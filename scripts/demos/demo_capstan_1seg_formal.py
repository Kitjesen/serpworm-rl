#!/usr/bin/env python3
"""Formal 3-Column Single-Segment Capstan Friction Comparison (Top: 3 Separate 3D Animations, Bottom: 3 Separate Data Panels).

Publication-grade academic layout (1440x810, 16:9, pure white canvas, zero decorative pill boxes):
  - Column 1: Smooth Capstan (现行光滑模型)
  - Column 2: Exact 2-Span Complementarity (精确双段粘滑)
  - Column 3: Explicit Stick-Slip Approximation (显式粘滑近似)
Each column has its own identical close-up 3D MuJoCo viewport on top and its own synchronized
dual-channel telemetry chart (compression stroke + single-tendon T1/T2 tensions) directly below.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

PKG_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PKG_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serpworm.config import HardwareConfig
from serpworm.paths import ASSETS_DIR
from serpworm.physics.gpu_core import GPUCore, cross, matrix, mul, norm, rotate, unit

torch.set_num_threads(1)

OUT_DIR = PKG_ROOT / 'runs' / 'capstan_1seg_formal'
_ARTIFACT_ENV = os.environ.get('SERPWORM_ARTIFACT_DIR')
ARTIFACT_DIR: Path | None = Path(_ARTIFACT_ENV) if _ARTIFACT_ENV else None
MESH_DIR = ASSETS_DIR / 'cad' / 'meshes' / 'longworm2'

MODELS: tuple[dict[str, Any], ...] = (
    {'key': 'smooth', 'tag': '(a) 现行光滑模型', 'short': '现行光滑', 'color': '#1D4ED8', 't1_color': '#60A5FA', 'rgb': (29, 78, 216)},
    {'key': 'exact_2span', 'tag': '(b) 精确双段粘滑', 'short': '精确双段', 'color': '#047857', 't1_color': '#34D399', 'rgb': (4, 120, 87)},
    {'key': 'explicit_stickslip', 'tag': '(c) 显式粘滑近似', 'short': '显式近似', 'color': '#B91C1C', 't1_color': '#F87171', 'rgb': (185, 28, 28)},
)

# Configured below after sign verification (default placeholder, verified before running)
EXPLICIT_SPD_SIGN = -1.0
EXPLICIT_DS_M = 0.0005


def _step_clamped_1seg(
    core: GPUCore,
    target_rad: torch.Tensor,
    p1_ref: torch.Tensor,
    pm_custom: torch.Tensor,
    is_2span: torch.Tensor,
    is_expl: torch.Tensor,
    st: dict[str, Any],
) -> None:
    h = core.dt
    B = core.batch
    x = core.x[:, :2]
    v = core.v[:, :2]
    q = x[..., 3:]
    R = matrix(q)
    f = torch.zeros_like(core.v)

    radius = core.horn_radius
    guide = core.guide_dist
    chord = torch.sqrt(guide**2 + radius**2 - 2.0 * guide * radius * torch.cos(core.servo)).clamp_min(1e-9)
    lever = (radius * guide * torch.sin(core.servo).abs() / chord).clamp_max(radius)

    cap_mt = core.env_dynamics[:, 3:4].clamp(0.0, 0.6)
    sv_db = core.env_dynamics[:, 6:7].clamp_min(0.0)

    t_sm_srv = core.tension * torch.exp(cap_mt[:, :, None])
    t_2s_srv = st['t1_prev']
    t_ex_srv = core.tension / st['eta_expl'].clamp_min(0.2)
    t_for_srv = torch.where(
        is_2span[:, None, None], t_2s_srv,
        torch.where(is_expl[:, None, None], t_ex_srv, t_sm_srv),
    )
    load = t_for_srv.reshape(B, 2, 2).sum(-1) * lever

    s_max = core.servo_speed_limit[:, None] * h
    raw_err = target_rad - core.servo
    eff_err = torch.sign(raw_err) * (raw_err.abs() - sv_db).clamp_min(0.0)
    inc = eff_err.clamp(-s_max, s_max)
    inc = inc * torch.where(inc > 0.0, (1.0 - load / core.servo_torque_limit[:, None]).clamp(0.0, 1.0), 1.0)
    sv_vel = inc / h
    core.servo.add_(inc)

    chord = torch.sqrt(guide**2 + radius**2 - 2.0 * guide * radius * torch.cos(core.servo))
    takeup = (chord - (guide - radius)).reshape(B, 1, 2).repeat_interleave(2, -1)

    ar = rotate(q[:, 0::2, None, :].expand(-1, -1, 4, -1), core.anchors.expand(B, 1, -1, -1))
    br = rotate(q[:, 1::2, None, :].expand(-1, -1, 4, -1), core.guides.expand(B, 1, -1, -1))
    delta = x[:, 1::2, None, :3] + br - x[:, 0::2, None, :3] - ar
    length = norm(delta)
    direction = delta / length.clamp_min(1e-15)
    va = v[:, 0::2, None, :3] + cross(v[:, 0::2, None, 3:].expand_as(ar), ar)
    vb = v[:, 1::2, None, :3] + cross(v[:, 1::2, None, 3:].expand_as(br), br)
    env_slack = core.env_dynamics[:, 4:5, None]
    rest = (0.101 + env_slack) - takeup
    ext = length[..., 0] - rest
    spd = ((vb - va) * direction).sum(-1)
    cap_mt_t = cap_mt[:, :, None]

    # 1. Smooth Capstan (exact gpu_core.py)
    c_eff = core.tendon_damping + 12.0 * cap_mt_t
    ten_raw_sm = torch.where(ext >= 0.0, (core.tendon_stiffness * ext + c_eff * spd).clamp_min(0.0), 0.0)
    sv_vel_t = sv_vel.reshape(B, 1, 2).repeat_interleave(2, -1)
    pull_frac = torch.tanh(sv_vel_t.clamp_min(0.0) / 0.35)
    ten_sm = ten_raw_sm * torch.exp(-cap_mt_t * pull_frac)
    t1_sm = ten_sm * torch.exp(cap_mt_t)

    # 2. Exact 2-Span Complementarity
    k_each = 2.0 * core.tendon_stiffness
    e_seg_0 = length[..., 0] - (0.101 + env_slack)
    e_srv_0 = takeup
    tot = e_srv_0 + e_seg_0
    r_hi = torch.exp(cap_mt_t)
    r_lo = torch.exp(-cap_mt_t)
    s_pull = (r_lo * e_srv_0 - e_seg_0) / (1.0 + r_lo)
    s_rel = (r_hi * e_srv_0 - e_seg_0) / (1.0 + r_hi)
    s_old = st['s_hole']
    s_new = torch.where(tot > 0.0, s_old.clamp(s_pull, s_rel), 0.5 * (e_srv_0 - e_seg_0))
    stick_2s = (tot > 1e-6) & (s_old > s_pull + 1e-9) & (s_old < s_rel - 1e-9)
    st['s_hole'].copy_(s_new)
    t1_2s = torch.where(tot > 0.0, k_each * (e_srv_0 - s_new).clamp_min(0.0), 0.0)
    ten_2s = torch.where(tot > 0.0, (k_each * (e_seg_0 + s_new) + core.tendon_damping * spd).clamp_min(0.0), 0.0)
    st['t1_prev'].copy_(t1_2s)

    # 3. Explicit Stick-Slip Approximation
    ten_raw_ex = torch.where(ext >= 0.0, (core.tendon_stiffness * ext + core.tendon_damping * spd).clamp_min(0.0), 0.0)
    lever_t = lever.reshape(B, 1, 2).repeat_interleave(2, -1)
    v_guide = sv_vel_t * lever_t + EXPLICIT_SPD_SIGN * spd
    d_log_eta = -v_guide * h / EXPLICIT_DS_M
    log_eta_new = (torch.log(st['eta_expl'].clamp_min(1e-6)) + d_log_eta).clamp(-cap_mt_t, cap_mt_t)
    stick_ex = (log_eta_new.abs() < (cap_mt_t - 1e-6)) & (ten_raw_ex > 0.1)
    eta_new = torch.where(ten_raw_ex > 1e-6, torch.exp(log_eta_new), torch.ones_like(log_eta_new))
    st['eta_expl'].copy_(eta_new)
    ten_ex = (ten_raw_ex * eta_new).clamp_max(120.0)
    t1_ex = ten_ex / eta_new.clamp_min(0.2)

    ten_sel = torch.where(is_2span[:, None, None], ten_2s, torch.where(is_expl[:, None, None], ten_ex, ten_sm))
    t1_sel = torch.where(is_2span[:, None, None], t1_2s, torch.where(is_expl[:, None, None], t1_ex, t1_sm))
    stick_sel = torch.where(is_2span[:, None, None], stick_2s, torch.where(is_expl[:, None, None], stick_ex, torch.zeros_like(stick_2s)))

    core.tension.copy_(ten_sel)
    st['last_t1'].copy_(t1_sel)
    st['last_stick'].copy_(stick_sel)

    force = core.tension[..., None] * direction
    f[:, 0:2:2, :3] += force.sum(-2)
    f[:, 1:2:2, :3] -= force.sum(-2)
    f[:, 0:2:2, 3:] += cross(ar, force).sum(-2)
    f[:, 1:2:2, 3:] -= cross(br, force).sum(-2)

    sep = x[:, 1::2, :3] - x[:, 0::2, :3]
    axis = -R[:, 0::2, :, 0]
    gap = (sep * axis).sum(-1)
    push = 50000.0 * (core.stop_gap - gap).clamp_min(0.0)[..., None] * axis
    f[:, 1:2:2, :3] += push
    f[:, 0:2:2, :3] -= push
    f[:, 0:2:2, 3:] -= cross(sep, push)

    f[:, 0, :3] -= 0.8 * v[:, 0, :3]
    f[:, 0, 3:] -= 0.025 * v[:, 0, 3:]

    elastic, K, M = core.beam()
    vg = (core.v + h * core.gravity)[:, core.edges].reshape(B, core.S, core.E, 12)
    rhs = core.scatter_vector((M @ vg[..., None])[..., 0]) + h * (f + elastic)
    rhs[:, :2] += (pm_custom @ (core.v[:, :2] + h * core.gravity[:, :2])[..., None])[..., 0]
    A = (1.0 + h * core.rayleigh_alpha) * M + (h * core.rayleigh_beta_env[:, None, None, None, None] + h * h) * K
    vnew = core.solve(A, rhs, (1.0 + h * core.rayleigh_alpha) * pm_custom)

    core.v.copy_(vnew)
    core.v[:, 1] = 0.0
    core.x[..., :3].add_(h * core.v[..., :3])
    omega = 0.5 * h * core.v[..., 3:]
    theta = norm(omega)
    exp_q = torch.cat((omega * torch.sinc(theta / math.pi), torch.cos(theta)), -1)
    core.x[..., 3:].copy_(unit(mul(exp_q, core.x[..., 3:])))
    core.x[:, 1] = p1_ref


def simulate_1seg() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    hw = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
    dt = 0.001
    duration_s = 2.0
    n_steps = round(duration_s / dt)
    fps = 30
    frame_times = np.linspace(0.0, duration_s, int(round(duration_s * fps)) + 1)

    core = GPUCore(
        count=1, batch=6, dt=dt, device='cpu', dtype=torch.float64,
        plate_solver='segment_lu', hardware=hw, elements_per_strip=16, ground_contact=False,
    )
    core.env_dynamics[:, 3] = torch.tensor([0.18, 0.18, 0.18, 0.36, 0.36, 0.36], dtype=torch.float64)
    is_2span = torch.tensor([False, True, False, False, True, False], dtype=torch.bool)
    is_expl = torch.tensor([False, False, True, False, False, True], dtype=torch.bool)

    pm_custom = core.plate_mass()
    pm_custom[:, 0] *= (0.115 / float(hw.plate_masses_kg[0]))
    p1_ref = core.x[:, 1].clone()

    st = {
        's_hole': torch.zeros((6, 1, 4), dtype=torch.float64),
        't1_prev': torch.zeros((6, 1, 4), dtype=torch.float64),
        'eta_expl': torch.ones((6, 1, 4), dtype=torch.float64),
        'last_t1': torch.zeros((6, 1, 4), dtype=torch.float64),
        'last_stick': torch.zeros((6, 1, 4), dtype=torch.bool),
    }

    zero_tgt = torch.zeros((6, 2), dtype=torch.float64)
    for _ in range(250):
        _step_clamped_1seg(core, zero_tgt, p1_ref, pm_custom, is_2span, is_expl, st)

    gap0 = torch.linalg.vector_norm(core.x[:, 1, :3] - core.x[:, 0, :3], dim=-1).numpy() * 1000.0

    t_s = np.zeros(n_steps + 1)
    cmd_deg = np.zeros(n_steps + 1)
    servo_deg = np.zeros((n_steps + 1, 6))
    gap_mm = np.zeros((n_steps + 1, 6))
    comp_mm = np.zeros((n_steps + 1, 6))
    t2_single_n = np.zeros((n_steps + 1, 6))
    t1_single_n = np.zeros((n_steps + 1, 6))
    stick_flag = np.zeros((n_steps + 1, 6), dtype=np.int32)

    frames_x = []
    frames_t = []
    next_fi = 0

    t0 = time.perf_counter()
    for k in range(n_steps + 1):
        ts = k * dt
        if ts < 0.7:
            ang = math.pi * (ts / 0.7)
        elif ts < 1.0:
            ang = math.pi
        elif ts < 1.7:
            ang = math.pi * (1.0 - (ts - 1.0) / 0.7)
        else:
            ang = 0.0

        if k > 0:
            tgt = torch.full((6, 2), ang, dtype=torch.float64)
            _step_clamped_1seg(core, tgt, p1_ref, pm_custom, is_2span, is_expl, st)

        t_s[k] = ts
        cmd_deg[k] = math.degrees(ang)
        servo_deg[k] = np.degrees(core.servo[:, 0].numpy())
        cur_gap = torch.linalg.vector_norm(core.x[:, 1, :3] - core.x[:, 0, :3], dim=-1).numpy() * 1000.0
        gap_mm[k] = cur_gap
        comp_mm[k] = gap0 - cur_gap
        t2_single_n[k] = core.tension[:, 0, :].mean(-1).numpy()
        t1_single_n[k] = st['last_t1'][:, 0, :].mean(-1).numpy()
        stick_flag[k] = st['last_stick'][:, 0, :].any(-1).numpy().astype(np.int32)

        if next_fi < len(frame_times) and ts + 0.5 * dt >= frame_times[next_fi]:
            frames_t.append(ts)
            frames_x.append(core.x.numpy().astype(np.float32).copy())
            next_fi += 1

    print(f'[Simulate] E=16 6-lane finished in {time.perf_counter()-t0:.2f} s', flush=True)

    summary: dict[str, Any] = {
        'elements_per_strip': 16, 'dt_ms': 1.0, 'duration_s': duration_s,
        'rest_gap_mm': round(float(gap0[0]), 2),
        'explicit_spd_sign': EXPLICIT_SPD_SIGN,
        'explicit_ds_mm': EXPLICIT_DS_M * 1000.0,
    }
    for g_idx, mt in enumerate((0.18, 0.36)):
        base = g_idx * 3
        ent: dict[str, Any] = {}
        for m_i, sp in enumerate(MODELS):
            ln = base + m_i
            ip = int(np.argmin(np.abs(servo_deg[:700, ln] - 110.0)))
            ir = 1000 + int(np.argmin(np.abs(servo_deg[1000:1700, ln] - 110.0)))
            ent[sp['key']] = {
                'min_gap_mm': round(float(gap_mm[:, ln].min()), 2),
                'max_comp_mm': round(float(comp_mm[:, ln].max()), 2),
                'mid_hyst_mm': round(float(abs(gap_mm[ip, ln] - gap_mm[ir, ln])), 2),
                'peak_t2_single_n': round(float(t2_single_n[:, ln].max()), 2),
                'peak_t1_single_n': round(float(t1_single_n[:, ln].max()), 2),
                'hold_t2_pp_n': round(float(np.ptp(t2_single_n[750:1000, ln])), 3),
                'stick_frac': round(float(stick_flag[:, ln].mean()), 3),
            }
        dg_sm = np.abs(gap_mm[:, base + 0] - gap_mm[:, base + 1])
        dt_sm = np.abs(t2_single_n[:, base + 0] - t2_single_n[:, base + 1])
        dg_ex = np.abs(gap_mm[:, base + 2] - gap_mm[:, base + 1])
        dt_ex = np.abs(t2_single_n[:, base + 2] - t2_single_n[:, base + 1])
        ent['smooth_vs_exact2span'] = {
            'mean_gap_diff_mm': round(float(dg_sm.mean()), 2),
            'max_gap_diff_mm': round(float(dg_sm.max()), 2),
            'mean_t2_single_diff_n': round(float(dt_sm.mean()), 3),
            'max_t2_single_diff_n': round(float(dt_sm.max()), 2),
        }
        ent['explicit_vs_exact2span'] = {
            'mean_gap_diff_mm': round(float(dg_ex.mean()), 2),
            'max_gap_diff_mm': round(float(dg_ex.max()), 2),
            'mean_t2_single_diff_n': round(float(dt_ex.mean()), 3),
            'max_t2_single_diff_n': round(float(dt_ex.max()), 2),
        }
        summary[f'mu_theta_{mt:.2f}'] = ent

    pack = {
        't_s': t_s, 'cmd_deg': cmd_deg, 'servo_deg': servo_deg,
        'gap_mm': gap_mm, 'comp_mm': comp_mm,
        't2_single_n': t2_single_n, 't1_single_n': t1_single_n,
        'stick_flag': stick_flag,
        'frames_x': np.asarray(frames_x), 'frames_t': np.asarray(frames_t), 'fps': fps,
        'strip_nodes': core.nodes.numpy(), 'transforms': core.transforms.numpy(),
        'mounts': core.mounts.numpy(), 'anchors': core.anchors.numpy(), 'guides': core.guides.numpy(),
    }
    return summary, pack


def _build_single_unit_mjcf(mounts: np.ndarray, x_rest_front: float, width: int, height: int) -> str:
    """Dedicated single-unit studio MJCF (pure white ground, crisp studio lighting, zero shadow acne)."""
    mesh_dir = MESH_DIR.resolve().as_posix()
    extras_p1 = []
    for sy in (+0.0285, -0.0285):
        extras_p1.append(
            f'<geom type="box" pos="0.018 {sy:+.4f} 0.000" size="0.017 0.0102 0.0088" '
            f'material="mat_servo_purple" contype="0" conaffinity="0"/>'
        )
    for s in range(2):
        mx, my, mz = mounts[1, s]
        strut_mid = 0.5 * (-0.064 + mz)
        strut_half = max(0.005, 0.5 * abs(-0.064 - mz))
        extras_p1.append(
            f'<geom type="box" pos="{mx:.5f} {my:.5f} {strut_mid:.5f}" '
            f'size="0.005 0.0035 {strut_half:.5f}" material="mat_strut" contype="0" conaffinity="0"/>'
        )
        extras_p1.append(
            f'<geom type="mesh" mesh="wheel_mesh" pos="{mx:.5f} {my:.5f} {mz:.5f}" '
            f'material="mat_wheel" contype="0" conaffinity="0"/>'
        )

    x_37 = x_rest_front - 0.037
    return f"""<mujoco model="serpworm_1seg_single_panel">
  <compiler angle="radian" meshdir="{mesh_dir}" autolimits="true"/>
  <visual>
    <global offwidth="{width}" offheight="{height}" fovy="34"/>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.02" zfar="8.0"/>
    <headlight ambient="0.30 0.31 0.32" diffuse="0.10 0.10 0.10" specular="0.0 0.0 0.0"/>
  </visual>
  <asset>
    <mesh name="plate_front" file="front2_Link.STL"/>
    <mesh name="plate_back"  file="back2_Link.STL"/>
    <mesh name="wheel_mesh"  file="w2-1_Link.STL"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="1.0 1.0 1.0" rgb2="0.98 0.985 0.99" width="256" height="256"/>
    <material name="mat_floor"        rgba="0.985 0.987 0.992 1.0" specular="0.01" shininess="0.02" reflectance="0.0"/>
    <material name="mat_plate_front"  rgba="0.72 0.78 0.85 0.78" specular="0.80" shininess="0.75" reflectance="0.08"/>
    <material name="mat_plate_back"   rgba="0.28 0.33 0.40 1.0"  specular="0.62" shininess="0.58" reflectance="0.06"/>
    <material name="mat_servo_purple" rgba="0.44 0.23 0.64 1.0"  specular="0.78" shininess="0.72" reflectance="0.10"/>
    <material name="mat_strut"        rgba="0.40 0.45 0.52 1.0"  specular="0.50" shininess="0.45"/>
    <material name="mat_wheel"        rgba="0.15 0.16 0.18 1.0"  specular="0.20" shininess="0.18"/>
  </asset>
  <worldbody>
    <light name="key"  directional="true" pos="0 0 2.4" dir="0.35 0.32 -0.88" diffuse="0.58 0.58 0.57" specular="0.88 0.88 0.86" castshadow="false"/>
    <light name="rim"  directional="true" pos="0 0 2.0" dir="-0.55 -0.25 -0.79" diffuse="0.22 0.24 0.26" specular="0.72 0.74 0.78" castshadow="false"/>
    <light name="fill" directional="true" pos="0 0 2.0" dir="-0.10 0.62 -0.78" diffuse="0.18 0.19 0.20" specular="0.15 0.15 0.16" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 0" size="4 4 0.1" material="mat_floor"/>
    <geom name="ref_0"  type="box" pos="{x_rest_front:.4f} 0 0.0005" size="0.0010 0.11 0.0005" rgba="0.58 0.62 0.68 0.75" contype="0" conaffinity="0"/>
    <geom name="ref_37" type="box" pos="{x_37:.4f} 0 0.0005" size="0.0010 0.11 0.0005" rgba="0.06 0.52 0.38 0.70" contype="0" conaffinity="0"/>
    <body name="p0" mocap="true" pos="0 0 0.13">
      <geom type="mesh" mesh="plate_front" material="mat_plate_front" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.004 0 -0.003" size="0.015 0.0098 0.0088" material="mat_servo_purple" contype="0" conaffinity="0"/>
    </body>
    <body name="p1" mocap="true" pos="0 0 0.13">
      <geom type="mesh" mesh="plate_back" material="mat_plate_back" contype="0" conaffinity="0"/>
      {chr(10).join('      ' + g for g in extras_p1)}
    </body>
  </worldbody>
</mujoco>
"""


def _add_single_unit_geoms(scene, x_row, strip_nodes, transforms, anchors, guides):
    import mujoco
    from serpworm.probes.render_mujoco import _quat_xyzw_to_mat

    steel_rgba = np.array([0.36, 0.49, 0.61, 1.0], dtype=np.float32)
    tendon_rgba = np.array([0.14, 0.16, 0.20, 0.96], dtype=np.float32)

    R_plates = [_quat_xyzw_to_mat(x_row[p, 3:7]) for p in range(2)]
    p_plates = x_row[:2, :3].astype(np.float64).copy()
    n_elems = strip_nodes.shape[1] - 1

    for sid in range(strip_nodes.shape[0]):
        nodes = strip_nodes[sid]
        pts = x_row[nodes, :3].astype(np.float64).copy()
        p_f, p_b = int(nodes[0]), int(nodes[-1])
        pts[0] = p_plates[p_f] + R_plates[p_f] @ transforms[sid, 0, 0, :3]
        pts[-1] = p_plates[p_b] + R_plates[p_b] @ transforms[sid, -1, 1, :3]
        seg_center = 0.5 * (p_plates[p_f] + p_plates[p_b])
        for e in range(n_elems):
            if scene.ngeom >= scene.maxgeom:
                return
            a, b = pts[e], pts[e + 1]
            diff = b - a
            length = float(np.linalg.norm(diff))
            if length < 1e-6:
                continue
            mid = 0.5 * (a + b)
            ex = diff / length
            rad = mid - seg_center
            ez = rad - np.dot(rad, ex) * ex
            n_ez = float(np.linalg.norm(ez))
            ez = np.array([0.0, 0.0, 1.0]) if n_ez < 1e-6 else ez / n_ez
            ey = np.cross(ez, ex)
            ey /= max(float(np.linalg.norm(ey)), 1e-9)
            ez = np.cross(ex, ey)
            rot = np.column_stack((ex, ey, ez)).reshape(9)
            size = np.array([0.58 * length, 0.0100, 0.00055], dtype=np.float64)
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_BOX), size, mid, rot, steel_rgba)
            g.matid = -1; g.shininess = 0.92; g.specular = 0.95; g.reflectance = 0.18
            scene.ngeom += 1

    for k in range(4):
        if scene.ngeom >= scene.maxgeom:
            return
        pt_a = p_plates[0] + R_plates[0] @ anchors[k]
        pt_b = p_plates[1] + R_plates[1] @ guides[k]
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE), np.zeros(3), np.zeros(3), np.eye(3).reshape(9), tendon_rgba)
        mujoco.mjv_connector(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE), 0.0016, pt_a, pt_b)
        scene.ngeom += 1


def _render_3col_data_row(
    t_s: np.ndarray,
    comp_3: np.ndarray,   # (N, 3)
    t2_3: np.ndarray,     # (N, 3)
    t1_3: np.ndarray,     # (N, 3)
    stick_3: np.ndarray,  # (N, 3)
    t_cur: float,
    width: int,
    height: int,
    y_comp_max: float,
    y_ten_max: float,
) -> np.ndarray:
    """Render 3 separate column-aligned synchronized data panels (2 stacked plots per column, pure white)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    dpi = 100
    fig, axes = plt.subplots(2, 3, figsize=(width / dpi, height / dpi), dpi=dpi, sharex=True)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.048, right=0.985, top=0.90, bottom=0.14, wspace=0.17, hspace=0.28)

    sl = slice(None, None, 5)
    ts = t_s[sl]
    mask = ts <= t_cur + 1e-6
    ref_comp = comp_3[sl, 1]  # Exact 2-Span reference dashed hairline

    for col, sp in enumerate(MODELS):
        # Upper subplot: Compression stroke (mm)
        ax_top = axes[0, col]
        ax_top.set_facecolor('white')
        for side in ('top', 'right'):
            ax_top.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax_top.spines[side].set_color('#94A3B8')
            ax_top.spines[side].set_linewidth(0.8)
        ax_top.grid(True, linestyle='--', linewidth=0.45, alpha=0.35, color='#CBD5E1')
        if col != 1:
            ax_top.plot(ts, ref_comp, color='#94A3B8', ls=':', lw=1.0, label='精确双段参考')
        c_col = comp_3[sl, col]
        ax_top.plot(ts, c_col, color=sp['color'], lw=0.8, alpha=0.18)
        if mask.any():
            ax_top.plot(ts[mask], c_col[mask], color=sp['color'], lw=1.9, label='压缩量')
        ax_top.axvline(t_cur, color='#334155', lw=0.9, ls=':')
        ax_top.set_ylim(-2.0, y_comp_max)
        ax_top.set_xlim(0.0, float(ts[-1]))
        ax_top.tick_params(labelsize=8.2, colors='#475569')
        if col == 0:
            ax_top.set_ylabel('压缩量 / mm', fontsize=8.8, color='#1E293B')
        ax_top.legend(frameon=False, loc='upper right', fontsize=7.8)

        # Lower subplot: Single-tendon T1 (servo side) & T2 (segment side)
        ax_bot = axes[1, col]
        ax_bot.set_facecolor('white')
        for side in ('top', 'right'):
            ax_bot.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax_bot.spines[side].set_color('#94A3B8')
            ax_bot.spines[side].set_linewidth(0.8)
        ax_bot.grid(True, linestyle='--', linewidth=0.45, alpha=0.35, color='#CBD5E1')
        # Subtle stick phase shading on Exact 2-Span
        if col == 1:
            stk = stick_3[sl, col] > 0
            ax_bot.fill_between(ts, 0, y_ten_max, where=stk, color='#E2E8F0', alpha=0.45, label='粘滞区')
        t1_c = t1_3[sl, col]
        t2_c = t2_3[sl, col]
        ax_bot.plot(ts, t1_c, color=sp['t1_color'], lw=0.7, alpha=0.22)
        ax_bot.plot(ts, t2_c, color=sp['color'], lw=0.8, alpha=0.20)
        if mask.any():
            ax_bot.plot(ts[mask], t1_c[mask], color=sp['t1_color'], lw=1.5, ls='--', label='舵机侧 T1')
            ax_bot.plot(ts[mask], t2_c[mask], color=sp['color'], lw=1.9, ls='-', label='体节侧 T2')
        ax_bot.axvline(t_cur, color='#334155', lw=0.9, ls=':')
        ax_bot.set_ylim(-1.0, y_ten_max)
        ax_bot.set_xlim(0.0, float(ts[-1]))
        ax_bot.set_xlabel('时间 / s', fontsize=8.6, color='#1E293B', labelpad=2)
        ax_bot.tick_params(labelsize=8.2, colors='#475569')
        if col == 0:
            ax_bot.set_ylabel('单根张力 / N', fontsize=8.8, color='#1E293B')
        ax_bot.legend(frameon=False, loc='upper right', fontsize=7.6, ncol=2)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def render_formal_3col_figure(summary: dict[str, Any], pack: dict[str, np.ndarray], out_png: Path) -> None:
    """Formal IEEE-style 3-column x 2-row comparison figure (mu_theta = 0.36; pure white, zero clutter)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.0), dpi=300, sharex=True)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.058, right=0.985, top=0.865, bottom=0.095, wspace=0.16, hspace=0.24)

    t = pack['t_s']
    comp_3 = pack['comp_mm'][:, 3:6]
    t2_3 = pack['t2_single_n'][:, 3:6]
    t1_3 = pack['t1_single_n'][:, 3:6]
    stick_3 = pack['stick_flag'][:, 3:6]

    y_comp_max = max(44.0, float(comp_3.max()) * 1.12)
    y_ten_max = max(24.0, float(max(t1_3.max(), t2_3.max())) * 1.12)

    for col, sp in enumerate(MODELS):
        ax_t = axes[0, col]
        ax_b = axes[1, col]
        for ax in (ax_t, ax_b):
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#475569')
            ax.spines['bottom'].set_color('#475569')
            ax.spines['left'].set_linewidth(0.85)
            ax.spines['bottom'].set_linewidth(0.85)
            ax.grid(True, linestyle='--', linewidth=0.45, alpha=0.35, color='#CBD5E1')
            ax.tick_params(labelsize=8.8, colors='#334155')

        if col != 1:
            ax_t.plot(t, comp_3[:, 1], color='#94A3B8', ls=':', lw=1.1, label='精确双段参考')
        ax_t.plot(t, comp_3[:, col], color=sp['color'], lw=1.8, label='前端板压缩量')
        ax_t.set_ylim(-2.0, y_comp_max)
        ax_t.set_title(sp['tag'], fontsize=11.0, fontweight='bold', color='#0F172A', pad=8)
        ax_t.legend(frameon=False, loc='upper right', fontsize=8.4)
        if col == 0:
            ax_t.set_ylabel('前端板压缩量 / mm', fontsize=9.6, color='#0F172A')

        if col == 1:
            ax_b.fill_between(t, 0, y_ten_max, where=(stick_3[:, col] > 0), color='#E2E8F0', alpha=0.55, label='粘滞区')
        ax_b.plot(t, t1_3[:, col], color=sp['t1_color'], ls='--', lw=1.5, label='舵机侧 T1')
        ax_b.plot(t, t2_3[:, col], color=sp['color'], ls='-', lw=1.8, label='体节侧 T2')
        ax_b.set_ylim(-1.0, y_ten_max)
        ax_b.set_xlabel('时间 / s', fontsize=9.5, color='#0F172A')
        ax_b.legend(frameon=False, loc='upper right', fontsize=8.2, ncol=2)
        if col == 0:
            ax_b.set_ylabel('单根拉绳张力 / N', fontsize=9.6, color='#0F172A')

    fig.suptitle('单体节拉绳导孔 Capstan 三模型独立对比（全网格 E=16，μθ = 0.36，dt = 1.0 ms）', fontsize=12.8, fontweight='bold', color='#0F172A', y=0.955)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'[Figure] Saved {out_png} ({out_png.stat().st_size / 1024:.1f} KiB)', flush=True)


def render_formal_3col_video(pack: dict[str, np.ndarray], out_mp4: Path, out_gif: Path, out_preview: Path) -> None:
    """Render formal 1440x810 3-column video: Top = 3 separate close-up 3D viewports, Bottom = 3 data panels."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from serpworm.probes.render_mujoco import _quat_xyzw_to_wxyz

    W_TOT, H_HDR, H_3D, H_PLOT = 1440, 54, 420, 336
    H_TOT = H_HDR + H_3D + H_PLOT  # 810
    W_COL = W_TOT // 3             # 480

    # Lanes 3, 4, 5 -> mu_theta = 0.36
    x_traj = pack['frames_x'][:, 3:6]  # (T, 3, D, 7)
    t_frames = pack['frames_t']
    fps = int(pack['fps'])
    t_s = pack['t_s']
    comp_3 = pack['comp_mm'][:, 3:6]
    t2_3 = pack['t2_single_n'][:, 3:6]
    t1_3 = pack['t1_single_n'][:, 3:6]
    stick_3 = pack['stick_flag'][:, 3:6]

    y_comp_max = max(44.0, float(comp_3.max()) * 1.12)
    y_ten_max = max(24.0, float(max(t1_3.max(), t2_3.max())) * 1.12)

    x_rest_front = float(x_traj[0, 0, 0, 0])
    xml = _build_single_unit_mjcf(pack['mounts'], x_rest_front, W_COL, H_3D)
    model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H_3D, width=W_COL)
    mocap_p0 = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'p0')]
    mocap_p1 = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'p1')]

    # Identical close-up 3/4 camera with clean top breathing room above the steel arch for text header
    x_mid = 0.5 * (float(x_traj[0, 0, 0, 0]) + float(x_traj[0, 0, 1, 0]))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [x_mid - 0.005, 0.0, 0.145]
    cam.azimuth = 118.0
    cam.elevation = -22.0
    cam.distance = 0.52

    font_title = ImageFont.load_default()
    font_sub = ImageFont.load_default()
    font_col = ImageFont.load_default()
    font_num = ImageFont.load_default()
    for fp in ('C:/Windows/Fonts/msyhbd.ttc', 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
        if Path(fp).is_file():
            font_title = ImageFont.truetype(fp, 21)
            font_col = ImageFont.truetype(fp, 18)
            break
    for fp in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
        if Path(fp).is_file():
            font_sub = ImageFont.truetype(fp, 16)
            font_num = ImageFont.truetype(fp, 15)
            break

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W_TOT}x{H_TOT}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    T = x_traj.shape[0]
    preview_fi = int(round(0.42 * (T - 1)))  # ~0.83 s peak hold

    t0 = time.perf_counter()
    for fi in range(T):
        idx_k = int(np.argmin(np.abs(t_s - t_frames[fi])))
        canvas = np.full((H_TOT, W_TOT, 3), 255, dtype=np.uint8)

        # Render each of the 3 single-segment models with its OWN dedicated identical camera
        for col in range(3):
            xr = x_traj[fi, col]
            mj_data.mocap_pos[mocap_p0] = xr[0, :3].astype(np.float64)
            mj_data.mocap_quat[mocap_p0] = _quat_xyzw_to_wxyz(xr[0, 3:7])
            mj_data.mocap_pos[mocap_p1] = xr[1, :3].astype(np.float64)
            mj_data.mocap_quat[mocap_p1] = _quat_xyzw_to_wxyz(xr[1, 3:7])
            mujoco.mj_forward(model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            _add_single_unit_geoms(renderer.scene, xr, pack['strip_nodes'], pack['transforms'], pack['anchors'], pack['guides'])
            rgb_col = renderer.render()
            canvas[H_HDR:H_HDR + H_3D, col * W_COL:(col + 1) * W_COL] = rgb_col

        # Render aligned 3-column synchronized data panel directly underneath
        rgb_data = _render_3col_data_row(t_s, comp_3, t2_3, t1_3, stick_3, float(t_frames[fi]), W_TOT, H_PLOT, y_comp_max, y_ten_max)
        canvas[H_HDR + H_3D:, :] = rgb_data

        # Formal crisp typography & thin structural hairlines (zero colored background boxes)
        im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(im)

        # Top header bar (pure white with 1px slate bottom rule)
        draw.text((24, 14), '单体节拉绳导孔 Capstan 摩擦模型对比（μθ = 0.36，全网格 E=16）', fill=(15, 23, 42), font=font_title)
        meta_txt = f"t = {t_frames[fi]:.2f} s   |   舵机指令 {pack['cmd_deg'][idx_k]:.0f}°   |   dt = 1.0 ms"
        mb = draw.textbbox((0, 0), meta_txt, font=font_sub)
        draw.text((W_TOT - (mb[2] - mb[0]) - 24, 18), meta_txt, fill=(71, 85, 105), font=font_sub)
        draw.line([(0, H_HDR - 1), (W_TOT, H_HDR - 1)], fill=(203, 213, 225), width=1)
        draw.line([(0, H_HDR + H_3D), (W_TOT, H_HDR + H_3D)], fill=(203, 213, 225), width=1)

        # Subtle vertical column dividers + formal column titles & live numeric readouts
        for col, sp in enumerate(MODELS):
            x_left = col * W_COL
            if col > 0:
                draw.line([(x_left, H_HDR), (x_left, H_TOT)], fill=(226, 232, 240), width=1)
            draw.text((x_left + 18, H_HDR + 12), sp['tag'], fill=sp['rgb'], font=font_col)
            readout = f"压缩 {comp_3[idx_k, col]:4.1f} mm    T2 {t2_3[idx_k, col]:4.1f} N    T1 {t1_3[idx_k, col]:4.1f} N"
            draw.text((x_left + 18, H_HDR + 38), readout, fill=(51, 65, 85), font=font_num)

        frame = np.asarray(im)
        if fi == preview_fi:
            im.save(out_preview)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode('utf-8', errors='replace')
    rc = proc.wait()
    renderer.close()
    if rc != 0:
        raise RuntimeError(f'ffmpeg failed ({rc}): {err[-400:]}')
    print(f'[Video] Saved {out_mp4} ({out_mp4.stat().st_size / (1024*1024):.2f} MiB) in {time.perf_counter()-t0:.1f}s', flush=True)

    pal = out_gif.parent / '_formal_pal.png'
    vf = 'fps=15,scale=960:-1:flags=lanczos'
    subprocess.run(['ffmpeg', '-y', '-i', str(out_mp4), '-vf', f'{vf},palettegen=stats_mode=diff', str(pal)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg', '-y', '-i', str(out_mp4), '-i', str(pal), '-lavfi', f'{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3', str(out_gif)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if pal.is_file():
        pal.unlink()
    print(f'[GIF] Saved {out_gif} ({out_gif.stat().st_size / (1024*1024):.2f} MiB)', flush=True)


def save_csv(pack: dict[str, np.ndarray], out_csv: Path) -> None:
    cols = ['t_s', 'cmd_deg']
    for mt in ('018', '036'):
        for m in ('smooth', 'exact2span', 'explicit'):
            cols += [f'comp_{mt}_{m}_mm', f'gap_{mt}_{m}_mm', f't2_single_{mt}_{m}_n', f't1_single_{mt}_{m}_n']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(0, len(pack['t_s']), 10):
            row = [f"{pack['t_s'][k]:.3f}", f"{pack['cmd_deg'][k]:.2f}"]
            for ln in range(6):
                row += [
                    f"{pack['comp_mm'][k, ln]:.3f}",
                    f"{pack['gap_mm'][k, ln]:.3f}",
                    f"{pack['t2_single_n'][k, ln]:.3f}",
                    f"{pack['t1_single_n'][k, ln]:.3f}",
                ]
            w.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, pack = simulate_1seg()
    (OUT_DIR / 'capstan_1seg_formal_data.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    save_csv(pack, OUT_DIR / 'capstan_1seg_formal_timeseries.csv')

    curves_png = OUT_DIR / 'capstan_1seg_formal_curves.png'
    mp4_path = OUT_DIR / 'capstan_1seg_formal_demo.mp4'
    gif_path = OUT_DIR / 'capstan_1seg_formal_demo.gif'
    preview_png = OUT_DIR / 'capstan_1seg_formal_preview.png'

    render_formal_3col_figure(summary, pack, curves_png)
    render_formal_3col_video(pack, mp4_path, gif_path, preview_png)

    if ARTIFACT_DIR is not None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        for src in (curves_png, gif_path, preview_png):
            shutil.copy2(src, ARTIFACT_DIR / src.name)

    print('\n=== FORMAL 1-SEG SUMMARY ===', flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

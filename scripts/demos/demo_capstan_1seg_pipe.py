#!/usr/bin/env python3
"""Formal 3-Column Single-Segment In-Pipe Squeeze Comparison (Top: 3D Transparent Acrylic Pipe, Bottom: Synced Data).

Places the full-mesh (E=16) single pre-curved spring-steel segment coaxially inside a rigid
transparent acrylic cylinder (ID = 176 mm, r_in = 88 mm; rest strip OD = 163.8 mm, free 180 deg OD = 198.4 mm).
As the 2 tendon servos retract (0 -> 180 -> 0 deg), all 8 steel arches expand radially into the pipe wall,
flatten against the cylinder, generate strong radial anchoring force F_wall, and load the tendons heavily.

Compares 3 Capstan guide-hole friction models (mu_theta = 0.36) + Open-Air reference:
  (a) Smooth Capstan (现行光滑模型)
  (b) Exact 2-Span Complementarity (精确双段粘滑)
  (c) Explicit Stick-Slip Approximation (显式粘滑近似)
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
from serpworm.physics.pipe_geometry import build_pipe_mujoco_geoms, eval_pipe_contact_torch, make_straight_pipe

torch.set_num_threads(1)

OUT_DIR = PKG_ROOT / 'runs' / 'capstan_1seg_pipe'
_ARTIFACT_ENV = os.environ.get('SERPWORM_ARTIFACT_DIR')
ARTIFACT_DIR: Path | None = Path(_ARTIFACT_ENV) if _ARTIFACT_ENV else None
MESH_DIR = ASSETS_DIR / 'cad' / 'meshes' / 'longworm2'

MODELS: tuple[dict[str, Any], ...] = (
    {'key': 'smooth', 'tag': '(a) 现行光滑模型', 'short': '现行光滑', 'color': '#1D4ED8', 't1_color': '#60A5FA', 'rgb': (29, 78, 216)},
    {'key': 'exact_2span', 'tag': '(b) 精确双段粘滑', 'short': '精确双段', 'color': '#047857', 't1_color': '#34D399', 'rgb': (4, 120, 87)},
    {'key': 'explicit_stickslip', 'tag': '(c) 显式粘滑近似', 'short': '显式近似', 'color': '#B91C1C', 't1_color': '#F87171', 'rgb': (185, 28, 28)},
)

PIPE_R_IN_M = 0.088      # 176 mm ID (rest strip OD 163.8 mm -> 6.1 mm clearance; 180 deg free OD 198.4 mm -> strong wall squeeze!)
PIPE_K_WALL = 8000.0     # rigid acrylic pipe wall stiffness (N/m); k_scale=0.25 -> 2000 N/m per interior strip node
EXPLICIT_DS_M = 0.0005


def _step_clamped_1seg_in_pipe(
    core: GPUCore,
    target_rad: torch.Tensor,
    p1_ref: torch.Tensor,
    pm_custom: torch.Tensor,
    is_2span: torch.Tensor,
    is_expl: torch.Tensor,
    st: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """One FP64 implicit substep with coaxial pipe contact on all 8x15=120 steel-strip interior nodes + front plate."""
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

    # 1. Smooth Capstan
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

    # 3. Explicit Stick-Slip Approximation (physically correct sign: v_guide = sv_vel*lever - spd)
    ten_raw_ex = torch.where(ext >= 0.0, (core.tendon_stiffness * ext + core.tendon_damping * spd).clamp_min(0.0), 0.0)
    lever_t = lever.reshape(B, 1, 2).repeat_interleave(2, -1)
    v_guide = sv_vel_t * lever_t - spd
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

    # Mild air/clamp damping on cantilevered front plate
    f[:, 0, :3] -= 0.8 * v[:, 0, :3]
    f[:, 0, 3:] -= 0.025 * v[:, 0, 3:]

    # Coaxial cylindrical pipe contact on front plate + all 8x15=120 steel strip interior nodes (implicit SPD!)
    plate_K_c = torch.zeros((B, 2, 6, 6), device=core.device, dtype=core.dtype)
    f_pl, k_pl, fn_pl = eval_pipe_contact_torch(
        x[:, :, :3], v[:, :, :3], core.pipe_cfg,
        node_radius=0.074, dt=h, implicit_contact=True, k_scale=1.0, c_wall=18.0,
    )
    f[:, :2, :3] += f_pl
    plate_K_c[:, :, :3, :3] += k_pl

    ids = core.nodes[:, 1:-1].reshape(-1)
    pos_int = core.x[:, ids, :3]
    vel_int = core.v[:, ids, :3]
    f_st, k_st, fn_st = eval_pipe_contact_torch(
        pos_int, vel_int, core.pipe_cfg,
        node_radius=0.002, dt=h, implicit_contact=True, k_scale=0.25, c_wall=6.0,
    )
    strip_K = k_st.reshape(B, core.S, core.E - 1, 3, 3)

    elastic, K, M = core.beam()
    vg = (core.v + h * core.gravity)[:, core.edges].reshape(B, core.S, core.E, 12)
    rhs = core.scatter_vector((M @ vg[..., None])[..., 0]) + h * (f + elastic)
    rhs[:, :2] += (pm_custom @ (core.v[:, :2] + h * core.gravity[:, :2])[..., None])[..., 0]
    rhs[:, core.interior_ids, :3] += h * f_st
    A = (1.0 + h * core.rayleigh_alpha) * M + (h * core.rayleigh_beta_env[:, None, None, None, None] + h * h) * K
    plate_lhs = (1.0 + h * core.rayleigh_alpha) * pm_custom + plate_K_c
    vnew = core.solve(A, rhs, plate_lhs, strip_K=strip_K)

    core.v.copy_(vnew)
    core.v[:, 1] = 0.0
    core.x[..., :3].add_(h * core.v[..., :3])
    omega = 0.5 * h * core.v[..., 3:]
    theta = norm(omega)
    exp_q = torch.cat((omega * torch.sinc(theta / math.pi), torch.cos(theta)), -1)
    core.x[..., 3:].copy_(unit(mul(exp_q, core.x[..., 3:])))
    core.x[:, 1] = p1_ref

    # Return (total_radial_wall_force [B], per_node_wall_fn [B, S, E-1])
    total_fn = fn_st.sum(dim=-1) + fn_pl[:, 0]
    return total_fn, fn_st.reshape(B, core.S, core.E - 1)


def simulate_1seg_in_pipe() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Simulate 4 batched E=16 single segments at mu_theta=0.36:
      Lane 0: In-Pipe Smooth Capstan
      Lane 1: In-Pipe Exact 2-Span
      Lane 2: In-Pipe Explicit Stick-Slip
      Lane 3: Open-Air Exact 2-Span (reference without pipe!)
    """
    hw = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
    dt = 0.001
    duration_s = 2.0
    n_steps = round(duration_s / dt)
    fps = 30
    frame_times = np.linspace(0.0, duration_s, int(round(duration_s * fps)) + 1)

    core = GPUCore(
        count=1, batch=4, dt=dt, device='cpu', dtype=torch.float64,
        plate_solver='segment_lu', hardware=hw, elements_per_strip=16, ground_contact=False,
    )
    core.env_dynamics[:, 3] = 0.36
    is_2span = torch.tensor([False, True, False, True], dtype=torch.bool)
    is_expl = torch.tensor([False, False, True, False], dtype=torch.bool)

    pm_custom = core.plate_mass()
    pm_custom[:, 0] *= (0.115 / float(hw.plate_masses_kg[0]))
    p1_ref = core.x[:, 1].clone()
    zc = float(core.x[0, :2, 2].mean().item())
    yc = float(core.x[0, :2, 1].mean().item())

    # Coaxial rigid transparent acrylic pipe around lanes 0..2; lane 3 stays open air (k_wall=0)
    pipe_spec = make_straight_pipe(
        x_entry=0.06, x_exit=-0.18, r_in=PIPE_R_IN_M, y0=yc, z_center=zc, k_wall=PIPE_K_WALL,
    )
    for ln in range(3):
        core.pipe_cfg[ln] = torch.tensor(pipe_spec, dtype=torch.float64)

    st = {
        's_hole': torch.zeros((4, 1, 4), dtype=torch.float64),
        't1_prev': torch.zeros((4, 1, 4), dtype=torch.float64),
        'eta_expl': torch.ones((4, 1, 4), dtype=torch.float64),
        'last_t1': torch.zeros((4, 1, 4), dtype=torch.float64),
        'last_stick': torch.zeros((4, 1, 4), dtype=torch.bool),
    }

    zero_tgt = torch.zeros((4, 2), dtype=torch.float64)
    for _ in range(250):
        _step_clamped_1seg_in_pipe(core, zero_tgt, p1_ref, pm_custom, is_2span, is_expl, st)

    gap0 = torch.linalg.vector_norm(core.x[:, 1, :3] - core.x[:, 0, :3], dim=-1).numpy() * 1000.0
    ids = core.nodes[:, 1:-1].reshape(-1)

    t_s = np.zeros(n_steps + 1)
    cmd_deg = np.zeros(n_steps + 1)
    servo_deg = np.zeros((n_steps + 1, 4))
    gap_mm = np.zeros((n_steps + 1, 4))
    comp_mm = np.zeros((n_steps + 1, 4))
    t2_single_n = np.zeros((n_steps + 1, 4))
    t1_single_n = np.zeros((n_steps + 1, 4))
    wall_fn_n = np.zeros((n_steps + 1, 4))
    max_r_mm = np.zeros((n_steps + 1, 4))
    stick_flag = np.zeros((n_steps + 1, 4), dtype=np.int32)

    frames_x = []
    frames_fn = []
    frames_t = []
    next_fi = 0

    t0 = time.perf_counter()
    last_node_fn = torch.zeros((4, core.S, core.E - 1), dtype=torch.float64)
    last_tot_fn = torch.zeros(4, dtype=torch.float64)
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
            tgt = torch.full((4, 2), ang, dtype=torch.float64)
            last_tot_fn, last_node_fn = _step_clamped_1seg_in_pipe(core, tgt, p1_ref, pm_custom, is_2span, is_expl, st)

        t_s[k] = ts
        cmd_deg[k] = math.degrees(ang)
        servo_deg[k] = np.degrees(core.servo[:, 0].numpy())
        cur_gap = torch.linalg.vector_norm(core.x[:, 1, :3] - core.x[:, 0, :3], dim=-1).numpy() * 1000.0
        gap_mm[k] = cur_gap
        comp_mm[k] = gap0 - cur_gap
        t2_single_n[k] = core.tension[:, 0, :].mean(-1).numpy()
        t1_single_n[k] = st['last_t1'][:, 0, :].mean(-1).numpy()
        wall_fn_n[k] = last_tot_fn.numpy()
        p_int = core.x[:, ids, :3]
        max_r_mm[k] = torch.hypot(p_int[..., 1] - yc, p_int[..., 2] - zc).amax(-1).numpy() * 1000.0
        stick_flag[k] = st['last_stick'][:, 0, :].any(-1).numpy().astype(np.int32)

        if next_fi < len(frame_times) and ts + 0.5 * dt >= frame_times[next_fi]:
            frames_t.append(ts)
            frames_x.append(core.x.numpy().astype(np.float32).copy())
            frames_fn.append(last_node_fn.numpy().astype(np.float32).copy())
            next_fi += 1

    print(f'[Simulate Pipe] E=16 4-lane finished in {time.perf_counter()-t0:.2f} s', flush=True)

    names = ('in_pipe_smooth', 'in_pipe_exact_2span', 'in_pipe_explicit', 'open_air_exact_2span')
    summary: dict[str, Any] = {
        'pipe_inner_dia_mm': round(PIPE_R_IN_M * 2000.0, 1),
        'pipe_r_in_mm': round(PIPE_R_IN_M * 1000.0, 1),
        'mu_theta': 0.36,
        'dt_ms': 1.0,
        'rest_gap_mm': round(float(gap0[0]), 2),
        'rest_outer_dia_mm': round(float(max_r_mm[0, 0] * 2.0), 1),
    }
    for i, nm in enumerate(names):
        summary[nm] = {
            'max_comp_mm': round(float(comp_mm[:, i].max()), 2),
            'min_gap_mm': round(float(gap_mm[:, i].min()), 2),
            'max_outer_dia_mm': round(float(max_r_mm[:, i].max() * 2.0), 1),
            'peak_wall_fn_n': round(float(wall_fn_n[:, i].max()), 2),
            'hold_wall_fn_n': round(float(wall_fn_n[750:1000, i].mean()), 2),
            'peak_t2_single_n': round(float(t2_single_n[:, i].max()), 2),
            'peak_t1_single_n': round(float(t1_single_n[:, i].max()), 2),
            'peak_seg_axial_n': round(float(4.0 * t2_single_n[:, i].max()), 2),
            'max_servo_deg': round(float(servo_deg[:, i].max()), 1),
            'stick_frac': round(float(stick_flag[:, i].mean()), 3),
        }

    # Exact diffs vs in_pipe_exact_2span
    for idx, label in ((0, 'smooth_vs_exact2span'), (2, 'explicit_vs_exact2span')):
        summary[label] = {
            'mean_comp_diff_mm': round(float(np.abs(comp_mm[:, idx] - comp_mm[:, 1]).mean()), 2),
            'max_comp_diff_mm': round(float(np.abs(comp_mm[:, idx] - comp_mm[:, 1]).max()), 2),
            'mean_wall_fn_diff_n': round(float(np.abs(wall_fn_n[:, idx] - wall_fn_n[:, 1]).mean()), 2),
            'max_wall_fn_diff_n': round(float(np.abs(wall_fn_n[:, idx] - wall_fn_n[:, 1]).max()), 2),
            'mean_t2_diff_n': round(float(np.abs(t2_single_n[:, idx] - t2_single_n[:, 1]).mean()), 3),
            'max_t2_diff_n': round(float(np.abs(t2_single_n[:, idx] - t2_single_n[:, 1]).max()), 2),
        }

    pack = {
        't_s': t_s, 'cmd_deg': cmd_deg, 'servo_deg': servo_deg,
        'gap_mm': gap_mm, 'comp_mm': comp_mm,
        't2_single_n': t2_single_n, 't1_single_n': t1_single_n,
        'wall_fn_n': wall_fn_n, 'max_r_mm': max_r_mm, 'stick_flag': stick_flag,
        'pipe_spec': np.asarray(pipe_spec, dtype=np.float64),
        'frames_x': np.asarray(frames_x), 'frames_fn': np.asarray(frames_fn),
        'frames_t': np.asarray(frames_t), 'fps': fps,
        'strip_nodes': core.nodes.numpy(), 'transforms': core.transforms.numpy(),
        'anchors': core.anchors.numpy(), 'guides': core.guides.numpy(),
    }
    return summary, pack


def _build_in_pipe_mjcf(pipe_spec: np.ndarray, width: int, height: int) -> str:
    """Clean white-studio MJCF with a single coaxially-mounted transparent acrylic pipe (zero bottom wheel clash)."""
    mesh_dir = MESH_DIR.resolve().as_posix()
    # Custom crisp 36-stave full 360-deg transparent acrylic cylinder + two subtle brushed aluminum end rings
    y0, r_in, x_en, x_b, z_c = float(pipe_spec[1]), float(pipe_spec[2]), float(pipe_spec[3]), float(pipe_spec[4]), float(pipe_spec[9])
    seg_len = abs(x_en - x_b)
    x_mid = 0.5 * (x_en + x_b)
    r_out = r_in + 0.004
    r_mid = 0.5 * (r_in + r_out)
    n_staves = 36
    dth = 2.0 * math.pi / n_staves
    stave_hw = r_mid * math.tan(0.5 * dth) * 1.05
    pipe_geoms = []
    for k in range(n_staves):
        th = (k + 0.5) * dth
        c_t, s_t = math.cos(th), math.sin(th)
        py = y0 + r_mid * c_t
        pz = z_c + r_mid * s_t
        # Top/bottom silhouettes slightly more visible to read glass contour cleanly on white background
        mat = 'mat_acrylic_rim' if (abs(s_t) > 0.85 or abs(c_t) > 0.92) else 'mat_acrylic_tube'
        # Rotation: X along tube, Y radial out, Z circumferential -> euler around X by th
        qw = math.cos(0.5 * th)
        qx = math.sin(0.5 * th)
        pipe_geoms.append(
            f'    <geom type="box" pos="{x_mid:.4f} {py:.4f} {pz:.4f}" '
            f'quat="{qw:.5f} {qx:.5f} 0.0 0.0" '
            f'size="{0.5 * seg_len:.4f} 0.0020 {stave_hw:.4f}" '
            f'material="{mat}" contype="0" conaffinity="0"/>'
        )
    # Clean slim end-cap rings at x_en and x_b so the cylinder mouth is razor-clear
    for x_ring in (x_en, x_b):
        for k in range(36):
            th = (k + 0.5) * dth
            c_t, s_t = math.cos(th), math.sin(th)
            py = y0 + (r_in + 0.0045) * c_t
            pz = z_c + (r_in + 0.0045) * s_t
            qw, qx = math.cos(0.5 * th), math.sin(0.5 * th)
            pipe_geoms.append(
                f'    <geom type="box" pos="{x_ring:.4f} {py:.4f} {pz:.4f}" '
                f'quat="{qw:.5f} {qx:.5f} 0.0 0.0" '
                f'size="0.0028 0.0032 {stave_hw * 1.05:.4f}" '
                f'material="mat_alu_ring" contype="0" conaffinity="0"/>'
            )

    extras_p1 = [
        f'<geom type="box" pos="0.018 {sy:+.4f} 0.000" size="0.017 0.0102 0.0088" material="mat_servo_purple" contype="0" conaffinity="0"/>'
        for sy in (+0.0285, -0.0285)
    ]

    return f"""<mujoco model="serpworm_1seg_in_pipe">
  <compiler angle="radian" meshdir="{mesh_dir}" autolimits="true"/>
  <visual>
    <global offwidth="{width}" offheight="{height}" fovy="34"/>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.02" zfar="8.0"/>
    <headlight ambient="0.32 0.33 0.34" diffuse="0.10 0.10 0.10" specular="0.0 0.0 0.0"/>
  </visual>
  <asset>
    <mesh name="plate_front" file="front2_Link.STL"/>
    <mesh name="plate_back"  file="back2_Link.STL"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="1.0 1.0 1.0" rgb2="0.985 0.99 0.995" width="256" height="256"/>
    <material name="mat_floor"        rgba="0.985 0.988 0.992 1.0" specular="0.01" shininess="0.02"/>
    <material name="mat_acrylic_tube" rgba="0.56 0.76 0.92 0.11" specular="0.95" shininess="0.92" reflectance="0.04"/>
    <material name="mat_acrylic_rim"  rgba="0.38 0.62 0.84 0.24" specular="0.98" shininess="0.95" reflectance="0.08"/>
    <material name="mat_alu_ring"     rgba="0.46 0.54 0.64 0.88" specular="0.72" shininess="0.68"/>
    <material name="mat_plate_front"  rgba="0.72 0.78 0.85 0.80" specular="0.80" shininess="0.75"/>
    <material name="mat_plate_back"   rgba="0.28 0.33 0.40 1.0"  specular="0.62" shininess="0.58"/>
    <material name="mat_servo_purple" rgba="0.44 0.23 0.64 1.0"  specular="0.78" shininess="0.72"/>
  </asset>
  <worldbody>
    <light name="key"  directional="true" pos="0 0 2.4" dir="0.35 0.32 -0.88" diffuse="0.56 0.56 0.55" specular="0.88 0.88 0.86" castshadow="false"/>
    <light name="rim"  directional="true" pos="0 0 2.0" dir="-0.55 -0.25 -0.79" diffuse="0.22 0.24 0.26" specular="0.72 0.74 0.78" castshadow="false"/>
    <light name="fill" directional="true" pos="0 0 2.0" dir="-0.10 0.62 -0.78" diffuse="0.18 0.19 0.20" specular="0.15 0.15 0.16" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 0" size="4 4 0.1" material="mat_floor"/>
    <body name="p0" mocap="true" pos="0 0 0.1335">
      <geom type="mesh" mesh="plate_front" material="mat_plate_front" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.004 0 -0.003" size="0.015 0.0098 0.0088" material="mat_servo_purple" contype="0" conaffinity="0"/>
    </body>
    <body name="p1" mocap="true" pos="-0.1175 0 0.1335">
      <geom type="mesh" mesh="plate_back" material="mat_plate_back" contype="0" conaffinity="0"/>
      {chr(10).join('      ' + g for g in extras_p1)}
    </body>
{chr(10).join(pipe_geoms)}
  </worldbody>
</mujoco>
"""


def _add_in_pipe_strips(scene, x_row, node_fn, strip_nodes, transforms, anchors, guides):
    """Render 8 deforming steel strips; elements actively pressing the pipe wall shift smoothly from steel blue to warm amber."""
    import mujoco
    from serpworm.probes.render_mujoco import _quat_xyzw_to_mat

    base_rgb = np.array([0.34, 0.48, 0.60], dtype=np.float32)
    hot_rgb = np.array([0.86, 0.34, 0.12], dtype=np.float32)  # contact pressure highlight on wall-pressed segments
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

            # Local contact intensity on element endpoints (interior nodes are indices 0..E-2)
            fn_a = float(node_fn[sid, e - 1]) if e > 0 else 0.0
            fn_b = float(node_fn[sid, e]) if e < n_elems - 1 else 0.0
            heat = min(1.0, max(fn_a, fn_b) / 4.5)
            rgb = (1.0 - heat) * base_rgb + heat * hot_rgb
            rgba = np.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)

            size = np.array([0.58 * length, 0.0100, 0.00060], dtype=np.float64)
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_BOX), size, mid, rot, rgba)
            g.matid = -1; g.shininess = 0.90; g.specular = 0.92; g.reflectance = 0.16
            scene.ngeom += 1

    for k in range(4):
        if scene.ngeom >= scene.maxgeom:
            return
        pt_a = p_plates[0] + R_plates[0] @ anchors[k]
        pt_b = p_plates[1] + R_plates[1] @ guides[k]
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE), np.zeros(3), np.zeros(3), np.eye(3).reshape(9), tendon_rgba)
        mujoco.mjv_connector(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE), 0.0016, pt_a, pt_b)
        scene.ngeom += 1


def _render_pipe_3col_data(
    t_s: np.ndarray,
    comp_3: np.ndarray,
    comp_open: np.ndarray,
    wall_3: np.ndarray,
    t2_3: np.ndarray,
    t1_3: np.ndarray,
    stick_3: np.ndarray,
    t_cur: float,
    width: int,
    height: int,
    y_wall_max: float,
    y_ten_max: float,
) -> np.ndarray:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    dpi = 100
    fig, axes = plt.subplots(2, 3, figsize=(width / dpi, height / dpi), dpi=dpi, sharex=True)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.052, right=0.985, top=0.91, bottom=0.14, wspace=0.18, hspace=0.28)

    sl = slice(None, None, 5)
    ts = t_s[sl]
    mask = ts <= t_cur + 1e-6

    for col, sp in enumerate(MODELS):
        # Upper: Pipe wall radial anchoring force F_wall (N)
        ax_t = axes[0, col]
        ax_t.set_facecolor('white')
        for s in ('top', 'right'):
            ax_t.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax_t.spines[s].set_color('#94A3B8'); ax_t.spines[s].set_linewidth(0.8)
        ax_t.grid(True, linestyle='--', linewidth=0.45, alpha=0.35, color='#CBD5E1')
        w_c = wall_3[sl, col]
        ax_t.plot(ts, w_c, color='#D97706', lw=0.8, alpha=0.22)
        if mask.any():
            ax_t.plot(ts[mask], w_c[mask], color='#D97706', lw=1.9, label='管壁总撑紧力 F_wall')
        ax_t.axvline(t_cur, color='#334155', lw=0.9, ls=':')
        ax_t.set_ylim(-2.0, y_wall_max)
        ax_t.set_xlim(0.0, float(ts[-1]))
        ax_t.tick_params(labelsize=8.2, colors='#475569')
        if col == 0:
            ax_t.set_ylabel('管壁撑紧力 / N', fontsize=8.8, color='#1E293B')
        ax_t.legend(frameon=False, loc='upper right', fontsize=7.8)

        # Lower: Single-tendon T1 & T2
        ax_b = axes[1, col]
        ax_b.set_facecolor('white')
        for s in ('top', 'right'):
            ax_b.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax_b.spines[s].set_color('#94A3B8'); ax_b.spines[s].set_linewidth(0.8)
        ax_b.grid(True, linestyle='--', linewidth=0.45, alpha=0.35, color='#CBD5E1')
        if col == 1:
            ax_b.fill_between(ts, 0, y_ten_max, where=(stick_3[sl, col] > 0), color='#E2E8F0', alpha=0.50, label='粘滞区')
        t1_c = t1_3[sl, col]
        t2_c = t2_3[sl, col]
        ax_b.plot(ts, t1_c, color=sp['t1_color'], lw=0.7, alpha=0.22)
        ax_b.plot(ts, t2_c, color=sp['color'], lw=0.8, alpha=0.20)
        if mask.any():
            ax_b.plot(ts[mask], t1_c[mask], color=sp['t1_color'], lw=1.5, ls='--', label='舵机侧 T1')
            ax_b.plot(ts[mask], t2_c[mask], color=sp['color'], lw=1.9, ls='-', label='体节侧 T2')
        ax_b.axvline(t_cur, color='#334155', lw=0.9, ls=':')
        ax_b.set_ylim(-1.0, y_ten_max)
        ax_b.set_xlim(0.0, float(ts[-1]))
        ax_b.set_xlabel('时间 / s', fontsize=8.6, color='#1E293B', labelpad=2)
        ax_b.tick_params(labelsize=8.2, colors='#475569')
        if col == 0:
            ax_b.set_ylabel('单根张力 / N', fontsize=8.8, color='#1E293B')
        ax_b.legend(frameon=False, loc='upper right', fontsize=7.6, ncol=2)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def render_pipe_3col_figure(summary: dict[str, Any], pack: dict[str, np.ndarray], out_png: Path) -> None:
    """Formal 3-column x 3-row figure (Row 1: Comp vs Open-Air, Row 2: Wall Force F_wall, Row 3: Single-Tendon T1/T2)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 3, figsize=(14.4, 8.8), dpi=300, sharex=True)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.058, right=0.985, top=0.895, bottom=0.075, wspace=0.16, hspace=0.24)

    t = pack['t_s']
    comp_3 = pack['comp_mm'][:, :3]
    comp_open = pack['comp_mm'][:, 3]
    wall_3 = pack['wall_fn_n'][:, :3]
    t2_3 = pack['t2_single_n'][:, :3]
    t1_3 = pack['t1_single_n'][:, :3]
    t2_open = pack['t2_single_n'][:, 3]
    stick_3 = pack['stick_flag'][:, :3]

    y_comp_max = max(42.0, float(max(comp_3.max(), comp_open.max())) * 1.12)
    y_wall_max = max(20.0, float(wall_3.max()) * 1.14)
    y_ten_max = max(24.0, float(max(t1_3.max(), t2_3.max())) * 1.14)

    for col, sp in enumerate(MODELS):
        for r in range(3):
            ax = axes[r, col]
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#475569'); ax.spines['bottom'].set_color('#475569')
            ax.spines['left'].set_linewidth(0.85); ax.spines['bottom'].set_linewidth(0.85)
            ax.grid(True, linestyle='--', linewidth=0.45, alpha=0.35, color='#CBD5E1')
            ax.tick_params(labelsize=8.6, colors='#334155')

        # Row 0: Compression inside pipe vs open-air reference
        ax0 = axes[0, col]
        ax0.plot(t, comp_open, color='#94A3B8', ls=':', lw=1.2, label='无管空载参考')
        ax0.plot(t, comp_3[:, col], color=sp['color'], lw=1.8, label='管内挤压压缩量')
        ax0.set_ylim(-2.0, y_comp_max)
        ax0.set_title(sp['tag'], fontsize=11.0, fontweight='bold', color='#0F172A', pad=7)
        ax0.legend(frameon=False, loc='upper right', fontsize=8.0)
        if col == 0:
            ax0.set_ylabel('前端板压缩量 / mm', fontsize=9.4, color='#0F172A')

        # Row 1: Pipe wall radial anchoring force F_wall
        ax1 = axes[1, col]
        ax1.plot(t, wall_3[:, col], color='#D97706', lw=1.8, label='8 片钢片管壁总撑紧力')
        ax1.set_ylim(-2.0, y_wall_max)
        ax1.legend(frameon=False, loc='upper right', fontsize=8.0)
        if col == 0:
            ax1.set_ylabel('管壁总撑紧力 / N', fontsize=9.4, color='#0F172A')

        # Row 2: Single-tendon T1 & T2 vs open-air T2
        ax2 = axes[2, col]
        if col == 1:
            ax2.fill_between(t, 0, y_ten_max, where=(stick_3[:, col] > 0), color='#E2E8F0', alpha=0.55, label='粘滞区')
        ax2.plot(t, t2_open, color='#94A3B8', ls=':', lw=1.1, label='空载 T2')
        ax2.plot(t, t1_3[:, col], color=sp['t1_color'], ls='--', lw=1.5, label='管内舵机侧 T1')
        ax2.plot(t, t2_3[:, col], color=sp['color'], ls='-', lw=1.8, label='管内体节侧 T2')
        ax2.set_ylim(-1.0, y_ten_max)
        ax2.set_xlabel('时间 / s', fontsize=9.4, color='#0F172A')
        ax2.legend(frameon=False, loc='upper right', fontsize=7.6, ncol=2)
        if col == 0:
            ax2.set_ylabel('单根拉绳张力 / N', fontsize=9.4, color='#0F172A')

    fig.suptitle(
        f"单体节 Ø{PIPE_R_IN_M*2000:.0f} mm 透明圆管内挤压三模型对比（μθ = 0.36，全网格 E=16，dt = 1.0 ms）",
        fontsize=12.6, fontweight='bold', color='#0F172A', y=0.96,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'[Figure] Saved {out_png} ({out_png.stat().st_size / 1024:.1f} KiB)', flush=True)


def render_pipe_3col_video(pack: dict[str, np.ndarray], out_mp4: Path, out_gif: Path, out_preview: Path) -> None:
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from serpworm.probes.render_mujoco import _quat_xyzw_to_wxyz

    W_TOT, H_HDR, H_3D, H_PLOT = 1440, 54, 420, 336
    H_TOT = H_HDR + H_3D + H_PLOT
    W_COL = W_TOT // 3

    x_traj = pack['frames_x'][:, :3]
    fn_traj = pack['frames_fn'][:, :3]
    t_frames = pack['frames_t']
    fps = int(pack['fps'])
    t_s = pack['t_s']
    comp_3 = pack['comp_mm'][:, :3]
    comp_open = pack['comp_mm'][:, 3]
    wall_3 = pack['wall_fn_n'][:, :3]
    t2_3 = pack['t2_single_n'][:, :3]
    t1_3 = pack['t1_single_n'][:, :3]
    stick_3 = pack['stick_flag'][:, :3]

    y_wall_max = max(20.0, float(wall_3.max()) * 1.14)
    y_ten_max = max(24.0, float(max(t1_3.max(), t2_3.max())) * 1.14)

    xml = _build_in_pipe_mjcf(pack['pipe_spec'], W_COL, H_3D)
    model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H_3D, width=W_COL)
    mocap_p0 = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'p0')]
    mocap_p1 = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'p1')]

    # Clean elevated 3/4 view centered on the coaxial acrylic cylinder with clean top headroom
    x_mid = 0.5 * (float(x_traj[0, 0, 0, 0]) + float(x_traj[0, 0, 1, 0]))
    zc = float(pack['pipe_spec'][9])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [x_mid - 0.005, 0.0, zc + 0.022]
    cam.azimuth = 118.0
    cam.elevation = -20.0
    cam.distance = 0.54

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
    preview_fi = int(round(0.42 * (T - 1)))

    t0 = time.perf_counter()
    for fi in range(T):
        idx_k = int(np.argmin(np.abs(t_s - t_frames[fi])))
        canvas = np.full((H_TOT, W_TOT, 3), 255, dtype=np.uint8)

        for col in range(3):
            xr = x_traj[fi, col]
            mj_data.mocap_pos[mocap_p0] = xr[0, :3].astype(np.float64)
            mj_data.mocap_quat[mocap_p0] = _quat_xyzw_to_wxyz(xr[0, 3:7])
            mj_data.mocap_pos[mocap_p1] = xr[1, :3].astype(np.float64)
            mj_data.mocap_quat[mocap_p1] = _quat_xyzw_to_wxyz(xr[1, 3:7])
            mujoco.mj_forward(model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            _add_in_pipe_strips(renderer.scene, xr, fn_traj[fi, col], pack['strip_nodes'], pack['transforms'], pack['anchors'], pack['guides'])
            canvas[H_HDR:H_HDR + H_3D, col * W_COL:(col + 1) * W_COL] = renderer.render()

        rgb_data = _render_pipe_3col_data(t_s, comp_3, comp_open, wall_3, t2_3, t1_3, stick_3, float(t_frames[fi]), W_TOT, H_PLOT, y_wall_max, y_ten_max)
        canvas[H_HDR + H_3D:, :] = rgb_data

        im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(im)

        draw.text((24, 14), f'单体节 Ø{PIPE_R_IN_M*2000:.0f} mm 管道内挤压三模型对比（μθ = 0.36，全网格 E=16）', fill=(15, 23, 42), font=font_title)
        meta_txt = f"t = {t_frames[fi]:.2f} s   |   舵机指令 {pack['cmd_deg'][idx_k]:.0f}°   |   橙色高亮 = 钢片压壁接触区"
        mb = draw.textbbox((0, 0), meta_txt, font=font_sub)
        draw.text((W_TOT - (mb[2] - mb[0]) - 24, 18), meta_txt, fill=(71, 85, 105), font=font_sub)
        draw.line([(0, H_HDR - 1), (W_TOT, H_HDR - 1)], fill=(203, 213, 225), width=1)
        draw.line([(0, H_HDR + H_3D), (W_TOT, H_HDR + H_3D)], fill=(203, 213, 225), width=1)

        for col, sp in enumerate(MODELS):
            x_left = col * W_COL
            if col > 0:
                draw.line([(x_left, H_HDR), (x_left, H_TOT)], fill=(226, 232, 240), width=1)
            draw.text((x_left + 16, H_HDR + 10), sp['tag'], fill=sp['rgb'], font=font_col)
            readout = f"撑壁 {wall_3[idx_k, col]:5.1f} N   压缩 {comp_3[idx_k, col]:4.1f} mm   T2 {t2_3[idx_k, col]:4.1f} / T1 {t1_3[idx_k, col]:4.1f} N"
            draw.text((x_left + 16, H_HDR + 36), readout, fill=(51, 65, 85), font=font_num)

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

    pal = out_gif.parent / '_pipe_pal.png'
    vf = 'fps=15,scale=960:-1:flags=lanczos'
    subprocess.run(['ffmpeg', '-y', '-i', str(out_mp4), '-vf', f'{vf},palettegen=stats_mode=diff', str(pal)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg', '-y', '-i', str(out_mp4), '-i', str(pal), '-lavfi', f'{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3', str(out_gif)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if pal.is_file():
        pal.unlink()
    print(f'[GIF] Saved {out_gif} ({out_gif.stat().st_size / (1024*1024):.2f} MiB)', flush=True)


def save_csv(pack: dict[str, np.ndarray], out_csv: Path) -> None:
    cols = ['t_s', 'cmd_deg']
    for tag in ('pipe_smooth', 'pipe_exact2span', 'pipe_explicit', 'open_exact2span'):
        cols += [f'comp_{tag}_mm', f'wall_fn_{tag}_n', f't2_{tag}_n', f't1_{tag}_n', f'od_{tag}_mm']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(0, len(pack['t_s']), 10):
            row = [f"{pack['t_s'][k]:.3f}", f"{pack['cmd_deg'][k]:.2f}"]
            for ln in range(4):
                row += [
                    f"{pack['comp_mm'][k, ln]:.3f}",
                    f"{pack['wall_fn_n'][k, ln]:.3f}",
                    f"{pack['t2_single_n'][k, ln]:.3f}",
                    f"{pack['t1_single_n'][k, ln]:.3f}",
                    f"{pack['max_r_mm'][k, ln] * 2.0:.2f}",
                ]
            w.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, pack = simulate_1seg_in_pipe()
    (OUT_DIR / 'capstan_1seg_pipe_data.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    save_csv(pack, OUT_DIR / 'capstan_1seg_pipe_timeseries.csv')

    curves_png = OUT_DIR / 'capstan_1seg_pipe_curves.png'
    mp4_path = OUT_DIR / 'capstan_1seg_pipe_demo.mp4'
    gif_path = OUT_DIR / 'capstan_1seg_pipe_demo.gif'
    preview_png = OUT_DIR / 'capstan_1seg_pipe_preview.png'

    render_pipe_3col_figure(summary, pack, curves_png)
    render_pipe_3col_video(pack, mp4_path, gif_path, preview_png)

    if ARTIFACT_DIR is not None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        for src in (curves_png, gif_path, preview_png):
            shutil.copy2(src, ARTIFACT_DIR / src.name)

    print('\n=== IN-PIPE 1-SEG SUMMARY ===', flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

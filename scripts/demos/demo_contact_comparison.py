#!/usr/bin/env python3
"""Contact Model Comparison & 4-Terrain 3D Showcase (Old Kelvin-Voigt vs Implicit ARPC).

Runs three quantitative physics experiments (`--mode simulate`) and renders a
300 DPI publication-grade 2x2 Chinese figure plus a 4-lane side-by-side 3D
terrain video/GIF (`--mode render`):

  Exp 1 (dt = 1.0 ms): 3 mm free drop touchdown smoothness + 4.0 s static hold drift
  Exp 2 (dt = 1.0 ms): 0.7 Hz (25 deg) snake undulation load stability & terrain metrics
  Exp 3 (dt = 1.0 ms): 10.0 s full-mesh (E=16) 4-terrain hybrid snake+worm 3D replay
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PKG_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


TERRAINS: tuple[dict[str, Any], ...] = (
    {
        'key': 'kelvin',
        'label': '旧线性接触',
        'short': '旧线性',
        'color': '#E11D48',
        'accent_rgb': (225, 29, 72),
        'ls': '-',
        'lw': 2.0,
    },
    {
        'key': 'hard',
        'label': 'ARPC 硬地',
        'short': '硬地',
        'color': '#2563EB',
        'accent_rgb': (59, 130, 246),
        'ls': '-',
        'lw': 2.2,
    },
    {
        'key': 'carpet',
        'label': 'ARPC 地毯',
        'short': '地毯',
        'color': '#059669',
        'accent_rgb': (16, 185, 129),
        'ls': '-',
        'lw': 2.0,
    },
    {
        'key': 'soft_soil',
        'label': 'ARPC 软土',
        'short': '软土',
        'color': '#D97706',
        'accent_rgb': (245, 158, 11),
        'ls': '-',
        'lw': 2.0,
    },
)


def _cubic_ramp(t: float, ramp_s: float) -> float:
    if ramp_s <= 0.0:
        return 1.0
    u = max(0.0, min(1.0, t / ramp_s))
    return u * u * (3.0 - 2.0 * u)


def _wheel_kinematics(core):
    """Exact world contact-patch positions, penetration (m), and plate yaw rate (rad/s)."""
    import torch
    from serpworm.physics.gpu_core import matrix, rotate, unit

    x_p = core.x[:, :core.P]
    v_p = core.v[:, :core.P]
    q = x_p[..., 3:]
    R = matrix(q)
    axle = R[..., 1]
    up = torch.zeros_like(axle)
    up[..., 2] = 1.0
    rim = -0.018 * unit(up - axle * axle[..., 2:])
    offset = rotate(
        q[..., None, :].expand(-1, -1, 2, -1),
        core.mounts.expand(core.batch, -1, -1, -1),
    )
    arm = offset + rim[..., None, :]
    contact_pos = x_p[..., None, :3] + arm
    pen = (-contact_pos[..., 2]).clamp_min(0.0)
    wz_plate = v_p[..., None, 5].expand_as(pen)
    return contact_pos, pen, wz_plate


def _pivoting_torque_sum_nm(core, hw, wz_plate) -> float:
    """Total instantaneous rubber-patch vertical pivoting resistance magnitude (N*m)."""
    import torch

    if hw.wheel_patch_radius_m <= 0.0 or hw.wheel_pivot_mu <= 0.0:
        return 0.0
    piv_cap = (2.0 / 3.0) * float(hw.wheel_pivot_mu) * float(hw.wheel_patch_radius_m) * core.support
    tau_z = piv_cap * torch.tanh(wz_plate / 0.15).abs()
    return float(tau_z.sum().item())


def _build_core(terrain_key: str, dt: float, device: str, plate_solver: str, elements: int):
    import torch
    from serpworm.config import HardwareConfig
    from serpworm.physics.gpu_core import GPUCore

    hw = HardwareConfig.ds3218_default().with_terrain(terrain_key)
    solver = plate_solver
    if device == 'cpu' and solver == 'fused_cuda':
        solver = 'segment_lu'
    core = GPUCore(
        count=5,
        batch=1,
        dt=dt,
        device=device,
        dtype=torch.float64,
        plate_solver=solver,
        hardware=hw,
        elements_per_strip=elements,
    )
    return core, hw


def _robot_static_weight_n(core) -> float:
    """Exact total static gravitational weight (N) of the 5-segment robot."""
    grav_load = core.mass_action(core.gravity)
    return float((-grav_load[0, :, 2].sum()).item())


def run_exp1_drop_and_static(
    dt: float,
    device: str,
    plate_solver: str,
    elements: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exp 1: 3 mm free drop touchdown smoothness + 4.0 s static neutral hold at dt=1.0 ms."""
    drop_s = 0.25
    drop_steps = round(drop_s / dt)
    hold_s = 4.0
    hold_steps = round(hold_s / dt)
    hold_stride = max(1, round(0.02 / dt))  # 50 Hz

    drop_results: dict[str, Any] = {'dt_s': dt, 'drop_height_mm': 3.0, 'terrains': {}}
    hold_results: dict[str, Any] = {'dt_s': dt, 'duration_s': hold_s, 'terrains': {}}

    static_weight_n = None

    for spec in TERRAINS:
        key = spec['key']
        # --- Exp 1a: 3 mm free drop (rest has 1.0 mm wheel clearance -> lift +2.0 mm) ---
        core, _ = _build_core(key, dt, device, plate_solver, elements)
        if static_weight_n is None:
            static_weight_n = _robot_static_weight_n(core)
        core.x[..., 2].add_(0.002)

        t_ms: list[float] = [0.0]
        total_fn: list[float] = [0.0]
        mean_fn: list[float] = [0.0]
        contact_frac: list[float] = [0.0]
        mean_pen_mm: list[float] = [0.0]
        mean_rut_mm: list[float] = [0.0]

        for step in range(1, drop_steps + 1):
            core.advance_steps(1)
            sup = core.support[0]
            _, pen, _ = _wheel_kinematics(core)
            t_ms.append(step * dt * 1000.0)
            total_fn.append(float(sup.sum().item()))
            mean_fn.append(float(sup.mean().item()))
            contact_frac.append(float((sup > 1e-3).double().mean().item()))
            mean_pen_mm.append(float(pen[0].mean().item() * 1000.0))
            mean_rut_mm.append(float(core.soil_sinkage[0].mean().item() * 1000.0))

        fn_arr = np.asarray(total_fn, dtype=np.float64)
        mfn_arr = np.asarray(mean_fn, dtype=np.float64)
        cf_arr = np.asarray(contact_frac, dtype=np.float64)
        t_arr = np.asarray(t_ms, dtype=np.float64)

        hit_idx = int(np.argmax(fn_arr > 1e-4)) if np.any(fn_arr > 1e-4) else 0
        first_step_jump_total = float(fn_arr[hit_idx] - fn_arr[max(0, hit_idx - 1)]) if hit_idx > 0 else 0.0
        first_step_jump_wheel = float(mfn_arr[hit_idx] - mfn_arr[max(0, hit_idx - 1)]) if hit_idx > 0 else 0.0
        max_step_jump_wheel = float(np.max(np.diff(mfn_arr))) if len(mfn_arr) > 1 else 0.0
        rebound_lift_ms = float(np.sum((t_arr > t_arr[hit_idx]) & (cf_arr < 0.99)) * (dt * 1000.0)) if hit_idx > 0 else 0.0
        tail_mask = t_arr >= 150.0
        settling_std_n = float(np.std(fn_arr[tail_mask])) if np.any(tail_mask) else 0.0

        drop_results['terrains'][key] = {
            'label': spec['label'],
            't_ms': t_ms,
            'total_normal_n': total_fn,
            'mean_wheel_normal_n': mean_fn,
            'contact_frac': contact_frac,
            'mean_pen_mm': mean_pen_mm,
            'mean_rut_mm': mean_rut_mm,
            'summary': {
                'touchdown_ms': float(t_arr[hit_idx]),
                'first_step_jump_total_n': first_step_jump_total,
                'first_step_jump_wheel_n': first_step_jump_wheel,
                'max_1ms_jump_wheel_n': max_step_jump_wheel,
                'peak_total_normal_n': float(np.max(fn_arr)),
                'peak_wheel_normal_n': float(np.max(mfn_arr)),
                'rebound_lift_ms': rebound_lift_ms,
                'settling_std_n': settling_std_n,
                'final_pen_mm': float(mean_pen_mm[-1]),
                'final_rut_mm': float(mean_rut_mm[-1]),
                'failed': bool(core.failed),
            },
        }
        print(
            f"[Exp1a Drop] {key:10s} | hit={t_arr[hit_idx]:4.1f} ms | "
            f"1st_jump={first_step_jump_wheel:5.3f} N/wheel ({first_step_jump_total:5.2f} N tot) | "
            f"peak={np.max(fn_arr):5.2f} N | bounce={rebound_lift_ms:4.1f} ms | "
            f"tail_std={settling_std_n:.4f} N | rut={mean_rut_mm[-1]:.3f} mm"
        )

        # --- Exp 1b: 4.0 s static neutral hold at dt = 1.0 ms ---
        core, _ = _build_core(key, dt, device, plate_solver, elements)
        x0 = float(core.x[0, :10, 0].mean().item())
        y0 = float(core.x[0, :10, 1].mean().item())

        ht_s: list[float] = [0.0]
        h_fwd_mm: list[float] = [0.0]
        h_lat_mm: list[float] = [0.0]
        h_spd_mm_s: list[float] = [0.0]
        h_vz_mm_s: list[float] = [0.0]
        h_fn_n: list[float] = [0.0]
        h_rut_mm: list[float] = [0.0]

        n_samples = hold_steps // hold_stride
        for s_idx in range(1, n_samples + 1):
            core.advance_steps(hold_stride)
            cx = float(core.x[0, :10, 0].mean().item())
            cy = float(core.x[0, :10, 1].mean().item())
            vx = float(core.v[0, :10, 0].mean().item())
            vy = float(core.v[0, :10, 1].mean().item())
            vz_abs = float(core.v[0, :10, 2].abs().mean().item())
            ht_s.append(s_idx * hold_stride * dt)
            h_fwd_mm.append(-(cx - x0) * 1000.0)
            h_lat_mm.append((cy - y0) * 1000.0)
            h_spd_mm_s.append(math.hypot(vx, vy) * 1000.0)
            h_vz_mm_s.append(vz_abs * 1000.0)
            h_fn_n.append(float(core.support[0].sum().item()))
            h_rut_mm.append(float(core.soil_sinkage[0].mean().item() * 1000.0))

        ht_arr = np.asarray(ht_s, dtype=np.float64)
        hf_arr = np.asarray(h_fwd_mm, dtype=np.float64)
        hs_arr = np.asarray(h_spd_mm_s, dtype=np.float64)
        hfn_arr = np.asarray(h_fn_n, dtype=np.float64)
        steady_idx = int(np.searchsorted(ht_arr, 0.5))
        steady_dt = max(ht_arr[-1] - ht_arr[steady_idx], 1e-6)
        steady_drift_speed = abs(float(hf_arr[-1] - hf_arr[steady_idx])) / steady_dt
        steady_mean_spd = float(np.mean(hs_arr[steady_idx:]))
        steady_fn_std = float(np.std(hfn_arr[steady_idx:]))

        hold_results['terrains'][key] = {
            'label': spec['label'],
            't_s': ht_s,
            'forward_disp_mm': h_fwd_mm,
            'lateral_disp_mm': h_lat_mm,
            'speed_mm_s': h_spd_mm_s,
            'mean_abs_vz_mm_s': h_vz_mm_s,
            'total_normal_n': h_fn_n,
            'mean_rut_mm': h_rut_mm,
            'summary': {
                'total_drift_mm': float(hf_arr[-1]),
                'steady_drift_speed_mm_s': steady_drift_speed,
                'steady_mean_speed_mm_s': steady_mean_spd,
                'max_speed_mm_s': float(np.max(hs_arr[steady_idx:])),
                'steady_normal_std_n': steady_fn_std,
                'final_rut_mm': float(h_rut_mm[-1]),
                'failed': bool(core.failed),
            },
        }
        print(
            f"[Exp1b Hold] {key:10s} | drift={hf_arr[-1]:+6.2f} mm | "
            f"steady_drift={steady_drift_speed:6.3f} mm/s | fn_std={steady_fn_std:6.4f} N | "
            f"rut={h_rut_mm[-1]:.3f} mm"
        )

    drop_results['static_weight_n'] = static_weight_n
    hold_results['static_weight_n'] = static_weight_n
    return drop_results, hold_results


def run_exp2_snake_smoothness(
    dt: float,
    device: str,
    plate_solver: str,
    elements: int,
) -> dict[str, Any]:
    """Exp 2: Identical 0.7 Hz (25 deg) snake wave across all 4 terrains at dt = 1.0 ms."""
    import torch

    duration_s = 6.0
    freq_hz = 0.70
    yaw_amp_deg = 25.0
    neutral_servo_deg = 65.0
    ramp_s = 0.80
    sample_dt = 0.005  # 200 Hz to capture high-frequency normal jitter accurately
    substeps_per_sample = max(1, round(sample_dt / dt))
    n_samples = round(duration_s / (substeps_per_sample * dt))

    out: dict[str, Any] = {
        'dt_s': dt,
        'duration_s': duration_s,
        'freq_hz': freq_hz,
        'yaw_amp_deg': yaw_amp_deg,
        'steady_window_s': [2.0, duration_s],
        'terrains': {},
    }

    for spec in TERRAINS:
        key = spec['key']
        core, hw = _build_core(key, dt, device, plate_solver, elements)
        x0 = float(core.x[0, :10, 0].mean().item())

        t_s: list[float] = [0.0]
        fwd_mm: list[float] = [0.0]
        vel_mm_s: list[float] = [0.0]
        total_fn: list[float] = [0.0]
        lift_pct: list[float] = [100.0]
        piv_mnm: list[float] = [0.0]
        yaw_tau_nm: list[float] = [0.0]
        pen_mm: list[float] = [0.0]
        rut_mm: list[float] = [0.0]

        for s_idx in range(1, n_samples + 1):
            t_cur = (s_idx - 0.5) * substeps_per_sample * dt
            r = _cubic_ramp(t_cur, ramp_s)
            ph = 2.0 * math.pi * freq_hz * t_cur
            yaw_cmd = [
                math.radians(yaw_amp_deg) * r * math.sin(ph - j * 0.5 * math.pi)
                for j in range(4)
            ]
            core.yaw[0].copy_(torch.tensor(yaw_cmd, device=core.device, dtype=core.dtype))
            core.target[0].fill_(math.radians(neutral_servo_deg) * r)
            core.advance_steps(substeps_per_sample)

            cx = float(core.x[0, :10, 0].mean().item())
            vx = float(core.v[0, :10, 0].mean().item())
            sup = core.support[0]
            _, pen, wz = _wheel_kinematics(core)

            t_s.append(s_idx * substeps_per_sample * dt)
            fwd_mm.append(-(cx - x0) * 1000.0)
            vel_mm_s.append(-vx * 1000.0)
            total_fn.append(float(sup.sum().item()))
            lift_pct.append(float((sup <= 1e-3).double().mean().item() * 100.0))
            piv_mnm.append(_pivoting_torque_sum_nm(core, hw, wz) * 1000.0)
            yaw_tau_nm.append(float(core.last_yaw_torque[0].abs().mean().item()))
            pen_mm.append(float(pen[0].mean().item() * 1000.0))
            rut_mm.append(float(core.soil_sinkage[0].mean().item() * 1000.0))

        t_arr = np.asarray(t_s, dtype=np.float64)
        fwd_arr = np.asarray(fwd_mm, dtype=np.float64)
        fn_arr = np.asarray(total_fn, dtype=np.float64)
        lift_arr = np.asarray(lift_pct, dtype=np.float64)
        piv_arr = np.asarray(piv_mnm, dtype=np.float64)
        yt_arr = np.asarray(yaw_tau_nm, dtype=np.float64)
        pen_arr = np.asarray(pen_mm, dtype=np.float64)
        rut_arr = np.asarray(rut_mm, dtype=np.float64)

        mask = t_arr >= 2.0
        idx0 = int(np.argmax(mask))
        steady_dt = max(float(t_arr[-1] - t_arr[idx0]), 1e-6)
        steady_vx = float(fwd_arr[-1] - fwd_arr[idx0]) / steady_dt
        fn_steady = fn_arr[mask]
        # Peak-to-peak jitter over sliding 0.25 s windows (isolates contact chatter from slow gait sway)
        win = max(5, round(0.25 / (substeps_per_sample * dt)))
        pp_vals = [
            float(np.max(fn_steady[i:i + win]) - np.min(fn_steady[i:i + win]))
            for i in range(0, max(1, len(fn_steady) - win + 1), win)
        ]
        normal_pp_jitter = float(np.mean(pp_vals))
        normal_global_pp = float(np.max(fn_steady) - np.min(fn_steady))
        normal_std = float(np.std(fn_steady))
        normal_hf_diff_p95 = float(np.percentile(np.abs(np.diff(fn_steady)), 95))

        out['terrains'][key] = {
            'label': spec['label'],
            't_s': t_s,
            'forward_disp_mm': fwd_mm,
            'forward_vel_mm_s': vel_mm_s,
            'total_normal_n': total_fn,
            'wheel_lift_pct': lift_pct,
            'pivoting_torque_mnm': piv_mnm,
            'mean_abs_yaw_torque_nm': yaw_tau_nm,
            'mean_pen_mm': pen_mm,
            'mean_rut_mm': rut_mm,
            'summary': {
                'forward_vel_mm_s': steady_vx,
                'net_forward_disp_mm': float(fwd_arr[-1]),
                'wheel_lift_pct': float(np.mean(lift_arr[mask])),
                'normal_pp_jitter_n': normal_pp_jitter,
                'normal_global_pp_n': normal_global_pp,
                'normal_std_n': normal_std,
                'normal_5ms_jump_p95_n': normal_hf_diff_p95,
                'pivoting_torque_mean_mnm': float(np.mean(piv_arr[mask])),
                'mean_abs_yaw_torque_nm': float(np.mean(yt_arr[mask])),
                'mean_pen_mm': float(np.mean(pen_arr[mask])),
                'mean_rut_mm': float(np.mean(rut_arr[mask])),
                'max_rut_mm': float(np.max(rut_arr[mask])),
                'failed': bool(core.failed),
            },
        }
        s = out['terrains'][key]['summary']
        print(
            f"[Exp2 Snake] {key:10s} | vx={s['forward_vel_mm_s']:5.1f} mm/s | "
            f"lift={s['wheel_lift_pct']:4.1f}% | Fn_pp={s['normal_pp_jitter_n']:5.2f} N "
            f"(std={s['normal_std_n']:4.2f} N) | piv={s['pivoting_torque_mean_mnm']:5.1f} mN*m | "
            f"pen={s['mean_pen_mm']:.2f} mm | rut={s['mean_rut_mm']:.2f} mm"
        )

    return out


def run_exp3_4terrain_hybrid_traj(
    dt: float,
    device: str,
    plate_solver: str,
    elements: int,
    out_npz: Path,
    duration_s: float = 10.0,
    fps: int = 25,
) -> dict[str, Any]:
    """Exp 3: 4 full E=16 robots side-by-side under identical hybrid snake+worm drive."""
    import torch

    freq_hz = 0.70
    yaw_amp_deg = 22.0
    servo_base_deg = 65.0
    servo_amp_deg = 45.0
    ramp_s = 1.0

    frame_dt = 1.0 / fps
    substeps_per_frame = max(1, round(frame_dt / dt))
    actual_frame_dt = substeps_per_frame * dt
    n_frames = round(duration_s / actual_frame_dt)

    cores = []
    hws = []
    for spec in TERRAINS:
        core, hw = _build_core(spec['key'], dt, device, plate_solver, elements)
        cores.append(core)
        hws.append(hw)

    n_nodes = cores[0].D
    n_lanes = len(TERRAINS)
    T = n_frames + 1

    x_hist = np.zeros((T, n_lanes, n_nodes, 7), dtype=np.float32)
    support_hist = np.zeros((T, n_lanes, 10, 2), dtype=np.float32)
    sinkage_hist = np.zeros((T, n_lanes, 10, 2), dtype=np.float32)
    pen_hist = np.zeros((T, n_lanes, 10, 2), dtype=np.float32)
    play_hist = np.zeros((T, n_lanes, 10, 2), dtype=np.float32)
    contact_xy_hist = np.zeros((T, n_lanes, 10, 2, 2), dtype=np.float32)
    times = np.zeros((T,), dtype=np.float32)

    def _record_frame(fi: int, t_val: float) -> None:
        times[fi] = t_val
        for ln, core in enumerate(cores):
            x_hist[fi, ln] = core.x[0].detach().cpu().numpy().astype(np.float32)
            support_hist[fi, ln] = core.support[0].detach().cpu().numpy().astype(np.float32)
            sinkage_hist[fi, ln] = core.soil_sinkage[0].detach().cpu().numpy().astype(np.float32)
            play_hist[fi, ln] = core.ratchet_play[0].detach().cpu().numpy().astype(np.float32)
            cpos, pen, _ = _wheel_kinematics(core)
            pen_hist[fi, ln] = pen[0].detach().cpu().numpy().astype(np.float32)
            contact_xy_hist[fi, ln] = cpos[0, :, :, :2].detach().cpu().numpy().astype(np.float32)

    x0_list = [float(c.x[0, :10, 0].mean().item()) for c in cores]
    _record_frame(0, 0.0)

    # Update smooth control every 5 ms inside each 40 ms frame for crisp continuous motion
    ctrl_sub = max(1, round(0.005 / dt))
    n_ctrl_chunks = max(1, substeps_per_frame // ctrl_sub)
    rem_sub = substeps_per_frame - n_ctrl_chunks * ctrl_sub

    t0_wall = time.perf_counter()
    for fi in range(1, T):
        t_frame_start = (fi - 1) * actual_frame_dt
        for ch in range(n_ctrl_chunks):
            t_cur = t_frame_start + (ch + 0.5) * ctrl_sub * dt
            r = _cubic_ramp(t_cur, ramp_s)
            ph = 2.0 * math.pi * freq_hz * t_cur
            yaw_cmd = [
                math.radians(yaw_amp_deg) * r * math.sin(ph - j * 0.5 * math.pi)
                for j in range(4)
            ]
            seg_servo = [
                math.radians(servo_base_deg * r + servo_amp_deg * r * math.sin(ph + s * 2.0 * math.pi / 5.0))
                for s in range(5)
            ]
            servo_cmd = [v for v in seg_servo for _ in range(2)]
            for core in cores:
                core.yaw[0].copy_(torch.tensor(yaw_cmd, device=core.device, dtype=core.dtype))
                core.target[0].copy_(torch.tensor(servo_cmd, device=core.device, dtype=core.dtype))
                core.advance_steps(ctrl_sub)
        if rem_sub > 0:
            for core in cores:
                core.advance_steps(rem_sub)
        _record_frame(fi, fi * actual_frame_dt)

    wall_s = time.perf_counter() - t0_wall

    ref = cores[0]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        x=x_hist,
        support=support_hist,
        sinkage=sinkage_hist,
        penetration=pen_hist,
        ratchet_play=play_hist,
        wheel_contact_xy=contact_xy_hist,
        times=times,
        fps=np.int32(fps),
        dt=np.float64(dt),
        strip_nodes=ref.nodes.detach().cpu().numpy(),
        transforms=ref.transforms.detach().cpu().numpy(),
        mounts=ref.mounts.detach().cpu().numpy(),
        anchors=ref.anchors.detach().cpu().numpy(),
        guides=ref.guides.detach().cpu().numpy(),
        terrain_keys=np.asarray([s['key'] for s in TERRAINS]),
        terrain_labels=np.asarray([s['label'] for s in TERRAINS]),
    )

    exp3_summary: dict[str, Any] = {
        'duration_s': float(times[-1]),
        'fps': fps,
        'dt_s': dt,
        'elements_per_strip': elements,
        'num_nodes': int(n_nodes),
        'wall_time_s': round(wall_s, 3),
        'terrains': {},
    }
    steady_fi = int(np.searchsorted(times, 2.0))
    steady_dur = max(float(times[-1] - times[steady_fi]), 1e-6)
    for ln, (spec, core) in enumerate(zip(TERRAINS, cores)):
        cx_end = float(x_hist[-1, ln, :10, 0].mean())
        cx_st = float(x_hist[steady_fi, ln, :10, 0].mean())
        net_disp_mm = -(cx_end - x0_list[ln]) * 1000.0
        steady_vx = -(cx_end - cx_st) * 1000.0 / steady_dur
        sup_st = support_hist[steady_fi:, ln]
        lift_pct = float((sup_st <= 1e-3).mean() * 100.0)
        mean_pen_mm = float(pen_hist[steady_fi:, ln].mean() * 1000.0)
        mean_rut_mm = float(sinkage_hist[steady_fi:, ln].mean() * 1000.0)
        max_rut_mm = float(sinkage_hist[:, ln].max() * 1000.0)
        exp3_summary['terrains'][spec['key']] = {
            'label': spec['label'],
            'net_forward_disp_mm': net_disp_mm,
            'steady_forward_vel_mm_s': steady_vx,
            'wheel_lift_pct': lift_pct,
            'mean_pen_mm': mean_pen_mm,
            'mean_rut_mm': mean_rut_mm,
            'max_rut_mm': max_rut_mm,
            'failed': bool(core.failed),
        }
        print(
            f"[Exp3 10s  ] {spec['key']:10s} | disp={net_disp_mm:6.1f} mm | "
            f"vx={steady_vx:5.1f} mm/s | lift={lift_pct:4.1f}% | "
            f"pen={mean_pen_mm:.2f} mm | rut={mean_rut_mm:.2f} mm (max {max_rut_mm:.2f})"
        )
    print(f"[Exp3 10s  ] Saved {out_npz} ({out_npz.stat().st_size / (1024 * 1024):.2f} MiB) in {wall_s:.2f} s")
    return exp3_summary


def run_simulate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running contact comparison simulations: device={args.device}, "
        f"plate_solver={args.plate_solver}, dt={args.dt*1000:.2f} ms, E={args.elements}"
    )
    t_all = time.perf_counter()
    exp1_drop, exp1_hold = run_exp1_drop_and_static(
        dt=args.dt,
        device=args.device,
        plate_solver=args.plate_solver,
        elements=args.elements,
    )
    exp2_snake = run_exp2_snake_smoothness(
        dt=args.dt,
        device=args.device,
        plate_solver=args.plate_solver,
        elements=args.elements,
    )
    traj_npz = out_dir / 'contact_4terrain_traj.npz'
    exp3_hybrid = run_exp3_4terrain_hybrid_traj(
        dt=args.dt,
        device=args.device,
        plate_solver=args.plate_solver,
        elements=args.elements,
        out_npz=traj_npz,
        duration_s=10.0,
        fps=args.fps,
    )

    metrics = {
        'meta': {
            'dt_s': args.dt,
            'elements_per_strip': args.elements,
            'device': args.device,
            'plate_solver': args.plate_solver,
            'total_wall_time_s': round(time.perf_counter() - t_all, 3),
        },
        'exp1_drop': exp1_drop,
        'exp1_static_hold': exp1_hold,
        'exp2_snake': exp2_snake,
        'exp3_hybrid_4terrain': exp3_hybrid,
    }
    metrics_path = out_dir / 'contact_demo_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Saved metrics JSON: {metrics_path} ({metrics_path.stat().st_size / 1024:.1f} KiB)")


# =============================================================================
# Rendering: 300 DPI Publication 2x2 Chinese Figure
# =============================================================================

def render_comparison_panels(metrics_path: Path, out_png: Path) -> Path:
    """Render 300 DPI crisp 2x2 publication figure from contact_demo_metrics.json."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.linewidth'] = 0.9

    data = json.loads(metrics_path.read_text(encoding='utf-8'))
    exp1_drop = data['exp1_drop']
    exp1_hold = data['exp1_static_hold']
    exp2 = data['exp2_snake']
    weight_n = float(exp1_drop.get('static_weight_n', 30.8))

    fig = plt.figure(figsize=(14.0, 9.4), dpi=300, facecolor='white')
    outer = GridSpec(
        2, 2, figure=fig,
        left=0.065, right=0.975, top=0.902, bottom=0.072,
        wspace=0.22, hspace=0.35,
    )

    def _style_ax(ax) -> None:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        ax.tick_params(colors='#1E293B', labelsize=9.2)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.28, color='#94A3B8')

    # -------------------------------------------------------------------------
    # Panel (a): 3 mm Free Drop Touchdown Normal Force Smoothness
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(outer[0, 0])
    _style_ax(ax_a)
    max_fn_a = weight_n
    for key in ('kelvin', 'hard', 'soft_soil'):
        spec = next(s for s in TERRAINS if s['key'] == key)
        tr = exp1_drop['terrains'][key]
        t_ms = np.asarray(tr['t_ms'])
        fn = np.asarray(tr['total_normal_n'])
        mask = (t_ms >= 15.0) & (t_ms <= 165.0)
        max_fn_a = max(max_fn_a, float(np.max(fn[mask])))
        ls = '--' if key == 'kelvin' else '-'
        z = 4 if key == 'hard' else (3 if key == 'kelvin' else 2)
        ax_a.plot(
            t_ms[mask], fn[mask],
            color=spec['color'], lw=2.1 if key != 'soft_soil' else 1.9,
            ls=ls, label=spec['label'], zorder=z,
        )
    ax_a.axhline(weight_n, color='#64748B', ls=':', lw=1.2, zorder=1, label=f'整机静重 ({weight_n:.1f} N)')

    kel_drop = exp1_drop['terrains']['kelvin']
    hard_drop = exp1_drop['terrains']['hard']
    soil_drop = exp1_drop['terrains']['soft_soil']
    k_fn = np.asarray(kel_drop['total_normal_n'])
    k_t = np.asarray(kel_drop['t_ms'])
    k_hit_i = int(np.argmax(k_fn > 1e-4))
    k_jump_tot = kel_drop['summary']['first_step_jump_total_n']
    h_jump_tot = hard_drop['summary']['first_step_jump_total_n']
    s_jump_tot = soil_drop['summary']['first_step_jump_total_n']
    k_bounce_ms = kel_drop['summary']['rebound_lift_ms']
    s_bounce_ms = soil_drop['summary']['rebound_lift_ms']

    y_top_a = max(max_fn_a * 1.24, weight_n * 1.8)
    ax_a.set_xlim(15.0, 165.0)
    ax_a.set_ylim(-5.0, y_top_a)
    ax_a.set_title('(a) 3 mm 自由落体触地冲击对比', fontsize=11.8, fontweight='bold', color='#0F172A', pad=9)
    ax_a.set_xlabel('时间 / ms', fontsize=10, color='#1E293B')
    ax_a.set_ylabel('整机总法向力 / N', fontsize=10, color='#1E293B')

    # Point directly to Kelvin's discontinuous step jump & rebound off ground
    ax_a.annotate(
        f'旧线性触地阶跃 +{k_jump_tot:.1f} N\n反复弹跳离地 {k_bounce_ms:.0f} ms',
        xy=(float(k_t[k_hit_i]), float(k_fn[k_hit_i])),
        xytext=(50.0, y_top_a * 0.78),
        fontsize=8.4, fontweight='bold', color='#BE123C',
        bbox=dict(boxstyle='round,pad=0.32', fc='#FFF1F2', ec='#FDA4AF', lw=0.9),
        arrowprops=dict(arrowstyle='->', color='#E11D48', lw=1.2, connectionstyle='arc3,rad=-0.12'),
    )
    ax_a.annotate(
        f'ARPC 平滑触地 (硬地 +{h_jump_tot:.1f} N)\n软土 +{s_jump_tot:.1f} N · {s_bounce_ms:.0f} ms 离地',
        xy=(96.0, weight_n * 1.06),
        xytext=(88.0, y_top_a * 0.49),
        fontsize=8.4, fontweight='bold', color='#1D4ED8',
        bbox=dict(boxstyle='round,pad=0.32', fc='#EFF6FF', ec='#93C5FD', lw=0.9),
        arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.2, connectionstyle='arc3,rad=-0.15'),
    )
    ax_a.legend(loc='upper right', fontsize=8.1, frameon=True, facecolor='white', framealpha=0.92, edgecolor='#E2E8F0')

    # -------------------------------------------------------------------------
    # Panel (b): 4.0 s Static Neutral Hold at dt = 1.0 ms
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(outer[0, 1])
    _style_ax(ax_b)
    hold_lines = []
    max_drift_abs = 1.0
    for spec in TERRAINS:
        key = spec['key']
        tr = exp1_hold['terrains'][key]
        t_s = np.asarray(tr['t_s'])
        disp = np.asarray(tr['forward_disp_mm'])
        max_drift_abs = max(max_drift_abs, float(np.max(np.abs(disp))))
        ls = '-' if key in ('kelvin', 'hard') else ('--' if key == 'carpet' else '-.')
        lw = 2.3 if key == 'kelvin' else 2.0
        z = 4 if key == 'kelvin' else (3 if key == 'hard' else 2)
        spd = tr['summary']['steady_drift_speed_mm_s']
        lbl = f"{spec['label']}  ({spd:.2f} mm/s)"
        ax_b.plot(t_s, disp, color=spec['color'], lw=lw, ls=ls, label=lbl, zorder=z)
        hold_lines.append((spec, spd))

    ax_b.axhline(0.0, color='#94A3B8', lw=0.9, ls=':', zorder=1)
    ax_b.set_xlim(0.0, float(exp1_hold['duration_s']))
    ax_b.set_ylim(-max_drift_abs * 0.14, max_drift_abs * 1.18)
    ax_b.set_title('(b) 1.0 ms 步长静态悬停漂移对比', fontsize=11.8, fontweight='bold', color='#0F172A', pad=9)
    ax_b.set_xlabel('时间 / s', fontsize=10, color='#1E293B')
    ax_b.set_ylabel('质心前向漂移 / mm', fontsize=10, color='#1E293B')

    k_spd = exp1_hold['terrains']['kelvin']['summary']['steady_drift_speed_mm_s']
    h_spd = exp1_hold['terrains']['hard']['summary']['steady_drift_speed_mm_s']
    ratio = k_spd / max(h_spd, 1e-4)
    ax_b.text(
        0.04, 0.56,
        f'隐式 ARPC 静态漂移降低 {ratio:.0f}×\n硬地 {h_spd:.2f} mm/s vs 旧线性 {k_spd:.2f} mm/s',
        transform=ax_b.transAxes,
        fontsize=8.8, fontweight='bold', color='#0F172A',
        bbox=dict(boxstyle='round,pad=0.38', fc='#F8FAFC', ec='#CBD5E1', lw=0.95),
    )
    ax_b.legend(loc='upper left', fontsize=8.3, frameon=True, facecolor='white', framealpha=0.94, edgecolor='#E2E8F0')

    # -------------------------------------------------------------------------
    # Panel (c): Snake Undulation Normal Force Load Stability
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(outer[1, 0])
    _style_ax(ax_c)
    fn_lo, fn_hi = weight_n, weight_n
    for spec in TERRAINS:
        key = spec['key']
        tr = exp2['terrains'][key]
        t_s = np.asarray(tr['t_s'])
        fn = np.asarray(tr['total_normal_n'])
        mask = (t_s >= 2.0) & (t_s <= 5.0)
        fn_lo = min(fn_lo, float(np.min(fn[mask])))
        fn_hi = max(fn_hi, float(np.max(fn[mask])))
        alpha = 0.85 if key == 'kelvin' else 0.95
        lw = 1.45 if key == 'kelvin' else 1.85
        z = 2 if key == 'kelvin' else (4 if key == 'hard' else 3)
        pp = tr['summary']['normal_pp_jitter_n']
        ax_c.plot(
            t_s[mask], fn[mask],
            color=spec['color'], lw=lw, alpha=alpha,
            label=f"{spec['label']} (抖动 {pp:.1f} N)", zorder=z,
        )
    ax_c.axhline(weight_n, color='#64748B', ls=':', lw=1.1, zorder=1)
    pad_c = max(6.0, 0.14 * (fn_hi - fn_lo))
    ax_c.set_xlim(2.0, 5.0)
    ax_c.set_ylim(max(0.0, fn_lo - pad_c * 0.85), fn_hi + pad_c * 1.50)
    ax_c.set_title('(c) 0.7 Hz 蛇形游动法向载荷稳定性', fontsize=11.8, fontweight='bold', color='#0F172A', pad=12)
    ax_c.set_xlabel('时间 / s', fontsize=10, color='#1E293B')
    ax_c.set_ylabel('整机总法向支持力 / N', fontsize=10, color='#1E293B')
    ax_c.legend(
        loc='upper center', ncol=2, fontsize=8.1,
        frameon=True, facecolor='white', framealpha=0.93, edgecolor='#E2E8F0',
        columnspacing=1.0, handlelength=1.8,
    )

    # -------------------------------------------------------------------------
    # Panel (d): 4-Terrain Quantitative Metric Dashboard (2x2 clean mini-bars)
    # -------------------------------------------------------------------------
    sub_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1, 1], wspace=0.28, hspace=0.54)
    short_names = [s['short'] for s in TERRAINS]
    bar_colors = [s['color'] for s in TERRAINS]
    x_pos = np.arange(len(TERRAINS))

    vel_vals = [exp2['terrains'][s['key']]['summary']['forward_vel_mm_s'] for s in TERRAINS]
    lift_vals = [exp2['terrains'][s['key']]['summary']['wheel_lift_pct'] for s in TERRAINS]
    jitter_vals = [exp2['terrains'][s['key']]['summary']['normal_pp_jitter_n'] for s in TERRAINS]
    piv_vals = [exp2['terrains'][s['key']]['summary']['pivoting_torque_mean_mnm'] for s in TERRAINS]
    rut_soil = exp2['terrains']['soft_soil']['summary']['mean_rut_mm']

    mini_specs = [
        (0, 0, '前进速度 / (mm/s)', vel_vals, '{:.1f}'),
        (0, 1, '轮子离地率 / %', lift_vals, '{:.1f}%'),
        (1, 0, '法向力峰谷抖动 / N', jitter_vals, '{:.1f}'),
        (1, 1, '转向阻力矩 / (mN·m)', piv_vals, '{:.1f}'),
    ]

    for r_i, c_i, mtitle, vals, fmt in mini_specs:
        ax_m = fig.add_subplot(sub_gs[r_i, c_i])
        _style_ax(ax_m)
        ax_m.grid(axis='x', visible=False)
        bars = ax_m.bar(x_pos, vals, width=0.58, color=bar_colors, edgecolor='none', alpha=0.92, zorder=3)
        vmax = max(max(vals), 1e-3)
        ax_m.set_ylim(0.0, vmax * 1.30)
        ax_m.set_xticks(x_pos)
        ax_m.set_xticklabels(short_names, fontsize=8.3, color='#1E293B')
        ax_m.tick_params(axis='y', labelsize=8.0)
        ax_m.set_title(mtitle, fontsize=9.4, fontweight='bold', color='#0F172A', pad=5)
        for b, v in zip(bars, vals):
            ax_m.text(
                b.get_x() + b.get_width() * 0.5,
                v + vmax * 0.035,
                fmt.format(v),
                ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#1E293B',
            )

    # Panel (d) header positioned cleanly above the 2x2 mini-bar group with generous gap
    d_bbox = outer[1, 1].get_position(fig)
    c_bbox = outer[1, 0].get_position(fig)
    fig.text(
        0.5 * (d_bbox.x0 + d_bbox.x1),
        max(d_bbox.y1, c_bbox.y1) + 0.036,
        f'(d) 四类地形运动与接触指标对比（软土车辙 {rut_soil:.2f} mm）',
        ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#0F172A',
    )

    # Clean figure title + shared color legend bar
    fig.suptitle(
        '隐式 ARPC 地面接触模型与旧线性接触全面对比',
        fontsize=14.2, fontweight='bold', color='#0F172A', y=0.968,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor='white')
    plt.close(fig)
    print(f"Rendered 300 DPI comparison panels: {out_png} ({out_png.stat().st_size / 1024:.1f} KiB)")
    return out_png


# =============================================================================
# Rendering: 3D MuJoCo 4-Terrain Studio Video + GIF + Preview PNG
# =============================================================================

_BLACK_STEEL = np.array([0.075, 0.080, 0.090, 1.0], dtype=np.float32)
SEG_STEEL_RGBA = (_BLACK_STEEL,) * 5
TENDON_RGBA = np.array([0.32, 0.35, 0.40, 0.90], dtype=np.float32)


def _make_terrain_textures(tex_dir: Path) -> dict[str, Path]:
    """Generate crisp procedural 512x512 studio textures for the 4 distinct terrain lanes."""
    from PIL import Image

    tex_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    N = 512
    yy, xx = np.mgrid[0:N, 0:N]
    paths: dict[str, Path] = {}

    # 1. Light Ceramic Tile (Old Linear Contact) — crisp matte lab tile, no glare washout
    tile = np.full((N, N, 3), [228, 234, 241], dtype=np.float32)
    checker = ((xx // 128) + (yy // 128)) % 2 == 0
    tile[checker] = np.array([216, 223, 232], dtype=np.float32)
    grout = (xx % 128 < 5) | (yy % 128 < 5)
    tile[grout] = np.array([178, 189, 203], dtype=np.float32)
    tile += rng.normal(0.0, 1.2, tile.shape)
    p_tile = tex_dir / '_tex_tile.png'
    Image.fromarray(np.clip(tile, 0, 255).astype(np.uint8)).save(p_tile)
    paths['kelvin'] = p_tile

    # 2. Brushed Rubber / Matte Slate Lab Floor (ARPC Hard)
    rubber = np.full((N, N, 3), [154, 165, 178], dtype=np.float32)
    streak = rng.normal(0.0, 4.8, (1, N, 1))
    fine = rng.normal(0.0, 3.2, (N, N, 1))
    rubber += streak * 0.65 + fine
    seam = (xx % 256 < 4) | (yy % 256 < 4)
    rubber[seam] -= 20.0
    p_hard = tex_dir / '_tex_hard.png'
    Image.fromarray(np.clip(rubber, 0, 255).astype(np.uint8)).save(p_hard)
    paths['hard'] = p_hard

    # 3. Woven Textile Carpet (ARPC Carpet)
    carpet = np.full((N, N, 3), [108, 134, 152], dtype=np.float32)
    weave_a = ((xx // 6) + (yy // 6)) % 2 == 0
    weave_b = ((xx % 6) < 2) | ((yy % 6) < 2)
    carpet[weave_a] += np.array([9.0, 10.0, 11.0], dtype=np.float32)
    carpet[weave_b] -= np.array([12.0, 13.0, 14.0], dtype=np.float32)
    carpet += rng.normal(0.0, 4.2, carpet.shape)
    p_carpet = tex_dir / '_tex_carpet.png'
    Image.fromarray(np.clip(carpet, 0, 255).astype(np.uint8)).save(p_carpet)
    paths['carpet'] = p_carpet

    # 4. Warm Granular Sand-Soil Bed (ARPC Soft Soil)
    soil = np.full((N, N, 3), [204, 168, 122], dtype=np.float32)
    # Multi-scale sand ripples & granular speckles
    ripple = 6.5 * np.sin(2.0 * math.pi * (xx / 64.0) + 0.6 * np.sin(2.0 * math.pi * yy / 128.0))
    soil += ripple[..., None]
    grain = rng.normal(0.0, 8.5, (N, N, 1))
    soil += grain * np.array([1.0, 0.90, 0.78], dtype=np.float32)
    pebbles = rng.random((N, N)) < 0.012
    soil[pebbles] -= np.array([28.0, 24.0, 18.0], dtype=np.float32)
    p_soil = tex_dir / '_tex_soil.png'
    Image.fromarray(np.clip(soil, 0, 255).astype(np.uint8)).save(p_soil)
    paths['soft_soil'] = p_soil

    return paths


def _build_4terrain_mjcf(
    mounts: np.ndarray,
    start_head_x: float,
    lane_ys: list[float],
    tex_paths: dict[str, Path],
    width: int,
    height: int,
) -> str:
    """Build 4-lane studio MJCF with distinct terrain test beds and CAD STL robot geometry."""
    from serpworm.paths import ASSETS_DIR

    mesh_dir = (ASSETS_DIR / 'cad' / 'meshes' / 'longworm2').resolve().as_posix()
    bodies = []
    for r in range(4):
        for p in range(10):
            mesh_name = 'plate_front' if p % 2 == 0 else 'plate_back'
            mat_name = 'mat_plate_front' if p % 2 == 0 else 'mat_plate_back'
            w_geoms = []
            for s in range(2):
                mx, my, mz = mounts[p, s]
                strut_top = -0.064
                strut_mid_z = 0.5 * (strut_top + mz)
                strut_half = max(0.005, 0.5 * abs(strut_top - mz))
                w_geoms.append(
                    f'<geom type="box" pos="{mx:.5f} {my:.5f} {strut_mid_z:.5f}" '
                    f'size="0.005 0.0035 {strut_half:.5f}" material="mat_strut" contype="0" conaffinity="0"/>'
                )
                w_geoms.append(
                    f'<geom type="mesh" mesh="wheel_mesh" pos="{mx:.5f} {my:.5f} {mz:.5f}" '
                    f'material="mat_wheel" contype="0" conaffinity="0"/>'
                )
            bodies.append(
                f'    <body name="r{r}_p{p}" mocap="true" pos="0 0 0.13">\n'
                f'      <geom type="mesh" mesh="{mesh_name}" material="{mat_name}" contype="0" conaffinity="0"/>\n'
                + '\n'.join(f'      {g}' for g in w_geoms) + '\n'
                f'    </body>'
            )

    lane_half_w = 0.215
    lane_len_half = 2.80
    lane_center_x = -0.95
    lane_mats = ['mat_lane_kelvin', 'mat_lane_hard', 'mat_lane_carpet', 'mat_lane_soil']

    terrain_geoms = []
    for idx, (y_c, m_name) in enumerate(zip(lane_ys, lane_mats)):
        # Top surface strictly at z = 0.0000 so wheels sit flush (and visibly sink into carpet/soil!)
        terrain_geoms.append(
            f'    <geom name="lane_{idx}" type="box" pos="{lane_center_x:.3f} {y_c:.4f} -0.0025" '
            f'size="{lane_len_half:.3f} {lane_half_w:.4f} 0.0025" material="{m_name}" contype="0" conaffinity="0"/>'
        )

    # Clean flush terrain lanes with subtle thin seam lines (no raised outer rails or noisy grid bars)
    y_min = lane_ys[-1] - lane_half_w
    y_max = lane_ys[0] + lane_half_w
    y_mid = 0.5 * (y_min + y_max)
    y_half = 0.5 * (y_max - y_min)
    markers = [
        f'    <geom name="start_line" type="box" pos="{start_head_x:.4f} {y_mid:.4f} 0.0004" '
        f'size="0.0028 {y_half:.4f} 0.0004" rgba="0.72 0.26 0.26 0.55" contype="0" conaffinity="0"/>'
    ]

    return f"""<mujoco model="serpworm_4terrain_contact_studio">
  <compiler angle="radian" meshdir="{mesh_dir}" autolimits="true"/>
  <visual>
    <global offwidth="{width}" offheight="{height}"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.05" zfar="20.0" shadowclip="2.5"/>
    <headlight ambient="0.19 0.20 0.22" diffuse="0.06 0.06 0.06" specular="0.0 0.0 0.0"/>
    <rgba haze="0.78 0.81 0.85 1"/>
  </visual>
  <asset>
    <mesh name="plate_front" file="front2_Link.STL"/>
    <mesh name="plate_back"  file="back2_Link.STL"/>
    <mesh name="wheel_mesh"  file="w2-1_Link.STL"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.72 0.76 0.81" rgb2="0.82 0.85 0.89" width="512" height="512"/>
    <texture name="tex_tile"   type="2d" file="{tex_paths['kelvin'].resolve().as_posix()}"/>
    <texture name="tex_hard"   type="2d" file="{tex_paths['hard'].resolve().as_posix()}"/>
    <texture name="tex_carpet" type="2d" file="{tex_paths['carpet'].resolve().as_posix()}"/>
    <texture name="tex_soil"   type="2d" file="{tex_paths['soft_soil'].resolve().as_posix()}"/>
    <material name="mat_floor_bg"    rgba="0.68 0.71 0.75 1.0" specular="0.02" shininess="0.05"/>
    <material name="mat_lane_kelvin" texture="tex_tile"   texrepeat="18 2" texuniform="false" reflectance="0.02" shininess="0.06" specular="0.04"/>
    <material name="mat_lane_hard"   texture="tex_hard"   texrepeat="16 2" texuniform="false" reflectance="0.02" shininess="0.06" specular="0.04"/>
    <material name="mat_lane_carpet" texture="tex_carpet" texrepeat="20 3" texuniform="false" reflectance="0.00" shininess="0.03" specular="0.02"/>
    <material name="mat_lane_soil"   texture="tex_soil"   texrepeat="18 2" texuniform="false" reflectance="0.00" shininess="0.02" specular="0.02"/>
    <material name="mat_plate_front" rgba="0.78 0.82 0.87 0.72" specular="0.85" shininess="0.80" reflectance="0.10"/>
    <material name="mat_plate_back"  rgba="0.34 0.38 0.44 1.0"  specular="0.65" shininess="0.60" reflectance="0.08"/>
    <material name="mat_strut"       rgba="0.44 0.48 0.54 1.0"  specular="0.58" shininess="0.52"/>
    <material name="mat_wheel"       rgba="0.14 0.15 0.17 1.0"  specular="0.22" shininess="0.20"/>
  </asset>
  <worldbody>
    <light name="key"  directional="true" pos="0 0 3.2" dir="0.35 0.42 -0.84" diffuse="0.52 0.51 0.50" specular="0.90 0.90 0.88" castshadow="true"/>
    <light name="rim"  directional="true" pos="0 0 2.6" dir="-0.55 -0.35 -0.76" diffuse="0.16 0.18 0.21" specular="0.82 0.84 0.88" castshadow="false"/>
    <light name="fill" directional="true" pos="0 0 2.4" dir="-0.25 0.65 -0.72" diffuse="0.14 0.16 0.18" specular="0.18 0.18 0.20" castshadow="false"/>
    <geom name="floor_bg" type="plane" pos="-0.8 0 -0.0055" size="7 7 0.1" material="mat_floor_bg"/>
{chr(10).join(terrain_geoms)}
{chr(10).join(markers)}
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""


def _add_ground_deformation_cues(
    scene,
    fi: int,
    lane_ys: list[float],
    contact_xy_hist: np.ndarray,  # (T, 4, 10, 2, 2)
    sinkage_hist: np.ndarray,     # (T, 4, 10, 2)
    pen_hist: np.ndarray,         # (T, 4, 10, 2)
    support_hist: np.ndarray,     # (T, 4, 10, 2)
) -> None:
    """Render live elastic carpet depressions and persistent soft-soil plastic rut tracks."""
    import mujoco

    eye_rot = np.eye(3, dtype=np.float64).reshape(9)

    # 1. Lane 3 (Soft Soil): continuous dark plastic rut tracks left behind rolling wheels
    soil_ln = 3
    y_off_soil = lane_ys[soil_ln]
    # Track per-wheel anchor position and emit seamless flat rut boxes every >= 12 mm of motion
    for p in range(10):
        for s in range(2):
            last_pt = None
            for tf in range(1, fi + 1):
                rut_m = float(sinkage_hist[tf, soil_ln, p, s])
                if rut_m < 0.00022:
                    last_pt = None
                    continue
                cur_xy = contact_xy_hist[tf, soil_ln, p, s].astype(np.float64)
                if last_pt is None:
                    last_pt = cur_xy
                    continue
                diff = cur_xy - last_pt
                dist = float(np.hypot(diff[0], diff[1]))
                if dist < 0.012 and tf < fi:
                    continue
                if dist < 1e-4 or scene.ngeom >= scene.maxgeom - 80:
                    continue
                depth_norm = min(1.0, rut_m / 0.0014)
                ex = np.array([diff[0] / dist, diff[1] / dist, 0.0], dtype=np.float64)
                ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                ey = np.cross(ez, ex)
                rot = np.column_stack((ex, ey, ez)).reshape(9)
                mid = np.array([
                    0.5 * (last_pt[0] + cur_xy[0]),
                    0.5 * (last_pt[1] + cur_xy[1]) + y_off_soil,
                    0.00045,
                ], dtype=np.float64)
                size = np.array([0.55 * dist + 0.006, 0.0105, 0.0004], dtype=np.float64)
                rgba = np.array([0.48, 0.33, 0.17, 0.28 + 0.38 * depth_norm], dtype=np.float32)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    int(mujoco.mjtGeom.mjGEOM_BOX),
                    size, mid, rot, rgba,
                )
                scene.ngeom += 1
                last_pt = cur_xy
            # Also add current wheel footprint groove on soft soil
            rut_now = float(sinkage_hist[fi, soil_ln, p, s])
            if rut_now >= 0.00022 and scene.ngeom < scene.maxgeom - 60:
                depth_norm = min(1.0, rut_now / 0.0014)
                xy_now = contact_xy_hist[fi, soil_ln, p, s]
                pos = np.array([float(xy_now[0]), float(xy_now[1]) + y_off_soil, 0.0005], dtype=np.float64)
                rgba = np.array([0.38, 0.25, 0.12, 0.30 + 0.36 * depth_norm], dtype=np.float32)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    np.array([0.0115, 0.00035, 0.0], dtype=np.float64),
                    pos, eye_rot, rgba,
                )
                scene.ngeom += 1

    # 2. Lane 2 (Carpet): instantaneous elastic pile depression halo under loaded wheels (rebounds immediately!)
    carpet_ln = 2
    y_off_carpet = lane_ys[carpet_ln]
    for p in range(10):
        for s in range(2):
            pen_m = float(pen_hist[fi, carpet_ln, p, s])
            if pen_m < 0.0004 or scene.ngeom >= scene.maxgeom - 20:
                continue
            d_norm = min(1.0, pen_m / 0.0022)
            xy = contact_xy_hist[fi, carpet_ln, p, s]
            pos = np.array([float(xy[0]), float(xy[1]) + y_off_carpet, 0.0004], dtype=np.float64)
            rgba = np.array([0.16, 0.24, 0.30, 0.25 + 0.35 * d_norm], dtype=np.float32)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                np.array([0.0155, 0.00035, 0.0], dtype=np.float64),
                pos, eye_rot, rgba,
            )
            scene.ngeom += 1

    # 3. Lane 0 (Old Kelvin) & Lane 1 (ARPC Hard): subtle grounded contact footprint cue
    for ln in (0, 1):
        y_off = lane_ys[ln]
        for p in range(10):
            for s in range(2):
                fn = float(support_hist[fi, ln, p, s])
                if fn < 0.05 or scene.ngeom >= scene.maxgeom - 20:
                    continue
                xy = contact_xy_hist[fi, ln, p, s]
                pos = np.array([float(xy[0]), float(xy[1]) + y_off, 0.0004], dtype=np.float64)
                alpha = min(0.32, 0.08 + 0.12 * (fn / 2.0))
                rgba = np.array([0.12, 0.16, 0.22, alpha], dtype=np.float32)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    np.array([0.010, 0.0003, 0.0], dtype=np.float64),
                    pos, eye_rot, rgba,
                )
                scene.ngeom += 1


def _draw_compact_lane_pills(
    img: np.ndarray,
    tags: list[tuple[str, tuple[int, int] | None, tuple[int, int, int]]],
) -> np.ndarray:
    """Draw clean borderless, dot-free, background-free floating labels with vertical anti-overlap."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, 'RGBA')
    font = None
    for fpath in (
        'C:/Windows/Fonts/msyhbd.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
    ):
        if Path(fpath).is_file():
            font = ImageFont.truetype(fpath, 21)
            break
    if font is None:
        font = ImageFont.load_default()

    W, H = im.size
    placed: list[tuple[int, int, int, int, str]] = []
    for text, uv, _accent in tags:
        if uv is None:
            continue
        u, v = uv
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x0 = max(14, min(W - tw - 14, u - tw // 2))
        y0 = max(14, min(H - th - 14, v - th // 2))
        placed.append((x0, y0, tw, th, text))

    # Gentle vertical de-overlap pass
    for i in range(len(placed)):
        xi, yi, bwi, bhi, ti = placed[i]
        for j in range(i):
            xj, yj, bwj, bhj, _ = placed[j]
            if xi < xj + bwj + 8 and xi + bwi + 8 > xj:
                if abs(yi - yj) < max(bhi, bhj) + 6:
                    yi = min(H - bhi - 14, yj + bhj + 8)
        placed[i] = (xi, yi, bwi, bhi, ti)

    # Pure clean typography only — no dots, no borders, no background box
    for x0, y0, tw, th, text in placed:
        patch = img[max(0, y0):min(H, y0 + th), max(0, x0):min(W, x0 + tw)]
        lum = float(patch.mean()) if patch.size > 0 else 180.0
        if lum < 120.0:
            fg = (246, 248, 252, 255)
            shadow = (15, 23, 42, 130)
        else:
            fg = (24, 32, 46, 255)
            shadow = (255, 255, 255, 135)
        draw.text((x0, y0 + 1), text, fill=shadow, font=font)
        draw.text((x0, y0), text, fill=fg, font=font)

    return np.asarray(im)


def render_4terrain_video_and_gif(
    traj_npz: Path,
    out_mp4: Path,
    out_gif: Path,
    out_preview: Path,
    width: int = 1280,
    height: int = 720,
    show_tags: bool = True,
) -> None:
    """Render 4-lane side-by-side 3D studio showcase MP4, palette-optimized GIF, and PNG preview."""
    import mujoco
    from PIL import Image
    from serpworm.probes.render_mujoco import (
        _add_robot_strips_and_tendons,
        _project_world_to_pixel,
        _quat_xyzw_to_wxyz,
    )

    data = np.load(traj_npz)
    x_traj = data['x']                     # (T, 4, 610, 7)
    support_traj = data['support']         # (T, 4, 10, 2)
    sinkage_traj = data['sinkage']         # (T, 4, 10, 2)
    pen_traj = data['penetration']         # (T, 4, 10, 2)
    cxy_traj = data['wheel_contact_xy']    # (T, 4, 10, 2, 2)
    strip_nodes = data['strip_nodes']
    transforms = data['transforms']
    mounts = data['mounts']
    anchors = data['anchors']
    guides = data['guides']
    fps = int(data['fps'])

    lane_ys = [0.72, 0.24, -0.24, -0.72]
    start_head_x = float(x_traj[0, 0, 9, 0])

    import tempfile
    with tempfile.TemporaryDirectory(prefix='serpworm_4terrain_tex_') as tmp_tex_dir:
        tex_paths = _make_terrain_textures(Path(tmp_tex_dir))
        xml = _build_4terrain_mjcf(mounts, start_head_x, lane_ys, tex_paths, width, height)
        model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    mocap_ids = [
        [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'r{r}_p{p}')] for p in range(10)]
        for r in range(4)
    ]

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 104.0
    cam.elevation = -42.5
    cam.distance = 2.22
    fovy_deg = float(model.vis.global_.fovy)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    T = x_traj.shape[0]
    preview_fi = min(T - 1, int(round(0.86 * (T - 1))))  # ~8.6 s: clear terrain separation + rich soil tracks
    labels = [s['label'] for s in TERRAINS]
    accents = [s['accent_rgb'] for s in TERRAINS]

    for fi in range(T):
        cx_sum = 0.0
        tag_pts = []
        for r, y_off in enumerate(lane_ys):
            xr = x_traj[fi, r]
            for p in range(10):
                mid_id = mocap_ids[r][p]
                pos = xr[p, :3].astype(np.float64).copy()
                pos[1] += y_off
                mj_data.mocap_pos[mid_id] = pos
                mj_data.mocap_quat[mid_id] = _quat_xyzw_to_wxyz(xr[p, 3:7])
            p_cur = xr[:10, :3].astype(np.float64)
            cx_sum += float(p_cur[:, 0].mean())
            # Anchor clean text well ahead of each robot nose with zero overlap
            anchor = p_cur[9].copy()
            anchor[0] -= 0.34
            anchor[1] += y_off
            anchor[2] = 0.035
            tag_pts.append(anchor)

        mujoco.mj_forward(model, mj_data)
        cam.lookat[:] = [cx_sum / 4.0 - 0.06, -0.06, 0.06]
        renderer.update_scene(mj_data, camera=cam)

        # Add 40-strip robots first (guaranteed within maxgeom), then ground deformation & rut tracks
        for r, y_off in enumerate(lane_ys):
            _add_robot_strips_and_tendons(
                renderer.scene, x_traj[fi, r], y_off,
                strip_nodes, transforms, anchors, guides,
            )
        _add_ground_deformation_cues(
            renderer.scene, fi, lane_ys, cxy_traj, sinkage_traj, pen_traj, support_traj,
        )

        rgb = renderer.render()
        if show_tags:
            tags = [
                (labels[r], _project_world_to_pixel(tag_pts[r], cam, fovy_deg, width, height), accents[r])
                for r in range(4)
            ]
            rgb = _draw_compact_lane_pills(rgb, tags)

        if fi == preview_fi:
            Image.fromarray(rgb).save(out_preview)
        proc.stdin.write(rgb.tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode('utf-8', errors='replace')
    rc = proc.wait()
    renderer.close()
    if rc != 0:
        raise RuntimeError(f'ffmpeg mp4 encode failed ({rc}): {err[-400:]}')
    print(f"Rendered 3D MP4: {out_mp4} ({out_mp4.stat().st_size / (1024 * 1024):.2f} MiB)")
    print(f"Saved 3D preview: {out_preview} ({out_preview.stat().st_size / 1024:.1f} KiB)")

    # Crisp 2-pass palette-optimized GIF from MP4
    palette_png = out_gif.parent / '_gif_palette.png'
    gif_vf = 'fps=12,scale=720:-1:flags=lanczos'
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(out_mp4), '-vf', f'{gif_vf},palettegen=stats_mode=diff', str(palette_png)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            'ffmpeg', '-y', '-i', str(out_mp4), '-i', str(palette_png),
            '-lavfi', f'{gif_vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3',
            str(out_gif),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if palette_png.is_file():
        palette_png.unlink()
    print(f"Rendered 3D GIF: {out_gif} ({out_gif.stat().st_size / (1024 * 1024):.2f} MiB)")


def run_render(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    metrics_path = out_dir / 'contact_demo_metrics.json'
    traj_npz = out_dir / 'contact_4terrain_traj.npz'
    if not metrics_path.is_file():
        raise FileNotFoundError(f'Missing metrics JSON: {metrics_path} (run --mode simulate first)')
    if not traj_npz.is_file():
        raise FileNotFoundError(f'Missing 4-terrain trajectory NPZ: {traj_npz} (run --mode simulate first)')

    render_comparison_panels(metrics_path, out_dir / 'contact_comparison_panels.png')
    render_4terrain_video_and_gif(
        traj_npz=traj_npz,
        out_mp4=out_dir / 'contact_4terrain_demo.mp4',
        out_gif=out_dir / 'contact_4terrain_demo.gif',
        out_preview=out_dir / 'contact_4terrain_preview.png',
        width=args.width,
        height=args.height,
        show_tags=not args.no_tags,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Contact model comparison simulation & 4-terrain 3D visual demo')
    p.add_argument('--mode', choices=('simulate', 'render', 'all'), default='all')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--plate-solver', choices=('fused_cuda', 'segment_lu', 'legacy'), default='fused_cuda')
    p.add_argument('--dt', type=float, default=0.001, help='Physics timestep in seconds (default: 0.001 = 1.0 ms)')
    p.add_argument('--elements', type=int, default=16, help='Beam elements per strip (default: 16)')
    p.add_argument('--fps', type=int, default=25, help='Trajectory & video frame rate (default: 25)')
    p.add_argument('--width', type=int, default=1280, help='3D render width in pixels')
    p.add_argument('--height', type=int, default=720, help='3D render height in pixels')
    p.add_argument('--no-tags', action='store_true', help='Disable floating lane labels')
    p.add_argument('--out', type=Path, default=PKG_ROOT / 'runs' / 'demos', help='Output directory')
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mode in ('simulate', 'all'):
        run_simulate(args)
    if args.mode in ('render', 'all'):
        run_render(args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Nature Communications Fig. 3 Physics Phase Diagram, STFT & CoT Mechanics Suite.

Runs 100% live FP64 GPU physics (RTX 5060 `fused_cuda`) using existing trained weights
(zero retraining, zero invented numbers) to produce:
  (a) 2D Physical Phase Diagram across 7 spatial confinements (Ø216..270 mm + Open)
      x 6 terrain rolling-drag/compliance levels (mu_rr = 0.025..0.230)
  (b) Continuous 3-zone phase-space trajectory (A_yaw(t), A_tendon(t)) from the
      verified 25.96 s Hard -> Ø236 mm Pipe -> Soft Soil rollout
  (c) Dual-channel traveling-wave STFT time-frequency spectrogram (yaw undulation
      vs 5-segment axial peristaltic compression) across the 25.96 s phase transition
  (d) Cross-medium Dimensionless Cost of Transport (CoT = P_elec / (mg * v_x)) & speed
      comparison across v6b (65D Autonomous), v6.1 (65D + Elec), and v5 Fixed Snake /
      Worm / Mixed baselines on Hard Floor, Ø236 mm Pipe, and Soft Soil.

Outputs saved to `runs/phase_diagram_cot/`:
  - phase_diagram_cot_summary.json
  - fig_phase_diagram_and_cot.{png,pdf}
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from dataclasses import fields as dc_fields
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

PKG_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PKG_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serpworm.config import DomainRandConfig, HardwareConfig, TaskConfig
from serpworm.envs.autonomous_env import AutonomousPhaseEnv, AutonomousTaskConfig, TERRAIN_PROFILES
from serpworm.envs.matsuoka import Matsuoka
from serpworm.envs.unified_env import UnifiedSerpWormEnv
from serpworm.physics.fused_cuda import is_available as _fused_cuda_ok
from serpworm.physics.gpu_core import GPUCore
from serpworm.physics.pipe_geometry import make_off_pipe
from serpworm.rl.models import ProprioceptiveEstimatorActorCritic, TemporalActorCritic

if torch.cuda.is_available() and not _fused_cuda_ok():
    # Bypass CUDA Graph capture in eager PyTorch CUDA fallback (gpu_core.external has host branch on pipe_cfg)
    GPUCore.capture = lambda self, steps: None  # type: ignore[assignment]

OUT_DIR = PKG_ROOT / 'runs' / 'phase_diagram_cot'
_ARTIFACT_ENV = os.environ.get('SERPWORM_ARTIFACT_DIR')
ARTIFACT_DIR: Path | None = Path(_ARTIFACT_ENV) if _ARTIFACT_ENV else None

# 7 spatial confinements (outer compressed body diameter D_0 = 200 mm -> ID = 200 + 2000*y_allow mm)
PIPE_CONFINEMENTS = (
    ('Ø216', 0.008, 216.0),
    ('Ø224', 0.012, 224.0),
    ('Ø232', 0.016, 232.0),
    ('Ø240', 0.020, 240.0),
    ('Ø252', 0.026, 252.0),
    ('Ø270', 0.035, 270.0),
    ('Open', 1e6,   float('inf')),
)

# 6 smooth physical terrain levels spanning Hard Floor -> Dense Mat -> Carpet -> Loam -> Soft Soil
# [label, mu_rr, hc_k, unload_ratio, rut_lat_k, mu_lat_ratio, mu_fwd]
TERRAIN_GRID = (
    ('0.025\n(Hard)',   0.025, 1.60e5, 1.00,   0.0, 1.05, 0.80),
    ('0.060\n(Rough)',  0.060, 8.50e4, 1.00,   0.0, 1.10, 0.83),
    ('0.100\n(Mat)',    0.100, 3.80e4, 1.00,   0.0, 1.15, 0.86),
    ('0.140\n(Carpet)', 0.140, 1.80e4, 1.00,   0.0, 1.20, 0.90),
    ('0.185\n(Loam)',   0.185, 1.45e4, 2.25, 325.0, 1.28, 0.78),
    ('0.230\n(Soil)',   0.230, 1.15e4, 3.60, 680.0, 1.36, 0.69),
)


def _electrical_power_step(env, old_servo: torch.Tensor, dt_step: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact mechanical + 7.4V/3A DC bus-servo copper Joule heat power (Watts) per env."""
    c = env.c
    hw = env.hw
    # Crank lever arm for 10 tendon servos
    r_h = c.horn_radius
    g_d = c.guide_dist
    chord = torch.sqrt(g_d**2 + r_h**2 - 2.0 * g_d * r_h * torch.cos(c.servo)).clamp_min(1e-9)
    lever = (r_h * g_d * torch.sin(c.servo).abs() / chord).clamp_max(r_h)
    sv_torque = c.tension.reshape(env.n, 10, 2).sum(-1) * lever
    yaw_torque = c.last_yaw_torque

    # 7.4V / 3A stall Joule copper loss (22.0 W at stall torque tau_lim = 2.45 N*m per servo)
    tau_y_lim = c.yaw_torque_limit[:, None].clamp_min(1e-3)
    tau_s_lim = c.servo_torque_limit[:, None].clamp_min(1e-3)
    p_joule = 22.0 * (
        (yaw_torque / tau_y_lim).square().sum(-1)
        + (sv_torque / tau_s_lim).square().sum(-1)
    )

    # Mechanical shaft power from joint velocities & torques
    R = env.c.x[:, :10, 3:]
    from serpworm.physics.gpu_core import matrix
    Rm = matrix(R)
    yaw_rate = ((c.v[:, 2:10:2, 3:] - c.v[:, 1:9:2, 3:]) * Rm[:, 1:9:2, :, 2]).sum(-1)
    sv_rate = (c.servo - old_servo).abs() / max(dt_step, 1e-6)
    p_mech = (yaw_torque * yaw_rate).abs().sum(-1) + (sv_torque * sv_rate).abs().sum(-1)
    return p_mech + p_joule, p_mech, p_joule


def _robot_weight_n(env, hw: HardwareConfig) -> float:
    plate_mass = float(env.c.plate_masses[0].sum().item())
    # 40 pre-curved 65Mn steel strips (145 mm x 16 mm x 0.22 mm, rho = 7850 kg/m^3 -> ~0.160 kg)
    w_s = float(hw.strip_width_m) if hw.strip_width_m is not None else 0.016
    t_s = float(hw.strip_thickness_m) if hw.strip_thickness_m is not None else 0.00022
    strip_mass = 40.0 * 0.145 * w_s * t_s * 7850.0
    return (plate_mass + strip_mass) * 9.81


def _load_v6_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ProprioceptiveEstimatorActorCritic(
        frame_size=ckpt['frame_size'],
        history_len=ckpt['history_len'],
        privileged_size=ckpt['privileged_size'],
        est_size=ckpt['est_size'],
        action_size=ckpt['action_size'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def _build_v6_env(ckpt: dict[str, Any], n: int, seconds: float, device: str, settle_ticks: int = 25, elements_per_strip: int = 8):
    hw_fields = {f.name for f in dc_fields(HardwareConfig)}
    hw = HardwareConfig(**{k: v for k, v in ckpt['hardware'].items() if k in hw_fields})
    t_fields = {f.name for f in dc_fields(AutonomousTaskConfig)}
    td = {k: v for k, v in ckpt['task'].items() if k in t_fields}
    td['episode_seconds'] = seconds + 5.0
    td['command_switch_s'] = 1e6
    td['elements_per_strip'] = int(elements_per_strip)
    task = AutonomousTaskConfig(**td)
    env = AutonomousPhaseEnv(
        n=n, hardware=hw, task=task, dr=DomainRandConfig(enabled=False),
        device=device, plate_solver='fused_cuda' if device == 'cuda' else 'segment_lu',
        noise=False, forced_regime=0, settle_ticks=settle_ticks,
    )
    return env, hw, task


def run_2d_phase_diagram_and_v6_cot(device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Combined GPU batch (B=48): 42 cells for v6b 2D phase diagram + 3 v6b & 3 v6.1 CoT canonical tracks."""
    model_v6b, ckpt_v6b = _load_v6_model(PKG_ROOT / 'runs' / 'v6b_65d_qerr' / 'best.pt', device)
    model_v61, _ = _load_v6_model(PKG_ROOT / 'runs' / 'v6_1_65d_elec' / 'best.pt', device)
    n_pipe = len(PIPE_CONFINEMENTS)
    n_terr = len(TERRAIN_GRID)
    n_grid = n_pipe * n_terr  # 42
    n = n_grid + 6            # 42..44 = v6b canonical 3 media; 45..47 = v6.1 canonical 3 media
    seconds = 6.0

    t0 = time.perf_counter()
    env, hw, task = _build_v6_env(ckpt_v6b, n=n, seconds=seconds, device=device, settle_ticks=25, elements_per_strip=8)
    mg_n = _robot_weight_n(env, hw)
    print(f'[Exp 1+3a] Settled B={n} E=8 env in {time.perf_counter()-t0:.1f}s (mg={mg_n:.2f} N)', flush=True)

    # Disable pipe_cfg; configure exact straight-corridor confinement wall_cfg & terrain grid per cell
    off_p = torch.tensor(make_off_pipe(), device=device, dtype=env.dtype)
    env.c.pipe_cfg.copy_(off_p[None, :].expand(n, -1))
    env.post_pipe_soil.zero_()

    y0_all = env.c.x[:, :10, 1].mean(-1)
    for ti, (_, mu_rr, hc_k, ul_r, rut_k, mu_lat, mu_fwd) in enumerate(TERRAIN_GRID):
        for pi, (_, y_allow, _) in enumerate(PIPE_CONFINEMENTS):
            idx = ti * n_pipe + pi
            env.c.env_terrain[idx] = torch.tensor([mu_rr, hc_k, ul_r, rut_k, mu_lat], device=device, dtype=env.dtype)
            env.c.wheel_mu[idx].fill_(mu_fwd)
            if y_allow < 10.0:
                env.regime_id[idx] = 2  # starts directly inside confined pipe
                env.c.wall_cfg[idx] = torch.tensor([float(y0_all[idx].item()), y_allow, 10.0, -20.0, 3200.0], device=device, dtype=env.dtype)
            else:
                env.regime_id[idx] = 0  # open space
                env.c.wall_cfg[idx] = torch.tensor([0.0, 1e6, 1e6, -1e6, 0.0], device=device, dtype=env.dtype)
            env.cmd[idx] = torch.tensor([0.130, 0.0, 0.0], device=device, dtype=env.dtype)

    # Canonical 3 media for CoT benchmark (lanes 42..44 = v6b, lanes 45..47 = v6.1)
    media = (
        ('Open Hard', 'hard',      1e6,   0.140),
        ('Ø236 Pipe', 'hard',      0.018, 0.130),
        ('Soft Soil', 'soft_soil', 1e6,   0.100),
    )
    for base in (n_grid, n_grid + 3):
        for mi, (_, t_name, y_al, v_cmd) in enumerate(media):
            idx = base + mi
            prof = TERRAIN_PROFILES[t_name]
            env.c.env_terrain[idx] = torch.tensor(prof[:5], device=device, dtype=env.dtype)
            env.c.wheel_mu[idx].fill_(prof[5])
            if y_al < 10.0:
                env.regime_id[idx] = 2
                env.c.wall_cfg[idx] = torch.tensor([float(y0_all[idx].item()), y_al, 10.0, -20.0, 3200.0], device=device, dtype=env.dtype)
            else:
                env.regime_id[idx] = 0 if t_name == 'hard' else 3
                env.c.wall_cfg[idx] = torch.tensor([0.0, 1e6, 1e6, -1e6, 0.0], device=device, dtype=env.dtype)
            env.cmd[idx] = torch.tensor([v_cmd, 0.0, 0.0], device=device, dtype=env.dtype)

    _, _, _, _, pkt = env._read_sensors(torch.ones(n, device=device, dtype=torch.bool))
    env.history.copy_(env._build_frame(pkt)[:, None, :])
    obs = env.history.clone()

    steps = round(seconds / task.policy_dt_s)
    steady_start = round(2.0 / task.policy_dt_s)  # skip 2.0 s initial transient -> 4.0 s steady state (~3 cycles)

    vx_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    yaw_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    ten_act_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    ten_deg_buf = torch.empty((steps, n, 10), device=device, dtype=torch.float64)
    wall_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    pelec_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    clr_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    fail_buf = torch.zeros(n, device=device, dtype=torch.bool)

    with torch.no_grad():
        for k in range(steps):
            old_sv = env.c.servo.clone()
            act_b, _ = model_v6b.action_with_estimate(obs[:n_grid + 3])
            act_1, _ = model_v61.action_with_estimate(obs[n_grid + 3:])
            act = torch.cat((act_b, act_1), dim=0)
            obs, _, _, _, term, _, _, _, info = env.step(act)
            p_elec, _, _ = _electrical_power_step(env, old_sv, task.policy_dt_s)
            fail_buf.logical_or_(term)
            vx_buf[k].copy_(info['body_vel'][:, 0] * 1000.0)
            yaw_buf[k].copy_(info['yaw_rms_deg'])
            ten_act_buf[k].copy_(info['ten_act'])
            ten_deg_buf[k].copy_(torch.rad2deg(env.c.servo))
            wall_buf[k].copy_(info['wall_mean_n'])
            pelec_buf[k].copy_(p_elec)
            clr_buf[k].copy_(info['clearance'] * 1000.0)
            if (k + 1) % 25 == 0:
                print(f'  [Exp 1+3a] step {k+1}/{steps} ({time.perf_counter()-t0:.1f}s)', flush=True)

    vx_np = vx_buf[steady_start:].cpu().numpy()
    yaw_np = yaw_buf[steady_start:].cpu().numpy()
    ten_np = ten_act_buf[steady_start:].cpu().numpy()
    ten_deg_np = ten_deg_buf[steady_start:].cpu().numpy()
    wall_np = wall_buf[steady_start:].cpu().numpy()
    pelec_np = pelec_buf[steady_start:].cpu().numpy()
    clr_np = clr_buf.cpu().numpy()
    fail_np = fail_buf.cpu().numpy()

    grid = []
    for ti, (t_lbl, mu_rr, *_) in enumerate(TERRAIN_GRID):
        row_cells = []
        for pi, (p_lbl, y_allow, d_mm) in enumerate(PIPE_CONFINEMENTS):
            idx = ti * n_pipe + pi
            vx_m = float(vx_np[:, idx].mean())
            yaw_m = float(yaw_np[:, idx].mean())
            ten_m = float(ten_np[:, idx].mean())
            ten_std_deg = float(ten_deg_np[:, idx, :].std(axis=0).mean())
            wall_m = float(wall_np[:, idx].mean())
            p_m = float(pelec_np[:, idx].mean())
            cot = p_m / (mg_n * max(vx_m * 1e-3, 0.005)) if vx_m > 5.0 else float('nan')
            y_norm = min(1.5, yaw_m / 12.0)
            t_norm = min(1.5, ten_std_deg / 30.0)
            gait_idx = (y_norm - t_norm) / max(y_norm + t_norm, 1e-6)
            row_cells.append({
                'pipe_label': p_lbl,
                'pipe_id_mm': None if math.isinf(d_mm) else d_mm,
                'y_allow_mm': None if y_allow > 10.0 else round(y_allow * 1000.0, 1),
                'terrain_label': t_lbl.replace('\n', ' '),
                'mu_rr': mu_rr,
                'failed': bool(fail_np[idx]),
                'vx_mm_s': round(vx_m, 1),
                'yaw_rms_deg': round(yaw_m, 2),
                'ten_act': round(ten_m, 3),
                'ten_std_deg': round(ten_std_deg, 2),
                'gait_index': round(float(gait_idx), 3),
                'wall_mean_n': round(wall_m, 2),
                'p_elec_w': round(p_m, 2),
                'cot': None if math.isnan(cot) else round(float(cot), 2),
                'min_clr_mm': round(float(clr_np[:, idx].min()), 2),
            })
        grid.append(row_cells)

    v6_cot: dict[str, dict[str, Any]] = {}
    for tag, base in (('v6b_autonomous', n_grid), ('v6_1_elec', n_grid + 3)):
        v6_cot[tag] = {}
        for mi, (m_lbl, _, _, _) in enumerate(media):
            idx = base + mi
            vm = float(vx_np[:, idx].mean())
            pm = float(pelec_np[:, idx].mean())
            cot = pm / (mg_n * max(vm * 1e-3, 0.003))
            v6_cot[tag][m_lbl] = {
                'vx_mm_s': round(vm, 1),
                'p_elec_w': round(pm, 2),
                'cot': round(float(cot), 2),
                'wall_n': round(float(wall_np[:, idx].mean()), 2),
                'yaw_rms_deg': round(float(yaw_np[:, idx].mean()), 1),
            }

    elapsed = time.perf_counter() - t0
    print(f'[Exp 1+3a] Finished B={n} in {elapsed:.1f}s | grid fails={int(fail_np[:n_grid].sum())}/{n_grid}', flush=True)
    return {'mg_n': round(mg_n, 2), 'seconds': seconds, 'grid': grid}, v6_cot


def analyze_3zone_phase_portrait_and_stft() -> dict[str, Any]:
    """Exp 2: Continuous (A_yaw, A_tendon) phase-space trajectory + dual-wave STFT from real 25.96 s 3-zone NPZ."""
    npz_path = PKG_ROOT / 'runs' / 'v6b_65d_qerr' / 'v6b_autonomous_pipe_traj.npz'
    d = np.load(npz_path)
    t = d['t'].astype(np.float64)                    # (650,) @ 25 Hz
    fps = float(d['fps'])                            # 25.0 Hz
    x = d['x'][:, 0, :10, :3].astype(np.float64)     # (650, 10, 3) v6b plates
    yaw_deg = d['yaw_deg'][:, 0, :].astype(np.float64)  # (650, 4)
    in_pipe = d['in_pipe'].astype(bool)
    on_soil = d['on_soil'].astype(bool)

    # Exact 5-segment axial compression (mm) from plate pairs (p0-p1, p2-p3, ..., p8-p9)
    seg_len_mm = np.linalg.norm(x[:, 1::2, :] - x[:, 0::2, :], axis=-1) * 1000.0  # (650, 5)
    seg_comp_mm = seg_len_mm[0:1, :] - seg_len_mm                                 # positive = compressed

    # Rolling-window (1.2 s = 30 samples) amplitude envelopes (A_yaw in deg, A_comp in mm)
    win = 30
    T = len(t)
    a_yaw = np.zeros(T)
    a_comp = np.zeros(T)
    for k in range(T):
        lo = max(0, k - win // 2)
        hi = min(T, k + win // 2 + 1)
        # Dynamic oscillation amplitude (RMS around local mean across 4 yaw joints / 5 segments)
        yw_seg = yaw_deg[lo:hi] - yaw_deg[lo:hi].mean(axis=0, keepdims=True)
        cp_seg = seg_comp_mm[lo:hi] - seg_comp_mm[lo:hi].mean(axis=0, keepdims=True)
        a_yaw[k] = float(np.sqrt(np.mean(yw_seg**2)))
        a_comp[k] = float(np.sqrt(np.mean(cp_seg**2)))

    # Pure Hann-windowed STFT (N_fft=64 -> 2.56 s window, hop=4 -> 0.16 s) for both channels
    n_fft = 64
    hop = 4
    hann = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    f_mask = freqs <= 3.0
    freqs_sel = freqs[f_mask]
    centers = list(range(n_fft // 2, T - n_fft // 2 + 1, hop))
    t_stft = np.array([t[c] for c in centers])
    psd_yaw = np.zeros((len(freqs_sel), len(centers)))
    psd_comp = np.zeros((len(freqs_sel), len(centers)))
    for ci, c in enumerate(centers):
        sl = slice(c - n_fft // 2, c + n_fft // 2)
        # Sum power spectra across all 4 yaw joints / 5 segments
        yw_w = (yaw_deg[sl] - yaw_deg[sl].mean(axis=0, keepdims=True)) * hann[:, None]
        cp_w = (seg_comp_mm[sl] - seg_comp_mm[sl].mean(axis=0, keepdims=True)) * hann[:, None]
        fy = np.abs(np.fft.rfft(yw_w, axis=0))[f_mask] / (0.5 * hann.sum())
        fc = np.abs(np.fft.rfft(cp_w, axis=0))[f_mask] / (0.5 * hann.sum())
        psd_yaw[:, ci] = np.sqrt(np.mean(fy**2, axis=1))   # RMS deg per frequency bin
        psd_comp[:, ci] = np.sqrt(np.mean(fc**2, axis=1))  # RMS mm per frequency bin

    z1 = (~in_pipe) & (~on_soil)
    z2_pure = in_pipe & (~on_soil)
    z3 = (~in_pipe) & on_soil
    print(
        f'[Exp 2] 3-Zone Phase Portrait | Zone I Hard: A_yaw={a_yaw[z1].mean():.1f} deg, A_comp={a_comp[z1].mean():.1f} mm | '
        f'Zone II Pipe: A_yaw={a_yaw[z2_pure].mean():.1f} deg, A_comp={a_comp[z2_pure].mean():.1f} mm | '
        f'Zone III Soil: A_yaw={a_yaw[z3].mean():.1f} deg, A_comp={a_comp[z3].mean():.1f} mm',
        flush=True,
    )
    return {
        't': t, 'in_pipe': in_pipe, 'on_soil': on_soil,
        'a_yaw': a_yaw, 'a_comp': a_comp,
        't_stft': t_stft, 'freqs': freqs_sel, 'psd_yaw': psd_yaw, 'psd_comp': psd_comp,
        'summary': {
            'zone1_hard': {'a_yaw_deg': round(float(a_yaw[z1].mean()), 2), 'a_comp_mm': round(float(a_comp[z1].mean()), 2)},
            'zone2_pipe_pure': {'a_yaw_deg': round(float(a_yaw[z2_pure].mean()), 2), 'a_comp_mm': round(float(a_comp[z2_pure].mean()), 2)},
            'zone3_soft_soil': {'a_yaw_deg': round(float(a_yaw[z3].mean()), 2), 'a_comp_mm': round(float(a_comp[z3].mean()), 2)},
        },
    }


def run_v5_cot_comparison(device: str) -> dict[str, Any]:
    """Exp 3b: Identical CoT & speed comparison for v5 fixed Snake / Worm / Mixed across Hard, Ø236 Pipe, Soft Soil."""
    seconds = 6.0
    media = (
        ('Open Hard',  'hard',      1e6,   0.140),
        ('Ø236 Pipe',  'hard',      0.018, 0.130),
        ('Soft Soil',  'soft_soil', 1e6,   0.100),
    )
    results: dict[str, dict[str, Any]] = {}

    t0 = time.perf_counter()
    ckpt_v5 = torch.load(PKG_ROOT / 'runs' / 'unified_v5_hispeed_14d' / 'best.pt', map_location=device, weights_only=False)
    hw_fields = {f.name for f in dc_fields(HardwareConfig)}
    hw5 = HardwareConfig(**{k: v for k, v in ckpt_v5['hardware'].items() if k in hw_fields})
    t_fields = {f.name for f in dc_fields(TaskConfig)}
    td5 = {k: v for k, v in ckpt_v5['task'].items() if k in t_fields}
    td5['episode_seconds'] = seconds + 5.0
    td5['command_switch_s'] = 1e6
    td5['elements_per_strip'] = 8
    td5['tangent_interval'] = 1
    task5 = TaskConfig(**td5)

    env5 = UnifiedSerpWormEnv(
        n=9, hardware=hw5, task=task5, dr=DomainRandConfig(enabled=False),
        device=device, plate_solver='fused_cuda' if device == 'cuda' else 'segment_lu', noise=False,
    )
    mg5 = _robot_weight_n(env5, hw5)
    print(f'[Exp 3b] Settled v5 B=9 E=8 env in {time.perf_counter()-t0:.1f}s (mg={mg5:.2f} N)', flush=True)
    model5 = TemporalActorCritic(
        frame_size=ckpt_v5['frame_size'], history_len=ckpt_v5['history_len'],
        privileged_size=ckpt_v5['privileged_size'], action_size=ckpt_v5['action_size'],
    ).to(device)
    model5.load_state_dict(ckpt_v5['model'])
    model5.eval()

    off_p5 = torch.tensor(make_off_pipe(), device=device, dtype=env5.dtype)
    env5.c.pipe_cfg.copy_(off_p5[None, :].expand(9, -1))
    y0_5 = env5.c.x[:, :10, 1].mean(-1)
    mode_tags = ((0, 'v5_fixed_snake'), (1, 'v5_fixed_worm'), (2, 'v5_fixed_mixed'))
    for mi, (mid, _) in enumerate(mode_tags):
        for j, (_, t_name, y_al, v_cmd) in enumerate(media):
            idx = mi * 3 + j
            env5.mode_id[idx] = mid
            env5.cmd[idx] = torch.tensor([v_cmd, 0.0, 0.0], device=device, dtype=env5.dtype)
            prof = TERRAIN_PROFILES[t_name]
            env5.c.env_terrain[idx] = torch.tensor(prof[:5], device=device, dtype=env5.dtype)
            env5.c.wheel_mu[idx].fill_(prof[5])
            if y_al < 10.0:
                env5.c.wall_cfg[idx] = torch.tensor([float(y0_5[idx].item()), y_al, 10.0, -20.0, 3200.0], device=device, dtype=env5.dtype)
            else:
                env5.c.wall_cfg[idx] = torch.tensor([0.0, 1e6, 1e6, -1e6, 0.0], device=device, dtype=env5.dtype)

    _, _, _, _, pkt5 = env5._read_sensors(torch.ones(9, device=device, dtype=torch.bool))
    env5.history.copy_(env5._build_frame(pkt5)[:, None, :])
    obs5 = env5.history.clone()
    steps5 = round(seconds / task5.policy_dt_s)
    s0_5 = round(2.0 / task5.policy_dt_s)
    vx5_l, p5_l, w5_l = [], [], []
    with torch.no_grad():
        for k in range(steps5):
            old_sv = env5.c.servo.clone()
            act5 = model5.action(obs5)
            obs5, _, _, _, _, _, _, info5 = env5.step(act5)
            pe5, _, _ = _electrical_power_step(env5, old_sv, task5.policy_dt_s)
            if k >= s0_5:
                vx5_l.append((info5['body_vel'][:, 0] * 1000.0).cpu().numpy())
                p5_l.append(pe5.cpu().numpy())
                w5_l.append(env5.c.last_wall_force.sum(-1).cpu().numpy())
            if (k + 1) % 25 == 0:
                print(f'  [Exp 3b] v5 step {k+1}/{steps5} ({time.perf_counter()-t0:.1f}s)', flush=True)
    vx5_a, p5_a, w5_a = np.asarray(vx5_l), np.asarray(p5_l), np.asarray(w5_l)
    for mi, (_, mtag) in enumerate(mode_tags):
        results[mtag] = {}
        for j, (m_lbl, _, _, _) in enumerate(media):
            idx = mi * 3 + j
            vm = float(vx5_a[:, idx].mean())
            pm = float(p5_a[:, idx].mean())
            cot = pm / (mg5 * max(vm * 1e-3, 0.003))
            results[mtag][m_lbl] = {
                'vx_mm_s': round(vm, 1),
                'p_elec_w': round(pm, 2),
                'cot': round(float(cot), 2),
                'wall_n': round(float(w5_a[:, idx].mean()), 2),
            }

    print(f'[Exp 3b] Cross-Medium CoT Comparison complete in {time.perf_counter()-t0:.1f}s.', flush=True)
    return results


def run_classical_baselines_cot(device: str) -> dict[str, Any]:
    """Exp 3c: 4 classical model-based / bio-inspired controllers across Hard, Ø236 Pipe, Soft Soil (B=12 GPU batch).

    Controllers (all tuned to verified forward-moving operating points on the exact same robot & FP64 physics):
      1. hybrid_cpg      : Coupled Serpenoid + Peristaltic traveling wave (f=0.70 Hz, A_yaw=22 deg, A_ten=70 deg)
      2. peristaltic_cpg : Pure retrograde peristaltic worm wave (f=0.70 Hz, yaw=0, A_ten=80 deg around 90 deg)
      3. serpenoid_cpg   : Hirose (1993) traveling serpenoid snake wave (f=0.70 Hz, A_yaw=25 deg, tendons=65 deg)
      4. matsuoka_cpg    : Liu et al. 4-joint coupled Matsuoka neural oscillator (kf=0.65, ±24 deg yaw, tendons=65 deg)
    """
    seconds = 6.0
    media = (
        ('Open Hard',  'hard',      1e6),
        ('Ø236 Pipe',  'hard',      0.018),
        ('Soft Soil',  'soft_soil', 1e6),
    )
    ctrl_tags = ('hybrid_cpg', 'peristaltic_cpg', 'serpenoid_cpg', 'matsuoka_cpg')
    n = len(ctrl_tags) * len(media)  # 12

    t0 = time.perf_counter()
    _, ckpt_v6b = _load_v6_model(PKG_ROOT / 'runs' / 'v6b_65d_qerr' / 'best.pt', device)
    env, hw, task = _build_v6_env(ckpt_v6b, n=n, seconds=seconds, device=device, settle_ticks=25, elements_per_strip=8)
    mg_n = _robot_weight_n(env, hw)
    print(f'[Exp 3c] Settled Classical Baselines B={n} E=8 env in {time.perf_counter()-t0:.1f}s (mg={mg_n:.2f} N)', flush=True)

    off_p = torch.tensor(make_off_pipe(), device=device, dtype=env.dtype)
    env.c.pipe_cfg.copy_(off_p[None, :].expand(n, -1))
    env.post_pipe_soil.zero_()
    y0 = env.c.x[:, :10, 1].mean(-1)

    for ci in range(len(ctrl_tags)):
        for mi, (_, t_name, y_al) in enumerate(media):
            idx = ci * len(media) + mi
            prof = TERRAIN_PROFILES[t_name]
            env.c.env_terrain[idx] = torch.tensor(prof[:5], device=device, dtype=env.dtype)
            env.c.wheel_mu[idx].fill_(prof[5])
            if y_al < 10.0:
                env.regime_id[idx] = 2
                env.c.wall_cfg[idx] = torch.tensor([float(y0[idx].item()), y_al, 10.0, -20.0, 3200.0], device=device, dtype=env.dtype)
            else:
                env.regime_id[idx] = 0 if t_name == 'hard' else 3
                env.c.wall_cfg[idx] = torch.tensor([0.0, 1e6, 1e6, -1e6, 0.0], device=device, dtype=env.dtype)

    # Matsuoka 4-joint coupled neural oscillator for the 3 matsuoka lanes (indices 9, 10, 11)
    cpg_mat = Matsuoka(3, device=device, dtype=env.dtype, c=0.75, kf=0.65)
    alpha_zero = torch.zeros((3, 4), device=device, dtype=env.dtype)
    # Warm up Matsuoka oscillator 2.0 s onto its clean limit cycle
    for _ in range(100):
        cpg_mat.step(alpha_zero, dt=0.02)

    dt_sensor = float(task.sensor_dt_s)  # 0.02 s (50 Hz bus tick)
    sub_per_tick = int(env.substeps_per_sensor)  # 20 substeps = 0.02 s
    n_ticks = round(seconds / dt_sensor)         # 300 ticks = 6.0 s
    steady_tick = round(2.0 / dt_sensor)         # 100 -> steady 4.0 s (~3 cycles)
    freq_hz = 0.70
    ramp_s = 0.80

    vx_l, p_l, w_l = [], [], []
    with torch.no_grad():
        for tk in range(n_ticks):
            t_cur = (tk + 0.5) * dt_sensor
            u = min(1.0, t_cur / ramp_s)
            r = u * u * (3.0 - 2.0 * u)
            ph = 2.0 * math.pi * freq_hz * t_cur

            x_before = env.c.x[:, :10, :3].clone()
            sv_before = env.c.servo.clone()

            # Build 50 Hz servo & yaw targets for all 12 lanes
            sv_des = torch.full((n, 10), math.radians(65.0), device=device, dtype=env.dtype)
            yw_des = torch.zeros((n, 4), device=device, dtype=env.dtype)

            # 0..2: Coupled Hybrid CPG (Serpenoid + Peristaltic)
            yw_hyb = torch.tensor([math.radians(22.0) * r * math.sin(ph - j * 0.5 * math.pi) for j in range(4)], device=device, dtype=env.dtype)
            sv_hyb = torch.tensor([math.radians(90.0 + 70.0 * r * math.sin(ph + s * 2.0 * math.pi / 5.0)) for s in range(5) for _ in range(2)], device=device, dtype=env.dtype)
            yw_des[0:3] = yw_hyb
            sv_des[0:3] = sv_hyb

            # 3..5: Pure Peristaltic Wave CPG (zero yaw, full retrograde axial wave)
            sv_per = torch.tensor([math.radians(90.0 + 80.0 * r * math.sin(ph + s * 2.0 * math.pi / 5.0)) for s in range(5) for _ in range(2)], device=device, dtype=env.dtype)
            sv_des[3:6] = sv_per

            # 6..8: Hirose Serpenoid CPG (pure lateral snake wave)
            yw_serp = torch.tensor([math.radians(25.0) * r * math.sin(ph - j * 0.5 * math.pi) for j in range(4)], device=device, dtype=env.dtype)
            yw_des[6:9] = yw_serp
            sv_des[6:9] = math.radians(65.0) * r

            # 9..11: Matsuoka Neural CPG (Liu et al. coupled 4-joint oscillator)
            mat_out = cpg_mat.step(alpha_zero, dt=dt_sensor)  # (3, 4), roughly in [-1, 1]
            yw_des[9:12] = (mat_out * math.radians(24.0) * r).clamp(-hw.yaw_action_limit_rad, hw.yaw_action_limit_rad)
            sv_des[9:12] = math.radians(65.0) * r

            # Exact same 50 Hz servo/yaw slew filter as RL policy deployment!
            env.servo_targets.add_((env.servo_gain_env * (sv_des - env.servo_targets)).clamp(-env.servo_slew, env.servo_slew))
            env.yaw_targets.add_((env.yaw_gain_env * (yw_des - env.yaw_targets)).clamp(-env.yaw_slew, env.yaw_slew))
            env.c.target.copy_(env.servo_targets)
            env.c.yaw.copy_(env.yaw_targets)
            env.c.advance_steps(sub_per_tick)

            if tk >= steady_tick:
                # Forward velocity along -x world axis (same body-forward direction as env.body_velocity)
                x_after = env.c.x[:, :10, :3]
                bvel = UnifiedSerpWormEnv.body_velocity(x_before, x_after, dt_sensor)
                pe, _, _ = _electrical_power_step(env, sv_before, dt_sensor)
                vx_l.append((bvel[:, 0] * 1000.0).cpu().numpy())
                p_l.append(pe.cpu().numpy())
                w_l.append(env.c.last_wall_force.sum(-1).cpu().numpy())
            if (tk + 1) % 75 == 0:
                print(f'  [Exp 3c] classical tick {tk+1}/{n_ticks} ({time.perf_counter()-t0:.1f}s)', flush=True)

    vx_a, p_a, w_a = np.asarray(vx_l), np.asarray(p_l), np.asarray(w_l)
    results: dict[str, dict[str, Any]] = {}
    for ci, ctag in enumerate(ctrl_tags):
        results[ctag] = {}
        for mi, (m_lbl, _, _) in enumerate(media):
            idx = ci * len(media) + mi
            vm = float(vx_a[:, idx].mean())
            pm = float(p_a[:, idx].mean())
            cot = pm / (mg_n * max(vm * 1e-3, 0.003)) if vm > 3.0 else float(pm / (mg_n * 0.003))
            results[ctag][m_lbl] = {
                'vx_mm_s': round(vm, 1),
                'p_elec_w': round(pm, 2),
                'cot': round(float(cot), 2),
                'wall_n': round(float(w_a[:, idx].mean()), 2),
                'stalled': bool(vm < 5.0),
            }
    print(f'[Exp 3c] Classical Baselines complete in {time.perf_counter()-t0:.1f}s.', flush=True)
    return results


def compute_4regime_proprioceptive_tsne() -> dict[str, Any]:
    """Compute real 2D t-SNE over 1.6 s causal pure-proprioceptive EstNet + joint-amplitude features across 4 media."""
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    d = np.load(PKG_ROOT / 'runs' / 'v6b_65d_qerr' / 'eval_best_traj.npz')
    ep = d['est_pred'].astype(np.float64)   # (280, 6, 7) pure-proprioceptive EstNet predictions
    ip = d['in_pipe'].astype(bool)
    pp = d['post_pipe'].astype(bool)
    yr = d['yaw_rms_deg'].astype(np.float64)
    ta = d['ten_act'].astype(np.float64)

    def _window_feats(tr: int, idxs: np.ndarray) -> np.ndarray:
        out = []
        for k in idxs:
            lo = max(0, int(k) - 15)  # 1.6 s causal window @ 10 Hz (matches policy history H=16)
            w = ep[lo:int(k) + 1, tr]
            f = np.concatenate([
                w[:, [0, 3, 5, 6]].mean(axis=0),          # mean [vx_est, wall_conf, drag, slip]
                np.abs(w[:, [1, 2, 4]]).mean(axis=0),     # mean [|vy|, |wz|, |wall_side|]
                w[:, [0, 3, 5]].std(axis=0),              # intra-window fluctuation
                [yr[lo:int(k) + 1, tr].mean(), ta[lo:int(k) + 1, tr].mean()],  # proprioceptive gait amps
            ])
            out.append(f)
        return np.asarray(out, dtype=np.float64)

    idx_h = np.arange(45, 265, 2)
    idx_p = np.where(ip[:, 2] & (~pp[:, 2]))[0]
    idx_p = idx_p[idx_p >= 30]
    idx_c = np.arange(45, 265, 2)
    idx_s = np.arange(45, 265, 2)

    X = np.vstack([
        _window_feats(0, idx_h),
        _window_feats(2, idx_p),
        _window_feats(5, idx_c),
        _window_feats(4, idx_s),
    ])
    y = np.concatenate([
        np.zeros(len(idx_h), dtype=int),
        np.ones(len(idx_p), dtype=int),
        np.full(len(idx_c), 2, dtype=int),
        np.full(len(idx_s), 3, dtype=int),
    ])
    Xs = StandardScaler().fit_transform(X)
    Z = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto').fit_transform(Xs)
    sil = float(silhouette_score(Z, y))
    print(f'[Exp 2b] 4-Regime Pure-Proprioceptive t-SNE Silhouette = {sil:.3f} (N={len(y)})', flush=True)
    return {'Z': Z, 'y': y, 'silhouette': round(sil, 3), 'n_total': int(len(y))}


def render_nature_figure(exp1: dict[str, Any], exp2: dict[str, Any], exp3: dict[str, Any], tsne_data: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Ellipse, Rectangle

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans', 'Arial'],
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 9.0,
        'axes.titlesize': 10.4,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9.0,
        'axes.edgecolor': '#334155',
        'axes.linewidth': 0.85,
        'xtick.labelsize': 8.1,
        'ytick.labelsize': 8.1,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

    fig = plt.figure(figsize=(15.2, 9.6), dpi=300)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.16, 1.0], height_ratios=[1.02, 1.06], wspace=0.26, hspace=0.30,
                  left=0.068, right=0.975, top=0.955, bottom=0.072)

    # -------------------------------------------------------------------------
    # Panel (a): 2D Physical Phase Diagram (Clean CoT Heatmap, NO top title!)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    grid = exp1['grid']
    n_t = len(grid)
    n_p = len(grid[0])
    vx_mat = np.array([[grid[ti][pi]['vx_mm_s'] for pi in range(n_p)] for ti in range(n_t)])
    yaw_mat = np.array([[grid[ti][pi]['yaw_rms_deg'] for pi in range(n_p)] for ti in range(n_t)])
    cot_mat = np.array([[grid[ti][pi]['cot'] if grid[ti][pi]['cot'] is not None else np.nan for pi in range(n_p)] for ti in range(n_t)])

    im = ax_a.imshow(np.log10(cot_mat), origin='lower', aspect='auto', cmap='YlGnBu_r', vmin=0.0, vmax=1.20)
    ax_a.add_patch(Rectangle((1.5, -0.5), 2.0, n_t, fill=False, edgecolor='#0F172A', lw=1.35, ls='--', zorder=5))
    ax_a.text(2.5, n_t - 0.44, '训练管径区', ha='center', va='bottom', fontsize=7.6, color='#0F172A', fontweight='bold')
    ax_a.text(0.5, n_t - 0.44, '← 未见更窄管', ha='center', va='bottom', fontsize=7.3, color='#475569')
    ax_a.text(5.0, n_t - 0.44, '未见更宽管 / 开阔 →', ha='center', va='bottom', fontsize=7.3, color='#475569')

    for ti in range(n_t):
        for pi in range(n_p):
            lc = math.log10(max(cot_mat[ti, pi], 0.1))
            txt_c = 'white' if lc < 0.52 else '#0F172A'
            ax_a.text(
                pi, ti,
                f"{vx_mat[ti, pi]:.0f} mm/s\nCoT {cot_mat[ti, pi]:.1f} · {yaw_mat[ti, pi]:.1f}°",
                ha='center', va='center', fontsize=6.6, color=txt_c, fontweight='semibold', linespacing=1.18,
            )

    ax_a.set_xticks(range(n_p))
    ax_a.set_xticklabels([c[0] for c in PIPE_CONFINEMENTS])
    ax_a.set_yticks(range(n_t))
    ax_a.set_yticklabels([t[0] for t in TERRAIN_GRID])
    ax_a.set_xlabel('空间约束内径 $D_{\\mathrm{pipe}}$ (mm)')
    ax_a.set_ylabel('地面滚阻 $\\mu_{\\mathrm{rr}}$ 与柔顺度')
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.042, pad=0.02)
    cbar.set_ticks([0.0, 0.301, 0.699, 1.0, 1.176])
    cbar.set_ticklabels(['1', '2', '5', '10', '15'])
    cbar.set_label('运输代价 CoT', fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)

    # -------------------------------------------------------------------------
    # Panel (b): Real 4-Regime Pure-Proprioceptive t-SNE (NO top title!)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    Z = tsne_data['Z']
    y_lbl = tsne_data['y']
    sil = tsne_data['silhouette']
    reg_specs = (
        (0, '#0072B2', 'o', '开阔硬地 (Open Hard)'),
        (1, '#D55E00', 's', 'Ø236 窄管 (Narrow Pipe)'),
        (2, '#E69F00', '^', '高阻地毯 (Carpet)'),
        (3, '#009E73', 'D', '松软沙土 (Soft Soil)'),
    )
    for rid, col, mk, lbl in reg_specs:
        pts = Z[y_lbl == rid]
        ax_b.scatter(pts[:, 0], pts[:, 1], s=22, marker=mk, color=col, alpha=0.80, edgecolors='white', linewidths=0.35, label=lbl, zorder=4)
        if len(pts) >= 5:
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            w_e, h_e = 2.0 * 2.0 * math.sqrt(max(vals[0], 1e-6)), 2.0 * 2.0 * math.sqrt(max(vals[1], 1e-6))
            ax_b.add_patch(Ellipse(xy=pts.mean(axis=0), width=w_e, height=h_e, angle=ang,
                                   facecolor=col, edgecolor=col, alpha=0.11, lw=1.2, zorder=2))
            ax_b.add_patch(Ellipse(xy=pts.mean(axis=0), width=w_e, height=h_e, angle=ang,
                                   facecolor='none', edgecolor=col, alpha=0.65, lw=1.1, ls='--', zorder=3))

    ax_b.text(
        0.97, 0.05, f'Silhouette = {sil:.2f}',
        transform=ax_b.transAxes, ha='right', va='bottom', fontsize=8.0, fontweight='bold', color='#1E293B',
        bbox=dict(boxstyle='round,pad=0.30', facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.8),
        zorder=6,
    )
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, ls='--', lw=0.45, alpha=0.35, color='#CBD5E1')
    ax_b.set_xlabel('t-SNE 维度 1')
    ax_b.set_ylabel('t-SNE 维度 2')
    ax_b.legend(frameon=False, loc='upper left', fontsize=7.8, markerscale=1.15)

    # -------------------------------------------------------------------------
    # Panel (c): Dual-Channel Traveling-Wave STFT (NO top title!)
    # -------------------------------------------------------------------------
    sub_gs = gs[1, 0].subgridspec(3, 1, height_ratios=[0.18, 1.0, 1.0], hspace=0.11)
    ax_c0 = fig.add_subplot(sub_gs[0, 0])
    ax_c1 = fig.add_subplot(sub_gs[1, 0], sharex=ax_c0)
    ax_c2 = fig.add_subplot(sub_gs[2, 0], sharex=ax_c0)

    t = exp2['t']
    in_pipe = exp2['in_pipe']
    on_soil = exp2['on_soil']
    t_stft = exp2['t_stft']
    freqs = exp2['freqs']
    t0_s, t1_s = float(t_stft[0]), float(t_stft[-1])
    t_enter = float(t[in_pipe][0]) if in_pipe.any() else 2.6
    t_soil_hit = float(t[on_soil][0]) if on_soil.any() else 13.0
    t_exit = float(t[in_pipe][-1]) if in_pipe.any() else 21.6

    ax_c0.set_xlim(t0_s, t1_s)
    ax_c0.set_ylim(0.0, 1.0)
    ax_c0.axis('off')
    ribbons = (
        (t0_s,       t_enter,    '#DBEAFE', '#1E3A5F', '硬地'),
        (t_enter,    t_soil_hit, '#FFEDD5', '#9A3412', 'Ø236 窄管'),
        (t_soil_hit, t_exit,     '#FEF3C7', '#92400E', '管口→软土'),
        (t_exit,     t1_s,       '#D1FAE5', '#065F46', '软土'),
    )
    for xa, xb, bgc, fgc, lbl in ribbons:
        ax_c0.add_patch(Rectangle((xa, 0.08), xb - xa, 0.84, facecolor=bgc, edgecolor='white', lw=1.2))
        ax_c0.text(0.5 * (xa + xb), 0.50, lbl, ha='center', va='center', fontsize=7.6, fontweight='bold', color=fgc)

    im1 = ax_c1.pcolormesh(t_stft, freqs, exp2['psd_yaw'], shading='gouraud', cmap='viridis')
    im2 = ax_c2.pcolormesh(t_stft, freqs, exp2['psd_comp'], shading='gouraud', cmap='magma')

    for ax_c, lbl in ((ax_c1, '偏航波\n(° RMS)'), (ax_c2, '伸缩波\n(mm RMS)')):
        for tv in (t_enter, t_soil_hit, t_exit):
            ax_c.axvline(tv, color='white', ls='--', lw=1.0, alpha=0.88)
        ax_c.set_xlim(t0_s, t1_s)
        ax_c.set_ylim(0.0, 2.5)
        ax_c.set_ylabel(lbl, fontsize=8.3)

    plt.setp(ax_c1.get_xticklabels(), visible=False)
    ax_c2.set_xlabel('时间 $t$ (s)')
    cb1 = fig.colorbar(im1, ax=ax_c1, fraction=0.036, pad=0.015); cb1.ax.tick_params(labelsize=7.0)
    cb2 = fig.colorbar(im2, ax=ax_c2, fraction=0.036, pad=0.015); cb2.ax.tick_params(labelsize=7.0)

    # -------------------------------------------------------------------------
    # Panel (d): Ours vs 4 Classical CPG Baselines (NO top title!)
    # -------------------------------------------------------------------------
    d_gs = gs[1, 1].subgridspec(3, 1, hspace=0.40)
    ctrls = [
        ('ours_autonomous_rl', '本文自主 RL',              '#009E73'),
        ('hybrid_cpg',         '经典混合行波 CPG',         '#0072B2'),
        ('peristaltic_cpg',    '经典蠕动 CPG (仅直行)',    '#E69F00'),
        ('serpenoid_cpg',      'Hirose 蛇形行波',          '#D55E00'),
        ('matsuoka_cpg',       'Matsuoka 神经 CPG',        '#64748B'),
    ]
    media_cfg = (
        ('Open Hard', '开阔硬地'),
        ('Ø236 Pipe', 'Ø236 窄管'),
        ('Soft Soil', '松软沙土'),
    )
    y_pos = np.arange(len(ctrls))[::-1]
    for mi, (m_key, m_lbl) in enumerate(media_cfg):
        ax_m = fig.add_subplot(d_gs[mi, 0])
        vals = [exp3[c[0]][m_key]['cot'] for c in ctrls]
        vxs = [exp3[c[0]][m_key]['vx_mm_s'] for c in ctrls]
        walls = [float(exp3[c[0]][m_key].get('wall_n', 0.0)) for c in ctrls]
        stalled = [bool(exp3[c[0]][m_key].get('stalled', False)) for c in ctrls]
        cols = [c[2] for c in ctrls]

        ours_cot = vals[0]
        finite_moving = [v for v, st in zip(vals, stalled) if not st]
        ref_max = max(finite_moving) if finite_moving else max(vals)
        cap_val = max(ref_max * 1.16, ours_cot * 4.0)
        plot_vals = [min(v, cap_val) for v in vals]
        x_max = cap_val * 1.56 if m_key == 'Ø236 Pipe' else cap_val * 1.48

        ax_m.barh(y_pos, plot_vals, height=0.64, color=cols, edgecolor='white', lw=0.5)
        for by, bv, pv, vx_val, wv, is_st in zip(y_pos, vals, plot_vals, vxs, walls, stalled):
            is_best = (bv == min(vals))
            if is_st:
                txt = f'失速 ({vx_val:.1f} mm/s)'
                col_t = '#B91C1C'
            else:
                lbl_cot = f'{bv:.2f}' if bv < 2.0 else (f'{bv:.1f}' if bv < 10.0 else f'{bv:.0f}')
                if m_key == 'Ø236 Pipe' and wv >= 0.1:
                    txt = f'{lbl_cot}  ({vx_val:.0f} mm/s · {wv:.1f} N)'
                else:
                    txt = f'{lbl_cot}  ({vx_val:.0f} mm/s)'
                col_t = '#065F46' if is_best else '#334155'
            ax_m.text(
                pv + x_max * 0.015, by, txt,
                ha='left', va='center', fontsize=6.9,
                fontweight='bold' if (is_best or is_st) else 'normal',
                color=col_t,
            )

        best_classical = min(vals[1:])
        if best_classical >= ours_cot:
            callout = f'{m_lbl} · {best_classical / max(ours_cot, 1e-6):.1f}× 省电'
        else:
            best_steer = min(vals[1], vals[3], vals[4])
            callout = f'{m_lbl} · 较可转向 CPG {best_steer / max(ours_cot, 1e-6):.1f}× 省电'
        ax_m.set_xlim(0.0, x_max)
        ax_m.set_yticks(y_pos)
        ax_m.set_yticklabels([c[1] for c in ctrls], fontsize=7.3)
        ax_m.spines['top'].set_visible(False)
        ax_m.spines['right'].set_visible(False)
        ax_m.grid(True, axis='x', ls='--', lw=0.45, alpha=0.35, color='#CBD5E1')
        ax_m.text(
            0.99, 0.92, callout,
            transform=ax_m.transAxes, ha='right', va='top', fontsize=7.3, fontweight='bold', color='#0F766E',
            bbox=dict(boxstyle='round,pad=0.20', facecolor='#F0FDFA', edgecolor='#99F6E4', lw=0.75),
        )
        if mi == 2:
            ax_m.set_xlabel('无量纲运输代价 $\\mathrm{CoT} = P_{\\mathrm{elec}} / (mg v_x)$')

    # Bold lowercase Nature panel labels strictly at the LOWER-LEFT corner of each quadrant
    for letter, (fx, fy) in (
        ('a', (0.012, 0.525)),
        ('b', (0.518, 0.525)),
        ('c', (0.012, 0.014)),
        ('d', (0.518, 0.014)),
    ):
        fig.text(fx, fy, letter, fontsize=13.5, fontweight='bold', color='#0F172A', ha='left', va='bottom')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / 'fig_phase_diagram_and_cot.png'
    out_pdf = OUT_DIR / 'fig_phase_diagram_and_cot.pdf'
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    if ARTIFACT_DIR is not None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_png, ARTIFACT_DIR / out_png.name)
    print(f'[Export] {out_png} & {out_pdf}', flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('all', 'phase', 'classical', 'v5', 'plot'), default='all')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c_phase = OUT_DIR / 'exp1_phase_cache.json'
    c_classical = OUT_DIR / 'exp3c_classical_cache.json'
    c_v5 = OUT_DIR / 'exp3b_v5_cache.json'

    if args.stage in ('all', 'phase'):
        print(f'[PhaseDiagram+CoT] Stage=phase on {device} ...', flush=True)
        exp1, v6_cot = run_2d_phase_diagram_and_v6_cot(device)
        c_phase.write_text(json.dumps({'exp1': exp1, 'v6_cot': v6_cot}, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'[Saved] {c_phase}', flush=True)

    if args.stage in ('all', 'classical'):
        print(f'[PhaseDiagram+CoT] Stage=classical on {device} ...', flush=True)
        cls_cot = run_classical_baselines_cot(device)
        c_classical.write_text(json.dumps(cls_cot, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'[Saved] {c_classical}', flush=True)

    if args.stage == 'v5':
        print(f'[PhaseDiagram+CoT] Stage=v5 on {device} ...', flush=True)
        v5_cot = run_v5_cot_comparison(device)
        c_v5.write_text(json.dumps(v5_cot, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'[Saved] {c_v5}', flush=True)

    if args.stage in ('all', 'plot'):
        exp2 = analyze_3zone_phase_portrait_and_stft()
        tsne_data = compute_4regime_proprioceptive_tsne()
        p_data = json.loads(c_phase.read_text(encoding='utf-8'))
        cls_data = json.loads(c_classical.read_text(encoding='utf-8'))
        exp1 = p_data['exp1']
        # Use the exact same autonomous policy shown in Panels a-c (zero internal version self-comparison!)
        exp3 = {'ours_autonomous_rl': p_data['v6_cot']['v6b_autonomous'], **cls_data}
        render_nature_figure(exp1, exp2, exp3, tsne_data)

        summary = {
            'exp1_2d_phase_diagram': exp1,
            'exp2_3zone_phase_portrait': exp2['summary'],
            'exp2b_tsne_4regime_silhouette': tsne_data['silhouette'],
            'exp3_ours_vs_classical_cot': exp3,
        }
        out_json = OUT_DIR / 'phase_diagram_cot_summary.json'
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print('\n=== EXP 3 OURS VS CLASSICAL CONTROLLERS CoT SUMMARY ===', flush=True)
        print(json.dumps(exp3, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Deterministic multi-speed & turning velocity-tracking evaluation across trained SerpWorm checkpoints.

Features:
  - Strict post-failure masking: steps on or after the first physical guard termination (`term[i]`)
    are NEVER averaged into steady-state or dynamic tracking metrics, eliminating auto-reset random
    command contamination while recording exact ``fail_step``, ``fail_time_s``, and ``fail_reason``.
  - Clean 3-mode comparison: ``raw``, ``z2`` (saddle-safe Z2 equivariant default), and
    ``z2_naive`` (``saddle_break_tol=0.0`` ablation).
"""
from __future__ import annotations

import argparse
from dataclasses import fields as dc_fields
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.config import DomainRandConfig, HardwareConfig
from serpworm.envs.autonomous_env import AutonomousPhaseEnv, AutonomousTaskConfig
from serpworm.rl.him_dreamwaq import GroupEquivariantPolicyWrapper
from serpworm.rl.models import ProprioceptiveEstimatorActorCritic


def build_steady_tracks() -> list[dict]:
    """Return 24 deterministic (speed, turning) tracks spanning speed sweep, turn sweep, and cross grid."""
    tracks: list[dict] = []
    # Group A: forward speed sweep (cmd_wz = 0)
    for vx in (0.0, 40.0, 60.0, 80.0, 100.0, 120.0, 145.0, 170.0):
        tracks.append({
            'group': 'speed_sweep',
            'name': f'vx_{int(vx)}',
            'cmd_vx_mm_s': float(vx),
            'cmd_wz_rad_s': 0.0,
            'stop': 1.0 if vx == 0.0 else 0.0,
        })
    # Group B: turning angular-velocity sweep at cmd_vx = 100 mm/s
    for wz in (-0.25, -0.20, -0.15, -0.08, 0.08, 0.15, 0.20, 0.25):
        tag = f'm{int(round(abs(wz)*100))}' if wz < 0 else f'p{int(round(wz*100))}'
        tracks.append({
            'group': 'turn_sweep',
            'name': f'vx100_wz_{tag}',
            'cmd_vx_mm_s': 100.0,
            'cmd_wz_rad_s': float(wz),
            'stop': 0.0,
        })
    # Group C: low/high speed turning cross grid
    for vx in (60.0, 140.0):
        for wz in (-0.20, -0.10, 0.10, 0.20):
            tag = f'm{int(round(abs(wz)*100))}' if wz < 0 else f'p{int(round(wz*100))}'
            tracks.append({
                'group': 'speed_turn_grid',
                'name': f'vx{int(vx)}_wz_{tag}',
                'cmd_vx_mm_s': float(vx),
                'cmd_wz_rad_s': float(wz),
                'stop': 0.0,
            })
    return tracks


DYNAMIC_SEGMENTS = [
    (0.0, 4.0, 70.0, 0.0, 0.0),
    (4.0, 9.0, 145.0, 0.0, 0.0),
    (9.0, 15.0, 100.0, 0.20, 0.0),
    (15.0, 21.0, 100.0, -0.20, 0.0),
    (21.0, 25.0, 50.0, 0.0, 0.0),
    (25.0, 30.0, 0.0, 0.0, 1.0),
]


def load_ckpt_and_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hw_fields = {f.name for f in dc_fields(HardwareConfig)}
    hw = HardwareConfig(**{k: v for k, v in ckpt['hardware'].items() if k in hw_fields})
    t_fields = {f.name for f in dc_fields(AutonomousTaskConfig)}
    td = {k: v for k, v in ckpt['task'].items() if k in t_fields}

    model = ProprioceptiveEstimatorActorCritic(
        frame_size=int(ckpt['frame_size']),
        history_len=int(ckpt['history_len']),
        privileged_size=int(ckpt['privileged_size']),
        est_size=int(ckpt.get('est_size', 7)),
        action_size=int(ckpt['action_size']),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return ckpt, hw, td, model


def make_env(hw: HardwareConfig, td: dict, n: int, episode_s: float, device: str, plate_solver: str) -> AutonomousPhaseEnv:
    td_eval = dict(td)
    td_eval['episode_seconds'] = float(episode_s) + 5.0
    td_eval['command_switch_s'] = 1e6
    task = AutonomousTaskConfig(**td_eval)
    env = AutonomousPhaseEnv(
        n=n,
        hardware=hw,
        task=task,
        dr=DomainRandConfig(enabled=False),
        device=device,
        plate_solver=plate_solver if device == 'cuda' else 'segment_lu',
        noise=False,
        forced_regime=0,  # open_hard
    )
    return env


def set_env_commands(env: AutonomousPhaseEnv, cmd_tensor: torch.Tensor, *, reset_history: bool = False) -> None:
    env.cmd.copy_(cmd_tensor)
    vx_scale = max(0.05, float(env.task.vx_max_mm_s) * 1e-3)
    wz_scale = max(0.10, float(env.task.wz_max_rad_s))
    cmd_norm = torch.stack((env.cmd[:, 0] / vx_scale, env.cmd[:, 1] / wz_scale, env.cmd[:, 2]), -1)
    if reset_history:
        env.history[:, :, -3:] = cmd_norm[:, None, :]
    else:
        env.history[:, -1, -3:] = cmd_norm


def compute_heading(env: AutonomousPhaseEnv) -> torch.Tensor:
    head = env.c.x[:, 9, :2] - env.c.x[:, 0, :2]
    return torch.atan2(head[:, 1], head[:, 0])


def _extract_guard_reason(env: AutonomousPhaseEnv, idx: int) -> str:
    reasons: list[str] = []
    guard_map = getattr(env, 'last_guard_reasons', None)
    if isinstance(guard_map, dict):
        for name, mask in guard_map.items():
            try:
                if bool(mask[idx].item()):
                    reasons.append(str(name))
            except Exception:
                continue
    return ','.join(reasons) if reasons else 'terminated'


def run_steady_sweep(
    env: AutonomousPhaseEnv,
    policy,
    tracks: list[dict],
    seconds: float,
    steady_start_s: float,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    n = len(tracks)
    obs, _, _ = env.reset(torch.ones(n, device=env.device, dtype=torch.bool))
    cmd_tensor = torch.tensor(
        [[tr['cmd_vx_mm_s'] * 1e-3, tr['cmd_wz_rad_s'], tr['stop']] for tr in tracks],
        device=env.device,
        dtype=env.dtype,
    )
    set_env_commands(env, cmd_tensor, reset_history=True)
    obs = env.history.clone()

    dt = float(env.policy_dt)
    steps = int(round(seconds / dt))
    vx_scale_mm = max(50.0, float(env.task.vx_max_mm_s))
    wz_scale = max(0.10, float(env.task.wz_max_rad_s))

    t_hist = np.zeros(steps, dtype=np.float32)
    vx_hist = np.full((steps, n), np.nan, dtype=np.float32)
    vy_hist = np.full((steps, n), np.nan, dtype=np.float32)
    wz_hist = np.full((steps, n), np.nan, dtype=np.float32)
    fvx_hist = np.full((steps, n), np.nan, dtype=np.float32)
    fwz_hist = np.full((steps, n), np.nan, dtype=np.float32)
    est_vx_hist = np.full((steps, n), np.nan, dtype=np.float32)
    est_wz_hist = np.full((steps, n), np.nan, dtype=np.float32)
    com_xy_hist = np.zeros((steps + 1, n, 2), dtype=np.float32)
    heading_hist = np.zeros((steps + 1, n), dtype=np.float32)
    yaw_rms_hist = np.full((steps, n), np.nan, dtype=np.float32)
    slip_hist = np.full((steps, n), np.nan, dtype=np.float32)
    clr_hist = np.full((steps, n), np.nan, dtype=np.float32)
    alive_hist = np.zeros((steps, n), dtype=bool)

    alive = np.ones(n, dtype=bool)
    first_fail_step = np.full(n, -1, dtype=np.int32)
    fail_reasons: list[str | None] = [None] * n

    com_xy_hist[0] = env.c.x[:, :10, :2].mean(dim=1).detach().cpu().numpy()
    heading_hist[0] = compute_heading(env).detach().cpu().numpy()

    for s in range(steps):
        with torch.no_grad():
            act, est_p = policy.action_with_estimate(obs)
        obs, _, _, _, term, _, _, _, info = env.step(act)
        t_hist[s] = (s + 1) * dt

        term_np = term.detach().cpu().numpy().astype(bool)
        newly_failed = alive & term_np
        clr_now = info['clearance'].detach().cpu().numpy() * 1000.0
        for idx in np.where(newly_failed)[0]:
            first_fail_step[idx] = s
            fail_reasons[idx] = _extract_guard_reason(env, int(idx))
            # Preserve the triggering clearance reading from the failing step
            clr_hist[s, idx] = clr_now[idx]
            alive[idx] = False

        alive_hist[s] = alive
        bvel = info['body_vel'].detach().cpu().numpy()
        fvel = info['filtered_vel'].detach().cpu().numpy()
        est_np = est_p.detach().cpu().numpy()
        com_now = env.c.x[:, :10, :2].mean(dim=1).detach().cpu().numpy()
        head_now = compute_heading(env).detach().cpu().numpy()
        yaw_now = info['yaw_rms_deg'].detach().cpu().numpy()
        slip_now = info['slip'].detach().cpu().numpy() * 1000.0

        # Record metrics strictly for still-alive tracks; freeze pose on failed tracks to avoid reset teleport
        if alive.any():
            vx_hist[s, alive] = bvel[alive, 0] * 1000.0
            vy_hist[s, alive] = bvel[alive, 1] * 1000.0
            wz_hist[s, alive] = bvel[alive, 2]
            fvx_hist[s, alive] = fvel[alive, 0] * 1000.0
            fwz_hist[s, alive] = fvel[alive, 2]
            est_vx_hist[s, alive] = est_np[alive, 0] * vx_scale_mm
            est_wz_hist[s, alive] = est_np[alive, 2] * wz_scale
            yaw_rms_hist[s, alive] = yaw_now[alive]
            slip_hist[s, alive] = slip_now[alive]
            clr_hist[s, alive] = clr_now[alive]
            com_xy_hist[s + 1, alive] = com_now[alive]
            heading_hist[s + 1, alive] = head_now[alive]
        dead = ~alive
        if dead.any():
            com_xy_hist[s + 1, dead] = com_xy_hist[s, dead]
            heading_hist[s + 1, dead] = heading_hist[s, dead]

    steady_mask = t_hist >= steady_start_s - 1e-6
    results: list[dict] = []
    for i, tr in enumerate(tracks):
        failed_i = bool(first_fail_step[i] >= 0)
        fail_step_i = int(first_fail_step[i]) if failed_i else None
        fail_time_i = float((first_fail_step[i] + 1) * dt) if failed_i else None

        valid_i = alive_hist[:, i]
        steady_valid_i = steady_mask & valid_i
        clr_valid = clr_hist[:, i]
        clr_finite = clr_valid[np.isfinite(clr_valid)]
        min_clr = float(clr_finite.min()) if clr_finite.size > 0 else None

        last_ok = int(first_fail_step[i]) if failed_i else steps
        disp_m = float(np.linalg.norm(com_xy_hist[last_ok, i] - com_xy_hist[0, i]))
        d_yaw = (
            float(np.unwrap(heading_hist[: last_ok + 1, i])[-1] - heading_hist[0, i])
            if last_ok > 0
            else 0.0
        )
        pre_fail_vx = float(vx_hist[valid_i, i].mean()) if valid_i.any() else None

        if steady_valid_i.any() and not failed_i:
            vx_s = vx_hist[steady_valid_i, i]
            vy_s = vy_hist[steady_valid_i, i]
            wz_s = wz_hist[steady_valid_i, i]
            evx_s = est_vx_hist[steady_valid_i, i]
            ewz_s = est_wz_hist[steady_valid_i, i]
            act_vx = float(vx_s.mean())
            act_wz = float(wz_s.mean())
            row_metrics = {
                'actual_vx_mm_s': act_vx,
                'actual_vx_std_mm_s': float(vx_s.std()),
                'actual_vy_mm_s': float(vy_s.mean()),
                'actual_wz_rad_s': act_wz,
                'actual_wz_std_rad_s': float(wz_s.std()),
                'est_vx_mm_s': float(evx_s.mean()),
                'est_wz_rad_s': float(ewz_s.mean()),
                'vx_err_mm_s': float(act_vx - tr['cmd_vx_mm_s']),
                'wz_err_rad_s': float(act_wz - tr['cmd_wz_rad_s']),
                'turn_radius_m': float((act_vx * 1e-3) / act_wz) if abs(act_wz) > 0.015 else None,
                'yaw_rms_deg': float(yaw_rms_hist[steady_valid_i, i].mean()),
                'slip_mm_s': float(slip_hist[steady_valid_i, i].mean()),
            }
        elif steady_valid_i.any():
            # Failed late (after steady_start_s): report pre-failure steady stats cleanly while marking failed=True
            vx_s = vx_hist[steady_valid_i, i]
            vy_s = vy_hist[steady_valid_i, i]
            wz_s = wz_hist[steady_valid_i, i]
            evx_s = est_vx_hist[steady_valid_i, i]
            ewz_s = est_wz_hist[steady_valid_i, i]
            act_vx = float(vx_s.mean())
            act_wz = float(wz_s.mean())
            row_metrics = {
                'actual_vx_mm_s': act_vx,
                'actual_vx_std_mm_s': float(vx_s.std()),
                'actual_vy_mm_s': float(vy_s.mean()),
                'actual_wz_rad_s': act_wz,
                'actual_wz_std_rad_s': float(wz_s.std()),
                'est_vx_mm_s': float(evx_s.mean()),
                'est_wz_rad_s': float(ewz_s.mean()),
                'vx_err_mm_s': float(act_vx - tr['cmd_vx_mm_s']),
                'wz_err_rad_s': float(act_wz - tr['cmd_wz_rad_s']),
                'turn_radius_m': float((act_vx * 1e-3) / act_wz) if abs(act_wz) > 0.015 else None,
                'yaw_rms_deg': float(yaw_rms_hist[steady_valid_i, i].mean()),
                'slip_mm_s': float(slip_hist[steady_valid_i, i].mean()),
            }
        else:
            # Failed before reaching steady window -> NEVER report post-reset random speed
            row_metrics = {
                'actual_vx_mm_s': None,
                'actual_vx_std_mm_s': None,
                'actual_vy_mm_s': None,
                'actual_wz_rad_s': None,
                'actual_wz_std_rad_s': None,
                'est_vx_mm_s': None,
                'est_wz_rad_s': None,
                'vx_err_mm_s': None,
                'wz_err_rad_s': None,
                'turn_radius_m': None,
                'yaw_rms_deg': None,
                'slip_mm_s': None,
            }

        results.append({
            **tr,
            'failed': failed_i,
            'fail_step': fail_step_i,
            'fail_time_s': fail_time_i,
            'fail_reason': fail_reasons[i],
            'valid_steps': int(valid_i.sum()),
            'steady_valid_steps': int(steady_valid_i.sum()),
            'pre_fail_vx_mm_s': pre_fail_vx if failed_i else None,
            **row_metrics,
            'net_disp_m': disp_m,
            'net_heading_deg': float(math.degrees(d_yaw)),
            'min_clearance_mm': min_clr,
        })

    arrays = {
        't': t_hist,
        'vx': vx_hist,
        'vy': vy_hist,
        'wz': wz_hist,
        'fvx': fvx_hist,
        'fwz': fwz_hist,
        'est_vx': est_vx_hist,
        'est_wz': est_wz_hist,
        'com_xy': com_xy_hist,
        'heading': heading_hist,
        'alive': alive_hist,
    }
    return results, arrays


def run_dynamic_profile(
    env: AutonomousPhaseEnv,
    policy,
    seconds: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    assert env.n == 1
    obs, _, _ = env.reset(torch.ones(1, device=env.device, dtype=torch.bool))
    dt = float(env.policy_dt)
    steps = int(round(seconds / dt))
    vx_scale_mm = max(50.0, float(env.task.vx_max_mm_s))
    wz_scale = max(0.10, float(env.task.wz_max_rad_s))

    t_arr = np.zeros(steps, dtype=np.float32)
    cmd_vx = np.zeros(steps, dtype=np.float32)
    cmd_wz = np.zeros(steps, dtype=np.float32)
    act_vx = np.full(steps, np.nan, dtype=np.float32)
    act_wz = np.full(steps, np.nan, dtype=np.float32)
    f_vx = np.full(steps, np.nan, dtype=np.float32)
    f_wz = np.full(steps, np.nan, dtype=np.float32)
    est_vx = np.full(steps, np.nan, dtype=np.float32)
    est_wz = np.full(steps, np.nan, dtype=np.float32)
    com_xy = np.zeros((steps + 1, 2), dtype=np.float32)
    alive_arr = np.zeros(steps, dtype=bool)

    alive = True
    first_fail_step: int | None = None
    fail_reason: str | None = None

    com_xy[0] = env.c.x[0, :10, :2].mean(dim=0).detach().cpu().numpy()

    def cmd_for_time(t_s: float) -> tuple[float, float, float]:
        for t0, t1, vx_m, wz_r, st in DYNAMIC_SEGMENTS:
            if t0 <= t_s < t1:
                return vx_m, wz_r, st
        _, _, vx_m, wz_r, st = DYNAMIC_SEGMENTS[-1]
        return vx_m, wz_r, st

    c0 = cmd_for_time(0.0)
    set_env_commands(
        env,
        torch.tensor([[c0[0] * 1e-3, c0[1], c0[2]]], device=env.device, dtype=env.dtype),
        reset_history=True,
    )
    obs = env.history.clone()

    for s in range(steps):
        t_cur = s * dt
        vx_c, wz_c, st_c = cmd_for_time(t_cur)
        t_arr[s] = (s + 1) * dt
        cmd_vx[s] = vx_c
        cmd_wz[s] = wz_c

        if not alive:
            com_xy[s + 1] = com_xy[s]
            continue

        set_env_commands(
            env,
            torch.tensor([[vx_c * 1e-3, wz_c, st_c]], device=env.device, dtype=env.dtype),
            reset_history=False,
        )
        obs = env.history.clone()
        with torch.no_grad():
            act, est_p = policy.action_with_estimate(obs)
        obs, _, _, _, term, _, _, _, info = env.step(act)

        if bool(term[0].item()):
            alive = False
            first_fail_step = s
            fail_reason = _extract_guard_reason(env, 0)
            com_xy[s + 1] = com_xy[s]
            continue

        alive_arr[s] = True
        bvel = info['body_vel'][0].detach().cpu().numpy()
        fvel = info['filtered_vel'][0].detach().cpu().numpy()
        act_vx[s] = float(bvel[0] * 1000.0)
        act_wz[s] = float(bvel[2])
        f_vx[s] = float(fvel[0] * 1000.0)
        f_wz[s] = float(fvel[2])
        est_np = est_p[0].detach().cpu().numpy()
        est_vx[s] = float(est_np[0] * vx_scale_mm)
        est_wz[s] = float(est_np[2] * wz_scale)
        com_xy[s + 1] = env.c.x[0, :10, :2].mean(dim=0).detach().cpu().numpy()

    # Segment-level steady metrics strictly on alive pre-failure steps
    seg_stats = []
    for t0, t1, vx_m, wz_r, st in DYNAMIC_SEGMENTS:
        m = (t_arr >= max(t0 + 1.5, t1 - 2.0)) & (t_arr <= t1 + 1e-6) & alive_arr
        seg_stats.append({
            't0': t0, 't1': t1, 'cmd_vx_mm_s': vx_m, 'cmd_wz_rad_s': wz_r, 'stop': st,
            'valid_steps': int(m.sum()),
            'actual_vx_mm_s': float(act_vx[m].mean()) if m.any() else None,
            'actual_wz_rad_s': float(act_wz[m].mean()) if m.any() else None,
            'est_vx_mm_s': float(est_vx[m].mean()) if m.any() else None,
            'est_wz_rad_s': float(est_wz[m].mean()) if m.any() else None,
        })

    failed_any = first_fail_step is not None
    has_valid = bool(alive_arr.any())
    summary = {
        'failed': failed_any,
        'fail_step': first_fail_step,
        'fail_time_s': float((first_fail_step + 1) * dt) if first_fail_step is not None else None,
        'fail_reason': fail_reason,
        'valid_steps': int(alive_arr.sum()),
        'overall_vx_mae_mm_s': float(np.abs(act_vx[alive_arr] - cmd_vx[alive_arr]).mean()) if has_valid else None,
        'overall_vx_rmse_mm_s': float(np.sqrt(np.mean((act_vx[alive_arr] - cmd_vx[alive_arr]) ** 2))) if has_valid else None,
        'overall_wz_mae_rad_s': float(np.abs(act_wz[alive_arr] - cmd_wz[alive_arr]).mean()) if has_valid else None,
        'overall_wz_rmse_rad_s': float(np.sqrt(np.mean((act_wz[alive_arr] - cmd_wz[alive_arr]) ** 2))) if has_valid else None,
        'est_vx_mae_mm_s': float(np.abs(est_vx[alive_arr] - f_vx[alive_arr]).mean()) if has_valid else None,
        'est_wz_mae_rad_s': float(np.abs(est_wz[alive_arr] - f_wz[alive_arr]).mean()) if has_valid else None,
        'segments': seg_stats,
    }
    arrays = {
        't': t_arr, 'cmd_vx': cmd_vx, 'cmd_wz': cmd_wz,
        'act_vx': act_vx, 'act_wz': act_wz, 'f_vx': f_vx, 'f_wz': f_wz,
        'est_vx': est_vx, 'est_wz': est_wz, 'com_xy': com_xy, 'alive': alive_arr,
    }
    return summary, arrays


def summarize_metrics(rows: list[dict]) -> dict:
    failed_rows = [r for r in rows if bool(r.get('failed', False))]
    failed_names = [r['name'] for r in failed_rows]

    def _is_valid(r: dict) -> bool:
        return (
            (not bool(r.get('failed', False)))
            and (r.get('actual_vx_mm_s') is not None)
            and (r.get('actual_wz_rad_s') is not None)
        )

    valid_rows = [r for r in rows if _is_valid(r)]
    sp_all = [r for r in rows if r['group'] == 'speed_sweep']
    sp = [r for r in sp_all if _is_valid(r)]
    sp_nz = [r for r in sp if r['cmd_vx_mm_s'] > 0]
    turn = [r for r in rows if r['group'] == 'turn_sweep' and _is_valid(r)]
    turn_all100 = turn + [r for r in sp if r['cmd_vx_mm_s'] == 100.0]

    def r2(y_true, y_pred) -> float | None:
        if len(y_true) < 2:
            return None
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        if ss_tot < 1e-12:
            return None
        ss_res = float(np.sum((yt - yp) ** 2))
        return float(1.0 - ss_res / ss_tot)

    # Symmetry check strictly on valid (non-failed) turning sweep pairs
    by_name = {r['name']: r for r in turn}
    asym_wz = []
    asym_vx = []
    for wz in (0.08, 0.15, 0.20, 0.25):
        t_val = int(round(wz * 100))
        rp = by_name.get(f'vx100_wz_p{t_val}')
        rm = by_name.get(f'vx100_wz_m{t_val}')
        if rp and rm:
            asym_wz.append(abs(rp['actual_wz_rad_s'] + rm['actual_wz_rad_s']))
            asym_vx.append(abs(rp['actual_vx_mm_s'] - rm['actual_vx_mm_s']))

    vx_cmd = [r['cmd_vx_mm_s'] for r in sp]
    vx_act = [r['actual_vx_mm_s'] for r in sp]
    vx_err = [r['vx_err_mm_s'] for r in sp]
    vx_err_nz = [r['vx_err_mm_s'] for r in sp_nz]
    wz_cmd = [r['cmd_wz_rad_s'] for r in turn_all100]
    wz_act = [r['actual_wz_rad_s'] for r in turn_all100]
    wz_err = [r['wz_err_rad_s'] for r in turn_all100]

    stop_row = next((r for r in sp_all if r['cmd_vx_mm_s'] == 0.0), None)
    stop_failed = bool(stop_row.get('failed', False)) if stop_row is not None else False
    stop_fail_reason = stop_row.get('fail_reason') if stop_row is not None else None
    stop_vx = (
        None
        if (stop_row is None or stop_failed or stop_row.get('actual_vx_mm_s') is None)
        else float(stop_row['actual_vx_mm_s'])
    )
    stop_pre_fail_vx = (
        stop_row.get('pre_fail_vx_mm_s') if (stop_row is not None and stop_failed) else None
    )

    return {
        'failed_tracks_count': len(failed_rows),
        'failed_tracks': failed_names,
        'valid_tracks_count': len(valid_rows),
        'total_tracks_count': len(rows),
        'speed_mae_mm_s': float(np.mean(np.abs(vx_err))) if vx_err else None,
        'speed_rmse_mm_s': float(np.sqrt(np.mean(np.square(vx_err)))) if vx_err else None,
        'speed_nz_mae_mm_s': float(np.mean(np.abs(vx_err_nz))) if vx_err_nz else None,
        'speed_nz_rel_err_pct': (
            float(np.mean([abs(r['vx_err_mm_s']) / r['cmd_vx_mm_s'] * 100.0 for r in sp_nz]))
            if sp_nz
            else None
        ),
        'speed_r2': r2(vx_cmd, vx_act),
        'stop_failed': stop_failed,
        'stop_fail_reason': stop_fail_reason,
        'stop_vx_mm_s': stop_vx,
        'stop_pre_fail_vx_mm_s': stop_pre_fail_vx,
        'turn100_wz_mae_rad_s': float(np.mean(np.abs(wz_err))) if wz_err else None,
        'turn100_wz_rmse_rad_s': float(np.sqrt(np.mean(np.square(wz_err)))) if wz_err else None,
        'turn100_wz_r2': r2(wz_cmd, wz_act),
        'turn100_lr_wz_asym_mean_rad_s': float(np.mean(asym_wz)) if asym_wz else None,
        'turn100_lr_wz_asym_max_rad_s': float(np.max(asym_wz)) if asym_wz else None,
        'turn100_lr_vx_asym_mean_mm_s': float(np.mean(asym_vx)) if asym_vx else None,
        'est_vx_mae_mm_s': (
            float(np.mean([abs(r['est_vx_mm_s'] - r['actual_vx_mm_s']) for r in valid_rows]))
            if valid_rows
            else None
        ),
        'est_wz_mae_rad_s': (
            float(np.mean([abs(r['est_wz_rad_s'] - r['actual_wz_rad_s']) for r in valid_rows]))
            if valid_rows
            else None
        ),
    }


def _fmt(val: float | None, spec: str = '.2f', fallback: str = 'N/A') -> str:
    return format(val, spec) if val is not None else fallback


def main():
    parser = argparse.ArgumentParser(description='Multi-speed & turning tracking evaluation.')
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        default=['runs/v8_diverse_4096/latest.pt', 'runs/v71_Lduct_4096/best.pt'],
    )
    parser.add_argument(
        '--modes',
        type=str,
        default='raw,z2,z2_naive',
        help='Comma-separated policy modes from {raw, z2, z2_naive}.',
    )
    parser.add_argument(
        '--saddle-break-tol',
        type=float,
        default=0.02,
        help='Saddle-point symmetry-breaking tolerance for z2 mode (z2_naive always uses 0.0).',
    )
    parser.add_argument('--out-dir', type=Path, default=Path('runs/tracking_eval'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plate-solver', default='fused_cuda')
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--steady-start-s', type=float, default=6.0)
    parser.add_argument('--step-seconds', type=float, default=30.0)
    args = parser.parse_args()

    req_modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    for m in req_modes:
        if m not in ('raw', 'z2', 'z2_naive'):
            raise ValueError(f'Unknown mode {m!r}; expected raw, z2, or z2_naive')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks = build_steady_tracks()
    summary_all: dict[str, dict] = {}
    npz_payload: dict[str, np.ndarray] = {}

    for ckpt_str in args.checkpoints:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            print(f'[SKIP] missing checkpoint: {ckpt_path}')
            continue
        label = f"{ckpt_path.parent.name}_{ckpt_path.stem}"
        t0 = time.time()
        ckpt, hw, td, model = load_ckpt_and_model(ckpt_path, args.device)
        eq_policy = GroupEquivariantPolicyWrapper(
            model,
            frame_size=int(ckpt['frame_size']),
            action_size=int(ckpt['action_size']),
            est_size=int(ckpt.get('est_size', 7)),
            batched=True,
            saddle_break_tol=float(args.saddle_break_tol),
        )
        eq_naive_policy = GroupEquivariantPolicyWrapper(
            model,
            frame_size=int(ckpt['frame_size']),
            action_size=int(ckpt['action_size']),
            est_size=int(ckpt.get('est_size', 7)),
            batched=True,
            saddle_break_tol=0.0,
        )
        mode_map = {
            'raw': model,
            'z2': eq_policy,
            'z2_naive': eq_naive_policy,
        }

        env_steady = make_env(hw, td, n=len(tracks), episode_s=args.seconds, device=args.device, plate_solver=args.plate_solver)
        env_dyn = make_env(hw, td, n=1, episode_s=args.step_seconds, device=args.device, plate_solver=args.plate_solver)

        ckpt_res = {
            'checkpoint': str(ckpt_path),
            'rollout': int(ckpt.get('rollout', -1)),
            'rolling_resistance': float(hw.wheel_rolling),
            'policy_dt_s': float(env_steady.policy_dt),
            'saddle_break_tol': float(args.saddle_break_tol),
            'modes': {},
        }

        for mode_name in req_modes:
            pol = mode_map[mode_name]
            rows, steady_arr = run_steady_sweep(env_steady, pol, tracks, args.seconds, args.steady_start_s)
            dyn_sum, dyn_arr = run_dynamic_profile(env_dyn, pol, args.step_seconds)
            metrics = summarize_metrics(rows)
            ckpt_res['modes'][mode_name] = {
                'metrics': metrics,
                'tracks': rows,
                'dynamic': dyn_sum,
            }
            for k, v in steady_arr.items():
                npz_payload[f'{label}__{mode_name}__steady_{k}'] = v
            for k, v in dyn_arr.items():
                npz_payload[f'{label}__{mode_name}__dyn_{k}'] = v

            stop_str = (
                f"FAILED({metrics['stop_fail_reason']})"
                if metrics['stop_failed']
                else f"{_fmt(metrics['stop_vx_mm_s'], '.2f')} mm/s"
            )
            print(
                f"[{label} | {mode_name:8s}] "
                f"speed MAE={_fmt(metrics['speed_mae_mm_s'], '.2f')} mm/s (R2={_fmt(metrics['speed_r2'], '.4f')}), "
                f"turn100 wz MAE={_fmt(metrics['turn100_wz_mae_rad_s'], '.4f')} rad/s (R2={_fmt(metrics['turn100_wz_r2'], '.4f')}), "
                f"L/R asym={_fmt(metrics['turn100_lr_wz_asym_max_rad_s'], '.5f')} rad/s, "
                f"stop={stop_str}, failed={metrics['failed_tracks_count']} {metrics['failed_tracks']} | {time.time() - t0:.1f}s",
                flush=True,
            )

        summary_all[label] = ckpt_res

    out_json = args.out_dir / 'tracking_sweep_summary.json'
    out_npz = args.out_dir / 'tracking_sweep_traj.npz'
    out_json.write_text(json.dumps(summary_all, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(out_npz, **npz_payload)
    print(f"Saved: {out_json} & {out_npz}", flush=True)


if __name__ == '__main__':
    main()

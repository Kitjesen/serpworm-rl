"""Fast Deterministic Local Empirical Benchmarks for v8 HIMDreamWaQ & GroupEquivariantPolicyWrapper.

Executes two real, reproducible empirical evaluations (exact dt=1.0 ms FP64 physics, deterministic
seeds) and writes verified metrics to ``runs/v8_ablation/v8_ablation_summary.json``:

  Benchmark (a): Exact Z2 Group-Equivariant Inference on REAL trained checkpoints
    - Evaluates ``runs/v6_1_65d_elec/best.pt`` (65D + EstNet), ``runs/v6a_51d_pure/best.pt``
      (51D + EstNet), and ``runs/unified_v5_hispeed_14d/best.pt`` (54D TemporalActorCritic).
    - Measures action & EstNet mirror-asymmetry error before vs after ``GroupEquivariantPolicyWrapper``
      on both real closed-loop rollout observations and randomized proprioceptive probes.
    - Runs short deterministic closed-loop ``AutonomousPhaseEnv`` tracks (base & equivariant
      batched in parallel at exact dt=1.0 ms) comparing straight-line yaw bias and Left/Right turn
      mirror symmetry (+0.20 vs -0.20 rad/s) before vs after wrapping.

  Benchmark (b): Self-Supervised Representation & Unseen-OOD Phase-Transition Benchmark
    - Collects genuine 65D multi-zone transitions from real 40-strip FP64 ``AutonomousPhaseEnv``
      rollouts (dt=1.0 ms) driven by ``runs/v6_1_65d_elec/best.pt`` across:
        * Nominal train zones: Open Hard Floor, Confined Narrow Pipe, Soft Soil;
        * Held-out OOD test zones with unseen friction, rolling drag, wall stiffness, and slopes.
    - Trains & compares three encoders under identical budgets:
        1. ``EstNet_v6`` (supervised 7D SmoothL1 only — current v6/v7 baseline);
        2. ``DreamWaQ_VAE`` (EstNet + beta-VAE latent context + residual next-obs decoder);
        3. ``HIMDreamWaQ_v8`` (DreamWaQ + HIMLoco symmetric InfoNCE + SwAV prototypes).
    - Measures held-out OOD next-step proprioceptive prediction MSE, 7D EstNet MAE, latent zone
      separability (Silhouette score, multi-class Fisher ratio, OOD linear-probe zone accuracy),
      parameter footprint, and single-frame 1-thread CPU latency (ms).
"""
from __future__ import annotations

import argparse
from dataclasses import fields as dc_fields
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from serpworm.config import DomainRandConfig, HardwareConfig
from serpworm.deploy.policy_runtime import load_checkpoint_model_and_meta
from serpworm.envs.autonomous_env import AutonomousPhaseEnv, AutonomousTaskConfig, mirror_autonomous_batch
from serpworm.rl.him_dreamwaq import (
    GroupEquivariantPolicyWrapper,
    HIMDreamWaQActorCritic,
    HIMDreamWaQConfig,
)
from serpworm.rl.models import ProprioceptiveEstimatorActorCritic

__all__ = [
    'compute_silhouette_score',
    'compute_fisher_discriminant_ratio',
    'linear_probe_zone_accuracy',
    'run_equivariance_benchmark',
    'run_representation_and_latency_benchmark',
    'run_all_benchmarks',
    'main',
]

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Pure-NumPy Clustering / Separability Metrics (zero external sklearn fragility)
# ---------------------------------------------------------------------------

def compute_silhouette_score(x: np.ndarray, labels: np.ndarray, metric: str = 'euclidean') -> float:
    """Exact mean Silhouette coefficient over ``x [N, D]`` and integer class labels ``[N]``."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = x.shape[0]
    uniq = np.unique(labels)
    if n < 3 or uniq.size < 2:
        return 0.0

    if metric == 'cosine':
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        xn = x / np.maximum(norms, 1e-12)
        dist = np.clip(1.0 - xn @ xn.T, 0.0, 2.0)
    else:
        sq = np.sum(x * x, axis=1, keepdims=True)
        dist = np.sqrt(np.maximum(sq + sq.T - 2.0 * (x @ x.T), 0.0))
    np.fill_diagonal(dist, 0.0)

    sil = np.zeros(n, dtype=np.float64)
    for i in range(n):
        c = labels[i]
        same = (labels == c)
        same[i] = False
        a_i = float(dist[i, same].mean()) if np.any(same) else 0.0
        b_i = float('inf')
        for other in uniq:
            if other == c:
                continue
            mask_o = labels == other
            if np.any(mask_o):
                b_i = min(b_i, float(dist[i, mask_o].mean()))
        denom = max(a_i, b_i)
        sil[i] = 0.0 if denom <= 1e-12 else (b_i - a_i) / denom
    return float(sil.mean())


def compute_fisher_discriminant_ratio(x: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Fisher scatter ratio ``tr(S_B) / (tr(S_W) + eps)`` (higher = crisper clusters)."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    uniq = np.unique(labels)
    if x.shape[0] < 3 or uniq.size < 2:
        return 0.0
    mu_all = x.mean(axis=0, keepdims=True)
    sb = 0.0
    sw = 0.0
    for c in uniq:
        xc = x[labels == c]
        if xc.shape[0] == 0:
            continue
        mu_c = xc.mean(axis=0, keepdims=True)
        sb += float(xc.shape[0] * np.sum((mu_c - mu_all) ** 2))
        sw += float(np.sum((xc - mu_c) ** 2))
    return float(sb / max(sw, 1e-12))


def linear_probe_zone_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 3,
    ridge: float = 1e-2,
) -> float:
    """Closed-form L2-regularized linear probe trained on nominal zones, tested on unseen OOD zones."""
    x_tr = np.asarray(x_train, dtype=np.float64)
    x_te = np.asarray(x_test, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.int64)
    y_te = np.asarray(y_test, dtype=np.int64)
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(x_tr.std(axis=0, keepdims=True), 1e-6)
    z_tr = np.hstack(((x_tr - mu) / sd, np.ones((x_tr.shape[0], 1))))
    z_te = np.hstack(((x_te - mu) / sd, np.ones((x_te.shape[0], 1))))

    oh = np.eye(num_classes, dtype=np.float64)[y_tr]
    reg = ridge * np.eye(z_tr.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(z_tr.T @ z_tr + reg, z_tr.T @ oh)
    pred = np.argmax(z_te @ w, axis=1)
    return float((pred == y_te).mean())


# ---------------------------------------------------------------------------
# Benchmark (a): Exact Z2 Group-Equivariant Inference on Real Checkpoints
# ---------------------------------------------------------------------------

def _measure_checkpoint_static_symmetry(ckpt_path: Path, seed: int = 123) -> dict[str, Any]:
    model, meta = load_checkpoint_model_and_meta(ckpt_path, device='cpu')
    model.eval()
    eq_exact = GroupEquivariantPolicyWrapper(model, batched=False).eval()
    eq_batched = GroupEquivariantPolicyWrapper(model, batched=True).eval()

    gen = torch.Generator().manual_seed(seed)
    mean_f = model.obs_norm.mean.detach().float().view(1, 1, -1)
    std_f = model.obs_norm.var.detach().clamp_min(1e-6).sqrt().float().view(1, 1, -1)
    obs = mean_f + 0.75 * std_f * torch.randn((128, meta.history_len, meta.frame_size), generator=gen)
    m_obs = eq_exact.mirror_obs(obs)

    with torch.inference_mode():
        raw_a = model.action(obs)
        raw_am = model.action(m_obs)
        raw_diff = (raw_am - eq_exact.mirror_action(raw_a)).abs()

        eq_a = eq_exact.action(obs)
        eq_am = eq_exact.action(m_obs)
        eq_diff = (eq_am - eq_exact.mirror_action(eq_a)).abs()

        eqb_a = eq_batched.action(obs)
        eqb_am = eq_batched.action(m_obs)
        eqb_diff = (eqb_am - eq_batched.mirror_action(eqb_a)).abs()

        sym_obs = 0.5 * (obs[:16] + m_obs[:16])
        yaw_sl = slice(10, 14) if meta.action_size == 14 else slice(5, 9)
        raw_neutral_yaw = float(model.action(sym_obs)[:, yaw_sl].abs().max().item())
        eq_neutral_yaw = float(eq_exact.action(sym_obs)[:, yaw_sl].abs().max().item())

        out: dict[str, Any] = {
            'checkpoint': str(ckpt_path.relative_to(REPO_ROOT)).replace('\\', '/'),
            'model_type': meta.model_type,
            'frame_size': meta.frame_size,
            'history_len': meta.history_len,
            'action_size': meta.action_size,
            'rollout': meta.rollout,
            'action_asym_before_mean': round(float(raw_diff.mean().item()), 6),
            'action_asym_before_max': round(float(raw_diff.max().item()), 6),
            'action_asym_before_rms': round(float(raw_diff.square().mean().sqrt().item()), 6),
            'action_asym_after_exact_mean': float(eq_diff.mean().item()),
            'action_asym_after_exact_max': float(eq_diff.max().item()),
            'action_asym_after_batched_max': float(eqb_diff.max().item()),
            'neutral_state_yaw_bias_before': round(raw_neutral_yaw, 6),
            'neutral_state_yaw_bias_after': round(eq_neutral_yaw, 8),
        }

        if eq_exact.has_estimator:
            _, raw_e = model.action_with_estimate(obs)
            _, raw_em = model.action_with_estimate(m_obs)
            est_diff_before = (raw_em - eq_exact.mirror_est(raw_e)).abs()

            _, eq_e = eq_exact.action_with_estimate(obs)
            _, eq_em = eq_exact.action_with_estimate(m_obs)
            est_diff_after = (eq_em - eq_exact.mirror_est(eq_e)).abs()

            out.update({
                'est_asym_before_mean': round(float(est_diff_before.mean().item()), 6),
                'est_asym_before_max': round(float(est_diff_before.max().item()), 6),
                'est_vy_asym_before_mean': round(float(est_diff_before[:, 1].mean().item()), 6),
                'est_wz_asym_before_mean': round(float(est_diff_before[:, 2].mean().item()), 6),
                'est_wall_side_asym_before_mean': round(float(est_diff_before[:, 4].mean().item()), 6),
                'est_asym_after_exact_mean': float(est_diff_after.mean().item()),
                'est_asym_after_exact_max': float(est_diff_after.max().item()),
            })
    return out


def _run_closed_loop_symmetry_track(
    ckpt_path: Path,
    steps: int = 15,
    elements_per_strip: int = 8,
    physics_dt_s: float = 0.001,
    seed: int = 7,
) -> dict[str, Any]:
    """Deterministic closed-loop AutonomousPhaseEnv comparison: Base vs Equivariant in one batched env.

    Runs 6 lanes in parallel (at exact ``dt=1.0 ms`` FP64 ARPC physics):
      Lanes 0..2 (Base policy):        straight (140, 0), turn_L (90, +0.20), turn_R (90, -0.20)
      Lanes 3..5 (Equivariant wrapper): straight (140, 0), turn_L (90, +0.20), turn_R (90, -0.20)
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hw_fields = {f.name for f in dc_fields(HardwareConfig)}
    hw = HardwareConfig(**{k: v for k, v in ckpt['hardware'].items() if k in hw_fields})
    t_fields = {f.name for f in dc_fields(AutonomousTaskConfig)}
    td = {k: v for k, v in ckpt['task'].items() if k in t_fields}
    td['elements_per_strip'] = int(elements_per_strip)
    td['physics_dt_s'] = float(physics_dt_s)
    td['episode_seconds'] = steps * float(td.get('policy_dt_s', 0.10)) + 2.0
    td['command_switch_s'] = 1e6
    task = AutonomousTaskConfig(**td)

    base_model, meta = load_checkpoint_model_and_meta(ckpt_path, device='cpu')
    eq_model = GroupEquivariantPolicyWrapper(base_model, batched=False).eval()

    env = AutonomousPhaseEnv(
        n=6,
        hardware=hw,
        task=task,
        dr=DomainRandConfig(enabled=False),
        device='cpu',
        plate_solver='segment_lu',
        seed=seed,
        noise=False,
        forced_regime=0,
        settle_ticks=12,
    )
    cmds = torch.tensor(
        [
            [0.140, 0.0, 0.0],
            [0.090, +0.20, 0.0],
            [0.090, -0.20, 0.0],
            [0.140, 0.0, 0.0],
            [0.090, +0.20, 0.0],
            [0.090, -0.20, 0.0],
        ],
        dtype=env.dtype,
    )
    env.cmd.copy_(cmds)
    _, _, _, _, pkt = env._read_sensors(torch.ones(6, dtype=torch.bool))
    env.history.copy_(env._build_frame(pkt)[:, None, :])
    obs = env.history.clone()

    p0 = env.c.x[:, :10, :3].clone().numpy()
    vx_hist = np.zeros((steps, 6), dtype=np.float64)
    wz_hist = np.zeros((steps, 6), dtype=np.float64)
    yaw_hist = np.zeros((steps, 6), dtype=np.float64)
    est_err_base = 0.0
    est_err_eq = 0.0
    failed = np.zeros(6, dtype=bool)
    vx_scale_mm = max(50.0, task.vx_max_mm_s)
    rollout_obs_samples: list[torch.Tensor] = []

    with torch.inference_mode():
        for k in range(steps):
            rollout_obs_samples.append(obs[:3].clone())
            act_b, est_b = base_model.action_with_estimate(obs[:3])
            act_e, est_e = eq_model.action_with_estimate(obs[3:])
            act = torch.cat((act_b, act_e), dim=0)
            obs, _, est_t, _, term, _, _, _, info = env.step(act)
            failed |= term.numpy()
            vx_hist[k] = info['body_vel'][:, 0].numpy() * 1000.0
            wz_hist[k] = info['body_vel'][:, 2].numpy()
            yaw_hist[k] = info['yaw_rms_deg'].numpy()
            est_err_base += float((est_b[:, 0] - est_t[:3, 0]).abs().mean().item() * vx_scale_mm)
            est_err_eq += float((est_e[:, 0] - est_t[3:, 0]).abs().mean().item() * vx_scale_mm)

    p1 = env.c.x[:, :10, :3].clone().numpy()
    heading_deg = []
    for i in range(6):
        fwd0 = p0[i, 9, :2] - p0[i, 0, :2]
        fwd0 /= max(float(np.linalg.norm(fwd0)), 1e-9)
        fwd1 = p1[i, 9, :2] - p1[i, 0, :2]
        fwd1 /= max(float(np.linalg.norm(fwd1)), 1e-9)
        ang = math.degrees(math.atan2(fwd0[0] * fwd1[1] - fwd0[1] * fwd1[0], float(fwd0 @ fwd1)))
        heading_deg.append(ang)

    tail = slice(steps // 3, steps)
    results_by_mode: dict[str, Any] = {}
    for mode_name, off, est_sum in (('base', 0, est_err_base), ('equivariant', 3, est_err_eq)):
        i0, i1, i2 = off, off + 1, off + 2
        results_by_mode[mode_name] = {
            'failed_any': bool(failed[off:off + 3].any()),
            'straight_vx_mm_s': round(float(vx_hist[tail, i0].mean()), 2),
            'straight_mean_wz_rad_s': round(float(wz_hist[tail, i0].mean()), 6),
            'straight_abs_heading_drift_deg': round(abs(heading_deg[i0]), 4),
            'straight_yaw_rms_deg': round(float(yaw_hist[tail, i0].mean()), 2),
            'turn_L_wz_rad_s': round(float(wz_hist[tail, i1].mean()), 5),
            'turn_R_wz_rad_s': round(float(wz_hist[tail, i2].mean()), 5),
            'turn_LR_wz_bias_rad_s': round(abs(float(wz_hist[tail, i1].mean() + wz_hist[tail, i2].mean())), 6),
            'turn_LR_heading_asym_deg': round(abs(heading_deg[i1] + heading_deg[i2]), 4),
            'turn_LR_vx_asym_mm_s': round(abs(float(vx_hist[tail, i1].mean() - vx_hist[tail, i2].mean())), 3),
            'est_vx_mae_mm_s': round(est_sum / max(1, steps), 2),
        }

    traj_obs = torch.cat(rollout_obs_samples, dim=0)
    m_traj = eq_model.mirror_obs(traj_obs)
    with torch.inference_mode():
        raw_a, raw_e = base_model.action_with_estimate(traj_obs)
        raw_am, raw_em = base_model.action_with_estimate(m_traj)
        eq_a, eq_e = eq_model.action_with_estimate(traj_obs)
        eq_am, eq_em = eq_model.action_with_estimate(m_traj)
        on_traj_metrics = {
            'num_obs_frames': int(traj_obs.shape[0]),
            'on_traj_action_asym_before_mean': round(float((raw_am - eq_model.mirror_action(raw_a)).abs().mean().item()), 6),
            'on_traj_action_asym_before_max': round(float((raw_am - eq_model.mirror_action(raw_a)).abs().max().item()), 6),
            'on_traj_action_asym_after_max': float((eq_am - eq_model.mirror_action(eq_a)).abs().max().item()),
            'on_traj_est_asym_before_mean': round(float((raw_em - eq_model.mirror_est(raw_e)).abs().mean().item()), 6),
            'on_traj_est_asym_before_max': round(float((raw_em - eq_model.mirror_est(raw_e)).abs().max().item()), 6),
            'on_traj_est_asym_after_max': float((eq_em - eq_model.mirror_est(eq_e)).abs().max().item()),
        }

    return {
        'checkpoint': str(ckpt_path.relative_to(REPO_ROOT)).replace('\\', '/'),
        'frame_size': meta.frame_size,
        'steps': steps,
        'sim_seconds': round(steps * float(task.policy_dt_s), 2),
        'physics_dt_ms': round(physics_dt_s * 1000.0, 2),
        'elements_per_strip': int(elements_per_strip),
        'on_trajectory_symmetry': on_traj_metrics,
        'closed_loop_tracks': results_by_mode,
    }


def run_equivariance_benchmark(
    closed_loop_steps: int = 15,
    seed: int = 7,
) -> dict[str, Any]:
    """Run Benchmark (a): static & closed-loop Z2 group-equivariance audit on real checkpoints."""
    ckpts = [
        REPO_ROOT / 'runs/v6_1_65d_elec/best.pt',
        REPO_ROOT / 'runs/v6a_51d_pure/best.pt',
        REPO_ROOT / 'runs/unified_v5_hispeed_14d/best.pt',
    ]
    static_results = []
    for p in ckpts:
        if p.is_file():
            static_results.append(_measure_checkpoint_static_symmetry(p, seed=seed + 100))

    closed_loop = []
    for p in (REPO_ROOT / 'runs/v6_1_65d_elec/best.pt', REPO_ROOT / 'runs/v6a_51d_pure/best.pt'):
        if p.is_file():
            closed_loop.append(_run_closed_loop_symmetry_track(p, steps=closed_loop_steps, seed=seed))

    return {
        'static_checkpoint_audit': static_results,
        'closed_loop_autonomous_eval': closed_loop,
    }


# ---------------------------------------------------------------------------
# Benchmark (b): Self-Supervised Representation & Unseen OOD Zone Separation
# ---------------------------------------------------------------------------

def _collect_multizone_physics_dataset(
    driver_ckpt: Path,
    steps_per_lane: int = 24,
    elements_per_strip: int = 8,
    physics_dt_s: float = 0.001,
    seed: int = 42,
) -> dict[str, Any]:
    """Collect real 65D multi-zone transitions (3 nominal train lanes + 3 unseen OOD shift lanes).

    Zones (exact dt=1.0 ms FP64 ARPC + pipe wall contact):
      0 = Hard Floor (nominal flat mu=0.80 vs OOD +3.0 deg slope, mu=0.50, rolling=0.055)
      1 = Confined Pipe (nominal y_allow=18 mm, k_w=3200 vs OOD tighter 15 mm, k_w=3800, mu=0.60)
      2 = Soft Soil (nominal rolling=0.22, rut_k=650 vs OOD -2.5 deg slope, rolling=0.26, rut_k=750, mu=0.55)
    """
    ckpt = torch.load(driver_ckpt, map_location='cpu', weights_only=False)
    hw_fields = {f.name for f in dc_fields(HardwareConfig)}
    hw = HardwareConfig(**{k: v for k, v in ckpt['hardware'].items() if k in hw_fields})
    t_fields = {f.name for f in dc_fields(AutonomousTaskConfig)}
    td = {k: v for k, v in ckpt['task'].items() if k in t_fields}
    td['elements_per_strip'] = int(elements_per_strip)
    td['physics_dt_s'] = float(physics_dt_s)
    td['episode_seconds'] = (steps_per_lane + 10) * float(td.get('policy_dt_s', 0.10)) + 4.0
    td['command_switch_s'] = 1e6
    task = AutonomousTaskConfig(**td)

    driver, _ = load_checkpoint_model_and_meta(driver_ckpt, device='cpu')
    driver.eval()

    n_lanes = 6
    env = AutonomousPhaseEnv(
        n=n_lanes,
        hardware=hw,
        task=task,
        dr=DomainRandConfig(enabled=False),
        device='cpu',
        plate_solver='segment_lu',
        seed=seed,
        noise=True,
        forced_regime=0,
        settle_ticks=12,
    )

    env.c.wall_cfg.zero_()
    env.c.wall_cfg[:, 1] = 1e6
    env.c.wall_cfg[:, 2] = 1e6
    env.c.wall_cfg[:, 3] = -1e6
    env.c.wall_cfg[:, 4] = 0.0
    env.c.ground_geom.zero_()

    terr_ids = torch.tensor([0, 0, 2, 0, 0, 2], dtype=torch.long)
    env._apply_terrain_ids(torch.ones(n_lanes, dtype=torch.bool), terr_ids)

    head_x = float(env.c.x[1, 9, 0].item())
    y0_1 = float(env.c.x[1, :10, 1].mean().item())
    env.regime_id[1] = 2
    env.has_entered_pipe[1] = True
    env.c.wall_cfg[1] = torch.tensor([y0_1, 0.018, 10.0, head_x - 3.0, 3200.0], dtype=env.dtype)

    # Lane 3: OOD Hard Floor (+3.0 deg slope, mu=0.50, rolling=0.055)
    env.c.env_terrain[3, 0] = 0.055
    env.c.wheel_mu[3].fill_(0.50)
    env.c.ground_geom[3, 0] = math.tan(math.radians(3.0))

    # Lane 4: OOD Tighter Stiff Pipe (y_allow=0.015 m, k_wall=3800, mu=0.60)
    y0_4 = float(env.c.x[4, :10, 1].mean().item())
    env.regime_id[4] = 2
    env.has_entered_pipe[4] = True
    env.c.wall_cfg[4] = torch.tensor([y0_4, 0.015, 10.0, head_x - 3.0, 3800.0], dtype=env.dtype)
    env.c.wheel_mu[4].fill_(0.60)

    # Lane 5: OOD High-Drag Soft Soil + Slope (-2.5 deg, rolling=0.260, rut_k=750, mu=0.55)
    env.c.env_terrain[5] = torch.tensor([0.260, 1.0e4, 3.8, 750.0, 1.40], dtype=env.dtype)
    env.c.wheel_mu[5].fill_(0.55)
    env.c.ground_geom[5, 0] = math.tan(math.radians(-2.5))

    env.c.align_pose_to_ground(torch.tensor([False, False, False, True, False, True], dtype=torch.bool))

    env.cmd[0] = torch.tensor([0.140, 0.0, 0.0], dtype=env.dtype)
    env.cmd[1] = torch.tensor([0.130, 0.0, 0.0], dtype=env.dtype)
    env.cmd[2] = torch.tensor([0.105, 0.0, 0.0], dtype=env.dtype)
    env.cmd[3] = torch.tensor([0.135, 0.0, 0.0], dtype=env.dtype)
    env.cmd[4] = torch.tensor([0.125, 0.0, 0.0], dtype=env.dtype)
    env.cmd[5] = torch.tensor([0.100, 0.0, 0.0], dtype=env.dtype)

    _, _, _, _, pkt = env._read_sensors(torch.ones(n_lanes, dtype=torch.bool))
    env.history.copy_(env._build_frame(pkt)[:, None, :])
    obs = env.history.clone()
    est_tgt = env._build_est_target()

    obs_list, nobs_list, act_list, est_list, zone_list, split_list = [], [], [], [], [], []
    lane_zones = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    lane_splits = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    gen = torch.Generator().manual_seed(seed + 17)
    warmup = 4

    with torch.inference_mode():
        for step in range(steps_per_lane + warmup):
            act = driver.action(obs)
            act_exec = (act + 0.05 * torch.randn(act.shape, generator=gen)).clamp(-1.0, 1.0)
            nobs, _, nest, _, term, _, tobs, _, _ = env.step(act_exec)
            if step >= warmup:
                valid = ~term
                if bool(valid.any()):
                    obs_list.append(obs[valid].clone())
                    nobs_list.append(tobs[valid, -1, :].clone())
                    act_list.append(act_exec[valid].clone())
                    est_list.append(est_tgt[valid].clone())
                    zone_list.append(lane_zones[valid].clone())
                    split_list.append(lane_splits[valid].clone())
            obs, est_tgt = nobs, nest

    if not obs_list:
        raise RuntimeError('Multi-zone physics rollout produced zero valid transitions.')

    all_obs = torch.cat(obs_list, dim=0)
    all_nobs = torch.cat(nobs_list, dim=0)
    all_act = torch.cat(act_list, dim=0)
    all_est = torch.cat(est_list, dim=0)
    all_zone = torch.cat(zone_list, dim=0)
    all_split = torch.cat(split_list, dim=0)

    def _pack_with_z2_mirror(mask: torch.Tensor) -> dict[str, torch.Tensor]:
        o = all_obs[mask]
        no = all_nobs[mask]
        a = all_act[mask]
        e = all_est[mask]
        z = all_zone[mask]
        dummy_priv = torch.zeros((o.shape[0], env.privileged_size), dtype=torch.float32)
        mo, _, me, ma = mirror_autonomous_batch(o, dummy_priv, e, a, include_q_err=task.include_q_err)
        mno, _, _, _ = mirror_autonomous_batch(no[:, None, :], dummy_priv, e, a, include_q_err=task.include_q_err)
        return {
            'obs': torch.cat((o, mo), dim=0),
            'next_obs': torch.cat((no, mno[:, 0, :]), dim=0),
            'act': torch.cat((a, ma), dim=0),
            'est': torch.cat((e, me), dim=0),
            'zone': torch.cat((z, z), dim=0),
        }

    tr_pack = _pack_with_z2_mirror(all_split == 0)
    te_pack = _pack_with_z2_mirror(all_split == 1)
    return {
        'frame_size': int(env.frame_size),
        'history_len': int(env.history_len),
        'privileged_size': int(env.privileged_size),
        'action_size': int(env.action_size),
        'train': tr_pack,
        'test_ood': te_pack,
        'zone_names': ['hard_floor', 'confined_pipe', 'soft_soil'],
    }


def _train_and_eval_estnet_baseline(
    data: dict[str, Any],
    *,
    device: str = 'cpu',
    train_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train v6 supervised-only EstNet baseline and evaluate its frozen features on unseen OOD zones."""
    torch.manual_seed(seed)
    tr = {k: v.to(device) for k, v in data['train'].items()}
    te = {k: v.to(device) for k, v in data['test_ood'].items()}
    f_sz, h_len, p_sz, a_sz = data['frame_size'], data['history_len'], data['privileged_size'], data['action_size']
    sens_dim = f_sz - 3

    model = ProprioceptiveEstimatorActorCritic(
        frame_size=f_sz, history_len=h_len, privileged_size=p_sz, est_size=7, action_size=a_sz
    ).to(device)
    model.obs_norm.update(tr['obs'][:, -1, :])

    est_params = list(model.est_tcn.parameters()) + list(model.est_mlp.parameters())
    opt = torch.optim.AdamW(est_params, lr=lr, weight_decay=1e-4)
    n_tr = tr['obs'].shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(train_steps):
        idx = torch.randint(0, n_tr, (min(batch_size, n_tr),), generator=gen, device=device)
        est_p = model.estimate(tr['obs'][idx])
        loss = F.smooth_l1_loss(est_p, tr['est'][idx], beta=0.15)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(est_params, 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        tr_seq_n = model.obs_norm(tr['obs'])
        te_seq_n = model.obs_norm(te['obs'])
        tr_est_p = model._estimate_from_norm(tr_seq_n)
        te_est_p = model._estimate_from_norm(te_seq_n)
        tr_feat = model.est_mlp[:-1](model._encode_seq(tr_seq_n, model.est_tcn))
        te_feat = model.est_mlp[:-1](model._encode_seq(te_seq_n, model.est_tcn))
        tr_next_sens = model.obs_norm(tr['next_obs'])[:, :sens_dim]
        te_next_sens = model.obs_norm(te['next_obs'])[:, :sens_dim]

    # Fit read-only next-step sensor decoder on FROZEN 7D EstNet outputs + action (exact matched input to DreamWaQ minus z_t)
    probe_dec = nn.Sequential(
        nn.Linear(7 + a_sz, 192),
        nn.LayerNorm(192),
        nn.SiLU(),
        nn.Linear(192, 128),
        nn.LayerNorm(128),
        nn.SiLU(),
        nn.Linear(128, sens_dim),
    ).to(device)
    opt_p = torch.optim.AdamW(probe_dec.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(train_steps):
        idx = torch.randint(0, n_tr, (min(batch_size, n_tr),), generator=gen, device=device)
        inp = torch.cat((tr_est_p[idx], tr['act'][idx]), dim=-1)
        pred = probe_dec(inp)
        loss_p = F.mse_loss(pred, tr_next_sens[idx])
        opt_p.zero_grad(set_to_none=True)
        loss_p.backward()
        opt_p.step()

    probe_dec.eval()
    with torch.no_grad():
        te_pred_sens = probe_dec(torch.cat((te_est_p, te['act']), dim=-1))
        next_obs_mse = float(F.mse_loss(te_pred_sens, te_next_sens).item())
        dyn_mse = float(F.mse_loss(te_pred_sens[:, 14:45], te_next_sens[:, 14:45]).item())
        est_l1 = (te_est_p - te['est']).abs()
        est_smooth_l1 = float(F.smooth_l1_loss(te_est_p, te['est'], beta=0.15).item())

    y_tr = tr['zone'].cpu().numpy()
    y_te = te['zone'].cpu().numpy()
    emb_tr = tr_feat.cpu().numpy()
    emb_te = te_feat.cpu().numpy()
    est_tr_np = tr_est_p.cpu().numpy()
    est_te_np = te_est_p.cpu().numpy()

    return {
        'model': 'EstNet_v6_supervised_only',
        'ood_next_obs_mse': round(next_obs_mse, 5),
        'ood_contact_dyn_mse': round(dyn_mse, 5),
        'ood_est_smooth_l1': round(est_smooth_l1, 5),
        'ood_est_vx_mae_mm_s': round(float(est_l1[:, 0].mean().item() * 175.0), 2),
        'ood_est_wall_mae': round(float(est_l1[:, 3].mean().item()), 4),
        'ood_est_drag_mae': round(float(est_l1[:, 5].mean().item()), 4),
        'ood_latent_silhouette_euclidean': round(compute_silhouette_score(emb_te, y_te, 'euclidean'), 4),
        'ood_latent_silhouette_cosine': round(compute_silhouette_score(emb_te, y_te, 'cosine'), 4),
        'ood_latent_fisher_ratio': round(compute_fisher_discriminant_ratio(emb_te, y_te), 4),
        'ood_latent_linear_probe_acc': round(linear_probe_zone_accuracy(emb_tr, y_tr, emb_te, y_te), 4),
        'ood_est7d_silhouette_euclidean': round(compute_silhouette_score(est_te_np, y_te, 'euclidean'), 4),
        'ood_est7d_fisher_ratio': round(compute_fisher_discriminant_ratio(est_te_np, y_te), 4),
        'ood_est7d_linear_probe_acc': round(linear_probe_zone_accuracy(est_tr_np, y_tr, est_te_np, y_te), 4),
    }


def _train_and_eval_him_dreamwaq_variant(
    data: dict[str, Any],
    *,
    variant_name: str,
    cfg: HIMDreamWaQConfig,
    device: str = 'cpu',
    train_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a HIMDreamWaQActorCritic variant (DreamWaQ-only or full HIM-DreamWaQ) & evaluate on OOD."""
    torch.manual_seed(seed)
    tr = {k: v.to(device) for k, v in data['train'].items()}
    te = {k: v.to(device) for k, v in data['test_ood'].items()}
    f_sz, h_len, p_sz, a_sz = data['frame_size'], data['history_len'], data['privileged_size'], data['action_size']
    sens_dim = f_sz - 3

    model = HIMDreamWaQActorCritic(
        frame_size=f_sz, history_len=h_len, privileged_size=p_sz, est_size=7, action_size=a_sz, config=cfg
    ).to(device)
    model.obs_norm.update(tr['obs'][:, -1, :])

    aux_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith(('actor_', 'critic_', 'z_gate_head', 'logstd'))
    ]
    opt = torch.optim.AdamW(aux_params, lr=lr, weight_decay=1e-4)
    n_tr = tr['obs'].shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)

    model.train()
    last_metrics: dict[str, float] = {}
    for _ in range(train_steps):
        idx = torch.randint(0, n_tr, (min(batch_size, n_tr),), generator=gen, device=device)
        loss, m = model.compute_aux_losses(
            tr['obs'][idx], tr['next_obs'][idx], tr['act'][idx], tr['est'][idx]
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(aux_params, 1.0)
        opt.step()
        model.update_ema_target()
        last_metrics = {k: float(m[k].item()) for k in ('est_loss', 'rec_loss', 'kl_loss', 'infonce_loss', 'swav_loss')}

    model.eval()
    with torch.no_grad():
        tr_seq_n = model.obs_norm(tr['obs'])
        te_seq_n = model.obs_norm(te['obs'])
        tr_ctx = model._encode_context_from_norm(tr_seq_n, deterministic=True)
        te_ctx = model._encode_context_from_norm(te_seq_n, deterministic=True)
        te_next_sens = model.obs_norm(te['next_obs'])[:, :sens_dim]
        te_pred_sens = model.decode_next_sensor(te['act'], te_ctx.est_pred, te_ctx.z_mu)

        next_obs_mse = float(F.mse_loss(te_pred_sens, te_next_sens).item())
        dyn_mse = float(F.mse_loss(te_pred_sens[:, 14:45], te_next_sens[:, 14:45]).item())
        est_l1 = (te_ctx.est_pred - te['est']).abs()
        est_smooth_l1 = float(F.smooth_l1_loss(te_ctx.est_pred, te['est'], beta=0.15).item())

        tr_him = F.normalize(
            model.him_src_proj(torch.cat((tr_ctx.z_mu, tr['act']), dim=-1)), dim=-1
        ).cpu().numpy()
        te_him = F.normalize(
            model.him_src_proj(torch.cat((te_ctx.z_mu, te['act']), dim=-1)), dim=-1
        ).cpu().numpy()

    y_tr = tr['zone'].cpu().numpy()
    y_te = te['zone'].cpu().numpy()
    z_tr = tr_ctx.z_mu.cpu().numpy()
    z_te = te_ctx.z_mu.cpu().numpy()
    joint_tr = np.concatenate((tr_ctx.est_pred.cpu().numpy(), z_tr), axis=-1)
    joint_te = np.concatenate((te_ctx.est_pred.cpu().numpy(), z_te), axis=-1)

    return {
        'model': variant_name,
        'ood_next_obs_mse': round(next_obs_mse, 5),
        'ood_contact_dyn_mse': round(dyn_mse, 5),
        'ood_est_smooth_l1': round(est_smooth_l1, 5),
        'ood_est_vx_mae_mm_s': round(float(est_l1[:, 0].mean().item() * 175.0), 2),
        'ood_est_wall_mae': round(float(est_l1[:, 3].mean().item()), 4),
        'ood_est_drag_mae': round(float(est_l1[:, 5].mean().item()), 4),
        'ood_latent_silhouette_euclidean': round(compute_silhouette_score(z_te, y_te, 'euclidean'), 4),
        'ood_latent_silhouette_cosine': round(compute_silhouette_score(z_te, y_te, 'cosine'), 4),
        'ood_latent_fisher_ratio': round(compute_fisher_discriminant_ratio(z_te, y_te), 4),
        'ood_latent_linear_probe_acc': round(linear_probe_zone_accuracy(z_tr, y_tr, z_te, y_te), 4),
        'ood_joint23d_silhouette_euclidean': round(compute_silhouette_score(joint_te, y_te, 'euclidean'), 4),
        'ood_joint23d_fisher_ratio': round(compute_fisher_discriminant_ratio(joint_te, y_te), 4),
        'ood_joint23d_linear_probe_acc': round(linear_probe_zone_accuracy(joint_tr, y_tr, joint_te, y_te), 4),
        'ood_him_sphere_silhouette_cosine': round(compute_silhouette_score(te_him, y_te, 'cosine'), 4),
        'ood_him_sphere_fisher_ratio': round(compute_fisher_discriminant_ratio(te_him, y_te), 4),
        'ood_him_sphere_linear_probe_acc': round(linear_probe_zone_accuracy(tr_him, y_tr, te_him, y_te), 4),
        'final_train_losses': {k: round(v, 5) for k, v in last_metrics.items()},
    }


def _benchmark_single_frame_cpu_latency(
    model: nn.Module,
    frame_size: int,
    history_len: int = 16,
    warmup: int = 50,
    iters: int = 400,
) -> dict[str, float]:
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = model.to('cpu').eval()
    x = torch.randn(1, history_len, frame_size)
    fn = model.action_with_estimate if hasattr(model, 'action_with_estimate') else model.action
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                fn(x)
            ms = np.empty(iters, dtype=np.float64)
            for i in range(iters):
                t0 = time.perf_counter()
                fn(x)
                ms[i] = (time.perf_counter() - t0) * 1000.0
    finally:
        torch.set_num_threads(prev_threads)

    return {
        'mean_ms': round(float(ms.mean()), 4),
        'p50_ms': round(float(np.percentile(ms, 50)), 4),
        'p95_ms': round(float(np.percentile(ms, 95)), 4),
        'p99_ms': round(float(np.percentile(ms, 99)), 4),
    }


def run_representation_and_latency_benchmark(
    steps_per_lane: int = 24,
    train_steps: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Benchmark (b): OOD multi-zone representation ablation + CPU single-frame latency."""
    driver_ckpt = REPO_ROOT / 'runs/v6_1_65d_elec/best.pt'
    data = _collect_multizone_physics_dataset(driver_ckpt, steps_per_lane=steps_per_lane, seed=seed)
    nn_dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    res_estnet = _train_and_eval_estnet_baseline(data, device=nn_dev, train_steps=train_steps, seed=seed)
    res_dreamwaq = _train_and_eval_him_dreamwaq_variant(
        data,
        variant_name='DreamWaQ_betaVAE_only',
        cfg=HIMDreamWaQConfig(beta_kl=0.002, est_coef=1.0, rec_coef=0.5, infonce_coef=0.0, swav_coef=0.0),
        device=nn_dev,
        train_steps=train_steps,
        seed=seed,
    )
    res_full = _train_and_eval_him_dreamwaq_variant(
        data,
        variant_name='HIMDreamWaQ_v8_full',
        cfg=HIMDreamWaQConfig(
            beta_kl=0.002, est_coef=1.0, rec_coef=0.5, infonce_coef=0.25, swav_coef=0.05,
            temperature=0.10, use_ema_target=True,
        ),
        device=nn_dev,
        train_steps=train_steps,
        seed=seed,
    )

    npz_path = REPO_ROOT / 'runs/v6b_65d_qerr/v6b_autonomous_pipe_traj.npz'
    logged_npz_ref: dict[str, Any] | None = None
    if npz_path.is_file():
        d = np.load(npz_path, allow_pickle=True)
        in_pipe = d['in_pipe'].astype(bool)
        on_soil = d['on_soil'].astype(bool)
        zone_ref = np.where(in_pipe, 1, np.where(on_soil, 2, 0))
        est_p = d['est_pred'].astype(np.float64)
        logged_npz_ref = {
            'source': 'runs/v6b_65d_qerr/v6b_autonomous_pipe_traj.npz',
            'frames': int(zone_ref.shape[0]),
            'zone_counts': {
                'hard': int((zone_ref == 0).sum()),
                'pipe': int((zone_ref == 1).sum()),
                'soft_soil': int((zone_ref == 2).sum()),
            },
            'logged_v6b_est7d_silhouette': round(compute_silhouette_score(est_p, zone_ref, 'euclidean'), 4),
            'logged_v6b_est7d_fisher': round(compute_fisher_discriminant_ratio(est_p, zone_ref), 4),
        }

    v6_65 = ProprioceptiveEstimatorActorCritic(frame_size=65, history_len=16, privileged_size=108, action_size=14)
    v6_51 = ProprioceptiveEstimatorActorCritic(frame_size=51, history_len=16, privileged_size=108, action_size=14)
    v8_65 = HIMDreamWaQActorCritic(frame_size=65, history_len=16, privileged_size=108, action_size=14)
    v8_51 = HIMDreamWaQActorCritic(frame_size=51, history_len=16, privileged_size=108, action_size=14)

    v6_deploy_params = sum(
        p.numel() for m in (v6_65.est_tcn, v6_65.est_mlp, v6_65.actor_tcn, v6_65.actor_mlp) for p in m.parameters()
    )
    v6_total_params = sum(p.numel() for p in v6_65.parameters())
    v8_params = v8_65.parameter_summary()

    latency = {
        'v6_estnet_65d': _benchmark_single_frame_cpu_latency(v6_65, 65),
        'v6_estnet_51d': _benchmark_single_frame_cpu_latency(v6_51, 51),
        'v8_him_dreamwaq_65d': _benchmark_single_frame_cpu_latency(v8_65, 65),
        'v8_him_dreamwaq_51d': _benchmark_single_frame_cpu_latency(v8_51, 51),
        'v6_equivariant_65d_exact': _benchmark_single_frame_cpu_latency(
            GroupEquivariantPolicyWrapper(v6_65, batched=False), 65
        ),
        'v6_equivariant_65d_batched': _benchmark_single_frame_cpu_latency(
            GroupEquivariantPolicyWrapper(v6_65, batched=True), 65
        ),
        'v8_equivariant_65d_exact': _benchmark_single_frame_cpu_latency(
            GroupEquivariantPolicyWrapper(v8_65, batched=False), 65
        ),
        'v8_equivariant_65d_batched': _benchmark_single_frame_cpu_latency(
            GroupEquivariantPolicyWrapper(v8_65, batched=True), 65
        ),
    }

    return {
        'dataset': {
            'driver_checkpoint': 'runs/v6_1_65d_elec/best.pt',
            'frame_size': data['frame_size'],
            'history_len': data['history_len'],
            'physics_dt_ms': 1.0,
            'elements_per_strip': 8,
            'train_transitions_nominal': int(data['train']['obs'].shape[0]),
            'test_transitions_ood': int(data['test_ood']['obs'].shape[0]),
            'zones': data['zone_names'],
            'ood_shifts': {
                'hard_floor': 'slope +3.0 deg, mu=0.50, rolling=0.055',
                'confined_pipe': 'y_allow=15 mm, k_wall=3800 N/m, mu=0.60',
                'soft_soil': 'slope -2.5 deg, rolling=0.260, hc_k=1.0e4, unload=3.8, rut_k=750, mu=0.55',
            },
        },
        'nn_train_device': nn_dev,
        'train_steps': int(train_steps),
        'models': [res_estnet, res_dreamwaq, res_full],
        'logged_3zone_reference': logged_npz_ref,
        'parameter_parity_65d': {
            'v6_deploy_params': int(v6_deploy_params),
            'v6_total_params': int(v6_total_params),
            **v8_params,
            'deploy_param_ratio_v8_vs_v6': round(v8_params['deploy_params'] / max(1, v6_deploy_params), 3),
        },
        'cpu_1thread_latency_ms': latency,
    }


def run_all_benchmarks(
    out_path: Path | None = None,
    *,
    closed_loop_steps: int = 12,
    repr_steps_per_lane: int = 20,
    train_steps: int = 220,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute both empirical v8 benchmarks deterministically and save JSON summary."""
    t0 = time.monotonic()
    bench_a = run_equivariance_benchmark(closed_loop_steps=closed_loop_steps, seed=seed)
    t_a = time.monotonic() - t0

    t1 = time.monotonic()
    bench_b = run_representation_and_latency_benchmark(
        steps_per_lane=repr_steps_per_lane, train_steps=train_steps, seed=seed
    )
    t_b = time.monotonic() - t1

    summary = {
        'schema': 'serpworm_v8_ablation_v1',
        'seed': int(seed),
        'physics_device': 'cpu',
        'nn_device': bench_b['nn_train_device'],
        'torch_version': torch.__version__,
        'elapsed_s': {
            'benchmark_a_equivariance': round(t_a, 2),
            'benchmark_b_representation_latency': round(t_b, 2),
            'total': round(t_a + t_b, 2),
        },
        'benchmark_a_equivariance': bench_a,
        'benchmark_b_representation_latency': bench_b,
    }

    target = out_path if out_path is not None else (REPO_ROOT / 'runs/v8_ablation/v8_ablation_summary.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description='SerpWorm-RL v8 HIM-DreamWaQ & Group-Equivariant Empirical Benchmarks')
    p.add_argument(
        '--out',
        type=Path,
        default=REPO_ROOT / 'runs/v8_ablation/v8_ablation_summary.json',
        help='Output JSON path',
    )
    p.add_argument('--closed-loop-steps', type=int, default=12)
    p.add_argument('--repr-steps', type=int, default=20)
    p.add_argument('--train-steps', type=int, default=220)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    summary = run_all_benchmarks(
        args.out,
        closed_loop_steps=args.closed_loop_steps,
        repr_steps_per_lane=args.repr_steps,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

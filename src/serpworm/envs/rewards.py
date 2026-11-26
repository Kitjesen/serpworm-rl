"""Locomotion and velocity reward objectives for SerpWorm-RL.

This module unifies:
1. ``locomotion_reward`` — the mainline multi-mode reward for
   ``UnifiedSerpWormEnv`` (straight speed tracking, steering, tight stop lock,
   mixed-mode dual-channel synergy, soft wheel-clearance barrier, smoothness,
   slip, and mechanical power regularization).
2. ``velocity_reward`` — the single-mode velocity tracking objective
   (``sustain_v4``) and archived numerical-comparison formulas used by
   ``TensorEnv``, fixed-wave replay probes, and contract tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

__all__ = [
    'UnifiedRewardConfig',
    'AutonomousRewardConfig',
    'locomotion_reward',
    'autonomous_phase_reward',
    'velocity_reward',
]


@dataclass
class UnifiedRewardConfig:
    """Weights and physical scales for :func:`locomotion_reward` (8-module physical objective)."""

    # 1. Forward speed tracking (short-window vx, tau=0.25 s)
    vx_sigma_ratio: float = 0.22
    vx_sigma_min_m_s: float = 0.012
    vx_track_weight: float = 1.65
    vx_progress_weight: float = 1.05
    vx_overshoot_weight: float = 0.65
    vx_progress_min_cmd_m_s: float = 0.020
    vx_progress_clip_m_s: tuple[float, float] = (-0.03, 0.14)

    # Stop lock (sub-1.5 mm/s standstill without saturation wall-slamming)
    stop_vx_sigma_m_s: float = 0.0025
    stop_wz_sigma_rad_s: float = 0.020
    stop_lock_weight: float = 1.85
    stop_creep_weight: float = 1.10
    stop_act_weight: float = 0.85
    stop_edge_sat_weight: float = 0.45

    # 2. Responsive + cycle-averaged yaw steering & heading tracking
    wz_sigma_ratio: float = 0.42
    wz_sigma_min_rad_s: float = 0.070
    wz_track_weight: float = 1.65
    wz_progress_weight: float = 1.35   # strong signed directional turning gradient when |wz_cmd| > 0
    wz_quad_weight: float = 0.20
    wz_turn_short_blend: float = 0.65  # blends tau=0.25s wz_short with tau=1.2s wz_avg during active turns

    # 3. Cycle-averaged net lateral drift (long-window vy_avg, frees periodic snake recoil)
    vy_penalty_weight: float = 0.08
    vy_scale_m_s: float = 0.025

    # 4. Efficiency-based slip (slip / (|vx| + v0), fair to wide-track one-way-wheel snake)
    slip_eff_weight: float = 0.04
    slip_eff_v0_m_s: float = 0.045

    # 5. Posture & ground-contact stability
    upright_weight: float = 0.35
    contact_weight: float = 0.15

    # 6. Hardware safety & anti-sticking boundaries
    barrier_weight: float = 0.45
    stop_shock_weight: float = 0.06
    stop_shock_thresh_n: float = 45.0  # penalize violent stop slamming (> 1 mm past stop), not normal full stroke
    edge_sat_weight: float = 0.08      # gentle rebound repulsion when moving & |cmd_state| > 0.90

    # 7. Zero-template gait emergence (mixed synergy + snake 4-joint balance & symmetry)
    synergy_weight: float = 0.60
    synergy_activity_scale: float = 0.65  # requires genuine dual-channel wave amplitude, not +/-2 deg jitter
    snake_activity_scale: float = 0.72    # requires full synchronized stroke across all 4 yaw joints
    snake_balance_weight: float = 0.42    # encourages 4-joint propulsion (frees steering hinges during turns)
    snake_symmetry_weight: float = 0.14   # keeps straight-command snake/mixed centered around 0 deg

    # 8. Action smoothness and mechanical power regularization
    rate_weight_move: float = 0.024
    rate_weight_stop: float = 0.75
    acc_weight_move: float = 0.024
    acc_weight_stop: float = 0.70
    work_weight: float = 0.005

    # Survival and physical termination
    alive_bonus: float = 0.05
    failure_penalty: float = 3.0

    @classmethod
    def v5_utm_calibrated(cls) -> UnifiedRewardConfig:
        """v5 reward weights retuned for 7x stiffer UTM-calibrated steel (42.7 N at 45 mm),
        high-speed tracking up to 240 mm/s, and non-punitive full-stroke stop clearance.
        """
        return cls(
            vx_sigma_ratio=0.16,
            vx_overshoot_weight=0.85,
            vx_progress_clip_m_s=(-0.03, 0.24),
            stop_shock_thresh_n=75.0,
            work_weight=0.0014,
        )


def _smooth_abs(x: torch.Tensor, eps: float = 5e-4) -> torch.Tensor:
    """C-infinity nonnegative magnitude matching |x| away from zero with zero slope at 0."""
    return torch.hypot(x, torch.full_like(x, eps)) - eps


def locomotion_reward(
    velocity: torch.Tensor,          # [B, 3]: short-window filtered (vx, vy, wz)
    command: torch.Tensor,           # [B, 3]: (vx_cmd, wz_cmd, stop_flag)
    mode_id: torch.Tensor,           # [B]: 0=snake, 1=worm, 2=mixed
    action_rate_sq: torch.Tensor,    # [B]: mean squared normalized target delta per 0.1 s
    action_acc_sq: torch.Tensor,     # [B]: mean squared second diff of normalized action
    slip: torch.Tensor,              # [B]: supported-wheel slip magnitude (m/s)
    work_rate: torch.Tensor,         # [B]: yaw+tendon mechanical power proxy (W)
    clearance: torch.Tensor,         # [B]: minimum certified wheel clearance (m)
    yaw_activity: torch.Tensor,      # [B]: mean |dq_yaw| normalized
    tendon_activity: torch.Tensor,   # [B]: mean |dservo| normalized
    failed: torch.Tensor,            # [B]: bool physical termination
    *,
    long_velocity: torch.Tensor | None = None,     # [B, 3]: cycle-averaged (tau=1.2 s) (vx, vy, wz)
    yaw_joint_act: torch.Tensor | None = None,     # [B, 4]: per-joint normalized |dq_yaw|
    upright_err: torch.Tensor | None = None,       # [B]: mean(1 - R_zz) across 10 plates
    contact_frac: torch.Tensor | None = None,      # [B]: fraction of 20 wheels supporting load
    stop_force_max: torch.Tensor | None = None,    # [B]: max mechanical stop force (N) across 5 segments
    edge_sat_sq: torch.Tensor | None = None,       # [B]: mean squared excess of |cmd_state| above 0.90
    long_mean_yaw_sq: torch.Tensor | None = None,  # [B]: mean squared cycle-averaged yaw angle (normalized)
    policy_dt: float = 0.1,
    clearance_min_m: float = 0.002,
    clearance_soft_m: float = 0.0042,
    config: UnifiedRewardConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute 8-module physical reward for unified 3-mode snake/worm/mixed RL."""
    policy_dt = float(policy_dt)
    if not math.isfinite(policy_dt) or policy_dt <= 0.0:
        raise ValueError('policy_dt must be finite and positive')

    cfg = config if config is not None else UnifiedRewardConfig()
    lv = velocity if long_velocity is None else long_velocity
    vx = velocity[:, 0]
    vy_avg, wz_avg = lv[:, 1], lv[:, 2]
    wz_short = velocity[:, 2]
    vx_cmd, wz_cmd, stop_flag = command[:, 0], command[:, 1], command[:, 2]
    is_stop = stop_flag > 0.5
    is_snake = mode_id == 0
    is_mixed = mode_id == 2
    is_turning = (~is_stop) & (wz_cmd.abs() > 0.015)
    zeros = torch.zeros_like(vx)
    ones = torch.ones_like(vx)

    # Effective steering yaw rate: fast short-window blend when actively turning for crisp
    # credit assignment across command switches; long cycle-average when straight to free snake sway.
    wz_eff = torch.where(
        is_turning,
        cfg.wz_turn_short_blend * wz_short + (1.0 - cfg.wz_turn_short_blend) * wz_avg,
        wz_avg,
    )
    wz_dir_prog = torch.where(
        is_turning,
        (wz_eff * wz_cmd.sign() / wz_cmd.abs().clamp_min(0.05)).clamp(-1.5, 1.0),
        zeros,
    )

    # 1. Forward speed tracking (sharp target peak + global progress + overspeed damping) & tight stop lock
    vx_cmd_safe = vx_cmd.clamp_min(cfg.vx_progress_min_cmd_m_s)
    sigma_vx = (cfg.vx_sigma_ratio * vx_cmd.abs()).clamp_min(cfg.vx_sigma_min_m_s)
    vx_track_move = torch.exp(-((vx - vx_cmd) / sigma_vx).square())
    vx_lo, vx_hi = cfg.vx_progress_clip_m_s
    vx_progress = (vx.clamp(vx_lo, vx_hi) / vx_cmd_safe).clamp(-1.0, 1.0)
    vx_overshoot = ((vx - vx_cmd).clamp_min(0.0) / vx_cmd_safe).clamp_max(1.5)
    turn_compliance = torch.where(is_turning, 0.65 + 0.35 * wz_dir_prog.clamp(0.0, 1.0), ones)
    move_score = (
        turn_compliance * (cfg.vx_track_weight * vx_track_move + cfg.vx_progress_weight * vx_progress)
        - cfg.vx_overshoot_weight * vx_overshoot
    )

    stop_vx = torch.exp(-(vx / cfg.stop_vx_sigma_m_s).square())
    stop_wz = torch.exp(-(wz_short / cfg.stop_wz_sigma_rad_s).square())
    stop_creep = (
        (_smooth_abs(vx) / 0.005).clamp_max(2.5)
        + 0.35 * (_smooth_abs(wz_short) / 0.03).clamp_max(2.0)
    )
    stop_act_pen = cfg.stop_act_weight * (
        yaw_activity.clamp(0.0, 2.0) + tendon_activity.clamp(0.0, 2.0)
    )
    stop_score = cfg.stop_lock_weight * stop_vx * stop_wz - cfg.stop_creep_weight * stop_creep - stop_act_pen
    r_vx = torch.where(is_stop, stop_score, move_score)

    # 2. Yaw steering & heading tracking (two-scale Gaussian prevents vanishing gradient near wz=0)
    progress_gate = torch.where(
        is_stop,
        zeros,
        (vx / vx_cmd_safe).clamp(0.0, 1.0),
    )
    sigma_wz = (cfg.wz_sigma_ratio * wz_cmd.abs()).clamp_min(cfg.wz_sigma_min_rad_s)
    wz_err_n = (wz_eff - wz_cmd) / sigma_wz
    wz_gauss = 0.70 * torch.exp(-wz_err_n.square()) + 0.30 * torch.exp(-(0.45 * wz_err_n).square())
    wz_gate = torch.where(is_turning, 0.40 + 0.60 * progress_gate, 0.20 + 0.80 * progress_gate)
    r_wz = torch.where(
        is_stop,
        zeros,
        cfg.wz_track_weight * wz_gate * wz_gauss
        + cfg.wz_progress_weight * wz_dir_prog
        - cfg.wz_quad_weight * ((wz_eff - wz_cmd) / 0.14).square().clamp_max(2.5),
    )

    # 3. Cycle-averaged net lateral drift
    r_vy = -cfg.vy_penalty_weight * (vy_avg / cfg.vy_scale_m_s).square().clamp_max(2.0)

    # 4. Efficiency-based slip ratio (fair to wide-track one-way-wheel snake)
    slip_eff = (slip / (_smooth_abs(vx) + cfg.slip_eff_v0_m_s)).clamp_max(3.0)
    r_slip = -cfg.slip_eff_weight * slip_eff

    # 5. Continuous posture & ground-contact stability
    r_posture = (
        (-cfg.upright_weight * upright_err.clamp_max(0.5) if upright_err is not None else zeros)
        + (cfg.contact_weight * (contact_frac - 0.5).clamp(-0.5, 0.5) if contact_frac is not None else zeros)
    )

    # 6. Hardware safety & anti-sticking boundaries (stronger edge repulsion during stop prevents wall-slamming)
    span = max(1e-4, clearance_soft_m - clearance_min_m)
    margin_deficit = ((clearance_soft_m - clearance) / span).clamp(0.0, 1.5)
    r_barrier = -cfg.barrier_weight * margin_deficit.square()
    if stop_force_max is not None:
        stop_excess = ((stop_force_max - cfg.stop_shock_thresh_n).clamp_min(0.0) / 50.0).clamp_max(2.0)
        r_shock = -cfg.stop_shock_weight * stop_excess.square()
    else:
        r_shock = zeros
    if edge_sat_sq is not None:
        edge_w = torch.where(is_stop, cfg.stop_edge_sat_weight, cfg.edge_sat_weight)
        r_edge = -edge_w * edge_sat_sq.clamp_max(2.0)
    else:
        r_edge = zeros

    # 7. Zero-template gait emergence (genuine mixed dual-channel synergy + snake 4-joint balance & symmetry)
    yaw_syn = (yaw_activity / cfg.synergy_activity_scale).clamp(0.0, 1.0)
    ten_syn = (tendon_activity / cfg.synergy_activity_scale).clamp(0.0, 1.0)
    both_active = torch.minimum(yaw_syn, ten_syn)
    if yaw_joint_act is not None:
        j_mix = (yaw_joint_act / cfg.synergy_activity_scale).clamp(0.0, 1.0)
        mix_yaw_coord = torch.where(
            is_turning,
            (j_mix.mean(-1) / 0.55).clamp(0.0, 1.0),
            0.55 * j_mix.mean(-1) + 0.45 * j_mix.amin(-1),
        )
        both_active = 0.55 * both_active + 0.45 * torch.minimum(ten_syn, mix_yaw_coord)

        snake_gate = torch.maximum(progress_gate, wz_dir_prog.clamp_min(0.0))
        j_norm = (yaw_joint_act / cfg.snake_activity_scale).clamp(0.0, 1.0)
        # Straight snake demands all 4 yaw joints undulate; turning frees steady Ackermann hinge joints
        j_balance = torch.where(
            is_turning,
            (j_norm.mean(-1) / 0.55).clamp(0.0, 1.0),
            0.45 * j_norm.mean(-1) + 0.55 * j_norm.amin(-1),
        )
        r_snake_bal = torch.where(is_snake & (~is_stop), cfg.snake_balance_weight * snake_gate * j_balance, zeros)
    else:
        r_snake_bal = zeros
    mix_gate = torch.maximum(progress_gate, 0.75 * wz_dir_prog.clamp(0.0, 1.0))
    r_synergy = torch.where(is_mixed & (~is_stop), cfg.synergy_weight * mix_gate * both_active, zeros)

    r_snake_sym = (
        torch.where((is_snake | is_mixed) & (~is_stop) & (~is_turning), -cfg.snake_symmetry_weight * long_mean_yaw_sq.clamp_max(1.5), zeros)
        if long_mean_yaw_sq is not None else zeros
    )

    # 8. Smoothness and mechanical power regularization
    rate_w = torch.where(is_stop, cfg.rate_weight_stop, cfg.rate_weight_move)
    acc_w = torch.where(is_stop, cfg.acc_weight_stop, cfg.acc_weight_move)
    r_reg = -(
        rate_w * action_rate_sq.clamp_max(6.0)
        + acc_w * action_acc_sq.clamp_max(6.0)
        + cfg.work_weight * work_rate.clamp_max(20.0)
    )

    # Survival & hard physical termination
    r_alive = torch.where(failed, -cfg.failure_penalty / max(policy_dt, 0.05), cfg.alive_bonus)

    total = policy_dt * (
        r_vx + r_wz + r_vy + r_slip + r_posture
        + r_barrier + r_shock + r_edge
        + r_synergy + r_snake_bal + r_snake_sym
        + r_reg + r_alive
    )
    terms = {
        'r_vx': (policy_dt * r_vx).detach(),
        'r_wz': (policy_dt * r_wz).detach(),
        'r_slip': (policy_dt * r_slip).detach(),
        'r_posture': (policy_dt * r_posture).detach(),
        'r_synergy': (policy_dt * r_synergy).detach(),
        'r_snake_bal': (policy_dt * r_snake_bal).detach(),
        'r_barrier': (policy_dt * r_barrier).detach(),
        'r_reg': (policy_dt * r_reg).detach(),
    }
    return total, terms


@dataclass
class AutonomousRewardConfig:
    """Pure-physics mode-free reward for v6 autonomous multi-terrain & pipe phase transitions.

    Zero mode IDs, zero gait templates, zero forced synergy/balance terms:
    gait transitions (snake <-> zero-slip worm <-> soil hybrid) and active proprioceptive
    probing emerge solely from velocity tracking, wall-impact cost, slip/power economy,
    and physical safety constraints.
    """

    # 1. Forward speed tracking (up to 240 mm/s for real UTM steel)
    vx_sigma_ratio: float = 0.18
    vx_sigma_min_m_s: float = 0.012
    vx_track_weight: float = 1.75
    vx_progress_weight: float = 1.35
    vx_overshoot_weight: float = 0.75
    vx_progress_min_cmd_m_s: float = 0.020
    vx_progress_clip_m_s: tuple[float, float] = (-0.03, 0.24)

    # Stop lock
    stop_vx_sigma_m_s: float = 0.0025
    stop_wz_sigma_rad_s: float = 0.020
    stop_lock_weight: float = 1.85
    stop_creep_weight: float = 1.10
    stop_act_weight: float = 0.80
    stop_edge_sat_weight: float = 0.45

    # 2. Yaw steering & heading tracking
    wz_sigma_ratio: float = 0.42
    wz_sigma_min_rad_s: float = 0.070
    wz_track_weight: float = 1.55
    wz_progress_weight: float = 1.25
    wz_quad_weight: float = 0.20
    wz_turn_short_blend: float = 0.65

    # 3. Lateral drift & pipe wall contact penalty
    # Straight pipes: strict wall_force_scale_n=2.5 N preserves zero-touch 0.1-0.3 N traversal.
    # Curved / sharp right-angle pipes (free_heading=True): allow 4-12 N glancing wall-guided
    # cornering & active dead-end probing without crushing forward arc-progress reward!
    vy_penalty_weight: float = 0.08
    vy_scale_m_s: float = 0.025
    wall_force_scale_n: float = 2.5
    wall_quad_weight: float = 0.75
    wall_lin_weight: float = 0.12
    curved_wall_force_scale_n: float = 12.0
    curved_wall_quad_weight: float = 0.18
    curved_wall_lin_weight: float = 0.05

    # 4. Efficiency-based slip & real DC-servo electrical power (mech + I^2R Joule heat)
    # Creates natural energetic separation: low-power pure snake on open hard floor,
    # zero-slip peristalsis + active probing inside narrow pipes, hybrid on soft soil.
    slip_eff_weight: float = 0.10
    slip_eff_v0_m_s: float = 0.040
    work_weight: float = 0.0028
    work_clamp_w: float = 40.0

    # 5. Posture & hardware safety
    upright_weight: float = 0.35
    contact_weight: float = 0.15
    barrier_weight: float = 0.45
    stop_shock_weight: float = 0.06
    stop_shock_thresh_n: float = 75.0
    edge_sat_weight: float = 0.06
    straight_yaw_sym_weight: float = 0.10

    # 6. Action smoothness
    rate_weight_move: float = 0.022
    rate_weight_stop: float = 0.75
    acc_weight_move: float = 0.022
    acc_weight_stop: float = 0.70

    # 7. Survival
    alive_bonus: float = 0.05
    failure_penalty: float = 3.0


def autonomous_phase_reward(
    velocity: torch.Tensor,          # [B, 3]: short-window filtered (vx, vy, wz)
    command: torch.Tensor,           # [B, 3]: (vx_cmd, wz_cmd, stop_flag)
    wall_force_max: torch.Tensor,    # [B]: max total lateral wall force over policy step (N)
    wall_force_mean: torch.Tensor,   # [B]: mean total lateral wall force over policy step (N)
    action_rate_sq: torch.Tensor,    # [B]: mean squared normalized target delta
    action_acc_sq: torch.Tensor,     # [B]: mean squared action acceleration
    slip: torch.Tensor,              # [B]: supported-wheel slip magnitude (m/s)
    work_rate: torch.Tensor,         # [B]: mechanical power proxy (W)
    clearance: torch.Tensor,         # [B]: min certified wheel clearance (m)
    yaw_activity: torch.Tensor,      # [B]: mean |dq_yaw| normalized
    tendon_activity: torch.Tensor,   # [B]: mean |dservo| normalized
    failed: torch.Tensor,            # [B]: bool physical termination
    *,
    long_velocity: torch.Tensor | None = None,
    upright_err: torch.Tensor | None = None,
    contact_frac: torch.Tensor | None = None,
    stop_force_max: torch.Tensor | None = None,
    edge_sat_sq: torch.Tensor | None = None,
    long_mean_yaw_sq: torch.Tensor | None = None,
    free_heading: torch.Tensor | None = None,
    policy_dt: float = 0.1,
    clearance_min_m: float = 0.002,
    clearance_soft_m: float = 0.0042,
    config: AutonomousRewardConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute pure-physics mode-free reward for autonomous terrain & pipe phase transitions."""
    policy_dt = float(policy_dt)
    cfg = config if config is not None else AutonomousRewardConfig()
    lv = velocity if long_velocity is None else long_velocity
    vx = velocity[:, 0]
    vy_avg, wz_avg = lv[:, 1], lv[:, 2]
    wz_short = velocity[:, 2]
    vx_cmd, wz_cmd, stop_flag = command[:, 0], command[:, 1], command[:, 2]
    is_stop = stop_flag > 0.5
    is_turning = (~is_stop) & (wz_cmd.abs() > 0.015)
    zeros = torch.zeros_like(vx)
    ones = torch.ones_like(vx)

    wz_eff = torch.where(
        is_turning,
        cfg.wz_turn_short_blend * wz_short + (1.0 - cfg.wz_turn_short_blend) * wz_avg,
        wz_avg,
    )
    wz_dir_prog = torch.where(
        is_turning,
        (wz_eff * wz_cmd.sign() / wz_cmd.abs().clamp_min(0.05)).clamp(-1.5, 1.0),
        zeros,
    )

    # 1. Forward speed tracking & tight stop lock
    vx_cmd_safe = vx_cmd.clamp_min(cfg.vx_progress_min_cmd_m_s)
    sigma_vx = (cfg.vx_sigma_ratio * vx_cmd.abs()).clamp_min(cfg.vx_sigma_min_m_s)
    vx_track_move = torch.exp(-((vx - vx_cmd) / sigma_vx).square())
    vx_lo, vx_hi = cfg.vx_progress_clip_m_s
    vx_progress = (vx.clamp(vx_lo, vx_hi) / vx_cmd_safe).clamp(-1.0, 1.0)
    vx_overshoot = ((vx - vx_cmd).clamp_min(0.0) / vx_cmd_safe).clamp_max(1.5)
    turn_compliance = torch.where(is_turning, 0.65 + 0.35 * wz_dir_prog.clamp(0.0, 1.0), ones)
    move_score = (
        turn_compliance * (cfg.vx_track_weight * vx_track_move + cfg.vx_progress_weight * vx_progress)
        - cfg.vx_overshoot_weight * vx_overshoot
    )

    stop_vx = torch.exp(-(vx / cfg.stop_vx_sigma_m_s).square())
    stop_wz = torch.exp(-(wz_short / cfg.stop_wz_sigma_rad_s).square())
    stop_creep = (
        (_smooth_abs(vx) / 0.005).clamp_max(2.5)
        + 0.35 * (_smooth_abs(wz_short) / 0.03).clamp_max(2.0)
    )
    stop_act_pen = cfg.stop_act_weight * (
        yaw_activity.clamp(0.0, 2.0) + tendon_activity.clamp(0.0, 2.0)
    )
    stop_score = cfg.stop_lock_weight * stop_vx * stop_wz - cfg.stop_creep_weight * stop_creep - stop_act_pen
    r_vx = torch.where(is_stop, stop_score, move_score)

    # 2. Yaw steering & heading tracking (relaxed when blindly navigating curved/right-angle pipes!)
    progress_gate = torch.where(is_stop, zeros, (vx / vx_cmd_safe).clamp(0.0, 1.0))
    sigma_wz = (cfg.wz_sigma_ratio * wz_cmd.abs()).clamp_min(cfg.wz_sigma_min_rad_s)
    wz_err_n = (wz_eff - wz_cmd) / sigma_wz
    wz_gauss = 0.70 * torch.exp(-wz_err_n.square()) + 0.30 * torch.exp(-(0.45 * wz_err_n).square())
    wz_gate = torch.where(is_turning, 0.40 + 0.60 * progress_gate, 0.20 + 0.80 * progress_gate)
    r_wz_raw = (
        cfg.wz_track_weight * wz_gate * wz_gauss
        + cfg.wz_progress_weight * wz_dir_prog
        - cfg.wz_quad_weight * ((wz_eff - wz_cmd) / 0.14).square().clamp_max(2.5)
    )
    if free_heading is not None:
        r_wz = torch.where(is_stop, zeros, torch.where(free_heading, cfg.wz_track_weight * progress_gate, r_wz_raw))
    else:
        r_wz = torch.where(is_stop, zeros, r_wz_raw)

    # 3. Lateral drift & quadratic pipe-wall impact penalty
    # Straight pipes keep strict 2.5 N penalty; curved/L-duct pipes (free_heading) permit glancing wall guidance!
    r_vy = -cfg.vy_penalty_weight * (vy_avg / cfg.vy_scale_m_s).square().clamp_max(2.0)
    if free_heading is not None:
        w_scale = torch.where(
            free_heading,
            torch.full_like(wall_force_max, max(cfg.curved_wall_force_scale_n, 1.0)),
            torch.full_like(wall_force_max, max(cfg.wall_force_scale_n, 0.5)),
        )
        w_qw = torch.where(
            free_heading,
            torch.full_like(wall_force_max, cfg.curved_wall_quad_weight),
            torch.full_like(wall_force_max, cfg.wall_quad_weight),
        )
        w_lw = torch.where(
            free_heading,
            torch.full_like(wall_force_max, cfg.curved_wall_lin_weight),
            torch.full_like(wall_force_max, cfg.wall_lin_weight),
        )
    else:
        w_scale = max(cfg.wall_force_scale_n, 0.5)
        w_qw = cfg.wall_quad_weight
        w_lw = cfg.wall_lin_weight
    w_norm_max = (wall_force_max / w_scale).clamp_max(4.0)
    w_norm_mean = (wall_force_mean / w_scale).clamp_max(4.0)
    r_wall = -(w_qw * w_norm_max.square() + w_lw * w_norm_mean)

    # 4. Slip efficiency & mechanical power economy
    slip_eff = (slip / (_smooth_abs(vx) + cfg.slip_eff_v0_m_s)).clamp_max(3.0)
    r_slip = -cfg.slip_eff_weight * slip_eff

    # 5. Posture & hardware safety
    r_posture = (
        (-cfg.upright_weight * upright_err.clamp_max(0.5) if upright_err is not None else zeros)
        + (cfg.contact_weight * (contact_frac - 0.5).clamp(-0.5, 0.5) if contact_frac is not None else zeros)
    )
    span = max(1e-4, clearance_soft_m - clearance_min_m)
    margin_deficit = ((clearance_soft_m - clearance) / span).clamp(0.0, 1.5)
    r_barrier = -cfg.barrier_weight * margin_deficit.square()
    if stop_force_max is not None:
        stop_excess = ((stop_force_max - cfg.stop_shock_thresh_n).clamp_min(0.0) / 50.0).clamp_max(2.0)
        r_shock = -cfg.stop_shock_weight * stop_excess.square()
    else:
        r_shock = zeros
    if edge_sat_sq is not None:
        edge_w = torch.where(is_stop, cfg.stop_edge_sat_weight, cfg.edge_sat_weight)
        r_edge = -edge_w * edge_sat_sq.clamp_max(2.0)
    else:
        r_edge = zeros
    no_free_h = (~free_heading) if free_heading is not None else ones.bool()
    r_sym = (
        torch.where((~is_stop) & (~is_turning) & no_free_h, -cfg.straight_yaw_sym_weight * long_mean_yaw_sq.clamp_max(1.5), zeros)
        if long_mean_yaw_sq is not None else zeros
    )

    # 6. Action smoothness & electrical power (mechanical + Joule heating)
    rate_w = torch.where(is_stop, cfg.rate_weight_stop, cfg.rate_weight_move)
    acc_w = torch.where(is_stop, cfg.acc_weight_stop, cfg.acc_weight_move)
    r_reg = -(
        rate_w * action_rate_sq.clamp_max(6.0)
        + acc_w * action_acc_sq.clamp_max(6.0)
        + cfg.work_weight * work_rate.clamp_max(cfg.work_clamp_w)
    )

    r_alive = torch.where(failed, -cfg.failure_penalty / max(policy_dt, 0.05), cfg.alive_bonus)
    total = policy_dt * (
        r_vx + r_wz + r_vy + r_wall + r_slip + r_posture
        + r_barrier + r_shock + r_edge + r_sym
        + r_reg + r_alive
    )
    terms = {
        'r_vx': (policy_dt * r_vx).detach(),
        'r_wz': (policy_dt * r_wz).detach(),
        'r_wall': (policy_dt * r_wall).detach(),
        'r_slip': (policy_dt * r_slip).detach(),
        'r_barrier': (policy_dt * r_barrier).detach(),
        'r_reg': (policy_dt * r_reg).detach(),
    }
    return total, terms


def velocity_reward(
    version: str,
    velocity: torch.Tensor,
    command: torch.Tensor,
    action_cost: torch.Tensor,
    slip: torch.Tensor,
    work: torch.Tensor,
    acceleration_cost: torch.Tensor | None = None,
    policy_dt: float = 0.2,
) -> torch.Tensor:
    """Return one scalar velocity-tracking reward per environment.

    Supported versions:
    - ``sustain_v4``: fixed physical speed scale with C1 positive exponential tail
      and rate/power rescaling by ``policy_dt``.
    - ``continuous_v3``, ``continuous_v2``, ``direct9d_v1``, ``tracking_v3``,
      ``tracking_v2``, ``legacy``: archived fixed-0.1 s comparison formulas.

    Parameters
    ----------
    velocity, command : Tensor [B, 3]
        ``(vx, vy, wz)`` in ``(m/s, m/s, rad/s)``.
    action_cost : Tensor [B]
        Nonnegative mean squared normalized applied target change over this macro action.
    slip : Tensor [B]
        Nonnegative supported-wheel slip magnitude (m/s).
    work : Tensor [B]
        Nonnegative integrated yaw-work proxy over this macro action (J).
    acceleration_cost : Tensor [B] | None
        Nonnegative mean squared second difference of normalized requested actions.
    policy_dt : float
        Macro action duration in seconds, finite and strictly positive.
    """
    if version == 'sustain_v4':
        policy_dt = float(policy_dt)
        if not math.isfinite(policy_dt) or policy_dt <= 0:
            raise ValueError('policy_dt must be finite and positive')
        h = lambda u: torch.hypot(u, torch.full_like(u, .002)) - .002
        moving_gain = (h(command[:, 0]) - h(velocity[:, 0] - command[:, 0])) / .04
        # Stop is an explicit task command with a tighter physical speed scale.
        gx = torch.where(command[:, 0] == 0., -h(velocity[:, 0]) / .008, moving_gain)
        yz_error = (velocity[:, 1:] - command[:, 1:]) / velocity.new_tensor([.005, .03])
        yz_cost = .5 * (torch.hypot(torch.ones_like(yz_error), yz_error) - 1).sum(-1)
        dt_ratio = .1 / policy_dt
        rate_cost = action_cost * dt_ratio ** 2
        acceleration = (acceleration_cost if acceleration_cost is not None else 0.) * dt_ratio ** 4
        penalty = .1 * rate_cost + .005 * acceleration + .05 * slip / .01 + .002 * work / policy_dt
        z = gx - yz_cost - penalty
        # Bound the unused exponential branch to prevent overflow/NaN gradients.
        tail = .5 * torch.exp(2 * (z + .5).clamp_max(0.))
        return policy_dt * torch.where(z >= -.5, 1 + z, tail)

    if version == 'continuous_v3':
        scales = torch.stack(((.2 * command[:, 0].abs()).clamp_min(.008),
                              torch.full_like(command[:, 0], .005),
                              torch.full_like(command[:, 0], .03)), -1)
        error = (velocity - command) / scales
        cost = ((torch.hypot(torch.ones_like(error), error) - 1) * velocity.new_tensor([1., .5, .5])).sum(-1)
        penalty = .01 * action_cost + .005 * (acceleration_cost if acceleration_cost is not None else 0.) + .005 * slip / .01 + .002 * work
        return .1 / (1 + cost + penalty / .1)

    if version == 'continuous_v2':
        scales = velocity.new_tensor([.008, .005, .03])
        weights = velocity.new_tensor([1., .5, .5])
        error = (velocity - command) / scales
        cost = torch.hypot(torch.ones_like(error), error) - 1
        tracking = 1 - (weights * cost).sum(-1)
        return .1 * tracking - .01 * action_cost - .005 * (acceleration_cost if acceleration_cost is not None else 0.) - .005 * slip / .01 - .002 * work

    ev = velocity[:, :2] - command[:, :2]
    ew = velocity[:, 2] - command[:, 2]

    if version == 'direct9d_v1':
        scales = velocity.new_tensor([.004, .004, .02])
        error = (velocity - command).square() / scales.square()
        active = command.abs() > velocity.new_tensor([5e-4, 5e-4, 2e-3])
        axis_score = (1 - error).clamp(-1., 1.)
        active_count = active.sum(-1)
        moving_score = (axis_score * active).sum(-1) / active_count.clamp_min(1)
        stop_error = error.mean(-1)
        stop_score = (1 - stop_error).clamp(-1., 1.)
        tracking = torch.where(active_count > 0, moving_score, stop_score)
        return .1 * tracking - .01 * action_cost - .005 * (acceleration_cost if acceleration_cost is not None else 0.) - .005 * slip / .01 - .002 * work

    if version == 'legacy':
        tracking = (-ev.square().sum(-1) / .006 ** 2).exp() + .5 * (-(ew / .025).square()).exp()
    elif version == 'tracking_v2':
        scale_x = command[:, 0].abs().clamp_min(.004)
        error = (ev[:, 0] / scale_x).square() + .25 * (ev[:, 1] / .004).square() + .5 * (ew / .02).square()
        tracking = (1 - error).clamp(0, 1)
    elif version == 'tracking_v3':
        if acceleration_cost is None:
            raise ValueError('tracking_v3 needs the requested-action second difference')
        scale_x = command[:, 0].abs().clamp_min(.004)
        error = (ev[:, 0] / scale_x).square() + .5 * (ev[:, 1] / .004).square() + (ew / .02).square()
        tracking = 1 / (1 + error)
        return .1 * (tracking - .05 * action_cost - .025 * acceleration_cost - .02 * slip / .01) - .01 * work
    else:
        raise ValueError(version)

    return .1 * (tracking - .02 * action_cost - .02 * slip / .01) - .01 * work

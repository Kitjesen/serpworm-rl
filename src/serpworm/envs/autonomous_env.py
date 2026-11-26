"""Mode-free Pure-Proprioceptive Autonomous Phase-Transition GPU Environment (v6).

Key design principles (100% deployable on FashionStar UART servos + single 6-axis IMU):
- ZERO mode_id, ZERO gait templates, ZERO action masks: full 14D actuation (10 independent
  tendons + 4 inter-segment yaw servos) is always open.
- Actor observation strictly excludes linear accelerometer, global position, velocity, slip,
  wall force, and terrain labels. Supports exact ablation switch:
    * include_q_err=False -> 51D/frame: [q(14), dq(14), gyro(3), proj_g(3), prev_action(14), cmd(3)]
    * include_q_err=True  -> 65D/frame: adds pure-software q_err = (q_target - q_meas)/scale (14D)
- Returns 7D EstNet supervision targets [vx, vy, wz, wall_confine, wall_side, terrain_drag, slip]
  for joint training of the pure-proprioceptive state/contact estimator.
- Runs all 4 physical regimes (open hard floor, pipe entry->traverse->exit, direct pipe->exit,
  carpet & soft soil with mid-episode transitions) simultaneously inside one captured CUDA Graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from serpworm.config import DomainRandConfig, HardwareConfig, TaskConfig
from serpworm.envs.rewards import AutonomousRewardConfig, autonomous_phase_reward
from serpworm.envs.unified_env import STATE_KEYS, UnifiedSerpWormEnv
from serpworm.physics.gpu_core import GPUCore, matrix
from serpworm.physics.pipe_geometry import eval_pipe_arc_progress_torch, eval_pipe_membership_torch
from serpworm.physics.wheel_envelope import active_pair_indices, certified_clearance

__all__ = [
    'REGIME_NAMES',
    'REGIME_NAMES_V7',
    'REGIME_NAMES_V8',
    'TERRAIN_NAMES',
    'TERRAIN_PROFILES',
    'AutonomousTaskConfig',
    'AutonomousPhaseEnv',
    'mirror_autonomous_batch',
]

REGIME_NAMES = ('open_hard', 'pipe_entry_exit', 'pipe_direct_exit', 'soft_terrain')
REGIME_NAMES_V7 = ('open_hard', 'straight_pipe', 'bend90_pipe', 'scurve_pipe', 'right_angle_L', 'soft_terrain')
REGIME_NAMES_V8 = (
    'open_hard', 'straight_pipe', 'bend90_pipe', 'scurve_pipe', 'right_angle_L',
    'u_turn_180', 'taper_reducer', 'double_L_Z', 'triple_wave', 'natural_terrain',
)
TERRAIN_NAMES = ('hard', 'carpet', 'soft_soil', 'gravel', 'grass')

# [wheel_rolling, ground_hc_k, ground_soil_unload_ratio, ground_rut_lat_k, wheel_mu_lat_ratio, wheel_mu]
TERRAIN_PROFILES = {
    'hard':      (0.025, 1.6e5, 1.00,   0.0, 1.05, 0.80),
    'carpet':    (0.140, 1.8e4, 1.00,   0.0, 1.20, 0.90),
    'soft_soil': (0.220, 1.2e4, 3.50, 650.0, 1.35, 0.70),
    'gravel':    (0.175, 3.2e4, 2.00, 280.0, 0.92, 0.62),  # loose crushed pebbles + micro-bumps
    'grass':     (0.195, 1.5e4, 1.35, 140.0, 1.25, 0.68),  # compliant turf + blade rolling drag
}

_PAIR_SWAP_10 = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8)
_TENSION_SWAP_20 = tuple(4 * s + k for s in range(5) for k in (2, 3, 0, 1))
_SUPPORT_SWAP_20 = tuple(2 * p + k for p in range(10) for k in (1, 0))


@dataclass
class AutonomousTaskConfig:
    """Task & multi-regime configuration for mode-free v6 autonomous phase-transition RL."""

    policy_dt_s: float = 0.10
    sensor_dt_s: float = 0.02
    physics_dt_s: float = 0.001     # 1 kHz FP64 implicit ARPC + implicit wall contact
    elements_per_strip: int = 16
    episode_seconds: float = 24.0   # long enough for approach -> pipe traversal -> exit recovery
    history_len: int = 16           # 1.6 s causal history (covers >1 full gait + probing cycle)

    include_q_err: bool = False     # Ablation switch: False=51D pure obs, True=65D (+software q_target-q)

    vx_min_mm_s: float = 35.0
    vx_max_mm_s: float = 175.0
    wz_max_rad_s: float = 0.26
    stop_fraction: float = 0.08
    straight_fraction: float = 0.45
    command_switch_s: float = 6.0

    # Regime mixture: (open_hard, pipe_entry_exit, pipe_direct_exit, soft_terrain)
    regime_probs: tuple[float, float, float, float] = (0.25, 0.35, 0.15, 0.25)

    # v7 curved & right-angle transparent pipe mode (uses unified pipe_cfg[B, 12] + 600-node strip contact)
    curved_pipes: bool = False
    curved_regime_probs: tuple[float, float, float, float, float, float] = (0.15, 0.15, 0.20, 0.20, 0.15, 0.15)

    # v8 expanded 10-regime diverse mode: 8 long/wide-diameter pipe types + gravel & grass natural terrains (no steps)
    diverse_v8: bool = False
    diverse_v8_probs: tuple[float, ...] = (0.07, 0.09, 0.10, 0.10, 0.11, 0.12, 0.11, 0.11, 0.09, 0.10)

    # Pipe geometry ranges (allowable plate-center |y| around 200 mm outer steel arch -> ID 230-250 mm)
    pipe_y_allow_range_m: tuple[float, float] = (0.015, 0.025)
    pipe_entry_ahead_range_m: tuple[float, float] = (0.28, 0.60)
    pipe_length_range_m: tuple[float, float] = (0.70, 1.15)
    pipe_direct_exit_range_m: tuple[float, float] = (0.50, 0.95)
    pipe_k_wall_range: tuple[float, float] = (2800.0, 3800.0)

    action_mode: str = 'delta'
    delta_step_scale: float = 0.50
    stage: int = 1


def mirror_autonomous_batch(
    obs: torch.Tensor,
    priv: torch.Tensor,
    est_tgt: torch.Tensor,
    latent: torch.Tensor,
    include_q_err: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact y -> -y left-right reflection of v6 observations, privileged state, EstNet targets, and 14D latents."""
    dev = obs.device
    p10 = torch.tensor(_PAIR_SWAP_10, device=dev, dtype=torch.long)
    t20 = torch.tensor(_TENSION_SWAP_20, device=dev, dtype=torch.long)
    s20 = torch.tensor(_SUPPORT_SWAP_20, device=dev, dtype=torch.long)

    m_obs = obs.clone()
    # q (0:14), dq (14:28)
    m_obs[..., 0:10] = obs[..., p10]
    m_obs[..., 10:14] = -obs[..., 10:14]
    m_obs[..., 14:24] = obs[..., 14 + p10]
    m_obs[..., 24:28] = -obs[..., 24:28]
    off = 28
    if include_q_err:
        m_obs[..., off:off + 10] = obs[..., off + p10]
        m_obs[..., off + 10:off + 14] = -obs[..., off + 10:off + 14]
        off += 14
    # Gyro (wx, wy, wz) -> (-wx, +wy, -wz)
    m_obs[..., off + 0] = -obs[..., off + 0]
    m_obs[..., off + 2] = -obs[..., off + 2]
    # Projected gravity (gx, gy, gz) -> (+gx, -gy, +gz)
    m_obs[..., off + 4] = -obs[..., off + 4]
    off += 6
    # Previous 14D action
    m_obs[..., off:off + 10] = obs[..., off + p10]
    m_obs[..., off + 10:off + 14] = -obs[..., off + 10:off + 14]
    off += 14
    # Command [vx, wz, stop] -> [+vx, -wz, +stop]
    m_obs[..., off + 1] = -obs[..., off + 1]

    m_lat = latent.clone()
    m_lat[..., 0:10] = latent[..., p10]
    m_lat[..., 10:14] = -latent[..., 10:14]

    m_est = est_tgt.clone()
    m_est[..., 1] = -est_tgt[..., 1]   # vy
    m_est[..., 2] = -est_tgt[..., 2]   # wz
    m_est[..., 4] = -est_tgt[..., 4]   # wall_side

    m_priv = priv.clone()
    for base in (0, 3, 6, 90):
        m_priv[..., base + 1] = -priv[..., base + 1]
        m_priv[..., base + 2] = -priv[..., base + 2]
    m_priv[..., 34:54] = priv[..., 34 + t20]
    m_priv[..., 54:74] = priv[..., 54 + s20]
    # Extra v6 privileged channels (93..101): negate wall_signed (95) and dy_center (101)
    m_priv[..., 95] = -priv[..., 95]
    m_priv[..., 101] = -priv[..., 101]
    # Extended DR/physics channels (102..107): negate slope_y at index 107
    if priv.shape[-1] > 107:
        m_priv[..., 107] = -priv[..., 107]
    return m_obs, m_priv, m_est, m_lat


class AutonomousPhaseEnv:
    """Batch-parallel GPU environment for v6 mode-free pure-proprioceptive phase transitions."""

    def __init__(
        self,
        n: int = 1024,
        hardware: HardwareConfig | None = None,
        task: AutonomousTaskConfig | None = None,
        dr: DomainRandConfig | None = None,
        reward_config: AutonomousRewardConfig | None = None,
        device: str = 'cuda',
        plate_solver: str = 'fused_cuda',
        seed: int = 7,
        noise: bool = True,
        forced_regime: int | None = None,
        settle_ticks: int = 120,
    ):
        self.n = int(n)
        if hardware is None:
            hardware = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
            hardware.yaw_slew_rad_s = 6.8
        if hardware.tendon_pairing != 'independent_10d':
            raise ValueError('AutonomousPhaseEnv requires 14D independent_10d hardware')
        self.hw = hardware
        self.task = task or AutonomousTaskConfig()
        self.dr = dr or DomainRandConfig()
        self.reward_config = reward_config or AutonomousRewardConfig()
        self.noise = bool(noise)
        self.forced_regime = forced_regime
        self.stage = int(self.task.stage)
        self.include_q_err = bool(self.task.include_q_err)

        self.sensor_dt = float(self.task.sensor_dt_s)
        self.policy_dt = float(self.task.policy_dt_s)
        self.physics_dt = float(self.task.physics_dt_s)
        self.elements_per_strip = int(self.task.elements_per_strip)
        self.steps_per_action = round(self.policy_dt / self.sensor_dt)
        self.substeps_per_sensor = round(self.sensor_dt / self.physics_dt)
        self.episode_policy_steps = round(self.task.episode_seconds / self.policy_dt)
        self.command_switch_steps = max(1, round(self.task.command_switch_s / self.policy_dt))
        self.history_len = int(self.task.history_len)
        self.action_size = 14
        self.est_size = 7

        self.yaw_gain = 1.0 - math.exp(-self.sensor_dt / max(self.hw.yaw_filter_tau_s, 1e-4))
        self.yaw_slew = self.hw.yaw_slew_rad_s * self.sensor_dt
        self.servo_gain = 1.0 - math.exp(-self.sensor_dt / max(self.hw.servo_filter_tau_s, 1e-4))
        self.servo_slew = self.hw.servo_slew_rad_s * self.sensor_dt

        if device != 'cuda':
            solver = 'segment_lu'
        elif plate_solver == 'fused_cuda':
            from serpworm.physics.fused_cuda import is_available as fused_ok
            solver = 'fused_cuda' if fused_ok() else 'segment_lu'
        else:
            solver = plate_solver

        graph_steps = self.substeps_per_sensor if solver == 'fused_cuda' else 1
        if solver != 'fused_cuda':
            for cand in (8, 5, 4, 2):
                if self.substeps_per_sensor % cand == 0 and cand <= 8:
                    graph_steps = cand
                    break

        # 1. Single-instance neutral settling on open hard floor
        settler = GPUCore(
            5, 1, dt=self.physics_dt, device=device, plate_solver=solver,
            hardware=self.hw, elements_per_strip=self.elements_per_strip, tangent_interval=1,
        )
        if device == 'cuda':
            settler.capture(graph_steps)
        pre_ticks = min(20, max(1, int(settle_ticks) // 6))
        post_ticks = max(1, int(settle_ticks) - pre_ticks)
        for _ in range(pre_ticks):
            settler.advance_steps(self.substeps_per_sensor)
        settler.target.fill_(self.hw.servo_neutral_rad)
        for _ in range(post_ticks):
            settler.advance_steps(self.substeps_per_sensor)
        if bool(settler.failed) or not bool(torch.isfinite(settler.x).all()):
            raise RuntimeError('Neutral settling failed for AutonomousPhaseEnv')
        settled_single = {k: getattr(settler, k).clone() for k in STATE_KEYS}
        del settler

        self.c = GPUCore(
            5, self.n, dt=self.physics_dt, device=device, plate_solver=solver,
            hardware=self.hw, elements_per_strip=self.elements_per_strip, tangent_interval=1,
        )
        for k, val in settled_single.items():
            getattr(self.c, k).copy_(val.expand(self.n, *val.shape[1:]))
        if device == 'cuda':
            self.c.capture(graph_steps)
        self.initial = {k: getattr(self.c, k).clone() for k in STATE_KEYS}

        self.device, self.dtype = self.c.device, self.c.dtype
        self.rng = torch.Generator(device=self.device).manual_seed(int(seed))

        # Control & state buffers
        self.servo_neutral = torch.full((self.n, 10), self.hw.servo_neutral_rad, device=self.device, dtype=self.dtype)
        self.servo_targets = self.servo_neutral.clone()
        self.yaw_targets = torch.zeros((self.n, 4), device=self.device, dtype=self.dtype)
        self.prev_action = torch.zeros((self.n, 14), device=self.device, dtype=self.dtype)
        self.prev_action_delta = torch.zeros((self.n, 14), device=self.device, dtype=self.dtype)
        self.cmd = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.step_count = torch.zeros(self.n, device=self.device, dtype=torch.long)

        # Regime & dynamic terrain tracking (ONLY for environment simulation, Critic, & logging)
        self.regime_id = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.terrain_id = torch.zeros(self.n, device=self.device, dtype=torch.long)  # 0=hard, 1=carpet, 2=soft_soil
        self.post_pipe_soil = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.has_entered_pipe = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.has_exited_pipe = torch.zeros(self.n, device=self.device, dtype=torch.bool)

        # Sensing & kinematics buffers
        self.prev_angles = torch.zeros((self.n, 14), device=self.device, dtype=self.dtype)
        self.dq = torch.zeros((self.n, 14), device=self.device, dtype=self.dtype)
        self.delayed_raw = torch.zeros((self.n, 20), device=self.device, dtype=self.dtype)
        self.filtered_vel = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.long_filtered_vel = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.long_mean_yaw = torch.zeros((self.n, 4), device=self.device, dtype=self.dtype)
        self.last_body_vel = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.last_clearance = torch.full((self.n,), 0.03, device=self.device, dtype=self.dtype)
        self.wall_ema = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.wall_signed_ema = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_wall_max = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_wall_mean = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_strip_wall_mean = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_min_strip_z = torch.full((self.n,), 0.05, device=self.device, dtype=self.dtype)
        self.prev_pipe_arc = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.filtered_v_arc = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_v_arc = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_slip_scalar = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_guard_reasons: dict[str, torch.Tensor] = {}
        self.last_reward_terms: dict[str, torch.Tensor] = {}

        # DR & sensor bias state
        self.dr_params = torch.zeros((self.n, 4), device=self.device, dtype=self.dtype)
        self.dr_scale = 1.0 if self.dr.enabled else 0.0
        self.joint_bias = torch.zeros((self.n, 14), device=self.device, dtype=self.dtype)
        self.gyro_bias = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.gravity_rot = torch.eye(3, device=self.device, dtype=self.dtype)[None].repeat(self.n, 1, 1)
        self.servo_gain_env = torch.full((self.n, 1), self.servo_gain, device=self.device, dtype=self.dtype)
        self.yaw_gain_env = torch.full((self.n, 1), self.yaw_gain, device=self.device, dtype=self.dtype)
        self.base_wheel_mu = torch.full((self.n,), self.hw.wheel_mu, device=self.device, dtype=self.dtype)
        self.dr_mu_scale = torch.ones((self.n,), device=self.device, dtype=self.dtype)
        self.dr_terr_scale = torch.ones((self.n, 5), device=self.device, dtype=self.dtype)
        self._push_interval_steps = (
            max(1, round(float(getattr(self.dr, 'push_interval_s', 0.0)) / self.policy_dt))
            if float(getattr(self.dr, 'push_interval_s', 0.0)) > 0.0 else 0
        )

        # Frame size:
        # 51D = 14 q + 14 dq + 3 gyro + 3 proj_g + 14 prev_act + 3 cmd
        # 65D = 51D + 14 q_err (pure software target - encoder)
        self.frame_size = 51 + (14 if self.include_q_err else 0)
        self.privileged_size = 108  # 102 core v6 + 6 extended DR/physics (backlash, capstan, yaw_sf, beta, slope_x, slope_y)
        self.history = torch.zeros((self.n, self.history_len, self.frame_size), device=self.device, dtype=torch.float32)

        self.angle_scale = torch.tensor(
            [max(self.hw.servo_max_rad, 1e-3)] * 10 + [max(self.hw.yaw_action_limit_rad, 1e-3)] * 4,
            device=self.device, dtype=self.dtype,
        )
        self.noise_scale = torch.tensor(
            [0.002] * 14 + [0.01] * 3 + [0.005] * 3, device=self.device, dtype=self.dtype,
        )
        self.frame_flip = torch.tensor([-1.0, -1.0, 1.0], device=self.device, dtype=self.dtype)
        self.no_reset = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        owners = torch.arange(10, device=self.device).repeat_interleave(2)
        self.wheel_pair = owners[:, None] != owners[None, :]
        self.wheel_pair_indices = active_pair_indices(self.wheel_pair)
        if self.task.diverse_v8:
            probs = self.task.diverse_v8_probs
        elif self.task.curved_pipes:
            probs = self.task.curved_regime_probs
        else:
            probs = self.task.regime_probs
        self.regime_cdf = torch.tensor(probs, device=self.device, dtype=self.dtype).cumsum(0)
        self.regime_cdf = self.regime_cdf / self.regime_cdf[-1]

        # Prebuild terrain table [5, 6] on device (hard, carpet, soft_soil, gravel, grass)
        self._use_gpipe_mode = bool(self.task.curved_pipes or self.task.diverse_v8)
        self._hard_rolling_nom = float(self.hw.wheel_rolling) if self._use_gpipe_mode else float(TERRAIN_PROFILES['hard'][0])
        hard_prof = (self._hard_rolling_nom, *TERRAIN_PROFILES['hard'][1:])
        self._terrain_table = torch.tensor(
            [
                hard_prof,
                TERRAIN_PROFILES['carpet'],
                TERRAIN_PROFILES['soft_soil'],
                TERRAIN_PROFILES['gravel'],
                TERRAIN_PROFILES['grass'],
            ],
            device=self.device, dtype=self.dtype,
        )

        self.obs, self.critic_obs, self.est_target = self.reset(torch.ones(self.n, device=self.device, dtype=torch.bool))

    def set_stage(self, stage: int, dr_scale: float | None = None) -> None:
        self.stage = int(stage)
        if self.stage >= 3:
            self.dr.enabled = True
            if dr_scale is not None:
                self.dr_scale = float(max(0.0, min(1.0, dr_scale)))
            elif self.dr_scale <= 0.0:
                self.dr_scale = 1.0

    def set_dr_scale(self, scale: float) -> None:
        self.dr_scale = float(max(0.0, min(1.0, scale)))

    def _rand(self, *shape):
        return torch.rand(shape, generator=self.rng, device=self.device, dtype=self.dtype)

    def _randn(self, *shape):
        return torch.randn(shape, generator=self.rng, device=self.device, dtype=self.dtype)

    def _is_pipe_regime(self) -> torch.Tensor:
        if self.task.diverse_v8:
            return (self.regime_id >= 1) & (self.regime_id <= 8)
        if self.task.curved_pipes:
            return (self.regime_id >= 1) & (self.regime_id <= 4)
        return (self.regime_id == 1) | (self.regime_id == 2)

    def _is_soil_regime(self) -> torch.Tensor:
        if self.task.diverse_v8:
            return self.regime_id == 9
        return (self.regime_id == 5) if self.task.curved_pipes else (self.regime_id == 3)

    def _apply_terrain_ids(self, mask: torch.Tensor, terr_ids: torch.Tensor) -> None:
        """Write per-env ARPC terrain tensors in-place (zero CUDA graph recapture)."""
        prof = self._terrain_table[terr_ids]  # [B, 6]
        self.terrain_id.copy_(torch.where(mask, terr_ids, self.terrain_id))
        terr_eff = prof[:, :5] * self.dr_terr_scale
        if self._use_gpipe_mode:
            terr_eff[:, 0] = terr_eff[:, 0].clamp(0.020, 0.280)
        terr_eff[:, 2] = torch.where(prof[:, 2] <= 1.0 + 1e-6, prof[:, 2], terr_eff[:, 2].clamp_min(1.05))
        self.c.env_terrain.copy_(torch.where(mask[:, None], terr_eff, self.c.env_terrain))
        self.base_wheel_mu.copy_(torch.where(mask, prof[:, 5], self.base_wheel_mu))
        eff_mu = (self.base_wheel_mu * self.dr_mu_scale).clamp(0.35, 1.25)
        self.c.wheel_mu.copy_(torch.where(mask[:, None, None], eff_mu[:, None, None], self.c.wheel_mu))

    def _sample_regimes_and_geometry(self, mask: torch.Tensor) -> None:
        if self.task.diverse_v8:
            if self.forced_regime is not None:
                reg = torch.full((self.n,), int(self.forced_regime), device=self.device, dtype=torch.long)
            else:
                u = self._rand(self.n)
                reg = torch.zeros(self.n, device=self.device, dtype=torch.long)
                for rid in range(1, len(self.regime_cdf)):
                    reg = torch.where(u >= self.regime_cdf[rid - 1], torch.full_like(reg, rid), reg)
            self.regime_id.copy_(torch.where(mask, reg, self.regime_id))

            # Natural outdoor terrain regime (reg=9): 35% gravel, 35% grass, 20% soft_soil, 10% carpet (ZERO steps!)
            u_nat = self._rand(self.n)
            nat_pick = torch.where(
                u_nat < 0.35, torch.full_like(reg, 3),
                torch.where(u_nat < 0.70, torch.full_like(reg, 4),
                            torch.where(u_nat < 0.90, torch.full_like(reg, 2), torch.ones_like(reg))),
            )
            terr = torch.where(reg == 9, nat_pick, torch.zeros_like(reg))
            self._apply_terrain_ids(mask, terr)
            p_soil = torch.where((reg == 1) | (reg == 6), 0.55, 0.35)
            self.post_pipe_soil.copy_(torch.where(mask, self._rand(self.n) < p_soil, self.post_pipe_soil))

            # Disable legacy straight wall_cfg; populate unified pipe_cfg[B, 12] across all 8 analytical pipe types
            wcfg = torch.zeros((self.n, 5), device=self.device, dtype=self.dtype)
            wcfg[:, 1:4] = torch.tensor([1e6, 1e6, -1e6], device=self.device, dtype=self.dtype)
            self.c.wall_cfg.copy_(torch.where(mask[:, None], wcfg, self.c.wall_cfg))

            head_x = self.c.x[:, 9, 0]
            y0 = self.c.x[:, :10, 1].mean(-1)
            sgn = torch.where(self._rand(self.n) < 0.5, 1.0, -1.0)
            ahead = 0.04 + 0.22 * self._rand(self.n)
            lo_k, hi_k = self.task.pipe_k_wall_range
            kw = lo_k + (hi_k - lo_k) * self._rand(self.n)

            pcfg = torch.zeros((self.n, 12), device=self.device, dtype=self.dtype)
            pcfg[:, 2] = 1e6
            pcfg[:, 3] = 1e6
            pcfg[:, 4] = -1e6
            pcfg[:, 5] = 0.52
            pcfg[:, 7] = sgn
            pcfg[:, 8] = 0.90
            pcfg[:, 9] = 0.118
            pcfg[:, 11] = 0.008

            # Regime 1: Wide-diameter-span long straight cylinder (ID 216-310 mm, length 1.10-2.10 m)
            m1 = reg == 1
            x_en1 = head_x - ahead
            x_ex1 = x_en1 - (1.10 + 1.00 * self._rand(self.n))
            r1 = 0.108 + 0.047 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m1, 1.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m1, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m1, r1, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m1, x_en1, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m1, x_ex1, pcfg[:, 4])
            pcfg[:, 10] = torch.where(m1, kw, pcfg[:, 10])

            # Regime 2: Long 90-deg circular arc (ID 238-285 mm, R=0.46-0.68 m, L_exit=0.90-1.40 m -> 2.0-2.8 m total)
            m2 = reg == 2
            x_en2 = head_x - ahead
            x_b2 = x_en2 - (0.22 + 0.20 * self._rand(self.n))
            R2 = 0.46 + 0.22 * self._rand(self.n)
            r2 = 0.119 + 0.0235 * self._rand(self.n)
            lex2 = 0.90 + 0.50 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m2, 2.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m2, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m2, r2, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m2, x_en2, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m2, x_b2, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m2, R2, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m2, 0.5 * math.pi, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m2, lex2, pcfg[:, 8])
            pcfg[:, 10] = torch.where(m2, kw, pcfg[:, 10])

            # Regime 3: Long S-curve double-arc (ID 238-285 mm, R=0.48-0.68 m, +/-50..75 deg -> 2.4-3.4 m total)
            m3 = reg == 3
            x_en3 = head_x - ahead
            x_b3 = x_en3 - (0.20 + 0.18 * self._rand(self.n))
            R3 = 0.48 + 0.20 * self._rand(self.n)
            ang3 = math.radians(50.0) + math.radians(25.0) * self._rand(self.n)
            r3 = 0.119 + 0.0235 * self._rand(self.n)
            lex3 = 0.85 + 0.50 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m3, 3.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m3, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m3, r3, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m3, x_en3, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m3, x_b3, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m3, R3, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m3, ang3, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m3, lex3, pcfg[:, 8])
            pcfg[:, 10] = torch.where(m3, kw, pcfg[:, 10])

            # Regime 4: Long sharp 90-deg L-duct (W=256-300 mm, H=235-275 mm, r_fil=6-18 mm -> 1.5-2.2 m total)
            m4 = reg == 4
            x_en4 = head_x - ahead
            x_b4 = x_en4 - (0.32 + 0.26 * self._rand(self.n))
            hw4 = 0.128 + 0.022 * self._rand(self.n)
            h4 = 0.235 + 0.040 * self._rand(self.n)
            rf4 = 0.006 + 0.012 * self._rand(self.n)
            lex4 = 0.95 + 0.50 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m4, 4.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m4, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m4, hw4, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m4, x_en4, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m4, x_b4, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m4, 0.0, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m4, 0.5 * math.pi, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m4, lex4, pcfg[:, 8])
            pcfg[:, 9] = torch.where(m4, h4, pcfg[:, 9])
            pcfg[:, 10] = torch.where(m4, kw, pcfg[:, 10])
            pcfg[:, 11] = torch.where(m4, rf4, pcfg[:, 11])

            # Regime 5: 180-deg Hairpin U-Turn pipe (ID 242-285 mm, R=0.48-0.65 m, L_exit=0.90-1.40 m -> 2.8-3.8 m!)
            m5 = reg == 5
            x_en5 = head_x - ahead
            x_b5 = x_en5 - (0.22 + 0.18 * self._rand(self.n))
            R5 = 0.48 + 0.17 * self._rand(self.n)
            r5 = 0.121 + 0.0215 * self._rand(self.n)
            lex5 = 0.90 + 0.50 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m5, 5.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m5, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m5, r5, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m5, x_en5, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m5, x_b5, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m5, R5, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m5, math.pi, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m5, lex5, pcfg[:, 8])
            pcfg[:, 10] = torch.where(m5, kw, pcfg[:, 10])

            # Regime 6: Smooth conical taper / reducer (ID 285-325 mm -> 216-234 mm, 1.6-2.4 m total)
            m6 = reg == 6
            x_en6 = head_x - ahead
            x_b6 = x_en6 - (0.22 + 0.18 * self._rand(self.n))
            L_tap6 = 0.50 + 0.35 * self._rand(self.n)
            r_in6 = 0.1425 + 0.020 * self._rand(self.n)
            r_out6 = 0.108 + 0.009 * self._rand(self.n)
            lex6 = 0.80 + 0.45 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m6, 6.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m6, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m6, r_in6, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m6, x_en6, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m6, x_b6, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m6, L_tap6, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m6, 0.0, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m6, lex6, pcfg[:, 8])
            pcfg[:, 10] = torch.where(m6, kw, pcfg[:, 10])
            pcfg[:, 11] = torch.where(m6, r_out6, pcfg[:, 11])

            # Regime 7: Z-shaped double 90-deg sharp square chicane (W=258-296 mm, L_mid=0.58-0.82 m -> 2.1-2.8 m!)
            m7 = reg == 7
            x_en7 = head_x - ahead
            x_b7 = x_en7 - (0.30 + 0.20 * self._rand(self.n))
            L_mid7 = 0.58 + 0.24 * self._rand(self.n)
            hw7 = 0.129 + 0.019 * self._rand(self.n)
            h7 = 0.240 + 0.035 * self._rand(self.n)
            rf7 = 0.006 + 0.010 * self._rand(self.n)
            lex7 = 0.85 + 0.40 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m7, 7.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m7, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m7, hw7, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m7, x_en7, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m7, x_b7, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m7, L_mid7, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m7, 0.5 * math.pi, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m7, lex7, pcfg[:, 8])
            pcfg[:, 9] = torch.where(m7, h7, pcfg[:, 9])
            pcfg[:, 10] = torch.where(m7, kw, pcfg[:, 10])
            pcfg[:, 11] = torch.where(m7, rf7, pcfg[:, 11])

            # Regime 8: Long triple-wave serpentine pipe (+phi -> -2*phi -> +phi, 3.2-4.0 m total!)
            m8 = reg == 8
            x_en8 = head_x - ahead
            x_b8 = x_en8 - (0.20 + 0.16 * self._rand(self.n))
            R8 = 0.50 + 0.16 * self._rand(self.n)
            ang8 = math.radians(50.0) + math.radians(12.0) * self._rand(self.n)
            r8 = 0.121 + 0.0215 * self._rand(self.n)
            lex8 = 0.80 + 0.45 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m8, 8.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m8, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m8, r8, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m8, x_en8, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m8, x_b8, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m8, R8, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m8, ang8, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m8, lex8, pcfg[:, 8])
            pcfg[:, 10] = torch.where(m8, kw, pcfg[:, 10])

            self.c.pipe_cfg.copy_(torch.where(mask[:, None], pcfg, self.c.pipe_cfg))
            self.has_entered_pipe.copy_(torch.where(mask, torch.zeros_like(mask), self.has_entered_pipe))
            self.has_exited_pipe.copy_(torch.where(mask, torch.zeros_like(mask), self.has_exited_pipe))
            return

        if self.task.curved_pipes:
            if self.forced_regime is not None:
                reg = torch.full((self.n,), int(self.forced_regime), device=self.device, dtype=torch.long)
            else:
                u = self._rand(self.n)
                reg = torch.zeros(self.n, device=self.device, dtype=torch.long)
                for rid in range(1, len(self.regime_cdf)):
                    reg = torch.where(u >= self.regime_cdf[rid - 1], torch.full_like(reg, rid), reg)
            self.regime_id.copy_(torch.where(mask, reg, self.regime_id))

            soil_pick = torch.where(self._rand(self.n) < 0.40, torch.ones_like(reg), torch.full_like(reg, 2))
            terr = torch.where(reg == 5, soil_pick, torch.zeros_like(reg))
            self._apply_terrain_ids(mask, terr)
            p_soil = torch.where(reg == 1, 0.50, 0.35)
            self.post_pipe_soil.copy_(torch.where(mask, self._rand(self.n) < p_soil, self.post_pipe_soil))

            # Disable legacy straight wall_cfg; populate unified pipe_cfg[B, 12]
            wcfg = torch.zeros((self.n, 5), device=self.device, dtype=self.dtype)
            wcfg[:, 1] = 1e6
            wcfg[:, 2] = 1e6
            wcfg[:, 3] = -1e6
            self.c.wall_cfg.copy_(torch.where(mask[:, None], wcfg, self.c.wall_cfg))

            head_x = self.c.x[:, 9, 0]
            y0 = self.c.x[:, :10, 1].mean(-1)
            sgn = torch.where(self._rand(self.n) < 0.5, 1.0, -1.0)
            ahead = 0.04 + 0.22 * self._rand(self.n)
            lo_k, hi_k = self.task.pipe_k_wall_range
            kw = lo_k + (hi_k - lo_k) * self._rand(self.n)

            pcfg = torch.zeros((self.n, 12), device=self.device, dtype=self.dtype)
            pcfg[:, 2] = 1e6
            pcfg[:, 3] = 1e6
            pcfg[:, 4] = -1e6
            pcfg[:, 5] = 0.50
            pcfg[:, 7] = sgn
            pcfg[:, 8] = 0.75
            pcfg[:, 9] = 0.118
            pcfg[:, 11] = 0.008

            # Regime 1: straight transparent cylinder (ID 232-248 mm)
            m1 = reg == 1
            x_en1 = head_x - ahead
            x_ex1 = x_en1 - (0.75 + 0.35 * self._rand(self.n))
            r1 = 0.116 + 0.008 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m1, 1.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m1, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m1, r1, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m1, x_en1, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m1, x_ex1, pcfg[:, 4])
            pcfg[:, 10] = torch.where(m1, kw, pcfg[:, 10])

            # Regime 2: 90-deg smooth circular arc pipe (ID 244-256 mm, R=0.48-0.56 m)
            m2 = reg == 2
            x_en2 = head_x - ahead
            x_b2 = x_en2 - (0.18 + 0.14 * self._rand(self.n))
            R2 = 0.48 + 0.08 * self._rand(self.n)
            r2 = 0.122 + 0.006 * self._rand(self.n)
            lex2 = 0.60 + 0.20 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m2, 2.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m2, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m2, r2, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m2, x_en2, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m2, x_b2, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m2, R2, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m2, 0.5 * math.pi, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m2, lex2, pcfg[:, 8])
            pcfg[:, 10] = torch.where(m2, kw, pcfg[:, 10])

            # Regime 3: S-curve double-arc pipe (ID 244-256 mm, R=0.50-0.58 m, +/-55..65 deg)
            m3 = reg == 3
            x_en3 = head_x - ahead
            x_b3 = x_en3 - (0.16 + 0.14 * self._rand(self.n))
            R3 = 0.50 + 0.08 * self._rand(self.n)
            ang3 = math.radians(55.0) + math.radians(10.0) * self._rand(self.n)
            r3 = 0.122 + 0.006 * self._rand(self.n)
            lex3 = 0.55 + 0.20 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m3, 3.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m3, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m3, r3, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m3, x_en3, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m3, x_b3, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m3, R3, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m3, ang3, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m3, lex3, pcfg[:, 8])
            pcfg[:, 10] = torch.where(m3, kw, pcfg[:, 10])

            # Regime 4: Sharp 90-deg right-angle transparent square L-duct (W=256-268 mm, H=245 mm, r_fillet=8 mm)
            m4 = reg == 4
            x_en4 = head_x - ahead
            x_b4 = x_en4 - (0.28 + 0.14 * self._rand(self.n))
            hw4 = 0.128 + 0.006 * self._rand(self.n)
            lex4 = 0.65 + 0.20 * self._rand(self.n)
            pcfg[:, 0] = torch.where(m4, 4.0, pcfg[:, 0])
            pcfg[:, 1] = torch.where(m4, y0, pcfg[:, 1])
            pcfg[:, 2] = torch.where(m4, hw4, pcfg[:, 2])
            pcfg[:, 3] = torch.where(m4, x_en4, pcfg[:, 3])
            pcfg[:, 4] = torch.where(m4, x_b4, pcfg[:, 4])
            pcfg[:, 5] = torch.where(m4, 0.0, pcfg[:, 5])
            pcfg[:, 6] = torch.where(m4, 0.5 * math.pi, pcfg[:, 6])
            pcfg[:, 8] = torch.where(m4, lex4, pcfg[:, 8])
            pcfg[:, 9] = torch.where(m4, 0.245, pcfg[:, 9])
            pcfg[:, 10] = torch.where(m4, kw, pcfg[:, 10])
            pcfg[:, 11] = torch.where(m4, 0.008, pcfg[:, 11])

            self.c.pipe_cfg.copy_(torch.where(mask[:, None], pcfg, self.c.pipe_cfg))
            self.has_entered_pipe.copy_(torch.where(mask, torch.zeros_like(mask), self.has_entered_pipe))
            self.has_exited_pipe.copy_(torch.where(mask, torch.zeros_like(mask), self.has_exited_pipe))
            return

        if self.forced_regime is not None:
            reg = torch.full((self.n,), int(self.forced_regime), device=self.device, dtype=torch.long)
        else:
            u = self._rand(self.n)
            reg = torch.zeros(self.n, device=self.device, dtype=torch.long)
            reg = torch.where(u >= self.regime_cdf[0], torch.ones_like(reg), reg)
            reg = torch.where(u >= self.regime_cdf[1], torch.full_like(reg, 2), reg)
            reg = torch.where(u >= self.regime_cdf[2], torch.full_like(reg, 3), reg)
        self.regime_id.copy_(torch.where(mask, reg, self.regime_id))

        # Initial terrain per regime: 0,1,2 start on hard; 3 starts on carpet (40%) or soft_soil (60%)
        soil_pick = torch.where(self._rand(self.n) < 0.40, torch.ones_like(reg), torch.full_like(reg, 2))
        terr = torch.where(reg == 3, soil_pick, torch.zeros_like(reg))
        self._apply_terrain_ids(mask, terr)
        self.post_pipe_soil.copy_(torch.where(mask, self._rand(self.n) < 0.50, self.post_pipe_soil))

        # Pipe wall geometry: [y_center, y_allow, x_entry, x_exit, k_wall]
        head_x = self.c.x[:, 9, 0]  # ~ -0.83 m
        lo_y, hi_y = self.task.pipe_y_allow_range_m
        y_allow = lo_y + (hi_y - lo_y) * self._rand(self.n)
        lo_a, hi_a = self.task.pipe_entry_ahead_range_m
        ahead = lo_a + (hi_a - lo_a) * self._rand(self.n)
        lo_l, hi_l = self.task.pipe_length_range_m
        plen = lo_l + (hi_l - lo_l) * self._rand(self.n)
        lo_d, hi_d = self.task.pipe_direct_exit_range_m
        dlen = lo_d + (hi_d - lo_d) * self._rand(self.n)
        lo_k, hi_k = self.task.pipe_k_wall_range
        kw = lo_k + (hi_k - lo_k) * self._rand(self.n)

        wcfg = torch.zeros((self.n, 5), device=self.device, dtype=self.dtype)
        wcfg[:, 1] = 1e6
        wcfg[:, 2] = 1e6
        wcfg[:, 3] = -1e6
        wcfg[:, 4] = 0.0

        # Regime 1: approach -> pipe entry -> pipe exit
        is_r1 = reg == 1
        wcfg[:, 0] = torch.where(is_r1, self.c.x[:, :10, 1].mean(-1), wcfg[:, 0])
        wcfg[:, 1] = torch.where(is_r1, y_allow, wcfg[:, 1])
        wcfg[:, 2] = torch.where(is_r1, head_x - ahead, wcfg[:, 2])
        wcfg[:, 3] = torch.where(is_r1, head_x - ahead - plen, wcfg[:, 3])
        wcfg[:, 4] = torch.where(is_r1, kw, wcfg[:, 4])

        # Regime 2: starts directly inside narrow pipe -> finite exit
        is_r2 = reg == 2
        wcfg[:, 0] = torch.where(is_r2, self.c.x[:, :10, 1].mean(-1), wcfg[:, 0])
        wcfg[:, 1] = torch.where(is_r2, y_allow, wcfg[:, 1])
        wcfg[:, 2] = torch.where(is_r2, torch.full_like(head_x, 10.0), wcfg[:, 2])
        wcfg[:, 3] = torch.where(is_r2, head_x - dlen, wcfg[:, 3])
        wcfg[:, 4] = torch.where(is_r2, kw, wcfg[:, 4])

        self.c.wall_cfg.copy_(torch.where(mask[:, None], wcfg, self.c.wall_cfg))
        self.has_entered_pipe.copy_(torch.where(mask, is_r2, self.has_entered_pipe))
        self.has_exited_pipe.copy_(torch.where(mask, torch.zeros_like(mask), self.has_exited_pipe))

    def _sample_commands(self, in_or_before_pipe: torch.Tensor) -> torch.Tensor:
        vx_lo = self.task.vx_min_mm_s * 1e-3
        vx_hi = self.task.vx_max_mm_s * 1e-3
        vx = vx_lo + (vx_hi - vx_lo) * self._rand(self.n)
        # Keep pipe episodes brisk enough (70..175 mm/s) that exiting into open space rewards speed recovery
        vx_pipe = 0.070 + (vx_hi - 0.070) * self._rand(self.n)
        vx = torch.where(in_or_before_pipe, vx_pipe, vx)

        is_stop = (self._rand(self.n) < self.task.stop_fraction) & (~in_or_before_pipe)
        vx = torch.where(is_stop, torch.zeros_like(vx), vx)

        wz = torch.zeros_like(vx)
        if self.stage >= 2:
            raw_wz = (2.0 * self._rand(self.n) - 1.0) * self.task.wz_max_rad_s
            straight = self._rand(self.n) < self.task.straight_fraction
            can_steer = (~is_stop) & (~straight) & (~in_or_before_pipe)
            wz = torch.where(can_steer, raw_wz, wz)

        return torch.stack((vx, wz, is_stop.to(self.dtype)), dim=-1)

    def _apply_domain_rand(self, mask: torch.Tensor) -> None:
        scale = float(self.dr_scale) if self.dr.enabled else 0.0
        eye3 = torch.eye(3, device=self.device, dtype=self.dtype)[None]
        nom_dyn = torch.tensor(
            [
                self.hw.ratchet_backlash_rad,
                self.hw.wheel_static_mu_ratio,
                self.hw.wheel_patch_radius_m,
                self.hw.tendon_capstan_mu_theta,
                self.hw.tendon_slack_m,
                self.hw.yaw_static_friction_nm,
                self.hw.servo_deadband_rad,
            ],
            device=self.device, dtype=self.dtype,
        )[None]
        if (not self.dr.enabled) or scale <= 1e-6:
            self.dr_mu_scale.copy_(torch.where(mask, 1.0, self.dr_mu_scale))
            self.dr_terr_scale.copy_(torch.where(mask[:, None], 1.0, self.dr_terr_scale))
            self.c.env_dynamics.copy_(torch.where(mask[:, None], nom_dyn, self.c.env_dynamics))
            self.c.rayleigh_beta_env.copy_(torch.where(mask, self.hw.rayleigh_beta, self.c.rayleigh_beta_env))
            self.c.ground_geom.copy_(torch.where(mask[:, None], 0.0, self.c.ground_geom))
            self.c.mass_scale.copy_(torch.where(mask, 1.0, self.c.mass_scale))
            self.c.plate_masses.copy_(torch.where(mask[:, None], self.c.plate_masses_base[None, :], self.c.plate_masses))
            self.c.yaw_kp.copy_(torch.where(mask, self.hw.yaw_kp_nm_rad, self.c.yaw_kp))
            self.c.yaw_kd.copy_(torch.where(mask, self.hw.yaw_kd_nms_rad, self.c.yaw_kd))
            self.c.yaw_torque_limit.copy_(torch.where(mask, self.hw.yaw_torque_limit_nm, self.c.yaw_torque_limit))
            self.c.servo_torque_limit.copy_(torch.where(mask, self.hw.servo_torque_limit_nm, self.c.servo_torque_limit))
            self.c.yaw_speed_limit.copy_(torch.where(mask, self.hw.yaw_speed_limit_rad_s, self.c.yaw_speed_limit))
            self.c.servo_speed_limit.copy_(torch.where(mask, self.hw.servo_speed_limit_rad_s, self.c.servo_speed_limit))
            self.servo_gain_env.copy_(torch.where(mask[:, None], self.servo_gain, self.servo_gain_env))
            self.yaw_gain_env.copy_(torch.where(mask[:, None], self.yaw_gain, self.yaw_gain_env))
            self.joint_bias.copy_(torch.where(mask[:, None], 0.0, self.joint_bias))
            self.gyro_bias.copy_(torch.where(mask[:, None], 0.0, self.gyro_bias))
            self.gravity_rot.copy_(torch.where(mask[:, None, None], eye3, self.gravity_rot))
            self.dr_params.copy_(torch.where(mask[:, None], 0.0, self.dr_params))
            return

        def _interp(lo: float, hi: float, nom: float) -> tuple[float, float]:
            return nom + scale * (lo - nom), nom + scale * (hi - nom)

        def _sample(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * self._rand(self.n)

        mus = _sample(*_interp(0.70, 1.25, 1.0))
        roll_lo, roll_hi = (0.42, 1.65) if self._use_gpipe_mode else (0.75, 1.35)
        terr_s = torch.stack(
            (
                _sample(*_interp(roll_lo, roll_hi, 1.0)),
                _sample(*_interp(0.65, 1.45, 1.0)),
                _sample(*_interp(0.75, 1.30, 1.0)),
                _sample(*_interp(0.60, 1.40, 1.0)),
                _sample(*_interp(0.85, 1.20, 1.0)),
            ),
            dim=-1,
        )

        bl_lo, bl_hi = getattr(self.dr, 'ratchet_backlash_deg_range', (4.0, 18.0))
        backlash = _sample(*_interp(math.radians(bl_lo), math.radians(bl_hi), self.hw.ratchet_backlash_rad)).clamp_min(0.0)
        static_mu_r = _sample(*_interp(*getattr(self.dr, 'wheel_static_mu_ratio_range', (1.0, 1.35)), self.hw.wheel_static_mu_ratio)).clamp_min(1.0)
        if self.hw.wheel_patch_radius_m > 0.0:
            patch_r = _sample(*_interp(*getattr(self.dr, 'wheel_patch_radius_range', (0.004, 0.013)), self.hw.wheel_patch_radius_m)).clamp_min(0.0)
        else:
            patch_r = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        cap_mt = _sample(*_interp(*getattr(self.dr, 'capstan_mu_theta_range', (0.04, 0.36)), self.hw.tendon_capstan_mu_theta)).clamp(0.0, 0.6)
        slack_jit = float(getattr(self.dr, 'tendon_slack_jitter_m', 0.0)) * scale
        slack_new = (self.hw.tendon_slack_m + (2.0 * self._rand(self.n) - 1.0) * slack_jit).clamp_min(0.0)
        yaw_sf = _sample(*_interp(*getattr(self.dr, 'yaw_static_friction_range', (0.0, 0.18)), self.hw.yaw_static_friction_nm)).clamp_min(0.0)
        db_lo, db_hi = getattr(self.dr, 'servo_deadband_deg_range', (0.0, 1.8))
        sv_db = _sample(*_interp(math.radians(db_lo), math.radians(db_hi), self.hw.servo_deadband_rad)).clamp_min(0.0)
        dyn_new = torch.stack((backlash, static_mu_r, patch_r, cap_mt, slack_new, yaw_sf, sv_db), dim=-1)

        ms = _sample(*_interp(*self.dr.mass_scale_range, 1.0))
        pm_jit = float(getattr(self.dr, 'plate_mass_jitter', 0.0)) * scale
        if pm_jit > 0.0:
            pm_mult = 1.0 + (2.0 * self._rand(self.n, self.c.P) - 1.0) * pm_jit
            pm_new = (self.c.plate_masses_base[None, :] * pm_mult).clamp_min(0.08)
        else:
            pm_new = self.c.plate_masses_base[None, :].expand(self.n, -1)
        rb_lo, rb_hi = getattr(self.dr, 'rayleigh_beta_range', (0.0002, 0.0018))
        rb_new = _sample(*_interp(rb_lo, rb_hi, self.hw.rayleigh_beta)).clamp(1e-5, 0.005)

        slope_max = math.tan(math.radians(float(getattr(self.dr, 'ground_slope_deg_max', 0.0)) * scale))
        wave_max = float(getattr(self.dr, 'ground_wave_amp_max_m', 0.0)) * scale
        wk_lo, wk_hi = getattr(self.dr, 'ground_wave_k_range', (6.0, 14.0))
        gg_new = torch.stack(
            (
                (2.0 * self._rand(self.n) - 1.0) * slope_max,
                (2.0 * self._rand(self.n) - 1.0) * slope_max,
                self._rand(self.n) * wave_max,
                _sample(wk_lo, wk_hi),
                _sample(wk_lo, wk_hi),
            ),
            dim=-1,
        )

        kps = _sample(*_interp(*self.dr.yaw_kp_scale_range, 1.0))
        kds = _sample(*_interp(*getattr(self.dr, 'yaw_kd_scale_range', (0.70, 1.30)), 1.0))
        ts = _sample(*_interp(*self.dr.torque_scale_range, 1.0))
        ss = _sample(*_interp(*self.dr.speed_scale_range, 1.0))
        tau_s = _sample(*_interp(*getattr(self.dr, 'filter_tau_scale_range', (0.70, 1.45)), 1.0)).clamp_min(0.25)
        sg_new = (1.0 - torch.exp(-self.sensor_dt / (max(self.hw.servo_filter_tau_s, 1e-4) * tau_s)))[:, None]
        yg_new = (1.0 - torch.exp(-self.sensor_dt / (max(self.hw.yaw_filter_tau_s, 1e-4) * tau_s)))[:, None]

        self.dr_mu_scale.copy_(torch.where(mask, mus, self.dr_mu_scale))
        self.dr_terr_scale.copy_(torch.where(mask[:, None], terr_s, self.dr_terr_scale))
        self.c.env_dynamics.copy_(torch.where(mask[:, None], dyn_new, self.c.env_dynamics))
        self.c.rayleigh_beta_env.copy_(torch.where(mask, rb_new, self.c.rayleigh_beta_env))
        self.c.ground_geom.copy_(torch.where(mask[:, None], gg_new, self.c.ground_geom))
        self.c.mass_scale.copy_(torch.where(mask, ms, self.c.mass_scale))
        self.c.plate_masses.copy_(torch.where(mask[:, None], pm_new, self.c.plate_masses))
        self.c.yaw_kp.copy_(torch.where(mask, self.hw.yaw_kp_nm_rad * kps, self.c.yaw_kp))
        self.c.yaw_kd.copy_(torch.where(mask, self.hw.yaw_kd_nms_rad * kds, self.c.yaw_kd))
        self.c.yaw_torque_limit.copy_(torch.where(mask, self.hw.yaw_torque_limit_nm * ts, self.c.yaw_torque_limit))
        self.c.servo_torque_limit.copy_(torch.where(mask, self.hw.servo_torque_limit_nm * ts, self.c.servo_torque_limit))
        if self.hw.yaw_speed_limit_rad_s > 0:
            self.c.yaw_speed_limit.copy_(torch.where(mask, self.hw.yaw_speed_limit_rad_s * ss, self.c.yaw_speed_limit))
        self.c.servo_speed_limit.copy_(torch.where(mask, self.hw.servo_speed_limit_rad_s * ss, self.c.servo_speed_limit))
        self.servo_gain_env.copy_(torch.where(mask[:, None], sg_new, self.servo_gain_env))
        self.yaw_gain_env.copy_(torch.where(mask[:, None], yg_new, self.yaw_gain_env))

        jb_rad = scale * math.radians(float(getattr(self.dr, 'joint_bias_deg', 0.0)))
        gb_rad_s = scale * float(getattr(self.dr, 'gyro_bias_rad_s', 0.0))
        gt_rad = scale * math.radians(float(getattr(self.dr, 'gravity_tilt_bias_deg', 0.0)))
        if jb_rad > 0.0:
            jb = (2.0 * self._rand(self.n, 14) - 1.0) * jb_rad
            self.joint_bias.copy_(torch.where(mask[:, None], jb, self.joint_bias))
        else:
            self.joint_bias.copy_(torch.where(mask[:, None], 0.0, self.joint_bias))
        if gb_rad_s > 0.0:
            gb = (2.0 * self._rand(self.n, 3) - 1.0) * gb_rad_s
            self.gyro_bias.copy_(torch.where(mask[:, None], gb, self.gyro_bias))
        else:
            self.gyro_bias.copy_(torch.where(mask[:, None], 0.0, self.gyro_bias))
        if gt_rad > 0.0:
            rx = (2.0 * self._rand(self.n) - 1.0) * gt_rad
            ry = (2.0 * self._rand(self.n) - 1.0) * gt_rad
            cx, sx = torch.cos(rx), torch.sin(rx)
            cy, sy = torch.cos(ry), torch.sin(ry)
            R_tilt = torch.zeros((self.n, 3, 3), device=self.device, dtype=self.dtype)
            R_tilt[:, 0, 0] = cy; R_tilt[:, 0, 1] = sx * sy; R_tilt[:, 0, 2] = cx * sy
            R_tilt[:, 1, 1] = cx; R_tilt[:, 1, 2] = -sx
            R_tilt[:, 2, 0] = -sy; R_tilt[:, 2, 1] = sx * cy; R_tilt[:, 2, 2] = cx * cy
            self.gravity_rot.copy_(torch.where(mask[:, None, None], R_tilt, self.gravity_rot))
        else:
            self.gravity_rot.copy_(torch.where(mask[:, None, None], eye3, self.gravity_rot))

        dr_vec = torch.stack(((mus - 1.0) / 0.25, (ms - 1.0) / 0.1, (ts - 1.0) / 0.15, (kps - 1.0) / 0.15), -1)
        self.dr_params.copy_(torch.where(mask[:, None], dr_vec, self.dr_params))

    def _apply_velocity_push(self) -> None:
        if (not self.dr.enabled) or self.dr_scale <= 1e-6 or self._push_interval_steps <= 0:
            return
        push_p = float(getattr(self.dr, 'push_prob', 0.0)) * float(self.dr_scale)
        xy_max = float(getattr(self.dr, 'push_xy_vel_max', 0.0)) * float(self.dr_scale)
        wz_max = float(getattr(self.dr, 'push_yaw_vel_max', 0.0)) * float(self.dr_scale)
        if push_p <= 0.0 or (xy_max <= 0.0 and wz_max <= 0.0):
            return
        due = (self.step_count > 0) & (self.step_count.remainder(self._push_interval_steps) == 0)
        if not bool(due.any()):
            return
        fire = due & (self._rand(self.n) < push_p)
        if not bool(fire.any()):
            return
        dvx = (2.0 * self._rand(self.n) - 1.0) * xy_max
        dvy = (2.0 * self._rand(self.n) - 1.0) * xy_max
        dwz = (2.0 * self._rand(self.n) - 1.0) * wz_max
        com = self.c.x[:, :10, :2].mean(dim=1, keepdim=True)
        rx = self.c.x[:, :, 0] - com[:, :, 0]
        ry = self.c.x[:, :, 1] - com[:, :, 1]
        fm = fire[:, None].to(self.dtype)
        self.c.v[:, :, 0].add_(fm * (dvx[:, None] - dwz[:, None] * ry))
        self.c.v[:, :, 1].add_(fm * (dvy[:, None] + dwz[:, None] * rx))
        self.c.v[:, :, 5].add_(fm * dwz[:, None])

    def _geometry(self):
        x, v = self.c.x[:, :10], self.c.v[:, :10]
        R = matrix(x[..., 3:])
        rel = R[:, 1:9:2].transpose(-1, -2) @ R[:, 2:10:2]
        q = torch.atan2(rel[..., 1, 0], rel[..., 0, 0])
        return x, v, R, q

    def _read_sensors(self, reset_mask: torch.Tensor):
        x, v, R, q = self._geometry()
        angles = torch.cat((self.c.servo, q), -1) + self.joint_bias
        gyro = torch.einsum('bij,bj->bi', R[:, 4].transpose(-1, -2), v[:, 4, 3:]) * self.frame_flip + self.gyro_bias
        proj_g = torch.einsum('bij,bj->bi', self.gravity_rot, -R[:, 4, 2, :] * self.frame_flip)
        raw = torch.cat((angles, gyro, proj_g), -1)
        if self.noise:
            raw = raw + self._randn(*raw.shape) * (self.noise_scale * self.dr.obs_noise_scale)
        packet = torch.where(reset_mask[:, None], raw, self.delayed_raw)
        self.delayed_raw.copy_(raw)
        prev = torch.where(reset_mask[:, None], raw[:, :14], self.prev_angles)
        old_dq = torch.where(reset_mask[:, None], 0.0, self.dq)
        self.dq.copy_(old_dq + 0.35 * ((packet[:, :14] - prev) / self.sensor_dt - old_dq))
        self.prev_angles.copy_(packet[:, :14])
        return x, v, R, q, packet

    def _build_frame(self, packet: torch.Tensor) -> torch.Tensor:
        vx_scale = max(0.05, self.task.vx_max_mm_s * 1e-3)
        wz_scale = max(0.10, self.task.wz_max_rad_s)
        cmd_norm = torch.stack((self.cmd[:, 0] / vx_scale, self.cmd[:, 1] / wz_scale, self.cmd[:, 2]), -1)
        ang_norm = packet[:, :14] / self.angle_scale
        dq_norm = (self.dq / 3.0).clamp(-3.0, 3.0)
        parts = [ang_norm, dq_norm]
        if self.include_q_err:
            tgt_all = torch.cat((self.servo_targets, self.yaw_targets), -1)
            q_err = ((tgt_all - packet[:, :14]) / self.angle_scale).clamp(-2.0, 2.0)
            parts.append(q_err)
        parts.extend((packet[:, 14:20], self.prev_action, cmd_norm))
        return torch.cat(parts, -1).float()

    def _pipe_membership(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(in_pipe_plates [B, 10], signed_lat_offset [B, 10])`` across both pipe_cfg & legacy wall_cfg."""
        w = self.c.wall_cfg
        px = x[:, :, 0]
        active_w = (w[:, 4:5] > 0.0) & (w[:, 1:2] < 10.0)
        in_w = active_w & (px <= w[:, 2:3] + 0.08) & (px >= w[:, 3:4])
        lat_w = x[:, :, 1] - w[:, 0:1]

        use_pc = (self.c.pipe_cfg[:, 0] > 0.5) & (self.c.pipe_cfg[:, 10] > 0.0)
        if not bool(use_pc.any()):
            return in_w, lat_w
        in_p, lat_p = eval_pipe_membership_torch(x[:, :, :2], self.c.pipe_cfg)
        return torch.where(use_pc[:, None], in_p, in_w), torch.where(use_pc[:, None], lat_p, lat_w)

    def _in_pipe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self._pipe_membership(x)[0].any(dim=-1)

    def _pipe_arc_body(self, x: torch.Tensor) -> torch.Tensor:
        s_plates = eval_pipe_arc_progress_torch(x[:, :10, :2], self.c.pipe_cfg)
        return 0.65 * s_plates.mean(-1) + 0.35 * s_plates[:, 7:10].mean(-1)

    def _build_est_target(self) -> torch.Tensor:
        vx_scale = max(0.05, self.task.vx_max_mm_s * 1e-3)
        wz_scale = max(0.10, self.task.wz_max_rad_s)
        vel_n = torch.stack((
            self.filtered_vel[:, 0] / vx_scale,
            self.filtered_vel[:, 1] / 0.04,
            self.filtered_vel[:, 2] / wz_scale,
        ), -1).clamp(-3.0, 3.0)
        wall_conf = (self.wall_ema / 4.0).clamp(0.0, 1.0)
        wall_side = (self.wall_signed_ema / 4.0).clamp(-1.0, 1.0)
        drag_denom = max(0.22 - self._hard_rolling_nom, 0.05) if self._use_gpipe_mode else 0.20
        terr_drag = ((self.c.env_terrain[:, 0] - self._hard_rolling_nom) / drag_denom).clamp(0.0, 1.0)
        slip_n = (self.last_slip_scalar / 0.08).clamp(0.0, 1.0)
        return torch.cat((vel_n, wall_conf[:, None], wall_side[:, None], terr_drag[:, None], slip_n[:, None]), -1).float()

    def _build_privileged(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        vx_scale = max(0.05, self.task.vx_max_mm_s * 1e-3)
        wz_scale = max(0.10, self.task.wz_max_rad_s)
        vscale = torch.tensor([vx_scale, 0.02, wz_scale], device=self.device, dtype=self.dtype)
        cmd_vel = torch.stack((self.cmd[:, 0], torch.zeros_like(self.cmd[:, 0]), self.cmd[:, 1]), -1)
        vel_err = (self.filtered_vel - cmd_vel) / vscale
        gap = ((x[:, 1::2, :3] - x[:, 0::2, :3]) * (-R[:, 0::2, :, 0])).sum(-1)
        slip_mag = self.c.last_slip.norm(dim=-1).mean(-1)
        yaw_trq_rms = self.c.last_yaw_torque.square().mean(-1, keepdim=True).sqrt() / max(self.hw.yaw_torque_limit_nm, 0.1)

        in_p_mat, lat_off = self._pipe_membership(x)
        in_pipe = in_p_mat.any(dim=-1)
        w = self.c.wall_cfg
        wall_active_frac = in_p_mat.to(self.dtype).mean(-1)
        use_pc = (self.c.pipe_cfg[:, 0] > 0.5) & (self.c.pipe_cfg[:, 10] > 0.0)
        y_allow_w = torch.where(w[:, 4] > 0.0, (w[:, 1] / 0.05).clamp(0.0, 2.0), torch.full_like(w[:, 1], 2.0))
        y_allow_p = ((self.c.pipe_cfg[:, 2] - 0.095) / 0.05).clamp(0.0, 2.0)
        y_allow_n = torch.where(use_pc, y_allow_p, y_allow_w)
        dy_center = torch.where(in_pipe, (lat_off.mean(-1) / 0.05).clamp(-2.0, 2.0), torch.zeros_like(w[:, 0]))

        z_g_plates, _ = self.c.ground_height_and_normal(x[:, :, :2])
        return torch.cat((
            self.last_body_vel / vscale,                     # 0..2
            self.filtered_vel / vscale,                      # 3..5
            vel_err,                                         # 6..8
            (x[:, :, 2] - z_g_plates - 0.133) / 0.05,        # 9..18
            R[:, :, 2, 2],                                   # 19..28
            (gap - 0.100) / 0.03,                            # 29..33
            self.c.tension.reshape(self.n, 20) / 15.0,       # 34..53
            self.c.support.reshape(self.n, 20) / 5.0,        # 54..73
            slip_mag / 0.05,                                 # 74..83
            (self.last_clearance[:, None] - 0.01) / 0.01,    # 84
            yaw_trq_rms,                                     # 85
            self.dr_params,                                  # 86..89
            self.long_filtered_vel / vscale,                 # 90..92
            (self.last_wall_max[:, None] / 10.0).clamp(0.0, 3.0),     # 93
            (self.wall_ema[:, None] / 10.0).clamp(0.0, 3.0),          # 94
            (self.wall_signed_ema[:, None] / 10.0).clamp(-3.0, 3.0),  # 95
            wall_active_frac[:, None],                                # 96
            y_allow_n[:, None],                                       # 97
            (self.c.env_terrain[:, 0:1] / 0.25),                      # 98 rolling
            (self.c.env_terrain[:, 3:4] / 800.0),                     # 99 rut_lat_k
            in_pipe.to(self.dtype)[:, None],                          # 100
            dy_center[:, None],                                       # 101
            self.c.env_dynamics[:, 0:1] / 0.30,                       # 102 ratchet_backlash
            self.c.env_dynamics[:, 3:4] / 0.35,                       # 103 capstan_mu_theta
            self.c.env_dynamics[:, 5:6] / 0.20,                       # 104 yaw_static_friction
            (self.c.rayleigh_beta_env[:, None] - 0.0005) / 0.001,     # 105 rayleigh_beta
            self.c.ground_geom[:, 0:1] / 0.10,                        # 106 slope_x
            self.c.ground_geom[:, 1:2] / 0.10,                        # 107 slope_y
        ), -1).float()

    def map_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mode-free 14D action mapping: ALL 10 tendons + 4 yaw servos always open."""
        a = action.to(dtype=self.dtype).clamp(-1.0, 1.0)
        if self.task.action_mode == 'delta':
            u = (self.prev_action + a * float(self.task.delta_step_scale)).clamp(-1.0, 1.0)
        else:
            u = a
        t_act = u[:, :10]
        y_act = u[:, 10:14]
        pos_span = self.hw.servo_max_rad - self.hw.servo_neutral_rad
        neg_span = self.hw.servo_neutral_rad - self.hw.servo_min_rad
        servo_des = (self.servo_neutral + t_act.clamp_min(0.0) * pos_span + t_act.clamp_max(0.0) * neg_span).clamp(
            self.hw.servo_min_rad, self.hw.servo_max_rad
        )
        yaw_des = y_act * self.hw.yaw_action_limit_rad
        return servo_des, yaw_des, u

    def reset(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for name in STATE_KEYS:
            val = getattr(self.c, name)
            val.copy_(torch.where(mask.reshape(self.n, *([1] * (val.ndim - 1))), self.initial[name], val))
        self.c.last_wall_force.copy_(torch.where(mask[:, None], 0.0, self.c.last_wall_force))
        self.c.last_strip_wall_force.copy_(torch.where(mask[:, None], 0.0, self.c.last_strip_wall_force))
        self.c.last_strip_min_z.copy_(torch.where(mask[:, None], 1.0, self.c.last_strip_min_z))
        self.c.last_slip.copy_(torch.where(mask[:, None, None, None], 0.0, self.c.last_slip))
        self.c.last_yaw_torque.copy_(torch.where(mask[:, None], 0.0, self.c.last_yaw_torque))
        self._apply_domain_rand(mask)
        self._sample_regimes_and_geometry(mask)
        # Keep pipe regimes strictly flat so walls stay aligned; apply gravel pebble bumps / grass turf undulations / slope DR on open terrains
        pipe_flat = mask & self._is_pipe_regime()
        self.c.ground_geom.copy_(torch.where(pipe_flat[:, None], 0.0, self.c.ground_geom))
        if self.task.diverse_v8:
            is_grv = mask & (~self._is_pipe_regime()) & (self.terrain_id == 3)
            if bool(is_grv.any()):
                sl_g = math.tan(math.radians(3.5))
                gg_grv = torch.stack((
                    (2.0 * self._rand(self.n) - 1.0) * sl_g,
                    (2.0 * self._rand(self.n) - 1.0) * sl_g,
                    0.006 + 0.007 * self._rand(self.n),
                    18.0 + 14.0 * self._rand(self.n),
                    18.0 + 14.0 * self._rand(self.n),
                ), dim=-1)
                self.c.ground_geom.copy_(torch.where(is_grv[:, None], gg_grv, self.c.ground_geom))
            is_grs = mask & (~self._is_pipe_regime()) & (self.terrain_id == 4)
            if bool(is_grs.any()):
                sl_r = math.tan(math.radians(4.5))
                gg_grs = torch.stack((
                    (2.0 * self._rand(self.n) - 1.0) * sl_r,
                    (2.0 * self._rand(self.n) - 1.0) * sl_r,
                    0.008 + 0.009 * self._rand(self.n),
                    7.0 + 7.0 * self._rand(self.n),
                    7.0 + 7.0 * self._rand(self.n),
                ), dim=-1)
                self.c.ground_geom.copy_(torch.where(is_grs[:, None], gg_grs, self.c.ground_geom))
        self.c.align_pose_to_ground(mask)

        self.servo_targets.copy_(torch.where(mask[:, None], self.servo_neutral, self.servo_targets))
        self.c.target.copy_(torch.where(mask[:, None], self.servo_neutral, self.c.target))
        self.c.yaw.copy_(torch.where(mask[:, None], 0.0, self.c.yaw))
        for buf in (self.yaw_targets, self.prev_action, self.prev_action_delta,
                    self.filtered_vel, self.long_filtered_vel, self.long_mean_yaw, self.last_body_vel):
            buf.copy_(torch.where(mask[:, None], 0.0, buf))
        for buf1 in (self.wall_ema, self.wall_signed_ema, self.last_wall_max, self.last_wall_mean,
                     self.last_strip_wall_mean, self.last_slip_scalar, self.filtered_v_arc, self.last_v_arc):
            buf1.copy_(torch.where(mask, 0.0, buf1))
        arc0 = self._pipe_arc_body(self.c.x[:, :10])
        self.prev_pipe_arc.copy_(torch.where(mask, arc0, self.prev_pipe_arc))
        self.last_min_strip_z.copy_(torch.where(mask, 0.05, self.last_min_strip_z))
        self.step_count.copy_(torch.where(mask, 0, self.step_count))
        self.last_clearance.copy_(torch.where(mask, 0.03, self.last_clearance))

        in_or_pre_pipe = self._is_pipe_regime()
        new_cmds = self._sample_commands(in_or_pre_pipe)
        self.cmd.copy_(torch.where(mask[:, None], new_cmds, self.cmd))

        saved = (self.delayed_raw.clone(), self.prev_angles.clone(), self.dq.clone())
        x, _, R, _, packet = self._read_sensors(mask)
        for cur, old in zip((self.delayed_raw, self.prev_angles, self.dq), saved):
            cur.copy_(torch.where(mask[:, None], cur, old))

        frame = self._build_frame(packet)
        self.history.copy_(torch.where(mask[:, None, None], frame[:, None, :], self.history))
        return self.history.clone(), self._build_privileged(x, R), self._build_est_target()

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        c = self.c
        x0 = c.x[:, :10, :3].clone()
        old_servo_tgt = self.servo_targets.clone()
        old_yaw_tgt = self.yaw_targets.clone()
        old_servo_act = c.servo.clone()
        _, _, _, old_q = self._geometry()

        servo_des, yaw_des, phys_act = self.map_action(action)
        act_delta = phys_act - self.prev_action
        act_acc = act_delta - self.prev_action_delta
        acc_sq = act_acc.square().mean(-1)

        delay_prob = self.dr.action_delay_prob * float(self.dr_scale) if self.dr.enabled else 0.0
        max_delay = max(1, min(self.steps_per_action - 1, int(getattr(self.dr, 'max_action_delay_ticks', 2))))
        if delay_prob > 0.0:
            has_delay = self._rand(self.n) < delay_prob
            delay_ticks = torch.where(
                has_delay,
                1 + torch.floor(self._rand(self.n) * max_delay).to(torch.long).clamp_max(max_delay - 1),
                torch.zeros(self.n, device=self.device, dtype=torch.long),
            )[:, None]
        else:
            delay_ticks = None

        self._apply_velocity_push()

        work = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        wall_max = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        wall_sum = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        strip_wall_sum = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        wall_signed_sum = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        min_strip_z = torch.full((self.n,), 1.0, device=self.device, dtype=self.dtype)
        packet = None
        for sub in range(self.steps_per_action):
            prev_sv = c.servo.clone()
            if delay_ticks is not None:
                d_mask = sub < delay_ticks
                s_cmd = torch.where(d_mask, self.servo_targets, servo_des)
                y_cmd = torch.where(d_mask, self.yaw_targets, yaw_des)
            else:
                s_cmd, y_cmd = servo_des, yaw_des
            self.servo_targets.add_((self.servo_gain_env * (s_cmd - self.servo_targets)).clamp(-self.servo_slew, self.servo_slew))
            self.yaw_targets.add_((self.yaw_gain_env * (y_cmd - self.yaw_targets)).clamp(-self.yaw_slew, self.yaw_slew))
            c.target.copy_(self.servo_targets)
            c.yaw.copy_(self.yaw_targets)
            c.advance_steps(self.substeps_per_sensor)
            x, v, R, q, packet = self._read_sensors(self.no_reset)

            # Aggregate wall normal forces across all 10 plates + all 600 steel-strip beam nodes
            wf_plates = c.last_wall_force  # [B, 10]
            wf_strips_seg = c.last_strip_wall_force.reshape(self.n, 5, 8).sum(-1)  # [B, 5]
            wf_strip_tot = wf_strips_seg.sum(-1)
            wf_tot = wf_plates.sum(-1) + wf_strip_tot
            wall_max = torch.maximum(wall_max, wf_tot)
            wall_sum.add_(wf_tot)
            strip_wall_sum.add_(wf_strip_tot)
            min_strip_z = torch.minimum(min_strip_z, c.last_strip_min_z.amin(-1))
            _, lat_p = self._pipe_membership(x)
            sgn_p = torch.where(lat_p >= 0.0, 1.0, -1.0)
            sgn_seg = torch.where(0.5 * (lat_p[:, 0::2] + lat_p[:, 1::2]) >= 0.0, 1.0, -1.0)
            wall_signed_sum.add_((wf_plates * sgn_p).sum(-1) + (wf_strips_seg * sgn_seg).sum(-1))

            rate = ((v[:, 2:10:2, 3:] - v[:, 1:9:2, 3:]) * R[:, 1:9:2, :, 2]).sum(-1)
            yaw_work = (c.last_yaw_torque * rate).abs().sum(-1) * self.sensor_dt
            sv_load = c.tension.reshape(self.n, 10, 2).sum(-1) * self.hw.tendon_horn_radius_m
            tendon_work = (sv_load * (c.servo - prev_sv).abs()).sum(-1)
            work += yaw_work + tendon_work

        wall_mean = wall_sum / float(self.steps_per_action)
        strip_wall_mean = strip_wall_sum / float(self.steps_per_action)
        wall_signed_mean = wall_signed_sum / float(self.steps_per_action)
        self.last_wall_max.copy_(wall_max)
        self.last_wall_mean.copy_(wall_mean)
        self.last_strip_wall_mean.copy_(strip_wall_mean)
        self.last_min_strip_z.copy_(min_strip_z)
        self.wall_ema.add_(0.45 * (wall_mean - self.wall_ema))
        self.wall_signed_ema.add_(0.45 * (wall_signed_mean - self.wall_signed_ema))

        self.step_count.add_(1)
        self.last_body_vel.copy_(UnifiedSerpWormEnv.body_velocity(x0, x, self.policy_dt))
        vel_alpha = 1.0 - math.exp(-self.policy_dt / 0.25)
        self.filtered_vel.add_(vel_alpha * (self.last_body_vel - self.filtered_vel))
        arc_now = self._pipe_arc_body(x)
        v_arc_inst = ((arc_now - self.prev_pipe_arc) / self.policy_dt).clamp(-0.30, 0.40)
        self.prev_pipe_arc.copy_(arc_now)
        self.last_v_arc.copy_(v_arc_inst)
        self.filtered_v_arc.add_(vel_alpha * (v_arc_inst - self.filtered_v_arc))
        long_alpha = 1.0 - math.exp(-self.policy_dt / 1.20)
        self.long_filtered_vel.add_(long_alpha * (self.last_body_vel - self.long_filtered_vel))
        yaw_avg_alpha = 1.0 - math.exp(-self.policy_dt / 1.50)
        self.long_mean_yaw.add_(yaw_avg_alpha * (q - self.long_mean_yaw))

        # Track dynamic pipe entry & pipe exit -> trigger post-pipe terrain / high-speed incentive!
        in_pipe_now = self._in_pipe_mask(x)
        self.has_entered_pipe.logical_or_(in_pipe_now)
        just_exited = self.has_entered_pipe & (~in_pipe_now) & (~self.has_exited_pipe) & self._is_pipe_regime()
        if bool(just_exited.any()):
            self.has_exited_pipe.logical_or_(just_exited)
            # Pipe exits enter soft soil / gravel / grass or fast open hard floor!
            to_soil = just_exited & self.post_pipe_soil
            if bool(to_soil.any()):
                if self.task.diverse_v8:
                    u_ps = self._rand(self.n)
                    ps_t = torch.where(u_ps < 0.34, torch.full_like(self.terrain_id, 2),
                                       torch.where(u_ps < 0.67, torch.full_like(self.terrain_id, 3),
                                                   torch.full_like(self.terrain_id, 4)))
                    self._apply_terrain_ids(to_soil, ps_t)
                else:
                    self._apply_terrain_ids(to_soil, torch.full_like(self.terrain_id, 2))
            fast_open = just_exited & (~self.post_pipe_soil)
            if bool(fast_open.any()):
                vx_boost = 0.135 + 0.040 * self._rand(self.n)
                self.cmd[:, 0] = torch.where(fast_open, vx_boost, self.cmd[:, 0])
                if self.stage >= 2:
                    wz_post = (2.0 * self._rand(self.n) - 1.0) * (0.75 * self.task.wz_max_rad_s)
                    want_turn = fast_open & (self._rand(self.n) < 0.40)
                    self.cmd[:, 1] = torch.where(want_turn, wz_post, self.cmd[:, 1])

        active_wheels = c.support > 1e-6
        contact_frac = active_wheels.to(self.dtype).mean((1, 2))
        slip = (c.last_slip.norm(dim=-1) * active_wheels).sum((1, 2)) / active_wheels.sum((1, 2)).clamp_min(1)
        self.last_slip_scalar.copy_(slip)

        s_span = max(self.hw.servo_max_rad - self.hw.servo_min_rad, 1e-3)
        y_span = max(self.hw.yaw_action_limit_rad, 1e-3)
        s_delta_n = (self.servo_targets - old_servo_tgt) / (0.5 * s_span)
        y_delta_n = (self.yaw_targets - old_yaw_tgt) / y_span
        rate_sq = (s_delta_n.square().sum(-1) + y_delta_n.square().sum(-1)) / 14.0

        yaw_act = (q - old_q).abs().mean(-1) / max(0.20 * self.hw.yaw_action_limit_rad, 1e-3)
        ten_act = (c.servo - old_servo_act).abs().mean(-1) / max(0.15 * s_span, 1e-3)
        upright_err = (1.0 - R[:, :, 2, 2]).clamp_min(0.0).mean(-1)
        gap = ((x[:, 1::2, :3] - x[:, 0::2, :3]) * (-R[:, 0::2, :, 0])).sum(-1)
        stop_force_max = (50000.0 * (self.hw.plate_stop_gap_m - gap).clamp_min(0.0)).amax(-1)
        edge_sat_sq = ((phys_act.abs() - 0.90).clamp_min(0.0) / 0.10).square().mean(-1)
        long_mean_yaw_sq = (self.long_mean_yaw / y_span).square().mean(-1)

        centers_f32 = (x[:, :, None, :3] + torch.einsum('bpij,psj->bpsi', R, c.mounts)).reshape(self.n, 20, 3).float()
        wheel_axes_f32 = R[..., :, 1].repeat_interleave(2, dim=1).float()
        clearance_f32, _ = certified_clearance(centers_f32, wheel_axes_f32, self.wheel_pair, pair_indices=self.wheel_pair_indices)
        clearance = clearance_f32.to(self.dtype)
        self.last_clearance.copy_(clearance)

        z_g_plates, _ = c.ground_height_and_normal(x[:, :, :2])
        finite_ok = torch.isfinite(c.x).all((1, 2)) & torch.isfinite(c.v).all((1, 2))
        guards = {
            'joint_limit': q.abs().amax(-1) > self.hw.yaw_hard_limit_rad,
            'tilt': R[:, :, 2, 2].amin(-1) < self.hw.min_upright_cos,
            'height': (x[:, :, 2] - z_g_plates).amin(-1) < self.hw.min_plate_height_m,
            'wheel_clearance': clearance < self.hw.wheel_clearance_min_m,
            'nonfinite': ~finite_ok,
        }
        failed = torch.stack(tuple(guards.values()), 0).any(0)
        self.last_guard_reasons = {k: v.clone() for k, v in guards.items()}

        free_heading = (c.pipe_cfg[:, 0] > 1.5) & (c.pipe_cfg[:, 10] > 0.0) & in_pipe_now
        vel_for_rew = self.filtered_vel
        if bool(free_heading.any()):
            vel_for_rew = self.filtered_vel.clone()
            vel_for_rew[:, 0] = torch.where(free_heading, self.filtered_v_arc, self.filtered_vel[:, 0])
        reward, terms = autonomous_phase_reward(
            vel_for_rew, self.cmd, wall_max, wall_mean,
            rate_sq, acc_sq, slip, work / self.policy_dt,
            clearance, yaw_act, ten_act, failed,
            long_velocity=self.long_filtered_vel,
            upright_err=upright_err,
            contact_frac=contact_frac,
            stop_force_max=stop_force_max,
            edge_sat_sq=edge_sat_sq,
            long_mean_yaw_sq=long_mean_yaw_sq,
            free_heading=free_heading,
            policy_dt=self.policy_dt,
            clearance_min_m=self.hw.wheel_clearance_min_m,
            clearance_soft_m=self.hw.wheel_clearance_soft_m,
            config=self.reward_config,
        )
        self.last_reward_terms = terms

        self.prev_action.copy_(phys_act)
        self.prev_action_delta.copy_(act_delta)

        terminated = failed
        truncated = (self.step_count >= self.episode_policy_steps) & ~terminated
        done = terminated | truncated

        # Mid-episode command & terrain switching (for open/soil regimes; pipe regimes stay straight while inside pipe)
        switch = (self.step_count.remainder(self.command_switch_steps) == 0) & (~done)
        if bool(switch.any()):
            in_or_pre = in_pipe_now | (self._is_pipe_regime() & (~self.has_entered_pipe))
            new_c = self._sample_commands(in_or_pre)
            self.cmd.copy_(torch.where(switch[:, None], new_c, self.cmd))
            # Natural-terrain regime randomly transitions among hard / carpet / soft_soil / gravel / grass mid-episode!
            sw_terr = switch & self._is_soil_regime()
            if bool(sw_terr.any()):
                u_t = self._rand(self.n)
                if self.task.diverse_v8:
                    next_t = torch.where(
                        u_t < 0.12, torch.zeros_like(self.terrain_id),
                        torch.where(u_t < 0.24, torch.ones_like(self.terrain_id),
                                    torch.where(u_t < 0.44, torch.full_like(self.terrain_id, 2),
                                                torch.where(u_t < 0.72, torch.full_like(self.terrain_id, 3),
                                                            torch.full_like(self.terrain_id, 4)))),
                    )
                else:
                    next_t = torch.where(u_t < 0.25, torch.zeros_like(self.terrain_id),
                                         torch.where(u_t < 0.60, torch.ones_like(self.terrain_id),
                                                     torch.full_like(self.terrain_id, 2)))
                self._apply_terrain_ids(sw_terr, next_t)

        frame = torch.nan_to_num(self._build_frame(packet), nan=0.0, posinf=0.0, neginf=0.0)
        shifted = torch.cat((self.history[:, 1:, :], frame[:, None, :]), dim=1)
        self.history.copy_(shifted)
        terminal_obs = self.history.clone()
        terminal_priv = torch.nan_to_num(self._build_privileged(x, R), nan=0.0, posinf=0.0, neginf=0.0)
        terminal_est = torch.nan_to_num(self._build_est_target(), nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.where(
            failed,
            torch.full_like(reward, -self.reward_config.failure_penalty),
            torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0),
        )

        reset_obs, reset_priv, reset_est = self.reset(done)
        self.obs = torch.where(done[:, None, None], reset_obs, terminal_obs)
        self.critic_obs = torch.where(done[:, None], reset_priv, terminal_priv)
        self.est_target = torch.where(done[:, None], reset_est, terminal_est)

        info = {
            'body_vel': self.last_body_vel.detach().clone(),
            'arc_vel': self.last_v_arc.detach().clone(),
            'filtered_vel': self.filtered_vel.detach().clone(),
            'cmd': self.cmd.detach().clone(),
            'regime_id': self.regime_id.detach().clone(),
            'terrain_id': self.terrain_id.detach().clone(),
            'pipe_type': c.pipe_cfg[:, 0].round().to(torch.long).detach().clone(),
            'in_pipe': in_pipe_now.detach().clone(),
            'post_pipe': self.has_exited_pipe.detach().clone(),
            'wall_max_n': wall_max.detach().clone(),
            'wall_mean_n': wall_mean.detach().clone(),
            'strip_wall_mean_n': strip_wall_mean.detach().clone(),
            'min_strip_z_m': min_strip_z.detach().clone(),
            'clearance': clearance.detach().clone(),
            'yaw_rms_deg': (q.square().mean(-1).sqrt() * (180.0 / math.pi)).detach().clone(),
            'max_yaw_deg': (q.abs().amax(-1) * (180.0 / math.pi)).detach().clone(),
            'ten_act': ten_act.detach().clone(),
            'slip': slip.detach().clone(),
        }
        return self.obs, self.critic_obs, self.est_target, reward.float(), terminated, truncated, terminal_obs, terminal_priv, info

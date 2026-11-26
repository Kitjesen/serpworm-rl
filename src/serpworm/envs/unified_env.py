"""Unified multi-mode GPU tensor environment (snake / worm / mixed in one policy).

Features:
- Real 40-strip pre-curved spring-steel FP64 physics (GPUCore, supports fused_cuda,
  SOFA-free parametric mesh E=6..32, frozen-tangent reuse, and CUDA Graph capture).
- Configurable HardwareConfig (DS3218 realistic defaults or calibrated specs).
- Single unified policy conditioned on 3D mode_onehot + [vx_cmd, wz_cmd, stop_flag].
- Clean physical action masking per mode + mid-episode mode & command switching.
- 8-frame temporal observation tensor [B, H, D_obs] for 1D-TCN/GRU actors + rich
  asymmetric privileged critic state [B, D_priv].
- Stage-aware curriculum (1: straight+stop, 2: +steering+mode_switch, 3: +DR).
"""
from __future__ import annotations

import math
import torch

from serpworm.config import DomainRandConfig, HardwareConfig, TaskConfig, TendonPairing
from serpworm.envs.rewards import UnifiedRewardConfig, locomotion_reward
from serpworm.physics.gpu_core import GPUCore, matrix
from serpworm.physics.wheel_envelope import active_pair_indices, certified_clearance

__all__ = ['MODE_NAMES', 'STATE_KEYS', 'UnifiedSerpWormEnv', 'mirror_rollout_batch']

MODE_NAMES = ('snake', 'worm', 'mixed')
STATE_KEYS = (
    'x', 'v', 'servo', 'target', 'yaw', 'spin', 'angle',
    'memory', 'ratchet_play', 'soil_sinkage', 'tension', 'support',
)

_PAIR_SWAP_10 = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8)
_TENSION_SWAP_20 = tuple(4 * s + k for s in range(5) for k in (2, 3, 0, 1))
_SUPPORT_SWAP_20 = tuple(2 * p + k for p in range(10) for k in (1, 0))


def mirror_rollout_batch(
    obs: torch.Tensor,
    priv: torch.Tensor,
    latent: torch.Tensor,
    tendon_pairing: TendonPairing = 'paired_5d',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact y -> -y left-right reflection of unified observations, privileged state, and latents."""
    dev = obs.device
    p10 = torch.tensor(_PAIR_SWAP_10, device=dev, dtype=torch.long)
    t20 = torch.tensor(_TENSION_SWAP_20, device=dev, dtype=torch.long)
    s20 = torch.tensor(_SUPPORT_SWAP_20, device=dev, dtype=torch.long)

    m_obs = obs.clone()
    m_obs[..., 0:10] = obs[..., p10]
    m_obs[..., 10:14] = -obs[..., 10:14]
    m_obs[..., 14:24] = obs[..., 14 + p10]
    m_obs[..., 24:28] = -obs[..., 24:28]
    # Gyro pseudo-vector (wx, wy, wz) -> (-wx, +wy, -wz)
    m_obs[..., 28] = -obs[..., 28]
    m_obs[..., 30] = -obs[..., 30]
    # Projected gravity polar vector (gx, gy, gz) -> (+gx, -gy, +gz)
    m_obs[..., 32] = -obs[..., 32]
    # Previous action & latent
    m_lat = latent.clone()
    if tendon_pairing == 'paired_5d':
        m_obs[..., 39:43] = -obs[..., 39:43]
        m_lat[..., 5:9] = -latent[..., 5:9]
    else:
        m_obs[..., 34:44] = obs[..., 34 + p10]
        m_obs[..., 44:48] = -obs[..., 44:48]
        m_lat[..., 0:10] = latent[..., p10]
        m_lat[..., 10:14] = -latent[..., 10:14]
    # Command [vx, wz, stop] -> [vx, -wz, stop]
    m_obs[..., -2] = -obs[..., -2]

    m_priv = priv.clone()
    # Body vel (0..2), filtered vel (3..5), vel_err (6..8), long_filtered_vel (90..92): (+vx, -vy, -wz)
    for base in (0, 3, 6, 90):
        if base + 2 < priv.shape[-1]:
            m_priv[..., base + 1] = -priv[..., base + 1]
            m_priv[..., base + 2] = -priv[..., base + 2]
    m_priv[..., 34:54] = priv[..., 34 + t20]
    m_priv[..., 54:74] = priv[..., 54 + s20]
    # Extended privileged DR channels (93..100): negate slope_y at index 100
    if priv.shape[-1] > 100:
        m_priv[..., 100] = -priv[..., 100]
    return m_obs, m_priv, m_lat


class UnifiedSerpWormEnv:
    """Batch-parallel GPU environment for unified 3-mode snake/worm/mixed RL."""

    def __init__(
        self,
        n: int = 2048,
        hardware: HardwareConfig | None = None,
        task: TaskConfig | None = None,
        dr: DomainRandConfig | None = None,
        reward_config: UnifiedRewardConfig | None = None,
        device: str = 'cuda',
        plate_solver: str = 'fused_cuda',
        seed: int = 7,
        noise: bool = True,
        forced_mode: int | None = None,  # 0=snake, 1=worm, 2=mixed, None=mixed batch
    ):
        self.n = int(n)
        self.hw = hardware or HardwareConfig.ds3218_default()
        self.task = task or TaskConfig()
        self.dr = dr or DomainRandConfig()
        self.reward_config = reward_config or UnifiedRewardConfig()
        self.noise = bool(noise)
        self.forced_mode = forced_mode
        self.stage = int(self.task.stage)

        self.sensor_dt = float(self.task.sensor_dt_s)
        self.policy_dt = float(self.task.policy_dt_s)
        self.physics_dt = float(self.task.physics_dt_s)
        self.elements_per_strip = int(getattr(self.task, 'elements_per_strip', 16))
        self.tangent_interval = int(getattr(self.task, 'tangent_interval', 1))
        self.steps_per_action = round(self.policy_dt / self.sensor_dt)
        self.substeps_per_sensor = round(self.sensor_dt / self.physics_dt)
        if not math.isclose(self.steps_per_action * self.sensor_dt, self.policy_dt, abs_tol=1e-9):
            raise ValueError('policy_dt_s must be an integer multiple of sensor_dt_s')
        if not math.isclose(self.substeps_per_sensor * self.physics_dt, self.sensor_dt, abs_tol=1e-9):
            raise ValueError('sensor_dt_s must be an integer multiple of physics_dt_s')
        if self.substeps_per_sensor % self.tangent_interval != 0:
            raise ValueError('substeps_per_sensor must be divisible by tangent_interval')

        self.episode_policy_steps = round(self.task.episode_seconds / self.policy_dt)
        self.command_switch_steps = max(1, round(self.task.command_switch_s / self.policy_dt))
        self.history_len = int(self.task.history_len)
        self.action_size = self.hw.action_size
        self.tendon_action_size = self.hw.tendon_action_size

        # Actuator filter gains & slew limits per 0.02 s sensor tick
        self.yaw_gain = 1.0 - math.exp(-self.sensor_dt / max(self.hw.yaw_filter_tau_s, 1e-4))
        self.yaw_slew = self.hw.yaw_slew_rad_s * self.sensor_dt
        self.servo_gain = 1.0 - math.exp(-self.sensor_dt / max(self.hw.servo_filter_tau_s, 1e-4))
        self.servo_slew = self.hw.servo_slew_rad_s * self.sensor_dt

        # Fall back cleanly to segment_lu on CPU or if nvcc is absent
        if device != 'cuda':
            solver = 'segment_lu'
        elif plate_solver == 'fused_cuda':
            from serpworm.physics.fused_cuda import is_available as fused_ok
            solver = 'fused_cuda' if fused_ok() else 'segment_lu'
        else:
            solver = plate_solver
        t_int = self.tangent_interval if solver == 'segment_lu' else 1

        # Pick CUDA Graph capture length dividing substeps_per_sensor
        if solver == 'fused_cuda':
            graph_steps = self.substeps_per_sensor
        else:
            graph_steps = t_int
            for cand in (8, 5, 4, 2):
                step_cand = cand * t_int if t_int == 1 else t_int
                if self.substeps_per_sensor % step_cand == 0 and step_cand <= 8:
                    graph_steps = step_cand
                    break

        # 1. Fast single-instance neutral settling, then broadcast to all n envs
        settler = GPUCore(
            5, 1, dt=self.physics_dt, device=device, plate_solver=solver,
            hardware=self.hw, elements_per_strip=self.elements_per_strip, tangent_interval=t_int,
        )
        if device == 'cuda':
            settler.capture(graph_steps)
        for _ in range(20):
            settler.advance_steps(self.substeps_per_sensor)
        settler.target.fill_(self.hw.servo_neutral_rad)
        for _ in range(100):
            settler.advance_steps(self.substeps_per_sensor)
        if bool(settler.failed) or not bool(torch.isfinite(settler.x).all()):
            raise RuntimeError('Neutral settling failed for HardwareConfig ' + self.hw.name)
        settled_single = {k: getattr(settler, k).clone() for k in STATE_KEYS}
        del settler

        self.c = GPUCore(
            5, self.n, dt=self.physics_dt, device=device, plate_solver=solver,
            hardware=self.hw, elements_per_strip=self.elements_per_strip, tangent_interval=t_int,
        )
        for k, val in settled_single.items():
            getattr(self.c, k).copy_(val.expand(self.n, *val.shape[1:]))
        if device == 'cuda':
            self.c.capture(graph_steps)
        self.initial = {k: getattr(self.c, k).clone() for k in STATE_KEYS}

        self.device, self.dtype = self.c.device, self.c.dtype
        self.rng = torch.Generator(device=self.device).manual_seed(int(seed))

        # Control buffers
        self.servo_neutral = torch.full((self.n, 10), self.hw.servo_neutral_rad, device=self.device, dtype=self.dtype)
        self.servo_targets = self.servo_neutral.clone()
        self.yaw_targets = torch.zeros((self.n, 4), device=self.device, dtype=self.dtype)
        self.prev_action = torch.zeros((self.n, self.action_size), device=self.device, dtype=self.dtype)
        self.prev_action_delta = torch.zeros((self.n, self.action_size), device=self.device, dtype=self.dtype)

        # Mode & command buffers: mode_id in {0:snake, 1:worm, 2:mixed}, cmd=[vx, wz, stop_flag]
        self.mode_id = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.cmd = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.step_count = torch.zeros(self.n, device=self.device, dtype=torch.long)

        # Sensing & kinematics buffers
        self.prev_angles = torch.zeros((self.n, 14), device=self.device, dtype=self.dtype)
        self.dq = torch.zeros((self.n, 14), device=self.device, dtype=self.dtype)
        self.delayed_raw = torch.zeros((self.n, 20), device=self.device, dtype=self.dtype)
        self.filtered_vel = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.long_filtered_vel = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.long_mean_yaw = torch.zeros((self.n, 4), device=self.device, dtype=self.dtype)
        self.last_body_vel = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.last_clearance = torch.full((self.n,), 0.03, device=self.device, dtype=self.dtype)
        self.last_guard_reasons: dict[str, torch.Tensor] = {}
        self.last_reward_terms: dict[str, torch.Tensor] = {}

        # Per-env DR metadata (4D normalized scalars exposed to Critic only) + sensor bias/ramp state
        self.dr_params = torch.zeros((self.n, 4), device=self.device, dtype=self.dtype)
        self.dr_scale = 1.0 if self.dr.enabled else 0.0
        self.joint_bias = torch.zeros((self.n, 14), device=self.device, dtype=self.dtype)
        self.gyro_bias = torch.zeros((self.n, 3), device=self.device, dtype=self.dtype)
        self.gravity_rot = torch.eye(3, device=self.device, dtype=self.dtype)[None].repeat(self.n, 1, 1)
        self.servo_gain_env = torch.full((self.n, 1), self.servo_gain, device=self.device, dtype=self.dtype)
        self.yaw_gain_env = torch.full((self.n, 1), self.yaw_gain, device=self.device, dtype=self.dtype)
        self._push_interval_steps = (
            max(1, round(float(getattr(self.dr, 'push_interval_s', 0.0)) / self.policy_dt))
            if float(getattr(self.dr, 'push_interval_s', 0.0)) > 0.0 else 0
        )

        # Frame layout:
        # 14 angles + 14 dq + 3 gyro + 3 proj_gravity + action_size prev_act + 3 mode_onehot + 3 cmd
        self.frame_size = 34 + self.action_size + 6
        self.privileged_size = 101  # 93 core + 8 extended DR/physics (rolling, hc_k, backlash, capstan, yaw_sf, beta, slope_x, slope_y)
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
        self.mode_cdf = torch.tensor(self.task.mode_probs, device=self.device, dtype=self.dtype).cumsum(0)
        self.mode_cdf = self.mode_cdf / self.mode_cdf[-1]

        self.obs, self.critic_obs = self.reset(torch.ones(self.n, device=self.device, dtype=torch.bool))

    def set_stage(self, stage: int, dr_scale: float | None = None) -> None:
        self.stage = int(stage)
        if self.stage >= 3:
            self.dr.enabled = True
            if dr_scale is not None:
                self.dr_scale = float(max(0.0, min(1.0, dr_scale)))
            elif self.dr_scale <= 0.0:
                self.dr_scale = 1.0

    def set_dr_scale(self, scale: float) -> None:
        """Smoothly ramp Stage-3 domain-randomization strength in [0.0, 1.0]."""
        self.dr_scale = float(max(0.0, min(1.0, scale)))

    def _rand(self, *shape):
        return torch.rand(shape, generator=self.rng, device=self.device, dtype=self.dtype)

    def _randn(self, *shape):
        return torch.randn(shape, generator=self.rng, device=self.device, dtype=self.dtype)

    def _sample_modes(self) -> torch.Tensor:
        if self.forced_mode is not None:
            return torch.full((self.n,), int(self.forced_mode), device=self.device, dtype=torch.long)
        u = self._rand(self.n)
        m = torch.zeros(self.n, device=self.device, dtype=torch.long)
        m = torch.where(u >= self.mode_cdf[0], torch.ones_like(m), m)
        m = torch.where(u >= self.mode_cdf[1], torch.full_like(m, 2), m)
        return m

    def _sample_commands(self, mode_id: torch.Tensor) -> torch.Tensor:
        vx_lo = self.task.vx_min_mm_s * 1e-3
        vx_hi = self.task.vx_max_mm_s * 1e-3
        vx = vx_lo + (vx_hi - vx_lo) * self._rand(self.n)
        is_stop = self._rand(self.n) < self.task.stop_fraction
        vx = torch.where(is_stop, torch.zeros_like(vx), vx)

        wz = torch.zeros_like(vx)
        if self.stage >= 2:
            raw_wz = (2.0 * self._rand(self.n) - 1.0) * self.task.wz_max_rad_s
            straight = self._rand(self.n) < self.task.straight_fraction
            can_steer = (~is_stop) & (~straight) & (
                (mode_id != 1) | (self.hw.tendon_pairing == 'independent_10d')
            )
            wz = torch.where(can_steer, raw_wz, wz)

        stop_f = is_stop.to(self.dtype)
        return torch.stack((vx, wz, stop_f), dim=-1)

    def _apply_domain_rand(self, mask: torch.Tensor) -> None:
        scale = float(self.dr_scale) if self.dr.enabled else 0.0
        eye3 = torch.eye(3, device=self.device, dtype=self.dtype)[None]
        nom_terr = torch.tensor(
            [
                self.hw.wheel_rolling,
                self.hw.ground_hc_k,
                self.hw.ground_soil_unload_ratio,
                self.hw.ground_rut_lat_k,
                self.hw.wheel_mu_lat_ratio,
            ],
            device=self.device, dtype=self.dtype,
        )[None]
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
            self.c.wheel_mu.copy_(torch.where(mask[:, None, None], self.hw.wheel_mu, self.c.wheel_mu))
            self.c.env_terrain.copy_(torch.where(mask[:, None], nom_terr, self.c.env_terrain))
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

        # --- Axis 1: Ground & Multi-Terrain Contact DR ---
        mu = _sample(*_interp(*self.dr.wheel_mu_range, self.hw.wheel_mu))
        roll = _sample(*_interp(*getattr(self.dr, 'wheel_rolling_range', (0.015, 0.12)), self.hw.wheel_rolling))
        hc_scale = _sample(*_interp(*getattr(self.dr, 'ground_hc_k_scale_range', (0.15, 1.50)), 1.0))
        hc_k = self.hw.ground_hc_k * hc_scale
        mu_lat_r = _sample(*_interp(*getattr(self.dr, 'wheel_mu_lat_ratio_range', (0.85, 1.35)), self.hw.wheel_mu_lat_ratio))
        unload_r = torch.full((self.n,), self.hw.ground_soil_unload_ratio, device=self.device, dtype=self.dtype)
        rut_lat_k = torch.full((self.n,), self.hw.ground_rut_lat_k, device=self.device, dtype=self.dtype)

        if self.hw.contact_model == 'arpc_soft_soil':
            unload_r = (self.hw.ground_soil_unload_ratio * _sample(*_interp(0.75, 1.30, 1.0))).clamp_min(1.05)
            rut_lat_k = (self.hw.ground_rut_lat_k * _sample(*_interp(0.60, 1.40, 1.0))).clamp_min(0.0)
        elif self.hw.contact_model == 'arpc_hard':
            soil_p = float(getattr(self.dr, 'mixed_soil_prob', 0.0)) * scale
            if soil_p > 0.0:
                is_soil = self._rand(self.n) < soil_p
                lo_u, hi_u = getattr(self.dr, 'mixed_soil_unload_range', (1.3, 3.0))
                lo_rk, hi_rk = getattr(self.dr, 'mixed_soil_rut_lat_k_range', (100.0, 550.0))
                hc_soil = self.hw.ground_hc_k * _sample(0.08, 0.28)
                roll_soil = _sample(0.08, 0.22)
                unload_r = torch.where(is_soil, _sample(lo_u, hi_u), unload_r)
                rut_lat_k = torch.where(is_soil, _sample(lo_rk, hi_rk), rut_lat_k)
                hc_k = torch.where(is_soil, hc_soil, hc_k)
                roll = torch.where(is_soil, roll_soil, roll)

        terr_new = torch.stack((roll, hc_k, unload_r, rut_lat_k, mu_lat_r), dim=-1)

        if self.hw.contact_model != 'kelvin':
            bl_lo, bl_hi = getattr(self.dr, 'ratchet_backlash_deg_range', (4.0, 18.0))
            backlash = _sample(*_interp(math.radians(bl_lo), math.radians(bl_hi), self.hw.ratchet_backlash_rad)).clamp_min(0.0)
            static_mu_r = _sample(*_interp(*getattr(self.dr, 'wheel_static_mu_ratio_range', (1.0, 1.35)), self.hw.wheel_static_mu_ratio)).clamp_min(1.0)
            if self.hw.wheel_patch_radius_m > 0.0:
                patch_r = _sample(*_interp(*getattr(self.dr, 'wheel_patch_radius_range', (0.004, 0.013)), self.hw.wheel_patch_radius_m)).clamp_min(0.0)
            else:
                patch_r = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        else:
            backlash = torch.full((self.n,), self.hw.ratchet_backlash_rad, device=self.device, dtype=self.dtype)
            static_mu_r = torch.full((self.n,), self.hw.wheel_static_mu_ratio, device=self.device, dtype=self.dtype)
            patch_r = torch.full((self.n,), self.hw.wheel_patch_radius_m, device=self.device, dtype=self.dtype)

        # --- Axis 2: Structure, Damping & Asymmetric Plate Mass DR ---
        ms = _sample(*_interp(*self.dr.mass_scale_range, 1.0))
        pm_jit = float(getattr(self.dr, 'plate_mass_jitter', 0.0)) * scale
        if pm_jit > 0.0:
            pm_mult = 1.0 + (2.0 * self._rand(self.n, self.c.P) - 1.0) * pm_jit
            pm_new = (self.c.plate_masses_base[None, :] * pm_mult).clamp_min(0.08)
        else:
            pm_new = self.c.plate_masses_base[None, :].expand(self.n, -1)
        rb_lo, rb_hi = getattr(self.dr, 'rayleigh_beta_range', (0.0002, 0.0018))
        rb_new = _sample(*_interp(rb_lo, rb_hi, self.hw.rayleigh_beta)).clamp(1e-5, 0.005)

        # Ground slope & micro-undulation DR (z_g = sx*x + sy*y + amp*sin(kx*x)*cos(ky*y))
        slope_max = math.tan(math.radians(float(getattr(self.dr, 'ground_slope_deg_max', 0.0)) * scale))
        wave_max = float(getattr(self.dr, 'ground_wave_amp_max_m', 0.0)) * scale
        wk_lo, wk_hi = getattr(self.dr, 'ground_wave_k_range', (6.0, 14.0))
        g_sx = (2.0 * self._rand(self.n) - 1.0) * slope_max
        g_sy = (2.0 * self._rand(self.n) - 1.0) * slope_max
        g_amp = self._rand(self.n) * wave_max
        g_kx = _sample(wk_lo, wk_hi)
        g_ky = _sample(wk_lo, wk_hi)
        gg_new = torch.stack((g_sx, g_sy, g_amp, g_kx, g_ky), dim=-1)

        # --- Axis 3: Tendon Transmission & Actuators DR ---
        cap_mt = _sample(*_interp(*getattr(self.dr, 'capstan_mu_theta_range', (0.04, 0.36)), self.hw.tendon_capstan_mu_theta)).clamp(0.0, 0.6)
        slack_jit = float(getattr(self.dr, 'tendon_slack_jitter_m', 0.0)) * scale
        slack_new = (self.hw.tendon_slack_m + (2.0 * self._rand(self.n) - 1.0) * slack_jit).clamp_min(0.0)
        yaw_sf = _sample(*_interp(*getattr(self.dr, 'yaw_static_friction_range', (0.0, 0.18)), self.hw.yaw_static_friction_nm)).clamp_min(0.0)
        db_lo, db_hi = getattr(self.dr, 'servo_deadband_deg_range', (0.0, 1.8))
        sv_db = _sample(*_interp(math.radians(db_lo), math.radians(db_hi), self.hw.servo_deadband_rad)).clamp_min(0.0)
        dyn_new = torch.stack((backlash, static_mu_r, patch_r, cap_mt, slack_new, yaw_sf, sv_db), dim=-1)

        kps = _sample(*_interp(*self.dr.yaw_kp_scale_range, 1.0))
        kds = _sample(*_interp(*getattr(self.dr, 'yaw_kd_scale_range', (0.70, 1.30)), 1.0))
        ts = _sample(*_interp(*self.dr.torque_scale_range, 1.0))
        ss = _sample(*_interp(*self.dr.speed_scale_range, 1.0))
        tau_s = _sample(*_interp(*getattr(self.dr, 'filter_tau_scale_range', (0.70, 1.45)), 1.0)).clamp_min(0.25)
        sg_new = (1.0 - torch.exp(-self.sensor_dt / (max(self.hw.servo_filter_tau_s, 1e-4) * tau_s)))[:, None]
        yg_new = (1.0 - torch.exp(-self.sensor_dt / (max(self.hw.yaw_filter_tau_s, 1e-4) * tau_s)))[:, None]

        self.c.wheel_mu.copy_(torch.where(mask[:, None, None], mu[:, None, None], self.c.wheel_mu))
        self.c.env_terrain.copy_(torch.where(mask[:, None], terr_new, self.c.env_terrain))
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

        # --- Axis 4: Sensors & IMU Mounting Bias DR ---
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

        dr_vec = torch.stack(((mu - self.hw.wheel_mu) / 0.3, (ms - 1.0) / 0.1, (ts - 1.0) / 0.15, (kps - 1.0) / 0.15), -1)
        self.dr_params.copy_(torch.where(mask[:, None], dr_vec, self.dr_params))

    def _apply_velocity_push(self) -> None:
        """Axis 5: Rigid-body-consistent planar + yaw velocity disturbance impulse on all nodes."""
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
        com = self.c.x[:, :10, :2].mean(dim=1, keepdim=True)  # (B, 1, 2)
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
        angles = torch.cat((self.c.servo, q), -1) + self.joint_bias  # 14D
        gyro = torch.einsum('bij,bj->bi', R[:, 4].transpose(-1, -2), v[:, 4, 3:]) * self.frame_flip + self.gyro_bias  # 3D
        proj_g = torch.einsum('bij,bj->bi', self.gravity_rot, -R[:, 4, 2, :] * self.frame_flip)  # 3D
        raw = torch.cat((angles, gyro, proj_g), -1)  # 20D
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
        mode_oh = torch.nn.functional.one_hot(self.mode_id, num_classes=3).to(self.dtype)
        vx_scale = max(0.04, self.task.vx_max_mm_s * 1e-3)
        wz_scale = max(0.10, self.task.wz_max_rad_s)
        cmd_norm = torch.stack((self.cmd[:, 0] / vx_scale, self.cmd[:, 1] / wz_scale, self.cmd[:, 2]), -1)
        ang_norm = packet[:, :14] / self.angle_scale
        dq_norm = (self.dq / 3.0).clamp(-3.0, 3.0)
        return torch.cat((ang_norm, dq_norm, packet[:, 14:20], self.prev_action, mode_oh, cmd_norm), -1).float()

    def _build_privileged(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        vx_scale = max(0.04, self.task.vx_max_mm_s * 1e-3)
        wz_scale = max(0.10, self.task.wz_max_rad_s)
        vscale = torch.tensor([vx_scale, 0.02, wz_scale], device=self.device, dtype=self.dtype)
        cmd_vel = torch.stack((self.cmd[:, 0], torch.zeros_like(self.cmd[:, 0]), self.cmd[:, 1]), -1)
        vel_err = (self.filtered_vel - cmd_vel) / vscale
        gap = ((x[:, 1::2, :3] - x[:, 0::2, :3]) * (-R[:, 0::2, :, 0])).sum(-1)  # 5D
        slip_mag = self.c.last_slip.norm(dim=-1).mean(-1)  # 10D (mean per plate)
        yaw_trq_rms = (self.c.last_yaw_torque.square().mean(-1, keepdim=True).sqrt() / max(self.hw.yaw_torque_limit_nm, 0.1))
        z_g_plates, _ = self.c.ground_height_and_normal(x[:, :, :2])
        hc_log = torch.log10((self.c.env_terrain[:, 1:2] / 1.6e5).clamp_min(1e-3))
        return torch.cat((
            self.last_body_vel / vscale,                   # 0..2
            self.filtered_vel / vscale,                    # 3..5
            vel_err,                                       # 6..8
            (x[:, :, 2] - z_g_plates - 0.133) / 0.05,      # 9..18
            R[:, :, 2, 2],                                 # 19..28
            (gap - 0.100) / 0.03,                          # 29..33
            self.c.tension.reshape(self.n, 20) / 15.0,     # 34..53
            self.c.support.reshape(self.n, 20) / 5.0,      # 54..73
            slip_mag / 0.05,                               # 74..83
            (self.last_clearance[:, None] - 0.01) / 0.01,  # 84
            yaw_trq_rms,                                   # 85
            self.dr_params,                                # 86..89
            self.long_filtered_vel / vscale,               # 90..92
            self.c.env_terrain[:, 0:1] / 0.15,             # 93 rolling
            hc_log,                                        # 94 log10(hc_k / 1.6e5)
            self.c.env_dynamics[:, 0:1] / 0.30,            # 95 ratchet_backlash
            self.c.env_dynamics[:, 3:4] / 0.35,            # 96 capstan_mu_theta
            self.c.env_dynamics[:, 5:6] / 0.20,            # 97 yaw_static_friction
            (self.c.rayleigh_beta_env[:, None] - 0.0005) / 0.001,  # 98 rayleigh_beta
            self.c.ground_geom[:, 0:1] / 0.10,             # 99 slope_x
            self.c.ground_geom[:, 1:2] / 0.10,             # 100 slope_y
        ), -1).float()

    @staticmethod
    def body_velocity(before: torch.Tensor, after: torch.Tensor, dt: float) -> torch.Tensor:
        axis0 = before[:, 9, :2] - before[:, 0, :2]
        axis1 = after[:, 9, :2] - after[:, 0, :2]
        axis0 = axis0 / axis0.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        axis1 = axis1 / axis1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        forward = axis0 + axis1
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
        vel = (after[:, :, :2].mean(1) - before[:, :, :2].mean(1)) / dt
        turn = torch.atan2(axis0[:, 0] * axis1[:, 1] - axis0[:, 1] * axis1[:, 0], (axis0 * axis1).sum(-1)) / dt
        return torch.stack(((vel * forward).sum(-1), (vel * left).sum(-1), turn), -1)

    def map_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply delta/absolute action mapping and mode physical mask to servo & yaw targets."""
        a = action.to(dtype=self.dtype).clamp(-1.0, 1.0)
        if getattr(self.task, 'action_mode', 'delta') == 'delta':
            scale = float(getattr(self.task, 'delta_step_scale', 0.28))
            u = (self.prev_action + a * scale).clamp(-1.0, 1.0)
        else:
            u = a

        t_sz = self.tendon_action_size
        t_act = u[:, :t_sz]
        y_act = u[:, t_sz:t_sz + 4]

        is_snake = (self.mode_id == 0)[:, None]
        is_worm = (self.mode_id == 1)[:, None]
        t_eff = torch.where(is_snake, torch.zeros_like(t_act), t_act)
        y_eff = torch.where(is_worm, torch.zeros_like(y_act), y_act)

        c = t_eff.repeat_interleave(2, -1) if self.hw.tendon_pairing == 'paired_5d' else t_eff
        pos_span = self.hw.servo_max_rad - self.hw.servo_neutral_rad
        neg_span = self.hw.servo_neutral_rad - self.hw.servo_min_rad
        servo_des = self.servo_neutral + c.clamp_min(0.0) * pos_span + c.clamp_max(0.0) * neg_span
        servo_des = servo_des.clamp(self.hw.servo_min_rad, self.hw.servo_max_rad)
        yaw_des = y_eff * self.hw.yaw_action_limit_rad
        physical = torch.cat((t_eff, y_eff), -1)
        return servo_des, yaw_des, physical

    def reset(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for name in STATE_KEYS:
            val = getattr(self.c, name)
            val.copy_(torch.where(mask.reshape(self.n, *([1] * (val.ndim - 1))), self.initial[name], val))
        self._apply_domain_rand(mask)
        self.c.align_pose_to_ground(mask)
        self.servo_targets.copy_(torch.where(mask[:, None], self.servo_neutral, self.servo_targets))
        self.c.target.copy_(torch.where(mask[:, None], self.servo_neutral, self.c.target))
        self.c.yaw.copy_(torch.where(mask[:, None], 0.0, self.c.yaw))
        for buf in (self.yaw_targets, self.prev_action, self.prev_action_delta,
                    self.filtered_vel, self.long_filtered_vel, self.long_mean_yaw, self.last_body_vel):
            buf.copy_(torch.where(mask[:, None], 0.0, buf))
        self.step_count.copy_(torch.where(mask, 0, self.step_count))
        self.last_clearance.copy_(torch.where(mask, 0.03, self.last_clearance))

        new_modes = self._sample_modes()
        self.mode_id.copy_(torch.where(mask, new_modes, self.mode_id))
        new_cmds = self._sample_commands(self.mode_id)
        self.cmd.copy_(torch.where(mask[:, None], new_cmds, self.cmd))

        saved = (self.delayed_raw.clone(), self.prev_angles.clone(), self.dq.clone())
        x, _, R, _, packet = self._read_sensors(mask)
        for cur, old in zip((self.delayed_raw, self.prev_angles, self.dq), saved):
            cur.copy_(torch.where(mask[:, None], cur, old))

        frame = self._build_frame(packet)
        self.history.copy_(torch.where(mask[:, None, None], frame[:, None, :], self.history))
        priv = self._build_privileged(x, R)
        return self.history.clone(), priv

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

        t_sz = self.tendon_action_size
        is_snake = (self.mode_id == 0)[:, None]
        is_worm = (self.mode_id == 1)[:, None]
        active_mask = torch.cat((
            (~is_snake).expand(self.n, t_sz),
            (~is_worm).expand(self.n, 4),
        ), -1).to(self.dtype)
        active_count = active_mask.sum(-1).clamp_min(1.0)
        acc_sq = (act_acc.square() * active_mask).sum(-1) / active_count

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
            rate = ((v[:, 2:10:2, 3:] - v[:, 1:9:2, 3:]) * R[:, 1:9:2, :, 2]).sum(-1)
            yaw_work = (c.last_yaw_torque * rate).abs().sum(-1) * self.sensor_dt
            sv_load = c.tension.reshape(self.n, 10, 2).sum(-1) * 0.0235
            tendon_work = (sv_load * (c.servo - prev_sv).abs()).sum(-1)
            work += yaw_work + tendon_work

        self.step_count.add_(1)
        self.last_body_vel.copy_(self.body_velocity(x0, x, self.policy_dt))
        vel_alpha = 1.0 - math.exp(-self.policy_dt / 0.25)
        self.filtered_vel.add_(vel_alpha * (self.last_body_vel - self.filtered_vel))
        long_alpha = 1.0 - math.exp(-self.policy_dt / 1.20)
        self.long_filtered_vel.add_(long_alpha * (self.last_body_vel - self.long_filtered_vel))
        yaw_avg_alpha = 1.0 - math.exp(-self.policy_dt / 1.50)
        self.long_mean_yaw.add_(yaw_avg_alpha * (q - self.long_mean_yaw))

        active_wheels = c.support > 1e-6
        contact_frac = active_wheels.to(self.dtype).mean((1, 2))
        slip = (c.last_slip.norm(dim=-1) * active_wheels).sum((1, 2)) / active_wheels.sum((1, 2)).clamp_min(1)

        s_span = max(self.hw.servo_max_rad - self.hw.servo_min_rad, 1e-3)
        y_span = max(self.hw.yaw_action_limit_rad, 1e-3)
        s_delta_n = (self.servo_targets - old_servo_tgt) / (0.5 * s_span)
        y_delta_n = (self.yaw_targets - old_yaw_tgt) / y_span
        t_ch_mask = (~is_snake).expand(self.n, 10).to(self.dtype)
        y_ch_mask = (~is_worm).expand(self.n, 4).to(self.dtype)
        rate_sq = (
            (s_delta_n.square() * t_ch_mask).sum(-1) + (y_delta_n.square() * y_ch_mask).sum(-1)
        ) / (t_ch_mask.sum(-1) + y_ch_mask.sum(-1)).clamp_min(1.0)

        yaw_joint_act = (q - old_q).abs() / max(0.20 * self.hw.yaw_action_limit_rad, 1e-3)  # [B, 4]
        ten_act_mag = (c.servo - old_servo_act).abs().mean(-1) / max(0.15 * s_span, 1e-3)
        upright_err = (1.0 - R[:, :, 2, 2]).clamp_min(0.0).mean(-1)
        gap = ((x[:, 1::2, :3] - x[:, 0::2, :3]) * (-R[:, 0::2, :, 0])).sum(-1)
        stop_force_max = (50000.0 * (self.hw.plate_stop_gap_m - gap).clamp_min(0.0)).amax(-1)
        edge_excess = (phys_act.abs() - 0.90).clamp_min(0.0) / 0.10
        edge_sat_sq = (edge_excess.square() * active_mask).sum(-1) / active_count
        long_mean_yaw_sq = (self.long_mean_yaw / y_span).square().mean(-1)

        # Fast FP32 upper-triangular finite-cylinder wheel separation certificate (zero D2H sync)
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

        reward, terms = locomotion_reward(
            self.filtered_vel, self.cmd, self.mode_id,
            rate_sq, acc_sq, slip, work / self.policy_dt,
            clearance, yaw_joint_act.mean(-1), ten_act_mag, failed,
            long_velocity=self.long_filtered_vel,
            yaw_joint_act=yaw_joint_act,
            upright_err=upright_err,
            contact_frac=contact_frac,
            stop_force_max=stop_force_max,
            edge_sat_sq=edge_sat_sq,
            long_mean_yaw_sq=long_mean_yaw_sq,
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

        switch = (self.step_count.remainder(self.command_switch_steps) == 0) & (~done)
        if self.stage >= 2 and self.forced_mode is None and self.task.mode_switch_prob > 0:
            do_ms = switch & (self._rand(self.n) < self.task.mode_switch_prob)
            new_m = self._sample_modes()
            self.mode_id.copy_(torch.where(do_ms, new_m, self.mode_id))
        new_c = self._sample_commands(self.mode_id)
        self.cmd.copy_(torch.where(switch[:, None], new_c, self.cmd))

        frame = self._build_frame(packet)
        shifted = torch.cat((self.history[:, 1:, :], frame[:, None, :]), dim=1)
        self.history.copy_(shifted)
        terminal_obs = self.history.clone()
        terminal_priv = self._build_privileged(x, R)

        reset_obs, reset_priv = self.reset(done)
        self.obs = torch.where(done[:, None, None], reset_obs, terminal_obs)
        self.critic_obs = torch.where(done[:, None], reset_priv, terminal_priv)

        info = {
            'body_vel': self.last_body_vel.detach().clone(),
            'filtered_vel': self.filtered_vel.detach().clone(),
            'mode_id': self.mode_id.detach().clone(),
            'cmd': self.cmd.detach().clone(),
            'clearance': clearance.detach().clone(),
            'max_yaw_deg': (q.abs().amax(-1) * (180.0 / math.pi)).detach().clone(),
            'slip': slip.detach().clone(),
        }
        return self.obs, self.critic_obs, reward.float(), terminated, truncated, terminal_obs, terminal_priv, info

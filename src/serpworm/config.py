"""Hardware, domain-randomization, curriculum, and unified RL training configs.

All actuator parameters are isolated here so switching between legacy v5
contracts, DS3218 engineering defaults, and calibrated real-hardware specs
requires changing only configuration—never physics or RL source code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Literal

__all__ = [
    'ContactModel',
    'TendonPairing',
    'TendonModel',
    'HardwareConfig',
    'DomainRandConfig',
    'TaskConfig',
    'UnifiedTrainConfig',
]

ContactModel = Literal['kelvin', 'arpc_hard', 'arpc_carpet', 'arpc_soft_soil']
TendonPairing = Literal['paired_5d', 'independent_10d']
TendonModel = Literal['legacy_wrap', 'crank_calibrated']


@dataclass
class HardwareConfig:
    """Physical actuator and geometric guard parameters.

    Default preset ('bench_calibrated') matches bench servo measurements
    (full 0-180 deg cycle ~1.40 s, loaded mean speed ~257 deg/s = 4.49 rad/s,
    peak feedback speed 428-510 deg/s = 7.5-8.9 rad/s).
    """

    name: str = 'bench_calibrated'

    # 4 inter-segment yaw servos (same 25 kg*cm = 2.45 Nm model as tendon servos)
    yaw_kp_nm_rad: float = 10.0
    yaw_kd_nms_rad: float = 0.12
    yaw_torque_limit_nm: float = 2.45
    yaw_speed_limit_rad_s: float = 8.2  # matches 428-510 deg/s measured peak feedback
    yaw_filter_tau_s: float = 0.06
    yaw_slew_rad_s: float = 4.5         # ~257 deg/s smooth trajectory cap matching bench avg
    yaw_action_limit_deg: float = 29.0  # strictly inside 30.16 deg 2 mm cylinder clearance
    yaw_hard_limit_deg: float = 34.0

    # 10 tendon-retraction servos (25 kg*cm = 2.45 Nm; bench Exp 126/127/128)
    # 0 deg -> 116.6 mm (extended), 65 deg -> ~95 mm (mid-stroke), 135 deg -> ~74 mm (full compression to 75 mm stop)
    servo_torque_limit_nm: float = 2.45
    servo_speed_limit_rad_s: float = 8.2  # matches 428-510 deg/s measured peak feedback
    servo_filter_tau_s: float = 0.06
    servo_slew_rad_s: float = 8.2         # allows full measured peak speed without artificial clip
    servo_neutral_deg: float = 65.0
    servo_min_deg: float = 0.0
    servo_max_deg: float = 135.0
    tendon_pairing: TendonPairing = 'paired_5d'

    # Tendon kinematics model:
    # - 'legacy_wrap': historical wrap-pulley formula (r=23.5 mm, g=35 mm, slack=0, k=2000 N/m)
    # - 'crank_calibrated': eccentric crank-arm calibrated 1:1 to Exp 126-128 dual-laser bench data
    tendon_model: TendonModel = 'legacy_wrap'
    tendon_horn_radius_m: float = 0.0235
    tendon_guide_dist_m: float = 0.0350
    tendon_slack_m: float = 0.0
    tendon_stiffness_n_m: float = 2000.0
    tendon_damping_ns_m: float = 0.5
    tendon_y_m: float = 0.0285
    # Optional UTM-identified spring-steel strip overrides (None -> assets/inputs/parameters.json defaults)
    strip_width_m: float | None = None
    strip_thickness_m: float | None = None
    stress_free_bow_m: float | None = None
    # Wide thin-ribbon shell correction (E/(1-nu^2) plate bending + restrained-warping torsion)
    strip_thin_shell_compat: bool = False
    # Assembly prestress fraction in [0, 1]: 0=heat-set stress-free arch, 1=flat ribbon elastically buckled into clamps
    strip_prestress_frac: float = 0.0
    # Enable implicit ground & pipe-wall contact on all 600 steel-strip interior beam nodes
    strip_node_contact: bool = False
    # Structural Rayleigh damping: f_d = -alpha*M*v - beta*K*v (legacy defaults 0.05, 0.002)
    rayleigh_alpha: float = 0.05
    rayleigh_beta: float = 0.002
    # Optional per-plate asymmetric rigid mass profile in kg (len=2*count; None -> uniform 0.28 kg)
    plate_masses_kg: tuple[float, ...] | None = None
    # Tendon guide-hole Capstan friction product mu_cable * theta_wrap (rad, dimensionless)
    tendon_capstan_mu_theta: float = 0.0
    # Gearbox Coulomb static-friction breakaway torque (Nm) & tendon servo angular deadband (rad)
    yaw_static_friction_nm: float = 0.0
    servo_deadband_rad: float = 0.0

    # Mechanical limits & safety guards
    plate_stop_gap_m: float = 0.075
    wheel_clearance_min_m: float = 0.002
    wheel_clearance_soft_m: float = 0.0042  # frees 26-28.5 deg high-thrust snake yaw while guarding 2 mm hard limit
    min_plate_height_m: float = 0.075
    min_upright_cos: float = 0.70

    # Nominal ground friction & ARPC (Anisotropic Ratchet-Patch Contact) parameters
    wheel_mu: float = 0.80
    wheel_static_mu_ratio: float = 1.0  # Stribeck static/kinetic ratio mu_s / mu_k (>= 1.0)
    contact_model: ContactModel = 'kelvin'
    implicit_contact: bool = False
    ground_kn_n_m: float = 5000.0
    ground_cn_ns_m: float = 8.0
    ground_hc_k: float = 1.6e5          # Hunt-Crossley stiffness (N / m^p)
    ground_hc_alpha: float = 3.5        # Hunt-Crossley nonlinear dissipation (s / m)
    ground_soil_exp: float = 1.5        # Hunt-Crossley / Bekker power exponent p
    ground_soil_unload_ratio: float = 1.0  # unloading/loading stiffness ratio (>1 on plastic soil)
    ground_rut_relax_m: float = 0.025   # rolling relaxation length for plastic soil rut (m)
    ground_rut_lat_k: float = 400.0     # lateral rut-wall bulldozing stiffness (N / m)
    wheel_mu_lat_ratio: float = 1.0     # mu_lat / mu_fwd anisotropy ratio
    wheel_rolling: float = 0.06         # rolling resistance coefficient
    wheel_mass_kg: float = 0.009        # single passive ratchet wheel mass for spin inertia (kg; real measured=0.0179)
    wheel_patch_radius_m: float = 0.0   # rubber contact-patch radius for vertical pivoting torque (m)
    wheel_pivot_mu: float = 0.65        # vertical pivoting friction coefficient
    ratchet_backlash_rad: float = 0.0   # one-way ratchet pawl backlash deadzone (rad)

    def with_terrain(
        self,
        terrain: Literal['kelvin', 'hard', 'carpet', 'soft_soil'],
        implicit_contact: bool | None = None,
    ) -> HardwareConfig:
        """Return a copy configured with a ground-contact terrain preset."""
        if terrain == 'kelvin':
            use_imp = False if implicit_contact is None else bool(implicit_contact)
            return replace(
                self,
                contact_model='kelvin',
                implicit_contact=use_imp,
                ground_kn_n_m=5000.0,
                ground_cn_ns_m=8.0,
                ground_hc_k=1.6e5,
                ground_hc_alpha=3.5,
                ground_soil_exp=1.5,
                ground_soil_unload_ratio=1.0,
                ground_rut_relax_m=0.025,
                ground_rut_lat_k=400.0,
                wheel_mu=0.80,
                wheel_mu_lat_ratio=1.0,
                wheel_rolling=0.06,
                wheel_patch_radius_m=0.0,
                wheel_pivot_mu=0.65,
                ratchet_backlash_rad=0.0,
            )
        use_imp = True if implicit_contact is None else bool(implicit_contact)
        if terrain == 'hard':
            return replace(
                self,
                contact_model='arpc_hard',
                implicit_contact=use_imp,
                ground_hc_k=1.6e5,
                ground_hc_alpha=3.5,
                ground_soil_exp=1.5,
                ground_soil_unload_ratio=1.0,
                ground_rut_relax_m=0.025,
                ground_rut_lat_k=0.0,
                wheel_mu=0.80,
                wheel_mu_lat_ratio=1.05,
                wheel_rolling=0.06,
                wheel_patch_radius_m=0.008,
                wheel_pivot_mu=0.65,
                ratchet_backlash_rad=0.05,
            )
        if terrain == 'carpet':
            return replace(
                self,
                contact_model='arpc_carpet',
                implicit_contact=use_imp,
                ground_hc_k=1.8e4,
                ground_hc_alpha=5.0,
                ground_soil_exp=1.4,
                ground_soil_unload_ratio=1.0,
                ground_rut_relax_m=0.025,
                ground_rut_lat_k=0.0,
                wheel_mu=0.90,
                wheel_mu_lat_ratio=1.20,
                wheel_rolling=0.14,
                wheel_patch_radius_m=0.011,
                wheel_pivot_mu=0.75,
                ratchet_backlash_rad=0.05,
            )
        if terrain == 'soft_soil':
            return replace(
                self,
                contact_model='arpc_soft_soil',
                implicit_contact=use_imp,
                ground_hc_k=1.2e4,
                ground_hc_alpha=6.0,
                ground_soil_exp=1.35,
                ground_soil_unload_ratio=3.5,
                ground_rut_relax_m=0.030,
                ground_rut_lat_k=650.0,
                wheel_mu=0.70,
                wheel_mu_lat_ratio=1.35,
                wheel_rolling=0.22,
                wheel_patch_radius_m=0.013,
                wheel_pivot_mu=0.80,
                ratchet_backlash_rad=0.05,
            )
        raise ValueError(f'Unknown terrain preset: {terrain!r}')

    @property
    def action_size(self) -> int:
        return 9 if self.tendon_pairing == 'paired_5d' else 14

    @property
    def tendon_action_size(self) -> int:
        return 5 if self.tendon_pairing == 'paired_5d' else 10

    @property
    def yaw_action_limit_rad(self) -> float:
        return math.radians(self.yaw_action_limit_deg)

    @property
    def yaw_hard_limit_rad(self) -> float:
        return math.radians(self.yaw_hard_limit_deg)

    @property
    def servo_min_rad(self) -> float:
        return math.radians(self.servo_min_deg)

    @property
    def servo_max_rad(self) -> float:
        return math.radians(self.servo_max_deg)

    @property
    def servo_neutral_rad(self) -> float:
        return math.radians(self.servo_neutral_deg)

    @classmethod
    def ds3218_default(cls) -> HardwareConfig:
        """Engineering-derated DS3218 5 V defaults (~68% stall torque, ~61% no-load speed)."""
        return cls()

    @classmethod
    def bench_exp128_calibrated(cls, tendon_pairing: TendonPairing = 'independent_10d') -> HardwareConfig:
        """25 kg*cm bench preset jointly calibrated to UTM 5-cycle compression (RMSE=0.09 N)
        and Exp 126-128 dual-laser stretch & unilateral deflection.

        Full 0-180 deg crank sweep reproduces the measured smooth S-curve (36.8 mm bilateral,
        24 mm center / 35 vs 13 mm unilateral deflection) with independent left/right tendon actuation.
        """
        return cls(
            name='bench_exp128_calibrated',
            servo_neutral_deg=90.0,
            servo_min_deg=0.0,
            servo_max_deg=180.0,
            tendon_pairing=tendon_pairing,
            tendon_model='crank_calibrated',
            tendon_horn_radius_m=0.0250,
            tendon_guide_dist_m=0.0360,
            tendon_slack_m=0.0114,
            tendon_stiffness_n_m=8000.0,
            tendon_damping_ns_m=2.5,
            tendon_y_m=0.0285,
            strip_thickness_m=0.000220,
            stress_free_bow_m=0.0300,
            plate_stop_gap_m=0.068,
        )

    @classmethod
    def v5_realistic(
        cls,
        tendon_pairing: TendonPairing = 'paired_5d',
        terrain: Literal['kelvin', 'hard', 'carpet', 'soft_soil'] = 'hard',
        ratchet_backlash_deg: float = 10.0,
        wheel_rolling: float | None = None,
    ) -> HardwareConfig:
        """Full v5 physical preset: UTM-calibrated steel (t=0.22 mm, bow=30 mm, 42.7 N),
        realistic assembly Rayleigh damping (beta=0.0005 s vs 4x overdamped 0.002 s),
        asymmetric 10-plate mass profile (front/rear yaw-servo + head-electronics bias, 2.76 kg total),
        Exp 126-128 eccentric crank-arm tendons with Capstan guide-hole friction (mu*theta=0.18),
        gearbox static-friction breakaway (0.06 Nm) & deadband (0.5 deg),
        and implicit ARPC contact with Stribeck static-friction boost (1.18x) and ratchet backlash.
        """
        base = cls.bench_exp128_calibrated(tendon_pairing=tendon_pairing).with_terrain(
            terrain, implicit_contact=(terrain != 'kelvin')
        )
        # Real 3D-printed mechanical ratchet wheels on hard floor exhibit mu_rr ~ 0.060 (vs slick 0.025 ball bearings)
        roll = (0.060 if terrain == 'hard' else base.wheel_rolling) if wheel_rolling is None else float(wheel_rolling)
        # Scale-measured 5-segment hardware plate mass profile (kg; 2,604.5 g plates + 152.0 g 40x3.8 g strips = 2,756.5 g total):
        # - Even plates (p=0,2,4,6,8 = PCB front): p0 tail end has no yaw servo (146.0 g);
        #   p2,4,6,8 carry PCB (41.0) + dual-axis yaw servo (80.8) + Al U-bracket (20.4) + mount (5.8) + 2 legs (39.2) + 2 wheels (35.8) + hardware (~30.0) = 253.0 g
        # - Odd plates (p=1,3,5,7,9 = carbon-fiber rear): CF plate (31.8) + 2x single-axis tendon servos (152.4) + 2 legs (39.2) + 2 wheels (35.8) + hardware (~30.1) = 289.3 g
        # Single full segment = 253.0 + 289.3 + 8*3.8 = 572.7 g (exact match to measured 572.7 g!)
        asym_masses = (0.1460, 0.2893, 0.2530, 0.2893, 0.2530, 0.2893, 0.2530, 0.2893, 0.2530, 0.2893)
        return replace(
            base,
            name=f'v5_realistic_{terrain}',
            rayleigh_alpha=0.05,
            rayleigh_beta=0.0005,
            plate_masses_kg=asym_masses,
            wheel_mass_kg=0.0179,
            tendon_capstan_mu_theta=0.18,
            yaw_static_friction_nm=0.06,
            servo_deadband_rad=math.radians(0.5),
            wheel_static_mu_ratio=1.18,
            wheel_rolling=roll,
            ratchet_backlash_rad=math.radians(float(ratchet_backlash_deg)),
        )

    @classmethod
    def legacy_v5(cls) -> HardwareConfig:
        """Exact v5 contract parameters for numerical parity tests."""
        return cls(
            name='legacy_v5',
            yaw_kp_nm_rad=2.0,
            yaw_kd_nms_rad=0.03,
            yaw_torque_limit_nm=0.5,
            yaw_speed_limit_rad_s=2.0,
            yaw_filter_tau_s=0.2,
            yaw_slew_rad_s=1.0,
            yaw_action_limit_deg=45.0,
            yaw_hard_limit_deg=46.0,
            servo_torque_limit_nm=0.5,
            servo_speed_limit_rad_s=2.0,
            servo_filter_tau_s=0.2,
            servo_slew_rad_s=1.0,
            servo_neutral_deg=90.0,
            servo_min_deg=0.0,
            servo_max_deg=180.0,
            tendon_pairing='paired_5d',
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: str | Path) -> HardwareConfig:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if isinstance(data.get('plate_masses_kg'), list):
            data['plate_masses_kg'] = tuple(float(v) for v in data['plate_masses_kg'])
        return cls(**data)


@dataclass
class DomainRandConfig:
    """Comprehensive 5-axis per-episode & per-step domain randomization for sim-to-real transfer.

    Axis 1 (Ground & Multi-Terrain Contact): wheel_mu, wheel_rolling, mu_lat_ratio, Stribeck static
      friction boost, Hunt-Crossley stiffness, contact patch radius, ratchet pawl backlash, mixed soil.
    Axis 2 (Structure & Asymmetric Mass): global mass scale + independent per-plate mass/COM jitter.
    Axis 3 (Tendon Transmission & Actuators): Capstan guide-hole friction hysteresis, hand-tied tendon
      slack jitter, yaw gearbox static-friction breakaway, servo deadband, torque/speed voltage sag,
      PD gains, first-order filter tau, random bus command latency.
    Axis 4 (Sensors & IMU Mounting): white noise scale, joint zero-offset bias, MEMS gyro drift bias,
      IMU mounting gravity tilt bias.
    Axis 5 (External Disturbances): periodic random planar/yaw velocity pushes (tether tug / bumps).
    """

    enabled: bool = False
    dr_ramp_rollouts: int = 20          # smooth linear ramp-in rollouts upon entering Stage 3

    # --- Axis 1: Ground & Multi-Terrain Contact ---
    wheel_mu_range: tuple[float, float] = (0.42, 1.15)
    wheel_rolling_range: tuple[float, float] = (0.025, 0.14)
    wheel_mu_lat_ratio_range: tuple[float, float] = (0.85, 1.35)
    wheel_static_mu_ratio_range: tuple[float, float] = (1.00, 1.35)
    ground_hc_k_scale_range: tuple[float, float] = (0.15, 1.50)
    wheel_patch_radius_range: tuple[float, float] = (0.004, 0.013)
    ratchet_backlash_deg_range: tuple[float, float] = (4.0, 18.0)
    mixed_soil_prob: float = 0.25       # fraction of hard-floor envs mixed with light compliant/plastic soil
    mixed_soil_unload_range: tuple[float, float] = (1.3, 3.0)
    mixed_soil_rut_lat_k_range: tuple[float, float] = (100.0, 550.0)
    ground_slope_deg_max: float = 3.5   # per-episode planar floor slope (+/- deg in X and Y)
    ground_wave_amp_max_m: float = 0.0025  # per-episode micro-undulation / floor seam amplitude (m)
    ground_wave_k_range: tuple[float, float] = (6.0, 14.0)  # spatial frequency (rad/m -> 0.45-1.05 m wavelength)

    # --- Axis 2: Structure, Damping & Asymmetric Mass ---
    mass_scale_range: tuple[float, float] = (0.85, 1.15)
    plate_mass_jitter: float = 0.15     # independent per-plate +/-15% mass/COM asymmetry jitter
    rayleigh_beta_range: tuple[float, float] = (0.0002, 0.0018)  # per-env steel+clamp structural damping (s)

    # --- Axis 3: Tendon Transmission & Actuators ---
    capstan_mu_theta_range: tuple[float, float] = (0.04, 0.36)
    tendon_slack_jitter_m: float = 0.0030   # +/-3.0 mm hand-tied braided cord slack uncertainty
    yaw_static_friction_range: tuple[float, float] = (0.00, 0.18)  # gearbox Coulomb breakaway (Nm)
    servo_deadband_deg_range: tuple[float, float] = (0.0, 1.8)     # bus servo angular deadband (deg)
    yaw_kp_scale_range: tuple[float, float] = (0.75, 1.25)
    yaw_kd_scale_range: tuple[float, float] = (0.70, 1.30)
    torque_scale_range: tuple[float, float] = (0.72, 1.15)  # covers 14-servo bus voltage sag & battery drain
    speed_scale_range: tuple[float, float] = (0.72, 1.15)
    filter_tau_scale_range: tuple[float, float] = (0.70, 1.45)  # actuator response lag randomization
    action_delay_prob: float = 0.35     # probability of extra 1..max_action_delay_ticks (20-40 ms) bus lag
    max_action_delay_ticks: int = 2

    # --- Axis 4: Sensors & IMU Mounting ---
    obs_noise_scale: float = 1.0
    joint_bias_deg: float = 2.0         # per-episode servo/joint zero-offset bias (+/- deg)
    gyro_bias_rad_s: float = 0.04       # per-episode MEMS IMU zero-rate gyro bias (+/- rad/s)
    gravity_tilt_bias_deg: float = 2.0  # per-episode IMU mounting roll/pitch tilt error (+/- deg)

    # --- Axis 5: External Velocity Push Disturbances ---
    push_interval_s: float = 4.0        # interval between random disturbance checks (0 disables)
    push_prob: float = 0.50             # probability of impulse on each check
    push_xy_vel_max: float = 0.06       # max planar velocity perturbation (m/s)
    push_yaw_vel_max: float = 0.25      # max whole-body yaw velocity perturbation (rad/s)


@dataclass
class TaskConfig:
    """Multi-mode locomotion command ranges and timing."""

    policy_dt_s: float = 0.10       # 10 Hz end-to-end control
    sensor_dt_s: float = 0.02       # 50 Hz servo/IMU tick
    physics_dt_s: float = 0.0005    # 2 kHz FP64 real spring-steel solver (verified 0.00 mm/s static drift)
    elements_per_strip: int = 16    # Corotational beam elements per strip (SOFA-free parametric topology)
    tangent_interval: int = 1       # Substeps per beam tangent factorization (1 = exact every step)
    episode_seconds: float = 20.0
    history_len: int = 16           # 1.6 s ordered temporal window (covers full snake/worm wave cycle)

    # Command ranges (calibrated to 25 kg*cm bench limits: snake ~89, worm ~126, mixed ~127 mm/s)
    vx_min_mm_s: float = 15.0
    vx_max_mm_s: float = 90.0
    wz_max_rad_s: float = 0.22      # ~12.6 deg/s strong yaw steering tracking target
    stop_fraction: float = 0.12
    straight_fraction: float = 0.35 # 65% of moving snake/mixed episodes actively train ±wz steering

    # Mode distribution in a unified batch: (snake, worm, mixed) — tilted toward harder multi-joint modes
    mode_probs: tuple[float, float, float] = (0.45, 0.15, 0.40)
    command_switch_s: float = 5.0
    mode_switch_prob: float = 0.35   # probability of switching gait mode on command switch

    # Action representation: 'delta' integrates normalized velocity (eliminates 10 Hz ratchet buzzing);
    # 'absolute' maps policy output directly to target angles.
    action_mode: Literal['delta', 'absolute'] = 'delta'
    delta_step_scale: float = 0.38  # max |du| per 0.1 s (matches real 26 kg*cm 8.2 rad/s servo envelope)

    # Curriculum stage: 1 = straight+stop, 2 = +steering+mode_switch, 3 = +domain_rand
    stage: int = 1


@dataclass
class UnifiedTrainConfig:
    """End-to-end asymmetric PPO configuration for unified 3-mode training (v5 tuned defaults)."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig.ds3218_default)
    task: TaskConfig = field(default_factory=TaskConfig)
    dr: DomainRandConfig = field(default_factory=DomainRandConfig)

    envs: int = 1024
    horizon: int = 48
    rollouts: int = 150
    epochs: int = 5
    minibatches: int = 24
    lr: float = 5e-4
    min_lr: float = 5e-5
    max_lr: float = 1.2e-3
    target_kl: float = 0.013
    gamma: float = 0.994
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0025
    max_grad_norm: float = 0.8
    initial_std: float = 0.42
    stage_std_floor: float = 0.22   # gentle exploration floor rebound on Stage 2 / Stage 3 entry
    reward_preset: Literal['legacy', 'v5_utm'] = 'v5_utm'
    seed: int = 7
    plate_solver: str = 'fused_cuda'
    compile_model: bool = False     # torch.compile TemporalActorCritic on CUDA
    amp_bf16: bool = False          # bfloat16 autocast during PPO minibatch forward
    symmetry_aug: bool = False      # 2x left-right reflection augmentation per rollout

    # Auto-curriculum progression rollouts (0 disables auto advance)
    stage2_rollout: int = 22
    stage3_rollout: int = 85
    eval_every: int = 25

"""Real-time SerpWorm policy deployment runtime, TorchScript export, and safety watchdog.

Runs trained v5 (``TemporalActorCritic``) and v6/v7 (``ProprioceptiveEstimatorActorCritic``,
both 51D pure-proprioceptive and 65D with software ``q_err = (q_target - q_meas) / scale``)
checkpoints or exported TorchScript ``.ts`` bundles at 10 Hz policy / 50 Hz servo tick with
exact bit-compatible sensor preprocessing, live EstNet telemetry, and hardware safety guards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields as dc_fields
from enum import IntFlag
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch import nn

from serpworm.config import HardwareConfig
from serpworm.deploy.hardware_io import HardwareBridge, SensorReading, TelemetryRecorder
from serpworm.rl.models import ProprioceptiveEstimatorActorCritic, TemporalActorCritic

__all__ = [
    'SafetyFault',
    'SafetyConfig',
    'SafetyStatus',
    'SafetyGuard',
    'LatencyTracker',
    'EstNetPrediction',
    'DeployMetadata',
    'PolicyStepResult',
    'ServoTickResult',
    'DeployableTemporalActor',
    'DeployableEstimatorActor',
    'SerpWormPolicyRuntime',
    'export_checkpoint_to_torchscript',
    'load_checkpoint_model_and_meta',
]


class SafetyFault(IntFlag):
    """Bitmask of active real-robot safety watchdog conditions."""

    NONE = 0
    JOINT_LIMIT = 1 << 0
    TILT = 1 << 1
    BUS_TIMEOUT = 1 << 2
    NONFINITE = 1 << 3
    MANUAL_ESTOP = 1 << 4
    COMMAND_CLAMPED = 1 << 5


@dataclass
class SafetyConfig:
    """Real-robot watchdog thresholds and fallback behavior."""

    yaw_hard_limit_deg: float = 34.0
    servo_min_deg: float = 0.0
    servo_max_deg: float = 180.0
    servo_neutral_deg: float = 90.0
    min_upright_cos: float = 0.70  # upright has proj_g_z ≈ -1.0 -> upright_cos = -proj_g_z
    bus_timeout_s: float = 0.08
    hard_timeout_s: float = 0.25
    timeout_mode: Literal['hold_last_safe', 'soft_neutral_ramp'] = 'soft_neutral_ramp'
    neutral_ramp_rate_rad_s: float = 1.5
    estop_on_joint_limit: bool = True
    estop_on_tilt: bool = True

    @classmethod
    def from_hardware(cls, hw: HardwareConfig, **overrides: Any) -> SafetyConfig:
        base = dict(
            yaw_hard_limit_deg=float(hw.yaw_hard_limit_deg),
            servo_min_deg=float(hw.servo_min_deg),
            servo_max_deg=float(hw.servo_max_deg),
            servo_neutral_deg=float(hw.servo_neutral_deg),
            min_upright_cos=float(hw.min_upright_cos),
        )
        base.update(overrides)
        return cls(**base)

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


@dataclass
class SafetyStatus:
    """Per-tick safety watchdog assessment."""

    ok: bool
    estop_active: bool
    faults: SafetyFault
    reasons: tuple[str, ...]
    upright_cos: float
    max_abs_yaw_deg: float
    sensor_age_s: float
    mode: Literal['normal', 'hold_last_safe', 'soft_neutral_ramp']


class SafetyGuard:
    """Hard joint-limit clamp, tilt E-stop, and bus-timeout hold/neutral-ramp watchdog."""

    def __init__(self, config: SafetyConfig | None = None):
        self.cfg = config or SafetyConfig()
        self.estop_latched: bool = False
        self.latched_faults: SafetyFault = SafetyFault.NONE
        self._last_valid_ts: float | None = None
        self._last_safe_servo = np.full(10, self.cfg.servo_neutral_rad, dtype=np.float64)
        self._last_safe_yaw = np.zeros(4, dtype=np.float64)

    def reset(
        self,
        servo_rad: np.ndarray | None = None,
        yaw_rad: np.ndarray | None = None,
        timestamp_s: float | None = None,
    ) -> None:
        self.estop_latched = False
        self.latched_faults = SafetyFault.NONE
        self._last_valid_ts = float(timestamp_s) if timestamp_s is not None else None
        if servo_rad is not None:
            self._last_safe_servo = np.clip(
                np.asarray(servo_rad, dtype=np.float64).reshape(10),
                self.cfg.servo_min_rad,
                self.cfg.servo_max_rad,
            )
        else:
            self._last_safe_servo = np.full(10, self.cfg.servo_neutral_rad, dtype=np.float64)
        if yaw_rad is not None:
            self._last_safe_yaw = np.clip(
                np.asarray(yaw_rad, dtype=np.float64).reshape(4),
                -self.cfg.yaw_hard_limit_rad,
                self.cfg.yaw_hard_limit_rad,
            )
        else:
            self._last_safe_yaw = np.zeros(4, dtype=np.float64)

    def trigger_manual_estop(self) -> None:
        self.estop_latched = True
        self.latched_faults |= SafetyFault.MANUAL_ESTOP

    def clear_estop(self) -> None:
        self.estop_latched = False
        self.latched_faults = SafetyFault.NONE

    def _step_toward_neutral(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        max_step = max(self.cfg.neutral_ramp_rate_rad_s, 0.05) * max(dt_s, 1e-4)
        neutral_s = np.full(10, self.cfg.servo_neutral_rad, dtype=np.float64)
        neutral_y = np.zeros(4, dtype=np.float64)
        s_next = self._last_safe_servo + np.clip(neutral_s - self._last_safe_servo, -max_step, max_step)
        y_next = self._last_safe_yaw + np.clip(neutral_y - self._last_safe_yaw, -max_step, max_step)
        s_next = np.clip(s_next, self.cfg.servo_min_rad, self.cfg.servo_max_rad)
        y_next = np.clip(y_next, -self.cfg.yaw_hard_limit_rad, self.cfg.yaw_hard_limit_rad)
        self._last_safe_servo = s_next
        self._last_safe_yaw = y_next
        return s_next.copy(), y_next.copy()

    def filter_targets(
        self,
        servo_target: np.ndarray,
        yaw_target: np.ndarray,
        sensor: SensorReading,
        *,
        dt_s: float = 0.02,
        now_s: float | None = None,
        proj_g_flu: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, SafetyStatus]:
        """Validate sensor health & clamp or override actuator targets safely."""
        s_req = np.asarray(servo_target, dtype=np.float64).reshape(10)
        y_req = np.asarray(yaw_target, dtype=np.float64).reshape(4)
        faults = self.latched_faults
        reasons: list[str] = []

        # 1. Non-finite protection on both sensors and commands
        finite_sensor = (
            np.all(np.isfinite(sensor.servo_rad))
            and np.all(np.isfinite(sensor.yaw_rad))
            and np.all(np.isfinite(sensor.gyro_rad_s))
            and np.all(np.isfinite(sensor.proj_g))
        )
        finite_cmd = np.all(np.isfinite(s_req)) and np.all(np.isfinite(y_req))
        if not (finite_sensor and finite_cmd):
            faults |= SafetyFault.NONFINITE
            self.latched_faults |= SafetyFault.NONFINITE
            self.estop_latched = True
            reasons.append('nonfinite_values')

        # 2. Bus timeout check
        cur_t = float(sensor.timestamp_s if now_s is None else now_s)
        if self._last_valid_ts is None and sensor.bus_ok:
            self._last_valid_ts = float(sensor.timestamp_s)
        if sensor.bus_ok and finite_sensor:
            sensor_age_s = max(0.0, cur_t - float(sensor.timestamp_s))
            if sensor_age_s <= self.cfg.bus_timeout_s:
                self._last_valid_ts = float(sensor.timestamp_s)
        else:
            ref_t = self._last_valid_ts if self._last_valid_ts is not None else cur_t
            sensor_age_s = max(self.cfg.bus_timeout_s + 1e-6, cur_t - ref_t)

        bus_timeout = (not sensor.bus_ok) or (sensor_age_s > self.cfg.bus_timeout_s)
        if bus_timeout:
            faults |= SafetyFault.BUS_TIMEOUT
            reasons.append(f'bus_timeout({sensor_age_s * 1e3:.1f}ms)')

        # 3. Tilt E-stop check (upright has proj_g_z ≈ -1.0 -> upright_cos = -proj_g_z)
        g_vec = sensor.proj_g if proj_g_flu is None else np.asarray(proj_g_flu, dtype=np.float64).reshape(3)
        upright_cos = float(-g_vec[2]) if np.isfinite(g_vec[2]) else -1.0
        if finite_sensor and self.cfg.estop_on_tilt and upright_cos < self.cfg.min_upright_cos:
            faults |= SafetyFault.TILT
            self.latched_faults |= SafetyFault.TILT
            self.estop_latched = True
            reasons.append(f'tilt_estop(cos={upright_cos:.3f}<{self.cfg.min_upright_cos:.2f})')

        # 4. Measured hard yaw-joint limit check
        max_abs_yaw_rad = float(np.max(np.abs(sensor.yaw_rad))) if finite_sensor else float('inf')
        max_abs_yaw_deg = math.degrees(max_abs_yaw_rad) if math.isfinite(max_abs_yaw_rad) else float('inf')
        if finite_sensor and self.cfg.estop_on_joint_limit and max_abs_yaw_rad > self.cfg.yaw_hard_limit_rad:
            faults |= SafetyFault.JOINT_LIMIT
            self.latched_faults |= SafetyFault.JOINT_LIMIT
            self.estop_latched = True
            reasons.append(f'yaw_hard_limit({max_abs_yaw_deg:.1f}deg>{self.cfg.yaw_hard_limit_deg:.1f}deg)')

        if SafetyFault.MANUAL_ESTOP in self.latched_faults:
            reasons.append('manual_estop')

        # 5. Select output target mode: E-stop -> soft neutral ramp; bus timeout -> hold or soft ramp; normal -> hard clamp
        if self.estop_latched:
            s_out, y_out = self._step_toward_neutral(dt_s)
            mode: Literal['normal', 'hold_last_safe', 'soft_neutral_ramp'] = 'soft_neutral_ramp'
        elif bus_timeout:
            if self.cfg.timeout_mode == 'hold_last_safe' and sensor_age_s <= self.cfg.hard_timeout_s:
                s_out = self._last_safe_servo.copy()
                y_out = self._last_safe_yaw.copy()
                mode = 'hold_last_safe'
            else:
                s_out, y_out = self._step_toward_neutral(dt_s)
                mode = 'soft_neutral_ramp'
        else:
            s_out = np.clip(s_req, self.cfg.servo_min_rad, self.cfg.servo_max_rad)
            y_out = np.clip(y_req, -self.cfg.yaw_hard_limit_rad, self.cfg.yaw_hard_limit_rad)
            if np.max(np.abs(s_out - s_req)) > 1e-9 or np.max(np.abs(y_out - y_req)) > 1e-9:
                faults |= SafetyFault.COMMAND_CLAMPED
            self._last_safe_servo = s_out.copy()
            self._last_safe_yaw = y_out.copy()
            mode = 'normal'

        critical_faults = faults & ~SafetyFault.COMMAND_CLAMPED
        status = SafetyStatus(
            ok=(critical_faults == SafetyFault.NONE) and (not self.estop_latched),
            estop_active=bool(self.estop_latched),
            faults=faults,
            reasons=tuple(reasons),
            upright_cos=upright_cos,
            max_abs_yaw_deg=max_abs_yaw_deg,
            sensor_age_s=sensor_age_s,
            mode=mode,
        )
        return s_out, y_out, status


class LatencyTracker:
    """Lightweight rolling-window per-tick execution latency telemetry (ms)."""

    def __init__(self, budget_ms: float, max_samples: int = 10000):
        self.budget_ms = float(budget_ms)
        self._max_samples = max(1, int(max_samples))
        self._samples: list[float] = []
        self._write_idx: int = 0
        self._total_count: int = 0
        self._last_ms: float = 0.0
        self.overruns: int = 0

    def reset(self) -> None:
        self._samples.clear()
        self._write_idx = 0
        self._total_count = 0
        self._last_ms = 0.0
        self.overruns = 0

    def record(self, latency_ms: float) -> float:
        val = float(latency_ms)
        if len(self._samples) < self._max_samples:
            self._samples.append(val)
        else:
            self._samples[self._write_idx] = val
        self._write_idx = (self._write_idx + 1) % self._max_samples
        self._total_count += 1
        self._last_ms = val
        if val > self.budget_ms:
            self.overruns += 1
        return val

    def summary(self) -> dict[str, float]:
        if self._total_count == 0:
            return {
                'count': 0.0,
                'last_ms': 0.0,
                'mean_ms': 0.0,
                'p50_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0,
                'max_ms': 0.0,
                'budget_ms': self.budget_ms,
                'overruns': 0.0,
            }
        arr = np.asarray(self._samples, dtype=np.float64)
        return {
            'count': float(self._total_count),
            'last_ms': float(self._last_ms),
            'mean_ms': float(arr.mean()),
            'p50_ms': float(np.percentile(arr, 50)),
            'p95_ms': float(np.percentile(arr, 95)),
            'p99_ms': float(np.percentile(arr, 99)),
            'max_ms': float(arr.max()),
            'budget_ms': self.budget_ms,
            'overruns': float(self.overruns),
        }


@dataclass(frozen=True)
class EstNetPrediction:
    """Live pure-proprioceptive EstNet prediction (v6/v7 checkpoints).

    Physical fields match ``AutonomousPhaseEnv._build_est_target`` denormalization:
      - ``vx`` (m/s), ``vy`` (m/s), ``wz`` (rad/s)
      - ``wall_conf`` in ``[0, 1]`` (pipe confinement intensity)
      - ``wall_side`` in ``[-1, 1]`` (signed lateral wall bias)
      - ``terrain_drag`` in ``[0, 1]`` (soft terrain rolling drag index)
      - ``slip`` in ``[0, 1]`` (normalized wheel slip intensity; ``slip_mm_s = slip * 80``)
    """

    vx: float
    vy: float
    wz: float
    wall_conf: float
    wall_side: float
    terrain_drag: float
    slip: float
    raw: tuple[float, ...]

    @classmethod
    def from_raw(cls, raw_vec: Sequence[float] | np.ndarray, vx_scale: float, wz_scale: float) -> EstNetPrediction:
        arr = np.asarray(raw_vec, dtype=np.float64).reshape(-1)
        if arr.size < 7:
            raise ValueError(f'Expected 7D EstNet prediction, got shape {arr.shape}')
        return cls(
            vx=float(arr[0] * vx_scale),
            vy=float(arr[1] * 0.04),
            wz=float(arr[2] * wz_scale),
            wall_conf=float(arr[3]),
            wall_side=float(arr[4]),
            terrain_drag=float(arr[5]),
            slip=float(arr[6]),
            raw=tuple(float(x) for x in arr[:7]),
        )

    @property
    def vx_mm_s(self) -> float:
        return self.vx * 1000.0

    @property
    def vy_mm_s(self) -> float:
        return self.vy * 1000.0

    @property
    def slip_mm_s(self) -> float:
        return self.slip * 80.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'vx_m_s': self.vx,
            'vx_mm_s': self.vx_mm_s,
            'vy_m_s': self.vy,
            'vy_mm_s': self.vy_mm_s,
            'wz_rad_s': self.wz,
            'wall_conf': self.wall_conf,
            'wall_side': self.wall_side,
            'terrain_drag': self.terrain_drag,
            'slip': self.slip,
            'slip_mm_s': self.slip_mm_s,
            'raw': list(self.raw),
        }


@dataclass
class DeployMetadata:
    """Self-contained deployment contract serialized alongside exported TorchScript ``.ts``."""

    schema: str = 'serpworm_deploy_v1'
    model_type: Literal['autonomous_est', 'unified_temporal'] = 'autonomous_est'
    source_schema: str = 'serpworm_autonomous_v6'
    frame_size: int = 65
    history_len: int = 16
    action_size: int = 14
    est_size: int = 7
    include_q_err: bool = True
    tendon_pairing: Literal['paired_5d', 'independent_10d'] = 'independent_10d'
    action_mode: Literal['delta', 'absolute'] = 'delta'
    delta_step_scale: float = 0.50
    mean_bound: float = 4.0
    policy_dt_s: float = 0.10
    sensor_dt_s: float = 0.02
    vx_max_mm_s: float = 175.0
    wz_max_rad_s: float = 0.26
    vx_scale_floor: float = 0.05  # 0.05 in AutonomousPhaseEnv, 0.04 in UnifiedSerpWormEnv
    dq_ema_alpha: float = 0.35
    frame_flip: tuple[float, float, float] = (-1.0, -1.0, 1.0)
    hardware: dict[str, Any] = field(
        default_factory=lambda: HardwareConfig.v5_realistic(tendon_pairing='independent_10d').to_dict()
    )
    torchscript_file: str = 'policy_actor.ts'
    rollout: int = -1
    stage: int = 3
    source_checkpoint: str = ''

    def __post_init__(self) -> None:
        self.frame_flip = tuple(float(v) for v in self.frame_flip)  # type: ignore[assignment]

    def build_hardware_config(self) -> HardwareConfig:
        valid_fields = {f.name for f in dc_fields(HardwareConfig)}
        filtered = {k: v for k, v in self.hardware.items() if k in valid_fields}
        if isinstance(filtered.get('plate_masses_kg'), list):
            filtered['plate_masses_kg'] = tuple(float(v) for v in filtered['plate_masses_kg'])
        return HardwareConfig(**filtered)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['frame_flip'] = list(self.frame_flip)
        return d

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
        return out

    @classmethod
    def load(cls, path: str | Path) -> DeployMetadata:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        valid = {f.name for f in dc_fields(cls)}
        data = {k: v for k, v in raw.items() if k in valid}
        if 'frame_flip' in data:
            data['frame_flip'] = tuple(float(v) for v in data['frame_flip'])
        return cls(**data)


@dataclass
class PolicyStepResult:
    """Output of a single 10 Hz policy inference step."""

    raw_action: np.ndarray
    physical_action: np.ndarray
    servo_des_rad: np.ndarray
    yaw_des_rad: np.ndarray
    est_pred: EstNetPrediction | None
    latency_ms: float


@dataclass
class ServoTickResult:
    """Output of a single 50 Hz actuator filter + safety tick."""

    servo_cmd_rad: np.ndarray
    yaw_cmd_rad: np.ndarray
    safety: SafetyStatus
    latency_ms: float


# ---------------------------------------------------------------------------
# Inference-Only TorchScript-Exportable Actor Wrappers
# ---------------------------------------------------------------------------

class DeployableTemporalActor(nn.Module):
    """Critic-free deployable wrapper for ``TemporalActorCritic`` (v3/v5)."""

    def __init__(self, model: TemporalActorCritic):
        super().__init__()
        self.obs_clip = float(model.obs_norm.clip)
        mean_f32 = model.obs_norm.mean.detach().to(dtype=torch.float32)
        inv_std_f32 = model.obs_norm.var.detach().clamp_min(1e-8).rsqrt().to(dtype=torch.float32)
        self.register_buffer('obs_mean', mean_f32)
        self.register_buffer('obs_inv_std', inv_std_f32)
        self.actor_tcn = model.actor_tcn
        self.actor_mlp = model.actor_mlp

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        seq_n = ((obs_seq - self.obs_mean) * self.obs_inv_std).clamp(-self.obs_clip, self.obs_clip)
        feat = self.actor_tcn(seq_n.transpose(1, 2))
        enc = torch.cat((feat.flatten(1), seq_n[:, -1, :]), dim=-1)
        return self.actor_mlp(enc).tanh()


class DeployableEstimatorActor(nn.Module):
    """Critic-free deployable wrapper for ``ProprioceptiveEstimatorActorCritic`` (v6/v7)."""

    def __init__(self, model: ProprioceptiveEstimatorActorCritic):
        super().__init__()
        self.obs_clip = float(model.obs_norm.clip)
        self.est_size = int(model.est_size)
        self.mean_bound = float(model.mean_bound)
        mean_f32 = model.obs_norm.mean.detach().to(dtype=torch.float32)
        inv_std_f32 = model.obs_norm.var.detach().clamp_min(1e-8).rsqrt().to(dtype=torch.float32)
        self.register_buffer('obs_mean', mean_f32)
        self.register_buffer('obs_inv_std', inv_std_f32)
        self.est_tcn = model.est_tcn
        self.est_mlp = model.est_mlp
        self.actor_tcn = model.actor_tcn
        self.actor_mlp = model.actor_mlp

    def forward(self, obs_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_n = ((obs_seq - self.obs_mean) * self.obs_inv_std).clamp(-self.obs_clip, self.obs_clip)
        seq_t = seq_n.transpose(1, 2)
        latest = seq_n[:, -1, :]

        est_enc = torch.cat((self.est_tcn(seq_t).flatten(1), latest), dim=-1)
        raw_est = self.est_mlp(est_enc)
        if self.est_size == 7:
            vel = raw_est[:, :3]
            w_conf = torch.sigmoid(raw_est[:, 3:4])
            w_side = torch.tanh(raw_est[:, 4:5])
            drag_slip = torch.sigmoid(raw_est[:, 5:7])
            est_pred = torch.cat((vel, w_conf, w_side, drag_slip), dim=-1)
        else:
            est_pred = raw_est

        act_enc = torch.cat((self.actor_tcn(seq_t).flatten(1), latest, est_pred), dim=-1)
        raw_mean = self.actor_mlp(act_enc)
        mean = self.mean_bound * torch.tanh(raw_mean / self.mean_bound)
        action = mean.float().tanh()
        return action, est_pred


def load_checkpoint_model_and_meta(
    ckpt_path: str | Path,
    device: str | torch.device = 'cpu',
) -> tuple[TemporalActorCritic | ProprioceptiveEstimatorActorCritic, DeployMetadata]:
    """Load any v3/v5/v6/v7 ``.pt`` checkpoint and construct its ``DeployMetadata``."""
    path = Path(ckpt_path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt['model']
    source_schema = str(ckpt.get('schema', ''))
    is_autonomous = ('est_size' in ckpt) or ('est_tcn.0.weight' in state_dict) or ('autonomous' in source_schema)

    hw_dict = dict(ckpt['hardware'])
    task_dict = dict(ckpt['task'])
    frame_size = int(ckpt['frame_size'])
    history_len = int(ckpt['history_len'])
    privileged_size = int(ckpt['privileged_size'])
    action_size = int(ckpt['action_size'])

    if is_autonomous:
        est_size = int(ckpt.get('est_size', 7))
        mean_bound = float(ckpt.get('mean_bound', 4.0))
        include_q_err = bool(ckpt.get('include_q_err', task_dict.get('include_q_err', frame_size >= 65)))
        model: TemporalActorCritic | ProprioceptiveEstimatorActorCritic = ProprioceptiveEstimatorActorCritic(
            frame_size=frame_size,
            history_len=history_len,
            privileged_size=privileged_size,
            est_size=est_size,
            action_size=action_size,
            mean_bound=mean_bound,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        meta = DeployMetadata(
            model_type='autonomous_est',
            source_schema=source_schema or 'serpworm_autonomous_v6',
            frame_size=frame_size,
            history_len=history_len,
            action_size=action_size,
            est_size=est_size,
            include_q_err=include_q_err,
            tendon_pairing='independent_10d',
            action_mode=str(task_dict.get('action_mode', 'delta')),  # type: ignore[arg-type]
            delta_step_scale=float(task_dict.get('delta_step_scale', 0.50)),
            mean_bound=mean_bound,
            policy_dt_s=float(task_dict.get('policy_dt_s', 0.10)),
            sensor_dt_s=float(task_dict.get('sensor_dt_s', 0.02)),
            vx_max_mm_s=float(task_dict.get('vx_max_mm_s', 175.0)),
            wz_max_rad_s=float(task_dict.get('wz_max_rad_s', 0.26)),
            vx_scale_floor=0.05,
            dq_ema_alpha=0.35,
            frame_flip=(-1.0, -1.0, 1.0),
            hardware=hw_dict,
            rollout=int(ckpt.get('rollout', -1)),
            stage=int(ckpt.get('stage', 3)),
            source_checkpoint=str(path),
        )
    else:
        model = TemporalActorCritic(
            frame_size=frame_size,
            history_len=history_len,
            privileged_size=privileged_size,
            action_size=action_size,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        pairing = str(hw_dict.get('tendon_pairing', 'paired_5d' if action_size == 9 else 'independent_10d'))
        meta = DeployMetadata(
            model_type='unified_temporal',
            source_schema=source_schema or 'serpworm_unified_ppo_v1',
            frame_size=frame_size,
            history_len=history_len,
            action_size=action_size,
            est_size=0,
            include_q_err=False,
            tendon_pairing=pairing,  # type: ignore[arg-type]
            action_mode=str(task_dict.get('action_mode', 'delta')),  # type: ignore[arg-type]
            delta_step_scale=float(task_dict.get('delta_step_scale', 0.38)),
            mean_bound=0.0,
            policy_dt_s=float(task_dict.get('policy_dt_s', 0.10)),
            sensor_dt_s=float(task_dict.get('sensor_dt_s', 0.02)),
            vx_max_mm_s=float(task_dict.get('vx_max_mm_s', 90.0)),
            wz_max_rad_s=float(task_dict.get('wz_max_rad_s', 0.22)),
            vx_scale_floor=0.04,
            dq_ema_alpha=0.35,
            frame_flip=(-1.0, -1.0, 1.0),
            hardware=hw_dict,
            rollout=int(ckpt.get('rollout', -1)),
            stage=int(ckpt.get('stage', 3)),
            source_checkpoint=str(path),
        )
    return model, meta


def export_checkpoint_to_torchscript(
    ckpt_path: str | Path,
    out_dir: str | Path,
    *,
    device: str = 'cpu',
    ts_filename: str = 'policy_actor.ts',
    meta_filename: str = 'deploy_meta.json',
    verify_samples: int = 16,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Export a ``.pt`` checkpoint to frozen TorchScript ``.ts`` + ``deploy_meta.json`` with parity verification."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, meta = load_checkpoint_model_and_meta(ckpt_path, device=device)
    model.eval()

    if isinstance(model, ProprioceptiveEstimatorActorCritic):
        deploy_mod: nn.Module = DeployableEstimatorActor(model).to(device).eval()
    else:
        deploy_mod = DeployableTemporalActor(model).to(device).eval()

    example = torch.zeros((1, meta.history_len, meta.frame_size), device=device, dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(deploy_mod, example, strict=False)
        frozen = torch.jit.freeze(traced)

    ts_path = out / ts_filename
    frozen.save(str(ts_path))
    meta.torchscript_file = ts_filename
    meta_path = meta.save(out / meta_filename)

    # Reload from disk and verify numerical parity against raw PyTorch model
    reloaded = torch.jit.load(str(ts_path), map_location=device)
    reloaded.eval()
    gen = torch.Generator(device=device).manual_seed(12345)
    test_obs = torch.randn((max(1, int(verify_samples)), meta.history_len, meta.frame_size), generator=gen, device=device, dtype=torch.float32)

    action_max_diff = 0.0
    est_max_diff = 0.0
    with torch.inference_mode():
        for i in range(test_obs.shape[0]):
            x = test_obs[i:i + 1]
            if isinstance(model, ProprioceptiveEstimatorActorCritic):
                ref_act, ref_est = model.action_with_estimate(x)
                ts_act, ts_est = reloaded(x)
                action_max_diff = max(action_max_diff, float((ref_act - ts_act).abs().max().item()))
                est_max_diff = max(est_max_diff, float((ref_est - ts_est).abs().max().item()))
            else:
                ref_act = model.action(x)
                ts_act = reloaded(x)
                action_max_diff = max(action_max_diff, float((ref_act - ts_act).abs().max().item()))

    if action_max_diff >= atol or (meta.model_type == 'autonomous_est' and est_max_diff >= atol):
        raise RuntimeError(
            f'TorchScript export parity failed (atol={atol}): '
            f'action_max_diff={action_max_diff:.3e}, est_max_diff={est_max_diff:.3e}'
        )

    return {
        'checkpoint': str(ckpt_path),
        'torchscript_path': str(ts_path),
        'meta_path': str(meta_path),
        'model_type': meta.model_type,
        'frame_size': meta.frame_size,
        'history_len': meta.history_len,
        'action_size': meta.action_size,
        'include_q_err': meta.include_q_err,
        'est_size': meta.est_size,
        'verify_samples': int(verify_samples),
        'action_max_abs_diff': action_max_diff,
        'est_max_abs_diff': est_max_diff if meta.model_type == 'autonomous_est' else None,
        'parity_ok': True,
    }


# ---------------------------------------------------------------------------
# Real-Time Policy Runtime (10 Hz policy / 50 Hz servo tick)
# ---------------------------------------------------------------------------

class SerpWormPolicyRuntime:
    """Production real-robot runtime replicating exact training preprocessing & actuator filtering."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        meta_path: str | Path | None = None,
        device: str = 'cpu',
        num_threads: int | None = 1,
        apply_frame_flip: bool = True,
        simulate_bus_delay: bool = False,
        safety_config: SafetyConfig | None = None,
    ):
        if device == 'cpu' and num_threads is not None:
            torch.set_num_threads(int(num_threads))

        path = Path(model_path)
        self.device = torch.device(device)
        self.apply_frame_flip = bool(apply_frame_flip)
        self.simulate_bus_delay = bool(simulate_bus_delay)

        if path.suffix.lower() in ('.ts', '.torchscript'):
            resolved_meta = Path(meta_path) if meta_path is not None else (path.parent / 'deploy_meta.json')
            if not resolved_meta.exists():
                raise FileNotFoundError(f'Missing deploy metadata JSON for TorchScript model: {resolved_meta}')
            self.meta = DeployMetadata.load(resolved_meta)
            self._backend = 'torchscript'
            self._runner: nn.Module = torch.jit.load(str(path), map_location=self.device)
            self._runner.eval()
            self._raw_model: nn.Module | None = None
        else:
            raw_model, self.meta = load_checkpoint_model_and_meta(path, device=self.device)
            self._backend = 'pytorch'
            self._raw_model = raw_model
            if isinstance(raw_model, ProprioceptiveEstimatorActorCritic):
                self._runner = DeployableEstimatorActor(raw_model).to(self.device).eval()
            else:
                self._runner = DeployableTemporalActor(raw_model).to(self.device).eval()

        self._init_runtime_state(safety_config=safety_config)

    @classmethod
    def from_model(
        cls,
        model: TemporalActorCritic | ProprioceptiveEstimatorActorCritic,
        meta: DeployMetadata,
        *,
        device: str = 'cpu',
        num_threads: int | None = 1,
        apply_frame_flip: bool = True,
        simulate_bus_delay: bool = False,
        safety_config: SafetyConfig | None = None,
    ) -> SerpWormPolicyRuntime:
        """Build runtime directly from an in-memory PyTorch model and ``DeployMetadata``."""
        obj = cls.__new__(cls)
        if device == 'cpu' and num_threads is not None:
            torch.set_num_threads(int(num_threads))
        obj.device = torch.device(device)
        obj.apply_frame_flip = bool(apply_frame_flip)
        obj.simulate_bus_delay = bool(simulate_bus_delay)
        obj.meta = meta
        obj._backend = 'pytorch'
        obj._raw_model = model.to(obj.device).eval()
        if isinstance(model, ProprioceptiveEstimatorActorCritic):
            obj._runner = DeployableEstimatorActor(model).to(obj.device).eval()
        else:
            obj._runner = DeployableTemporalActor(model).to(obj.device).eval()
        obj._init_runtime_state(safety_config=safety_config)
        return obj

    def _init_runtime_state(self, safety_config: SafetyConfig | None) -> None:
        self.hw = self.meta.build_hardware_config()
        self.sensor_dt = float(self.meta.sensor_dt_s)
        self.policy_dt = float(self.meta.policy_dt_s)
        self.steps_per_action = max(1, round(self.policy_dt / self.sensor_dt))
        self.history_len = int(self.meta.history_len)
        self.frame_size = int(self.meta.frame_size)
        self.action_size = int(self.meta.action_size)
        self.include_q_err = bool(self.meta.include_q_err)
        self.has_estnet = (self.meta.model_type == 'autonomous_est') and (self.meta.est_size > 0)

        # Exact filter gains & slew limits per sensor tick (matching AutonomousPhaseEnv / UnifiedSerpWormEnv)
        self.servo_gain = 1.0 - math.exp(-self.sensor_dt / max(self.hw.servo_filter_tau_s, 1e-4))
        self.servo_slew = self.hw.servo_slew_rad_s * self.sensor_dt
        self.yaw_gain = 1.0 - math.exp(-self.sensor_dt / max(self.hw.yaw_filter_tau_s, 1e-4))
        self.yaw_slew = self.hw.yaw_slew_rad_s * self.sensor_dt

        self.angle_scale = np.array(
            [max(self.hw.servo_max_rad, 1e-3)] * 10 + [max(self.hw.yaw_action_limit_rad, 1e-3)] * 4,
            dtype=np.float64,
        )
        self.frame_flip = np.asarray(self.meta.frame_flip, dtype=np.float64).reshape(3)
        self.dq_ema_alpha = float(self.meta.dq_ema_alpha)
        self.vx_scale = max(float(self.meta.vx_scale_floor), float(self.meta.vx_max_mm_s) * 1e-3)
        self.wz_scale = max(0.10, float(self.meta.wz_max_rad_s))

        self.safety = SafetyGuard(safety_config or SafetyConfig.from_hardware(self.hw))
        self.policy_latency = LatencyTracker(budget_ms=self.policy_dt * 1000.0)
        self.tick_latency = LatencyTracker(budget_ms=self.sensor_dt * 1000.0)

        # Runtime state buffers
        self.servo_neutral = np.full(10, self.hw.servo_neutral_rad, dtype=np.float64)
        self.servo_targets = self.servo_neutral.copy()
        self.yaw_targets = np.zeros(4, dtype=np.float64)
        self.servo_des = self.servo_neutral.copy()
        self.yaw_des = np.zeros(4, dtype=np.float64)
        self.prev_action = np.zeros(self.action_size, dtype=np.float64)
        self._pending_phys_action = np.zeros(self.action_size, dtype=np.float64)
        self._last_raw_action = np.zeros(self.action_size, dtype=np.float64)

        self.prev_angles = np.zeros(14, dtype=np.float64)
        self.dq = np.zeros(14, dtype=np.float64)
        self.delayed_raw = np.zeros(20, dtype=np.float64)
        self._last_packet = np.zeros(20, dtype=np.float64)
        self.history = np.zeros((self.history_len, self.frame_size), dtype=np.float32)
        self._obs_tensor = torch.zeros((1, self.history_len, self.frame_size), device=self.device, dtype=torch.float32)

        self.cmd = np.zeros(3, dtype=np.float64)
        self.mode_id: int = 2  # mixed default for v5
        self.last_est: EstNetPrediction | None = None
        self.last_safety: SafetyStatus | None = None
        self._initialized = False

    def _pack_raw_sensor(self, sensor: SensorReading) -> np.ndarray:
        angles = sensor.angles_rad
        if self.apply_frame_flip:
            gyro = sensor.gyro_rad_s * self.frame_flip
            proj_g = sensor.proj_g * self.frame_flip
        else:
            gyro = sensor.gyro_rad_s.copy()
            proj_g = sensor.proj_g.copy()
        return np.concatenate((angles, gyro, proj_g))

    def ingest_sensor(self, sensor: SensorReading, *, is_reset: bool = False) -> np.ndarray:
        """Ingest one 50 Hz sensor reading, updating optional 1-tick bus delay & EMA ``dq``."""
        raw = self._pack_raw_sensor(sensor)
        if is_reset:
            packet = raw.copy()
            self.delayed_raw = raw.copy()
            self.prev_angles = packet[:14].copy()
            self.dq = np.zeros(14, dtype=np.float64)
        else:
            if self.simulate_bus_delay:
                packet = self.delayed_raw.copy()
                self.delayed_raw = raw.copy()
            else:
                packet = raw.copy()
                self.delayed_raw = raw.copy()
            dq_inst = (packet[:14] - self.prev_angles) / self.sensor_dt
            self.dq = self.dq + self.dq_ema_alpha * (dq_inst - self.dq)
            self.prev_angles = packet[:14].copy()
        self._last_packet = packet
        return packet

    def build_frame(
        self,
        packet: np.ndarray | None = None,
        cmd: Sequence[float] | np.ndarray | None = None,
        mode_id: int | None = None,
    ) -> np.ndarray:
        """Construct single observation frame matching ``AutonomousPhaseEnv`` / ``UnifiedSerpWormEnv``."""
        pkt = self._last_packet if packet is None else np.asarray(packet, dtype=np.float64).reshape(20)
        c = self.cmd if cmd is None else np.asarray(cmd, dtype=np.float64).reshape(3)
        mid = self.mode_id if mode_id is None else int(mode_id)

        ang_norm = pkt[:14] / self.angle_scale
        dq_norm = np.clip(self.dq / 3.0, -3.0, 3.0)
        cmd_norm = np.array([c[0] / self.vx_scale, c[1] / self.wz_scale, c[2]], dtype=np.float64)

        parts: list[np.ndarray] = [ang_norm, dq_norm]
        if self.include_q_err:
            tgt_all = np.concatenate((self.servo_targets, self.yaw_targets))
            q_err = np.clip((tgt_all - pkt[:14]) / self.angle_scale, -2.0, 2.0)
            parts.append(q_err)

        parts.append(pkt[14:20])
        parts.append(self.prev_action)

        if self.meta.model_type == 'unified_temporal':
            mode_oh = np.zeros(3, dtype=np.float64)
            mode_oh[max(0, min(2, mid))] = 1.0
            parts.append(mode_oh)

        parts.append(cmd_norm)
        frame = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(
        self,
        initial_sensor: SensorReading,
        cmd: Sequence[float] = (0.0, 0.0, 0.0),
        mode_id: int = 2,
    ) -> np.ndarray:
        """Reset all filters, rolling 16-frame causal history buffer, and safety watchdog."""
        self.servo_targets = self.servo_neutral.copy()
        self.yaw_targets = np.zeros(4, dtype=np.float64)
        self.servo_des = self.servo_neutral.copy()
        self.yaw_des = np.zeros(4, dtype=np.float64)
        self.prev_action = np.zeros(self.action_size, dtype=np.float64)
        self._pending_phys_action = np.zeros(self.action_size, dtype=np.float64)
        self._last_raw_action = np.zeros(self.action_size, dtype=np.float64)
        self.cmd = np.asarray(cmd, dtype=np.float64).reshape(3).copy()
        self.mode_id = int(mode_id)
        self.last_est = None
        self.last_safety = None
        self.policy_latency.reset()
        self.tick_latency.reset()
        self.safety.reset(self.servo_targets, self.yaw_targets, timestamp_s=initial_sensor.timestamp_s)

        packet = self.ingest_sensor(initial_sensor, is_reset=True)
        frame = self.build_frame(packet, self.cmd, self.mode_id)
        self.history[:] = frame[None, :]
        self._initialized = True
        return self.history.copy()

    def map_action(
        self,
        raw_action: Sequence[float] | np.ndarray,
        mode_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact delta/absolute action mapping + optional v5 mode mask to desired rad targets."""
        a = np.clip(np.asarray(raw_action, dtype=np.float64).reshape(self.action_size), -1.0, 1.0)
        if self.meta.action_mode == 'delta':
            u = np.clip(self.prev_action + a * float(self.meta.delta_step_scale), -1.0, 1.0)
        else:
            u = a

        pos_span = self.hw.servo_max_rad - self.hw.servo_neutral_rad
        neg_span = self.hw.servo_neutral_rad - self.hw.servo_min_rad

        if self.meta.model_type == 'unified_temporal':
            mid = self.mode_id if mode_id is None else int(mode_id)
            t_sz = 5 if self.meta.tendon_pairing == 'paired_5d' else 10
            t_act = u[:t_sz]
            y_act = u[t_sz:t_sz + 4]
            t_eff = np.zeros_like(t_act) if mid == 0 else t_act
            y_eff = np.zeros_like(y_act) if mid == 1 else y_act
            c = np.repeat(t_eff, 2) if self.meta.tendon_pairing == 'paired_5d' else t_eff
            phys_act = np.concatenate((t_eff, y_eff))
        else:
            c = u[:10]
            y_eff = u[10:14]
            phys_act = u

        servo_des = self.servo_neutral + np.maximum(c, 0.0) * pos_span + np.minimum(c, 0.0) * neg_span
        servo_des = np.clip(servo_des, self.hw.servo_min_rad, self.hw.servo_max_rad)
        yaw_des = y_eff * self.hw.yaw_action_limit_rad
        return servo_des, yaw_des, phys_act

    def infer_action(self, history_override: np.ndarray | None = None) -> tuple[np.ndarray, EstNetPrediction | None, float]:
        """Run deterministic policy (+ EstNet) inference on ``[1, H, D]`` causal history."""
        hist = self.history if history_override is None else np.asarray(history_override, dtype=np.float32)
        t0 = time.perf_counter()
        self._obs_tensor[0].copy_(torch.from_numpy(hist))
        with torch.inference_mode():
            if self.has_estnet:
                act_t, est_t = self._runner(self._obs_tensor)
                act_np = act_t[0].detach().cpu().numpy().astype(np.float64)
                est_np = est_t[0].detach().cpu().numpy().astype(np.float64)
                est_pred: EstNetPrediction | None = EstNetPrediction.from_raw(est_np, self.vx_scale, self.wz_scale)
            else:
                act_t = self._runner(self._obs_tensor)
                act_np = act_t[0].detach().cpu().numpy().astype(np.float64)
                est_pred = None
        lat_ms = (time.perf_counter() - t0) * 1000.0
        self.policy_latency.record(lat_ms)
        return act_np, est_pred, lat_ms

    def _sync_latest_frame_command(self) -> None:
        """Reflect live cmd/mode_id updates into ``history[-1]`` before inference (zero switch lag)."""
        cmd_norm = np.array(
            [self.cmd[0] / self.vx_scale, self.cmd[1] / self.wz_scale, self.cmd[2]],
            dtype=np.float32,
        )
        self.history[-1, -3:] = np.nan_to_num(cmd_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if self.meta.model_type == 'unified_temporal':
            mode_oh = np.zeros(3, dtype=np.float32)
            mode_oh[max(0, min(2, self.mode_id))] = 1.0
            self.history[-1, -6:-3] = mode_oh

    def begin_policy_step(
        self,
        cmd: Sequence[float] | np.ndarray | None = None,
        mode_id: int | None = None,
    ) -> PolicyStepResult:
        """Execute 10 Hz policy inference and latch desired targets for the next 5 servo ticks."""
        if not self._initialized:
            raise RuntimeError('SerpWormPolicyRuntime.reset() must be called before stepping')
        updated_ctrl = False
        if cmd is not None:
            self.cmd = np.asarray(cmd, dtype=np.float64).reshape(3).copy()
            updated_ctrl = True
        if mode_id is not None:
            self.mode_id = int(mode_id)
            updated_ctrl = True
        if updated_ctrl:
            self._sync_latest_frame_command()

        raw_act, est_pred, lat_ms = self.infer_action()
        servo_des, yaw_des, phys_act = self.map_action(raw_act, self.mode_id)
        self.servo_des = servo_des
        self.yaw_des = yaw_des
        self._pending_phys_action = phys_act
        self._last_raw_action = raw_act
        self.last_est = est_pred
        return PolicyStepResult(
            raw_action=raw_act,
            physical_action=phys_act,
            servo_des_rad=servo_des.copy(),
            yaw_des_rad=yaw_des.copy(),
            est_pred=est_pred,
            latency_ms=lat_ms,
        )

    def compute_servo_command(
        self,
        sensor: SensorReading,
        *,
        dt_s: float | None = None,
        now_s: float | None = None,
    ) -> ServoTickResult:
        """Advance 50 Hz first-order filter + slew cap and run ``SafetyGuard``."""
        t0 = time.perf_counter()
        dt = self.sensor_dt if dt_s is None else float(dt_s)
        if abs(dt - self.sensor_dt) <= 1e-9:
            s_gain, s_slew = self.servo_gain, self.servo_slew
            y_gain, y_slew = self.yaw_gain, self.yaw_slew
        else:
            s_gain = 1.0 - math.exp(-dt / max(self.hw.servo_filter_tau_s, 1e-4))
            s_slew = self.hw.servo_slew_rad_s * dt
            y_gain = 1.0 - math.exp(-dt / max(self.hw.yaw_filter_tau_s, 1e-4))
            y_slew = self.hw.yaw_slew_rad_s * dt

        cand_servo = self.servo_targets + np.clip(s_gain * (self.servo_des - self.servo_targets), -s_slew, s_slew)
        cand_yaw = self.yaw_targets + np.clip(y_gain * (self.yaw_des - self.yaw_targets), -y_slew, y_slew)

        proj_g_flu = sensor.proj_g * self.frame_flip if self.apply_frame_flip else sensor.proj_g
        safe_servo, safe_yaw, status = self.safety.filter_targets(
            cand_servo,
            cand_yaw,
            sensor,
            dt_s=dt,
            now_s=now_s,
            proj_g_flu=proj_g_flu,
        )
        self.servo_targets = safe_servo.copy()
        self.yaw_targets = safe_yaw.copy()
        self.last_safety = status
        lat_ms = (time.perf_counter() - t0) * 1000.0
        self.tick_latency.record(lat_ms)
        return ServoTickResult(
            servo_cmd_rad=safe_servo,
            yaw_cmd_rad=safe_yaw,
            safety=status,
            latency_ms=lat_ms,
        )

    def end_policy_step(
        self,
        cmd: Sequence[float] | np.ndarray | None = None,
        mode_id: int | None = None,
    ) -> np.ndarray:
        """Commit ``prev_action`` and append latest preprocessed frame into 16-frame history."""
        if cmd is not None:
            self.cmd = np.asarray(cmd, dtype=np.float64).reshape(3).copy()
        if mode_id is not None:
            self.mode_id = int(mode_id)
        self.prev_action = self._pending_phys_action.copy()
        frame = self.build_frame(self._last_packet, self.cmd, self.mode_id)
        self.history[:-1] = self.history[1:]
        self.history[-1] = frame
        return self.history.copy()

    def run_policy_cycle(
        self,
        bridge: HardwareBridge,
        cmd: Sequence[float] = (0.10, 0.0, 0.0),
        *,
        mode_id: int | None = None,
        recorder: TelemetryRecorder | None = None,
        sleep_realtime: bool = False,
        last_sensor: SensorReading | None = None,
    ) -> tuple[PolicyStepResult, ServoTickResult, SensorReading]:
        """Run one full 10 Hz policy step + ``steps_per_action`` (5x) 50 Hz servo ticks."""
        sensor_cur = bridge.read_sensors() if last_sensor is None else last_sensor
        if not self._initialized:
            self.reset(sensor_cur, cmd=cmd, mode_id=self.mode_id if mode_id is None else mode_id)

        policy_res = self.begin_policy_step(cmd=cmd, mode_id=mode_id)
        last_tick: ServoTickResult | None = None
        for sub in range(self.steps_per_action):
            t_tick_start = time.perf_counter()
            last_tick = self.compute_servo_command(sensor_cur)
            if last_tick.safety.estop_active:
                bridge.emergency_stop()
            else:
                bridge.write_targets(last_tick.servo_cmd_rad, last_tick.yaw_cmd_rad)
            sensor_cur = bridge.read_sensors()
            self.ingest_sensor(sensor_cur, is_reset=False)

            if recorder is not None:
                recorder.record(
                    timestamp_s=sensor_cur.timestamp_s,
                    sensor=sensor_cur,
                    servo_cmd_rad=last_tick.servo_cmd_rad,
                    yaw_cmd_rad=last_tick.yaw_cmd_rad,
                    cmd=self.cmd,
                    action=policy_res.physical_action,
                    est_pred=policy_res.est_pred.raw if policy_res.est_pred is not None else None,
                    policy_latency_ms=policy_res.latency_ms if sub == 0 else 0.0,
                    tick_latency_ms=last_tick.latency_ms,
                    safety_flags=int(last_tick.safety.faults),
                    estop=last_tick.safety.estop_active,
                )
            if sleep_realtime:
                remain = self.sensor_dt - (time.perf_counter() - t_tick_start)
                if remain > 0.0:
                    time.sleep(remain)

        self.end_policy_step(cmd=self.cmd, mode_id=self.mode_id)
        assert last_tick is not None
        return policy_res, last_tick, sensor_cur

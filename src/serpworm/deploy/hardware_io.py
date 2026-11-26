"""Hardware abstraction, serial bus adapter, HIL mock bridge, and telemetry logging.

Supports the 5-segment SerpWorm robot (10 tendon servos + 4 inter-segment yaw
FashionStar 26KG serial bus servos + single 6-axis IMU) with clean separation
between the abstract protocol, deterministic HIL simulation, real UART bus
driver (with safe dry-run fallback), and compressed ``.npz`` telemetry capture.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import struct
import time
from typing import Any, Sequence

import numpy as np

from serpworm.config import HardwareConfig

try:
    import serial as _pyserial  # type: ignore[import-untyped]
except ImportError:
    _pyserial = None

__all__ = [
    'FS_REQ_HEADER',
    'FS_RESP_HEADER',
    'FS_CMD_PING',
    'FS_CMD_SET_ANGLE',
    'FS_CMD_SET_DAMPING',
    'FS_CMD_QUERY_ANGLE',
    'FS_CMD_QUERY_MONITOR',
    'SensorReading',
    'HardwareBridge',
    'MockHardwareConfig',
    'MockHardwareBridge',
    'FashionStarConfig',
    'FashionStarSerialBridge',
    'TelemetryRecorder',
    'encode_fs_packet',
    'decode_fs_packet',
    'parse_imu_line',
    'parse_imu_binary_frame',
]

# FashionStar UART serial bus servo protocol constants (- + S half-duplex bus)
FS_REQ_HEADER = b'\x12\x4C'
FS_RESP_HEADER = b'\x05\x1C'
FS_CMD_PING = 0x01
FS_CMD_SET_ANGLE = 0x08
FS_CMD_SET_DAMPING = 0x09
FS_CMD_QUERY_ANGLE = 0x0A
FS_CMD_QUERY_MONITOR = 0x16


@dataclass
class SensorReading:
    """Single 50 Hz proprioceptive sensor snapshot from 14 bus servos + 6-axis IMU.

    Conventions (matching ``AutonomousPhaseEnv._read_sensors``):
      - ``servo_rad``: 10 tendon servo measured angles (rad).
      - ``yaw_rad``: 4 inter-segment yaw joint measured angles (rad).
      - ``gyro_rad_s``: 3-axis angular rate (rad/s) in IMU/plate frame.
      - ``proj_g``: 3-axis projected unit gravity vector in IMU/plate frame
        (upright robot has ``proj_g ≈ [0, 0, -1]``).
      - ``timestamp_s``: monotonic capture time (s).
      - ``bus_ok``: False if a serial bus timeout or CRC error occurred.
    """

    servo_rad: np.ndarray
    yaw_rad: np.ndarray
    gyro_rad_s: np.ndarray
    proj_g: np.ndarray
    timestamp_s: float
    bus_ok: bool = True
    servo_temps_c: np.ndarray | None = None
    servo_voltage_v: np.ndarray | None = None
    servo_current_ma: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.servo_rad = np.array(self.servo_rad, dtype=np.float64, copy=True).reshape(10)
        self.yaw_rad = np.array(self.yaw_rad, dtype=np.float64, copy=True).reshape(4)
        self.gyro_rad_s = np.array(self.gyro_rad_s, dtype=np.float64, copy=True).reshape(3)
        self.proj_g = np.array(self.proj_g, dtype=np.float64, copy=True).reshape(3)
        self.timestamp_s = float(self.timestamp_s)
        self.bus_ok = bool(self.bus_ok)

    @property
    def angles_rad(self) -> np.ndarray:
        """Concatenated 14D joint angles ``[servo(10), yaw(4)]``."""
        return np.concatenate((self.servo_rad, self.yaw_rad))


class HardwareBridge(ABC):
    """Abstract real-time I/O interface for SerpWorm deployment & HIL loops."""

    def connect(self) -> None:
        """Open hardware transports if applicable."""

    def close(self) -> None:
        """Release hardware transports safely."""

    @abstractmethod
    def read_sensors(self) -> SensorReading:
        """Read current 14D servo/yaw angles and 6-axis IMU gyro + gravity."""

    @abstractmethod
    def write_targets(self, servo_rad: np.ndarray, yaw_rad: np.ndarray) -> None:
        """Command 10 tendon servo targets (rad) and 4 yaw servo targets (rad)."""

    @abstractmethod
    def emergency_stop(self) -> None:
        """Immediately put all 14 actuators into safe damped / neutral / torque-off state."""

    def __enter__(self) -> HardwareBridge:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class MockHardwareConfig:
    """Configuration for deterministic first-order HIL mock robot plant."""

    dt_s: float = 0.02
    servo_tau_s: float = 0.06
    yaw_tau_s: float = 0.06
    servo_speed_limit_rad_s: float = 8.2
    yaw_speed_limit_rad_s: float = 8.2
    servo_min_rad: float = 0.0
    servo_max_rad: float = math.pi
    servo_neutral_rad: float = 0.5 * math.pi
    yaw_hard_limit_rad: float = math.radians(34.0)
    latency_ticks: int = 1
    angle_noise_std: float = 0.002
    gyro_noise_std: float = 0.01
    gravity_noise_std: float = 0.005
    seed: int = 42

    @classmethod
    def from_hardware(
        cls,
        hw: HardwareConfig,
        dt_s: float = 0.02,
        latency_ticks: int = 1,
        noise: bool = True,
        seed: int = 42,
    ) -> MockHardwareConfig:
        return cls(
            dt_s=float(dt_s),
            servo_tau_s=float(hw.servo_filter_tau_s),
            yaw_tau_s=float(hw.yaw_filter_tau_s),
            servo_speed_limit_rad_s=float(hw.servo_speed_limit_rad_s),
            yaw_speed_limit_rad_s=float(hw.yaw_speed_limit_rad_s),
            servo_min_rad=float(hw.servo_min_rad),
            servo_max_rad=float(hw.servo_max_rad),
            servo_neutral_rad=float(hw.servo_neutral_rad),
            yaw_hard_limit_rad=float(hw.yaw_hard_limit_rad),
            latency_ticks=int(max(0, latency_ticks)),
            angle_noise_std=0.002 if noise else 0.0,
            gyro_noise_std=0.01 if noise else 0.0,
            gravity_noise_std=0.005 if noise else 0.0,
            seed=int(seed),
        )


class MockHardwareBridge(HardwareBridge):
    """First-order servo lag + configurable bus latency/noise HIL mock bridge."""

    def __init__(self, config: MockHardwareConfig | None = None):
        self.cfg = config or MockHardwareConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._servo_pos = np.full(10, self.cfg.servo_neutral_rad, dtype=np.float64)
        self._yaw_pos = np.zeros(4, dtype=np.float64)
        self._prev_yaw_pos = np.zeros(4, dtype=np.float64)
        self._servo_cmd = self._servo_pos.copy()
        self._yaw_cmd = self._yaw_pos.copy()
        self._cmd_queue: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._t_s = 0.0
        self._tick_count = 0
        self.estop_called = False
        self.estop_count = 0

        # Fault-injection overrides for safety watchdog HIL verification
        self._injected_tilt_rad = 0.0
        self._injected_bus_timeout = False
        self._injected_yaw_override: tuple[int, float] | None = None

        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._servo_pos = np.full(10, self.cfg.servo_neutral_rad, dtype=np.float64)
        self._yaw_pos = np.zeros(4, dtype=np.float64)
        self._prev_yaw_pos = np.zeros(4, dtype=np.float64)
        self._servo_cmd = self._servo_pos.copy()
        self._yaw_cmd = self._yaw_pos.copy()
        self._cmd_queue.clear()
        for _ in range(self.cfg.latency_ticks):
            self._cmd_queue.append((self._servo_pos.copy(), self._yaw_pos.copy()))
        self._t_s = 0.0
        self._tick_count = 0
        self.estop_called = False
        self.estop_count = 0
        self.clear_faults()

    def inject_tilt(self, tilt_deg: float) -> None:
        """Inject a simulated body roll tilt (deg) into IMU projected gravity."""
        self._injected_tilt_rad = math.radians(float(tilt_deg))

    def inject_bus_timeout(self, active: bool = True) -> None:
        """Force ``read_sensors()`` to return stale / bus_ok=False packets."""
        self._injected_bus_timeout = bool(active)

    def inject_joint_limit(self, yaw_index: int, angle_rad: float) -> None:
        """Force a single measured yaw joint angle to test hard-limit E-stop."""
        self._injected_yaw_override = (int(yaw_index), float(angle_rad))

    def clear_faults(self) -> None:
        self._injected_tilt_rad = 0.0
        self._injected_bus_timeout = False
        self._injected_yaw_override = None

    def _step_plant(self, target_servo: np.ndarray, target_yaw: np.ndarray, dt: float) -> None:
        self._prev_yaw_pos = self._yaw_pos.copy()
        s_alpha = 1.0 - math.exp(-dt / max(self.cfg.servo_tau_s, 1e-4))
        y_alpha = 1.0 - math.exp(-dt / max(self.cfg.yaw_tau_s, 1e-4))
        s_max_step = self.cfg.servo_speed_limit_rad_s * dt
        y_max_step = self.cfg.yaw_speed_limit_rad_s * dt

        ds = np.clip(s_alpha * (target_servo - self._servo_pos), -s_max_step, s_max_step)
        dy = np.clip(y_alpha * (target_yaw - self._yaw_pos), -y_max_step, y_max_step)
        self._servo_pos = np.clip(self._servo_pos + ds, self.cfg.servo_min_rad, self.cfg.servo_max_rad)
        self._yaw_pos = np.clip(self._yaw_pos + dy, -self.cfg.yaw_hard_limit_rad * 1.2, self.cfg.yaw_hard_limit_rad * 1.2)
        self._t_s += dt
        self._tick_count += 1

    def write_targets(self, servo_rad: np.ndarray, yaw_rad: np.ndarray) -> None:
        s_in = np.asarray(servo_rad, dtype=np.float64).reshape(10).copy()
        y_in = np.asarray(yaw_rad, dtype=np.float64).reshape(4).copy()
        self._servo_cmd = s_in
        self._yaw_cmd = y_in
        if self.cfg.latency_ticks > 0:
            self._cmd_queue.append((s_in, y_in))
            eff_servo, eff_yaw = self._cmd_queue.popleft()
        else:
            eff_servo, eff_yaw = s_in, y_in
        if self.estop_called:
            eff_servo = self._servo_pos.copy()
            eff_yaw = self._yaw_pos.copy()
        self._step_plant(eff_servo, eff_yaw, self.cfg.dt_s)

    def read_sensors(self) -> SensorReading:
        dt = max(self.cfg.dt_s, 1e-4)
        yaw_vel = (self._yaw_pos - self._prev_yaw_pos) / dt
        # Physically plausible plate-4 IMU gyro from center yaw joint motion
        wz = 0.25 * float(yaw_vel[1] + yaw_vel[2])
        wx = 0.02 * math.sin(2.0 * math.pi * 0.8 * self._t_s)
        wy = 0.02 * math.cos(2.0 * math.pi * 0.8 * self._t_s)
        gyro = np.array([wx, wy, wz], dtype=np.float64)

        tilt = self._injected_tilt_rad
        proj_g = np.array([0.0, -math.sin(tilt), -math.cos(tilt)], dtype=np.float64)

        servo_meas = self._servo_pos.copy()
        yaw_meas = self._yaw_pos.copy()
        if self._injected_yaw_override is not None:
            idx, val = self._injected_yaw_override
            yaw_meas[idx] = val

        if self.cfg.angle_noise_std > 0.0:
            servo_meas = servo_meas + self._rng.normal(0.0, self.cfg.angle_noise_std, size=10)
            yaw_meas = yaw_meas + self._rng.normal(0.0, self.cfg.angle_noise_std, size=4)
        if self.cfg.gyro_noise_std > 0.0:
            gyro = gyro + self._rng.normal(0.0, self.cfg.gyro_noise_std, size=3)
        if self.cfg.gravity_noise_std > 0.0:
            proj_g = proj_g + self._rng.normal(0.0, self.cfg.gravity_noise_std, size=3)
            norm = float(np.linalg.norm(proj_g))
            if norm > 1e-6:
                proj_g = proj_g / norm

        return SensorReading(
            servo_rad=servo_meas,
            yaw_rad=yaw_meas,
            gyro_rad_s=gyro,
            proj_g=proj_g,
            timestamp_s=self._t_s,
            bus_ok=not self._injected_bus_timeout,
            servo_temps_c=np.full(14, 34.0, dtype=np.float32),
            servo_voltage_v=np.full(14, 7.4, dtype=np.float32),
            servo_current_ma=np.full(14, 240.0, dtype=np.float32),
        )

    def emergency_stop(self) -> None:
        self.estop_called = True
        self.estop_count += 1
        self._t_s += self.cfg.dt_s

    def clear_estop(self) -> None:
        self.estop_called = False


# ---------------------------------------------------------------------------
# FashionStar 26KG Serial Bus Servo (- + S) + IMU UART Protocol Adapter
# ---------------------------------------------------------------------------

def encode_fs_packet(cmd_id: int, payload: bytes = b'', header: bytes = FS_REQ_HEADER) -> bytes:
    """Encode a FashionStar UART serial bus packet with 8-bit sum checksum."""
    if not (0 <= cmd_id <= 0xFF):
        raise ValueError(f'cmd_id out of byte range: {cmd_id}')
    if len(payload) > 0xFF:
        raise ValueError(f'payload exceeds 255 bytes: {len(payload)}')
    body = header + bytes((cmd_id, len(payload))) + payload
    checksum = sum(body) & 0xFF
    return body + bytes((checksum,))


def decode_fs_packet(buf: bytes, expected_header: bytes = FS_RESP_HEADER) -> tuple[int, bytes]:
    """Decode and validate a FashionStar UART packet; returns ``(cmd_id, payload)``."""
    if len(buf) < 5:
        raise ValueError(f'Packet too short ({len(buf)} bytes)')
    if buf[:2] != expected_header:
        raise ValueError(f'Unexpected header {buf[:2].hex()!r} (expected {expected_header.hex()!r})')
    cmd_id = buf[2]
    data_len = buf[3]
    expected_len = 5 + data_len
    if len(buf) < expected_len:
        raise ValueError(f'Truncated packet: need {expected_len} bytes, got {len(buf)}')
    frame = buf[:expected_len]
    checksum = sum(frame[:-1]) & 0xFF
    if checksum != frame[-1]:
        raise ValueError(f'Checksum mismatch: computed 0x{checksum:02X}, got 0x{frame[-1]:02X}')
    return cmd_id, frame[4:-1]


def parse_imu_line(line: str | bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse ASCII CSV IMU line ``wx,wy,wz,gx,gy,gz`` into ``(gyro_rad_s, proj_g)``."""
    text = line.decode('ascii', errors='strict') if isinstance(line, (bytes, bytearray)) else str(line)
    parts = [p.strip() for p in text.strip().split(',') if p.strip()]
    if len(parts) != 6:
        raise ValueError(f'Expected 6 comma-separated IMU floats, got {len(parts)}: {text!r}')
    vals = np.array([float(p) for p in parts], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError('Non-finite IMU value in ASCII frame')
    gyro = vals[:3]
    proj_g = vals[3:6]
    g_norm = float(np.linalg.norm(proj_g))
    if g_norm > 1e-6:
        proj_g = proj_g / g_norm
    return gyro, proj_g


def parse_imu_binary_frame(frame: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse 28-byte binary IMU packet: ``0xAA 0x55`` + 6x float32 LE + uint16 sum checksum."""
    if len(frame) < 28 or frame[0] != 0xAA or frame[1] != 0x55:
        raise ValueError('Invalid binary IMU frame header/length')
    payload = frame[2:26]
    expected_chk = sum(frame[:26]) & 0xFFFF
    got_chk = struct.unpack('<H', frame[26:28])[0]
    if expected_chk != got_chk:
        raise ValueError(f'Binary IMU checksum mismatch: {expected_chk} != {got_chk}')
    vals = np.array(struct.unpack('<6f', payload), dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError('Non-finite IMU value in binary frame')
    gyro = vals[:3]
    proj_g = vals[3:6]
    g_norm = float(np.linalg.norm(proj_g))
    if g_norm > 1e-6:
        proj_g = proj_g / g_norm
    return gyro, proj_g


@dataclass
class FashionStarConfig:
    """Serial bus + calibration configuration for 14 FashionStar 26KG servos & IMU."""

    servo_port: str | None = None
    servo_baudrate: int = 115200
    imu_port: str | None = None
    imu_baudrate: int = 115200
    timeout_s: float = 0.008
    servo_ids: tuple[int, ...] = tuple(range(14))  # 0..9 tendon, 10..13 yaw
    zero_offsets_rad: tuple[float, ...] = (0.0,) * 14
    directions: tuple[float, ...] = (1.0,) * 14
    move_interval_ms: int = 20
    power_limit_mw: int = 0  # 0 = servo default full rated power
    damping_estop_mw: int = 600
    servo_neutral_rad: float = 0.5 * math.pi
    dry_run: bool = False
    allow_dry_run_fallback: bool = True

    def __post_init__(self) -> None:
        if len(self.servo_ids) != 14:
            raise ValueError('FashionStarConfig requires exactly 14 servo_ids (10 tendon + 4 yaw)')
        if len(self.zero_offsets_rad) != 14 or len(self.directions) != 14:
            raise ValueError('zero_offsets_rad and directions must each have length 14')


class FashionStarSerialBridge(HardwareBridge):
    """FashionStar 26KG UART serial bus adapter (``- + S``) + 6-axis IMU parser.

    Operates against real serial ports when ``pyserial`` and ports are available,
    or runs in safe dry-run mode (encoding/validating real wire packets without
    opening hardware ports) when ``dry_run=True`` or ``allow_dry_run_fallback=True``
    and no port / ``pyserial`` is present.
    """

    def __init__(self, config: FashionStarConfig | None = None):
        self.cfg = config or FashionStarConfig()
        self._servo_ser: Any = None
        self._imu_ser: Any = None
        self.is_dry_run: bool = bool(self.cfg.dry_run)
        self._connected: bool = False
        self._estop_active: bool = False

        self._zero_offsets = np.asarray(self.cfg.zero_offsets_rad, dtype=np.float64)
        self._directions = np.asarray(self.cfg.directions, dtype=np.float64)

        # Last known/echoed state & wire-frame inspection buffer
        self._echo_servo = np.full(10, self.cfg.servo_neutral_rad, dtype=np.float64)
        self._echo_yaw = np.zeros(4, dtype=np.float64)
        self._last_gyro = np.zeros(3, dtype=np.float64)
        self._last_proj_g = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.last_tx_frames: list[bytes] = []
        self._t0 = time.monotonic()

    def connect(self) -> None:
        if self._connected:
            return
        if self.cfg.dry_run:
            self.is_dry_run = True
            self._connected = True
            return
        if _pyserial is None or not self.cfg.servo_port:
            if self.cfg.allow_dry_run_fallback:
                self.is_dry_run = True
                self._connected = True
                return
            raise RuntimeError(
                'FashionStarSerialBridge requires pyserial and a valid servo_port '
                '(or set dry_run=True / allow_dry_run_fallback=True).'
            )
        try:
            self._servo_ser = _pyserial.Serial(
                self.cfg.servo_port,
                baudrate=self.cfg.servo_baudrate,
                timeout=self.cfg.timeout_s,
            )
            if self.cfg.imu_port:
                self._imu_ser = _pyserial.Serial(
                    self.cfg.imu_port,
                    baudrate=self.cfg.imu_baudrate,
                    timeout=self.cfg.timeout_s,
                )
            self.is_dry_run = False
            self._connected = True
        except Exception as exc:
            self.close()
            if self.cfg.allow_dry_run_fallback:
                self.is_dry_run = True
                self._connected = True
                return
            raise RuntimeError(f'Failed to open serial bus ({self.cfg.servo_port}): {exc}') from exc

    def close(self) -> None:
        for attr in ('_servo_ser', '_imu_ser'):
            ser = getattr(self, attr, None)
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._connected = False

    def rad_to_raw_0p1deg(self, joint_index: int, angle_rad: float) -> int:
        deg = math.degrees((float(angle_rad) - float(self._zero_offsets[joint_index])) * float(self._directions[joint_index]))
        raw = int(round(deg * 10.0))
        return max(-32768, min(32767, raw))

    def raw_0p1deg_to_rad(self, joint_index: int, raw_0p1deg: int) -> float:
        deg = float(raw_0p1deg) * 0.1
        return math.radians(deg) * float(self._directions[joint_index]) + float(self._zero_offsets[joint_index])

    def build_set_angle_packet(self, joint_index: int, angle_rad: float, interval_ms: int | None = None) -> bytes:
        sid = int(self.cfg.servo_ids[joint_index])
        raw_ang = self.rad_to_raw_0p1deg(joint_index, angle_rad)
        dt_ms = max(0, min(65535, int(self.cfg.move_interval_ms if interval_ms is None else interval_ms)))
        pwr = max(0, min(65535, int(self.cfg.power_limit_mw)))
        payload = struct.pack('<BhHH', sid, raw_ang, dt_ms, pwr)
        return encode_fs_packet(FS_CMD_SET_ANGLE, payload)

    def build_query_angle_packet(self, joint_index: int) -> bytes:
        sid = int(self.cfg.servo_ids[joint_index])
        return encode_fs_packet(FS_CMD_QUERY_ANGLE, struct.pack('<B', sid))

    def build_damping_packet(self, joint_index: int, power_mw: int | None = None) -> bytes:
        sid = int(self.cfg.servo_ids[joint_index])
        pwr = max(0, min(65535, int(self.cfg.damping_estop_mw if power_mw is None else power_mw)))
        return encode_fs_packet(FS_CMD_SET_DAMPING, struct.pack('<BH', sid, pwr))

    @property
    def estop_active(self) -> bool:
        return self._estop_active

    def clear_estop(self) -> None:
        """Explicitly release latched hardware E-stop before resuming angle commands."""
        self._estop_active = False

    def write_targets(self, servo_rad: np.ndarray, yaw_rad: np.ndarray) -> None:
        if not self._connected:
            self.connect()
        if self._estop_active:
            # Never override hardware damping E-stop with full-power angle commands
            self.last_tx_frames = []
            return
        targets = np.concatenate((
            np.asarray(servo_rad, dtype=np.float64).reshape(10),
            np.asarray(yaw_rad, dtype=np.float64).reshape(4),
        ))
        frames = [self.build_set_angle_packet(i, float(targets[i])) for i in range(14)]
        self.last_tx_frames = frames
        if self.is_dry_run or self._servo_ser is None:
            # First-order echo tracking in dry-run so closed-loop HIL behaves smoothly
            self._echo_servo = self._echo_servo + 0.35 * (targets[:10] - self._echo_servo)
            self._echo_yaw = self._echo_yaw + 0.35 * (targets[10:14] - self._echo_yaw)
            return
        self._servo_ser.write(b''.join(frames))

    def _read_imu_hw(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self._imu_ser is None:
            return self._last_gyro.copy(), self._last_proj_g.copy(), True
        raw = self._imu_ser.readline()
        if not raw:
            return self._last_gyro.copy(), self._last_proj_g.copy(), False
        try:
            if len(raw) >= 28 and raw[0] == 0xAA and raw[1] == 0x55:
                gyro, proj_g = parse_imu_binary_frame(raw)
            else:
                gyro, proj_g = parse_imu_line(raw)
            self._last_gyro = gyro
            self._last_proj_g = proj_g
            return gyro, proj_g, True
        except ValueError:
            return self._last_gyro.copy(), self._last_proj_g.copy(), False

    def read_sensors(self) -> SensorReading:
        if not self._connected:
            self.connect()
        now = time.monotonic() - self._t0
        if self.is_dry_run or self._servo_ser is None:
            return SensorReading(
                servo_rad=self._echo_servo.copy(),
                yaw_rad=self._echo_yaw.copy(),
                gyro_rad_s=self._last_gyro.copy(),
                proj_g=self._last_proj_g.copy(),
                timestamp_s=now,
                bus_ok=True,
            )

        angles = np.concatenate((self._echo_servo, self._echo_yaw))
        bus_ok = True
        for i in range(14):
            pkt = self.build_query_angle_packet(i)
            self._servo_ser.write(pkt)
            resp = self._servo_ser.read(8)  # 2B hdr + 1B cmd + 1B len(=3) + 3B payload + 1B chk
            try:
                cmd_id, payload = decode_fs_packet(resp)
                if cmd_id != FS_CMD_QUERY_ANGLE or len(payload) < 3:
                    bus_ok = False
                    continue
                sid, raw_ang = struct.unpack('<Bh', payload[:3])
                if sid != int(self.cfg.servo_ids[i]):
                    bus_ok = False
                    continue
                angles[i] = self.raw_0p1deg_to_rad(i, raw_ang)
            except ValueError:
                bus_ok = False

        self._echo_servo = angles[:10].copy()
        self._echo_yaw = angles[10:14].copy()
        gyro, proj_g, imu_ok = self._read_imu_hw()
        return SensorReading(
            servo_rad=self._echo_servo.copy(),
            yaw_rad=self._echo_yaw.copy(),
            gyro_rad_s=gyro,
            proj_g=proj_g,
            timestamp_s=now,
            bus_ok=bool(bus_ok and imu_ok),
        )

    def emergency_stop(self) -> None:
        if not self._connected:
            self.connect()
        self._estop_active = True
        frames = [self.build_damping_packet(i) for i in range(14)]
        self.last_tx_frames = frames
        if (not self.is_dry_run) and self._servo_ser is not None:
            self._servo_ser.write(b''.join(frames))


# ---------------------------------------------------------------------------
# Compressed .npz Real-Robot / HIL Telemetry Recorder
# ---------------------------------------------------------------------------

class TelemetryRecorder:
    """Records synchronized sensor, command, EstNet, latency, and safety logs to ``.npz``."""

    def __init__(self, metadata: dict[str, Any] | None = None):
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._timestamps: list[float] = []
        self._servo_meas: list[np.ndarray] = []
        self._yaw_meas: list[np.ndarray] = []
        self._servo_cmd: list[np.ndarray] = []
        self._yaw_cmd: list[np.ndarray] = []
        self._gyro: list[np.ndarray] = []
        self._proj_g: list[np.ndarray] = []
        self._cmd: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._est_pred: list[np.ndarray] = []
        self._policy_lat_ms: list[float] = []
        self._tick_lat_ms: list[float] = []
        self._safety_flags: list[int] = []
        self._estop: list[bool] = []

    def __len__(self) -> int:
        return len(self._timestamps)

    def record(
        self,
        *,
        timestamp_s: float,
        sensor: SensorReading,
        servo_cmd_rad: np.ndarray,
        yaw_cmd_rad: np.ndarray,
        cmd: Sequence[float] | np.ndarray,
        action: Sequence[float] | np.ndarray,
        est_pred: Sequence[float] | np.ndarray | None = None,
        policy_latency_ms: float = 0.0,
        tick_latency_ms: float = 0.0,
        safety_flags: int = 0,
        estop: bool = False,
    ) -> None:
        self._timestamps.append(float(timestamp_s))
        self._servo_meas.append(np.asarray(sensor.servo_rad, dtype=np.float32).reshape(10).copy())
        self._yaw_meas.append(np.asarray(sensor.yaw_rad, dtype=np.float32).reshape(4).copy())
        self._servo_cmd.append(np.asarray(servo_cmd_rad, dtype=np.float32).reshape(10).copy())
        self._yaw_cmd.append(np.asarray(yaw_cmd_rad, dtype=np.float32).reshape(4).copy())
        self._gyro.append(np.asarray(sensor.gyro_rad_s, dtype=np.float32).reshape(3).copy())
        self._proj_g.append(np.asarray(sensor.proj_g, dtype=np.float32).reshape(3).copy())
        self._cmd.append(np.asarray(cmd, dtype=np.float32).reshape(3).copy())
        self._action.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
        if est_pred is None:
            self._est_pred.append(np.full(7, np.nan, dtype=np.float32))
        else:
            self._est_pred.append(np.asarray(est_pred, dtype=np.float32).reshape(7).copy())
        self._policy_lat_ms.append(float(policy_latency_ms))
        self._tick_lat_ms.append(float(tick_latency_ms))
        self._safety_flags.append(int(safety_flags))
        self._estop.append(bool(estop))

    def to_arrays(self) -> dict[str, np.ndarray]:
        n = len(self._timestamps)
        if n == 0:
            return {
                'timestamps_s': np.zeros((0,), dtype=np.float64),
                'servo_meas_rad': np.zeros((0, 10), dtype=np.float32),
                'yaw_meas_rad': np.zeros((0, 4), dtype=np.float32),
                'servo_cmd_rad': np.zeros((0, 10), dtype=np.float32),
                'yaw_cmd_rad': np.zeros((0, 4), dtype=np.float32),
                'gyro_rad_s': np.zeros((0, 3), dtype=np.float32),
                'proj_g': np.zeros((0, 3), dtype=np.float32),
                'cmd': np.zeros((0, 3), dtype=np.float32),
                'action': np.zeros((0, 14), dtype=np.float32),
                'est_pred': np.zeros((0, 7), dtype=np.float32),
                'policy_latency_ms': np.zeros((0,), dtype=np.float32),
                'tick_latency_ms': np.zeros((0,), dtype=np.float32),
                'safety_flags': np.zeros((0,), dtype=np.int32),
                'estop': np.zeros((0,), dtype=bool),
                'meta_json': np.array(json.dumps(self.metadata)),
            }
        return {
            'timestamps_s': np.asarray(self._timestamps, dtype=np.float64),
            'servo_meas_rad': np.stack(self._servo_meas, axis=0),
            'yaw_meas_rad': np.stack(self._yaw_meas, axis=0),
            'servo_cmd_rad': np.stack(self._servo_cmd, axis=0),
            'yaw_cmd_rad': np.stack(self._yaw_cmd, axis=0),
            'gyro_rad_s': np.stack(self._gyro, axis=0),
            'proj_g': np.stack(self._proj_g, axis=0),
            'cmd': np.stack(self._cmd, axis=0),
            'action': np.stack(self._action, axis=0),
            'est_pred': np.stack(self._est_pred, axis=0),
            'policy_latency_ms': np.asarray(self._policy_lat_ms, dtype=np.float32),
            'tick_latency_ms': np.asarray(self._tick_lat_ms, dtype=np.float32),
            'safety_flags': np.asarray(self._safety_flags, dtype=np.int32),
            'estop': np.asarray(self._estop, dtype=bool),
            'meta_json': np.array(json.dumps(self.metadata)),
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **self.to_arrays())
        return out

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        with np.load(Path(path), allow_pickle=False) as data:
            result: dict[str, Any] = {k: data[k] for k in data.files}
        if 'meta_json' in result:
            result['meta'] = json.loads(str(result['meta_json'].item()))
        else:
            result['meta'] = {}
        return result

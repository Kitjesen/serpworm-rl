"""Comprehensive unit & integration tests for SerpWorm real-robot deployment & sysid toolkit."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from serpworm.config import DomainRandConfig, HardwareConfig, TaskConfig
from serpworm.deploy.hardware_io import (
    FS_CMD_QUERY_ANGLE,
    FS_CMD_SET_ANGLE,
    FS_CMD_SET_DAMPING,
    FS_REQ_HEADER,
    FS_RESP_HEADER,
    FashionStarConfig,
    FashionStarSerialBridge,
    MockHardwareBridge,
    MockHardwareConfig,
    SensorReading,
    TelemetryRecorder,
    decode_fs_packet,
    encode_fs_packet,
    parse_imu_binary_frame,
    parse_imu_line,
)
from serpworm.deploy.policy_runtime import (
    DeployMetadata,
    LatencyTracker,
    SafetyConfig,
    SafetyFault,
    SafetyGuard,
    SerpWormPolicyRuntime,
    export_checkpoint_to_torchscript,
)
from serpworm.deploy.sysid_toolkit import (
    ActuatorNet,
    fit_capstan_from_pull_release,
    fit_rebound_rayleigh_beta,
    fit_slope_contact_params,
    train_actuator_net,
)
from serpworm.envs.autonomous_env import AutonomousPhaseEnv, AutonomousTaskConfig
from serpworm.envs.unified_env import UnifiedSerpWormEnv
from serpworm.rl.models import ProprioceptiveEstimatorActorCritic, TemporalActorCritic

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Exact Preprocessing & Actuator-Filter Parity vs AutonomousPhaseEnv / UnifiedSerpWormEnv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('include_q_err,expected_dim', [(False, 51), (True, 65)])
def test_autonomous_51d_and_65d_preprocessing_parity(include_q_err: bool, expected_dim: int):
    """Verify exact bit-compatible 51D & 65D sensor/history/action parity against AutonomousPhaseEnv."""
    hw = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
    hw.yaw_slew_rad_s = 6.8
    task = AutonomousTaskConfig(
        elements_per_strip=6,
        physics_dt_s=0.001,
        include_q_err=include_q_err,
        history_len=8,
        episode_seconds=20.0,
        command_switch_s=1e6,
        stage=2,
    )
    env = AutonomousPhaseEnv(
        n=1,
        hardware=hw,
        task=task,
        dr=DomainRandConfig(enabled=False),
        device='cpu',
        plate_solver='segment_lu',
        seed=17,
        noise=False,
        forced_regime=0,
        settle_ticks=4,
    )
    assert env.frame_size == expected_dim

    torch.manual_seed(101)
    model = ProprioceptiveEstimatorActorCritic(
        frame_size=env.frame_size,
        history_len=env.history_len,
        privileged_size=env.privileged_size,
        est_size=env.est_size,
        action_size=14,
    ).eval()
    # Populate non-trivial RunningNorm stats so normalization is genuinely exercised
    model.obs_norm.update(torch.randn(32, env.frame_size))

    meta = DeployMetadata(
        model_type='autonomous_est',
        frame_size=env.frame_size,
        history_len=env.history_len,
        action_size=14,
        est_size=7,
        include_q_err=include_q_err,
        tendon_pairing='independent_10d',
        action_mode=task.action_mode,
        delta_step_scale=task.delta_step_scale,
        mean_bound=model.mean_bound,
        policy_dt_s=task.policy_dt_s,
        sensor_dt_s=task.sensor_dt_s,
        vx_max_mm_s=task.vx_max_mm_s,
        wz_max_rad_s=task.wz_max_rad_s,
        vx_scale_floor=0.05,
        dq_ema_alpha=0.35,
        frame_flip=(-1.0, -1.0, 1.0),
        hardware=hw.to_dict(),
    )
    runtime = SerpWormPolicyRuntime.from_model(
        model,
        meta,
        device='cpu',
        num_threads=1,
        apply_frame_flip=True,
        simulate_bus_delay=True,
    )

    def _capture_plate_sensor(ts: float) -> SensorReading:
        _, v, R, q = env._geometry()
        gyro_plate = torch.einsum('bij,bj->bi', R[:, 4].transpose(-1, -2), v[:, 4, 3:])[0].cpu().numpy()
        proj_g_plate = (-R[:, 4, 2, :])[0].cpu().numpy()
        return SensorReading(
            servo_rad=env.c.servo[0].cpu().numpy(),
            yaw_rad=q[0].cpu().numpy(),
            gyro_rad_s=gyro_plate,
            proj_g=proj_g_plate,
            timestamp_s=ts,
            bus_ok=True,
        )

    # Set deterministic command & reset env + runtime from identical initial state
    env.cmd[0] = torch.tensor([0.125, -0.09, 0.0], dtype=env.dtype)
    _, _, _, _, init_pkt = env._read_sensors(torch.ones(1, dtype=torch.bool))
    env.history.copy_(env._build_frame(init_pkt)[:, None, :])
    env.obs = env.history.clone()

    t_sim = 0.0
    runtime.reset(_capture_plate_sensor(t_sim), cmd=env.cmd[0].cpu().numpy())
    assert float(np.max(np.abs(runtime.history - env.obs[0].cpu().numpy()))) < 1e-6

    # Capture each 50 Hz sub-tick sensor reading inside env.step without modifying autonomous_env.py
    captured_ticks: list[SensorReading] = []
    orig_read_sensors = env._read_sensors

    def _wrapped_read_sensors(reset_mask: torch.Tensor):
        nonlocal t_sim
        out = orig_read_sensors(reset_mask)
        # Only capture the 5 actual 50 Hz physics sub-ticks (skip the masked-out reset(done) call)
        if reset_mask is env.no_reset:
            t_sim += task.sensor_dt_s
            captured_ticks.append(_capture_plate_sensor(t_sim))
        return out

    env._read_sensors = _wrapped_read_sensors  # type: ignore[method-assign]
    try:
        for _ in range(6):
            # Compare policy & EstNet inference on identical obs
            with torch.no_grad():
                ref_act_t, ref_est_t = model.action_with_estimate(env.obs)
            pol = runtime.begin_policy_step(cmd=env.cmd[0].cpu().numpy())
            assert float(np.max(np.abs(pol.raw_action - ref_act_t[0].numpy()))) < 1e-5
            assert pol.est_pred is not None
            assert float(np.max(np.abs(np.array(pol.est_pred.raw) - ref_est_t[0].numpy()))) < 1e-5

            captured_ticks.clear()
            env_obs, _, _, _, term, trunc, _, _, _ = env.step(ref_act_t)
            assert not bool(term[0]) and not bool(trunc[0])
            assert len(captured_ticks) == runtime.steps_per_action

            for tick_sensor in captured_ticks:
                tick_res = runtime.compute_servo_command(tick_sensor)
                assert tick_res.safety.ok
                runtime.ingest_sensor(tick_sensor, is_reset=False)
            runtime.end_policy_step(cmd=env.cmd[0].cpu().numpy())

            # Exact parity on filtered targets, EMA dq, and full 16-frame causal history (51D or 65D)
            assert float(np.max(np.abs(runtime.servo_targets - env.servo_targets[0].cpu().numpy()))) < 1e-6
            assert float(np.max(np.abs(runtime.yaw_targets - env.yaw_targets[0].cpu().numpy()))) < 1e-6
            assert float(np.max(np.abs(runtime.dq - env.dq[0].cpu().numpy()))) < 1e-6
            assert float(np.max(np.abs(runtime.history - env_obs[0].cpu().numpy()))) < 1e-6
    finally:
        env._read_sensors = orig_read_sensors  # type: ignore[method-assign]


def test_unified_v5_preprocessing_and_mode_mask_parity():
    """Verify exact 54D v5 TemporalActorCritic preprocessing & snake/worm/mixed mode masks vs UnifiedSerpWormEnv."""
    hw = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
    task = TaskConfig(
        elements_per_strip=6,
        physics_dt_s=0.001,
        history_len=6,
        episode_seconds=15.0,
        command_switch_s=1e6,
        stage=2,
    )
    env = UnifiedSerpWormEnv(
        n=3,
        hardware=hw,
        task=task,
        dr=DomainRandConfig(enabled=False),
        device='cpu',
        plate_solver='segment_lu',
        seed=23,
        noise=False,
    )
    assert env.frame_size == 54

    torch.manual_seed(202)
    model = TemporalActorCritic(
        frame_size=env.frame_size,
        history_len=env.history_len,
        privileged_size=env.privileged_size,
        action_size=14,
    ).eval()
    model.obs_norm.update(torch.randn(16, env.frame_size))

    meta = DeployMetadata(
        model_type='unified_temporal',
        source_schema='serpworm_unified_ppo_v1',
        frame_size=env.frame_size,
        history_len=env.history_len,
        action_size=14,
        est_size=0,
        include_q_err=False,
        tendon_pairing='independent_10d',
        action_mode=task.action_mode,
        delta_step_scale=task.delta_step_scale,
        mean_bound=0.0,
        policy_dt_s=task.policy_dt_s,
        sensor_dt_s=task.sensor_dt_s,
        vx_max_mm_s=task.vx_max_mm_s,
        wz_max_rad_s=task.wz_max_rad_s,
        vx_scale_floor=0.04,
        dq_ema_alpha=0.35,
        frame_flip=(-1.0, -1.0, 1.0),
        hardware=hw.to_dict(),
    )
    runtimes = [
        SerpWormPolicyRuntime.from_model(
            model, meta, device='cpu', num_threads=1, apply_frame_flip=True, simulate_bus_delay=True
        )
        for _ in range(3)
    ]

    def _capture_lane(lane: int, ts: float) -> SensorReading:
        _, v, R, q = env._geometry()
        return SensorReading(
            servo_rad=env.c.servo[lane].cpu().numpy(),
            yaw_rad=q[lane].cpu().numpy(),
            gyro_rad_s=torch.einsum('bij,bj->bi', R[:, 4].transpose(-1, -2), v[:, 4, 3:])[lane].cpu().numpy(),
            proj_g=(-R[:, 4, 2, :])[lane].cpu().numpy(),
            timestamp_s=ts,
            bus_ok=True,
        )

    for lane in range(3):
        env.mode_id[lane] = lane  # 0=snake, 1=worm, 2=mixed
        env.cmd[lane] = torch.tensor([0.070, 0.08 if lane != 1 else 0.0, 0.0], dtype=env.dtype)
    _, _, _, _, init_pkt = env._read_sensors(torch.ones(3, dtype=torch.bool))
    env.history.copy_(env._build_frame(init_pkt)[:, None, :])
    env.obs = env.history.clone()

    t_sim = 0.0
    for lane in range(3):
        runtimes[lane].reset(_capture_lane(lane, t_sim), cmd=env.cmd[lane].cpu().numpy(), mode_id=lane)
        assert float(np.max(np.abs(runtimes[lane].history - env.obs[lane].cpu().numpy()))) < 1e-6

    captured: list[list[SensorReading]] = [[], [], []]
    orig_rs = env._read_sensors

    def _wrap_rs(reset_mask: torch.Tensor):
        nonlocal t_sim
        out = orig_rs(reset_mask)
        if reset_mask is env.no_reset:
            t_sim += task.sensor_dt_s
            for lane in range(3):
                captured[lane].append(_capture_lane(lane, t_sim))
        return out

    env._read_sensors = _wrap_rs  # type: ignore[method-assign]
    try:
        for _ in range(4):
            with torch.no_grad():
                ref_act = model.action(env.obs)
            for lane in range(3):
                pol = runtimes[lane].begin_policy_step(cmd=env.cmd[lane].cpu().numpy(), mode_id=lane)
                assert pol.est_pred is None
                assert float(np.max(np.abs(pol.raw_action - ref_act[lane].numpy()))) < 1e-5
                if lane == 0:  # snake masks tendons to 0
                    assert float(np.max(np.abs(pol.physical_action[:10]))) == 0.0
                elif lane == 1:  # worm masks yaw to 0
                    assert float(np.max(np.abs(pol.physical_action[10:14]))) == 0.0
                captured[lane].clear()

            env_obs, _, _, term, trunc, _, _, _ = env.step(ref_act)
            assert not bool(term.any()) and not bool(trunc.any())
            for lane in range(3):
                assert len(captured[lane]) == runtimes[lane].steps_per_action
                for tick_s in captured[lane]:
                    runtimes[lane].compute_servo_command(tick_s)
                    runtimes[lane].ingest_sensor(tick_s, is_reset=False)
                runtimes[lane].end_policy_step(cmd=env.cmd[lane].cpu().numpy(), mode_id=lane)
                assert float(np.max(np.abs(runtimes[lane].servo_targets - env.servo_targets[lane].cpu().numpy()))) < 1e-6
                assert float(np.max(np.abs(runtimes[lane].yaw_targets - env.yaw_targets[lane].cpu().numpy()))) < 1e-6
                assert float(np.max(np.abs(runtimes[lane].history - env_obs[lane].cpu().numpy()))) < 1e-6
    finally:
        env._read_sensors = orig_rs  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 2. TorchScript Export & Numerical Parity (< 1e-5)
# ---------------------------------------------------------------------------

def _save_synthetic_ckpt(path: Path, model_kind: str, include_q_err: bool = True) -> Path:
    torch.manual_seed(77)
    if model_kind == 'autonomous':
        hw = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
        task = AutonomousTaskConfig(include_q_err=include_q_err, history_len=16)
        frame_size = 65 if include_q_err else 51
        model = ProprioceptiveEstimatorActorCritic(
            frame_size=frame_size, history_len=16, privileged_size=108, est_size=7, action_size=14
        )
        model.obs_norm.update(torch.randn(64, frame_size))
        payload = {
            'schema': 'serpworm_autonomous_v6',
            'model': model.state_dict(),
            'frame_size': frame_size,
            'history_len': 16,
            'privileged_size': 108,
            'est_size': 7,
            'action_size': 14,
            'include_q_err': include_q_err,
            'hardware': hw.to_dict(),
            'task': asdict(task),
            'rollout': 120,
            'stage': 3,
        }
    else:
        pairing = 'paired_5d' if model_kind == 'unified_9d' else 'independent_10d'
        hw = HardwareConfig.v5_realistic(tendon_pairing=pairing, terrain='hard')
        task = TaskConfig(history_len=16)
        action_size = 9 if pairing == 'paired_5d' else 14
        frame_size = 34 + action_size + 6
        model = TemporalActorCritic(
            frame_size=frame_size, history_len=16, privileged_size=101, action_size=action_size
        )
        model.obs_norm.update(torch.randn(64, frame_size))
        payload = {
            'schema': 'serpworm_unified_ppo_v1',
            'model': model.state_dict(),
            'frame_size': frame_size,
            'history_len': 16,
            'privileged_size': 101,
            'action_size': action_size,
            'hardware': hw.to_dict(),
            'task': asdict(task),
            'rollout': 150,
            'stage': 3,
        }
    torch.save(payload, path)
    return path


@pytest.mark.parametrize(
    'kind,include_q_err',
    [
        ('autonomous', False),
        ('autonomous', True),
        ('unified_14d', False),
        ('unified_9d', False),
    ],
)
def test_torchscript_export_and_runtime_parity(tmp_path: Path, kind: str, include_q_err: bool):
    ckpt = _save_synthetic_ckpt(tmp_path / f'{kind}_{include_q_err}.pt', kind, include_q_err=include_q_err)
    out_dir = tmp_path / f'export_{kind}_{include_q_err}'
    report = export_checkpoint_to_torchscript(ckpt, out_dir, device='cpu', verify_samples=12, atol=1e-5)

    assert report['parity_ok'] is True
    assert report['action_max_abs_diff'] < 1e-5
    if kind == 'autonomous':
        assert report['est_max_abs_diff'] is not None and report['est_max_abs_diff'] < 1e-5

    rt_pt = SerpWormPolicyRuntime(ckpt, device='cpu', num_threads=1)
    rt_ts = SerpWormPolicyRuntime(report['torchscript_path'], device='cpu', num_threads=1)
    assert rt_pt.frame_size == rt_ts.frame_size
    assert rt_pt.has_estnet == rt_ts.has_estnet

    rng = np.random.default_rng(9)
    hist = rng.normal(0.0, 0.5, size=(rt_pt.history_len, rt_pt.frame_size)).astype(np.float32)
    a_pt, e_pt, _ = rt_pt.infer_action(hist)
    a_ts, e_ts, _ = rt_ts.infer_action(hist)
    assert float(np.max(np.abs(a_pt - a_ts))) < 1e-5
    if rt_pt.has_estnet:
        assert e_pt is not None and e_ts is not None
        assert float(np.max(np.abs(np.array(e_pt.raw) - np.array(e_ts.raw)))) < 1e-5


@pytest.mark.parametrize(
    'rel_ckpt',
    [
        'runs/v6_1_65d_elec/best.pt',
        'runs/v6a_51d_pure/best.pt',
        'runs/unified_v5_hispeed_14d/best.pt',
    ],
)
def test_real_local_checkpoint_export_parity(tmp_path: Path, rel_ckpt: str):
    ckpt_path = REPO_ROOT / rel_ckpt
    if not ckpt_path.exists():
        pytest.skip(f'Local checkpoint not present: {ckpt_path}')
    out_dir = tmp_path / ckpt_path.parent.name
    report = export_checkpoint_to_torchscript(ckpt_path, out_dir, device='cpu', verify_samples=10, atol=1e-5)
    assert report['parity_ok'] is True
    assert report['action_max_abs_diff'] < 1e-5
    if report['est_max_abs_diff'] is not None:
        assert report['est_max_abs_diff'] < 1e-5


# ---------------------------------------------------------------------------
# 3. SafetyGuard Watchdog Verification
# ---------------------------------------------------------------------------

def _nominal_sensor(ts: float = 0.0, yaw_deg: float = 0.0, tilt_deg: float = 0.0, bus_ok: bool = True) -> SensorReading:
    tilt = math.radians(tilt_deg)
    return SensorReading(
        servo_rad=np.full(10, 0.5 * math.pi, dtype=np.float64),
        yaw_rad=np.full(4, math.radians(yaw_deg), dtype=np.float64),
        gyro_rad_s=np.zeros(3, dtype=np.float64),
        proj_g=np.array([0.0, -math.sin(tilt), -math.cos(tilt)], dtype=np.float64),
        timestamp_s=ts,
        bus_ok=bus_ok,
    )


def test_safety_guard_clamps_tilt_joint_limit_timeout_and_nonfinite():
    cfg = SafetyConfig(
        yaw_hard_limit_deg=34.0,
        servo_min_deg=0.0,
        servo_max_deg=180.0,
        servo_neutral_deg=90.0,
        min_upright_cos=0.70,
        bus_timeout_s=0.08,
        hard_timeout_s=0.20,
        timeout_mode='hold_last_safe',
        neutral_ramp_rate_rad_s=2.0,
    )
    guard = SafetyGuard(cfg)
    guard.reset(timestamp_s=0.0)

    # 1. Upright & within limits -> ok=True, commanded target clamped cleanly
    s_ok, y_ok, st = guard.filter_targets(
        np.full(10, 4.5),  # > pi, should clamp
        np.full(4, math.radians(40.0)),  # > 34 deg, should clamp
        _nominal_sensor(ts=0.02),
        dt_s=0.02,
    )
    assert st.ok is True and st.estop_active is False
    assert SafetyFault.COMMAND_CLAMPED in st.faults
    assert float(np.max(s_ok)) == pytest.approx(cfg.servo_max_rad)
    assert float(np.max(np.abs(y_ok))) == pytest.approx(cfg.yaw_hard_limit_rad)

    # 2. Transient bus timeout in hold_last_safe mode -> holds last safe targets
    s_hold, y_hold, st_to = guard.filter_targets(
        np.full(10, 1.0),
        np.zeros(4),
        _nominal_sensor(ts=0.02, bus_ok=False),
        dt_s=0.02,
        now_s=0.12,
    )
    assert st_to.ok is False and st_to.estop_active is False
    assert SafetyFault.BUS_TIMEOUT in st_to.faults
    assert st_to.mode == 'hold_last_safe'
    assert np.allclose(s_hold, s_ok) and np.allclose(y_hold, y_ok)

    # 3. Persistent bus timeout > hard_timeout_s -> escalates to soft_neutral_ramp
    s_ramp, y_ramp, st_ramp = guard.filter_targets(
        np.full(10, 1.0),
        np.zeros(4),
        _nominal_sensor(ts=0.02, bus_ok=False),
        dt_s=0.02,
        now_s=0.30,
    )
    assert st_ramp.mode == 'soft_neutral_ramp'
    assert float(np.mean(np.abs(s_ramp - cfg.servo_neutral_rad))) < float(np.mean(np.abs(s_hold - cfg.servo_neutral_rad)))
    assert float(np.max(np.abs(y_ramp))) < float(np.max(np.abs(y_hold)))

    # 4. Tilt E-stop (55 deg -> cos(55)=0.5736 < 0.70) latches E-stop
    guard.reset(servo_rad=np.full(10, 2.4), yaw_rad=np.full(4, 0.3), timestamp_s=1.0)
    s_tilt, y_tilt, st_tilt = guard.filter_targets(
        np.full(10, 2.6),
        np.full(4, 0.35),
        _nominal_sensor(ts=1.02, tilt_deg=55.0),
        dt_s=0.02,
    )
    assert st_tilt.estop_active is True and st_tilt.ok is False
    assert SafetyFault.TILT in st_tilt.faults
    assert st_tilt.mode == 'soft_neutral_ramp'
    assert float(np.max(np.abs(s_tilt - 2.4))) <= cfg.neutral_ramp_rate_rad_s * 0.02 + 1e-9

    # Remains latched even if tilt clears, until clear_estop()
    _, _, st_still_latched = guard.filter_targets(
        np.full(10, 1.5), np.zeros(4), _nominal_sensor(ts=1.04, tilt_deg=0.0), dt_s=0.02
    )
    assert st_still_latched.estop_active is True
    guard.clear_estop()

    # 5. Measured hard yaw-joint limit E-stop (36 deg > 34 deg)
    _, _, st_jl = guard.filter_targets(
        np.full(10, 1.5), np.zeros(4), _nominal_sensor(ts=1.06, yaw_deg=36.0), dt_s=0.02
    )
    assert st_jl.estop_active is True and SafetyFault.JOINT_LIMIT in st_jl.faults
    guard.clear_estop()

    # 6. Non-finite sensor E-stop
    bad_sensor = _nominal_sensor(ts=1.08)
    bad_sensor.gyro_rad_s[0] = float('nan')
    s_nf, y_nf, st_nf = guard.filter_targets(np.full(10, 1.5), np.zeros(4), bad_sensor, dt_s=0.02)
    assert st_nf.estop_active is True and SafetyFault.NONFINITE in st_nf.faults
    assert np.all(np.isfinite(s_nf)) and np.all(np.isfinite(y_nf))


# ---------------------------------------------------------------------------
# 4. MockHardwareBridge HIL Loop, FashionStar Serial Protocol & TelemetryRecorder .npz
# ---------------------------------------------------------------------------

def test_mock_hardware_hil_loop_and_telemetry_npz_roundtrip(tmp_path: Path):
    ckpt = _save_synthetic_ckpt(tmp_path / 'v6_65d.pt', 'autonomous', include_q_err=True)
    export = export_checkpoint_to_torchscript(ckpt, tmp_path / 'export', device='cpu', verify_samples=6)
    rt = SerpWormPolicyRuntime(export['torchscript_path'], device='cpu', num_threads=1)

    bridge = MockHardwareBridge(MockHardwareConfig.from_hardware(rt.hw, dt_s=0.02, latency_ticks=1, noise=True, seed=5))
    recorder = TelemetryRecorder(metadata={'run_name': 'pytest_hil', 'frame_size': rt.frame_size})

    sensor = bridge.read_sensors()
    rt.reset(sensor, cmd=(0.12, 0.0, 0.0))

    for step in range(25):
        if step == 20:
            bridge.inject_tilt(60.0)
        pol, tick, sensor = rt.run_policy_cycle(
            bridge,
            cmd=(0.12, 0.0, 0.0),
            recorder=recorder,
            last_sensor=sensor,
        )
        assert pol.est_pred is not None
        assert np.all(np.isfinite(pol.est_pred.raw))
        assert np.all(np.isfinite(tick.servo_cmd_rad))

    assert bridge.estop_called is True
    assert rt.policy_latency.summary()['count'] == 25.0
    assert rt.policy_latency.summary()['p50_ms'] > 0.0
    assert len(recorder) == 25 * rt.steps_per_action

    npz_path = recorder.save(tmp_path / 'hil_telemetry.npz')
    loaded = TelemetryRecorder.load(npz_path)
    assert loaded['meta']['run_name'] == 'pytest_hil'
    assert loaded['servo_meas_rad'].shape == (125, 10)
    assert loaded['yaw_meas_rad'].shape == (125, 4)
    assert loaded['est_pred'].shape == (125, 7)
    assert np.all(np.isfinite(loaded['servo_cmd_rad']))
    assert np.all(np.isfinite(loaded['est_pred']))
    assert bool(loaded['estop'].any()) is True


def test_fashionstar_serial_protocol_and_dryrun():
    # Request/response packet roundtrip & checksum validation
    payload = struct.pack('<BhHH', 3, 900, 20, 0)
    pkt = encode_fs_packet(FS_CMD_SET_ANGLE, payload, header=FS_REQ_HEADER)
    cmd_id, decoded = decode_fs_packet(pkt, expected_header=FS_REQ_HEADER)
    assert cmd_id == FS_CMD_SET_ANGLE and decoded == payload

    corrupt = bytearray(pkt)
    corrupt[-1] ^= 0xFF
    with pytest.raises(ValueError, match='Checksum mismatch'):
        decode_fs_packet(bytes(corrupt), expected_header=FS_REQ_HEADER)

    # Response angle packet decode
    resp = encode_fs_packet(FS_CMD_QUERY_ANGLE, struct.pack('<Bh', 10, -155), header=FS_RESP_HEADER)
    rcmd, rpay = decode_fs_packet(resp, expected_header=FS_RESP_HEADER)
    assert rcmd == FS_CMD_QUERY_ANGLE
    assert struct.unpack('<Bh', rpay) == (10, -155)

    # IMU ASCII & binary frame parsing
    gyro_a, g_a = parse_imu_line('0.1,-0.2,0.3,0.0,0.0,-9.81\n')
    assert np.allclose(gyro_a, [0.1, -0.2, 0.3])
    assert np.allclose(g_a, [0.0, 0.0, -1.0], atol=1e-6)

    bin_body = b'\xAA\x55' + struct.pack('<6f', 0.05, -0.05, 0.2, 0.0, 0.0, -1.0)
    bin_pkt = bin_body + struct.pack('<H', sum(bin_body) & 0xFFFF)
    gyro_b, g_b = parse_imu_binary_frame(bin_pkt)
    assert np.allclose(gyro_b, [0.05, -0.05, 0.2], atol=1e-5)
    assert np.allclose(g_b, [0.0, 0.0, -1.0], atol=1e-5)

    # Dry-run FashionStarSerialBridge lifecycle (works without physical serial port)
    bridge = FashionStarSerialBridge(FashionStarConfig(servo_port=None, allow_dry_run_fallback=True))
    with bridge:
        assert bridge.is_dry_run is True
        s0 = bridge.read_sensors()
        assert s0.bus_ok is True and s0.servo_rad.shape == (10,) and s0.yaw_rad.shape == (4,)
        bridge.write_targets(np.full(10, 1.2), np.array([0.1, -0.1, 0.2, -0.2]))
        assert len(bridge.last_tx_frames) == 14
        for frame in bridge.last_tx_frames:
            cid, _ = decode_fs_packet(frame, expected_header=FS_REQ_HEADER)
            assert cid == FS_CMD_SET_ANGLE
        bridge.emergency_stop()
        assert bridge.estop_active is True
        assert len(bridge.last_tx_frames) == 14
        for frame in bridge.last_tx_frames:
            cid, _ = decode_fs_packet(frame, expected_header=FS_REQ_HEADER)
            assert cid == FS_CMD_SET_DAMPING

        # Guard: write_targets() while E-stopped MUST emit zero FS_CMD_SET_ANGLE frames
        s_before = bridge.read_sensors()
        bridge.write_targets(np.full(10, 0.2), np.array([0.4, -0.4, 0.4, -0.4]))
        assert bridge.last_tx_frames == []
        s_after = bridge.read_sensors()
        assert np.allclose(s_after.servo_rad, s_before.servo_rad)
        assert np.allclose(s_after.yaw_rad, s_before.yaw_rad)

        # Explicit clear_estop() re-enables normal angle commands
        bridge.clear_estop()
        assert bridge.estop_active is False
        bridge.write_targets(np.full(10, 1.3), np.zeros(4))
        assert len(bridge.last_tx_frames) == 14
        for frame in bridge.last_tx_frames:
            cid, _ = decode_fs_packet(frame, expected_header=FS_REQ_HEADER)
            assert cid == FS_CMD_SET_ANGLE

    # Strict non-fallback mode raises cleanly when port absent
    strict_bridge = FashionStarSerialBridge(FashionStarConfig(servo_port=None, dry_run=False, allow_dry_run_fallback=False))
    with pytest.raises(RuntimeError):
        strict_bridge.connect()


def test_latency_tracker_ring_buffer_rollover():
    tracker = LatencyTracker(budget_ms=5.0, max_samples=4)
    samples = [1.0, 2.0, 6.0, 3.0, 4.0, 8.0, 2.5, 1.5, 9.0, 0.5]
    for s in samples:
        tracker.record(s)
    stats = tracker.summary()
    window = np.array(samples[-4:], dtype=np.float64)
    assert stats['count'] == float(len(samples))
    assert stats['last_ms'] == pytest.approx(samples[-1])
    assert stats['mean_ms'] == pytest.approx(float(window.mean()))
    assert stats['p50_ms'] == pytest.approx(float(np.percentile(window, 50)))
    assert stats['p99_ms'] == pytest.approx(float(np.percentile(window, 99)))
    assert stats['max_ms'] == pytest.approx(float(window.max()))
    assert stats['overruns'] == float(sum(1 for s in samples if s > 5.0))


def test_mid_episode_cmd_and_mode_switch_zero_lag_and_estop_bus_guard(tmp_path: Path):
    # 1. Autonomous 65D: live cmd switch updates history[-1, -3:] before inference (zero 100 ms lag)
    ckpt_v6 = _save_synthetic_ckpt(tmp_path / 'v6_switch.pt', 'autonomous', include_q_err=True)
    rt_v6 = SerpWormPolicyRuntime(ckpt_v6, device='cpu', num_threads=1)
    s0 = _nominal_sensor(ts=0.0)
    rt_v6.reset(s0, cmd=(0.10, 0.0, 0.0))
    new_cmd = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # immediate stop switch
    expected_hist = rt_v6.history.copy()
    expected_hist[-1, -3:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ref_act, _, _ = rt_v6.infer_action(expected_hist)
    pol = rt_v6.begin_policy_step(cmd=new_cmd)
    assert np.allclose(rt_v6.history[-1, -3:], [0.0, 0.0, 1.0], atol=1e-6)
    assert float(np.max(np.abs(pol.raw_action - ref_act))) < 1e-5

    # 2. Unified v5 54D: live cmd + mode_id switch updates both mode_oh [-6:-3] and cmd [-3:] before inference
    ckpt_v5 = _save_synthetic_ckpt(tmp_path / 'v5_switch.pt', 'unified_14d')
    rt_v5 = SerpWormPolicyRuntime(ckpt_v5, device='cpu', num_threads=1)
    rt_v5.reset(s0, cmd=(0.06, 0.0, 0.0), mode_id=0)
    new_cmd_v5 = np.array([0.08, -0.15, 0.0], dtype=np.float64)
    expected_v5 = rt_v5.history.copy()
    expected_v5[-1, -6:-3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # mode_id=2 (mixed)
    expected_v5[-1, -3:] = np.array(
        [new_cmd_v5[0] / rt_v5.vx_scale, new_cmd_v5[1] / rt_v5.wz_scale, 0.0], dtype=np.float32
    )
    ref_act_v5, _, _ = rt_v5.infer_action(expected_v5)
    pol_v5 = rt_v5.begin_policy_step(cmd=new_cmd_v5, mode_id=2)
    assert np.allclose(rt_v5.history[-1, -6:-3], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(rt_v5.history[-1, -3:], expected_v5[-1, -3:], atol=1e-6)
    assert float(np.max(np.abs(pol_v5.raw_action - ref_act_v5))) < 1e-5

    # 3. End-to-end run_policy_cycle on FashionStarSerialBridge under tilt E-stop emits only damping frames
    fs_bridge = FashionStarSerialBridge(FashionStarConfig(dry_run=True))
    with fs_bridge:
        rt_v6.reset(fs_bridge.read_sensors(), cmd=(0.10, 0.0, 0.0))
        fs_bridge._last_proj_g = np.array([0.0, -math.sin(math.radians(60.0)), -math.cos(math.radians(60.0))])
        _, tick, _ = rt_v6.run_policy_cycle(fs_bridge, cmd=(0.10, 0.0, 0.0))
        assert tick.safety.estop_active is True
        assert fs_bridge.estop_active is True
        assert len(fs_bridge.last_tx_frames) == 14
        for frame in fs_bridge.last_tx_frames:
            cid, _ = decode_fs_packet(frame, expected_header=FS_REQ_HEADER)
            assert cid == FS_CMD_SET_DAMPING


# ---------------------------------------------------------------------------
# 5. System Identification Toolkit (all 4 fitters on synthetic ground truth)
# ---------------------------------------------------------------------------

def test_sysid_fit_rebound_rayleigh_beta():
    rng = np.random.default_rng(123)
    fps = 240.0
    t = np.arange(0, 480, dtype=np.float64) / fps  # 2.0 s @ 240 fps
    omega_n_true = 2.0 * math.pi * 11.5
    alpha_true = 0.05
    beta_true = 0.00055
    sigma_true = 0.5 * (alpha_true + beta_true * (omega_n_true ** 2))
    zeta_true = sigma_true / omega_n_true
    omega_d_true = omega_n_true * math.sqrt(1.0 - zeta_true ** 2)
    z_clean = 0.132 + 0.018 * np.exp(-sigma_true * t) * np.cos(omega_d_true * t + 0.35)

    res_clean = fit_rebound_rayleigh_beta(t, z_clean, rayleigh_alpha=alpha_true)
    assert res_clean.rayleigh_beta == pytest.approx(beta_true, rel=1e-5)
    assert res_clean.zeta == pytest.approx(zeta_true, rel=1e-5)
    assert res_clean.r2 > 0.999999

    z_noisy = z_clean + rng.normal(0.0, 0.00025, size=t.size)  # 0.25 mm vision jitter
    res_noisy = fit_rebound_rayleigh_beta(t, z_noisy, rayleigh_alpha=alpha_true)
    assert res_noisy.rayleigh_beta == pytest.approx(beta_true, rel=0.04)
    assert res_noisy.zeta == pytest.approx(zeta_true, rel=0.04)
    assert res_noisy.r2 > 0.99
    assert res_noisy.n_peaks >= 10


def test_sysid_fit_slope_contact_params():
    mass = 2.76
    g = 9.80665
    roll_true = 0.060
    mu_true = 0.80
    ratio_true = 1.18
    roll_deg = math.degrees(math.atan(roll_true))
    kinetic_n = mu_true * mass * g
    static_n = ratio_true * kinetic_n

    res = fit_slope_contact_params(roll_deg, static_n, kinetic_n, mass, g=g)
    assert res.wheel_rolling == pytest.approx(roll_true, rel=1e-6)
    assert res.wheel_mu == pytest.approx(mu_true, rel=1e-6)
    assert res.wheel_static_mu_ratio == pytest.approx(ratio_true, rel=1e-6)
    overrides = res.to_hardware_overrides()
    assert set(overrides.keys()) == {'wheel_rolling', 'wheel_mu', 'wheel_static_mu_ratio'}

    # Multi-trial input
    res_multi = fit_slope_contact_params(
        [roll_deg - 0.1, roll_deg, roll_deg + 0.1],
        [static_n - 0.2, static_n, static_n + 0.2],
        [kinetic_n - 0.1, kinetic_n, kinetic_n + 0.1],
        mass,
        g=g,
    )
    assert res_multi.n_trials == 3
    assert res_multi.wheel_rolling == pytest.approx(roll_true, rel=0.02)
    assert res_multi.wheel_mu == pytest.approx(mu_true, rel=0.02)
    assert res_multi.wheel_rolling_std > 0.0


def test_sysid_fit_capstan_from_pull_release():
    mu_theta_true = 0.18
    t0 = np.linspace(6.0, 38.0, 12)
    t_pull = t0 * math.exp(mu_theta_true)
    t_rel = t0 * math.exp(-mu_theta_true)

    res = fit_capstan_from_pull_release(t_pull, t_rel)
    assert res.capstan_mu_theta == pytest.approx(mu_theta_true, rel=1e-6)
    assert res.capstan_mu_theta_ols == pytest.approx(mu_theta_true, rel=1e-6)
    assert res.transmission_efficiency == pytest.approx(math.exp(-mu_theta_true), rel=1e-6)
    assert res.r2_ratio > 0.999999
    assert res.quality == 'good'
    assert res.n_valid == 12 and res.n_rejected == 0

    # Inject one inverted outlier pair -> rejected cleanly with warning
    t_pull_bad = np.append(t_pull, 5.0)
    t_rel_bad = np.append(t_rel, 8.0)
    res_bad = fit_capstan_from_pull_release(t_pull_bad, t_rel_bad)
    assert res_bad.n_rejected == 1 and res_bad.n_valid == 12
    assert res_bad.capstan_mu_theta == pytest.approx(mu_theta_true, rel=1e-6)
    assert len(res_bad.warnings) >= 1


def test_sysid_actuator_net_mini_fit():
    rng = np.random.default_rng(7)
    T, J = 450, 4
    dt = 0.02
    t = np.arange(T, dtype=np.float32)[:, None] * dt
    freq = np.array([[0.7, 1.1, 1.5, 0.9]], dtype=np.float32)
    phase = np.array([[0.0, 0.8, 1.7, 2.4]], dtype=np.float32)
    q_des = 0.45 * np.sin(2.0 * math.pi * freq * t + phase) + 0.15 * np.cos(2.0 * math.pi * 2.1 * freq * t)
    q = np.zeros_like(q_des)
    dq = np.zeros_like(q_des)
    for k in range(1, T):
        q[k] = q[k - 1] + 0.28 * (q_des[k - 1] - q[k - 1])
        dq[k] = (q[k] - q[k - 1]) / dt

    q_err = q_des - q
    q_err_prev = np.vstack((q_err[:1], q_err[:-1]))
    # Nonlinear FashionStar-style servo law: PD + Coulomb friction + 1-tick lag + saturation
    torque = np.clip(
        8.0 * q_err + 1.2 * q_err_prev - 0.14 * dq - 0.08 * np.tanh(6.0 * dq),
        -2.45,
        2.45,
    ) + rng.normal(0.0, 0.015, size=q_err.shape).astype(np.float32)

    model, metrics = train_actuator_net(
        {'q_des': q_des, 'q': q, 'dq': dq, 'torque': torque},
        tap_strides=(0, 1, 2),
        hidden_dim=32,
        activation='softsign',
        torque_limit_nm=2.45,
        epochs=90,
        batch_size=128,
        lr=4e-3,
        seed=7,
    )
    assert isinstance(model, ActuatorNet)
    assert metrics.val_r2 > 0.95
    assert metrics.val_rmse_nm < 0.10
    assert metrics.loss_history[-1] < metrics.loss_history[0]

    # Verify ActuatorNet is TorchScript exportable for real-time C++/Python integration
    traced = torch.jit.trace(model.eval(), torch.zeros((1, 6), dtype=torch.float32))
    x_sample = torch.randn((4, 6), dtype=torch.float32)
    with torch.inference_mode():
        assert torch.allclose(model(x_sample), traced(x_sample), atol=1e-6)

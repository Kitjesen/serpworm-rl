#!/usr/bin/env python3
"""200-step Hardware-in-the-Loop (HIL) dry-run benchmark & TorchScript parity audit.

Exports a trained SerpWorm .pt checkpoint to TorchScript (.ts) + deploy_meta.json,
runs a 200-step (20 s @ 10 Hz policy / 1000 ticks @ 50 Hz servo) HIL closed-loop
dry-run against MockHardwareBridge, verifies exact numerical parity (< 1e-5) vs raw
PyTorch inference on every policy step, logs .npz telemetry, and prints CPU p50/p99
inference latency and EstNet telemetry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.deploy.hardware_io import MockHardwareBridge, MockHardwareConfig, TelemetryRecorder
from serpworm.deploy.policy_runtime import (
    SerpWormPolicyRuntime,
    export_checkpoint_to_torchscript,
    load_checkpoint_model_and_meta,
)

DEFAULT_CHECKPOINT_CANDIDATES = (
    REPO_ROOT / 'runs' / 'v6_1_65d_elec' / 'best.pt',
    REPO_ROOT / 'runs' / 'v6b_65d_qerr' / 'best.pt',
    REPO_ROOT / 'runs' / 'v6a_51d_pure' / 'best.pt',
    REPO_ROOT / 'runs' / 'unified_v5_hispeed_14d' / 'best.pt',
    REPO_ROOT / 'runs' / 'unified_v3_150r' / 'best.pt',
)


def resolve_default_checkpoint() -> Path:
    for cand in DEFAULT_CHECKPOINT_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        'No local checkpoint found in runs/. Pass --checkpoint <path/to/best.pt> explicitly.'
    )


def run_hil_dryrun(
    ckpt_path: Path,
    out_dir: Path,
    *,
    steps: int = 200,
    cmd_vx_mm_s: float = 130.0,
    cmd_wz_rad_s: float = 0.0,
    mode_id: int = 2,
    latency_ticks: int = 1,
    noise: bool = True,
    num_threads: int = 1,
    warmup_steps: int = 15,
    atol: float = 1e-5,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_report = export_checkpoint_to_torchscript(
        ckpt_path,
        out_dir,
        device='cpu',
        verify_samples=16,
        atol=atol,
    )

    ts_path = Path(export_report['torchscript_path'])
    meta_path = Path(export_report['meta_path'])

    # Load both exported TorchScript runtime and raw PyTorch reference model on CPU
    ts_runtime = SerpWormPolicyRuntime(
        ts_path,
        meta_path=meta_path,
        device='cpu',
        num_threads=num_threads,
        apply_frame_flip=True,
        simulate_bus_delay=False,
    )
    raw_model, _ = load_checkpoint_model_and_meta(ckpt_path, device='cpu')
    raw_model.eval()

    mock_cfg = MockHardwareConfig.from_hardware(
        ts_runtime.hw,
        dt_s=ts_runtime.sensor_dt,
        latency_ticks=latency_ticks,
        noise=noise,
        seed=42,
    )
    bridge = MockHardwareBridge(mock_cfg)
    recorder = TelemetryRecorder(metadata={
        'checkpoint': str(ckpt_path),
        'torchscript': str(ts_path),
        'model_type': ts_runtime.meta.model_type,
        'frame_size': ts_runtime.frame_size,
        'history_len': ts_runtime.history_len,
        'steps': int(steps),
    })

    cmd = np.array([cmd_vx_mm_s * 1e-3, cmd_wz_rad_s, 0.0], dtype=np.float64)

    # Warm up TorchScript CPU execution caches before timing the benchmark loop
    dummy_hist = np.zeros((ts_runtime.history_len, ts_runtime.frame_size), dtype=np.float32)
    for _ in range(max(1, int(warmup_steps))):
        ts_runtime.infer_action(dummy_hist)

    init_sensor = bridge.read_sensors()
    ts_runtime.reset(init_sensor, cmd=cmd, mode_id=mode_id)

    action_max_diff = 0.0
    est_max_diff = 0.0
    est_records: list[dict[str, Any]] = []
    estop_triggered = False

    sensor_cur = init_sensor
    for _ in range(int(steps)):
        # Capture exact pre-step history to verify TorchScript vs raw PyTorch model parity
        obs_t = torch.from_numpy(ts_runtime.history[None, :, :].copy())
        policy_res, tick_res, sensor_cur = ts_runtime.run_policy_cycle(
            bridge,
            cmd=cmd,
            mode_id=mode_id,
            recorder=recorder,
            sleep_realtime=False,
            last_sensor=sensor_cur,
        )
        with torch.inference_mode():
            if ts_runtime.has_estnet:
                ref_act_t, ref_est_t = raw_model.action_with_estimate(obs_t)  # type: ignore[union-attr]
                ref_act = ref_act_t[0].numpy().astype(np.float64)
                ref_est = ref_est_t[0].numpy().astype(np.float64)
                action_max_diff = max(action_max_diff, float(np.max(np.abs(policy_res.raw_action - ref_act))))
                assert policy_res.est_pred is not None
                est_raw = np.asarray(policy_res.est_pred.raw, dtype=np.float64)
                est_max_diff = max(est_max_diff, float(np.max(np.abs(est_raw - ref_est))))
                est_records.append(policy_res.est_pred.to_dict())
            else:
                ref_act_t = raw_model.action(obs_t)  # type: ignore[union-attr]
                ref_act = ref_act_t[0].numpy().astype(np.float64)
                action_max_diff = max(action_max_diff, float(np.max(np.abs(policy_res.raw_action - ref_act))))

        if tick_res.safety.estop_active:
            estop_triggered = True

    telemetry_path = recorder.save(out_dir / 'hil_dryrun_telemetry.npz')
    lat_summary = ts_runtime.policy_latency.summary()
    tick_summary = ts_runtime.tick_latency.summary()

    est_summary: dict[str, Any] | None = None
    if est_records:
        est_summary = {
            'last': est_records[-1],
            'mean_vx_mm_s': round(float(np.mean([r['vx_mm_s'] for r in est_records])), 2),
            'mean_vy_mm_s': round(float(np.mean([r['vy_mm_s'] for r in est_records])), 2),
            'mean_wz_rad_s': round(float(np.mean([r['wz_rad_s'] for r in est_records])), 4),
            'mean_wall_conf': round(float(np.mean([r['wall_conf'] for r in est_records])), 4),
            'mean_wall_side': round(float(np.mean([r['wall_side'] for r in est_records])), 4),
            'mean_terrain_drag': round(float(np.mean([r['terrain_drag'] for r in est_records])), 4),
            'mean_slip': round(float(np.mean([r['slip'] for r in est_records])), 4),
        }

    parity_ok = (action_max_diff < atol) and ((not ts_runtime.has_estnet) or (est_max_diff < atol))
    report = {
        'checkpoint': str(ckpt_path),
        'torchscript_path': str(ts_path),
        'meta_path': str(meta_path),
        'telemetry_npz': str(telemetry_path),
        'model_type': ts_runtime.meta.model_type,
        'frame_size': ts_runtime.frame_size,
        'history_len': ts_runtime.history_len,
        'action_size': ts_runtime.action_size,
        'include_q_err': ts_runtime.include_q_err,
        'policy_steps': int(steps),
        'servo_ticks': len(recorder),
        'cpu_threads': int(num_threads),
        'policy_latency_ms': {k: round(v, 4) for k, v in lat_summary.items()},
        'servo_tick_latency_ms': {k: round(v, 4) for k, v in tick_summary.items()},
        'parity': {
            'action_max_abs_diff': action_max_diff,
            'est_max_abs_diff': est_max_diff if ts_runtime.has_estnet else None,
            'atol': atol,
            'passed': parity_ok,
        },
        'estnet': est_summary,
        'safety': {
            'estop_triggered': estop_triggered,
            'last_ok': bool(ts_runtime.last_safety.ok) if ts_runtime.last_safety is not None else True,
            'last_upright_cos': round(ts_runtime.last_safety.upright_cos, 4) if ts_runtime.last_safety else 1.0,
            'last_max_abs_yaw_deg': round(ts_runtime.last_safety.max_abs_yaw_deg, 2) if ts_runtime.last_safety else 0.0,
        },
    }
    (out_dir / 'hil_dryrun_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not parity_ok:
        raise RuntimeError(
            f'HIL TorchScript parity check failed: action_max_abs_diff={action_max_diff:.3e}, '
            f'est_max_abs_diff={est_max_diff:.3e} (atol={atol})'
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run 200-step SerpWorm HIL dry-run benchmark & parity check.')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Path to .pt checkpoint (auto-detects if omitted).')
    parser.add_argument('--out-dir', type=Path, default=REPO_ROOT / 'runs' / 'deploy_hil_dryrun', help='Output directory.')
    parser.add_argument('--steps', type=int, default=200, help='Number of 10 Hz policy steps (default: 200).')
    parser.add_argument('--cmd-vx-mm-s', type=float, default=130.0, help='Forward speed command (mm/s).')
    parser.add_argument('--cmd-wz-rad-s', type=float, default=0.0, help='Yaw rate command (rad/s).')
    parser.add_argument('--mode-id', type=int, default=2, help='Gait mode ID for v5 checkpoints (0=snake,1=worm,2=mixed).')
    parser.add_argument('--latency-ticks', type=int, default=1, help='Mock bus delay in 20 ms ticks (default: 1).')
    parser.add_argument('--no-noise', action='store_true', help='Disable mock sensor noise.')
    parser.add_argument('--num-threads', type=int, default=1, help='CPU threads for TorchScript runtime (default: 1).')
    args = parser.parse_args(argv)

    ckpt_path = args.checkpoint if args.checkpoint is not None else resolve_default_checkpoint()
    report = run_hil_dryrun(
        ckpt_path,
        args.out_dir,
        steps=args.steps,
        cmd_vx_mm_s=args.cmd_vx_mm_s,
        cmd_wz_rad_s=args.cmd_wz_rad_s,
        mode_id=args.mode_id,
        latency_ticks=args.latency_ticks,
        noise=not args.no_noise,
        num_threads=args.num_threads,
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

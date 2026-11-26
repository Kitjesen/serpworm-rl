#!/usr/bin/env python3
"""End-to-end GPU physics & RL rollout benchmark across mesh (E), tangent reuse, dt, and solver."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.config import HardwareConfig, TaskConfig
from serpworm.envs.unified_env import UnifiedSerpWormEnv
from serpworm.physics.fused_cuda import is_available as fused_available
from serpworm.physics.gpu_core import GPUCore


def measure_accuracy(hw: HardwareConfig, device: str) -> list[dict]:
    """Compare 1.0 s active snake+worm actuation trajectories against E=16, dt=0.5ms, K=1 baseline."""
    seconds = 1.0
    configs = [
        ('baseline_E16_dt0.5_K1', 16, 0.0005, 1),
        ('frozen_E16_dt0.5_K2',   16, 0.0005, 2),
        ('frozen_E16_dt0.5_K4',   16, 0.0005, 4),
        ('mesh_E12_dt0.5_K1',     12, 0.0005, 1),
        ('mesh_E8_dt0.5_K1',       8, 0.0005, 1),
        ('mesh_E8_dt0.5_K4',       8, 0.0005, 4),
        ('mesh_E8_dt1.0_K1',       8, 0.0010, 1),
    ]
    traces = {}
    for label, E, dt, K in configs:
        c = GPUCore(5, 1, dt=dt, device=device, plate_solver='segment_lu', hardware=hw, elements_per_strip=E, tangent_interval=K)
        if device == 'cuda':
            c.capture(max(4, K))
        ticks = round(seconds / 0.02)
        substeps = round(0.02 / dt)
        for step in range(ticks):
            t = step * 0.02
            wave_sv = [hw.servo_neutral_rad + 0.55 * math.sin(2 * math.pi * 0.8 * t + 0.9 * (i // 2)) for i in range(10)]
            wave_yw = [0.38 * math.sin(2 * math.pi * 0.8 * t + 1.1 * j) for j in range(4)]
            c.target.copy_(torch.tensor([wave_sv], device=device, dtype=torch.float64))
            c.yaw.copy_(torch.tensor([wave_yw], device=device, dtype=torch.float64))
            c.advance_steps(substeps)
        traces[label] = (
            c.x[0, :10, :3].detach().cpu(),
            c.tension[0].detach().cpu(),
            c.support[0].detach().cpu(),
            bool(c.failed),
        )
        del c

    ref_x, ref_ten, ref_sup, _ = traces['baseline_E16_dt0.5_K1']
    rows = []
    for label, E, dt, K in configs:
        x, ten, sup, failed = traces[label]
        rows.append({
            'config': label,
            'E': E,
            'dt_ms': dt * 1000.0,
            'tangent_interval': K,
            'failed': failed,
            'plate_pos_max_err_mm': round(float((x - ref_x).abs().max().item()) * 1000.0, 4),
            'centroid_disp_err_mm': round(float((x.mean(0) - ref_x.mean(0)).norm().item()) * 1000.0, 4),
            'tension_max_err_N': round(float((ten - ref_ten).abs().max().item()), 4),
            'support_max_err_N': round(float((sup - ref_sup).abs().max().item()), 4),
        })
    return rows


def measure_throughput(batch: int, steps: int, hw: HardwareConfig, device: str) -> list[dict]:
    solvers = ['segment_lu'] + (['fused_cuda'] if fused_available() else [])
    candidates = []
    for solver in solvers:
        candidates.append((f'{solver}_E16_dt0.5_K1', solver, 16, 0.0005, 1))
        if solver == 'segment_lu':
            candidates.append((f'{solver}_E16_dt0.5_K4', solver, 16, 0.0005, 4))
        candidates.append((f'{solver}_E8_dt0.5_K1', solver, 8, 0.0005, 1))
        if solver == 'segment_lu':
            candidates.append((f'{solver}_E8_dt0.5_K4', solver, 8, 0.0005, 4))
            candidates.append((f'{solver}_E8_dt1.0_K2', solver, 8, 0.0010, 2))

    rows = []
    base_sps = None
    for label, solver, E, dt, K in candidates:
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        task = TaskConfig(
            physics_dt_s=dt, policy_dt_s=0.10, sensor_dt_s=0.02,
            elements_per_strip=E, tangent_interval=K, episode_seconds=20.0, stage=2,
        )
        env = UnifiedSerpWormEnv(n=batch, hardware=hw, task=task, device=device, plate_solver=solver, noise=True)
        act = torch.full((batch, hw.action_size), 0.35, device=device, dtype=torch.float32)
        env.step(act)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(act)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        sps = (batch * steps) / elapsed
        if base_sps is None:
            base_sps = sps
        vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == 'cuda' else 0.0
        row = {
            'config': label,
            'solver': env.c.plate_solver,
            'batch': batch,
            'E': E,
            'dt_ms': dt * 1000.0,
            'K': K,
            'policy_sps': round(sps, 1),
            'speedup_vs_base': round(sps / base_sps, 2),
            'vram_gb': round(vram_gb, 2),
        }
        print(
            f"{label:26s} | B={batch:4d} | {sps:8.1f} steps/s | {row['speedup_vs_base']:5.2f}x | VRAM={vram_gb:.2f} GB",
            flush=True,
        )
        rows.append(row)
        del env
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--steps', type=int, default=6)
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args(argv)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hw = HardwareConfig.ds3218_default()
    print(f"Running accuracy + throughput benchmark on {device} ({torch.cuda.get_device_name() if device == 'cuda' else 'cpu'}), nvcc_fused={fused_available()}")
    acc = measure_accuracy(hw, device)
    for r in acc:
        print(f"  [acc] {r['config']:24s} | max_plate_err={r['plate_pos_max_err_mm']:6.3f} mm | centroid_err={r['centroid_disp_err_mm']:6.3f} mm | ten_err={r['tension_max_err_N']:6.3f} N")
    perf = measure_throughput(args.batch, args.steps, hw, device)
    report = {'device': device, 'fused_cuda_compiled': fused_available(), 'accuracy_1s': acc, 'throughput': perf}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

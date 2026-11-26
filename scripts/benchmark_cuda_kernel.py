#!/usr/bin/env python3
"""Benchmark & verify FP64 parity of the lossless fused CUDA physics kernel (V3, E=16 & E=8)."""
import argparse
from pathlib import Path
import sys
import time
import torch

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.config import HardwareConfig
from serpworm.physics.gpu_core import GPUCore


def _verify_parity(elements: int, dt: float, hw: HardwareConfig) -> None:
    ref = GPUCore(5, 4, dt=dt, device='cuda', plate_solver='segment_lu', hardware=hw, elements_per_strip=elements)
    fus = GPUCore(5, 4, dt=dt, device='cuda', plate_solver='fused_cuda', hardware=hw, elements_per_strip=elements)
    for c in (ref, fus):
        c.target.fill_(1.1)
        c.yaw.copy_(torch.tensor([[0.35, -0.3, 0.25, -0.2]] * 4, device='cuda', dtype=torch.float64))
    ref.advance_steps(1)
    fus.advance_steps(1)
    pos_err1 = (ref.x[..., :3] - fus.x[..., :3]).abs().max().item()
    vel_err1 = (ref.v - fus.v).abs().max().item()
    print(f"[E={elements:2d}] FP64 Parity (  1 step ): max |dx| = {pos_err1:.3e} m | max |dv| = {vel_err1:.3e} m/s")
    ref.advance_steps(199)
    fus.advance_steps(199)
    pos_err = (ref.x[..., :3] - fus.x[..., :3]).abs().max().item()
    vel_err = (ref.v - fus.v).abs().max().item()
    qdot = (ref.x[..., 3:] * fus.x[..., 3:]).sum(-1).abs().clamp_max(1.0)
    ang_err = (2.0 * torch.acos(qdot)).max().item()
    print(
        f"[E={elements:2d}] FP64 Parity (200 steps): max |dx| = {pos_err:.3e} m | "
        f"max |dv| = {vel_err:.3e} m/s | max |dq| = {ang_err:.3e} rad | failed={bool(fus.failed)}"
    )
    assert (
        pos_err1 < 1e-10
        and vel_err1 < 1e-5
        and pos_err < 1e-9
        and vel_err < 1e-4
        and not bool(ref.failed)
        and not bool(fus.failed)
    ), f"Parity check failed for E={elements}: pos_err={pos_err:.3e}, vel_err={vel_err:.3e}"
    del ref, fus
    torch.cuda.empty_cache()


def _bench_solver(
    elements: int,
    solver: str,
    batch: int,
    dt: float,
    substeps: int,
    iters: int,
    hw: HardwareConfig,
    baseline_sps: float | None = None,
) -> float:
    torch.cuda.reset_peak_memory_stats()
    core = GPUCore(
        5, batch, dt=dt, device='cuda', plate_solver=solver, hardware=hw, elements_per_strip=elements,
    )
    if solver == 'fused_cuda':
        graph_steps = 40 if (substeps % 40 == 0) else substeps
        core.capture(graph_steps)
    else:
        core.capture(4)
    core.target.fill_(1.0)
    core.yaw.fill_(0.25)
    # Warmup
    core.advance_steps(40 if (substeps % 40 == 0) else substeps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        core.advance_steps(substeps)
    torch.cuda.synchronize()
    dt_wall = time.perf_counter() - t0
    sps = (batch * iters) / dt_wall
    ms_per_substep = dt_wall / (iters * substeps) * 1000.0
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    ref_sps = sps if baseline_sps is None else baseline_sps
    speedup = sps / ref_sps
    print(
        f"E={elements:2d} | {solver:10s} | B={batch:4d} dt={dt*1000:.2f}ms | "
        f"{ms_per_substep:6.3f} ms/substep | {sps:8.1f} policy-steps/s | "
        f"{speedup:6.2f}x vs E=16 segment_lu | VRAM={vram_gb:.2f} GB"
    )
    del core
    torch.cuda.empty_cache()
    return sps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--dt', type=float, default=0.0005)
    p.add_argument('--substeps', type=int, default=200)  # 0.1 s = 1 policy step at 10 Hz
    p.add_argument('--iters', type=int, default=5)
    args = p.parse_args()

    hw = HardwareConfig.ds3218_default()

    # 1. Exact FP64 parity checks across both <16> and <8> unrolled templates (1 step + 200 substeps)
    for elements in (16, 8):
        _verify_parity(elements, args.dt, hw)

    # 2. Dual-mesh throughput benchmark (E=16 and E=8 vs E=16 segment_lu baseline)
    base_sps: float | None = None
    for elements in (16, 8):
        for solver in ('segment_lu', 'fused_cuda'):
            sps = _bench_solver(
                elements=elements,
                solver=solver,
                batch=args.batch,
                dt=args.dt,
                substeps=args.substeps,
                iters=args.iters,
                hw=hw,
                baseline_sps=base_sps,
            )
            if elements == 16 and solver == 'segment_lu':
                base_sps = sps


if __name__ == '__main__':
    main()

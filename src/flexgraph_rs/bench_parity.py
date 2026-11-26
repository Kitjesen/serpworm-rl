"""Concise (< 5s) Python-vs-Rust FP64 parity & CPU speedup benchmark via `flexgraph_rs`."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from flexgraph_rs import (
    FlexGraphRsBatchSolver,
    FlexGraphRsSolver,
    build_serpworm_envelope_json,
    check_serpworm_gpucore_parity,
)
from serpworm.config import HardwareConfig
from serpworm.physics.gpu_core import GPUCore


def main() -> None:
    torch.set_num_threads(1)
    t_all = time.perf_counter()

    # 1. Exact FP64 parity checks against Python GPUCore (segment_lu CPU)
    # Use 40 steps @ dt=0.5 ms (20 ms > 14.3 ms touchdown) so ground-contact cases test ACTIVE wheel contact!
    parity_cases = [
        check_serpworm_gpucore_parity(
            case_name="SerpWorm-3 free-flight (bench_exp128, E=8)",
            n_segments=3,
            elements_per_strip=8,
            steps=15,
            dt=0.0005,
            ground_contact=False,
            hardware=HardwareConfig.bench_exp128_calibrated(),
            tendon_targets=[0.9, 0.3, 0.5, 1.1, 0.7, 0.2],
            yaw_targets=[0.25, -0.30],
        ),
        check_serpworm_gpucore_parity(
            case_name="SerpWorm-2 Kelvin active wheels (ds3218, E=6, 40 steps)",
            n_segments=2,
            elements_per_strip=6,
            steps=40,
            dt=0.0005,
            ground_contact=True,
            hardware=HardwareConfig.ds3218_default(),
            tendon_targets=[0.6, 0.6, 0.4, 0.8],
            yaw_targets=[0.15],
        ),
        check_serpworm_gpucore_parity(
            case_name="SerpWorm-2 ARPC hard implicit active (E=6, 40 steps)",
            n_segments=2,
            elements_per_strip=6,
            steps=40,
            dt=0.0005,
            ground_contact=True,
            hardware=HardwareConfig.ds3218_default().with_terrain("hard", implicit_contact=True),
            tendon_targets=[0.6, 0.6, 0.4, 0.8],
            yaw_targets=[0.15],
        ),
    ]

    print("# Python GPUCore vs Rust flexgraph-rs FP64 Parity")
    print("| Case | Max Pos Err (m) | Max Quat Err | Max Tension Err (N) | Max Support (N) | Support Err (N) |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for r in parity_cases:
        print(
            f"| {r.case_name} | {r.max_pos_err_m:.3e} | {r.max_quat_err:.3e} | "
            f"{r.max_tension_err_n:.3e} | {r.max_support_n:.3f} | {r.max_support_err_n:.3e} |"
        )
        assert r.max_pos_err_m < 1e-10, f"Position parity failed on {r.case_name}: {r.max_pos_err_m}"
        if r.ground_contact:
            assert r.max_support_n > 0.5, f"Expected active wheel contact on {r.case_name}, got {r.max_support_n}"

    # 2. Controlled CPU throughput comparison: SerpWorm-5 (E=16, 3660 DoFs, dt=0.5ms)
    n_seg, elems, dt = 5, 16, 0.0005
    py_steps = 15
    rs_steps = 60
    batch_b = 16
    batch_steps = 20

    py_core = GPUCore(
        count=n_seg,
        batch=1,
        dt=dt,
        device="cpu",
        dtype=torch.float64,
        plate_solver="segment_lu",
        ground_contact=True,
        elements_per_strip=elems,
    )
    py_core.target[0, :] = torch.tensor([0.7, 0.4, 0.5, 0.8, 0.6, 0.6, 0.4, 0.7, 0.5, 0.3], dtype=torch.float64)
    py_core.yaw[0, :] = torch.tensor([0.15, -0.15, 0.10, -0.10], dtype=torch.float64)
    py_core.physics()  # warmup
    t0 = time.perf_counter()
    for _ in range(py_steps):
        py_core.physics()
    py_sps = py_steps / (time.perf_counter() - t0)

    rs_single = FlexGraphRsSolver.for_serpworm(
        n_segments=n_seg,
        elements_per_strip=elems,
        ground_contact=True,
        dt=dt,
        contact_model="kelvin",
        implicit_contact=False,
    )
    rs_single.set_targets([0.7, 0.4, 0.5, 0.8, 0.6, 0.6, 0.4, 0.7, 0.5, 0.3, 0.15, -0.15, 0.10, -0.10])
    rs_single.step(dt=dt, steps=2)  # warmup
    t1 = time.perf_counter()
    rs_single.step(dt=dt, steps=rs_steps)
    rs_1t_sps = rs_steps / (time.perf_counter() - t1)
    rs_single.close()

    env_json = build_serpworm_envelope_json(
        n_segments=n_seg,
        elements_per_strip=elems,
        ground_contact=True,
        dt=dt,
    )
    cfg = {
        "solver_preset": "serpworm_kelvin_parity",
        "world_sdf": "kelvin_floor",
    }
    rs_batch = FlexGraphRsBatchSolver.from_json(env_json, batch_size=batch_b, dt=dt, config=cfg)
    acts = np.tile(
        np.array([0.7, 0.4, 0.5, 0.8, 0.6, 0.6, 0.4, 0.7, 0.5, 0.3, 0.15, -0.15, 0.10, -0.10], dtype=np.float64),
        (batch_b, 1),
    )
    rs_batch.step(substeps=2, actions=acts)  # warmup
    t2 = time.perf_counter()
    rs_batch.step(substeps=batch_steps, actions=acts)
    rs_batch_sps = (batch_b * batch_steps) / (time.perf_counter() - t2)
    rs_batch.close()

    total_s = time.perf_counter() - t_all
    print()
    print("# SerpWorm-5 (E=16, 3660 DoFs, FP64) CPU Speedup vs Python GPUCore (segment_lu)")
    print(f"- Python `GPUCore` (CPU 1-thread `segment_lu`): **{py_sps:.1f} steps/s**")
    print(f"- Rust `fg-ffi` (CPU 1-thread): **{rs_1t_sps:.1f} steps/s** (**{rs_1t_sps / py_sps:.2f}x** vs Python)")
    print(
        f"- Rust `fg-ffi` Rayon batch (`B={batch_b}`): **{rs_batch_sps:.1f} env-steps/s** "
        f"(**{rs_batch_sps / py_sps:.2f}x** vs Python)"
    )
    print(f"- Total wall-clock time: {total_s:.3f} s")

    summary = {
        "parity": [r.__dict__ for r in parity_cases],
        "serpworm5_py_cpu_sps": py_sps,
        "serpworm5_rust_1t_sps": rs_1t_sps,
        "serpworm5_rust_rayon_sps": rs_batch_sps,
        "speedup_1t": rs_1t_sps / py_sps,
        "speedup_rayon": rs_batch_sps / py_sps,
        "total_wall_s": total_s,
    }
    print("PARITY_BENCH_JSON=" + json.dumps(summary))


if __name__ == "__main__":
    main()

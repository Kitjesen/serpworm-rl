"""Exact FP64 parity verification between native Rust `fg-ffi` and `serpworm.physics.gpu_core.GPUCore`."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from flexgraph_rs.solver import FlexGraphRsSolver


@dataclass(frozen=True)
class ParityReport:
    case_name: str
    n_segments: int
    elements_per_strip: int
    steps: int
    dt: float
    ground_contact: bool
    contact_model: str
    implicit_contact: bool
    max_pos_err_m: float
    max_quat_err: float
    max_plate_vel_err: float
    max_vel_err: float
    max_tension_err_n: float
    max_support_err_n: float
    max_support_n: float
    rust_ms: float
    python_ms: float
    speedup: float


def _quat_hemisphere_err(q_ref: np.ndarray, q_rs: np.ndarray) -> float:
    dots = np.sum(q_ref * q_rs, axis=-1, keepdims=True)
    aligned = np.where(dots < 0.0, -q_rs, q_rs)
    return float(np.max(np.abs(q_ref - aligned)))


def check_serpworm_gpucore_parity(
    *,
    case_name: str = "serpworm_parity",
    n_segments: int = 3,
    elements_per_strip: int = 8,
    steps: int = 15,
    dt: float = 0.0005,
    ground_contact: bool = True,
    hardware: Any = None,
    tendon_targets: list[float] | None = None,
    yaw_targets: list[float] | None = None,
) -> ParityReport:
    """Run identical `SerpWorm-N` trajectories in Python `GPUCore` (FP64 CPU `segment_lu`)
    and native Rust `FlexGraphRsSolver`, returning exact element-wise FP64 discrepancies and timings.
    """
    import torch
    from serpworm.physics.gpu_core import GPUCore

    torch.set_num_threads(1)

    contact_model = str(getattr(hardware, "contact_model", "kelvin")) if hardware is not None else "kelvin"
    implicit_contact = bool(getattr(hardware, "implicit_contact", False)) if hardware is not None else False

    ref = GPUCore(
        count=n_segments,
        batch=1,
        dt=dt,
        device="cpu",
        dtype=torch.float64,
        plate_solver="segment_lu",
        ground_contact=ground_contact,
        hardware=hardware,
        elements_per_strip=elements_per_strip,
    )
    rs = FlexGraphRsSolver.for_serpworm(
        n_segments=n_segments,
        elements_per_strip=elements_per_strip,
        hardware=hardware,
        ground_contact=ground_contact,
        dt=dt,
        contact_model=contact_model,
        implicit_contact=implicit_contact,
    )

    n_t = 2 * n_segments
    n_y = max(0, n_segments - 1)
    t_vals = (
        tendon_targets
        if tendon_targets is not None
        else [0.4 + 0.15 * ((i * 3) % 5) for i in range(n_t)]
    )
    y_vals = (
        yaw_targets
        if yaw_targets is not None
        else [0.18 * (1.0 if i % 2 == 0 else -1.0) for i in range(n_y)]
    )

    ref.target[0, :] = torch.tensor(t_vals, dtype=torch.float64)
    if n_y > 0:
        ref.yaw[0, :] = torch.tensor(y_vals, dtype=torch.float64)
    rs.set_targets(t_vals + y_vals)

    t0 = time.perf_counter()
    for _ in range(steps):
        ref.physics()
    py_ms = (time.perf_counter() - t0) * 1e3

    t1 = time.perf_counter()
    diag = rs.step(dt=dt, steps=steps)
    rs_ms = (time.perf_counter() - t1) * 1e3
    assert not diag.failed, f"Rust solver failed during parity case {case_name}"

    ref_x = ref.x[0].cpu().numpy()
    ref_v = ref.v[0].cpu().numpy()
    ref_ten = ref.tension[0].reshape(-1).cpu().numpy()
    ref_sup = ref.support[0].reshape(-1).cpu().numpy()

    rs_x = np.array(rs.poses, copy=True)
    rs_v = np.array(rs.vels, copy=True)
    rs_ten = np.array(rs.tensions, copy=True)
    rs_sup = np.array(rs.contact_normals, copy=True)
    rs.close()

    n_plates = 2 * n_segments
    max_pos_err = float(np.max(np.abs(ref_x[:, :3] - rs_x[:, :3])))
    max_quat_err = _quat_hemisphere_err(ref_x[:, 3:], rs_x[:, 3:])
    max_plate_vel_err = float(np.max(np.abs(ref_v[:n_plates] - rs_v[:n_plates])))
    max_vel_err = float(np.max(np.abs(ref_v - rs_v)))
    max_ten_err = float(np.max(np.abs(ref_ten - rs_ten))) if ref_ten.size else 0.0
    max_sup_err = (
        float(np.max(np.abs(ref_sup - rs_sup)))
        if ground_contact and ref_sup.size and rs_sup.size
        else 0.0
    )
    max_sup = float(np.max(rs_sup)) if ground_contact and rs_sup.size else 0.0

    return ParityReport(
        case_name=case_name,
        n_segments=n_segments,
        elements_per_strip=elements_per_strip,
        steps=steps,
        dt=dt,
        ground_contact=ground_contact,
        contact_model=contact_model,
        implicit_contact=implicit_contact,
        max_pos_err_m=max_pos_err,
        max_quat_err=max_quat_err,
        max_plate_vel_err=max_plate_vel_err,
        max_vel_err=max_vel_err,
        max_tension_err_n=max_ten_err,
        max_support_err_n=max_sup_err,
        max_support_n=max_sup,
        rust_ms=rs_ms,
        python_ms=py_ms,
        speedup=py_ms / max(rs_ms, 1e-6),
    )

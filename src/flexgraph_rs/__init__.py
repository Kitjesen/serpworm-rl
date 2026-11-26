"""Zero-dependency Python ctypes bindings to the pure-Rust `flexgraph-rs` (`fg-ffi`) engine."""
from __future__ import annotations

from flexgraph_rs._ffi import (
    FgEnergyMomentum,
    FgStepDiag,
    FgWorldInfo,
    SCHUR_ROUTE_NAMES,
    find_native_library,
    load_native_lib,
)
from flexgraph_rs.parity import ParityReport, check_serpworm_gpucore_parity
from flexgraph_rs.solver import (
    AdjointGradResult,
    EnergyMomentum,
    FlexGraphRsBatchSolver,
    FlexGraphRsSolver,
    StepDiag,
    WorldInfo,
    build_serpworm_envelope_json,
)

__all__ = [
    "AdjointGradResult",
    "EnergyMomentum",
    "FgEnergyMomentum",
    "FgStepDiag",
    "FgWorldInfo",
    "FlexGraphRsBatchSolver",
    "FlexGraphRsSolver",
    "ParityReport",
    "SCHUR_ROUTE_NAMES",
    "StepDiag",
    "WorldInfo",
    "build_serpworm_envelope_json",
    "check_serpworm_gpucore_parity",
    "find_native_library",
    "load_native_lib",
]

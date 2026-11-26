"""Native Rust FFI backend bridge (``fg_ffi``) with graceful Python ``GraphSchurSolver`` fallback.

Detects and loads the compiled Rust ``fg-ffi`` shared library (``fg_ffi.dll`` on Windows,
``libfg_ffi.so`` on Linux, ``libfg_ffi.dylib`` on macOS) via ``ctypes`` (or optional PyO3
extension module ``fg_ffi`` / ``flexgraph_rs``) matching the exact C-ABI exported by
``rust/flexgraph-rs/crates/fg-ffi/src/lib.rs``:

1. **Vectorized Rayon Batch Handle (``FgBatchHandle``)**:
   - ``fg_batch_create_from_json``, ``fg_batch_create_from_json_with_config``, ``fg_batch_create_archetype``
   - ``fg_batch_dims``, ``fg_batch_size``, ``fg_batch_total_nodes``
   - ``fg_batch_reset_mask``, ``fg_batch_step_into``, ``fg_batch_get_all_nodes_into``, ``fg_batch_destroy``
2. **Single-Environment Handle (``FgWorldHandle`` / ``FgSimHandle``)**:
   - ``fg_world_create_from_json``, ``fg_world_create_archetype``, ``fg_world_reset``
   - ``fg_world_set_targets``, ``fg_world_step``, ``fg_world_get_state``, ``fg_world_get_info``
   - ``fg_world_poses_ptr``, ``fg_world_vels_ptr``, ``fg_world_tensions_ptr``, ``fg_world_contact_normals_ptr``
   - ``fg_world_free``, ``fg_last_error``, ``fg_string_free``

When the Rust library is unavailable, incompatible, or when running batched CUDA physics on GPU,
``RustGraphSolverBackend`` falls back cleanly with zero crashes to the FP64 PyTorch ``GraphSchurSolver``.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from flexgraph.contact import ContactState
from flexgraph.graph import RigidFlexGraph
from flexgraph.sdf_world import ComposableSDFWorld, HeightFieldSDF
from flexgraph.solver import GraphSchurSolver

__all__ = [
    'ALL_V2_C_ABI_SYMBOLS',
    'AdjointGradResult',
    'EnergyMomentum',
    'FFIBindings',
    'FgEnergyMomentum',
    'FgStepDiag',
    'FgWorldInfo',
    'FlexGraphRsBatchSolver',
    'FlexGraphRsSolver',
    'OPTIONAL_C_ABI_SYMBOLS',
    'REQUIRED_BATCH_C_ABI_SYMBOLS',
    'REQUIRED_C_ABI_SYMBOLS',
    'REQUIRED_WORLD_C_ABI_SYMBOLS',
    'RustGraphSolverBackend',
    'StepDiag',
    'V2_C_ABI_SYMBOLS',
    'WorldInfo',
    'benchmark_backend_parity',
    'create_native_world',
    'create_solver',
    'create_world_from_mjcf',
    'create_world_from_ros_urdf',
    'create_world_with_corrugated_pipe_mesh',
    'discover_ffi_candidates',
    'graph_to_rust_ir_dict',
    'graph_to_rust_ir_json',
    'load_flexgraph_ffi',
    'world_to_ffi_config_dict',
    'world_to_ffi_config_json',
]

try:
    from flexgraph_rs import (
        AdjointGradResult,
        EnergyMomentum,
        FlexGraphRsBatchSolver,
        FlexGraphRsSolver,
        StepDiag,
        WorldInfo,
    )
except Exception:  # pragma: no cover - optional sibling package
    AdjointGradResult = None  # type: ignore[assignment]
    EnergyMomentum = None  # type: ignore[assignment]
    FlexGraphRsBatchSolver = None  # type: ignore[assignment]
    FlexGraphRsSolver = None  # type: ignore[assignment]
    StepDiag = None  # type: ignore[assignment]
    WorldInfo = None  # type: ignore[assignment]

# Exact C-ABI structs matching #[repr(C)] definitions in rust/flexgraph-rs/crates/fg-ffi/src/lib.rs
class FgWorldInfo(ctypes.Structure):
    _fields_ = [
        ('num_bodies', ctypes.c_size_t),
        ('num_free_bodies', ctypes.c_size_t),
        ('num_beam_nodes', ctypes.c_size_t),
        ('total_nodes', ctypes.c_size_t),
        ('num_beams', ctypes.c_size_t),
        ('num_tendons', ctypes.c_size_t),
        ('num_tendon_actuators', ctypes.c_size_t),
        ('num_joint_actuators', ctypes.c_size_t),
        ('num_total_actuators', ctypes.c_size_t),
        ('num_contact_sites', ctypes.c_size_t),
        ('schur_route', ctypes.c_int32),
        ('cyclomatic_number', ctypes.c_size_t),
        ('step_count', ctypes.c_uint64),
        ('sim_time', ctypes.c_double),
        ('failed', ctypes.c_uint8),
    ]


class FgStepDiag(ctypes.Structure):
    _fields_ = [
        ('step_index', ctypes.c_uint64),
        ('sim_time', ctypes.c_double),
        ('newton_iters', ctypes.c_size_t),
        ('residual_inf', ctypes.c_double),
        ('residual_l2', ctypes.c_double),
        ('line_search_alpha', ctypes.c_double),
        ('ccd_toi', ctypes.c_double),
        ('active_site_contacts', ctypes.c_size_t),
        ('active_beam_contacts', ctypes.c_size_t),
        ('active_self_contacts', ctypes.c_size_t),
        ('lcp_iters', ctypes.c_size_t),
        ('lcp_residual', ctypes.c_double),
        ('max_quat_norm_err', ctypes.c_double),
        ('failed', ctypes.c_uint8),
    ]


class FgEnergyMomentum(ctypes.Structure):
    _fields_ = [
        ('kinetic_energy', ctypes.c_double),
        ('elastic_energy', ctypes.c_double),
        ('actuation_potential_energy', ctypes.c_double),
        ('gravity_potential_energy', ctypes.c_double),
        ('total_energy', ctypes.c_double),
        ('linear_momentum', ctypes.c_double * 3),
        ('angular_momentum', ctypes.c_double * 3),
    ]


REQUIRED_BATCH_C_ABI_SYMBOLS: tuple[str, ...] = (
    'fg_batch_create_from_json_with_config',
    'fg_batch_dims',
    'fg_batch_size',
    'fg_batch_total_nodes',
    'fg_batch_reset_mask',
    'fg_batch_step_into',
    'fg_batch_get_all_nodes_into',
    'fg_batch_destroy',
)

REQUIRED_WORLD_C_ABI_SYMBOLS: tuple[str, ...] = (
    'fg_world_create_from_json',
    'fg_world_reset',
    'fg_world_set_targets',
    'fg_world_step',
    'fg_world_get_state',
    'fg_world_get_info',
    'fg_world_free',
)

# Backwards-compatible alias referencing the primary batch C-ABI contract
REQUIRED_C_ABI_SYMBOLS: tuple[str, ...] = REQUIRED_BATCH_C_ABI_SYMBOLS

V2_C_ABI_SYMBOLS: tuple[str, ...] = (
    'fg_world_create_from_ros_urdf',
    'fg_world_create_from_mjcf',
    'fg_world_create_with_corrugated_pipe_mesh',
    'fg_world_adjoint_grad',
    'fg_world_get_diff_param',
    'fg_world_set_diff_param',
)

OPTIONAL_C_ABI_SYMBOLS: tuple[str, ...] = (
    'fg_last_error',
    'fg_string_free',
    'fg_batch_create_from_json',
    'fg_batch_create_archetype',
    'fg_world_create_from_yaml',
    'fg_world_create_from_urdf_flex',
    'fg_world_create_archetype',
    'fg_world_set_tendon_targets',
    'fg_world_set_joint_targets',
    'fg_world_poses_ptr',
    'fg_world_vels_ptr',
    'fg_world_tensions_ptr',
    'fg_world_contact_normals_ptr',
    'fg_world_get_energy_momentum',
    'fg_world_export_graph_json',
) + V2_C_ABI_SYMBOLS

ALL_V2_C_ABI_SYMBOLS: tuple[str, ...] = tuple(
    sorted(set(REQUIRED_BATCH_C_ABI_SYMBOLS) | set(REQUIRED_WORLD_C_ABI_SYMBOLS) | set(OPTIONAL_C_ABI_SYMBOLS))
)


@dataclass(frozen=True)
class FFIBindings:
    """Resolved native ``fg_ffi`` binding handle and diagnostic metadata."""

    available: bool
    mode: str  # 'cdylib' | 'pyo3' | 'cdylib_incomplete' | 'unavailable'
    library_path: Path | None = None
    abi_version: int | None = None
    exported_symbols: tuple[str, ...] = ()
    missing_symbols: tuple[str, ...] = ()
    supports_batch_handle: bool = False
    supports_world_handle: bool = False
    supports_v2: bool = False
    reason: str = ''
    cdll: Any = None
    pyo3_module: Any = None

    def last_error(self) -> str:
        if self.cdll is not None and hasattr(self.cdll, 'fg_last_error'):
            try:
                raw = self.cdll.fg_last_error()
                return raw.decode('utf-8', errors='replace') if raw else ''
            except Exception:
                return ''
        return ''

    def summary_dict(self) -> dict[str, Any]:
        return {
            'available': self.available,
            'mode': self.mode,
            'library_path': str(self.library_path) if self.library_path is not None else None,
            'abi_version': self.abi_version,
            'supports_batch_handle': self.supports_batch_handle,
            'supports_world_handle': self.supports_world_handle,
            'supports_v2': self.supports_v2,
            'exported_symbols': list(self.exported_symbols),
            'missing_symbols': list(self.missing_symbols),
            'reason': self.reason,
        }


def _platform_lib_names() -> tuple[str, ...]:
    if sys.platform.startswith('win'):
        return ('fg_ffi.dll', 'libfg_ffi.dll')
    if sys.platform == 'darwin':
        return ('libfg_ffi.dylib', 'fg_ffi.dylib')
    return ('libfg_ffi.so', 'fg_ffi.so')


def discover_ffi_candidates(extra_paths: list[str | Path] | None = None) -> list[Path]:
    """Return ordered candidate paths where compiled ``fg_ffi`` shared libraries may live."""
    candidates: list[Path] = []
    if extra_paths:
        for p in extra_paths:
            candidates.append(Path(p))

    for env_key in ('FLEXGRAPH_FFI_LIB', 'FLEXGRAPH_RS_LIB'):
        env_override = os.environ.get(env_key, '').strip()
        if env_override:
            candidates.append(Path(env_override))

    repo_root = Path(__file__).resolve().parents[2]
    target_root = repo_root / 'rust' / 'flexgraph-rs' / 'target'
    lib_names = _platform_lib_names()
    search_dirs = (
        target_root / 'release',
        target_root / 'release' / 'deps',
        target_root / 'debug',
        target_root / 'debug' / 'deps',
        Path(__file__).resolve().parent,
        repo_root / 'build',
    )
    for d in search_dirs:
        for name in lib_names:
            candidates.append(d / name)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for c in candidates:
        rc = c.resolve() if c.exists() else c
        if rc not in seen:
            seen.add(rc)
            ordered.append(c)
    return ordered


def _configure_cdll_signatures(lib: ctypes.CDLL, exported: set[str]) -> None:
    """Bind exact ctypes argtypes/restype matching all 36 symbols in ``fg-ffi/src/lib.rs``."""
    c_void_p = ctypes.c_void_p
    c_char_p = ctypes.c_char_p
    c_int = ctypes.c_int
    c_size_t = ctypes.c_size_t
    c_size_t_p = ctypes.POINTER(ctypes.c_size_t)
    c_double = ctypes.c_double
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_uint8 = ctypes.c_uint8
    c_uint8_p = ctypes.POINTER(ctypes.c_uint8)

    if 'fg_last_error' in exported:
        lib.fg_last_error.argtypes = []
        lib.fg_last_error.restype = c_char_p
    if 'fg_string_free' in exported:
        lib.fg_string_free.argtypes = [c_void_p]
        lib.fg_string_free.restype = None

    # --- Single-world handle (FgWorldHandle, lib.rs) ---
    for sym in (
        'fg_world_create_from_json',
        'fg_world_create_from_yaml',
        'fg_world_create_from_urdf_flex',
        'fg_world_create_archetype',
        'fg_world_create_from_ros_urdf',
        'fg_world_create_from_mjcf',
    ):
        if sym in exported:
            fn = getattr(lib, sym)
            fn.argtypes = [c_char_p, c_char_p]
            fn.restype = c_void_p
    if 'fg_world_create_with_corrugated_pipe_mesh' in exported:
        lib.fg_world_create_with_corrugated_pipe_mesh.argtypes = [
            c_char_p,
            c_char_p,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
        ]
        lib.fg_world_create_with_corrugated_pipe_mesh.restype = c_void_p
    if 'fg_world_reset' in exported:
        lib.fg_world_reset.argtypes = [c_void_p]
        lib.fg_world_reset.restype = c_int
    for sym in ('fg_world_set_targets', 'fg_world_set_tendon_targets', 'fg_world_set_joint_targets'):
        if sym in exported:
            fn = getattr(lib, sym)
            fn.argtypes = [c_void_p, c_double_p, c_size_t]
            fn.restype = c_int
    if 'fg_world_step' in exported:
        lib.fg_world_step.argtypes = [c_void_p, c_size_t, c_double, ctypes.POINTER(FgStepDiag)]
        lib.fg_world_step.restype = c_int
    if 'fg_world_get_state' in exported:
        lib.fg_world_get_state.argtypes = [c_void_p, c_double_p, c_size_t, c_double_p, c_size_t]
        lib.fg_world_get_state.restype = c_int
    for sym in ('fg_world_poses_ptr', 'fg_world_vels_ptr', 'fg_world_tensions_ptr', 'fg_world_contact_normals_ptr'):
        if sym in exported:
            fn = getattr(lib, sym)
            fn.argtypes = [c_void_p]
            fn.restype = c_double_p
    if 'fg_world_get_info' in exported:
        lib.fg_world_get_info.argtypes = [c_void_p, ctypes.POINTER(FgWorldInfo)]
        lib.fg_world_get_info.restype = c_int
    if 'fg_world_get_energy_momentum' in exported:
        lib.fg_world_get_energy_momentum.argtypes = [c_void_p, ctypes.POINTER(FgEnergyMomentum)]
        lib.fg_world_get_energy_momentum.restype = c_int
    if 'fg_world_export_graph_json' in exported:
        lib.fg_world_export_graph_json.argtypes = [c_void_p]
        lib.fg_world_export_graph_json.restype = c_void_p
    if 'fg_world_adjoint_grad' in exported:
        lib.fg_world_adjoint_grad.argtypes = [
            c_void_p,
            c_double_p,
            c_size_t,
            c_size_t,
            c_double,
            c_double_p,
            c_size_t,
            c_double,
            c_double,
            c_double,
            c_char_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_size_t,
        ]
        lib.fg_world_adjoint_grad.restype = c_int
    if 'fg_world_get_diff_param' in exported:
        lib.fg_world_get_diff_param.argtypes = [c_void_p, c_char_p, c_double_p]
        lib.fg_world_get_diff_param.restype = c_int
    if 'fg_world_set_diff_param' in exported:
        lib.fg_world_set_diff_param.argtypes = [c_void_p, c_char_p, c_double, c_uint8]
        lib.fg_world_set_diff_param.restype = c_int
    if 'fg_world_free' in exported:
        lib.fg_world_free.argtypes = [c_void_p]
        lib.fg_world_free.restype = None

    # --- Rayon vectorized batch handle (FgBatchHandle, lib.rs) ---
    if 'fg_batch_create_from_json' in exported:
        lib.fg_batch_create_from_json.argtypes = [c_char_p, c_size_t, c_double, c_size_t]
        lib.fg_batch_create_from_json.restype = c_void_p
    if 'fg_batch_create_from_json_with_config' in exported:
        lib.fg_batch_create_from_json_with_config.argtypes = [c_char_p, c_char_p, c_size_t, c_double, c_size_t]
        lib.fg_batch_create_from_json_with_config.restype = c_void_p
    if 'fg_batch_create_archetype' in exported:
        lib.fg_batch_create_archetype.argtypes = [c_char_p, c_char_p, c_size_t, c_double, c_size_t]
        lib.fg_batch_create_archetype.restype = c_void_p
    if 'fg_batch_dims' in exported:
        lib.fg_batch_dims.argtypes = [c_void_p, c_size_t_p, c_size_t_p, c_size_t_p]
        lib.fg_batch_dims.restype = c_int
    if 'fg_batch_size' in exported:
        lib.fg_batch_size.argtypes = [c_void_p]
        lib.fg_batch_size.restype = c_size_t
    if 'fg_batch_total_nodes' in exported:
        lib.fg_batch_total_nodes.argtypes = [c_void_p]
        lib.fg_batch_total_nodes.restype = c_size_t
    if 'fg_batch_reset_mask' in exported:
        lib.fg_batch_reset_mask.argtypes = [c_void_p, c_uint8_p]
        lib.fg_batch_reset_mask.restype = c_int
    if 'fg_batch_step_into' in exported:
        lib.fg_batch_step_into.argtypes = [
            c_void_p,
            c_size_t,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_uint8_p,
        ]
        lib.fg_batch_step_into.restype = c_int
    if 'fg_batch_get_all_nodes_into' in exported:
        lib.fg_batch_get_all_nodes_into.argtypes = [c_void_p, c_double_p, c_double_p]
        lib.fg_batch_get_all_nodes_into.restype = c_int
    if 'fg_batch_destroy' in exported:
        lib.fg_batch_destroy.argtypes = [c_void_p]
        lib.fg_batch_destroy.restype = None


def load_flexgraph_ffi(
    lib_path: str | Path | None = None,
    *,
    prefer_pyo3: bool = True,
) -> FFIBindings:
    """Discover and load native Rust ``fg_ffi`` bindings with zero-crash fallback metadata.

    Probes:
      1. Optional PyO3 extension module (``flexgraph_rs`` or ``fg_ffi``) when ``lib_path is None``
         and ``prefer_pyo3=True``.
      2. Compiled C-ABI shared library (``fg_ffi.dll`` / ``libfg_ffi.so`` / ``libfg_ffi.dylib``)
         from ``lib_path``, ``FLEXGRAPH_FFI_LIB``, ``FLEXGRAPH_RS_LIB``, or
         ``rust/flexgraph-rs/target/{release,debug}[/deps]``.
    """
    if lib_path is None and prefer_pyo3:
        for mod_name in ('flexgraph_rs', 'fg_ffi'):
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            world_cls = (
                getattr(mod, 'FlexGraphBatchWorld', None)
                or getattr(mod, 'FlexGraphWorld', None)
                or getattr(mod, 'RustWorld', None)
            )
            if world_cls is not None:
                abi_v = getattr(mod, 'ABI_VERSION', None)
                return FFIBindings(
                    available=True,
                    mode='pyo3',
                    library_path=Path(getattr(mod, '__file__', '')).resolve() if getattr(mod, '__file__', None) else None,
                    abi_version=int(abi_v) if abi_v is not None else None,
                    exported_symbols=(world_cls.__name__,),
                    missing_symbols=(),
                    supports_batch_handle=hasattr(mod, 'FlexGraphBatchWorld'),
                    supports_world_handle=True,
                    supports_v2=bool(abi_v and int(abi_v) >= 2),
                    reason='',
                    pyo3_module=mod,
                )

    candidates = [Path(lib_path)] if lib_path is not None else discover_ffi_candidates()
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        searched = ', '.join(str(p) for p in candidates[:4])
        return FFIBindings(
            available=False,
            mode='unavailable',
            library_path=None,
            missing_symbols=REQUIRED_C_ABI_SYMBOLS,
            reason=f'No compiled fg_ffi shared library found (searched: {searched})',
        )

    all_known = set(ALL_V2_C_ABI_SYMBOLS)
    last_reason = ''
    for cand in existing:
        try:
            cdll = ctypes.CDLL(str(cand))
        except OSError as exc:
            last_reason = f'Failed loading {cand}: {exc}'
            continue

        exported_set = {sym for sym in all_known if hasattr(cdll, sym)}
        _configure_cdll_signatures(cdll, exported_set)

        batch_missing = tuple(s for s in REQUIRED_BATCH_C_ABI_SYMBOLS if s not in exported_set)
        world_missing = tuple(s for s in REQUIRED_WORLD_C_ABI_SYMBOLS if s not in exported_set)
        has_batch = len(batch_missing) == 0
        has_world = len(world_missing) == 0
        has_v2 = all(s in exported_set for s in V2_C_ABI_SYMBOLS)

        if has_batch or has_world:
            return FFIBindings(
                available=True,
                mode='cdylib',
                library_path=cand.resolve(),
                abi_version=2 if has_v2 else 1,
                exported_symbols=tuple(sorted(exported_set)),
                missing_symbols=(),
                supports_batch_handle=has_batch,
                supports_world_handle=has_world,
                supports_v2=has_v2,
                reason='',
                cdll=cdll,
            )

        last_reason = (
            f'Loaded {cand.name} but required C-ABI symbols are missing '
            f'(missing batch: {", ".join(batch_missing)})'
        )
        return FFIBindings(
            available=False,
            mode='cdylib_incomplete',
            library_path=cand.resolve(),
            exported_symbols=tuple(sorted(exported_set)),
            missing_symbols=batch_missing,
            supports_batch_handle=False,
            supports_world_handle=False,
            supports_v2=False,
            reason=last_reason,
            cdll=cdll,
        )

    return FFIBindings(
        available=False,
        mode='unavailable',
        library_path=existing[0].resolve(),
        missing_symbols=REQUIRED_C_ABI_SYMBOLS,
        reason=last_reason or 'Unable to load candidate fg_ffi shared library',
    )


# ---------------------------------------------------------------------------
# Python RigidFlexGraph -> Rust fg-ir JSON & FfiConfigJson Serializers
# ---------------------------------------------------------------------------
def _pose7_to_se3_dict(p7: tuple[float, ...]) -> dict[str, list[float]]:
    return {
        'pos': [float(p7[0]), float(p7[1]), float(p7[2])],
        'rot': [float(p7[3]), float(p7[4]), float(p7[5]), float(p7[6])],
    }


def _diag_to_mat3(d: tuple[float, float, float]) -> list[list[float]]:
    return [
        [float(d[0]), 0.0, 0.0],
        [0.0, float(d[1]), 0.0],
        [0.0, 0.0, float(d[2])],
    ]


def graph_to_rust_ir_dict(
    graph: RigidFlexGraph,
    world: ComposableSDFWorld | None = None,
    *,
    dt: float = 0.001,
) -> dict[str, Any]:
    """Serialize a Python ``RigidFlexGraph`` into the exact Rust ``fg-ir`` serde JSON schema."""
    gravity = list(world.gravity_m_s2) if world is not None else [0.0, 0.0, -9.81]
    default_alpha = float(graph.beams[0].rayleigh_alpha) if graph.beams else 0.05
    default_beta = float(graph.beams[0].rayleigh_beta) if graph.beams else 0.002

    bodies = [
        {
            'id': idx,
            'name': b.name,
            'is_fixed': bool(b.fixed),
            'mass': float(b.mass_kg),
            'inertia': _diag_to_mat3(b.inertia_diag),
            'com_local': [0.0, 0.0, 0.0],
            'rest_pose': _pose7_to_se3_dict(b.rest_pose_7d),
            'collision_radius': float(b.collision_radius_m),
            'rayleigh_alpha': default_alpha,
            'tags': [],
        }
        for idx, b in enumerate(graph.bodies)
    ]

    beams = []
    for sid, b in enumerate(graph.beams):
        if b.section_type == 'rect':
            sec: dict[str, Any] = {
                'kind': 'rectangular',
                'width': float(b.width_m),
                'thickness': float(b.thickness_m),
            }
        elif b.section_type == 'circle':
            sec = {'kind': 'circular', 'radius': float(b.radius_m)}
        else:
            sec = {
                'kind': 'general',
                'area': float(b.area_m2 or 1e-6),
                'iy': float(b.iy_m4 or 1e-12),
                'iz': float(b.iz_m4 or 1e-12),
                'j_torsion': float(b.j_m4 or 1e-12),
                'shear_ky': 5.0 / 6.0,
                'shear_kz': 5.0 / 6.0,
            }
        beams.append(
            {
                'id': sid,
                'name': b.name,
                'body_a': int(b.body_a),
                'clamp_a': _pose7_to_se3_dict(b.clamp_a_7d),
                'body_b': None if int(b.body_b) < 0 else int(b.body_b),
                'clamp_b': None if int(b.body_b) < 0 else _pose7_to_se3_dict(b.clamp_b_7d),
                'n_elements': int(b.n_elem),
                'formulation': 'corotational_euler_bernoulli',
                'section': sec,
                'material': {
                    'youngs_modulus': float(b.youngs_modulus_pa),
                    'poisson_ratio': float(b.poisson_ratio),
                    'density': float(b.density_kg_m3),
                    'rayleigh_alpha': float(b.rayleigh_alpha),
                    'rayleigh_beta': float(b.rayleigh_beta),
                },
                'thin_shell_compat': bool(b.thin_shell_compat),
                'prestress_frac': float(b.prestress_frac),
                'centerline_rest_poses': [_pose7_to_se3_dict(p) for p in b.node_poses_7d],
                'element_lengths': [float(x) for x in b.element_lengths_m],
                'element_transforms': None,
                'interior_node_contact_radius': float(b.contact_node_radius_m) if b.enable_node_contact else None,
                'enable_self_contact': False,
            }
        )

    tendons = []
    for tid, ten in enumerate(graph.tendons):
        if ten.drive_mode == 'crank':
            dmode: dict[str, Any] = {
                'mode': 'eccentric_crank',
                'horn_radius': float(ten.horn_radius_m),
                'guide_dist': float(ten.guide_dist_m),
            }
        elif ten.drive_mode == 'legacy_wrap':
            dmode = {
                'mode': 'legacy_wrap',
                'horn_radius': float(ten.horn_radius_m),
                'guide_dist': float(ten.guide_dist_m),
            }
        elif ten.drive_mode == 'capstan':
            dmode = {'mode': 'capstan', 'drum_radius': float(ten.horn_radius_m)}
        else:
            dmode = {'mode': 'linear'}

        act_meta = graph.actuators[ten.actuator_idx] if 0 <= ten.actuator_idx < len(graph.actuators) else None
        angle_limits = [float(act_meta.min_cmd), float(act_meta.max_cmd)] if act_meta is not None else [0.0, 3.141592653589793]
        tendons.append(
            {
                'id': tid,
                'name': ten.name,
                'actuator_id': int(ten.actuator_idx),
                'waypoints': [
                    {'body_id': int(bi), 'local_pos': [float(v) for v in lp], 'wrap_angle_rad': 0.0}
                    for bi, lp in ten.waypoints
                ],
                'drive_mode': dmode,
                'rest_length': float(ten.rest_length_m),
                'slack': float(ten.slack_m),
                'stiffness': float(ten.stiffness_n_m),
                'damping': float(ten.damping_ns_m),
                'capstan_mu_theta': float(ten.capstan_mu_theta),
                'stick_slip_enabled': False,
                'servo_torque_limit': float(ten.torque_limit_nm),
                'servo_speed_limit': float(ten.speed_limit_rad_s),
                'servo_deadband': float(ten.deadband_rad),
                'angle_limits': angle_limits,
            }
        )

    joints = []
    for jid, jt in enumerate(graph.joints):
        if jt.joint_type == 'stop_gap':
            jtype: dict[str, Any] = {
                'kind': 'axial_stop_gap',
                'axis_local_a': [-1.0, 0.0, 0.0],
                'min_gap': float(jt.stop_gap_m),
                'stiffness': float(jt.k_stop),
            }
        else:
            jtype = {
                'kind': 'revolute',
                'axis_local_a': [float(v) for v in jt.axis_local],
                'anchor_a': [float(v) for v in jt.anchor_a],
                'anchor_b': [0.0, 0.0, 0.0],
                'pos_stiffness': float(jt.k_pos),
                'pos_damping': float(jt.c_pos),
                'align_stiffness': float(jt.k_align),
                'align_damping': float(jt.c_align),
            }
        joints.append(
            {
                'id': jid,
                'name': jt.name,
                'body_a': int(jt.body_a),
                'body_b': int(jt.body_b),
                'joint_type': jtype,
                'actuator_id': None if int(jt.actuator_idx) < 0 else int(jt.actuator_idx),
                'kp': float(jt.kp),
                'kd': float(jt.kd),
                'torque_limit': float(jt.tau_max),
                'speed_limit': float(jt.omega_max),
                'static_friction': float(jt.tau_static_fric),
                'deadband': 0.0,
                'limits': None,
                'implicit_coupling': bool(jt.implicit_coupling),
            }
        )

    contact_sites = []
    for cid, s in enumerate(graph.contact_sites):
        if s.contact_mode == 'ratchet_wheel':
            mu_lat_ratio = float(s.mu_lat) / max(float(s.mu_fwd), 1e-9)
            cmode: dict[str, Any] = {
                'mode': 'ratchet_wheel',
                'wheel_mass': float(s.wheel_mass_kg),
                'rolling_resistance': float(s.rolling),
                'backlash_rad': float(s.backlash_rad),
                'pivot_mu': float(s.pivot_mu),
                'mu_fwd': float(s.mu_fwd),
                'mu_lat_ratio': mu_lat_ratio,
                'static_mu_ratio': float(s.static_mu_ratio),
                'axle_axis_local': [float(v) for v in s.axle_local],
            }
        elif s.contact_mode == 'anisotropic_scale':
            cmode = {
                'mode': 'anisotropic_scale',
                'mu_fwd': float(s.mu_fwd),
                'mu_rev': float(s.mu_rev),
                'mu_lat': float(s.mu_lat),
                'pivot_mu': float(s.pivot_mu),
                'static_mu_ratio': float(s.static_mu_ratio),
                'fwd_axis_local': [float(v) for v in s.fwd_local],
            }
        else:
            cmode = {
                'mode': 'isotropic_pad',
                'mu': float(s.mu_fwd),
                'pivot_mu': float(s.pivot_mu),
                'static_mu_ratio': float(s.static_mu_ratio),
                'sdf_only': False,
            }
        contact_sites.append(
            {
                'id': cid,
                'name': s.name,
                'attachment': {'attach_type': 'rigid_body', 'body_id': int(s.owner_body)},
                'local_pos': [float(v) for v in s.local_pos],
                'local_frame': [0.0, 0.0, 0.0, 1.0],
                'radius': float(s.radius_m),
                'patch_radius': float(s.patch_radius_m),
                'mode': cmode,
                'material_profile_id': 0,
            }
        )

    sym_dict: dict[str, Any] = {'kind': 'trivial', 'elements': []}
    if graph.symmetry is not None:
        sname = graph.symmetry.name.lower()
        if 'z2xz2' in sname:
            skind: Any = 'z2x_z2_bidirectional'
        elif 'z2' in sname:
            skind = 'z2_bilateral'
        elif 'd4' in sname:
            skind = 'd4_dihedral'
        elif 'c4' in sname:
            skind = 'c4_cyclic'
        else:
            skind = {'custom': graph.symmetry.name}
        elems = []
        for idx, (ap, asg) in enumerate(zip(graph.symmetry.act_perms, graph.symmetry.act_signs)):
            op = graph.symmetry.obs_perms[idx] if idx < len(graph.symmetry.obs_perms) else []
            osg = graph.symmetry.obs_signs[idx] if idx < len(graph.symmetry.obs_signs) else []
            ep = graph.symmetry.est_perms[idx] if idx < len(graph.symmetry.est_perms) else []
            esg = graph.symmetry.est_signs[idx] if idx < len(graph.symmetry.est_signs) else []
            elems.append(
                {
                    'name': f'g{idx}',
                    'obs_perm': list(op),
                    'obs_sign': [float(v) for v in osg],
                    'act_perm': list(ap),
                    'act_sign': [float(v) for v in asg],
                    'est_perm': list(ep),
                    'est_sign': [float(v) for v in esg],
                }
            )
        sym_dict = {'kind': skind, 'elements': elems}

    return {
        'name': graph.name,
        'gravity': [float(v) for v in gravity],
        'bodies': bodies,
        'beams': beams,
        'tendons': tendons,
        'joints': joints,
        'contact_sites': contact_sites,
        'symmetry': sym_dict,
        'solver_hints': {
            'preferred_dt': float(dt),
            'rayleigh_alpha': default_alpha,
            'rayleigh_beta': default_beta,
            'force_route': None,
        },
    }


def graph_to_rust_ir_json(
    graph: RigidFlexGraph,
    world: ComposableSDFWorld | None = None,
    *,
    dt: float = 0.001,
) -> str:
    return json.dumps(graph_to_rust_ir_dict(graph, world=world, dt=dt), separators=(',', ':'))


def world_to_ffi_config_dict(
    world: ComposableSDFWorld | None = None,
    *,
    implicit_contact: bool = True,
    use_kelvin_normal: bool = False,
    enable_self_contact: bool = False,
    solver_preset: str | None = None,
    lcp_solver: str | None = None,
    sparse_backend: str | None = None,
    use_tight_inclusion_ccd: bool | None = None,
    world_sdf_override: str | None = None,
    use_strain_jacobian: bool = False,
    world_fixed_inertia: bool = True,
) -> dict[str, Any]:
    """Serialize solver & world SDF configuration into exact v2 ``FfiConfigJson`` (``fg-ffi/src/lib.rs``)."""
    preset = solver_preset if solver_preset is not None else ('serpworm_kelvin_parity' if use_kelvin_normal else 'default')
    cfg: dict[str, Any] = {
        'solver_preset': preset,
        'use_strain_jacobian': bool(use_strain_jacobian),
        'world_fixed_inertia': bool(world_fixed_inertia),
        'enable_self_contact': bool(enable_self_contact),
    }
    if lcp_solver is not None:
        cfg['lcp_solver'] = str(lcp_solver)
    elif not use_kelvin_normal and preset != 'delassus_socp':
        cfg['contact'] = {'backend': 'arpc', 'implicit': bool(implicit_contact)}
    if sparse_backend is not None:
        cfg['sparse_backend'] = str(sparse_backend)
    if use_tight_inclusion_ccd is not None:
        cfg['use_tight_inclusion_ccd'] = bool(use_tight_inclusion_ccd)

    if world_sdf_override is not None:
        cfg['world_sdf'] = str(world_sdf_override)
        return cfg

    if world is None:
        cfg['world_sdf'] = 'kelvin_floor' if use_kelvin_normal else 'flat_ground'
        cfg['ground_mat_id'] = 0
        return cfg

    if len(world.primitives) == 0:
        cfg['world_sdf'] = 'empty'
        return cfg

    prim_names = [type(p).__name__ for p in world.primitives]
    if 'SplineTubeSDF' in prim_names:
        cfg['world_sdf'] = 'endoscope_tube'
        return cfg

    p0 = world.primitives[0]
    if type(p0).__name__ == 'HeightFieldSDF':
        sx = float(p0.slope_x)
        sy = float(p0.slope_y)
        amp = float(p0.wave_amp_m)
        kx = float(p0.wave_kx)
        ky = float(p0.wave_ky)
        mat_id = int(p0.mat_id)
        is_flat = (sx == 0.0 and sy == 0.0 and amp == 0.0)
        if use_kelvin_normal and is_flat and mat_id == 0:
            cfg['world_sdf'] = 'kelvin_floor'
        elif is_flat:
            cfg['world_sdf'] = 'flat_ground'
            cfg['ground_mat_id'] = mat_id
        else:
            cfg['world_sdf'] = 'heightfield'
            cfg['ground_geom'] = [sx, sy, amp, kx, ky]
            cfg['ground_mat_id'] = mat_id
        return cfg

    cfg['world_sdf'] = 'flat_ground'
    cfg['ground_mat_id'] = int(getattr(p0, 'mat_id', 0))
    return cfg


def world_to_ffi_config_json(
    world: ComposableSDFWorld | None = None,
    *,
    implicit_contact: bool = True,
    use_kelvin_normal: bool = False,
    enable_self_contact: bool = False,
    solver_preset: str | None = None,
    lcp_solver: str | None = None,
    sparse_backend: str | None = None,
    use_tight_inclusion_ccd: bool | None = None,
    world_sdf_override: str | None = None,
    use_strain_jacobian: bool = False,
    world_fixed_inertia: bool = True,
) -> str:
    return json.dumps(
        world_to_ffi_config_dict(
            world,
            implicit_contact=implicit_contact,
            use_kelvin_normal=use_kelvin_normal,
            enable_self_contact=enable_self_contact,
            solver_preset=solver_preset,
            lcp_solver=lcp_solver,
            sparse_backend=sparse_backend,
            use_tight_inclusion_ccd=use_tight_inclusion_ccd,
            world_sdf_override=world_sdf_override,
            use_strain_jacobian=use_strain_jacobian,
            world_fixed_inertia=world_fixed_inertia,
        ),
        separators=(',', ':'),
    )


# ---------------------------------------------------------------------------
# Unified Rust / Python Solver Bridge
# ---------------------------------------------------------------------------
class RustGraphSolverBackend:
    """Drop-in solver bridge wrapping live Rust ``fg_ffi`` with safe ``GraphSchurSolver`` fallback.

    Exposes the full ``GraphSchurSolver`` tensor/state API so ``FlexGraphEnv`` and trainers can
    switch transparently via ``backend='auto' | 'python' | 'rust'``.
    """

    def __init__(
        self,
        graph: RigidFlexGraph,
        world: ComposableSDFWorld | None = None,
        *,
        batch: int = 1,
        dt: float = 0.001,
        device: str | torch.device = 'cpu',
        dtype: torch.dtype = torch.float64,
        implicit_contact: bool = True,
        use_kelvin_normal: bool = False,
        enable_self_contact: bool = False,
        self_contact_stiffness: float = 1500.0,
        backend: str = 'auto',
        strict_rust: bool = False,
        lib_path: str | Path | None = None,
        num_threads: int = 0,
        solver_preset: str | None = None,
        lcp_solver: str | None = None,
        sparse_backend: str | None = None,
        use_tight_inclusion_ccd: bool | None = None,
        world_sdf_override: str | None = None,
        use_strain_jacobian: bool = False,
        world_fixed_inertia: bool = True,
    ):
        backend_norm = str(backend).lower().strip()
        if backend_norm not in ('auto', 'python', 'rust'):
            raise ValueError(f"backend must be 'auto', 'python', or 'rust'; got {backend!r}")

        self.graph = graph
        self.world = world if world is not None else ComposableSDFWorld(primitives=[HeightFieldSDF()])
        self.batch = int(batch)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = dtype
        self.implicit_contact = bool(implicit_contact)
        self.use_kelvin_normal = bool(use_kelvin_normal)
        self.enable_self_contact = bool(enable_self_contact)
        self.num_threads = int(num_threads)
        self.solver_preset = solver_preset
        self.lcp_solver = lcp_solver
        self.sparse_backend = sparse_backend
        self.use_tight_inclusion_ccd = use_tight_inclusion_ccd
        self.world_sdf_override = world_sdf_override
        self.use_strain_jacobian = bool(use_strain_jacobian)
        self.world_fixed_inertia = bool(world_fixed_inertia)
        self.requested_backend = backend_norm
        self.strict_rust = bool(strict_rust)
        self.active_backend: str = 'python'
        self.fallback_reason: str = ''
        self.ffi_info: FFIBindings = FFIBindings(available=False, mode='unavailable', reason='python requested')

        self._py_solver: GraphSchurSolver | None = None
        self._native_handles: list[Any] = []
        self._native_mode: str = 'none'
        self._np_poses: np.ndarray | None = None
        self._np_vels: np.ndarray | None = None
        self._np_body_poses: np.ndarray | None = None
        self._np_body_vels: np.ndarray | None = None
        self._np_act3: np.ndarray | None = None
        self._np_summary: np.ndarray | None = None
        self._np_failed_u8: np.ndarray | None = None
        self._np_targets: np.ndarray | None = None

        self.sparsity = graph.analyze_sparsity()
        self.route = self.sparsity.recommended_route
        self.P = len(graph.bodies)
        self.n_actuators = len(graph.actuators)
        self.n_sites = len(graph.contact_sites)
        self.fixed_mask = torch.tensor([b.fixed for b in graph.bodies], device=self.device, dtype=torch.bool)
        self.free_body_ids = torch.where(~self.fixed_mask)[0]
        self.n_free = int(self.free_body_ids.numel())
        self.last_contact_summary: torch.Tensor | None = None
        self.failed_mask = torch.zeros((self.batch,), device=self.device, dtype=torch.bool)

        # Build tendon-group & joint-actuator mapping for Rust telemetry unpacking
        t_ids = sorted({t.actuator_idx for t in graph.tendons})
        self._tendon_act_groups: list[list[int]] = [
            [tid for tid, t in enumerate(graph.tendons) if t.actuator_idx == aid] for aid in t_ids
        ]
        j_pairs = sorted(
            ((jt.actuator_idx, jid) for jid, jt in enumerate(graph.joints) if jt.actuator_idx >= 0),
            key=lambda p: p[0],
        )
        self._joint_act_indices: list[int] = [jid for _, jid in j_pairs]

        if backend_norm in ('auto', 'rust'):
            if self.device.type != 'cpu':
                self.fallback_reason = (
                    f'Rust fg_ffi backend runs on CPU; using batched PyTorch GraphSchurSolver on {self.device}'
                )
            else:
                self.ffi_info = load_flexgraph_ffi(lib_path=lib_path)
                if self.ffi_info.available:
                    ok, err = self._try_init_native()
                    if ok:
                        self.active_backend = 'rust'
                    else:
                        self.fallback_reason = err
                else:
                    self.fallback_reason = self.ffi_info.reason

            if self.active_backend != 'rust' and backend_norm == 'rust' and self.strict_rust:
                raise RuntimeError(f'Rust fg_ffi backend unavailable: {self.fallback_reason}')

        if self.active_backend != 'rust':
            self._py_solver = GraphSchurSolver(
                graph,
                world=self.world,
                batch=self.batch,
                dt=self.dt,
                device=self.device,
                dtype=self.dtype,
                implicit_contact=self.implicit_contact,
                use_kelvin_normal=self.use_kelvin_normal,
                enable_self_contact=self.enable_self_contact,
                self_contact_stiffness=self_contact_stiffness,
            )
            self._mirror_python_attrs()

    @property
    def supports_v2(self) -> bool:
        return bool(self.ffi_info.supports_v2)

    def _mirror_python_attrs(self) -> None:
        assert self._py_solver is not None
        py = self._py_solver
        self.D = py.D
        self.rest = py.rest
        self.x = py.x
        self.v = py.v
        self.act_state = py.act_state
        self.act_target = py.act_target
        self.tendon_tensions = py.tendon_tensions
        self.joint_torques = py.joint_torques
        self.contact_state = py.contact_state
        self.last_site_normal = py.last_site_normal
        self.last_self_contact_energy = py.last_self_contact_energy
        self.failed = py.failed
        self.last_contact_summary = None

    def _try_init_native(self) -> tuple[bool, str]:
        graph_json = graph_to_rust_ir_json(self.graph, world=self.world, dt=self.dt).encode('utf-8')
        config_json = world_to_ffi_config_json(
            self.world,
            implicit_contact=self.implicit_contact,
            use_kelvin_normal=self.use_kelvin_normal,
            enable_self_contact=self.enable_self_contact,
            solver_preset=self.solver_preset,
            lcp_solver=self.lcp_solver,
            sparse_backend=self.sparse_backend,
            use_tight_inclusion_ccd=self.use_tight_inclusion_ccd,
            world_sdf_override=self.world_sdf_override,
            use_strain_jacobian=self.use_strain_jacobian,
            world_fixed_inertia=self.world_fixed_inertia,
        ).encode('utf-8')
        self.D = self.P + self.sparsity.n_beam_interior_nodes

        try:
            if self.ffi_info.mode == 'cdylib' and self.ffi_info.cdll is not None:
                lib = self.ffi_info.cdll
                if self.ffi_info.supports_batch_handle:
                    h = lib.fg_batch_create_from_json_with_config(
                        graph_json,
                        config_json,
                        ctypes.c_size_t(self.batch),
                        ctypes.c_double(self.dt),
                        ctypes.c_size_t(self.num_threads),
                    )
                    if not h:
                        err = self.ffi_info.last_error() or 'NULL FgBatchHandle'
                        return False, f'fg_batch_create_from_json_with_config failed: {err}'
                    nb = ctypes.c_size_t(0)
                    na = ctypes.c_size_t(0)
                    nbeams = ctypes.c_size_t(0)
                    rc = lib.fg_batch_dims(h, ctypes.byref(nb), ctypes.byref(na), ctypes.byref(nbeams))
                    if rc != 0:
                        lib.fg_batch_destroy(h)
                        return False, f'fg_batch_dims returned rc={rc}: {self.ffi_info.last_error()}'
                    total_nodes = int(lib.fg_batch_total_nodes(h))
                    if int(nb.value) != self.P or int(na.value) != self.n_actuators or total_nodes < self.P:
                        lib.fg_batch_destroy(h)
                        return (
                            False,
                            f'Rust FgBatchHandle dimension mismatch '
                            f'(rust nb={nb.value}, na={na.value}, D={total_nodes} vs '
                            f'py P={self.P}, N_act={self.n_actuators})',
                        )
                    self.D = total_nodes
                    self._native_handles = [h]
                    self._native_mode = 'cdylib_batch'
                elif self.ffi_info.supports_world_handle:
                    handles = []
                    for _ in range(self.batch):
                        h = lib.fg_world_create_from_json(graph_json, config_json)
                        if not h:
                            err = self.ffi_info.last_error() or 'NULL FgWorldHandle'
                            for prev in handles:
                                lib.fg_world_free(prev)
                            return False, f'fg_world_create_from_json failed: {err}'
                        handles.append(h)
                    info = FgWorldInfo()
                    if lib.fg_world_get_info(handles[0], ctypes.byref(info)) != 0:
                        for prev in handles:
                            lib.fg_world_free(prev)
                        return False, f'fg_world_get_info failed: {self.ffi_info.last_error()}'
                    if int(info.num_bodies) != self.P or int(info.num_total_actuators) != self.n_actuators:
                        for prev in handles:
                            lib.fg_world_free(prev)
                        return False, 'Rust FgWorldHandle dimension mismatch'
                    self.D = int(info.total_nodes)
                    self._native_handles = handles
                    self._native_mode = 'cdylib_world'
                else:
                    return False, 'Loaded cdylib supports neither FgBatchHandle nor FgWorldHandle'
            elif self.ffi_info.mode == 'pyo3' and self.ffi_info.pyo3_module is not None:
                mod = self.ffi_info.pyo3_module
                batch_cls = getattr(mod, 'FlexGraphBatchWorld', None)
                g_str = graph_json.decode('utf-8')
                c_str = config_json.decode('utf-8')
                if batch_cls is not None:
                    self._native_handles = [batch_cls.from_json(g_str, c_str, self.dt, self.batch)]
                    self._native_mode = 'pyo3_batched'
                else:
                    world_cls = getattr(mod, 'FlexGraphWorld', None) or getattr(mod, 'RustWorld', None)
                    self._native_handles = [world_cls.from_json(g_str, c_str, self.dt) for _ in range(self.batch)]
                    self._native_mode = 'pyo3'
            else:
                return False, f'Unsupported FFI mode: {self.ffi_info.mode}'
        except Exception as exc:
            self.close()
            return False, f'Native fg_ffi world initialization failed: {exc}'

        # Pre-allocated contiguous flat FP64/u8 buffers matching fg-ffi layouts
        self._np_poses = np.zeros((self.batch, self.D, 7), dtype=np.float64)
        self._np_vels = np.zeros((self.batch, self.D, 6), dtype=np.float64)
        self._np_body_poses = np.zeros((self.batch, self.P, 7), dtype=np.float64)
        self._np_body_vels = np.zeros((self.batch, self.P, 6), dtype=np.float64)
        self._np_act3 = np.zeros((self.batch, max(1, self.n_actuators), 3), dtype=np.float64)
        self._np_summary = np.zeros((self.batch, 7), dtype=np.float64)
        self._np_failed_u8 = np.zeros((self.batch,), dtype=np.uint8)
        self._np_targets = np.zeros((self.batch, max(1, self.n_actuators)), dtype=np.float64)

        self.x = torch.zeros((self.batch, self.D, 7), device=self.device, dtype=self.dtype)
        self.v = torch.zeros((self.batch, self.D, 6), device=self.device, dtype=self.dtype)
        self.act_state = torch.zeros((self.batch, self.n_actuators), device=self.device, dtype=self.dtype)
        self.act_target = torch.zeros((self.batch, self.n_actuators), device=self.device, dtype=self.dtype)
        self.tendon_tensions = torch.zeros((self.batch, len(self.graph.tendons)), device=self.device, dtype=self.dtype)
        self.joint_torques = torch.zeros((self.batch, len(self.graph.joints)), device=self.device, dtype=self.dtype)
        self.contact_state = ContactState.zeros(self.batch, self.n_sites, self.device, self.dtype)
        self.last_site_normal = torch.zeros((self.batch, self.n_sites), device=self.device, dtype=self.dtype)
        self.last_self_contact_energy = torch.zeros((self.batch,), device=self.device, dtype=self.dtype)
        self.last_contact_summary = torch.zeros((self.batch, 7), device=self.device, dtype=self.dtype)
        self.failed = torch.zeros((), device=self.device, dtype=torch.bool)
        self.failed_mask = torch.zeros((self.batch,), device=self.device, dtype=torch.bool)

        self._sync_from_native(refresh_batch_snapshot=True)
        self.rest = self.x.clone()
        return True, ''

    def _sync_from_native(self, *, refresh_batch_snapshot: bool = False) -> None:
        if not self._native_handles:
            return
        assert self._np_poses is not None and self._np_vels is not None
        assert self._np_body_poses is not None and self._np_body_vels is not None
        assert self._np_act3 is not None and self._np_summary is not None and self._np_failed_u8 is not None
        c_dbl_p = ctypes.POINTER(ctypes.c_double)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)

        if self._native_mode == 'cdylib_batch':
            lib = self.ffi_info.cdll
            h = self._native_handles[0]
            if refresh_batch_snapshot:
                lib.fg_batch_step_into(
                    h,
                    ctypes.c_size_t(0),
                    None,
                    self._np_body_poses.ctypes.data_as(c_dbl_p),
                    self._np_body_vels.ctypes.data_as(c_dbl_p),
                    self._np_act3.ctypes.data_as(c_dbl_p) if self.n_actuators > 0 else None,
                    self._np_summary.ctypes.data_as(c_dbl_p),
                    self._np_failed_u8.ctypes.data_as(c_u8_p),
                )
            lib.fg_batch_get_all_nodes_into(
                h,
                self._np_poses.ctypes.data_as(c_dbl_p),
                self._np_vels.ctypes.data_as(c_dbl_p),
            )
            self.x.copy_(torch.from_numpy(self._np_poses).to(device=self.device, dtype=self.dtype))
            self.v.copy_(torch.from_numpy(self._np_vels).to(device=self.device, dtype=self.dtype))
            if self.n_actuators > 0:
                self.act_state.copy_(
                    torch.from_numpy(self._np_act3[:, : self.n_actuators, 0]).to(device=self.device, dtype=self.dtype)
                )
                # Populate tendon_tensions & joint_torques from act_state[..., 2] loads
                loads = self._np_act3[:, : self.n_actuators, 2]
                slot = 0
                for members in self._tendon_act_groups:
                    if members and self.tendon_tensions.numel() > 0:
                        per_cord = loads[:, slot] / float(len(members))
                        for tid in members:
                            self.tendon_tensions[:, tid].copy_(torch.from_numpy(per_cord).to(device=self.device, dtype=self.dtype))
                    slot += 1
                for jid in self._joint_act_indices:
                    if self.joint_torques.numel() > 0:
                        self.joint_torques[:, jid].copy_(torch.from_numpy(loads[:, slot]).to(device=self.device, dtype=self.dtype))
                    slot += 1
            assert self.last_contact_summary is not None
            self.last_contact_summary.copy_(torch.from_numpy(self._np_summary).to(device=self.device, dtype=self.dtype))
            self.failed_mask.copy_(torch.from_numpy(self._np_failed_u8 != 0).to(device=self.device, dtype=torch.bool))
            any_fail = bool(self.failed_mask.any().item()) or (not bool(torch.isfinite(self.x).all().item()))
            self.failed.fill_(any_fail)
            return

        if self._native_mode == 'cdylib_world':
            lib = self.ffi_info.cdll
            n_ten = len(self.graph.tendons)
            has_tptr = hasattr(lib, 'fg_world_tensions_ptr')
            has_nptr = hasattr(lib, 'fg_world_contact_normals_ptr')
            for b, h in enumerate(self._native_handles):
                lib.fg_world_get_state(
                    h,
                    self._np_poses[b].ctypes.data_as(c_dbl_p),
                    ctypes.c_size_t(self.D * 7),
                    self._np_vels[b].ctypes.data_as(c_dbl_p),
                    ctypes.c_size_t(self.D * 6),
                )
                info = FgWorldInfo()
                lib.fg_world_get_info(h, ctypes.byref(info))
                self._np_failed_u8[b] = info.failed
                if has_tptr and n_ten > 0:
                    tptr = lib.fg_world_tensions_ptr(h)
                    if tptr:
                        arr = np.ctypeslib.as_array(tptr, shape=(n_ten,))
                        self.tendon_tensions[b].copy_(torch.from_numpy(arr).to(device=self.device, dtype=self.dtype))
                if has_nptr and self.n_sites > 0:
                    nptr = lib.fg_world_contact_normals_ptr(h)
                    if nptr:
                        arr = np.ctypeslib.as_array(nptr, shape=(self.n_sites,))
                        self.last_site_normal[b].copy_(torch.from_numpy(arr).to(device=self.device, dtype=self.dtype))
            self.x.copy_(torch.from_numpy(self._np_poses).to(device=self.device, dtype=self.dtype))
            self.v.copy_(torch.from_numpy(self._np_vels).to(device=self.device, dtype=self.dtype))
            self.failed_mask.copy_(torch.from_numpy(self._np_failed_u8 != 0).to(device=self.device, dtype=torch.bool))
            self.failed.fill_(bool(self.failed_mask.any().item()) or (not bool(torch.isfinite(self.x).all().item())))
            return

        if self._native_mode in ('pyo3_batched', 'pyo3'):
            for b, h in enumerate(self._native_handles):
                if self._native_mode == 'pyo3_batched':
                    self._np_poses[:] = np.asarray(h.get_poses(), dtype=np.float64).reshape(self.batch, self.D, 7)
                    self._np_vels[:] = np.asarray(h.get_vels(), dtype=np.float64).reshape(self.batch, self.D, 6)
                else:
                    self._np_poses[b] = np.asarray(h.get_poses(), dtype=np.float64).reshape(self.D, 7)
                    self._np_vels[b] = np.asarray(h.get_vels(), dtype=np.float64).reshape(self.D, 6)
            self.x.copy_(torch.from_numpy(self._np_poses).to(device=self.device, dtype=self.dtype))
            self.v.copy_(torch.from_numpy(self._np_vels).to(device=self.device, dtype=self.dtype))
            self.failed.fill_(not bool(torch.isfinite(self.x).all().item()))

    def reset(self) -> None:
        if self._py_solver is not None:
            self._py_solver.reset()
            self._mirror_python_attrs()
            return
        self.act_target.zero_()
        if self._native_mode == 'cdylib_batch':
            self.ffi_info.cdll.fg_batch_reset_mask(self._native_handles[0], None)
            self._sync_from_native(refresh_batch_snapshot=True)
        elif self._native_mode == 'cdylib_world':
            for h in self._native_handles:
                self.ffi_info.cdll.fg_world_reset(h)
            self.act_state.zero_()
            self._sync_from_native()
        elif self._native_mode in ('pyo3', 'pyo3_batched'):
            for h in self._native_handles:
                h.reset()
            self._sync_from_native()

    def reset_envs(self, env_mask: torch.Tensor) -> None:
        """Reset only the subset of batch environments where ``env_mask`` is True."""
        mask = env_mask.to(device=self.device, dtype=torch.bool)
        if not bool(mask.any().item()):
            return
        if bool(mask.all().item()):
            self.reset()
            return
        if self._py_solver is not None:
            py = self._py_solver
            py.x[mask] = py.rest[mask]
            py.v[mask] = 0.0
            py.act_state[mask] = 0.0
            py.act_target[mask] = 0.0
            py.tendon_tensions[mask] = 0.0
            py.joint_torques[mask] = 0.0
            py.last_site_normal[mask] = 0.0
            py.last_self_contact_energy[mask] = 0.0
            py.contact_state.rut[mask] = 0.0
            py.contact_state.memory[mask] = 0.0
            py.contact_state.spin[mask] = 0.0
            py.contact_state.angle[mask] = 0.0
            py.contact_state.ratchet_play[mask] = 0.0
            if bool(torch.isfinite(py.x).all().item()) and bool(torch.isfinite(py.v).all().item()):
                py.failed.zero_()
            return

        self.act_target[mask] = 0.0
        if self._native_mode == 'cdylib_batch':
            mask_u8 = np.ascontiguousarray(mask.detach().cpu().numpy().astype(np.uint8))
            self.ffi_info.cdll.fg_batch_reset_mask(
                self._native_handles[0],
                mask_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            )
            self._sync_from_native(refresh_batch_snapshot=True)
        elif self._native_mode == 'cdylib_world':
            for b in torch.where(mask)[0].cpu().tolist():
                self.ffi_info.cdll.fg_world_reset(self._native_handles[b])
            self.act_state[mask] = 0.0
            self._sync_from_native()
        elif self._native_mode == 'pyo3_batched':
            if hasattr(self._native_handles[0], 'reset_envs'):
                self._native_handles[0].reset_envs(torch.where(mask)[0].cpu().tolist())
            self._sync_from_native()
        elif self._native_mode == 'pyo3':
            for b in torch.where(mask)[0].cpu().tolist():
                self._native_handles[b].reset()
            self._sync_from_native()

    def step(self, targets: torch.Tensor | None = None, steps: int = 1) -> None:
        if self._py_solver is not None:
            self._py_solver.step(targets, steps=steps)
            self._mirror_python_attrs()
            return

        if targets is not None:
            tgt = torch.as_tensor(targets, device=self.device, dtype=self.dtype)
            if tgt.ndim == 1:
                tgt = tgt[None].expand(self.batch, -1)
            self.act_target.copy_(tgt)

        assert self._np_targets is not None and self._np_body_poses is not None and self._np_body_vels is not None
        assert self._np_act3 is not None and self._np_summary is not None and self._np_failed_u8 is not None
        if self.n_actuators > 0:
            np.copyto(self._np_targets[:, : self.n_actuators], self.act_target.detach().cpu().numpy())
        c_dbl_p = ctypes.POINTER(ctypes.c_double)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        n_steps = int(steps)

        if self._native_mode == 'cdylib_batch':
            lib = self.ffi_info.cdll
            act_ptr = self._np_targets.ctypes.data_as(c_dbl_p) if self.n_actuators > 0 else None
            rc = lib.fg_batch_step_into(
                self._native_handles[0],
                ctypes.c_size_t(n_steps),
                act_ptr,
                self._np_body_poses.ctypes.data_as(c_dbl_p),
                self._np_body_vels.ctypes.data_as(c_dbl_p),
                self._np_act3.ctypes.data_as(c_dbl_p) if self.n_actuators > 0 else None,
                self._np_summary.ctypes.data_as(c_dbl_p),
                self._np_failed_u8.ctypes.data_as(c_u8_p),
            )
            if rc != 0:
                self.failed.fill_(True)
            self._sync_from_native(refresh_batch_snapshot=False)
        elif self._native_mode == 'cdylib_world':
            lib = self.ffi_info.cdll
            diag = FgStepDiag()
            for b, h in enumerate(self._native_handles):
                if self.n_actuators > 0:
                    row = np.ascontiguousarray(self._np_targets[b, : self.n_actuators], dtype=np.float64)
                    lib.fg_world_set_targets(h, row.ctypes.data_as(c_dbl_p), ctypes.c_size_t(self.n_actuators))
                lib.fg_world_step(h, ctypes.c_size_t(n_steps), ctypes.c_double(self.dt), ctypes.byref(diag))
            self._sync_from_native()
        elif self._native_mode == 'pyo3_batched':
            h = self._native_handles[0]
            if hasattr(h, 'step_batch_into'):
                h.step_batch_into(self._np_targets, self._np_poses, self._np_vels, self._np_act3, n_steps)
            else:
                h.step_batch(self._np_targets, n_steps)
            self._sync_from_native()
        elif self._native_mode == 'pyo3':
            for b, h in enumerate(self._native_handles):
                h.step(self._np_targets[b, : self.n_actuators].tolist(), n_steps)
            self._sync_from_native()

    def close(self) -> None:
        if self._native_handles and self.ffi_info.cdll is not None:
            lib = self.ffi_info.cdll
            if self._native_mode == 'cdylib_batch' and hasattr(lib, 'fg_batch_destroy'):
                for h in self._native_handles:
                    try:
                        lib.fg_batch_destroy(h)
                    except Exception:
                        pass
            elif self._native_mode == 'cdylib_world' and hasattr(lib, 'fg_world_free'):
                for h in self._native_handles:
                    try:
                        lib.fg_world_free(h)
                    except Exception:
                        pass
        self._native_handles = []
        self._native_mode = 'none'

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def create_solver(
    graph: RigidFlexGraph,
    world: ComposableSDFWorld | None = None,
    *,
    backend: str = 'auto',
    strict_rust: bool = False,
    lib_path: str | Path | None = None,
    **kwargs: Any,
) -> GraphSchurSolver | RustGraphSolverBackend:
    """Factory returning either ``GraphSchurSolver`` (``backend='python'``) or ``RustGraphSolverBackend``."""
    b = str(backend).lower().strip()
    if b == 'python':
        return GraphSchurSolver(graph, world=world, **kwargs)
    return RustGraphSolverBackend(
        graph,
        world=world,
        backend=b,
        strict_rust=strict_rust,
        lib_path=lib_path,
        **kwargs,
    )


def benchmark_backend_parity(
    graph: RigidFlexGraph,
    world: ComposableSDFWorld | None = None,
    *,
    steps: int = 15,
    dt: float = 0.001,
    targets: torch.Tensor | None = None,
    batch: int = 1,
    lib_path: str | Path | None = None,
    use_kelvin_normal: bool = False,
    implicit_contact: bool = True,
) -> dict[str, Any]:
    """Benchmark Python ``GraphSchurSolver`` vs ``RustGraphSolverBackend`` and measure FP64 parity."""
    w = world if world is not None else ComposableSDFWorld(primitives=[HeightFieldSDF()])
    py_sim = GraphSchurSolver(
        graph,
        world=w,
        batch=batch,
        dt=dt,
        device='cpu',
        dtype=torch.float64,
        implicit_contact=implicit_contact,
        use_kelvin_normal=use_kelvin_normal,
    )
    rs_sim = RustGraphSolverBackend(
        graph,
        world=w,
        batch=batch,
        dt=dt,
        device='cpu',
        dtype=torch.float64,
        implicit_contact=implicit_contact,
        use_kelvin_normal=use_kelvin_normal,
        backend='auto',
        strict_rust=False,
        lib_path=lib_path,
    )

    if targets is None:
        n_act = len(graph.actuators)
        lo = torch.tensor([a.min_cmd for a in graph.actuators], dtype=torch.float64)
        hi = torch.tensor([a.max_cmd for a in graph.actuators], dtype=torch.float64)
        tgt = (0.5 * (lo + hi) + 0.15 * (hi - lo))[None].expand(batch, n_act).contiguous()
    else:
        tgt = torch.as_tensor(targets, dtype=torch.float64)

    t0 = time.perf_counter()
    py_sim.step(tgt, steps=steps)
    py_s = max(time.perf_counter() - t0, 1e-9)

    t1 = time.perf_counter()
    rs_sim.step(tgt, steps=steps)
    rs_s = max(time.perf_counter() - t1, 1e-9)

    pos_err = float((py_sim.x[..., :3] - rs_sim.x[..., :3]).abs().max().item())
    vel_err = float((py_sim.v - rs_sim.v).abs().max().item())
    act_err = (
        float((py_sim.act_state - rs_sim.act_state).abs().max().item())
        if py_sim.act_state.numel() > 0
        else 0.0
    )

    report = {
        'graph': graph.name,
        'route': py_sim.route,
        'steps': int(steps),
        'batch': int(batch),
        'dt': float(dt),
        'native_rust_available': rs_sim.active_backend == 'rust',
        'active_backend': rs_sim.active_backend,
        'fallback_reason': rs_sim.fallback_reason,
        'ffi': rs_sim.ffi_info.summary_dict(),
        'python_ms': round(py_s * 1000.0, 3),
        'backend_ms': round(rs_s * 1000.0, 3),
        'speedup_vs_python': round(py_s / rs_s, 3),
        'max_pos_err_m': pos_err,
        'max_vel_err': vel_err,
        'max_act_err': act_err,
        'python_failed': bool(py_sim.failed.item()),
        'backend_failed': bool(rs_sim.failed.item()),
    }
    rs_sim.close()
    return report


# ---------------------------------------------------------------------------
# Unified FlexGraph-RS v2 Single-World Factories (All 36/36 C-ABI Symbols)
# ---------------------------------------------------------------------------
def _require_rs_solver_and_lib(lib_path: str | Path | None = None, *, require_v2: bool = False) -> Path:
    if FlexGraphRsSolver is None:
        raise RuntimeError('flexgraph_rs sibling package is not importable')
    ffi = load_flexgraph_ffi(lib_path=lib_path, prefer_pyo3=False)
    if not ffi.available or ffi.library_path is None:
        raise RuntimeError(f'Native fg_ffi library unavailable: {ffi.reason}')
    if require_v2 and not ffi.supports_v2:
        missing_v2 = [s for s in V2_C_ABI_SYMBOLS if s not in ffi.exported_symbols]
        raise RuntimeError(f'Loaded fg_ffi library does not export v2 symbols: {missing_v2}')
    return ffi.library_path


def create_world_from_ros_urdf(
    urdf_xml: str,
    config: dict[str, Any] | str | None = None,
    *,
    lib_path: str | Path | None = None,
) -> Any:
    """Create a native FlexGraph-RS v2 world from standard ROS URDF XML (``fg_world_create_from_ros_urdf``)."""
    resolved = _require_rs_solver_and_lib(lib_path=lib_path, require_v2=True)
    return FlexGraphRsSolver.from_ros_urdf(urdf_xml, config=config, lib_path=resolved)


def create_world_from_mjcf(
    mjcf_xml: str,
    config: dict[str, Any] | str | None = None,
    *,
    lib_path: str | Path | None = None,
) -> Any:
    """Create a native FlexGraph-RS v2 world from MuJoCo MJCF XML (``fg_world_create_from_mjcf``)."""
    resolved = _require_rs_solver_and_lib(lib_path=lib_path, require_v2=True)
    return FlexGraphRsSolver.from_mjcf(mjcf_xml, config=config, lib_path=resolved)


def create_world_with_corrugated_pipe_mesh(
    graph_or_json: RigidFlexGraph | str,
    config: dict[str, Any] | str | None = None,
    *,
    length: float = 1.0,
    r_mean: float = 0.08,
    corrugation_amp: float = 0.006,
    n_waves: float = 8.0,
    origin: tuple[float, float, float] = (0.15, 0.0, 0.08),
    dt: float = 0.001,
    lib_path: str | Path | None = None,
) -> Any:
    """Mount a ``RigidFlexGraph`` or JSON IR inside a Parry3D corrugated pipe mesh (``fg_world_create_with_corrugated_pipe_mesh``)."""
    resolved = _require_rs_solver_and_lib(lib_path=lib_path, require_v2=True)
    g_json = (
        graph_to_rust_ir_json(graph_or_json, dt=dt)
        if isinstance(graph_or_json, RigidFlexGraph)
        else str(graph_or_json)
    )
    return FlexGraphRsSolver.with_corrugated_pipe_mesh(
        g_json,
        config=config,
        length=length,
        r_mean=r_mean,
        corrugation_amp=corrugation_amp,
        n_waves=n_waves,
        origin=origin,
        lib_path=resolved,
    )


def create_native_world(
    graph_or_archetype: RigidFlexGraph | str,
    world: ComposableSDFWorld | None = None,
    *,
    config: dict[str, Any] | str | None = None,
    dt: float = 0.001,
    lib_path: str | Path | None = None,
    **v2_cfg_kwargs: Any,
) -> Any:
    """Create a native Rust ``FlexGraphRsSolver`` from a Python ``RigidFlexGraph`` or Rust archetype name.

    Exposes zero-copy FP64 views, energy/momentum diagnostics, and v2 IFT reverse-mode adjoints
    (``adjoint_grad``, ``get_diff_param``, ``set_diff_param``) directly from ``flexgraph``.
    """
    resolved = _require_rs_solver_and_lib(lib_path=lib_path, require_v2=False)
    cfg_payload: dict[str, Any] | str | None = config
    if cfg_payload is None and (world is not None or v2_cfg_kwargs):
        cfg_payload = world_to_ffi_config_dict(world, **v2_cfg_kwargs)
    if isinstance(graph_or_archetype, RigidFlexGraph):
        g_json = graph_to_rust_ir_json(graph_or_archetype, world=world, dt=dt)
        return FlexGraphRsSolver.from_json(g_json, config=cfg_payload, lib_path=resolved)
    return FlexGraphRsSolver.from_archetype(str(graph_or_archetype), config=cfg_payload, lib_path=resolved)

"""Low-level ctypes declarations & native shared-library loader for `fg-ffi`."""
from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_double,
    c_int,
    c_int32,
    c_size_t,
    c_uint8,
    c_uint64,
    c_void_p,
)
import os
from pathlib import Path
import sys
from typing import Optional

SCHUR_ROUTE_NAMES = {
    0: "block_diagonal_pairs",
    1: "tree_chain",
    2: "sparse_general",
}


class FgWorldInfo(Structure):
    _fields_ = [
        ("num_bodies", c_size_t),
        ("num_free_bodies", c_size_t),
        ("num_beam_nodes", c_size_t),
        ("total_nodes", c_size_t),
        ("num_beams", c_size_t),
        ("num_tendons", c_size_t),
        ("num_tendon_actuators", c_size_t),
        ("num_joint_actuators", c_size_t),
        ("num_total_actuators", c_size_t),
        ("num_contact_sites", c_size_t),
        ("schur_route", c_int32),
        ("cyclomatic_number", c_size_t),
        ("step_count", c_uint64),
        ("sim_time", c_double),
        ("failed", c_uint8),
    ]


class FgStepDiag(Structure):
    _fields_ = [
        ("step_index", c_uint64),
        ("sim_time", c_double),
        ("newton_iters", c_size_t),
        ("residual_inf", c_double),
        ("residual_l2", c_double),
        ("line_search_alpha", c_double),
        ("ccd_toi", c_double),
        ("active_site_contacts", c_size_t),
        ("active_beam_contacts", c_size_t),
        ("active_self_contacts", c_size_t),
        ("lcp_iters", c_size_t),
        ("lcp_residual", c_double),
        ("max_quat_norm_err", c_double),
        ("failed", c_uint8),
    ]


class FgEnergyMomentum(Structure):
    _fields_ = [
        ("kinetic_energy", c_double),
        ("elastic_energy", c_double),
        ("actuation_potential_energy", c_double),
        ("gravity_potential_energy", c_double),
        ("total_energy", c_double),
        ("linear_momentum", c_double * 3),
        ("angular_momentum", c_double * 3),
    ]


def _default_lib_filename() -> str:
    if sys.platform.startswith("win"):
        return "fg_ffi.dll"
    if sys.platform == "darwin":
        return "libfg_ffi.dylib"
    return "libfg_ffi.so"


def find_native_library(lib_path: str | Path | None = None) -> Path:
    """Locate compiled `fg_ffi` shared library (prefers release, falls back to debug)."""
    if lib_path is not None:
        p = Path(lib_path)
        if not p.is_file():
            raise FileNotFoundError(f"fg-ffi library not found at {p}")
        return p

    env_path = os.environ.get("FLEXGRAPH_RS_LIB")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    fname = _default_lib_filename()
    repo_root = Path(__file__).resolve().parents[2]
    rust_target = repo_root / "rust" / "flexgraph-rs" / "target"
    candidates = [
        rust_target / "release" / fname,
        rust_target / "debug" / fname,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Could not find compiled `{fname}` under {rust_target}. "
        f"Run `cargo build --release -p fg-ffi` in `rust/flexgraph-rs/`."
    )


_LIB_CACHE: dict[Path, ctypes.CDLL] = {}


def load_native_lib(lib_path: str | Path | None = None) -> ctypes.CDLL:
    """Load and bind all C-ABI symbols of `fg-ffi`."""
    resolved = find_native_library(lib_path).resolve()
    cached = _LIB_CACHE.get(resolved)
    if cached is not None:
        return cached

    lib = ctypes.CDLL(str(resolved))

    # Error & string utilities
    lib.fg_last_error.argtypes = []
    lib.fg_last_error.restype = c_char_p
    lib.fg_string_free.argtypes = [c_void_p]
    lib.fg_string_free.restype = None

    # Single-world lifecycle
    lib.fg_world_create_from_json.argtypes = [c_char_p, c_char_p]
    lib.fg_world_create_from_json.restype = c_void_p

    lib.fg_world_create_from_yaml.argtypes = [c_char_p, c_char_p]
    lib.fg_world_create_from_yaml.restype = c_void_p

    lib.fg_world_create_from_urdf_flex.argtypes = [c_char_p, c_char_p]
    lib.fg_world_create_from_urdf_flex.restype = c_void_p

    lib.fg_world_create_archetype.argtypes = [c_char_p, c_char_p]
    lib.fg_world_create_archetype.restype = c_void_p

    lib.fg_world_reset.argtypes = [c_void_p]
    lib.fg_world_reset.restype = c_int

    lib.fg_world_set_targets.argtypes = [c_void_p, POINTER(c_double), c_size_t]
    lib.fg_world_set_targets.restype = c_int

    lib.fg_world_set_tendon_targets.argtypes = [c_void_p, POINTER(c_double), c_size_t]
    lib.fg_world_set_tendon_targets.restype = c_int

    lib.fg_world_set_joint_targets.argtypes = [c_void_p, POINTER(c_double), c_size_t]
    lib.fg_world_set_joint_targets.restype = c_int

    lib.fg_world_step.argtypes = [c_void_p, c_size_t, c_double, POINTER(FgStepDiag)]
    lib.fg_world_step.restype = c_int

    lib.fg_world_get_state.argtypes = [
        c_void_p,
        POINTER(c_double),
        c_size_t,
        POINTER(c_double),
        c_size_t,
    ]
    lib.fg_world_get_state.restype = c_int

    # Zero-copy FP64 view pointers
    lib.fg_world_poses_ptr.argtypes = [c_void_p]
    lib.fg_world_poses_ptr.restype = POINTER(c_double)

    lib.fg_world_vels_ptr.argtypes = [c_void_p]
    lib.fg_world_vels_ptr.restype = POINTER(c_double)

    lib.fg_world_tensions_ptr.argtypes = [c_void_p]
    lib.fg_world_tensions_ptr.restype = POINTER(c_double)

    lib.fg_world_contact_normals_ptr.argtypes = [c_void_p]
    lib.fg_world_contact_normals_ptr.restype = POINTER(c_double)

    lib.fg_world_get_info.argtypes = [c_void_p, POINTER(FgWorldInfo)]
    lib.fg_world_get_info.restype = c_int

    lib.fg_world_get_energy_momentum.argtypes = [c_void_p, POINTER(FgEnergyMomentum)]
    lib.fg_world_get_energy_momentum.restype = c_int

    lib.fg_world_export_graph_json.argtypes = [c_void_p]
    lib.fg_world_export_graph_json.restype = c_void_p

    lib.fg_world_free.argtypes = [c_void_p]
    lib.fg_world_free.restype = None

    # Vectorized Rayon batch RL API
    lib.fg_batch_create_from_json.argtypes = [c_char_p, c_size_t, c_double, c_size_t]
    lib.fg_batch_create_from_json.restype = c_void_p

    lib.fg_batch_create_from_json_with_config.argtypes = [
        c_char_p,
        c_char_p,
        c_size_t,
        c_double,
        c_size_t,
    ]
    lib.fg_batch_create_from_json_with_config.restype = c_void_p

    lib.fg_batch_create_archetype.argtypes = [
        c_char_p,
        c_char_p,
        c_size_t,
        c_double,
        c_size_t,
    ]
    lib.fg_batch_create_archetype.restype = c_void_p

    lib.fg_batch_dims.argtypes = [
        c_void_p,
        POINTER(c_size_t),
        POINTER(c_size_t),
        POINTER(c_size_t),
    ]
    lib.fg_batch_dims.restype = c_int

    lib.fg_batch_size.argtypes = [c_void_p]
    lib.fg_batch_size.restype = c_size_t

    lib.fg_batch_total_nodes.argtypes = [c_void_p]
    lib.fg_batch_total_nodes.restype = c_size_t

    lib.fg_batch_reset_mask.argtypes = [c_void_p, POINTER(c_uint8)]
    lib.fg_batch_reset_mask.restype = c_int

    lib.fg_batch_step_into.argtypes = [
        c_void_p,
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_uint8),
    ]
    lib.fg_batch_step_into.restype = c_int

    lib.fg_batch_get_all_nodes_into.argtypes = [
        c_void_p,
        POINTER(c_double),
        POINTER(c_double),
    ]
    lib.fg_batch_get_all_nodes_into.restype = c_int

    lib.fg_batch_destroy.argtypes = [c_void_p]
    lib.fg_batch_destroy.restype = None

    # v2 additive exports: ROS URDF, MJCF, Parry3D corrugated pipe mesh, IFT reverse-mode adjoint
    lib.fg_world_create_from_ros_urdf.argtypes = [c_char_p, c_char_p]
    lib.fg_world_create_from_ros_urdf.restype = c_void_p

    lib.fg_world_create_from_mjcf.argtypes = [c_char_p, c_char_p]
    lib.fg_world_create_from_mjcf.restype = c_void_p

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

    lib.fg_world_adjoint_grad.argtypes = [
        c_void_p,
        POINTER(c_double),
        c_size_t,
        c_size_t,
        c_double,
        POINTER(c_double),
        c_size_t,
        c_double,
        c_double,
        c_double,
        c_char_p,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_size_t,
    ]
    lib.fg_world_adjoint_grad.restype = c_int

    lib.fg_world_get_diff_param.argtypes = [c_void_p, c_char_p, POINTER(c_double)]
    lib.fg_world_get_diff_param.restype = c_int

    lib.fg_world_set_diff_param.argtypes = [c_void_p, c_char_p, c_double, c_uint8]
    lib.fg_world_set_diff_param.restype = c_int

    _LIB_CACHE[resolved] = lib
    return lib


def last_error_message(lib: ctypes.CDLL) -> str:
    raw = lib.fg_last_error()
    return raw.decode("utf-8", errors="replace") if raw else ""


def encode_opt_config(config: dict | str | None) -> Optional[bytes]:
    if config is None:
        return None
    if isinstance(config, str):
        return config.encode("utf-8")
    import json

    return json.dumps(config).encode("utf-8")

"""High-level Pythonic wrappers with zero-copy NumPy buffer views for `flexgraph-rs`."""
from __future__ import annotations

import ctypes
from ctypes import byref, c_double, c_size_t, c_uint8
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from flexgraph_rs._ffi import (
    FgEnergyMomentum,
    FgStepDiag,
    FgWorldInfo,
    SCHUR_ROUTE_NAMES,
    encode_opt_config,
    last_error_message,
    load_native_lib,
)


@dataclass(frozen=True)
class WorldInfo:
    num_bodies: int
    num_free_bodies: int
    num_beam_nodes: int
    total_nodes: int
    num_beams: int
    num_tendons: int
    num_tendon_actuators: int
    num_joint_actuators: int
    num_total_actuators: int
    num_contact_sites: int
    schur_route: str
    cyclomatic_number: int
    step_count: int
    sim_time: float
    failed: bool


@dataclass(frozen=True)
class StepDiag:
    step_index: int
    sim_time: float
    newton_iters: int
    residual_inf: float
    residual_l2: float
    line_search_alpha: float
    ccd_toi: float
    active_site_contacts: int
    active_beam_contacts: int
    active_self_contacts: int
    lcp_iters: int
    lcp_residual: float
    max_quat_norm_err: float
    failed: bool


@dataclass(frozen=True)
class EnergyMomentum:
    kinetic_energy: float
    elastic_energy: float
    actuation_potential_energy: float
    gravity_potential_energy: float
    total_energy: float
    linear_momentum: tuple[float, float, float]
    angular_momentum: tuple[float, float, float]


@dataclass(frozen=True)
class AdjointGradResult:
    """Output of Rust IFT reverse-mode trajectory adjoint (`fg_world_adjoint_grad`)."""

    loss: float
    grad_controls: np.ndarray
    grad_params: np.ndarray


def build_serpworm_envelope_json(
    n_segments: int = 5,
    elements_per_strip: int = 16,
    hardware: Any = None,
    ground_contact: bool = True,
    dt: float = 0.0005,
    material_profile_id: int = 0,
    yaw_motor_speed: float | None = None,
) -> str:
    """Serialize an exact `SerpWormConfig` JSON envelope matching `serpworm.physics.gpu_core.GPUCore`."""
    if hardware is not None:
        plate_masses = getattr(hardware, "plate_masses_kg", None)
        if plate_masses is not None:
            plate_masses = [float(x) for x in plate_masses[: 2 * n_segments]]
        y_slim = (
            float(hardware.yaw_speed_limit_rad_s)
            if yaw_motor_speed is None
            else float(yaw_motor_speed)
        )
        cfg = {
            "n_segments": int(n_segments),
            "elements_per_strip": int(elements_per_strip),
            "preferred_dt": float(dt),
            "beam_formulation": "corotational_euler_bernoulli",
            "strip_width_m": getattr(hardware, "strip_width_m", None),
            "strip_thickness_m": getattr(hardware, "strip_thickness_m", None),
            "stress_free_bow_m": getattr(hardware, "stress_free_bow_m", None),
            "strip_thin_shell_compat": bool(getattr(hardware, "strip_thin_shell_compat", False)),
            "strip_prestress_frac": float(getattr(hardware, "strip_prestress_frac", 0.0)),
            "strip_node_contact": bool(getattr(hardware, "strip_node_contact", False)),
            "rayleigh_alpha": float(getattr(hardware, "rayleigh_alpha", 0.05)),
            "rayleigh_beta": float(getattr(hardware, "rayleigh_beta", 0.002)),
            "plate_masses_kg": plate_masses,
            "tendon_crank_mode": getattr(hardware, "tendon_model", "legacy_wrap") == "crank_calibrated",
            "tendon_horn_radius_m": float(getattr(hardware, "tendon_horn_radius_m", 0.0235)),
            "tendon_guide_dist_m": float(getattr(hardware, "tendon_guide_dist_m", 0.0350)),
            "tendon_slack_m": float(getattr(hardware, "tendon_slack_m", 0.0)),
            "tendon_stiffness_n_m": float(getattr(hardware, "tendon_stiffness_n_m", 2000.0)),
            "tendon_damping_ns_m": float(getattr(hardware, "tendon_damping_ns_m", 0.5)),
            "tendon_y_m": float(getattr(hardware, "tendon_y_m", 0.0285)),
            "tendon_capstan_mu_theta": float(getattr(hardware, "tendon_capstan_mu_theta", 0.0)),
            "servo_torque_limit_nm": float(getattr(hardware, "servo_torque_limit_nm", 0.5)),
            "servo_speed_limit_rad_s": float(getattr(hardware, "servo_speed_limit_rad_s", 2.0)),
            "servo_deadband_rad": float(getattr(hardware, "servo_deadband_rad", 0.0)),
            "plate_stop_gap_m": float(getattr(hardware, "plate_stop_gap_m", 0.075)),
            "yaw_kp_nm_rad": float(getattr(hardware, "yaw_kp_nm_rad", 2.0)),
            "yaw_kd_nms_rad": float(getattr(hardware, "yaw_kd_nms_rad", 0.03)),
            "yaw_torque_limit_nm": float(getattr(hardware, "yaw_torque_limit_nm", 0.5)),
            "yaw_speed_limit_rad_s": y_slim,
            "yaw_static_friction_nm": float(getattr(hardware, "yaw_static_friction_nm", 0.0)),
            "ground_contact": bool(ground_contact),
            "wheel_mu": float(getattr(hardware, "wheel_mu", 0.80)),
            "wheel_mu_lat_ratio": float(getattr(hardware, "wheel_mu_lat_ratio", 1.0)),
            "wheel_rolling": float(getattr(hardware, "wheel_rolling", 0.06)),
            "wheel_mass_kg": float(getattr(hardware, "wheel_mass_kg", 0.009)),
            "wheel_patch_radius_m": float(getattr(hardware, "wheel_patch_radius_m", 0.0)),
            "wheel_pivot_mu": float(getattr(hardware, "wheel_pivot_mu", 0.65)),
            "ratchet_backlash_rad": float(getattr(hardware, "ratchet_backlash_rad", 0.0)),
            "wheel_static_mu_ratio": float(getattr(hardware, "wheel_static_mu_ratio", 1.0)),
            "material_profile_id": int(material_profile_id),
        }
    else:
        y_slim = -1.0 if yaw_motor_speed is None else float(yaw_motor_speed)
        cfg = {
            "n_segments": int(n_segments),
            "elements_per_strip": int(elements_per_strip),
            "preferred_dt": float(dt),
            "beam_formulation": "corotational_euler_bernoulli",
            "strip_width_m": None,
            "strip_thickness_m": None,
            "stress_free_bow_m": None,
            "strip_thin_shell_compat": False,
            "strip_prestress_frac": 0.0,
            "strip_node_contact": False,
            "rayleigh_alpha": 0.05,
            "rayleigh_beta": 0.002,
            "plate_masses_kg": None,
            "tendon_crank_mode": False,
            "tendon_horn_radius_m": 0.0235,
            "tendon_guide_dist_m": 0.0350,
            "tendon_slack_m": 0.0,
            "tendon_stiffness_n_m": 2000.0,
            "tendon_damping_ns_m": 0.5,
            "tendon_y_m": 0.0285,
            "tendon_capstan_mu_theta": 0.0,
            "servo_torque_limit_nm": 0.5,
            "servo_speed_limit_rad_s": 2.0,
            "servo_deadband_rad": 0.0,
            "plate_stop_gap_m": 0.075,
            "yaw_kp_nm_rad": 2.0,
            "yaw_kd_nms_rad": 0.03,
            "yaw_torque_limit_nm": 0.5,
            "yaw_speed_limit_rad_s": y_slim,
            "yaw_static_friction_nm": 0.0,
            "ground_contact": bool(ground_contact),
            "wheel_mu": 0.80,
            "wheel_mu_lat_ratio": 1.0,
            "wheel_rolling": 0.06,
            "wheel_mass_kg": 0.009,
            "wheel_patch_radius_m": 0.0,
            "wheel_pivot_mu": 0.65,
            "ratchet_backlash_rad": 0.0,
            "wheel_static_mu_ratio": 1.0,
            "material_profile_id": int(material_profile_id),
        }
    return json.dumps({"archetype": "serpworm", "config": cfg})


def _wrap_ro_f64_view(ptr: ctypes.POINTER(c_double), shape: tuple[int, ...]) -> np.ndarray:
    total = 1
    for dim in shape:
        total *= dim
    if total == 0 or not ptr:
        arr = np.empty(shape, dtype=np.float64)
        arr.setflags(write=False)
        return arr
    arr = np.ctypeslib.as_array(ptr, shape=shape)
    arr.setflags(write=False)
    return arr


class FlexGraphRsSolver:
    """Zero-copy Pythonic wrapper around a native Rust `FgWorldHandle`."""

    def __init__(self, handle: int, lib: ctypes.CDLL) -> None:
        self._lib = lib
        self._handle: int | None = None
        if not handle:
            raise RuntimeError(f"Failed to construct Rust FlexGraphWorld: {last_error_message(lib)}")
        self._handle = int(handle)
        self._info = self._query_info()

    @classmethod
    def from_json(
        cls,
        graph_json: str,
        config: dict | str | None = None,
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsSolver:
        lib = load_native_lib(lib_path)
        h = lib.fg_world_create_from_json(
            graph_json.encode("utf-8"),
            encode_opt_config(config),
        )
        return cls(h, lib)

    @classmethod
    def from_yaml(
        cls,
        graph_yaml: str,
        config: dict | str | None = None,
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsSolver:
        lib = load_native_lib(lib_path)
        h = lib.fg_world_create_from_yaml(
            graph_yaml.encode("utf-8"),
            encode_opt_config(config),
        )
        return cls(h, lib)

    @classmethod
    def from_urdf_flex(
        cls,
        urdf_xml: str,
        config: dict | str | None = None,
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsSolver:
        lib = load_native_lib(lib_path)
        h = lib.fg_world_create_from_urdf_flex(
            urdf_xml.encode("utf-8"),
            encode_opt_config(config),
        )
        return cls(h, lib)

    @classmethod
    def from_archetype(
        cls,
        archetype_name: str,
        config: dict | str | None = None,
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsSolver:
        lib = load_native_lib(lib_path)
        h = lib.fg_world_create_archetype(
            archetype_name.encode("utf-8"),
            encode_opt_config(config),
        )
        return cls(h, lib)

    @classmethod
    def from_ros_urdf(
        cls,
        urdf_xml: str,
        config: dict | str | None = None,
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsSolver:
        """Load a standard ROS URDF XML string via `urdf-rs` into a native Rust world."""
        lib = load_native_lib(lib_path)
        h = lib.fg_world_create_from_ros_urdf(
            urdf_xml.encode("utf-8"),
            encode_opt_config(config),
        )
        return cls(h, lib)

    @classmethod
    def from_mjcf(
        cls,
        mjcf_xml: str,
        config: dict | str | None = None,
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsSolver:
        """Load a MuJoCo MJCF XML string (`<mujoco>`) into a native Rust world."""
        lib = load_native_lib(lib_path)
        h = lib.fg_world_create_from_mjcf(
            mjcf_xml.encode("utf-8"),
            encode_opt_config(config),
        )
        return cls(h, lib)

    @classmethod
    def with_corrugated_pipe_mesh(
        cls,
        graph_json: str,
        config: dict | str | None = None,
        length: float = 1.0,
        r_mean: float = 0.08,
        corrugation_amp: float = 0.006,
        n_waves: float = 8.0,
        origin: tuple[float, float, float] = (0.15, 0.0, 0.08),
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsSolver:
        """Create a world mounted inside a `parry3d-f64` corrugated triangle-mesh pipe cavity."""
        lib = load_native_lib(lib_path)
        h = lib.fg_world_create_with_corrugated_pipe_mesh(
            graph_json.encode("utf-8"),
            encode_opt_config(config),
            float(length),
            float(r_mean),
            float(corrugation_amp),
            float(n_waves),
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
        )
        return cls(h, lib)

    @classmethod
    def for_serpworm(
        cls,
        n_segments: int = 5,
        elements_per_strip: int = 16,
        hardware: Any = None,
        ground_contact: bool = True,
        dt: float = 0.0005,
        contact_model: str = "kelvin",
        implicit_contact: bool = False,
        yaw_motor_speed: float | None = None,
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsSolver:
        """Construct a parity-exact Rust `SerpWorm-N` world matching `serpworm.physics.gpu_core.GPUCore`."""
        mat_map = {
            "kelvin": (0, "kelvin_floor"),
            "arpc_hard": (0, "hard_floor"),
            "arpc_soft_soil": (2, "bekker_soft_soil"),
            "arpc_carpet": (3, "carpet"),
        }
        if contact_model not in mat_map:
            raise ValueError(f"Unsupported contact_model: {contact_model!r}")
        mat_id, world_sdf = mat_map[contact_model]
        if not ground_contact:
            world_sdf = "empty"

        graph_json = build_serpworm_envelope_json(
            n_segments=n_segments,
            elements_per_strip=elements_per_strip,
            hardware=hardware,
            ground_contact=ground_contact,
            dt=dt,
            material_profile_id=mat_id,
            yaw_motor_speed=yaw_motor_speed,
        )

        if not ground_contact:
            solver_cfg: dict[str, Any] = {
                "solver_preset": "serpworm_kelvin_parity",
                "contact": {"backend": "none"},
                "world_sdf": "empty",
            }
        elif contact_model == "kelvin" and not implicit_contact:
            kn = float(getattr(hardware, "ground_kn_n_m", 5000.0)) if hardware is not None else 5000.0
            cn = float(getattr(hardware, "ground_cn_ns_m", 8.0)) if hardware is not None else 8.0
            solver_cfg = {
                "solver_preset": "serpworm_kelvin_parity",
                "contact": {"backend": "gpu_core_kelvin_parity", "kn": kn, "cn": cn},
                "world_sdf": "kelvin_floor",
            }
        else:
            solver_cfg = {
                "solver_preset": "serpworm_arpc_parity",
                "contact": {"backend": "arpc", "implicit": bool(implicit_contact)},
                "world_sdf": world_sdf,
            }
        return cls.from_json(graph_json, config=solver_cfg, lib_path=lib_path)

    def _require_handle(self) -> int:
        if self._handle is None:
            raise RuntimeError("FlexGraphRsSolver has already been closed/freed")
        return self._handle

    def _query_info(self) -> WorldInfo:
        h = self._require_handle()
        raw = FgWorldInfo()
        rc = self._lib.fg_world_get_info(h, byref(raw))
        if rc != 0:
            raise RuntimeError(f"fg_world_get_info failed ({rc}): {last_error_message(self._lib)}")
        return WorldInfo(
            num_bodies=int(raw.num_bodies),
            num_free_bodies=int(raw.num_free_bodies),
            num_beam_nodes=int(raw.num_beam_nodes),
            total_nodes=int(raw.total_nodes),
            num_beams=int(raw.num_beams),
            num_tendons=int(raw.num_tendons),
            num_tendon_actuators=int(raw.num_tendon_actuators),
            num_joint_actuators=int(raw.num_joint_actuators),
            num_total_actuators=int(raw.num_total_actuators),
            num_contact_sites=int(raw.num_contact_sites),
            schur_route=SCHUR_ROUTE_NAMES.get(int(raw.schur_route), "unknown"),
            cyclomatic_number=int(raw.cyclomatic_number),
            step_count=int(raw.step_count),
            sim_time=float(raw.sim_time),
            failed=bool(raw.failed),
        )

    @property
    def info(self) -> WorldInfo:
        self._info = self._query_info()
        return self._info

    @property
    def poses(self) -> np.ndarray:
        """Zero-copy read-only FP64 NumPy view `(total_nodes, 7)` directly over Rust state."""
        h = self._require_handle()
        ptr = self._lib.fg_world_poses_ptr(h)
        return _wrap_ro_f64_view(ptr, (self._info.total_nodes, 7))

    @property
    def vels(self) -> np.ndarray:
        """Zero-copy read-only FP64 NumPy view `(total_nodes, 6)` directly over Rust state."""
        h = self._require_handle()
        ptr = self._lib.fg_world_vels_ptr(h)
        return _wrap_ro_f64_view(ptr, (self._info.total_nodes, 6))

    @property
    def tensions(self) -> np.ndarray:
        """Zero-copy read-only FP64 NumPy view `(num_tendons,)` directly over Rust state."""
        h = self._require_handle()
        ptr = self._lib.fg_world_tensions_ptr(h)
        return _wrap_ro_f64_view(ptr, (self._info.num_tendons,))

    @property
    def contact_normals(self) -> np.ndarray:
        """Zero-copy read-only FP64 NumPy view `(num_contact_sites,)` directly over Rust state."""
        h = self._require_handle()
        ptr = self._lib.fg_world_contact_normals_ptr(h)
        return _wrap_ro_f64_view(ptr, (self._info.num_contact_sites,))

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy out `(poses [D, 7], vels [D, 6])` via `fg_world_get_state`."""
        h = self._require_handle()
        n = self._info.total_nodes
        poses = np.empty((n, 7), dtype=np.float64)
        vels = np.empty((n, 6), dtype=np.float64)
        rc = self._lib.fg_world_get_state(
            h,
            poses.ctypes.data_as(ctypes.POINTER(c_double)),
            poses.size,
            vels.ctypes.data_as(ctypes.POINTER(c_double)),
            vels.size,
        )
        if rc != 0:
            raise RuntimeError(f"fg_world_get_state failed ({rc}): {last_error_message(self._lib)}")
        return poses, vels

    def reset(self) -> None:
        h = self._require_handle()
        rc = self._lib.fg_world_reset(h)
        if rc != 0:
            raise RuntimeError(f"fg_world_reset failed ({rc}): {last_error_message(self._lib)}")

    def set_targets(self, targets: np.ndarray | list[float]) -> None:
        h = self._require_handle()
        arr = np.ascontiguousarray(targets, dtype=np.float64).reshape(-1)
        ptr = arr.ctypes.data_as(ctypes.POINTER(c_double)) if arr.size else None
        rc = self._lib.fg_world_set_targets(h, ptr, arr.size)
        if rc != 0:
            raise RuntimeError(f"fg_world_set_targets failed ({rc}): {last_error_message(self._lib)}")

    def set_tendon_targets(self, targets: np.ndarray | list[float]) -> None:
        h = self._require_handle()
        arr = np.ascontiguousarray(targets, dtype=np.float64).reshape(-1)
        ptr = arr.ctypes.data_as(ctypes.POINTER(c_double)) if arr.size else None
        rc = self._lib.fg_world_set_tendon_targets(h, ptr, arr.size)
        if rc != 0:
            raise RuntimeError(
                f"fg_world_set_tendon_targets failed ({rc}): {last_error_message(self._lib)}"
            )

    def set_joint_targets(self, targets: np.ndarray | list[float]) -> None:
        h = self._require_handle()
        arr = np.ascontiguousarray(targets, dtype=np.float64).reshape(-1)
        ptr = arr.ctypes.data_as(ctypes.POINTER(c_double)) if arr.size else None
        rc = self._lib.fg_world_set_joint_targets(h, ptr, arr.size)
        if rc != 0:
            raise RuntimeError(
                f"fg_world_set_joint_targets failed ({rc}): {last_error_message(self._lib)}"
            )

    def step(self, dt: float, steps: int = 1) -> StepDiag:
        h = self._require_handle()
        raw = FgStepDiag()
        rc = self._lib.fg_world_step(h, int(steps), float(dt), byref(raw))
        if rc != 0:
            raise RuntimeError(f"fg_world_step failed ({rc}): {last_error_message(self._lib)}")
        return StepDiag(
            step_index=int(raw.step_index),
            sim_time=float(raw.sim_time),
            newton_iters=int(raw.newton_iters),
            residual_inf=float(raw.residual_inf),
            residual_l2=float(raw.residual_l2),
            line_search_alpha=float(raw.line_search_alpha),
            ccd_toi=float(raw.ccd_toi),
            active_site_contacts=int(raw.active_site_contacts),
            active_beam_contacts=int(raw.active_beam_contacts),
            active_self_contacts=int(raw.active_self_contacts),
            lcp_iters=int(raw.lcp_iters),
            lcp_residual=float(raw.lcp_residual),
            max_quat_norm_err=float(raw.max_quat_norm_err),
            failed=bool(raw.failed),
        )

    def energy_momentum(self) -> EnergyMomentum:
        h = self._require_handle()
        raw = FgEnergyMomentum()
        rc = self._lib.fg_world_get_energy_momentum(h, byref(raw))
        if rc != 0:
            raise RuntimeError(
                f"fg_world_get_energy_momentum failed ({rc}): {last_error_message(self._lib)}"
            )
        return EnergyMomentum(
            kinetic_energy=float(raw.kinetic_energy),
            elastic_energy=float(raw.elastic_energy),
            actuation_potential_energy=float(raw.actuation_potential_energy),
            gravity_potential_energy=float(raw.gravity_potential_energy),
            total_energy=float(raw.total_energy),
            linear_momentum=(
                float(raw.linear_momentum[0]),
                float(raw.linear_momentum[1]),
                float(raw.linear_momentum[2]),
            ),
            angular_momentum=(
                float(raw.angular_momentum[0]),
                float(raw.angular_momentum[1]),
                float(raw.angular_momentum[2]),
            ),
        )

    def export_graph_json(self) -> str:
        h = self._require_handle()
        ptr = self._lib.fg_world_export_graph_json(h)
        if not ptr:
            raise RuntimeError(
                f"fg_world_export_graph_json failed: {last_error_message(self._lib)}"
            )
        try:
            return ctypes.string_at(ptr).decode("utf-8")
        finally:
            self._lib.fg_string_free(ptr)

    def adjoint_grad(
        self,
        controls: np.ndarray,
        dt: float,
        target_poses: np.ndarray,
        weight_pos: float = 1.0,
        weight_rot: float = 0.0,
        weight_ctrl: float = 0.0,
        diff_params: list[dict[str, Any]] | str | None = None,
    ) -> AdjointGradResult:
        """Compute exact Rust IFT reverse-mode trajectory gradients `(dL/du, dL/dtheta)`."""
        h = self._require_handle()
        ctrl = np.ascontiguousarray(controls, dtype=np.float64)
        if ctrl.ndim != 2:
            raise ValueError(f"controls must have shape (n_steps, n_act), got {ctrl.shape}")
        n_steps, n_act = ctrl.shape

        tgt = np.ascontiguousarray(target_poses, dtype=np.float64)
        if tgt.ndim != 3 or tgt.shape[0] != n_steps + 1 or tgt.shape[2] != 7:
            raise ValueError(
                f"target_poses must have shape ({n_steps + 1}, n_tracked_bodies, 7), got {tgt.shape}"
            )
        n_tracked = tgt.shape[1]

        if diff_params is None:
            dp_bytes = None
            raw_graph = json.loads(self.export_graph_json())
            n_params = len(raw_graph.get("diff_params", []))
        elif isinstance(diff_params, str):
            dp_bytes = diff_params.encode("utf-8")
            n_params = len(json.loads(diff_params))
        else:
            dp_bytes = json.dumps(diff_params).encode("utf-8")
            n_params = len(diff_params)

        out_loss = c_double(0.0)
        grad_u = np.zeros((n_steps, n_act), dtype=np.float64)
        grad_p = np.zeros((n_params,), dtype=np.float64)

        rc = self._lib.fg_world_adjoint_grad(
            h,
            ctrl.ctypes.data_as(ctypes.POINTER(c_double)) if ctrl.size else None,
            int(n_steps),
            int(n_act),
            float(dt),
            tgt.ctypes.data_as(ctypes.POINTER(c_double)) if tgt.size else None,
            int(n_tracked),
            float(weight_pos),
            float(weight_rot),
            float(weight_ctrl),
            dp_bytes,
            byref(out_loss),
            grad_u.ctypes.data_as(ctypes.POINTER(c_double)) if grad_u.size else None,
            grad_p.ctypes.data_as(ctypes.POINTER(c_double)) if grad_p.size else None,
            int(n_params),
        )
        if rc != 0:
            raise RuntimeError(
                f"fg_world_adjoint_grad failed ({rc}): {last_error_message(self._lib)}"
            )
        return AdjointGradResult(
            loss=float(out_loss.value),
            grad_controls=grad_u,
            grad_params=grad_p,
        )

    def get_diff_param(self, param_spec: dict[str, Any] | str) -> float:
        h = self._require_handle()
        p_bytes = (
            param_spec.encode("utf-8")
            if isinstance(param_spec, str)
            else json.dumps(param_spec).encode("utf-8")
        )
        out = c_double(0.0)
        rc = self._lib.fg_world_get_diff_param(h, p_bytes, byref(out))
        if rc != 0:
            raise RuntimeError(
                f"fg_world_get_diff_param failed ({rc}): {last_error_message(self._lib)}"
            )
        return float(out.value)

    def set_diff_param(
        self,
        param_spec: dict[str, Any] | str,
        value: float,
        clamp_bounds: bool = False,
    ) -> None:
        h = self._require_handle()
        p_bytes = (
            param_spec.encode("utf-8")
            if isinstance(param_spec, str)
            else json.dumps(param_spec).encode("utf-8")
        )
        rc = self._lib.fg_world_set_diff_param(
            h, p_bytes, float(value), 1 if clamp_bounds else 0
        )
        if rc != 0:
            raise RuntimeError(
                f"fg_world_set_diff_param failed ({rc}): {last_error_message(self._lib)}"
            )

    def close(self) -> None:
        if self._handle is not None:
            self._lib.fg_world_free(self._handle)
            self._handle = None

    def __enter__(self) -> FlexGraphRsSolver:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class FlexGraphRsBatchSolver:
    """Vectorized Rayon multi-environment wrapper over `FgBatchHandle` with preallocated NumPy buffers."""

    def __init__(self, handle: int, lib: ctypes.CDLL) -> None:
        self._lib = lib
        self._handle: int | None = None
        if not handle:
            raise RuntimeError(f"Failed to construct Rust FgBatchHandle: {last_error_message(lib)}")
        self._handle = int(handle)
        nb, na, nbeams = c_size_t(), c_size_t(), c_size_t()
        rc = lib.fg_batch_dims(handle, byref(nb), byref(na), byref(nbeams))
        if rc != 0:
            raise RuntimeError(f"fg_batch_dims failed: {last_error_message(lib)}")
        self.batch_size = int(lib.fg_batch_size(handle))
        self.num_bodies = int(nb.value)
        self.num_actuators = int(na.value)
        self.num_beams = int(nbeams.value)
        self.total_nodes = int(lib.fg_batch_total_nodes(handle))

        # Preallocated contiguous FP64/uint8 output buffers populated in-place by Rust Rayon threads
        self.poses = np.empty((self.batch_size, self.num_bodies, 7), dtype=np.float64)
        self.vels = np.empty((self.batch_size, self.num_bodies, 6), dtype=np.float64)
        self.act_state = np.empty((self.batch_size, self.num_actuators, 3), dtype=np.float64)
        self.contact_summary = np.empty((self.batch_size, 7), dtype=np.float64)
        self.failed = np.zeros((self.batch_size,), dtype=np.uint8)

    @classmethod
    def from_json(
        cls,
        graph_json: str,
        batch_size: int,
        dt: float,
        num_threads: int = 0,
        config: dict | str | None = None,
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsBatchSolver:
        lib = load_native_lib(lib_path)
        if config is None:
            h = lib.fg_batch_create_from_json(
                graph_json.encode("utf-8"),
                int(batch_size),
                float(dt),
                int(num_threads),
            )
        else:
            h = lib.fg_batch_create_from_json_with_config(
                graph_json.encode("utf-8"),
                encode_opt_config(config),
                int(batch_size),
                float(dt),
                int(num_threads),
            )
        return cls(h, lib)

    @classmethod
    def from_archetype(
        cls,
        archetype_name: str,
        batch_size: int,
        dt: float,
        num_threads: int = 0,
        config: dict | str | None = None,
        lib_path: str | Path | None = None,
    ) -> FlexGraphRsBatchSolver:
        lib = load_native_lib(lib_path)
        h = lib.fg_batch_create_archetype(
            archetype_name.encode("utf-8"),
            encode_opt_config(config),
            int(batch_size),
            float(dt),
            int(num_threads),
        )
        return cls(h, lib)

    def _require_handle(self) -> int:
        if self._handle is None:
            raise RuntimeError("FlexGraphRsBatchSolver has already been closed/destroyed")
        return self._handle

    def reset_mask(self, mask: np.ndarray | list[bool] | None = None) -> None:
        h = self._require_handle()
        if mask is None:
            rc = self._lib.fg_batch_reset_mask(h, None)
        else:
            arr = np.ascontiguousarray(mask, dtype=np.uint8).reshape(self.batch_size)
            rc = self._lib.fg_batch_reset_mask(h, arr.ctypes.data_as(ctypes.POINTER(c_uint8)))
        if rc != 0:
            raise RuntimeError(f"fg_batch_reset_mask failed ({rc}): {last_error_message(self._lib)}")

    def step(
        self,
        substeps: int = 1,
        actions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step all `B` environments in parallel via Rayon and populate preallocated NumPy arrays."""
        h = self._require_handle()
        if actions is None:
            act_ptr = None
        else:
            arr = np.ascontiguousarray(actions, dtype=np.float64)
            if arr.shape != (self.batch_size, self.num_actuators):
                raise ValueError(
                    f"actions shape {arr.shape} != expected {(self.batch_size, self.num_actuators)}"
                )
            act_ptr = arr.ctypes.data_as(ctypes.POINTER(c_double)) if arr.size else None

        rc = self._lib.fg_batch_step_into(
            h,
            int(substeps),
            act_ptr,
            self.poses.ctypes.data_as(ctypes.POINTER(c_double)) if self.poses.size else None,
            self.vels.ctypes.data_as(ctypes.POINTER(c_double)) if self.vels.size else None,
            self.act_state.ctypes.data_as(ctypes.POINTER(c_double)) if self.act_state.size else None,
            self.contact_summary.ctypes.data_as(ctypes.POINTER(c_double)),
            self.failed.ctypes.data_as(ctypes.POINTER(c_uint8)),
        )
        if rc != 0:
            raise RuntimeError(f"fg_batch_step_into failed ({rc}): {last_error_message(self._lib)}")
        return self.poses, self.vels, self.act_state, self.contact_summary, self.failed

    def get_all_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy all nodes `(B, total_nodes, 7)` and `(B, total_nodes, 6)` including beam interior nodes."""
        h = self._require_handle()
        all_poses = np.empty((self.batch_size, self.total_nodes, 7), dtype=np.float64)
        all_vels = np.empty((self.batch_size, self.total_nodes, 6), dtype=np.float64)
        rc = self._lib.fg_batch_get_all_nodes_into(
            h,
            all_poses.ctypes.data_as(ctypes.POINTER(c_double)),
            all_vels.ctypes.data_as(ctypes.POINTER(c_double)),
        )
        if rc != 0:
            raise RuntimeError(
                f"fg_batch_get_all_nodes_into failed ({rc}): {last_error_message(self._lib)}"
            )
        return all_poses, all_vels

    def close(self) -> None:
        if self._handle is not None:
            self._lib.fg_batch_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> FlexGraphRsBatchSolver:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

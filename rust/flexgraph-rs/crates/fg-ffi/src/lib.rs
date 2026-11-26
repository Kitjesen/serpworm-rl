//! `fg-ffi`: Zero-dependency pure-Rust C-ABI (`cdylib`) bridge for `flexgraph-rs`.
//!
//! Exposes two opaque handle families for zero-friction Python `ctypes` / C / C++ integration:
//! 1. **Single-environment handle** (`FgWorldHandle` / `FgSimHandle`):
//!    - `fg_world_create_from_json`, `fg_world_create_from_yaml`, `fg_world_create_from_urdf_flex`,
//!      `fg_world_create_archetype`
//!    - `fg_world_reset`, `fg_world_set_targets`, `fg_world_set_tendon_targets`, `fg_world_set_joint_targets`
//!    - `fg_world_step`, `fg_world_get_state`, `fg_world_get_info`, `fg_world_get_energy_momentum`
//!    - Zero-copy FP64 buffer pointers: `fg_world_poses_ptr`, `fg_world_vels_ptr`,
//!      `fg_world_tensions_ptr`, `fg_world_contact_normals_ptr`
//!    - `fg_world_export_graph_json`, `fg_world_free`
//! 2. **Vectorized Rayon multi-environment RL handle** (`FgBatchHandle`):
//!    - `fg_batch_create_from_json`, `fg_batch_create_from_json_with_config`, `fg_batch_create_archetype`
//!    - `fg_batch_dims`, `fg_batch_size`, `fg_batch_total_nodes`
//!    - `fg_batch_reset_mask`, `fg_batch_step_into`, `fg_batch_get_all_nodes_into`
//!    - `fg_batch_destroy`

#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_safety_doc)]

use fg_contact::{LcpSolverConfig, LcpSolverMethod};
use fg_ir::{DiffParamSpec, JointType, RigidFlexGraph, SchurRoute};
use fg_math::{Quat, Se3, Vec3, Vec6};
use fg_robots::{
    build_archetype_by_name, build_archetype_default_world, build_endoscope_spline_tube_sdf,
    load_graph_from_json_str, load_graph_from_mjcf_str, load_graph_from_ros_urdf_str,
    load_graph_from_urdf_flex_str, load_graph_from_yaml_str, EndoscopeConfig,
};
use fg_sdf::{
    corrugated_pipe_mesh, rubble_terrain_mesh, HeightFieldSdf, MaterialLibrary, PlaneSdf,
    SurfaceMaterial, WorldSdf, MAT_ACRYLIC_WALL, MAT_BEKKER_SOFT_SOIL, MAT_CARPET, MAT_GRAVEL,
    MAT_HARD_FLOOR,
};
use fg_solver::{
    compute_trajectory_adjoint, get_diff_param_value, set_diff_param_value, ContactBackend,
    FlexGraphWorld, IntegratorKind, PoseTrackingLoss, SolverConfig, SparseBackend, StepDiagnostics,
};
use rayon::prelude::*;
use serde::Deserialize;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::mem::{align_of, size_of};
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Arc;

// Compile-time soundness assertions guaranteeing `Se3` == `[f64; 7]` and `Vec6` == `[f64; 6]`
// with zero padding so zero-copy `*const f64` NumPy views are strictly sound.
const _: () = {
    assert!(size_of::<Se3>() == 7 * size_of::<f64>());
    assert!(align_of::<Se3>() == align_of::<f64>());
    assert!(size_of::<Vec6>() == 6 * size_of::<f64>());
    assert!(align_of::<Vec6>() == align_of::<f64>());
};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_last_error(msg: impl Into<String>) {
    let clean: String = msg
        .into()
        .chars()
        .map(|c| if c == '\0' { ' ' } else { c })
        .collect();
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(clean).unwrap_or_else(|_| CString::new("error").unwrap());
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new("").unwrap();
    });
}

/// Return a thread-local null-terminated UTF-8 string describing the most recent FFI error
/// (empty string if the last call succeeded). Pointer remains valid until the next FFI call
/// on the same thread.
#[no_mangle]
pub extern "C" fn fg_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

/// Free a C string allocated by `fg_world_export_graph_json`.
#[no_mangle]
pub unsafe extern "C" fn fg_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// C-ABI world metadata & topology summary returned by `fg_world_get_info`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct FgWorldInfo {
    pub num_bodies: usize,
    pub num_free_bodies: usize,
    pub num_beam_nodes: usize,
    pub total_nodes: usize,
    pub num_beams: usize,
    pub num_tendons: usize,
    pub num_tendon_actuators: usize,
    pub num_joint_actuators: usize,
    pub num_total_actuators: usize,
    pub num_contact_sites: usize,
    /// `0 = BlockDiagonalPairs`, `1 = TreeChain`, `2 = SparseGeneral`.
    pub schur_route: i32,
    pub cyclomatic_number: usize,
    pub step_count: u64,
    pub sim_time: f64,
    pub failed: u8,
}

/// C-ABI step diagnostics returned by `fg_world_step`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct FgStepDiag {
    pub step_index: u64,
    pub sim_time: f64,
    pub newton_iters: usize,
    pub residual_inf: f64,
    pub residual_l2: f64,
    pub line_search_alpha: f64,
    pub ccd_toi: f64,
    pub active_site_contacts: usize,
    pub active_beam_contacts: usize,
    pub active_self_contacts: usize,
    pub lcp_iters: usize,
    pub lcp_residual: f64,
    pub max_quat_norm_err: f64,
    pub failed: u8,
}

impl From<StepDiagnostics> for FgStepDiag {
    fn from(d: StepDiagnostics) -> Self {
        Self {
            step_index: d.step_index,
            sim_time: d.sim_time,
            newton_iters: d.newton_iters,
            residual_inf: d.residual_inf,
            residual_l2: d.residual_l2,
            line_search_alpha: d.line_search_alpha,
            ccd_toi: d.ccd_toi,
            active_site_contacts: d.active_site_contacts,
            active_beam_contacts: d.active_beam_contacts,
            active_self_contacts: d.active_self_contacts,
            lcp_iters: d.lcp_iters,
            lcp_residual: d.lcp_residual,
            max_quat_norm_err: d.max_quat_norm_err,
            failed: u8::from(d.failed),
        }
    }
}

/// C-ABI mechanical energy and spatial momentum budget report.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct FgEnergyMomentum {
    pub kinetic_energy: f64,
    pub elastic_energy: f64,
    pub actuation_potential_energy: f64,
    pub gravity_potential_energy: f64,
    pub total_energy: f64,
    pub linear_momentum: [f64; 3],
    pub angular_momentum: [f64; 3],
}

/// Optional JSON configuration envelope accepted by `fg_world_create_*` and `fg_batch_create_*`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct FfiConfigJson {
    /// Preset: `"default"` | `"serpworm_kelvin_parity"` | `"serpworm_arpc_parity"` |
    /// `"nonlinear_newmark"` | `"delassus_socp"`.
    solver_preset: Option<String>,
    /// Optional explicit full `SolverConfig` override.
    solver_config: Option<SolverConfig>,
    /// Optional granular overrides applied on top of `solver_preset`.
    integrator: Option<IntegratorKind>,
    contact: Option<ContactBackend>,
    /// Optional v2 LCP solver selector (`"block_pgs"` | `"alart_curnier_newton"` | `"clarabel_socp"`).
    lcp_solver: Option<String>,
    /// Optional v2 sparse direct solver backend (`"auto"` | `"custom_block_csc"` | `"faer"`).
    sparse_backend: Option<String>,
    /// Optional v2 Tight Inclusion conservative CCD flag (`Wang et al. TOG 2021`).
    use_tight_inclusion_ccd: Option<bool>,
    use_strain_jacobian: Option<bool>,
    world_fixed_inertia: Option<bool>,
    enable_self_contact: Option<bool>,
    newton_max_iters: Option<usize>,
    newton_tol: Option<f64>,
    /// World SDF preset: `"flat_ground"` | `"kelvin_floor"` | `"bekker_soft_soil"` | `"carpet"` |
    /// `"gravel"` | `"empty"` | `"endoscope_tube"` | `"archetype_default"` | `"heightfield"` |
    /// `"rubble_terrain"`.
    world_sdf: Option<String>,
    /// Optional `[slope_x, slope_y, wave_amp, wave_kx, wave_ky]` when `world_sdf == "heightfield"`.
    ground_geom: Option<[f64; 5]>,
    /// Material profile ID for `"heightfield"` / `"flat_ground"` (default 0 = hard floor).
    ground_mat_id: Option<usize>,
}

fn build_solver_and_sdf(
    config_json: Option<&str>,
    archetype_hint: Option<&str>,
) -> Result<(SolverConfig, WorldSdf), String> {
    let parsed: FfiConfigJson = match config_json {
        Some(s) if !s.trim().is_empty() => {
            serde_json::from_str(s).map_err(|e| format!("invalid config_json: {e}"))?
        }
        _ => FfiConfigJson::default(),
    };

    let mut solver_cfg = if let Some(explicit) = parsed.solver_config {
        explicit
    } else {
        match parsed
            .solver_preset
            .as_deref()
            .unwrap_or("default")
            .to_ascii_lowercase()
            .as_str()
        {
            "default" => SolverConfig::default(),
            "serpworm_kelvin_parity" | "kelvin_parity" => SolverConfig::serpworm_kelvin_parity(),
            "serpworm_arpc_parity" | "arpc_parity" => SolverConfig::serpworm_arpc_parity(),
            "nonlinear_newmark" | "newmark" => SolverConfig::nonlinear_newmark(),
            "delassus_socp" | "clarabel_socp" | "socp" => SolverConfig::delassus_socp(),
            other => return Err(format!("unknown solver_preset '{other}'")),
        }
    };

    if let Some(integ) = parsed.integrator {
        solver_cfg.integrator = integ;
    }
    if let Some(con) = parsed.contact {
        solver_cfg.contact = con;
    }
    if let Some(lcp_name) = parsed.lcp_solver.as_deref() {
        let method = match lcp_name.to_ascii_lowercase().as_str() {
            "block_pgs" | "pgs" => LcpSolverMethod::BlockPgs,
            "alart_curnier_newton" | "newton" | "ac_newton" => LcpSolverMethod::AlartCurnierNewton,
            "clarabel_socp" | "clarabel" | "socp" => LcpSolverMethod::ClarabelSocp,
            other => return Err(format!("unknown lcp_solver '{other}'")),
        };
        match &mut solver_cfg.contact {
            ContactBackend::DelassusLcp { lcp, .. } => {
                lcp.method = method;
            }
            other => {
                *other = ContactBackend::DelassusLcp {
                    lcp: LcpSolverConfig {
                        method,
                        ..LcpSolverConfig::default()
                    },
                    dedup_radius: 0.002,
                    baumgarte_beta: 0.2,
                };
            }
        }
    }
    if let Some(sb_name) = parsed.sparse_backend.as_deref() {
        solver_cfg.sparse_backend = match sb_name.to_ascii_lowercase().as_str() {
            "auto" => SparseBackend::Auto,
            "custom" | "custom_block_csc" | "block_csc" => SparseBackend::CustomBlockCsc,
            "faer" | "faer_supernodal" | "supernodal" => SparseBackend::FaerSupernodal,
            other => return Err(format!("unknown sparse_backend '{other}'")),
        };
    }
    if let Some(ti) = parsed.use_tight_inclusion_ccd {
        solver_cfg.use_tight_inclusion_ccd = ti;
        if ti {
            solver_cfg.enable_ccd_clamping = true;
        }
    }
    if let Some(sj) = parsed.use_strain_jacobian {
        solver_cfg.use_strain_jacobian = sj;
    }
    if let Some(wfi) = parsed.world_fixed_inertia {
        solver_cfg.world_fixed_inertia = wfi;
    }
    if let Some(sc) = parsed.enable_self_contact {
        solver_cfg.enable_self_contact = sc;
    }
    if let Some(mi) = parsed.newton_max_iters {
        solver_cfg.newton_max_iters = mi;
    }
    if let Some(tol) = parsed.newton_tol {
        solver_cfg.newton_tol = tol;
    }

    let sdf_kind = parsed
        .world_sdf
        .as_deref()
        .unwrap_or(if archetype_hint.is_some() {
            "archetype_default"
        } else {
            "flat_ground"
        })
        .to_ascii_lowercase();

    let sdf = match sdf_kind.as_str() {
        "archetype_default" => match archetype_hint {
            Some(name) => build_archetype_default_world(name)
                .unwrap_or_else(|_| WorldSdf::flat_ground()),
            None => WorldSdf::flat_ground(),
        },
        "flat_ground" | "hard_floor" => {
            let mat = parsed.ground_mat_id.unwrap_or(MAT_HARD_FLOOR);
            let mut w = WorldSdf::new(MaterialLibrary::standard());
            w.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, mat)));
            w
        }
        "kelvin_floor" => {
            let mut lib = MaterialLibrary::standard();
            if let Some(m0) = lib.get_mut(MAT_HARD_FLOOR) {
                *m0 = SurfaceMaterial::kelvin_floor();
            }
            let mut w = WorldSdf::new(lib);
            w.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, MAT_HARD_FLOOR)));
            w
        }
        "bekker_soft_soil" | "soft_soil" => {
            let mut w = WorldSdf::new(MaterialLibrary::standard());
            w.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, MAT_BEKKER_SOFT_SOIL)));
            w
        }
        "carpet" => {
            let mut w = WorldSdf::new(MaterialLibrary::standard());
            w.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, MAT_CARPET)));
            w
        }
        "gravel" => {
            let mut w = WorldSdf::new(MaterialLibrary::standard());
            w.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, MAT_GRAVEL)));
            w
        }
        "empty" | "none" => WorldSdf::new(MaterialLibrary::standard()),
        "endoscope_tube" | "spline_tube" => {
            let mut w = WorldSdf::new(MaterialLibrary::standard());
            let tube = build_endoscope_spline_tube_sdf(&EndoscopeConfig::default());
            w.add_unbounded(Arc::new(tube));
            w
        }
        "heightfield" => {
            let g = parsed.ground_geom.unwrap_or([0.0; 5]);
            let mat = parsed.ground_mat_id.unwrap_or(MAT_HARD_FLOOR);
            let mut w = WorldSdf::new(MaterialLibrary::standard());
            w.add_unbounded(Arc::new(HeightFieldSdf::analytic(
                g[0], g[1], g[2], g[3], g[4], 0.0, mat,
            )));
            w
        }
        "rubble_terrain" => {
            let mat = parsed.ground_mat_id.unwrap_or(MAT_GRAVEL);
            let rubble = rubble_terrain_mesh([-2.0, -2.0], [4.0, 4.0], 20, 20, 0.025, 42, mat)
                .map_err(|e| format!("failed to build rubble_terrain_mesh: {e}"))?;
            let mut w = WorldSdf::new(MaterialLibrary::standard());
            w.add_unbounded(Arc::new(rubble));
            w
        }
        other => return Err(format!("unknown world_sdf '{other}'")),
    };

    Ok((solver_cfg, sdf))
}

unsafe fn read_c_str<'a>(ptr: *const c_char, name: &str) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err(format!("null pointer passed for '{name}'"));
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|e| format!("invalid UTF-8 in '{name}': {e}"))
}

unsafe fn read_opt_c_str<'a>(ptr: *const c_char) -> Result<Option<&'a str>, String> {
    if ptr.is_null() {
        return Ok(None);
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map(Some)
        .map_err(|e| format!("invalid UTF-8 in optional string: {e}"))
}

fn route_to_i32(route: SchurRoute) -> i32 {
    match route {
        SchurRoute::BlockDiagonalPairs => 0,
        SchurRoute::TreeChain => 1,
        SchurRoute::SparseGeneral => 2,
    }
}

// ============================================================================
// 1. Single-Environment Handle (`FgWorldHandle` / `FgSimHandle`)
// ============================================================================

/// Opaque single-environment simulation handle owned by Rust across the C-ABI.
pub struct FgWorldHandle {
    world: FlexGraphWorld,
    /// Contiguous cache of per-tendon tensions (N) for zero-copy `fg_world_tensions_ptr`.
    tension_cache: Vec<f64>,
}

/// Alias matching sibling RL design docs (`FgSimHandle` == `FgWorldHandle`).
pub type FgSimHandle = FgWorldHandle;

impl FgWorldHandle {
    fn from_world(world: FlexGraphWorld) -> Self {
        let mut h = Self {
            tension_cache: vec![0.0; world.graph.tendons.len()],
            world,
        };
        h.sync_caches();
        h
    }

    fn sync_caches(&mut self) {
        self.tension_cache.resize(self.world.state.actuation.tendons.len(), 0.0);
        for (dst, src) in self
            .tension_cache
            .iter_mut()
            .zip(self.world.state.actuation.tendons.iter())
        {
            *dst = src.tension;
        }
    }
}

fn create_world_handle(
    graph: RigidFlexGraph,
    config_json: Option<&str>,
    archetype_hint: Option<&str>,
) -> Result<*mut FgWorldHandle, String> {
    let (solver_cfg, sdf) = build_solver_and_sdf(config_json, archetype_hint)?;
    let world = FlexGraphWorld::new(graph, sdf, solver_cfg).map_err(|e| e.to_string())?;
    Ok(Box::into_raw(Box::new(FgWorldHandle::from_world(world))))
}

/// Create a `FlexGraphWorld` from a JSON string (raw `RigidFlexGraph` or archetype envelope)
/// and optional `config_json` (may be NULL). Returns NULL on error; check `fg_last_error()`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_create_from_json(
    graph_json: *const c_char,
    config_json: *const c_char,
) -> *mut FgWorldHandle {
    clear_last_error();
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<*mut FgWorldHandle, String> {
        let g_str = read_c_str(graph_json, "graph_json")?;
        let c_str = read_opt_c_str(config_json)?;
        let graph = load_graph_from_json_str(g_str).map_err(|e| e.to_string())?;
        create_world_handle(graph, c_str, None)
    }));
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic inside fg_world_create_from_json");
            ptr::null_mut()
        }
    }
}

/// Create a `FlexGraphWorld` from a YAML string and optional `config_json` (may be NULL).
#[no_mangle]
pub unsafe extern "C" fn fg_world_create_from_yaml(
    graph_yaml: *const c_char,
    config_json: *const c_char,
) -> *mut FgWorldHandle {
    clear_last_error();
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<*mut FgWorldHandle, String> {
        let y_str = read_c_str(graph_yaml, "graph_yaml")?;
        let c_str = read_opt_c_str(config_json)?;
        let graph = load_graph_from_yaml_str(y_str).map_err(|e| e.to_string())?;
        create_world_handle(graph, c_str, None)
    }));
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic inside fg_world_create_from_yaml");
            ptr::null_mut()
        }
    }
}

/// Create a `FlexGraphWorld` from a URDF-Flex XML string and optional `config_json` (may be NULL).
#[no_mangle]
pub unsafe extern "C" fn fg_world_create_from_urdf_flex(
    urdf_xml: *const c_char,
    config_json: *const c_char,
) -> *mut FgWorldHandle {
    clear_last_error();
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<*mut FgWorldHandle, String> {
        let u_str = read_c_str(urdf_xml, "urdf_xml")?;
        let c_str = read_opt_c_str(config_json)?;
        let graph = load_graph_from_urdf_flex_str(u_str).map_err(|e| e.to_string())?;
        create_world_handle(graph, c_str, None)
    }));
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic inside fg_world_create_from_urdf_flex");
            ptr::null_mut()
        }
    }
}

/// Create a `FlexGraphWorld` for one of the 7 built-in archetypes (`serpworm_1/3/5/7`,
/// `bristle_worm`, `tendon_endoscope_3d`, `flexspine_centipede`, `tensegrity_6bar`,
/// `soft_octopus_arm_der`, `rigid_flex_grid_4x4`, `rigid_flex_grid_6x6`).
#[no_mangle]
pub unsafe extern "C" fn fg_world_create_archetype(
    archetype_name: *const c_char,
    config_json: *const c_char,
) -> *mut FgWorldHandle {
    clear_last_error();
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<*mut FgWorldHandle, String> {
        let name = read_c_str(archetype_name, "archetype_name")?;
        let c_str = read_opt_c_str(config_json)?;
        let graph = build_archetype_by_name(name).map_err(|e| e.to_string())?;
        create_world_handle(graph, c_str, Some(name))
    }));
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic inside fg_world_create_archetype");
            ptr::null_mut()
        }
    }
}

/// Create a `FlexGraphWorld` from a standard ROS URDF XML string (via `urdf-rs`) and optional
/// `config_json` (may be NULL). Returns NULL on error; check `fg_last_error()`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_create_from_ros_urdf(
    urdf_xml: *const c_char,
    config_json: *const c_char,
) -> *mut FgWorldHandle {
    clear_last_error();
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<*mut FgWorldHandle, String> {
        let u_str = read_c_str(urdf_xml, "urdf_xml")?;
        let c_str = read_opt_c_str(config_json)?;
        let graph = load_graph_from_ros_urdf_str(u_str).map_err(|e| e.to_string())?;
        create_world_handle(graph, c_str, None)
    }));
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic inside fg_world_create_from_ros_urdf");
            ptr::null_mut()
        }
    }
}

/// Create a `FlexGraphWorld` from a MuJoCo MJCF XML string (`<mujoco>`) and optional
/// `config_json` (may be NULL). Returns NULL on error; check `fg_last_error()`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_create_from_mjcf(
    mjcf_xml: *const c_char,
    config_json: *const c_char,
) -> *mut FgWorldHandle {
    clear_last_error();
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<*mut FgWorldHandle, String> {
        let m_str = read_c_str(mjcf_xml, "mjcf_xml")?;
        let c_str = read_opt_c_str(config_json)?;
        let graph = load_graph_from_mjcf_str(m_str).map_err(|e| e.to_string())?;
        create_world_handle(graph, c_str, None)
    }));
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic inside fg_world_create_from_mjcf");
            ptr::null_mut()
        }
    }
}

/// Create a `FlexGraphWorld` from `graph_json` mounted inside a `parry3d-f64` BVH-accelerated
/// corrugated triangle-mesh pipe cavity (`corrugated_pipe_mesh`) registered for both SDF queries
/// and Tight Inclusion Point-Triangle CCD.
#[no_mangle]
pub unsafe extern "C" fn fg_world_create_with_corrugated_pipe_mesh(
    graph_json: *const c_char,
    config_json: *const c_char,
    length: f64,
    r_mean: f64,
    corrugation_amp: f64,
    n_waves: f64,
    origin_x: f64,
    origin_y: f64,
    origin_z: f64,
) -> *mut FgWorldHandle {
    clear_last_error();
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<*mut FgWorldHandle, String> {
        let g_str = read_c_str(graph_json, "graph_json")?;
        let c_str = read_opt_c_str(config_json)?;
        let graph = load_graph_from_json_str(g_str).map_err(|e| e.to_string())?;
        let (solver_cfg, _) = build_solver_and_sdf(c_str, None)?;
        let pipe = corrugated_pipe_mesh(
            length,
            r_mean,
            corrugation_amp,
            n_waves,
            36,
            20,
            Vec3::new(origin_x, origin_y, origin_z),
            MAT_ACRYLIC_WALL,
        )
        .map_err(|e| format!("failed to build corrugated_pipe_mesh: {e}"))?;
        let mut world = FlexGraphWorld::new(
            graph,
            WorldSdf::new(MaterialLibrary::standard()),
            solver_cfg,
        )
        .map_err(|e| e.to_string())?;
        world.register_static_trimesh(Arc::new(pipe));
        Ok(Box::into_raw(Box::new(FgWorldHandle::from_world(world))))
    }));
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic inside fg_world_create_with_corrugated_pipe_mesh");
            ptr::null_mut()
        }
    }
}

/// Compute exact IFT reverse-mode trajectory gradients (`dL/du`, `dL/dtheta`) using Rust's cached
/// per-step factored bipartite operators (`fg_solver::compute_trajectory_adjoint`).
///
/// - `controls`: row-major `[n_steps, n_act]` actuator commands (may be NULL if `n_steps*n_act == 0`).
/// - `target_poses`: row-major `[n_steps + 1, n_tracked_bodies, 7]` target SE(3) poses `[px,py,pz,qx,qy,qz,qw]`.
/// - `diff_params_json`: optional JSON array of `DiffParamSpec` (if NULL/empty, uses `world.graph.diff_params`).
/// - `out_loss`: optional output pointer for scalar trajectory loss.
/// - `out_grad_controls`: optional output buffer `[n_steps * n_act]` for `dL/du`.
/// - `out_grad_params`: optional output buffer `[out_grad_params_len]` for `dL/dtheta` (`>= n_params`).
///
/// Preserves `world.state` on exit. Returns `0` on success.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn fg_world_adjoint_grad(
    handle: *mut FgWorldHandle,
    controls: *const f64,
    n_steps: usize,
    n_act: usize,
    dt: f64,
    target_poses: *const f64,
    n_tracked_bodies: usize,
    weight_pos: f64,
    weight_rot: f64,
    weight_ctrl: f64,
    diff_params_json: *const c_char,
    out_loss: *mut f64,
    out_grad_controls: *mut f64,
    out_grad_params: *mut f64,
    out_grad_params_len: usize,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_mut() else {
        set_last_error("null handle in fg_world_adjoint_grad");
        return -1;
    };

    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let mut ctrl_vecs = Vec::with_capacity(n_steps);
        if n_steps > 0 && n_act > 0 {
            if controls.is_null() {
                return Err("null controls pointer with non-zero n_steps * n_act".to_string());
            }
            let flat = std::slice::from_raw_parts(controls, n_steps * n_act);
            for k in 0..n_steps {
                ctrl_vecs.push(flat[k * n_act..(k + 1) * n_act].to_vec());
            }
        } else {
            for _ in 0..n_steps {
                ctrl_vecs.push(Vec::new());
            }
        }

        let mut targets_se3 = Vec::with_capacity(n_steps + 1);
        if n_tracked_bodies > 0 {
            if target_poses.is_null() {
                return Err("null target_poses pointer with non-zero n_tracked_bodies".to_string());
            }
            let flat = std::slice::from_raw_parts(target_poses, (n_steps + 1) * n_tracked_bodies * 7);
            for k in 0..=n_steps {
                let mut row = Vec::with_capacity(n_tracked_bodies);
                for b in 0..n_tracked_bodies {
                    let off = (k * n_tracked_bodies + b) * 7;
                    row.push(Se3::new(
                        Vec3::new(flat[off], flat[off + 1], flat[off + 2]),
                        Quat::new(flat[off + 3], flat[off + 4], flat[off + 5], flat[off + 6])
                            .normalized(),
                    ));
                }
                targets_se3.push(row);
            }
        } else {
            for _ in 0..=n_steps {
                targets_se3.push(Vec::new());
            }
        }

        let diff_params: Vec<DiffParamSpec> = match read_opt_c_str(diff_params_json)? {
            Some(s) if !s.trim().is_empty() => {
                serde_json::from_str(s).map_err(|e| format!("invalid diff_params_json: {e}"))?
            }
            _ => h.world.graph.diff_params.clone(),
        };

        if !out_grad_params.is_null() && out_grad_params_len < diff_params.len() {
            return Err(format!(
                "out_grad_params buffer too small: got {out_grad_params_len}, need {}",
                diff_params.len()
            ));
        }

        let mut loss = PoseTrackingLoss::new(targets_se3, weight_pos, weight_rot);
        loss.weight_ctrl = weight_ctrl;

        let adj = compute_trajectory_adjoint(&mut h.world, &ctrl_vecs, dt, &diff_params, &loss)
            .map_err(|e| e.to_string())?;

        if let Some(ol) = out_loss.as_mut() {
            *ol = adj.loss;
        }
        if !out_grad_controls.is_null() && n_steps > 0 && n_act > 0 {
            let dst = std::slice::from_raw_parts_mut(out_grad_controls, n_steps * n_act);
            for k in 0..n_steps {
                let row = &adj.grad_controls[k];
                for a in 0..n_act {
                    dst[k * n_act + a] = row.get(a).copied().unwrap_or(0.0);
                }
            }
        }
        if !out_grad_params.is_null() && !adj.grad_params.is_empty() {
            let dst = std::slice::from_raw_parts_mut(out_grad_params, adj.grad_params.len());
            dst.copy_from_slice(&adj.grad_params);
        }
        h.sync_caches();
        Ok(())
    }));

    match res {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -3
        }
        Err(_) => {
            set_last_error("panic inside fg_world_adjoint_grad");
            -4
        }
    }
}

/// Read the current physical scalar value of a JSON-serialized `DiffParamSpec` from `handle`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_get_diff_param(
    handle: *const FgWorldHandle,
    param_json: *const c_char,
    out_value: *mut f64,
) -> c_int {
    clear_last_error();
    let (Some(h), Some(out)) = (handle.as_ref(), out_value.as_mut()) else {
        set_last_error("null pointer in fg_world_get_diff_param");
        return -1;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<f64, String> {
        let p_str = read_c_str(param_json, "param_json")?;
        let spec: DiffParamSpec =
            serde_json::from_str(p_str).map_err(|e| format!("invalid param_json: {e}"))?;
        Ok(get_diff_param_value(&h.world, &spec))
    }));
    match res {
        Ok(Ok(val)) => {
            *out = val;
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -2
        }
        Err(_) => {
            set_last_error("panic inside fg_world_get_diff_param");
            -4
        }
    }
}

/// Set the physical scalar value of a JSON-serialized `DiffParamSpec` on `handle`
/// (rebuilding cached beam matrices automatically when required).
#[no_mangle]
pub unsafe extern "C" fn fg_world_set_diff_param(
    handle: *mut FgWorldHandle,
    param_json: *const c_char,
    value: f64,
    clamp_bounds: u8,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_mut() else {
        set_last_error("null handle in fg_world_set_diff_param");
        return -1;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let p_str = read_c_str(param_json, "param_json")?;
        let spec: DiffParamSpec =
            serde_json::from_str(p_str).map_err(|e| format!("invalid param_json: {e}"))?;
        set_diff_param_value(&mut h.world, &spec, value, clamp_bounds != 0);
        Ok(())
    }));
    match res {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -2
        }
        Err(_) => {
            set_last_error("panic inside fg_world_set_diff_param");
            -4
        }
    }
}

/// Reset world state to its initial rest configuration. Returns `0` on success.
#[no_mangle]
pub unsafe extern "C" fn fg_world_reset(handle: *mut FgWorldHandle) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_mut() else {
        set_last_error("null handle in fg_world_reset");
        return -1;
    };
    h.world.reset();
    h.sync_caches();
    0
}

/// Set unified actuator targets `[tendon_targets..., joint_targets...]`. Returns `0` on success.
#[no_mangle]
pub unsafe extern "C" fn fg_world_set_targets(
    handle: *mut FgWorldHandle,
    targets: *const f64,
    len: usize,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_mut() else {
        set_last_error("null handle in fg_world_set_targets");
        return -1;
    };
    if len == 0 {
        return 0;
    }
    if targets.is_null() {
        set_last_error("null targets pointer with non-zero len");
        return -2;
    }
    let slice = std::slice::from_raw_parts(targets, len);
    h.world.set_all_targets(slice);
    0
}

/// Set tendon actuator targets only. Returns `0` on success.
#[no_mangle]
pub unsafe extern "C" fn fg_world_set_tendon_targets(
    handle: *mut FgWorldHandle,
    targets: *const f64,
    len: usize,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_mut() else {
        set_last_error("null handle in fg_world_set_tendon_targets");
        return -1;
    };
    if len == 0 {
        return 0;
    }
    if targets.is_null() {
        set_last_error("null targets pointer with non-zero len");
        return -2;
    }
    let slice = std::slice::from_raw_parts(targets, len);
    h.world.set_tendon_targets(slice);
    0
}

/// Set articulated joint actuator targets only. Returns `0` on success.
#[no_mangle]
pub unsafe extern "C" fn fg_world_set_joint_targets(
    handle: *mut FgWorldHandle,
    targets: *const f64,
    len: usize,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_mut() else {
        set_last_error("null handle in fg_world_set_joint_targets");
        return -1;
    };
    if len == 0 {
        return 0;
    }
    if targets.is_null() {
        set_last_error("null targets pointer with non-zero len");
        return -2;
    }
    let slice = std::slice::from_raw_parts(targets, len);
    h.world.set_joint_targets(slice);
    0
}

/// Advance the world by `n_steps` consecutive time steps of size `dt` (s).
/// `out_diag` may be NULL. Returns `0` on success, nonzero error code otherwise.
#[no_mangle]
pub unsafe extern "C" fn fg_world_step(
    handle: *mut FgWorldHandle,
    n_steps: usize,
    dt: f64,
    out_diag: *mut FgStepDiag,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_mut() else {
        set_last_error("null handle in fg_world_step");
        return -1;
    };
    let res = catch_unwind(AssertUnwindSafe(|| h.world.step_n(n_steps, dt)));
    match res {
        Ok(Ok(diag)) => {
            h.sync_caches();
            if let Some(out) = out_diag.as_mut() {
                *out = FgStepDiag::from(diag);
            }
            0
        }
        Ok(Err(e)) => {
            h.sync_caches();
            set_last_error(e.to_string());
            -3
        }
        Err(_) => {
            set_last_error("panic inside fg_world_step");
            -4
        }
    }
}

/// Copy global node poses (`7 * total_nodes`) and/or velocities (`6 * total_nodes`) into caller
/// buffers. Either `poses_out` or `vels_out` may be NULL to skip copying that array.
/// Returns `0` on success.
#[no_mangle]
pub unsafe extern "C" fn fg_world_get_state(
    handle: *const FgWorldHandle,
    poses_out: *mut f64,
    poses_len: usize,
    vels_out: *mut f64,
    vels_len: usize,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_ref() else {
        set_last_error("null handle in fg_world_get_state");
        return -1;
    };
    let n = h.world.total_nodes();
    if !poses_out.is_null() {
        let req = n * 7;
        if poses_len < req {
            set_last_error(format!(
                "poses_out buffer too small: got {poses_len}, need {req}"
            ));
            return -2;
        }
        let src = h.world.state.poses.as_ptr() as *const f64;
        ptr::copy_nonoverlapping(src, poses_out, req);
    }
    if !vels_out.is_null() {
        let req = n * 6;
        if vels_len < req {
            set_last_error(format!(
                "vels_out buffer too small: got {vels_len}, need {req}"
            ));
            return -2;
        }
        let src = h.world.state.vels.as_ptr() as *const f64;
        ptr::copy_nonoverlapping(src, vels_out, req);
    }
    0
}

/// Zero-copy read-only pointer to contiguous FP64 node poses `[total_nodes, 7]`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_poses_ptr(handle: *const FgWorldHandle) -> *const f64 {
    match handle.as_ref() {
        Some(h) => h.world.state.poses.as_ptr() as *const f64,
        None => ptr::null(),
    }
}

/// Zero-copy read-only pointer to contiguous FP64 node velocities `[total_nodes, 6]`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_vels_ptr(handle: *const FgWorldHandle) -> *const f64 {
    match handle.as_ref() {
        Some(h) => h.world.state.vels.as_ptr() as *const f64,
        None => ptr::null(),
    }
}

/// Zero-copy read-only pointer to contiguous FP64 per-tendon tensions `[num_tendons]`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_tensions_ptr(handle: *const FgWorldHandle) -> *const f64 {
    match handle.as_ref() {
        Some(h) => h.tension_cache.as_ptr(),
        None => ptr::null(),
    }
}

/// Zero-copy read-only pointer to contiguous FP64 per-site contact normal forces `[num_contact_sites]`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_contact_normals_ptr(
    handle: *const FgWorldHandle,
) -> *const f64 {
    match handle.as_ref() {
        Some(h) => h.world.state.contact_normal_forces.as_ptr(),
        None => ptr::null(),
    }
}

/// Query topology metadata, actuator dimensions, Schur route, and simulation time.
#[no_mangle]
pub unsafe extern "C" fn fg_world_get_info(
    handle: *const FgWorldHandle,
    out_info: *mut FgWorldInfo,
) -> c_int {
    clear_last_error();
    let (Some(h), Some(out)) = (handle.as_ref(), out_info.as_mut()) else {
        set_last_error("null pointer in fg_world_get_info");
        return -1;
    };
    let w = &h.world;
    *out = FgWorldInfo {
        num_bodies: w.analysis.num_bodies,
        num_free_bodies: w.analysis.num_free_bodies,
        num_beam_nodes: w.analysis.indexing.num_beam_nodes,
        total_nodes: w.analysis.indexing.total_nodes,
        num_beams: w.graph.beams.len(),
        num_tendons: w.graph.tendons.len(),
        num_tendon_actuators: w.num_tendon_actuators(),
        num_joint_actuators: w.num_joint_actuators(),
        num_total_actuators: w.num_total_actuators(),
        num_contact_sites: w.graph.contact_sites.len(),
        schur_route: route_to_i32(w.analysis.route),
        cyclomatic_number: w.analysis.cyclomatic_number,
        step_count: w.state.step_count,
        sim_time: w.state.time,
        failed: u8::from(w.state.failed),
    };
    0
}

/// Compute mechanical energy and spatial momentum budget for conservation checks.
#[no_mangle]
pub unsafe extern "C" fn fg_world_get_energy_momentum(
    handle: *const FgWorldHandle,
    out_em: *mut FgEnergyMomentum,
) -> c_int {
    clear_last_error();
    let (Some(h), Some(out)) = (handle.as_ref(), out_em.as_mut()) else {
        set_last_error("null pointer in fg_world_get_energy_momentum");
        return -1;
    };
    let rep = h.world.energy_and_momentum();
    *out = FgEnergyMomentum {
        kinetic_energy: rep.kinetic_energy,
        elastic_energy: rep.elastic_energy,
        actuation_potential_energy: rep.actuation_potential_energy,
        gravity_potential_energy: rep.gravity_potential_energy,
        total_energy: rep.total_energy,
        linear_momentum: rep.linear_momentum.0,
        angular_momentum: rep.angular_momentum.0,
    };
    0
}

/// Export the active `RigidFlexGraph` as a newly allocated JSON C string.
/// Caller must release the returned pointer via `fg_string_free`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_export_graph_json(handle: *const FgWorldHandle) -> *mut c_char {
    clear_last_error();
    let Some(h) = handle.as_ref() else {
        set_last_error("null handle in fg_world_export_graph_json");
        return ptr::null_mut();
    };
    match h.world.graph.to_json_string() {
        Ok(s) => match CString::new(s) {
            Ok(cs) => cs.into_raw(),
            Err(e) => {
                set_last_error(format!("interior NUL in graph JSON: {e}"));
                ptr::null_mut()
            }
        },
        Err(e) => {
            set_last_error(e.to_string());
            ptr::null_mut()
        }
    }
}

/// Destroy a single-world handle previously returned by `fg_world_create_*`.
#[no_mangle]
pub unsafe extern "C" fn fg_world_free(handle: *mut FgWorldHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

// ============================================================================
// 2. Vectorized Rayon RL Batch Handle (`FgBatchHandle`)
// ============================================================================

/// Precomputed per-actuator metadata for fast lock-free telemetry extraction in `fg_batch_step_into`.
#[derive(Clone, Debug)]
struct BatchActuatorMap {
    /// Sorted unique tendon actuator IDs (`0..n_tendon_act`) and their member tendon indices.
    tendon_groups: Vec<(usize, Vec<usize>)>,
    /// Sorted unique joint actuator IDs paired with their `graph.joints` index.
    joint_indices: Vec<usize>,
    /// IMU body index matching `flexgraph.env.FlexGraphEnv` (`0` unless body 0 is fixed).
    imu_body_idx: usize,
    /// Total mass of non-fixed rigid bodies (kg) for normal-force weight normalization.
    total_free_mass_kg: f64,
    /// Per-contact-site lateral side sign (`+1` left y>0, `-1` right y<0, `0` center).
    site_side_signs: Vec<f64>,
}

impl BatchActuatorMap {
    fn from_world(w: &FlexGraphWorld) -> Self {
        let mut t_ids: Vec<usize> = w.graph.tendons.iter().map(|t| t.actuator_id).collect();
        t_ids.sort_unstable();
        t_ids.dedup();
        let tendon_groups: Vec<(usize, Vec<usize>)> = t_ids
            .into_iter()
            .map(|aid| {
                let members: Vec<usize> = w
                    .graph
                    .tendons
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.actuator_id == aid)
                    .map(|(idx, _)| idx)
                    .collect();
                (aid, members)
            })
            .collect();

        let mut j_pairs: Vec<(usize, usize)> = w
            .graph
            .joints
            .iter()
            .enumerate()
            .filter_map(|(idx, j)| j.actuator_id.map(|aid| (aid, idx)))
            .collect();
        j_pairs.sort_by_key(|&(aid, _)| aid);
        j_pairs.dedup_by_key(|p| p.0);
        let joint_indices: Vec<usize> = j_pairs.into_iter().map(|(_, idx)| idx).collect();

        let imu_body_idx = if w.graph.bodies.len() <= 1 {
            0
        } else if w.graph.bodies[0].is_fixed {
            1
        } else {
            0
        };
        let total_free_mass_kg = w
            .graph
            .bodies
            .iter()
            .filter(|b| !b.is_fixed)
            .map(|b| b.mass)
            .sum::<f64>()
            .max(1e-3);
        let site_side_signs = w
            .graph
            .contact_sites
            .iter()
            .map(|s| {
                let y = s.local_pos.y();
                if y > 1e-4 {
                    1.0
                } else if y < -1e-4 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();

        Self {
            tendon_groups,
            joint_indices,
            imu_body_idx,
            total_free_mass_kg,
            site_side_signs,
        }
    }
}

/// Compute `(pos_or_angle, vel_or_rate, torque_or_force)` for one actuated joint.
fn measure_joint_state(w: &FlexGraphWorld, j_idx: usize) -> [f64; 3] {
    let j = &w.graph.joints[j_idx];
    let pa = w.state.poses[j.body_a];
    let pb = w.state.poses[j.body_b];
    let va = w.state.vels[j.body_a];
    let vb = w.state.vels[j.body_b];
    let torque = w
        .state
        .actuation
        .last_joint_torque
        .get(j_idx)
        .copied()
        .unwrap_or(0.0);

    match j.joint_type {
        JointType::Revolute { axis_local_a, .. } => {
            let axis_l = axis_local_a.normalized_or(Vec3::EZ);
            let axis_w = pa.rot.rotate(axis_l);
            let q_rel = pa.rot.conj().mul(pb.rot).canonical();
            let phi_rel = q_rel.log();
            let angle = phi_rel.dot(axis_l);
            let rate = (vb.ang() - va.ang()).dot(axis_w);
            [angle, rate, torque]
        }
        JointType::Prismatic {
            axis_local_a,
            anchor_a,
            anchor_b,
            ..
        } => {
            let axis_l = axis_local_a.normalized_or(Vec3::EX);
            let axis_w = pa.rot.rotate(axis_l);
            let ra = pa.rot.rotate(anchor_a);
            let rb = pb.rot.rotate(anchor_b);
            let disp = ((pb.pos + rb) - (pa.pos + ra)).dot(axis_w);
            let v_rel = (vb.lin() + vb.ang().cross(rb)) - (va.lin() + va.ang().cross(ra));
            let rate = v_rel.dot(axis_w);
            [disp, rate, torque]
        }
        _ => [0.0, 0.0, torque],
    }
}

/// Extract 7D EstNet contact/proprioceptive summary:
/// `[body_vx, body_vy, body_wz, wall_conf, wall_side, drag, slip]`.
fn compute_contact_summary(w: &FlexGraphWorld, map: &BatchActuatorMap) -> [f64; 7] {
    let sanitize = |x: f64, lo: f64, hi: f64| {
        if x.is_finite() {
            x.clamp(lo, hi)
        } else {
            0.0
        }
    };

    let (vx, vy, wz) = if map.imu_body_idx < w.state.poses.len() {
        let q = w.state.poses[map.imu_body_idx].rot;
        let v = w.state.vels[map.imu_body_idx];
        let vb = q.inv_rotate(v.lin());
        let wb = q.inv_rotate(v.ang());
        (
            sanitize(vb.x(), -3.0, 3.0),
            sanitize(vb.y(), -3.0, 3.0),
            sanitize(wb.z(), -6.0, 6.0),
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    let weight_n = map.total_free_mass_kg * 9.81;
    let n_sites = w.state.contact_normal_forces.len();
    let (wall_conf, wall_side, slip) = if n_sites > 0 {
        let mut n_sum = 0.0_f64;
        let mut side_net = 0.0_f64;
        let mut slip_sum = 0.0_f64;
        let mut active_cnt = 0usize;
        for i in 0..n_sites {
            let fn_i = w.state.contact_normal_forces[i];
            let fn_pos = if fn_i.is_finite() { fn_i.max(0.0) } else { 0.0 };
            n_sum += fn_pos;
            side_net += fn_pos * map.site_side_signs[i];
            if fn_pos > 1e-9 {
                let s = w.state.contact_slips[i];
                let mag = s[0].hypot(s[1]);
                if mag.is_finite() {
                    slip_sum += mag;
                }
                active_cnt += 1;
            }
        }
        let wc = (n_sum / weight_n.max(1e-3)).tanh();
        let ws = (side_net / n_sum.max(1e-3)).clamp(-1.0, 1.0);
        let sl = if active_cnt > 0 {
            sanitize(slip_sum / (active_cnt as f64), 0.0, 10.0)
        } else {
            0.0
        };
        (sanitize(wc, 0.0, 1.0), sanitize(ws, -1.0, 1.0), sl)
    } else {
        (0.0, 0.0, 0.0)
    };

    let n_ten = w.state.actuation.tendons.len();
    let ten_load = if n_ten > 0 {
        w.state
            .actuation
            .tendons
            .iter()
            .map(|t| t.tension.abs())
            .sum::<f64>()
            / ((n_ten as f64) * 40.0)
    } else {
        0.0
    };
    let n_jnt = w.state.actuation.last_joint_torque.len();
    let jnt_load = if n_jnt > 0 {
        w.state
            .actuation
            .last_joint_torque
            .iter()
            .map(|t| t.abs())
            .sum::<f64>()
            / ((n_jnt as f64) * 2.5)
    } else {
        0.0
    };
    let drag = sanitize((ten_load + jnt_load).tanh(), 0.0, 1.0);

    [vx, vy, wz, wall_conf, wall_side, drag, slip]
}

/// Vectorized Rayon-parallel multi-environment simulation handle for RL training & benchmarking.
pub struct FgBatchHandle {
    worlds: Vec<FlexGraphWorld>,
    dt: f64,
    n_bodies: usize,
    n_act: usize,
    n_beams: usize,
    total_nodes: usize,
    act_map: BatchActuatorMap,
    pool: Option<rayon::ThreadPool>,
}

impl FgBatchHandle {
    fn execute_in_pool<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(p) => p.install(f),
            None => f(),
        }
    }
}

fn create_batch_handle(
    graph: RigidFlexGraph,
    config_json: Option<&str>,
    archetype_hint: Option<&str>,
    batch_size: usize,
    dt: f64,
    num_threads: usize,
) -> Result<*mut FgBatchHandle, String> {
    if batch_size == 0 {
        return Err("batch_size must be >= 1".into());
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(format!("invalid dt={dt} (must be finite and > 0)"));
    }
    let pool = if num_threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|e| format!("failed to build rayon ThreadPool({num_threads}): {e}"))?,
        )
    } else {
        None
    };

    let mut worlds = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let (solver_cfg, sdf) = build_solver_and_sdf(config_json, archetype_hint)?;
        let w = FlexGraphWorld::new(graph.clone(), sdf, solver_cfg).map_err(|e| e.to_string())?;
        worlds.push(w);
    }

    let first = &worlds[0];
    let n_bodies = first.analysis.num_bodies;
    let n_act = first.num_total_actuators();
    let n_beams = first.graph.beams.len();
    let total_nodes = first.total_nodes();
    let act_map = BatchActuatorMap::from_world(first);

    Ok(Box::into_raw(Box::new(FgBatchHandle {
        worlds,
        dt,
        n_bodies,
        n_act,
        n_beams,
        total_nodes,
        act_map,
        pool,
    })))
}

/// Create a Rayon-parallel batch of `batch_size` worlds from a JSON string (`RigidFlexGraph` or
/// archetype envelope). `num_threads == 0` uses the global Rayon thread pool.
#[no_mangle]
pub unsafe extern "C" fn fg_batch_create_from_json(
    json_ptr: *const c_char,
    batch_size: usize,
    dt: f64,
    num_threads: usize,
) -> *mut FgBatchHandle {
    fg_batch_create_from_json_with_config(json_ptr, ptr::null(), batch_size, dt, num_threads)
}

/// Create a Rayon-parallel batch of `batch_size` worlds from a JSON string + optional `config_json`.
#[no_mangle]
pub unsafe extern "C" fn fg_batch_create_from_json_with_config(
    json_ptr: *const c_char,
    config_json_ptr: *const c_char,
    batch_size: usize,
    dt: f64,
    num_threads: usize,
) -> *mut FgBatchHandle {
    clear_last_error();
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<*mut FgBatchHandle, String> {
        let g_str = read_c_str(json_ptr, "json_ptr")?;
        let c_str = read_opt_c_str(config_json_ptr)?;
        let graph = load_graph_from_json_str(g_str).map_err(|e| e.to_string())?;
        create_batch_handle(graph, c_str, None, batch_size, dt, num_threads)
    }));
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic inside fg_batch_create_from_json_with_config");
            ptr::null_mut()
        }
    }
}

/// Create a Rayon-parallel batch of `batch_size` worlds for one of the 5 built-in archetypes.
#[no_mangle]
pub unsafe extern "C" fn fg_batch_create_archetype(
    archetype_name: *const c_char,
    config_json_ptr: *const c_char,
    batch_size: usize,
    dt: f64,
    num_threads: usize,
) -> *mut FgBatchHandle {
    clear_last_error();
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<*mut FgBatchHandle, String> {
        let name = read_c_str(archetype_name, "archetype_name")?;
        let c_str = read_opt_c_str(config_json_ptr)?;
        let graph = build_archetype_by_name(name).map_err(|e| e.to_string())?;
        create_batch_handle(graph, c_str, Some(name), batch_size, dt, num_threads)
    }));
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic inside fg_batch_create_archetype");
            ptr::null_mut()
        }
    }
}

/// Query `(n_bodies, n_act, n_beams)` of a batch handle. Any output pointer may be NULL.
/// Returns `0` on success.
#[no_mangle]
pub unsafe extern "C" fn fg_batch_dims(
    handle: *const FgBatchHandle,
    out_n_bodies: *mut usize,
    out_n_act: *mut usize,
    out_n_beams: *mut usize,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_ref() else {
        set_last_error("null handle in fg_batch_dims");
        return -1;
    };
    if let Some(p) = out_n_bodies.as_mut() {
        *p = h.n_bodies;
    }
    if let Some(p) = out_n_act.as_mut() {
        *p = h.n_act;
    }
    if let Some(p) = out_n_beams.as_mut() {
        *p = h.n_beams;
    }
    0
}

/// Return batch size `B` (0 if handle is NULL).
#[no_mangle]
pub unsafe extern "C" fn fg_batch_size(handle: *const FgBatchHandle) -> usize {
    handle.as_ref().map_or(0, |h| h.worlds.len())
}

/// Return total node count (`n_bodies + n_beam_nodes`) per env (0 if handle is NULL).
#[no_mangle]
pub unsafe extern "C" fn fg_batch_total_nodes(handle: *const FgBatchHandle) -> usize {
    handle.as_ref().map_or(0, |h| h.total_nodes)
}

/// Selectively reset environments where `mask_u8_ptr[i] != 0` in parallel via Rayon.
/// If `mask_u8_ptr` is NULL, resets all `B` environments. Returns `0` on success.
#[no_mangle]
pub unsafe extern "C" fn fg_batch_reset_mask(
    handle: *mut FgBatchHandle,
    mask_u8_ptr: *const u8,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_mut() else {
        set_last_error("null handle in fg_batch_reset_mask");
        return -1;
    };
    let b = h.worlds.len();
    let mask_opt: Option<&[u8]> = if mask_u8_ptr.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(mask_u8_ptr, b))
    };
    let worlds = &mut h.worlds;
    match &h.pool {
        Some(pool) => pool.install(|| {
            worlds.par_iter_mut().enumerate().for_each(|(i, w)| {
                if mask_opt.is_none_or(|m| m[i] != 0) {
                    w.reset();
                }
            });
        }),
        None => {
            worlds.par_iter_mut().enumerate().for_each(|(i, w)| {
                if mask_opt.is_none_or(|m| m[i] != 0) {
                    w.reset();
                }
            });
        }
    }
    0
}

/// Step all `B` environments for `substeps` in parallel via Rayon and write outputs directly into
/// caller-owned contiguous row-major FP64/u8 buffers (any output pointer may be NULL to skip):
///
/// - `actions_ptr`: `[B, N_act]` f64 (NULL keeps current targets)
/// - `out_poses`: `[B, N_b, 7]` f64 (`[px, py, pz, qx, qy, qz, qw]` of rigid bodies `0..N_b`)
/// - `out_vels`: `[B, N_b, 6]` f64 (`[vx, vy, vz, wx, wy, wz]` of rigid bodies `0..N_b`)
/// - `out_act_state`: `[B, N_act, 3]` f64 (`[pos_or_len, vel, tension_or_torque]`)
/// - `out_contact_summary`: `[B, 7]` f64 (`[body_vx, body_vy, body_wz, wall_conf, wall_side, drag, slip]`)
/// - `out_failed_u8`: `[B]` u8 (`1` if failed, `0` otherwise)
///
/// Returns `0` on success (even if individual envs set `out_failed_u8[i] = 1`), negative on API error.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn fg_batch_step_into(
    handle: *mut FgBatchHandle,
    substeps: usize,
    actions_ptr: *const f64,
    out_poses: *mut f64,
    out_vels: *mut f64,
    out_act_state: *mut f64,
    out_contact_summary: *mut f64,
    out_failed_u8: *mut u8,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_mut() else {
        set_last_error("null handle in fg_batch_step_into");
        return -1;
    };

    let b = h.worlds.len();
    let n_b = h.n_bodies;
    let n_act = h.n_act;
    let dt = h.dt;

    // Validate & wrap raw pointer views as Send+Sync usize addresses so Rayon worker threads
    // write strictly disjoint per-env slices `[i*stride .. (i+1)*stride]`.
    let actions_addr = actions_ptr as usize;
    let poses_addr = out_poses as usize;
    let vels_addr = out_vels as usize;
    let act_addr = out_act_state as usize;
    let summary_addr = out_contact_summary as usize;
    let failed_addr = out_failed_u8 as usize;

    let res = catch_unwind(AssertUnwindSafe(|| {
        let act_map = &h.act_map;
        let worlds = &mut h.worlds;
        let mut run = || {
            worlds.par_iter_mut().enumerate().for_each(|(i, w)| {
                if actions_addr != 0 && n_act > 0 {
                    let act_slice = std::slice::from_raw_parts(
                        (actions_addr as *const f64).add(i * n_act),
                        n_act,
                    );
                    w.set_all_targets(act_slice);
                }

                let step_failed = if substeps > 0 {
                    match w.step_n(substeps, dt) {
                        Ok(d) => d.failed || w.state.failed,
                        Err(_) => {
                            w.state.failed = true;
                            true
                        }
                    }
                } else {
                    w.state.failed
                };

                if poses_addr != 0 && n_b > 0 {
                    let dst = (poses_addr as *mut f64).add(i * n_b * 7);
                    let src = w.state.poses.as_ptr() as *const f64;
                    ptr::copy_nonoverlapping(src, dst, n_b * 7);
                }

                if vels_addr != 0 && n_b > 0 {
                    let dst = (vels_addr as *mut f64).add(i * n_b * 6);
                    let src = w.state.vels.as_ptr() as *const f64;
                    ptr::copy_nonoverlapping(src, dst, n_b * 6);
                }

                if act_addr != 0 && n_act > 0 {
                    let dst = std::slice::from_raw_parts_mut(
                        (act_addr as *mut f64).add(i * n_act * 3),
                        n_act * 3,
                    );
                    let mut slot = 0usize;
                    for (aid, members) in &act_map.tendon_groups {
                        let pos = w
                            .state
                            .actuation
                            .tendon_servo_pos
                            .get(*aid)
                            .copied()
                            .unwrap_or(0.0);
                        let vel = w
                            .state
                            .actuation
                            .tendon_servo_vel
                            .get(*aid)
                            .copied()
                            .unwrap_or(0.0);
                        let ten: f64 = members
                            .iter()
                            .map(|&tid| w.state.actuation.tendons[tid].tension)
                            .sum();
                        dst[slot * 3] = pos;
                        dst[slot * 3 + 1] = vel;
                        dst[slot * 3 + 2] = ten;
                        slot += 1;
                    }
                    for &j_idx in &act_map.joint_indices {
                        let js = measure_joint_state(w, j_idx);
                        dst[slot * 3] = js[0];
                        dst[slot * 3 + 1] = js[1];
                        dst[slot * 3 + 2] = js[2];
                        slot += 1;
                    }
                }

                if summary_addr != 0 {
                    let dst = std::slice::from_raw_parts_mut(
                        (summary_addr as *mut f64).add(i * 7),
                        7,
                    );
                    let summary = compute_contact_summary(w, act_map);
                    dst.copy_from_slice(&summary);
                }

                if failed_addr != 0 {
                    *(failed_addr as *mut u8).add(i) = u8::from(step_failed);
                }
            });
        };

        match &h.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }
    }));

    let _ = b;
    match res {
        Ok(()) => 0,
        Err(_) => {
            set_last_error("panic inside fg_batch_step_into");
            -4
        }
    }
}

/// Copy ALL nodes (rigid bodies + beam interior nodes) for all `B` environments into
/// `out_poses [B, total_nodes, 7]` and/or `out_vels [B, total_nodes, 6]`. Either may be NULL.
#[no_mangle]
pub unsafe extern "C" fn fg_batch_get_all_nodes_into(
    handle: *const FgBatchHandle,
    out_poses: *mut f64,
    out_vels: *mut f64,
) -> c_int {
    clear_last_error();
    let Some(h) = handle.as_ref() else {
        set_last_error("null handle in fg_batch_get_all_nodes_into");
        return -1;
    };
    let n = h.total_nodes;
    let poses_addr = out_poses as usize;
    let vels_addr = out_vels as usize;

    h.execute_in_pool(|| {
        h.worlds.par_iter().enumerate().for_each(|(i, w)| {
            if poses_addr != 0 && n > 0 {
                let dst = (poses_addr as *mut f64).add(i * n * 7);
                let src = w.state.poses.as_ptr() as *const f64;
                ptr::copy_nonoverlapping(src, dst, n * 7);
            }
            if vels_addr != 0 && n > 0 {
                let dst = (vels_addr as *mut f64).add(i * n * 6);
                let src = w.state.vels.as_ptr() as *const f64;
                ptr::copy_nonoverlapping(src, dst, n * 6);
            }
        });
    });
    0
}

/// Destroy a batch handle previously returned by `fg_batch_create_*`.
#[no_mangle]
pub unsafe extern "C" fn fg_batch_destroy(handle: *mut FgBatchHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_single_world_lifecycle_zero_copy_and_all_5_archetypes() {
        let archetypes = [
            ("serpworm_1", 0, 0.0005),
            ("bristle_worm", 0, 0.001),
            ("tendon_endoscope_3d", 1, 0.001),
            ("flexspine_centipede", 1, 0.001),
            ("tensegrity_6bar", 2, 0.001),
        ];

        for (name, expected_route, dt) in archetypes {
            let c_name = CString::new(name).unwrap();
            let h = unsafe { fg_world_create_archetype(c_name.as_ptr(), ptr::null()) };
            assert!(
                !h.is_null(),
                "failed to create {name}: {:?}",
                unsafe { CStr::from_ptr(fg_last_error()) }
            );

            let mut info = FgWorldInfo::default();
            assert_eq!(unsafe { fg_world_get_info(h, &mut info) }, 0);
            assert_eq!(info.schur_route, expected_route, "route mismatch on {name}");
            assert!(info.total_nodes >= info.num_bodies);

            let targets = vec![0.05; info.num_total_actuators];
            assert_eq!(
                unsafe { fg_world_set_targets(h, targets.as_ptr(), targets.len()) },
                0
            );

            let mut diag = FgStepDiag::default();
            assert_eq!(unsafe { fg_world_step(h, 5, dt, &mut diag) }, 0);
            assert_eq!(diag.failed, 0, "step failed on {name}");
            assert_eq!(diag.step_index, 5);
            assert!(diag.max_quat_norm_err < 1e-13);

            // Verify zero-copy pointer matches copy-out fg_world_get_state bit-for-bit
            let mut poses_copy = vec![0.0; info.total_nodes * 7];
            let mut vels_copy = vec![0.0; info.total_nodes * 6];
            assert_eq!(
                unsafe {
                    fg_world_get_state(
                        h,
                        poses_copy.as_mut_ptr(),
                        poses_copy.len(),
                        vels_copy.as_mut_ptr(),
                        vels_copy.len(),
                    )
                },
                0
            );
            let poses_view =
                unsafe { std::slice::from_raw_parts(fg_world_poses_ptr(h), info.total_nodes * 7) };
            let vels_view =
                unsafe { std::slice::from_raw_parts(fg_world_vels_ptr(h), info.total_nodes * 6) };
            assert_eq!(poses_view, poses_copy.as_slice());
            assert_eq!(vels_view, vels_copy.as_slice());
            assert!(poses_view.iter().all(|v| v.is_finite()));
            assert!(vels_view.iter().all(|v| v.is_finite()));

            // Verify energy/momentum and JSON round-trip export -> recreate
            let mut em = FgEnergyMomentum::default();
            assert_eq!(unsafe { fg_world_get_energy_momentum(h, &mut em) }, 0);
            assert!(em.total_energy.is_finite());

            let exported = unsafe { fg_world_export_graph_json(h) };
            assert!(!exported.is_null());
            let h2 = unsafe { fg_world_create_from_json(exported, ptr::null()) };
            assert!(!h2.is_null());
            unsafe {
                fg_string_free(exported);
                fg_world_free(h2);
                fg_world_free(h);
            }
        }
    }

    #[test]
    fn test_ffi_batch_vectorized_rl_api_and_selective_reset() {
        let env_json = CString::new(r#"{"archetype": "tensegrity6bar"}"#).unwrap();
        let batch_size = 4usize;
        let bh = unsafe { fg_batch_create_from_json(env_json.as_ptr(), batch_size, 0.001, 2) };
        assert!(!bh.is_null());

        let (mut nb, mut na, mut nbeams) = (0usize, 0usize, 0usize);
        assert_eq!(
            unsafe { fg_batch_dims(bh, &mut nb, &mut na, &mut nbeams) },
            0
        );
        assert_eq!(nb, 6);
        assert_eq!(na, 24);
        assert_eq!(nbeams, 24);
        assert_eq!(unsafe { fg_batch_size(bh) }, batch_size);

        let mut actions = vec![0.0_f64; batch_size * na];
        for b in 0..batch_size {
            actions[b * na] = 0.005 * ((b + 1) as f64);
        }
        let mut poses = vec![0.0_f64; batch_size * nb * 7];
        let mut vels = vec![0.0_f64; batch_size * nb * 6];
        let mut act_state = vec![0.0_f64; batch_size * na * 3];
        let mut summary = vec![0.0_f64; batch_size * 7];
        let mut failed = vec![1_u8; batch_size];

        assert_eq!(
            unsafe {
                fg_batch_step_into(
                    bh,
                    5,
                    actions.as_ptr(),
                    poses.as_mut_ptr(),
                    vels.as_mut_ptr(),
                    act_state.as_mut_ptr(),
                    summary.as_mut_ptr(),
                    failed.as_mut_ptr(),
                )
            },
            0
        );
        assert!(failed.iter().all(|&f| f == 0));
        assert!(poses.iter().all(|v| v.is_finite()));
        assert!(vels.iter().all(|v| v.is_finite()));
        assert!(act_state.iter().all(|v| v.is_finite()));
        assert!(summary.iter().all(|v| v.is_finite()));

        // Selectively reset env 1 and env 3 only; env 0 and env 2 keep non-zero velocities
        let mask = [0_u8, 1_u8, 0_u8, 1_u8];
        assert_eq!(unsafe { fg_batch_reset_mask(bh, mask.as_ptr()) }, 0);
        // Query zero-step snapshot
        assert_eq!(
            unsafe {
                fg_batch_step_into(
                    bh,
                    0,
                    ptr::null(),
                    poses.as_mut_ptr(),
                    vels.as_mut_ptr(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    failed.as_mut_ptr(),
                )
            },
            0
        );
        let env0_vel_norm: f64 = vels[0..nb * 6].iter().map(|v| v * v).sum();
        let env1_vel_norm: f64 = vels[nb * 6..2 * nb * 6].iter().map(|v| v * v).sum();
        let env2_vel_norm: f64 = vels[2 * nb * 6..3 * nb * 6].iter().map(|v| v * v).sum();
        let env3_vel_norm: f64 = vels[3 * nb * 6..4 * nb * 6].iter().map(|v| v * v).sum();
        assert!(env0_vel_norm > 1e-8);
        assert_eq!(env1_vel_norm, 0.0);
        assert!(env2_vel_norm > 1e-8);
        assert_eq!(env3_vel_norm, 0.0);

        unsafe { fg_batch_destroy(bh) };
    }

    #[test]
    fn test_ffi_null_and_error_handling() {
        let null_h = unsafe { fg_world_create_from_json(ptr::null(), ptr::null()) };
        assert!(null_h.is_null());
        let err = unsafe { CStr::from_ptr(fg_last_error()).to_str().unwrap() };
        assert!(!err.is_empty());
        assert_ne!(unsafe { fg_world_step(ptr::null_mut(), 1, 0.001, ptr::null_mut()) }, 0);
    }

    #[test]
    fn test_ffi_v2_ros_urdf_mjcf_corrugated_pipe_and_adjoint() {
        // 1. Standard ROS URDF + Clarabel SOCP config
        let ros_urdf = CString::new(
            r#"<?xml version="1.0"?>
            <robot name="ffi_ros_arm">
              <link name="world"/>
              <link name="link1">
                <inertial><mass value="0.4"/><inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/></inertial>
                <collision><geometry><sphere radius="0.03"/></geometry></collision>
              </link>
              <joint name="j1" type="revolute">
                <parent link="world"/><child link="link1"/>
                <origin xyz="0 0 0.25" rpy="0 0 0"/><axis xyz="0 1 0"/>
                <limit lower="-1.5" upper="1.5" effort="5.0" velocity="4.0"/>
              </joint>
            </robot>"#,
        )
        .unwrap();
        let socp_cfg = CString::new(r#"{"lcp_solver":"clarabel_socp"}"#).unwrap();
        let h_urdf = unsafe { fg_world_create_from_ros_urdf(ros_urdf.as_ptr(), socp_cfg.as_ptr()) };
        assert!(!h_urdf.is_null());
        let mut diag = FgStepDiag::default();
        assert_eq!(unsafe { fg_world_step(h_urdf, 4, 0.001, &mut diag) }, 0);
        assert_eq!(diag.failed, 0);
        unsafe { fg_world_free(h_urdf) };

        // 2. MuJoCo MJCF
        let mjcf = CString::new(
            r#"<mujoco model="ffi_mjcf_ball">
              <option timestep="0.001" gravity="0 0 -9.81"/>
              <worldbody>
                <body name="ball" pos="0 0 0.15">
                  <freejoint name="root"/>
                  <geom name="g0" type="sphere" size="0.04" solref="0.02 1.0"/>
                </body>
              </worldbody>
            </mujoco>"#,
        )
        .unwrap();
        let h_mjcf = unsafe { fg_world_create_from_mjcf(mjcf.as_ptr(), ptr::null()) };
        assert!(!h_mjcf.is_null());
        assert_eq!(unsafe { fg_world_step(h_mjcf, 5, 0.001, &mut diag) }, 0);
        assert_eq!(diag.failed, 0);
        unsafe { fg_world_free(h_mjcf) };

        // 3. Corrugated triangle-mesh pipe cavity + Tight Inclusion CCD
        let g_pipe = build_archetype_by_name("bristle_worm").unwrap();
        let g_pipe_json = CString::new(g_pipe.to_json_string().unwrap()).unwrap();
        let ti_cfg = CString::new(r#"{"use_tight_inclusion_ccd":true}"#).unwrap();
        let h_pipe = unsafe {
            fg_world_create_with_corrugated_pipe_mesh(
                g_pipe_json.as_ptr(),
                ti_cfg.as_ptr(),
                1.2,
                0.15,
                0.006,
                6.0,
                0.2,
                0.0,
                0.08,
            )
        };
        assert!(!h_pipe.is_null());
        assert_eq!(unsafe { fg_world_step(h_pipe, 3, 0.001, &mut diag) }, 0);
        assert_eq!(diag.failed, 0);
        unsafe { fg_world_free(h_pipe) };

        // 4. RigidFlexGrid-4x4 with faer sparse backend
        let grid_name = CString::new("rigid_flex_grid_4x4").unwrap();
        let faer_cfg = CString::new(r#"{"sparse_backend":"faer"}"#).unwrap();
        let h_grid = unsafe { fg_world_create_archetype(grid_name.as_ptr(), faer_cfg.as_ptr()) };
        assert!(!h_grid.is_null());
        let mut info = FgWorldInfo::default();
        assert_eq!(unsafe { fg_world_get_info(h_grid, &mut info) }, 0);
        assert_eq!(info.schur_route, 2);
        assert_eq!(info.cyclomatic_number, 9);
        assert_eq!(unsafe { fg_world_step(h_grid, 3, 0.001, &mut diag) }, 0);
        assert_eq!(diag.failed, 0);
        unsafe { fg_world_free(h_grid) };

        // 5. IFT Reverse-Mode Adjoint gradient via C-ABI vs finite difference
        let sw_name = CString::new("serpworm_1").unwrap();
        let free_cfg = CString::new(
            r#"{"solver_preset":"serpworm_kelvin_parity","contact":{"backend":"none"},"world_sdf":"empty"}"#,
        )
        .unwrap();
        let h_adj = unsafe { fg_world_create_archetype(sw_name.as_ptr(), free_cfg.as_ptr()) };
        assert!(!h_adj.is_null());
        assert_eq!(unsafe { fg_world_get_info(h_adj, &mut info) }, 0);
        let n_steps = 4usize;
        let n_act = info.num_total_actuators;
        let nb = info.num_bodies;
        let controls = vec![0.25_f64; n_steps * n_act];
        let mut targets = vec![0.0_f64; (n_steps + 1) * nb * 7];
        for k in 0..=n_steps {
            for b in 0..nb {
                let off = (k * nb + b) * 7;
                targets[off] = 0.05 * (b as f64) + 0.02;
                targets[off + 2] = 0.05;
                targets[off + 6] = 1.0;
            }
        }
        let dp_json = CString::new(
            r#"[{"name":"E","kind":"beam_youngs_modulus","target_ids":[],"scale":1e11}]"#,
        )
        .unwrap();
        let mut loss = 0.0_f64;
        let mut grad_u = vec![0.0_f64; n_steps * n_act];
        let mut grad_p = vec![0.0_f64; 1];
        assert_eq!(
            unsafe {
                fg_world_adjoint_grad(
                    h_adj,
                    controls.as_ptr(),
                    n_steps,
                    n_act,
                    0.001,
                    targets.as_ptr(),
                    nb,
                    50.0,
                    1.0,
                    0.01,
                    dp_json.as_ptr(),
                    &mut loss,
                    grad_u.as_mut_ptr(),
                    grad_p.as_mut_ptr(),
                    grad_p.len(),
                )
            },
            0
        );
        assert!(loss > 0.0 && loss.is_finite());
        assert!(grad_u.iter().all(|v| v.is_finite()));
        assert!(grad_p[0].is_finite());

        // FD check on control[0]
        let eps = 1e-6;
        let mut ctrl_p = controls.clone();
        let mut ctrl_m = controls;
        ctrl_p[0] += eps;
        ctrl_m[0] -= eps;
        let (mut lp, mut lm) = (0.0_f64, 0.0_f64);
        let empty_dp = CString::new("[]").unwrap();
        assert_eq!(
            unsafe {
                fg_world_adjoint_grad(
                    h_adj,
                    ctrl_p.as_ptr(),
                    n_steps,
                    n_act,
                    0.001,
                    targets.as_ptr(),
                    nb,
                    50.0,
                    1.0,
                    0.01,
                    empty_dp.as_ptr(),
                    &mut lp,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                )
            },
            0
        );
        assert_eq!(
            unsafe {
                fg_world_adjoint_grad(
                    h_adj,
                    ctrl_m.as_ptr(),
                    n_steps,
                    n_act,
                    0.001,
                    targets.as_ptr(),
                    nb,
                    50.0,
                    1.0,
                    0.01,
                    empty_dp.as_ptr(),
                    &mut lm,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                )
            },
            0
        );
        let fd = (lp - lm) / (2.0 * eps);
        let rel = (grad_u[0] - fd).abs() / fd.abs().max(1e-8);
        assert!(rel < 1e-4, "FFI adjoint dL/du[0] mismatch: adj={}, fd={}, rel={rel}", grad_u[0], fd);
        unsafe { fg_world_free(h_adj) };
    }
}

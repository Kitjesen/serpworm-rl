//! Implicit Lie-Group Time Integration configuration, state, diagnostics, and
//! factored bipartite multi-RHS system helpers on $SE(3)$.

use crate::actuation::ActuationState;
use crate::continuum::{DerBeamFactor, DerBeamState};
use crate::schur::{CorotationalBeamFactor, LinearSolveError, RigidSchurFactor, SparseBackend};
use fg_contact::{ArpcSiteState, LcpSolverConfig, SelfContactEngine};
use fg_ir::GraphTopologyAnalysis;
use fg_math::{Se3, Vec3, Vec6};
use serde::{Deserialize, Serialize};

/// Implicit Lie-group time integration scheme selector.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IntegratorKind {
    /// 1-step linearly implicit Euler with exact $SE(3)$ exponential pose update
    /// (`< 1e-10` FP64 parity with `serpworm.physics.gpu_core.GPUCore`).
    #[default]
    LinearlyImplicitEuler,
    /// Full nonlinear implicit Backward Euler solved via Damped Newton-Raphson + Armijo merit
    /// line search (+ conservative ACCD clamping when IPC/SDF barriers are enabled).
    DampedNewtonEuler,
    /// Second-order implicit Lie-group Newmark-$\beta$ solved via Damped Newton-Raphson on $SE(3)$
    /// (default $\beta = 0.25, \gamma = 0.50$: unconditional stability & zero artificial drift).
    NewmarkBeta { beta: f64, gamma: f64 },
    /// Second-order L-stable implicit Lie-group BDF2 solved via Damped Newton-Raphson on $SE(3)$.
    Bdf2,
}

/// Contact backend selector wiring `fg-contact` Engines A, B, and C into `fg-solver`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum ContactBackend {
    /// Disable all environment/ground/SDF contacts (useful for free-flight conservation tests).
    None,
    /// Unified Implicit Anisotropic Patch Contact (ARPC: `RatchetWheel` / `AnisotropicScale` /
    /// `IsotropicPad` + Hunt-Crossley / Bekker soil rutting + SPD `Contact-Schur`).
    Arpc { implicit: bool },
    /// Exact bit-level parity mode matching `serpworm.physics.gpu_core.GPUCore` default
    /// `contact_model='kelvin', implicit_contact=False` (`wheel_stiction.contact` + `kn=5000, cn=8`).
    GpuCoreKelvinParity { kn: f64, cn: f64 },
    /// Compliance-Based Frictional Contact LCP / Delassus-Schur solver (Shen et al. IJRR 2026 &
    /// Moreau Contact-Implicit) using exact $C_c = J_c H^{-1} J_c^\top$ from the factored bipartite solver!
    DelassusLcp {
        lcp: LcpSolverConfig,
        /// Contact clustering deduplication radius (m).
        dedup_radius: f64,
        /// Baumgarte penetration stabilization factor in `[0, 1]` (`0.2` default).
        baumgarte_beta: f64,
    },
    /// C-IPC $C^2$ clamped log-barrier / Py-DiSMech $C^2$ soft-barrier against `WorldSdf`
    /// with conservative Lipschitz ACCD non-tunneling clamping.
    SdfBarrier {
        barrier: fg_contact::IpcBarrierConfig,
        mu: f64,
        friction_eps_v: f64,
        use_ccd: bool,
    },
}

impl Default for ContactBackend {
    fn default() -> Self {
        Self::Arpc { implicit: true }
    }
}

/// Complete solver configuration for [`crate::FlexGraphWorld`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolverConfig {
    pub integrator: IntegratorKind,
    pub contact: ContactBackend,
    /// When `true`, uses exact analytic `beam_strain_jacobian` (`J^T K J`) and `Quat::log`
    /// for corotational beams; when `false`, uses frozen-frame `T^T K T` + `log_legacy` for
    /// exact `< 1e-10` FP64 parity with `serpworm.physics.gpu_core`.
    pub use_strain_jacobian: bool,
    /// When `true`, keeps rigid-body inertia tensor fixed in world frame without gyroscopic
    /// $\omega \times I \omega$ (exact match to `serpworm.GPUCore.plate_mass`); when `false`,
    /// uses exact rotating world inertia $R I_b R^\top$ and angular-momentum-conserving Euler dynamics.
    pub world_fixed_inertia: bool,
    /// Enable Stribeck velocity-dependent static friction boost on revolute/prismatic joints
    /// (`false` matches `gpu_core.py:553` pure Coulomb deadband).
    pub joint_stribeck_friction: bool,
    /// Enable beam-beam self-contact (`fg_contact::SelfContactEngine`) for beams with
    /// `enable_self_contact = true`.
    pub enable_self_contact: bool,
    pub self_contact_barrier: fg_contact::IpcBarrierConfig,
    pub self_contact_mu: f64,
    pub self_contact_eps_v: f64,
    /// Enable conservative ACCD step clamping during damped Newton line search.
    pub enable_ccd_clamping: bool,
    /// Maximum Newton-Raphson iterations per time step (for `DampedNewtonEuler`, `NewmarkBeta`, `Bdf2`).
    pub newton_max_iters: usize,
    /// Convergence tolerance on `||r||_inf` for Newton-Raphson.
    pub newton_tol: f64,
    /// Maximum Armijo backtracking halvings per Newton iteration.
    pub line_search_max_iters: usize,
    /// Initial Levenberg-Marquardt regularization shift (0.0 disables unless needed).
    pub lm_initial: f64,
    /// SPD-project 11x11 DER bending/twisting Hessians for guaranteed descent in Newton/implicit steps.
    pub project_der_hessian_spd: bool,
    /// Sparse direct solver backend for cyclic `SchurRoute::SparseGeneral` graphs
    /// (`Auto`, `CustomBlockCsc`, or `FaerSupernodal`).
    #[serde(default)]
    pub sparse_backend: SparseBackend,
    /// Use Tight Inclusion conservative CCD (Wang et al. TOG 2021) for mesh/segment impacts
    /// when `enable_ccd_clamping == true`.
    #[serde(default)]
    pub use_tight_inclusion_ccd: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::Arpc { implicit: true },
            use_strain_jacobian: false,
            world_fixed_inertia: false,
            joint_stribeck_friction: false,
            enable_self_contact: false,
            self_contact_barrier: fg_contact::IpcBarrierConfig::default(),
            self_contact_mu: 0.25,
            self_contact_eps_v: 1.0e-2,
            enable_ccd_clamping: true,
            newton_max_iters: 25,
            newton_tol: 1.0e-10,
            line_search_max_iters: 12,
            lm_initial: 0.0,
            project_der_hessian_spd: true,
            sparse_backend: SparseBackend::Auto,
            use_tight_inclusion_ccd: false,
        }
    }
}

impl SolverConfig {
    /// Preset for exact `< 1e-10` FP64 parity with `serpworm.physics.gpu_core.GPUCore`
    /// (`contact_model='kelvin', implicit_contact=False, strain_jacobian=False`).
    pub fn serpworm_kelvin_parity() -> Self {
        Self {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::GpuCoreKelvinParity {
                kn: 5000.0,
                cn: 8.0,
            },
            use_strain_jacobian: false,
            world_fixed_inertia: true,
            joint_stribeck_friction: false,
            enable_self_contact: false,
            ..Self::default()
        }
    }

    /// Preset for exact `< 1e-10` FP64 parity with `serpworm.physics.gpu_core.GPUCore`
    /// using implicit ARPC (`arpc_hard`, `arpc_carpet`, `arpc_soft_soil`, or implicit `kelvin`).
    pub fn serpworm_arpc_parity() -> Self {
        Self {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::Arpc { implicit: true },
            use_strain_jacobian: false,
            world_fixed_inertia: true,
            joint_stribeck_friction: false,
            enable_self_contact: false,
            ..Self::default()
        }
    }

    /// Preset for nonlinear large-step (`dt = 1..5 ms`) Damped Newton-Raphson + Newmark-$\beta$
    /// on $SE(3)$ with exact analytic strain Jacobian and rotating rigid inertia.
    pub fn nonlinear_newmark() -> Self {
        Self {
            integrator: IntegratorKind::NewmarkBeta {
                beta: 0.25,
                gamma: 0.50,
            },
            contact: ContactBackend::Arpc { implicit: true },
            use_strain_jacobian: true,
            world_fixed_inertia: false,
            joint_stribeck_friction: true,
            ..Self::default()
        }
    }

    /// Preset for exact 3D second-order friction-cone SOCP contact (`LcpSolverMethod::ClarabelSocp`)
    /// coupled with the bipartite Delassus operator (`W = J_c H^{-1} J_c^T`).
    pub fn delassus_socp() -> Self {
        Self {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::DelassusLcp {
                lcp: LcpSolverConfig {
                    method: fg_contact::LcpSolverMethod::ClarabelSocp,
                    max_iters: 120,
                    tol: 1.0e-10,
                    compliance_reg: 1.0e-7,
                    ..LcpSolverConfig::default()
                },
                dedup_radius: 0.002,
                baumgarte_beta: 0.2,
            },
            ..Self::default()
        }
    }

    /// Build a [`SelfContactEngine`] from this config.
    pub fn self_contact_engine(&self) -> SelfContactEngine {
        SelfContactEngine::new(
            self.self_contact_barrier,
            self.self_contact_mu,
            self.self_contact_eps_v,
        )
    }
}

/// Complete dynamic state of a simulated `RigidFlexGraph` world.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldState {
    pub time: f64,
    pub step_count: u64,
    /// Global node $SE(3)$ poses (`0..num_bodies` rigid bodies, `num_bodies..total_nodes` beam nodes).
    pub poses: Vec<Se3>,
    /// Global node 6D spatial velocities `[v_lin; omega]` in world orientation.
    pub vels: Vec<Vec6>,
    /// Global node 6D spatial accelerations (used by `NewmarkBeta`).
    pub accels: Vec<Vec6>,
    /// Previous step Lie-algebra increments `Delta xi_{n-1}` (used by `Bdf2`).
    pub prev_dx: Vec<Vec6>,
    /// Previous step velocities `v_{n-1}` (used by `Bdf2`).
    pub prev_vels: Vec<Vec6>,
    /// Actuation state (tendon servos, stick-slip Capstan tensions, joint targets & telemetry).
    pub actuation: ActuationState,
    /// Per-contact-site persistent ARPC tribology/rutting state (len `graph.contact_sites.len()`).
    pub contact_states: Vec<ArpcSiteState>,
    /// Per-contact-site last normal force (N) for telemetry/RL observations.
    pub contact_normal_forces: Vec<f64>,
    /// Per-contact-site last slip velocity `[fwd, lat]` (m/s).
    pub contact_slips: Vec<[f64; 2]>,
    /// Per-beam DER Bishop-frame and edge twist state (`Some` for DER beams, `None` for corotational).
    pub der_states: Vec<Option<DerBeamState>>,
    /// Per-beam DER twist accelerations (used by `NewmarkBeta` / `Bdf2`).
    pub der_theta_accels: Vec<Vec<f64>>,
    /// Per-beam previous-step DER twist increments `dtheta_{n-1}` (used by `Bdf2`).
    pub prev_der_dtheta: Vec<Vec<f64>>,
    /// Per-beam previous-step DER twist velocities `theta_vel_{n-1}` (used by `Bdf2`).
    pub prev_der_theta_vel: Vec<Vec<f64>>,
    /// Previous-step world positions of all nodes (for IPC mollified friction).
    pub prev_positions: Vec<Vec3>,
    /// Warm-start impulses for `DelassusLcp` mode.
    pub lcp_warm_start: Vec<[f64; 3]>,
    /// Cumulative solver failure flag.
    pub failed: bool,
}

/// Per-step numerical diagnostics returned by [`crate::FlexGraphWorld::step`].
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct StepDiagnostics {
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
    pub failed: bool,
}

/// Mechanical energy and momentum budget report for conservation verification.
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct EnergyMomentumReport {
    pub kinetic_energy: f64,
    pub elastic_energy: f64,
    pub actuation_potential_energy: f64,
    pub gravity_potential_energy: f64,
    pub total_energy: f64,
    pub linear_momentum: Vec3,
    pub angular_momentum: Vec3,
}

/// Error returned when creating or stepping a [`crate::FlexGraphWorld`].
#[derive(Clone, Debug)]
pub enum SolverError {
    Ir(String),
    LinearSolve(LinearSolveError),
    InvalidDt(f64),
    NonFiniteState,
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ir(msg) => write!(f, "graph IR error: {msg}"),
            Self::LinearSolve(e) => write!(f, "linear solve error: {e}"),
            Self::InvalidDt(dt) => write!(f, "invalid dt={dt} (must be finite and > 0)"),
            Self::NonFiniteState => write!(f, "non-finite state detected"),
        }
    }
}

impl std::error::Error for SolverError {}

impl From<LinearSolveError> for SolverError {
    fn from(e: LinearSolveError) -> Self {
        Self::LinearSolve(e)
    }
}

/// Per-beam cached factor in a [`FactoredBipartiteSystem`].
#[derive(Clone, Debug)]
pub enum BeamFactorKind {
    Corotational(Box<CorotationalBeamFactor>),
    Der(Box<DerBeamFactor>),
}

/// Completely factored bipartite linear operator $H$ supporting $O(E + \mathrm{nnz}(L))$
/// multi-RHS solves `solve_rhs(rhs) -> H^{-1} rhs` (used both for velocity stepping and for
/// exact Delassus compliance matrix construction $C_c = J_c H^{-1} J_c^\top$!).
#[derive(Clone, Debug)]
pub struct FactoredBipartiteSystem {
    pub beam_factors: Vec<BeamFactorKind>,
    pub rigid_factor: RigidSchurFactor,
}

impl FactoredBipartiteSystem {
    /// Solve `H * x = rhs` for an arbitrary global 6D node RHS vector `rhs[0..total_nodes]`
    /// without re-factoring (exact for all corotational beams + all 3 rigid Schur routes;
    /// for DER beams, maps translational RHS onto DER vertices).
    pub fn solve_rhs(
        &self,
        rhs: &[Vec6],
        analysis: &GraphTopologyAnalysis,
    ) -> Result<Vec<Vec6>, LinearSolveError> {
        let empty_thetas: Vec<Option<&[f64]>> = vec![None; self.beam_factors.len()];
        let (out, _) = self.solve_rhs_with_der_thetas(rhs, &empty_thetas, analysis)?;
        Ok(out)
    }

    /// Solve the full coupled bipartite linear system `H * (x_nodes, x_der_thetas) = (rhs, der_theta_rhs)`
    /// without re-factoring, returning both global node 6D solutions and per-DER-beam edge twist
    /// solutions (used for exact $O(E + \mathrm{nnz}(L))$ IFT reverse-mode adjoints across DER rods!).
    #[allow(clippy::type_complexity)]
    pub fn solve_rhs_with_der_thetas(
        &self,
        rhs: &[Vec6],
        der_theta_rhs: &[Option<&[f64]>],
        analysis: &GraphTopologyAnalysis,
    ) -> Result<(Vec<Vec6>, Vec<Option<Vec<f64>>>), LinearSolveError> {
        let total = analysis.indexing.total_nodes;
        let mut r_free = vec![Vec6::ZERO; analysis.num_free_bodies];
        for (slot, &body_id) in analysis.free_body_ids.iter().enumerate() {
            r_free[slot] = rhs[body_id];
        }

        // Condense beam interior RHS onto free rigid bodies
        let mut beam_z0: Vec<Option<Vec<Vec6>>> = vec![None; self.beam_factors.len()];
        let mut der_z0: Vec<Option<Vec<f64>>> = vec![None; self.beam_factors.len()];
        for (s, fac) in self.beam_factors.iter().enumerate() {
            let map = &analysis.indexing.beam_maps[s];
            match fac {
                BeamFactorKind::Corotational(cf) => {
                    let int_rhs: Vec<Vec6> = map.free_node_ids.iter().map(|&nid| rhs[nid]).collect();
                    let (z0, ra, rb) = cf.condense_rhs(&int_rhs);
                    if let Some(u) = analysis.body_to_free_slot[map.node_a] {
                        r_free[u] += ra;
                    }
                    if let Some(bb) = map.node_b {
                        if let Some(v) = analysis.body_to_free_slot[bb] {
                            r_free[v] += rb;
                        }
                    }
                    beam_z0[s] = Some(z0);
                }
                BeamFactorKind::Der(df) => {
                    let int_rhs_verts: Vec<Vec3> =
                        map.free_node_ids.iter().map(|&nid| rhs[nid].lin()).collect();
                    let int_rhs_thetas = der_theta_rhs.get(s).and_then(|o| *o).unwrap_or(&[]);
                    let (z0, ra, rb) = df.condense_rhs_full(&int_rhs_verts, int_rhs_thetas);
                    if let Some(u) = analysis.body_to_free_slot[map.node_a] {
                        r_free[u] += ra;
                    }
                    if let Some(bb) = map.node_b {
                        if let Some(v) = analysis.body_to_free_slot[bb] {
                            r_free[v] += rb;
                        }
                    }
                    der_z0[s] = Some(z0);
                }
            }
        }

        // Solve rigid Schur system
        let x_free = self.rigid_factor.solve(&r_free)?;
        let mut out = vec![Vec6::ZERO; total];
        for (slot, &body_id) in analysis.free_body_ids.iter().enumerate() {
            out[body_id] = x_free[slot];
        }

        // Back-substitute beam interior nodes & DER twist DOFs
        let mut der_thetas_out: Vec<Option<Vec<f64>>> = vec![None; self.beam_factors.len()];
        for (s, fac) in self.beam_factors.iter().enumerate() {
            let map = &analysis.indexing.beam_maps[s];
            let va = out[map.node_a];
            let vb = map.node_b.map_or(Vec6::ZERO, |bb| out[bb]);
            match fac {
                BeamFactorKind::Corotational(cf) => {
                    if let Some(ref z0) = beam_z0[s] {
                        let vi = cf.back_substitute_with_z0(z0, va, vb);
                        for (i, &nid) in map.free_node_ids.iter().enumerate() {
                            out[nid] = vi[i];
                        }
                    }
                }
                BeamFactorKind::Der(df) => {
                    if let Some(ref z0) = der_z0[s] {
                        let (v_verts, v_thetas) = df.back_substitute_with_z0(z0, va, vb);
                        for (i, &nid) in map.free_node_ids.iter().enumerate() {
                            out[nid] = Vec6::from_parts(v_verts[i], Vec3::ZERO);
                        }
                        der_thetas_out[s] = Some(v_thetas);
                    }
                }
            }
        }
        Ok((out, der_thetas_out))
    }
}

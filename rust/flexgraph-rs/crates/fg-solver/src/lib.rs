//! `fg-solver`: Implicit Lie-Group Continuum & Bipartite Sparse Direct Solver for `flexgraph-rs`.
//!
//! Provides:
//! - **Dual Continuum Formulations** (`continuum`):
//!   - Cubic-Hermite Corotational 3D Euler-Bernoulli / Timoshenko beams (`< 1e-10` FP64 parity
//!     with `serpworm.physics.gpu_core`, plus exact analytic `beam_strain_jacobian` and
//!     Kirchhoff-Love / Vlasov thin-shell stiffening);
//!   - Discrete Elastic Rods (DER) / Cosserat rods with time-parallel Bishop frames, holonomy-aware
//!     discrete twist, exact 11x11 analytical Hessians (Jawed / Py-DiSMech 2025), and $O(E)$
//!     symmetric banded Schur condensation.
//! - **Bipartite Direct Solver** (`schur`):
//!   - Parallel `6x6` Block-Thomas elimination per beam (clamped-clamped, fixed Dirichlet base,
//!     and free cantilever tips `body_b = None`);
//!   - Compile-time Adaptive Rigid Schur Router: `BlockDiagonalPairs` (batched `12x12`/`6x6`),
//!     `TreeChain` ($O(N_b)$ zero-fill Block-Tree $LDL^\top$), and `SparseGeneral` (symbolic
//!     elimination-tree `6x6` Block-CSC $LDL^\top$);
//!   - Multi-RHS `FactoredBipartiteSystem::solve_rhs` enabling exact Delassus compliance
//!     $C_c = J_c H^{-1} J_c^\top$ evaluation without re-factoring.
//! - **Rich Actuation Mechanics** (`actuation`):
//!   - Multi-waypoint Capstan tendons (`EccentricCrank`, `LegacyWrap`, `Capstan`, `Linear`) with
//!     stick-slip Capstan hysteresis;
//!   - Articulated & compliant joints (`Revolute`, `Prismatic`, `Spherical`, `Weld`,
//!     `AxialStopGap`, `CableSpring`) with Stribeck static friction and DC torque-speed envelopes.
//! - **Implicit Lie-Group Time Integration & Tri-Engine Contact** (`integrator`, `world`):
//!   - `LinearlyImplicitEuler`, `DampedNewtonEuler`, `NewmarkBeta`, and `Bdf2` on $SE(3)$ with
//!     Armijo merit line search, conservative ACCD clamping, ARPC, Delassus-LCP, and Rayon
//!     multi-environment parallel stepping.

#![allow(clippy::needless_range_loop)]

pub mod actuation;
pub mod adjoint;
pub mod continuum;
pub mod integrator;
pub mod schur;
pub mod world;

pub use actuation::{
    advance_tendon_servos, evaluate_actuation, evaluate_actuation_with_masses, tendon_lever_arm,
    tendon_takeup, ActuationEval, ActuationState, TendonRuntimeState,
};
pub use adjoint::{
    compute_trajectory_adjoint, get_diff_param_value, set_diff_param_value, AdjointResult,
    PoseTrackingLoss, TrajectoryLoss,
};
pub use continuum::{
    project_spd_11x11, CorotationalBeamData, CorotationalBeamEval, DerAssembleOutput, DerBeamData,
    DerBeamFactor, DerBeamState, DER_HALF_BANDWIDTH,
};
pub use integrator::{
    BeamFactorKind, ContactBackend, EnergyMomentumReport, FactoredBipartiteSystem, IntegratorKind,
    SolverConfig, SolverError, StepDiagnostics, WorldState,
};
#[cfg(feature = "faer-sparse")]
pub use schur::{FaerSparseLdltFactor, FaerSparsePattern};
pub use schur::{
    factor_corotational_beams_parallel, solve_dense_reference, BeamSchurContribution,
    CorotationalBeamFactor, CorotationalBeamInput, LinearSolveError, PairBlocksFactor,
    RigidSchurFactor, RigidSchurMatrix, SparseBackend, SparseBlockLdltFactor, SymmetricBandedLdlt,
    TreeBlockLdltFactor,
};
pub use world::FlexGraphWorld;

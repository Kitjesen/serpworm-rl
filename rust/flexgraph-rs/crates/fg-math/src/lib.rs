//! `fg-math`: Pure-Rust FP64 Lie-group, spatial algebra, corotational beam,
//! and Discrete Elastic Rod (Cosserat/DER) mathematical primitives for `flexgraph-rs`.
//!
//! # Spatial Vector Convention
//! All 6D spatial twists and wrenches follow the `[linear (0..3); angular (3..6)]`
//! ordering to maintain bit-level convention parity with `serpworm.physics.gpu_core`
//! while providing exact Featherstone Plücker transformations and $SE(3)$ Lie-group
//! operators.
//!
//! # Quaternion Convention
//! Unit quaternions are stored in `[x, y, z, w]` (`[vector; scalar]`) layout with
//! canonical hemisphere `w >= 0` when extracted from rotations.

#![allow(clippy::needless_range_loop)]

pub mod beam;
pub mod cosserat;
pub mod lie;
pub mod solref;
pub mod spatial;
pub mod vec;

pub use beam::{
    beam_frame_transform, beam_local_strain, beam_strain_jacobian, cubic_hermite_midpoint,
    local_beam_matrices, BeamMaterial, BeamSection, CorotationalFrame,
};
pub use cosserat::{
    compute_bishop_frames, curvature_binormal, der_bending_energy_grad_hess,
    der_stretching_energy_grad_hess, der_twisting_energy_grad_hess, material_curvature,
    material_frame, parallel_transport, reference_twist, DerBendEval, DerStretchEval, DerTwistEval,
};
pub use lie::{
    se3_dexp_inv, se3_exp, se3_log, so3_dexp, so3_dexp_inv, so3_exp, so3_log, so3_right_jacobian,
    so3_right_jacobian_inv, Quat, Se3,
};
pub use solref::{SolRefImp, SOLIMP_MAX, SOLIMP_MIN};
pub use spatial::{
    adjoint_se3, coadjoint_se3, plucker_force_transform, plucker_motion_transform,
    point_force_to_wrench, point_wrench_jacobian, spatial_cross_force, spatial_cross_motion,
    SpatialInertia,
};
pub use vec::{Mat12, Mat3, Mat6, Vec12, Vec3, Vec6};

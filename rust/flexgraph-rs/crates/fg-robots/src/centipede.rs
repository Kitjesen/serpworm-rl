//! Archetype 4: `FlexSpine-Centipede` — multi-segment rigid torsos coupled by dual parallel
//! compliant steel leaf-spring spine beams, articulated left/right revolute hip legs, and
//! Bekker soft-soil isotropic footpads (`SchurRoute::TreeChain`, `Z2` bilateral symmetry).

use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, JointEdgeSpec,
    JointType, RigidBodySpec, RigidFlexGraph, SolverHints, SymmetryElement, SymmetryGroupKind,
    SymmetryGroupSpec,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};
use serde::{Deserialize, Serialize};

/// Configuration for `FlexSpine-Centipede` (`SchurRoute::TreeChain`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CentipedeConfig {
    /// Number of torso segments (`>= 3` recommended for torso-chain `TreeChain` routing).
    pub n_segments: usize,
    pub elements_per_spine: usize,
    pub seg_pitch_m: f64,
    pub torso_height_m: f64,
    /// When `true`, hip joints also add implicit edges to $\mathcal{G}_R$ (still an acyclic tree!).
    pub implicit_hip_coupling: bool,
    pub enable_self_contact: bool,
    pub preferred_dt: f64,
    /// Ground material profile ID (`2` = Bekker elasto-plastic soft soil).
    pub material_profile_id: usize,
}

impl Default for CentipedeConfig {
    fn default() -> Self {
        Self {
            n_segments: 3,
            elements_per_spine: 6,
            seg_pitch_m: 0.14,
            torso_height_m: 0.075,
            implicit_hip_coupling: true,
            enable_self_contact: true,
            preferred_dt: 0.001,
            material_profile_id: 2,
        }
    }
}

/// Build $\mathbb{Z}_2$ bilateral symmetry swapping left/right hip actuators on each segment.
pub fn make_centipede_z2_symmetry(n_segments: usize) -> SymmetryGroupSpec {
    let n_a = 2 * n_segments;
    let mut act_perm = Vec::with_capacity(n_a);
    let mut act_sign = Vec::with_capacity(n_a);
    for s in 0..n_segments {
        act_perm.push(2 * s + 1);
        act_perm.push(2 * s);
        act_sign.push(1.0);
        act_sign.push(1.0);
    }
    let mirror = SymmetryElement {
        name: "lr_mirror".to_string(),
        obs_perm: act_perm.clone(),
        obs_sign: act_sign.clone(),
        act_perm,
        act_sign,
        est_perm: Vec::new(),
        est_sign: Vec::new(),
    };
    SymmetryGroupSpec::from_generators(
        SymmetryGroupKind::Z2Bilateral,
        n_a,
        n_a,
        0,
        &[mirror],
        2,
    )
    .expect("Centipede Z2 symmetry closure must succeed")
}

/// Build the `FlexSpine-Centipede` rigid-flexible graph (`SchurRoute::TreeChain`).
pub fn build_flexspine_centipede(cfg: &CentipedeConfig) -> RigidFlexGraph {
    assert!(cfg.n_segments >= 2, "n_segments must be >= 2");
    assert!(cfg.elements_per_spine >= 2, "elements_per_spine must be >= 2");

    let mut bodies = Vec::with_capacity(3 * cfg.n_segments);
    let mut joints = Vec::with_capacity(2 * cfg.n_segments);
    let mut contact_sites = Vec::with_capacity(2 * cfg.n_segments);
    let mut torso_ids = Vec::with_capacity(cfg.n_segments);

    let z_body = cfg.torso_height_m;

    for s in 0..cfg.n_segments {
        let x_s = -(s as f64) * cfg.seg_pitch_m;
        let torso_id = bodies.len();
        torso_ids.push(torso_id);
        bodies.push(RigidBodySpec {
            id: torso_id,
            name: format!("torso_{s}"),
            is_fixed: false,
            mass: 0.35,
            inertia: Mat3::diag([0.0005, 0.0006, 0.0008]),
            com_local: Vec3::ZERO,
            rest_pose: Se3::new(Vec3::new(x_s, 0.0, z_body), Quat::IDENTITY),
            collision_radius: 0.04,
            rayleigh_alpha: 0.08,
            tags: vec![format!("segment_{s}"), "torso".to_string()],
        });

        for (side_i, y_sgn) in [1.0_f64, -1.0_f64].into_iter().enumerate() {
            let side_lbl = if side_i == 0 { "L" } else { "R" };
            let leg_id = bodies.len();
            bodies.push(RigidBodySpec {
                id: leg_id,
                name: format!("torso_{s}_leg_{side_lbl}"),
                is_fixed: false,
                mass: 0.06,
                inertia: Mat3::diag([0.00008, 0.00008, 0.00005]),
                com_local: Vec3::ZERO,
                rest_pose: Se3::new(
                    Vec3::new(x_s, y_sgn * 0.065, z_body - 0.030),
                    Quat::IDENTITY,
                ),
                collision_radius: 0.015,
                rayleigh_alpha: 0.08,
                tags: vec![format!("segment_{s}"), format!("leg_{side_lbl}")],
            });

            let act_id = 2 * s + side_i;
            let jid = joints.len();
            joints.push(JointEdgeSpec {
                id: jid,
                name: format!("hip_joint_{s}_{side_lbl}"),
                body_a: torso_id,
                body_b: leg_id,
                joint_type: JointType::Revolute {
                    axis_local_a: Vec3::EY,
                    anchor_a: Vec3::new(0.0, y_sgn * 0.065, -0.030),
                    anchor_b: Vec3::ZERO,
                    pos_stiffness: 12_000.0,
                    pos_damping: 15.0,
                    align_stiffness: 25.0,
                    align_damping: 0.15,
                },
                actuator_id: Some(act_id),
                kp: 8.0,
                kd: 0.15,
                torque_limit: 1.8,
                speed_limit: 6.0,
                static_friction: 0.02,
                deadband: 0.0,
                limits: Some([-0.6, 0.6]),
                implicit_coupling: cfg.implicit_hip_coupling,
                solref_imp: None,
            });

            let cid = contact_sites.len();
            contact_sites.push(ContactSiteSpec {
                id: cid,
                name: format!("foot_{s}_{side_lbl}"),
                attachment: ContactAttachment::RigidBody { body_id: leg_id },
                local_pos: Vec3::new(0.0, 0.0, -0.032),
                local_frame: Quat::IDENTITY,
                radius: 0.012,
                patch_radius: 0.012,
                mode: ContactMode::IsotropicPad {
                    mu: 0.85,
                    pivot_mu: 0.55,
                    static_mu_ratio: 1.12,
                    sdf_only: false,
                },
                material_profile_id: cfg.material_profile_id,
                solref_imp: None,
            });
        }
    }

    // Dual parallel compliant steel leaf-spring spine beams connecting consecutive torsos
    let leaf_mat = BeamMaterial {
        youngs_modulus: 200e9,
        poisson_ratio: 0.3,
        density: 7850.0,
        rayleigh_alpha: 0.08,
        rayleigh_beta: 0.001,
    };
    let leaf_sec = BeamSection::Rectangular {
        width: 0.018,
        thickness: 0.0006,
    };
    // Beam local +X points toward -world X (rotation of 180 deg around +Z)
    let q_neg_x = Quat::from_mat3(Mat3([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]));

    let mut beams = Vec::with_capacity(2 * (cfg.n_segments - 1));
    for s in 0..(cfg.n_segments - 1) {
        let ta = torso_ids[s];
        let tb = torso_ids[s + 1];
        let xa = bodies[ta].rest_pose.pos.x();
        let xb = bodies[tb].rest_pose.pos.x();
        for (y_off, lbl) in [(0.020_f64, "L"), (-0.020_f64, "R")] {
            let a_pt = Vec3::new(xa - 0.025, y_off, z_body);
            let b_pt = Vec3::new(xb + 0.025, y_off, z_body);
            let span = (b_pt - a_pt).norm();
            let elem_len = span / (cfg.elements_per_spine as f64);

            let mut poses = Vec::with_capacity(cfg.elements_per_spine + 1);
            for j in 0..=cfg.elements_per_spine {
                let u = (j as f64) / (cfg.elements_per_spine as f64);
                let pt = a_pt + u * (b_pt - a_pt);
                poses.push(Se3::new(pt, q_neg_x));
            }
            let clamp_a = Se3::new(Vec3::new(-0.025, y_off, 0.0), q_neg_x);
            let clamp_b = Se3::new(Vec3::new(0.025, y_off, 0.0), q_neg_x);
            let mut elem_tfs = vec![[Se3::IDENTITY, Se3::IDENTITY]; cfg.elements_per_spine];
            elem_tfs[0][0] = clamp_a;
            elem_tfs[cfg.elements_per_spine - 1][1] = clamp_b;

            let bid = beams.len();
            beams.push(BeamEdgeSpec {
                id: bid,
                name: format!("spine_{s}_{lbl}"),
                body_a: ta,
                clamp_a,
                body_b: Some(tb),
                clamp_b: Some(clamp_b),
                n_elements: cfg.elements_per_spine,
                formulation: BeamFormulation::CorotationalEulerBernoulli,
                section: leaf_sec,
                material: leaf_mat,
                thin_shell_compat: false,
                prestress_frac: 0.0,
                centerline_rest_poses: poses,
                element_lengths: vec![elem_len; cfg.elements_per_spine],
                element_transforms: Some(elem_tfs),
                interior_node_contact_radius: Some(0.003),
                enable_self_contact: cfg.enable_self_contact,
            });
        }
    }

    RigidFlexGraph {
        name: format!("flexspine_centipede_N{}", cfg.n_segments),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams,
        tendons: Vec::new(),
        joints,
        contact_sites,
        symmetry: make_centipede_z2_symmetry(cfg.n_segments),
        solver_hints: SolverHints {
            preferred_dt: cfg.preferred_dt,
            rayleigh_alpha: 0.08,
            rayleigh_beta: 0.001,
            force_route: None,
        },
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    }
}

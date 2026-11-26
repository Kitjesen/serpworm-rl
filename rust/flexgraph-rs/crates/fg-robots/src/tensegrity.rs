//! Archetype 5: `Tensegrity-6Bar` — classic 6-strut icosahedral tensegrity ball.
//!
//! Forms a genuinely **cyclic closed-loop rigid-flexible graph** ($K_{2,2,2}$ octahedron graph:
//! 6 free rigid struts, 12 undirected strut-pair couplings, 24 flexible elastic cables +
//! 24 prestressed linear tendon actuators, cyclomatic number $\beta_1 = 12 - 6 + 1 = 7$)
//! that automatically routes to `SchurRoute::SparseGeneral` (Sparse Block-$LDL^\top$).
//!
//! At the exact self-equilibrated aspect ratio $a = 2b$ (half-strut length $a = L/2$,
//! parallel-pair offset $b = L/4$), all 24 cables have identical length $d = \sqrt{6}\,b$
//! and equal cable tension $T$ produces exact zero net force and zero net torque on every
//! strut (internal strut compression $C = \sqrt{6}\,T$).

use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, RigidBodySpec,
    RigidFlexGraph, SolverHints, SymmetryElement, SymmetryGroupKind, SymmetryGroupSpec,
    TendonDriveMode, TendonEdgeSpec, TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};
use serde::{Deserialize, Serialize};

/// Configuration for the cyclic 6-strut icosahedral tensegrity ball (`SchurRoute::SparseGeneral`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensegrityConfig {
    /// Total length of each rigid strut (m).
    pub strut_length_m: f64,
    /// Outer radius of each rigid strut (m).
    pub strut_radius_m: f64,
    /// Mass of each rigid strut including endcaps (kg).
    pub strut_mass_kg: f64,
    /// Ground-contact endcap sphere radius (m).
    pub endcap_radius_m: f64,
    /// Number of beam elements along each elastic cable (`>= 2`).
    pub elements_per_cable: usize,
    /// Radius of each flexible elastic cable beam (m).
    pub cable_radius_m: f64,
    /// Young's modulus of the flexible elastomer/nylon cables (Pa).
    pub cable_youngs_modulus_pa: f64,
    /// Parallel actuated tendon stiffness (N/m) for prestress & active rolling control.
    pub tendon_stiffness_n_m: f64,
    /// Tendon damping (N*s/m).
    pub tendon_damping_ns_m: f64,
    /// Fractional tensile prestress `(1 - L_rest / L_geom)` in `[0, 0.25]` for self-equilibrium.
    pub prestress_strain: f64,
    /// Preferred integration timestep (s).
    pub preferred_dt: f64,
    /// Ground material profile ID.
    pub material_profile_id: usize,
}

impl Default for TensegrityConfig {
    fn default() -> Self {
        Self {
            strut_length_m: 0.40,
            strut_radius_m: 0.008,
            strut_mass_kg: 0.12,
            endcap_radius_m: 0.015,
            elements_per_cable: 2,
            cable_radius_m: 0.0015,
            cable_youngs_modulus_pa: 2.0e8, // Flexible polyurethane/nylon cable
            tendon_stiffness_n_m: 450.0,
            tendon_damping_ns_m: 2.0,
            prestress_strain: 0.05,
            preferred_dt: 0.001,
            material_profile_id: 0,
        }
    }
}

/// Canonical 24 cable endpoint pairs `((strut_a, sgn_a), (strut_b, sgn_b))` of the
/// 6-bar icosahedral tensegrity ball, where `sgn in {-1, +1}` selects endpoint `±a`
/// along each strut's local `+X` axis.
///
/// Strut layout:
/// - `0, 1`: parallel to world `+Z`, centers `(0, +b, z_c)` and `(0, -b, z_c)`
/// - `2, 3`: parallel to world `+X`, centers `(0, 0, z_c + b)` and `(0, 0, z_c - b)`
/// - `4, 5`: parallel to world `+Y`, centers `(+b, 0, z_c)` and `(-b, 0, z_c)`
pub fn tensegrity_6bar_cable_pairs() -> [((usize, i8), (usize, i8)); 24] {
    [
        // Z-pair (0, 1) <-> X-pair (2, 3) (8 cables)
        ((0, 1), (2, 1)),
        ((0, 1), (2, -1)),
        ((1, 1), (2, 1)),
        ((1, 1), (2, -1)),
        ((0, -1), (3, 1)),
        ((0, -1), (3, -1)),
        ((1, -1), (3, 1)),
        ((1, -1), (3, -1)),
        // Z-pair (0, 1) <-> Y-pair (4, 5) (8 cables)
        ((0, 1), (4, 1)),
        ((0, 1), (5, 1)),
        ((0, -1), (4, 1)),
        ((0, -1), (5, 1)),
        ((1, 1), (4, -1)),
        ((1, 1), (5, -1)),
        ((1, -1), (4, -1)),
        ((1, -1), (5, -1)),
        // X-pair (2, 3) <-> Y-pair (4, 5) (8 cables)
        ((2, 1), (4, 1)),
        ((2, 1), (4, -1)),
        ((3, 1), (4, 1)),
        ((3, 1), (4, -1)),
        ((2, -1), (5, 1)),
        ((2, -1), (5, -1)),
        ((3, -1), (5, 1)),
        ((3, -1), (5, -1)),
    ]
}

/// Construct an orthonormal right-handed rotation quaternion whose local `+X` axis
/// equals `dir` (unit vector).
fn frame_from_x_axis(dir: Vec3) -> Quat {
    let tx = dir.normalized();
    let helper = if tx.z().abs() < 0.9 {
        Vec3::EZ
    } else {
        Vec3::EY
    };
    let ty = helper.cross(tx).normalized();
    let tz = tx.cross(ty);
    Quat::from_mat3(Mat3([
        [tx.x(), ty.x(), tz.x()],
        [tx.y(), ty.y(), tz.y()],
        [tx.z(), ty.z(), tz.z()],
    ]))
}

/// Compute the maximum static force and torque residual across all 6 struts under
/// uniform unit cable tension `T = 1.0 N` (verifies exact self-equilibrated prestress).
pub fn tensegrity_6bar_equilibrium_residual(graph: &RigidFlexGraph) -> (f64, f64) {
    let mut forces = [Vec3::ZERO; 6];
    let mut torques = [Vec3::ZERO; 6];
    for tendon in &graph.tendons {
        let wa = tendon.waypoints[0];
        let wb = tendon.waypoints[1];
        let ba = &graph.bodies[wa.body_id];
        let bb = &graph.bodies[wb.body_id];
        let pa = ba.rest_pose.transform_point(wa.local_pos);
        let pb = bb.rest_pose.transform_point(wb.local_pos);
        let dir = (pb - pa).normalized();
        let ra = pa - ba.rest_pose.pos;
        let rb = pb - bb.rest_pose.pos;
        forces[wa.body_id] += dir;
        torques[wa.body_id] += ra.cross(dir);
        forces[wb.body_id] -= dir;
        torques[wb.body_id] += rb.cross(-dir);
    }
    let max_f = forces.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
    let max_t = torques.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
    (max_f, max_t)
}

/// Build cyclic 3-fold axis-permutation symmetry `C3` (X->Y->Z->X) over the 24 tensegrity cables.
fn make_tensegrity_c3_symmetry() -> SymmetryGroupSpec {
    // Under cyclic coordinate permutation (x, y, z) -> (y, z, x):
    // Struts map: 0(Z,+Y)->2(X,+Z)->4(Y,+X)->0, and 1(Z,-Y)->3(X,-Z)->5(Y,-X)->1
    let pairs = tensegrity_6bar_cable_pairs();
    let map_strut = |s: usize| -> usize {
        match s {
            0 => 2,
            1 => 3,
            2 => 4,
            3 => 5,
            4 => 0,
            5 => 1,
            _ => unreachable!(),
        }
    };
    let mut perm = Vec::with_capacity(24);
    for &((sa, ea), (sb, eb)) in &pairs {
        let a2 = (map_strut(sa), ea);
        let b2 = (map_strut(sb), eb);
        let idx = pairs
            .iter()
            .position(|&(p, q)| (p == a2 && q == b2) || (p == b2 && q == a2))
            .expect("icosahedral cable set is closed under C3 cyclic axis permutation");
        perm.push(idx);
    }
    let c3_gen = SymmetryElement {
        name: "c3_xyz".to_string(),
        obs_perm: perm.clone(),
        obs_sign: vec![1.0; 24],
        act_perm: perm,
        act_sign: vec![1.0; 24],
        est_perm: Vec::new(),
        est_sign: Vec::new(),
    };
    SymmetryGroupSpec::from_generators(
        SymmetryGroupKind::Custom("C3_icosahedral".to_string()),
        24,
        24,
        0,
        &[c3_gen],
        3,
    )
    .expect("C3 symmetry closure must succeed")
}

/// Build the cyclic 6-strut icosahedral tensegrity ball (`SchurRoute::SparseGeneral`, $\beta_1 = 7$).
pub fn build_tensegrity_6bar(cfg: &TensegrityConfig) -> RigidFlexGraph {
    assert!(cfg.strut_length_m > 0.0, "strut_length_m must be > 0");
    assert!(cfg.elements_per_cable >= 2, "elements_per_cable must be >= 2");

    let a = 0.5 * cfg.strut_length_m;
    let b = 0.5 * a; // Exact equilibrium aspect ratio a = 2*b => all 24 cables have length sqrt(6)*b
    let z_c = a + cfg.endcap_radius_m + 0.002;

    // Cylinder inertia with local +X along strut axis
    let m = cfg.strut_mass_kg;
    let r = cfg.strut_radius_m;
    let l = cfg.strut_length_m;
    let i_ax = 0.5 * m * r * r;
    let i_tr = (m / 12.0) * (3.0 * r * r + l * l);
    let strut_inertia = Mat3::diag([i_ax, i_tr, i_tr]);

    // 6 orthogonal rigid struts (local +X aligned with strut axis)
    let strut_defs: [(&str, Vec3, Vec3); 6] = [
        ("strut_z_pos_y", Vec3::new(0.0, b, z_c), Vec3::EZ),
        ("strut_z_neg_y", Vec3::new(0.0, -b, z_c), Vec3::EZ),
        ("strut_x_pos_z", Vec3::new(0.0, 0.0, z_c + b), Vec3::EX),
        ("strut_x_neg_z", Vec3::new(0.0, 0.0, z_c - b), Vec3::EX),
        ("strut_y_pos_x", Vec3::new(b, 0.0, z_c), Vec3::EY),
        ("strut_y_neg_x", Vec3::new(-b, 0.0, z_c), Vec3::EY),
    ];

    let mut bodies = Vec::with_capacity(6);
    for (id, &(name, center, axis)) in strut_defs.iter().enumerate() {
        let rot = frame_from_x_axis(axis);
        bodies.push(RigidBodySpec {
            id,
            name: name.to_string(),
            is_fixed: false,
            mass: m,
            inertia: strut_inertia,
            com_local: Vec3::ZERO,
            rest_pose: Se3::new(center, rot),
            collision_radius: cfg.endcap_radius_m,
            rayleigh_alpha: 0.12,
            tags: vec!["tensegrity_strut".to_string()],
        });
    }

    // 12 isotropic endcap ground contact sites (2 per strut)
    let mut contact_sites = Vec::with_capacity(12);
    for s in 0..6 {
        for &(sgn, lbl) in &[(-1.0_f64, "neg"), (1.0_f64, "pos")] {
            let cid = contact_sites.len();
            contact_sites.push(ContactSiteSpec {
                id: cid,
                name: format!("strut_{s}_cap_{lbl}"),
                attachment: ContactAttachment::RigidBody { body_id: s },
                local_pos: Vec3::new(sgn * a, 0.0, 0.0),
                local_frame: Quat::IDENTITY,
                radius: cfg.endcap_radius_m,
                patch_radius: 0.006,
                mode: ContactMode::IsotropicPad {
                    mu: 0.75,
                    pivot_mu: 0.45,
                    static_mu_ratio: 1.10,
                    sdf_only: false,
                },
                material_profile_id: cfg.material_profile_id,
                solref_imp: None,
            });
        }
    }

    // 24 flexible elastic cables (beams) + 24 parallel prestressed tendon actuators
    let cable_mat = BeamMaterial {
        youngs_modulus: cfg.cable_youngs_modulus_pa,
        poisson_ratio: 0.40,
        density: 1150.0,
        rayleigh_alpha: 0.12,
        rayleigh_beta: 0.002,
    };
    let cable_sec = BeamSection::Circular {
        radius: cfg.cable_radius_m,
    };

    let pairs = tensegrity_6bar_cable_pairs();
    let mut beams = Vec::with_capacity(24);
    let mut tendons = Vec::with_capacity(24);

    for (cid, &((sa, ea), (sb, eb))) in pairs.iter().enumerate() {
        let local_a = Vec3::new((ea as f64) * a, 0.0, 0.0);
        let local_b = Vec3::new((eb as f64) * a, 0.0, 0.0);
        let world_a = bodies[sa].rest_pose.transform_point(local_a);
        let world_b = bodies[sb].rest_pose.transform_point(local_b);
        let delta = world_b - world_a;
        let cable_len = delta.norm();
        let q_world = frame_from_x_axis(delta);

        // Local clamp orientation on body_a and body_b so that body.rot * clamp.rot == q_world
        let q_clamp_a = bodies[sa].rest_pose.rot.conj().mul(q_world).normalized();
        let q_clamp_b = bodies[sb].rest_pose.rot.conj().mul(q_world).normalized();
        let clamp_a = Se3::new(local_a, q_clamp_a);
        let clamp_b = Se3::new(local_b, q_clamp_b);

        let n_e = cfg.elements_per_cable;
        let elem_len = cable_len / (n_e as f64);
        let mut poses = Vec::with_capacity(n_e + 1);
        for j in 0..=n_e {
            let u = (j as f64) / (n_e as f64);
            poses.push(Se3::new(world_a + u * delta, q_world));
        }

        let mut elem_tfs = vec![[Se3::IDENTITY, Se3::IDENTITY]; n_e];
        elem_tfs[0][0] = clamp_a;
        elem_tfs[n_e - 1][1] = clamp_b;

        beams.push(BeamEdgeSpec {
            id: cid,
            name: format!("cable_beam_{cid}_s{sa}_s{sb}"),
            body_a: sa,
            clamp_a,
            body_b: Some(sb),
            clamp_b: Some(clamp_b),
            n_elements: n_e,
            formulation: BeamFormulation::CorotationalEulerBernoulli,
            section: cable_sec,
            material: cable_mat,
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: poses,
            element_lengths: vec![elem_len; n_e],
            element_transforms: Some(elem_tfs),
            interior_node_contact_radius: None,
            enable_self_contact: false,
        });

        let rest_len = cable_len * (1.0 - cfg.prestress_strain.clamp(0.0, 0.25));
        tendons.push(TendonEdgeSpec {
            id: cid,
            name: format!("cable_tendon_{cid}_s{sa}_s{sb}"),
            actuator_id: cid,
            waypoints: vec![
                TendonWaypoint {
                    body_id: sa,
                    local_pos: local_a,
                    wrap_angle_rad: 0.0,
                },
                TendonWaypoint {
                    body_id: sb,
                    local_pos: local_b,
                    wrap_angle_rad: 0.0,
                },
            ],
            drive_mode: TendonDriveMode::Linear,
            rest_length: rest_len,
            slack: 0.0,
            stiffness: cfg.tendon_stiffness_n_m,
            damping: cfg.tendon_damping_ns_m,
            capstan_mu_theta: 0.0,
            stick_slip_enabled: false,
            servo_torque_limit: 5.0,
            servo_speed_limit: 0.25,
            servo_deadband: 0.0,
            angle_limits: [-0.04, 0.06],
            solref_imp: None,
        });
    }

    RigidFlexGraph {
        name: "tensegrity_6bar_icosahedron".to_string(),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams,
        tendons,
        joints: Vec::new(),
        contact_sites,
        symmetry: make_tensegrity_c3_symmetry(),
        solver_hints: SolverHints {
            preferred_dt: cfg.preferred_dt,
            rayleigh_alpha: 0.12,
            rayleigh_beta: 0.002,
            force_route: None,
        },
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    }
}

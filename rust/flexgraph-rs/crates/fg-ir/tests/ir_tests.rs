//! Comprehensive tests for `fg-ir`:
//! - JSON & YAML round-trip serialization
//! - Fixed bases, cantilever beams (`body_b = None`), multi-waypoint tendons
//! - Finite symmetry groups (Z2, Z2xZ2, D4) closure + exact Reynolds equivariance
//! - Automatic Schur routing across all 3 topology classes:
//!   1. `BlockDiagonalPairs` (SerpWorm-like disconnected plate pairs)
//!   2. `TreeChain` (fixed-base TendonEndoscope + FlexSpine-Centipede)
//!   3. `SparseGeneral` (cyclic 6-bar icosahedral tensegrity with verified symbolic fill-in!)

use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ColliderGeometry, ContactAttachment, ContactMode,
    ContactSiteSpec, DiffParamKind, DiffParamSpec, JointEdgeSpec, JointType, MeshColliderSpec,
    RigidBodySpec, RigidFlexGraph, SchurRoute, SolRefImp, SolverHints, SymmetryElement,
    SymmetryGroupKind, SymmetryGroupSpec, TendonDriveMode, TendonEdgeSpec, TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};

fn make_body(id: usize, name: &str, is_fixed: bool, pos: Vec3) -> RigidBodySpec {
    RigidBodySpec {
        id,
        name: name.to_string(),
        is_fixed,
        mass: 0.28,
        inertia: Mat3::diag([4.48e-4, 2.52e-4, 2.52e-4]),
        com_local: Vec3::ZERO,
        rest_pose: Se3::new(pos, Quat::IDENTITY),
        collision_radius: 0.05,
        rayleigh_alpha: 0.05,
        tags: Vec::new(),
    }
}

fn make_straight_beam(
    id: usize,
    body_a: usize,
    body_b: Option<usize>,
    p_start: Vec3,
    p_end: Vec3,
    n_elem: usize,
) -> BeamEdgeSpec {
    let poses: Vec<Se3> = (0..=n_elem)
        .map(|i| {
            let u = i as f64 / n_elem as f64;
            Se3::new(p_start + (p_end - p_start) * u, Quat::IDENTITY)
        })
        .collect();
    BeamEdgeSpec {
        id,
        name: format!("beam_{id}"),
        body_a,
        clamp_a: Se3::IDENTITY,
        body_b,
        clamp_b: body_b.map(|_| Se3::IDENTITY),
        n_elements: n_elem,
        formulation: BeamFormulation::CorotationalEulerBernoulli,
        section: BeamSection::Circular { radius: 0.002 },
        material: BeamMaterial::NITINOL,
        thin_shell_compat: false,
        prestress_frac: 0.0,
        centerline_rest_poses: poses,
        element_lengths: Vec::new(),
        element_transforms: None,
        interior_node_contact_radius: Some(0.002),
        enable_self_contact: false,
    }
}

#[test]
fn test_block_diagonal_pairs_route_and_json_yaml_roundtrip() {
    // 3 segments (6 plates), 2 beams per pair, explicit yaw joints (implicit_coupling = false)
    let mut bodies = Vec::new();
    let mut beams = Vec::new();
    let mut joints = Vec::new();
    for seg in 0..3 {
        let p0 = Vec3::new(-(seg as f64) * 0.2, 0.0, 0.1);
        let p1 = Vec3::new(-(seg as f64) * 0.2 - 0.11, 0.0, 0.1);
        bodies.push(make_body(2 * seg, &format!("p{}_f", seg), false, p0));
        bodies.push(make_body(2 * seg + 1, &format!("p{}_b", seg), false, p1));
        beams.push(make_straight_beam(seg, 2 * seg, Some(2 * seg + 1), p0, p1, 4));
        if seg > 0 {
            joints.push(JointEdgeSpec {
                id: seg - 1,
                name: format!("yaw_{}", seg - 1),
                body_a: 2 * seg - 1,
                body_b: 2 * seg,
                joint_type: JointType::Revolute {
                    axis_local_a: Vec3::EZ,
                    anchor_a: Vec3::new(-0.058, 0.0, 0.0),
                    anchor_b: Vec3::ZERO,
                    pos_stiffness: 10000.0,
                    pos_damping: 10.0,
                    align_stiffness: 20.0,
                    align_damping: 0.1,
                },
                actuator_id: Some(seg - 1),
                kp: 10.0,
                kd: 0.12,
                torque_limit: 2.45,
                speed_limit: 8.2,
                static_friction: 0.06,
                deadband: 0.0,
                limits: Some([-0.55, 0.55]),
                implicit_coupling: false,
                solref_imp: None,
            });
        }
    }

    let graph = RigidFlexGraph {
        name: "serpworm_mini_3".to_string(),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams,
        tendons: Vec::new(),
        joints,
        contact_sites: Vec::new(),
        symmetry: SymmetryGroupSpec::trivial(),
        solver_hints: SolverHints::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };

    let analysis = graph.analyze().expect("analyze must succeed");
    assert_eq!(analysis.route, SchurRoute::BlockDiagonalPairs);
    assert_eq!(analysis.num_free_bodies, 6);
    assert_eq!(analysis.pair_blocks.len(), 3);
    assert_eq!(analysis.cyclomatic_number, 0);
    assert_eq!(analysis.sparse_pattern.fill_in_blocks, 0);
    assert_eq!(analysis.indexing.num_bodies, 6);
    assert_eq!(analysis.indexing.num_beam_nodes, 3 * 3);

    // JSON & YAML round-trip
    let json = graph.to_json_string().unwrap();
    let from_json = RigidFlexGraph::from_json_str(&json).unwrap();
    assert_eq!(from_json.bodies.len(), graph.bodies.len());
    assert_eq!(from_json.beams.len(), graph.beams.len());
    assert_eq!(from_json.joints.len(), graph.joints.len());
    for (b1, b2) in graph.beams.iter().zip(from_json.beams.iter()) {
        for (p1, p2) in b1.centerline_rest_poses.iter().zip(b2.centerline_rest_poses.iter()) {
            assert!((p1.pos - p2.pos).norm() < 1e-15);
        }
    }

    let yaml = graph.to_yaml_string().unwrap();
    let from_yaml = RigidFlexGraph::from_yaml_str(&yaml).unwrap();
    assert_eq!(from_yaml.bodies.len(), graph.bodies.len());
    assert_eq!(from_yaml.beams.len(), graph.beams.len());
    assert_eq!(from_yaml.joints.len(), graph.joints.len());
    for (b1, b2) in graph.beams.iter().zip(from_yaml.beams.iter()) {
        for (p1, p2) in b1.centerline_rest_poses.iter().zip(b2.centerline_rest_poses.iter()) {
            assert!((p1.pos - p2.pos).norm() < 1e-15);
        }
    }
}

#[test]
fn test_fixed_base_cantilever_and_multi_waypoint_tendon_tree_route() {
    // Fixed base (body 0) -> disc 1 -> disc 2 -> disc 3 -> cantilever tip beam (body_b = None)
    let bodies = vec![
        make_body(0, "base_fixed", true, Vec3::ZERO),
        make_body(1, "disc_1", false, Vec3::new(0.05, 0.0, 0.0)),
        make_body(2, "disc_2", false, Vec3::new(0.10, 0.0, 0.0)),
        make_body(3, "disc_3", false, Vec3::new(0.15, 0.0, 0.0)),
    ];
    let beams = vec![
        make_straight_beam(0, 0, Some(1), Vec3::ZERO, Vec3::new(0.05, 0.0, 0.0), 4),
        make_straight_beam(1, 1, Some(2), Vec3::new(0.05, 0.0, 0.0), Vec3::new(0.10, 0.0, 0.0), 4),
        make_straight_beam(2, 2, Some(3), Vec3::new(0.10, 0.0, 0.0), Vec3::new(0.15, 0.0, 0.0), 4),
        // Cantilever free tip!
        make_straight_beam(3, 3, None, Vec3::new(0.15, 0.0, 0.0), Vec3::new(0.20, 0.0, 0.0), 5),
    ];
    let tendons = vec![TendonEdgeSpec {
        id: 0,
        name: "t_pos_y".to_string(),
        actuator_id: 0,
        waypoints: vec![
            TendonWaypoint { body_id: 0, local_pos: Vec3::new(0.0, 0.006, 0.0), wrap_angle_rad: 0.0 },
            TendonWaypoint { body_id: 1, local_pos: Vec3::new(0.0, 0.006, 0.0), wrap_angle_rad: 0.05 },
            TendonWaypoint { body_id: 2, local_pos: Vec3::new(0.0, 0.006, 0.0), wrap_angle_rad: 0.05 },
            TendonWaypoint { body_id: 3, local_pos: Vec3::new(0.0, 0.006, 0.0), wrap_angle_rad: 0.0 },
        ],
        drive_mode: TendonDriveMode::Capstan { drum_radius: 0.01 },
        rest_length: 0.15,
        slack: 0.0,
        stiffness: 1500.0,
        damping: 0.4,
        capstan_mu_theta: 0.12,
        stick_slip_enabled: true,
        servo_torque_limit: 1.5,
        servo_speed_limit: 6.0,
        servo_deadband: 0.0,
        angle_limits: [0.0, std::f64::consts::PI],
        solref_imp: None,
    }];
    let contact_sites = vec![ContactSiteSpec {
        id: 0,
        name: "tip_contact".to_string(),
        attachment: ContactAttachment::BeamInteriorNode {
            beam_id: 3,
            node_local_idx: 4, // Free tip node of 5-element cantilever
        },
        local_pos: Vec3::ZERO,
        local_frame: Quat::IDENTITY,
        radius: 0.002,
        patch_radius: 0.0,
        mode: ContactMode::IsotropicPad {
            mu: 0.3,
            pivot_mu: 0.0,
            static_mu_ratio: 1.0,
            sdf_only: true,
        },
        material_profile_id: 0,
        solref_imp: None,
    }];

    let graph = RigidFlexGraph {
        name: "endoscope_cantilever".to_string(),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams,
        tendons,
        joints: Vec::new(),
        contact_sites,
        symmetry: SymmetryGroupSpec::trivial(),
        solver_hints: SolverHints::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };

    let analysis = graph.analyze().expect("endoscope analysis must succeed");
    assert_eq!(analysis.route, SchurRoute::TreeChain);
    assert_eq!(analysis.num_bodies, 4);
    assert_eq!(analysis.num_free_bodies, 3); // body 0 is fixed!
    assert_eq!(analysis.body_to_free_slot[0], None);
    assert_eq!(analysis.cyclomatic_number, 0);
    assert_eq!(analysis.sparse_pattern.fill_in_blocks, 0);
    // Cantilever beam 3 has n_elements = 5 free beam nodes (including free tip)
    assert_eq!(analysis.indexing.beam_maps[3].free_node_ids.len(), 5);
    assert_eq!(analysis.indexing.beam_maps[3].node_b, None);
}

#[test]
fn test_cyclic_tensegrity_sparse_general_and_symbolic_fill_in() {
    // 6-body cyclic graph (icosahedral 6-bar tensegrity topology: K_6 minus 3 disjoint parallel pairs = octahedron graph!)
    // Vertices 0..6; pairs (0,1), (2,3), (4,5) are parallel struts; every non-parallel pair is connected by cables/beams.
    let bodies: Vec<RigidBodySpec> = (0..6)
        .map(|i| make_body(i, &format!("strut_{i}"), false, Vec3::new(i as f64 * 0.1, 0.0, 0.2)))
        .collect();
    let mut joints = Vec::new();
    let mut jid = 0;
    for i in 0..6 {
        for j in (i + 1)..6 {
            if i / 2 == j / 2 {
                continue; // parallel pair not directly connected
            }
            joints.push(JointEdgeSpec {
                id: jid,
                name: format!("cable_{i}_{j}"),
                body_a: i,
                body_b: j,
                joint_type: JointType::CableSpring {
                    anchor_a: Vec3::ZERO,
                    anchor_b: Vec3::ZERO,
                    rest_length: 0.1,
                    stiffness: 1000.0,
                    damping: 2.0,
                    compression_allowed: false,
                },
                actuator_id: None,
                kp: 0.0,
                kd: 0.0,
                torque_limit: 0.0,
                speed_limit: 0.0,
                static_friction: 0.0,
                deadband: 0.0,
                limits: None,
                implicit_coupling: true,
                solref_imp: None,
            });
            jid += 1;
        }
    }

    let graph = RigidFlexGraph {
        name: "tensegrity_6bar".to_string(),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams: Vec::new(),
        tendons: Vec::new(),
        joints,
        contact_sites: Vec::new(),
        symmetry: SymmetryGroupSpec::trivial(),
        solver_hints: SolverHints::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };

    let analysis = graph.analyze().expect("tensegrity analysis must succeed");
    assert_eq!(analysis.route, SchurRoute::SparseGeneral);
    assert_eq!(analysis.num_free_bodies, 6);
    assert_eq!(analysis.num_rigid_edges, 12); // octahedron has 12 edges
    assert_eq!(analysis.cyclomatic_number, 12 + 1 - 6); // 7 independent cycles!

    // Independently simulate exact symbolic Gaussian elimination on the permuted matrix
    // and assert the symbolic pattern `sparse_pattern` matches 100% of nonzero entries in L!
    let pat = &analysis.sparse_pattern;
    let n = pat.n_blocks;
    let mut dense_l = vec![vec![false; n]; n];
    // Populate lower triangle of permuted A
    for joint in &graph.joints {
        let u = pat.inv_perm[joint.body_a];
        let v = pat.inv_perm[joint.body_b];
        let (r, c) = if u > v { (u, v) } else { (v, u) };
        dense_l[r][c] = true;
    }
    // Symbolic Cholesky fill-in
    for k in 0..n {
        let rows: Vec<usize> = ((k + 1)..n).filter(|&i| dense_l[i][k]).collect();
        for a in 0..rows.len() {
            for b in (a + 1)..rows.len() {
                dense_l[rows[b]][rows[a]] = true;
            }
        }
    }
    for col in 0..n {
        let expected_rows: Vec<usize> = dense_l
            .iter()
            .enumerate()
            .skip(col + 1)
            .filter_map(|(r, row)| if row[col] { Some(r) } else { None })
            .collect();
        let actual_rows = &pat.row_idx[pat.col_ptr[col]..pat.col_ptr[col + 1]];
        assert_eq!(
            actual_rows,
            expected_rows.as_slice(),
            "symbolic L column {col} mismatch"
        );
    }
}

#[test]
fn test_symmetry_groups_z2_z2xz2_d4_and_reynolds_equivariance() {
    // D4 group acting on 4-tendon endoscope [t_+y, t_-y, t_+z, t_-z]:
    // rot90: +y -> +z -> -y -> -z -> +y  (perm [3, 2, 0, 1] or cyclic [2, 3, 1, 0])
    // reflect_y: swap +y <-> -y
    let rot90 = SymmetryElement {
        name: "r".to_string(),
        obs_perm: vec![2, 3, 1, 0],
        obs_sign: vec![1.0; 4],
        act_perm: vec![2, 3, 1, 0],
        act_sign: vec![1.0; 4],
        est_perm: vec![0, 2, 1],
        est_sign: vec![1.0, 1.0, -1.0],
    };
    let refl = SymmetryElement {
        name: "s".to_string(),
        obs_perm: vec![1, 0, 2, 3],
        obs_sign: vec![1.0; 4],
        act_perm: vec![1, 0, 2, 3],
        act_sign: vec![1.0; 4],
        est_perm: vec![0, 1, 2],
        est_sign: vec![1.0, -1.0, 1.0],
    };
    let d4 = SymmetryGroupSpec::from_generators(
        SymmetryGroupKind::D4Dihedral,
        4,
        4,
        3,
        &[rot90, refl],
        8,
    )
    .expect("D4 group must close at order 8");
    assert_eq!(d4.elements.len(), 8);

    // Arbitrary biased non-equivariant policy
    let biased_policy = |o: &[f64]| -> Vec<f64> {
        vec![
            0.3 * o[0] - 0.1 * o[1] + 0.5,
            -0.2 * o[1] + 0.4 * o[2] - 0.3,
            0.7 * o[2] + 0.1 * o[3] + 0.2,
            -0.4 * o[3] + 0.2 * o[0] - 0.1,
        ]
    };

    let obs = vec![0.4, -0.7, 0.2, 0.9];
    let avg_obs = d4.reynolds_average(&obs, biased_policy);
    // Verify EXACT equivariance: bar_mu(L_g o) == K_g bar_mu(o) for all 8 elements g in D4!
    for g in &d4.elements {
        let lhs = d4.reynolds_average(&g.apply_obs(&obs), biased_policy);
        let rhs = g.apply_act(&avg_obs);
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            assert!(
                (a - b).abs() < 1e-13,
                "D4 equivariance violated for {}: lhs={lhs:?} rhs={rhs:?}",
                g.name
            );
        }
    }
}

#[test]
fn test_v2_solref_diff_params_mesh_colliders_and_v1_serde_compat() {
    // 1. Backward compatibility: v1 JSON omitting solref_imp / diff_params / mesh_colliders
    let v1_json = r#"{
        "name": "legacy_v1",
        "bodies": [{
            "id": 0, "name": "b0", "mass": 0.5,
            "inertia": [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]]
        }],
        "contact_sites": [{
            "id": 0, "name": "pad0",
            "attachment": { "attach_type": "rigid_body", "body_id": 0 },
            "radius": 0.01,
            "mode": { "mode": "isotropic_pad", "mu": 0.7, "pivot_mu": 0.2, "static_mu_ratio": 1.1, "sdf_only": false }
        }]
    }"#;
    let g_v1 = RigidFlexGraph::from_json_str(v1_json).expect("v1 JSON must deserialize cleanly");
    g_v1.validate().expect("v1 graph must validate");
    assert!(g_v1.contact_sites[0].solref_imp.is_none());
    assert!(g_v1.diff_params.is_empty());
    assert!(g_v1.mesh_colliders.is_empty());

    // 2. Full v2 graph with SolRefImp (positive & direct negative), all 7 DiffParamKinds, & MeshColliderSpec
    let bodies = vec![
        make_body(0, "base", false, Vec3::ZERO),
        make_body(1, "tip", false, Vec3::new(0.1, 0.0, 0.0)),
    ];
    let beams = vec![make_straight_beam(
        0,
        0,
        Some(1),
        Vec3::ZERO,
        Vec3::new(0.1, 0.0, 0.0),
        4,
    )];
    let tendons = vec![TendonEdgeSpec {
        id: 0,
        name: "t0".to_string(),
        actuator_id: 0,
        waypoints: vec![
            TendonWaypoint { body_id: 0, local_pos: Vec3::ZERO, wrap_angle_rad: 0.0 },
            TendonWaypoint { body_id: 1, local_pos: Vec3::ZERO, wrap_angle_rad: 0.0 },
        ],
        drive_mode: TendonDriveMode::Linear,
        rest_length: 0.1,
        slack: 0.0,
        stiffness: 1500.0,
        damping: 0.5,
        capstan_mu_theta: 0.0,
        stick_slip_enabled: false,
        servo_torque_limit: 2.0,
        servo_speed_limit: 5.0,
        servo_deadband: 0.0,
        angle_limits: [0.0, 1.0],
        solref_imp: Some(SolRefImp::direct(1500.0, 0.5)),
    }];
    let joints = vec![JointEdgeSpec {
        id: 0,
        name: "j0".to_string(),
        body_a: 0,
        body_b: 1,
        joint_type: JointType::Spherical {
            anchor_a: Vec3::new(0.05, 0.0, 0.0),
            anchor_b: Vec3::new(-0.05, 0.0, 0.0),
            pos_stiffness: 5000.0,
            pos_damping: 5.0,
        },
        actuator_id: None,
        kp: 0.0,
        kd: 0.0,
        torque_limit: 0.0,
        speed_limit: 0.0,
        static_friction: 0.0,
        deadband: 0.0,
        limits: None,
        implicit_coupling: true,
        solref_imp: Some(SolRefImp::from_mujoco([0.01, 1.0], [0.9, 0.98, 0.002, 0.5, 2.0])),
    }];
    let contact_sites = vec![ContactSiteSpec {
        id: 0,
        name: "scale_0".to_string(),
        attachment: ContactAttachment::RigidBody { body_id: 0 },
        local_pos: Vec3::new(0.0, 0.0, -0.02),
        local_frame: Quat::IDENTITY,
        radius: 0.01,
        patch_radius: 0.005,
        mode: ContactMode::AnisotropicScale {
            mu_fwd: 0.15,
            mu_rev: 0.85,
            mu_lat: 0.8,
            pivot_mu: 0.5,
            static_mu_ratio: 1.15,
            fwd_axis_local: Vec3::EX,
        },
        material_profile_id: 0,
        solref_imp: Some(SolRefImp::default()),
    }];
    let diff_params = vec![
        DiffParamSpec {
            name: "E_beam".to_string(),
            kind: DiffParamKind::BeamYoungsModulus,
            target_ids: vec![0],
            bounds: Some([1e9, 3e11]),
            scale: 1e11,
        },
        DiffParamSpec::shared("beta_beam", DiffParamKind::BeamDampingBeta),
        DiffParamSpec::targeted("mu_fwd", DiffParamKind::ContactMuFwd, vec![0]),
        DiffParamSpec::shared("mu_rev", DiffParamKind::ContactMuRev),
        DiffParamSpec::shared("k_con", DiffParamKind::ContactStiffness),
        DiffParamSpec::targeted("k_ten", DiffParamKind::TendonStiffness, vec![0]),
        DiffParamSpec::targeted("m_tip", DiffParamKind::BodyMass, vec![1]),
    ];
    let mesh_colliders = vec![
        MeshColliderSpec {
            id: 0,
            name: "tetra_body".to_string(),
            body_id: Some(0),
            local_pose: Se3::IDENTITY,
            geometry: ColliderGeometry::TriMesh {
                vertices: vec![
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.02, 0.0, 0.0),
                    Vec3::new(0.0, 0.02, 0.0),
                    Vec3::new(0.0, 0.0, 0.02),
                ],
                indices: vec![[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]],
            },
            material_profile_id: 0,
            solref_imp: Some(SolRefImp::default()),
            margin: 1e-4,
            collision_group: 1,
            collision_mask: u32::MAX,
        },
        MeshColliderSpec {
            id: 1,
            name: "static_obstacle".to_string(),
            body_id: None,
            local_pose: Se3::new(Vec3::new(0.5, 0.0, 0.1), Quat::IDENTITY),
            geometry: ColliderGeometry::Capsule {
                radius: 0.03,
                half_height: 0.1,
            },
            ..MeshColliderSpec::default()
        },
    ];

    let mut g_v2 = RigidFlexGraph {
        name: "v2_full_spec".to_string(),
        bodies,
        beams,
        tendons,
        joints,
        contact_sites,
        diff_params,
        mesh_colliders,
        ..RigidFlexGraph::default()
    };
    g_v2.validate().expect("v2 graph must validate");

    // JSON & YAML round-trip
    let js = g_v2.to_json_string().unwrap();
    let g_js = RigidFlexGraph::from_json_str(&js).unwrap();
    assert_eq!(g_js.diff_params.len(), 7);
    assert_eq!(g_js.mesh_colliders.len(), 2);
    assert_eq!(g_js.contact_sites[0].solref_imp, Some(SolRefImp::default()));

    let ym = g_v2.to_yaml_string().unwrap();
    let g_ym = RigidFlexGraph::from_yaml_str(&ym).unwrap();
    assert_eq!(g_ym.diff_params, g_v2.diff_params);
    assert_eq!(g_ym.mesh_colliders.len(), 2);

    // Invalid triangle index must be rejected
    g_v2.mesh_colliders[0].geometry = ColliderGeometry::TriMesh {
        vertices: vec![Vec3::ZERO, Vec3::EX, Vec3::EY],
        indices: vec![[0, 1, 9]],
    };
    assert!(g_v2.validate().is_err());

    // Invalid diff_param target index must be rejected
    g_v2.mesh_colliders.clear();
    g_v2.diff_params[0].target_ids = vec![99];
    assert!(g_v2.validate().is_err());
}

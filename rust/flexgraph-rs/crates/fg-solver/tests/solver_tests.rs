//! Integration & numerical physics verification suite for `fg-solver`.

use fg_contact::{BarrierKind, IpcBarrierConfig, LcpSolverConfig, LcpSolverMethod};
#[cfg(feature = "parry-mesh")]
use fg_ir::{ColliderGeometry, MeshColliderSpec};
use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, JointEdgeSpec,
    JointType, RigidBodySpec, RigidFlexGraph, SchurRoute, TendonDriveMode, TendonEdgeSpec,
    TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, SolRefImp, Vec3, Vec6};
use fg_sdf::{
    CrossSection, FunnelConfig, MaterialLibrary, PlaneSdf, RadiusProfile, SdfField, SplineKind,
    SplineTubeSdf, TubeWallMode, WorldSdf, MAT_ACRYLIC_WALL, MAT_HARD_FLOOR,
};
use fg_solver::{ContactBackend, FlexGraphWorld, IntegratorKind, SolverConfig};
use std::sync::Arc;

fn make_body(id: usize, pos: Vec3, is_fixed: bool, mass: f64, inertia_diag: [f64; 3]) -> RigidBodySpec {
    RigidBodySpec {
        id,
        name: format!("body_{id}"),
        is_fixed,
        mass,
        inertia: Mat3::diag(inertia_diag),
        com_local: Vec3::ZERO,
        rest_pose: Se3::new(pos, Quat::IDENTITY),
        collision_radius: 0.015,
        rayleigh_alpha: 0.05,
        tags: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn make_curved_steel_strip(
    id: usize,
    body_a: usize,
    body_b: usize,
    p_a: Vec3,
    p_b: Vec3,
    n_el: usize,
    bow: f64,
    alpha: f64,
    beta: f64,
) -> BeamEdgeSpec {
    let mut poses = Vec::with_capacity(n_el + 1);
    for i in 0..=n_el {
        let u = i as f64 / n_el as f64;
        let pos = p_a + (p_b - p_a) * u + Vec3::new(0.0, 0.0, 4.0 * bow * u * (1.0 - u));
        let tang = ((p_b - p_a) + Vec3::new(0.0, 0.0, 4.0 * bow * (1.0 - 2.0 * u))).normalized();
        let width_ax = Vec3::EY;
        let norm_ax = tang.cross(width_ax).normalized();
        let q = Quat::from_mat3(Mat3::from_cols(tang, width_ax, norm_ax));
        poses.push(Se3::new(pos, q));
    }
    BeamEdgeSpec {
        id,
        name: format!("strip_{id}"),
        body_a,
        clamp_a: Se3::new(Vec3::ZERO, poses[0].rot),
        body_b: Some(body_b),
        clamp_b: Some(Se3::new(Vec3::ZERO, poses[n_el].rot)),
        n_elements: n_el,
        formulation: BeamFormulation::CorotationalEulerBernoulli,
        section: BeamSection::Rectangular {
            width: 0.016,
            thickness: 0.0002,
        },
        material: BeamMaterial {
            rayleigh_alpha: alpha,
            rayleigh_beta: beta,
            ..BeamMaterial::SPRING_STEEL
        },
        thin_shell_compat: true,
        prestress_frac: 0.0,
        centerline_rest_poses: poses,
        element_lengths: Vec::new(),
        element_transforms: None,
        interior_node_contact_radius: None,
        enable_self_contact: false,
    }
}

#[test]
fn test_free_flight_momentum_and_quat_norm_conservation() {
    // Torque-free anisotropic rigid body tumbling in 3D zero-gravity vacuum + 0 Rayleigh damping
    for integrator in [
        IntegratorKind::DampedNewtonEuler,
        IntegratorKind::NewmarkBeta {
            beta: 0.25,
            gamma: 0.50,
        },
        IntegratorKind::Bdf2,
    ] {
        let mut body = make_body(0, Vec3::new(0.1, -0.2, 0.5), false, 0.45, [0.002, 0.005, 0.009]);
        body.rayleigh_alpha = 0.0;
        let graph = RigidFlexGraph {
            name: "free_tumbler".into(),
            gravity: Vec3::ZERO,
            bodies: vec![body],
            beams: Vec::new(),
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = SolverConfig {
            integrator,
            contact: ContactBackend::None,
            world_fixed_inertia: false,
            use_strain_jacobian: true,
            newton_tol: 1e-11,
            ..SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
        world.state.vels[0] = Vec6::from_parts(Vec3::new(0.8, -0.4, 0.3), Vec3::new(3.0, -5.0, 2.0));
        let budget0 = world.energy_and_momentum();

        let dt = 0.002;
        let mut max_q_err = 0.0_f64;
        for _ in 0..250 {
            let diag = world.step(dt).expect("free flight step ok");
            max_q_err = max_q_err.max(diag.max_quat_norm_err);
            assert!(
                diag.residual_inf < 1e-10,
                "{integrator:?} residual_inf = {:e}",
                diag.residual_inf
            );
        }
        let budget1 = world.energy_and_momentum();
        let lin_err = (budget1.linear_momentum - budget0.linear_momentum).norm();
        let ang_err = (budget1.angular_momentum - budget0.angular_momentum).norm();

        assert!(
            max_q_err < 1e-13,
            "{integrator:?} quat norm err = {max_q_err:e}"
        );
        assert!(
            lin_err < 1e-12,
            "{integrator:?} linear momentum drift = {lin_err:e}"
        );
        // DampedNewtonEuler uses exact world angular-momentum residual (I_w(q_{n+1}) w_{n+1} - L_n)/dt = 0!
        if matches!(integrator, IntegratorKind::DampedNewtonEuler) {
            assert!(
                ang_err < 1e-11,
                "DampedNewtonEuler angular momentum drift = {ang_err:e}"
            );
        } else {
            assert!(
                ang_err < 5e-4,
                "{integrator:?} 3D tumbling angular momentum drift = {ang_err:e}"
            );
        }
    }
}

#[test]
fn test_damped_newton_large_dt_stiff_corotational_steel_beam() {
    // 200 GPa thin spring-steel arched ribbon connecting two plates driven by tendon at large dt = 2 ms
    let p0 = Vec3::new(0.0, 0.0, 0.15);
    let p1 = Vec3::new(0.12, 0.0, 0.15);
    let strip = make_curved_steel_strip(0, 0, 1, p0, p1, 8, 0.035, 0.05, 0.002);
    let tendon = TendonEdgeSpec {
        id: 0,
        name: "t0".into(),
        actuator_id: 0,
        waypoints: vec![
            TendonWaypoint {
                body_id: 0,
                local_pos: Vec3::new(0.0, 0.02, 0.0),
                wrap_angle_rad: 0.0,
            },
            TendonWaypoint {
                body_id: 1,
                local_pos: Vec3::new(0.0, 0.02, 0.0),
                wrap_angle_rad: 0.0,
            },
        ],
        drive_mode: TendonDriveMode::Linear,
        rest_length: 0.12,
        slack: 0.0,
        stiffness: 1500.0,
        damping: 0.5,
        capstan_mu_theta: 0.0,
        stick_slip_enabled: false,
        servo_torque_limit: 5.0,
        servo_speed_limit: 0.2,
        servo_deadband: 0.0,
        angle_limits: [0.0, 0.04],
        solref_imp: None,
    };

    for integrator in [
        IntegratorKind::DampedNewtonEuler,
        IntegratorKind::NewmarkBeta {
            beta: 0.25,
            gamma: 0.50,
        },
        IntegratorKind::Bdf2,
    ] {
        let graph = RigidFlexGraph {
            name: "stiff_steel_pair".into(),
            gravity: Vec3::ZERO,
            bodies: vec![
                make_body(0, p0, false, 0.28, [4.5e-4, 2.5e-4, 2.5e-4]),
                make_body(1, p1, false, 0.28, [4.5e-4, 2.5e-4, 2.5e-4]),
            ],
            beams: vec![strip.clone()],
            tendons: vec![tendon.clone()],
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = SolverConfig {
            integrator,
            contact: ContactBackend::None,
            use_strain_jacobian: true,
            world_fixed_inertia: false,
            newton_max_iters: 30,
            newton_tol: 1e-10,
            ..SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
        world.set_tendon_targets(&[0.012]);

        let dt = 0.002; // 8x larger than baseline GPUCore dt=0.25ms!
        for step in 0..30 {
            let diag = world.step(dt).expect("stiff steel Newton step ok");
            // On E=200 GPa steel (K_axial ~ 4.3e7 N/m), FP64 1-ULP force floor is ~3e-9 N (rel < 1e-14)
            assert!(
                diag.residual_inf < 1e-8,
                "{integrator:?} step {step}: residual_inf = {:e}",
                diag.residual_inf
            );
            assert!(diag.max_quat_norm_err < 1e-13);
        }
        // Plates must pull closer under tendon retraction!
        let sep_final = (world.state.poses[1].pos - world.state.poses[0].pos).norm();
        assert!(
            sep_final < 0.119,
            "{integrator:?}: expected steel arch compression, sep_final={sep_final}"
        );
    }
}

#[test]
fn test_der_cantilever_catheter_with_capstan_tendon_in_3d_spline_tube() {
    // Fixed-base Nitinol DER cantilever rod with 10 elements + interior SDF contact inside a 3D curved SplineTubeSdf
    let n_el = 10;
    let l_tot = 0.25;
    let dl = l_tot / n_el as f64;
    let poses: Vec<Se3> = (0..=n_el)
        .map(|i| Se3::new(Vec3::new(i as f64 * dl, 0.0, 0.0), Quat::IDENTITY))
        .collect();
    let der_beam = BeamEdgeSpec {
        id: 0,
        name: "nitinol_der".into(),
        body_a: 0,
        clamp_a: Se3::IDENTITY,
        body_b: None, // free cantilever tip!
        clamp_b: None,
        n_elements: n_el,
        formulation: BeamFormulation::DiscreteElasticRod,
        section: BeamSection::HollowCircular {
            outer_radius: 0.0025,
            inner_radius: 0.0015,
        },
        material: BeamMaterial::NITINOL,
        thin_shell_compat: false,
        prestress_frac: 0.0,
        centerline_rest_poses: poses,
        element_lengths: Vec::new(),
        element_transforms: None,
        interior_node_contact_radius: Some(0.0025),
        enable_self_contact: false,
    };

    let graph = RigidFlexGraph {
        name: "der_endoscope".into(),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies: vec![make_body(0, Vec3::ZERO, true, 1.0, [1e-2, 1e-2, 1e-2])],
        beams: vec![der_beam],
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: Vec::new(),
        symmetry: Default::default(),
        solver_hints: Default::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };

    // 3D curved SplineTubeSdf surrounding the catheter
    let ctrl = vec![
        Vec3::new(-0.05, 0.0, 0.0),
        Vec3::new(0.10, 0.0, 0.0),
        Vec3::new(0.25, 0.03, -0.01),
        Vec3::new(0.40, 0.08, -0.02),
    ];
    let tube = SplineTubeSdf::new(
        ctrl,
        SplineKind::CatmullRom { tension: 0.5 },
        RadiusProfile::Constant(0.015),
        CrossSection::Circular,
        TubeWallMode::InteriorCavity,
        FunnelConfig::NONE,
        MAT_ACRYLIC_WALL,
        64,
    );
    let mut sdf = WorldSdf::new(MaterialLibrary::standard());
    sdf.add_unbounded(Arc::new(tube));

    let cfg = SolverConfig {
        integrator: IntegratorKind::LinearlyImplicitEuler,
        contact: ContactBackend::Arpc { implicit: true },
        ..SolverConfig::default()
    };
    let mut world = FlexGraphWorld::new(graph, sdf, cfg).unwrap();
    for _ in 0..80 {
        let diag = world.step(0.001).expect("DER catheter step ok");
        assert!(!diag.failed);
        assert!(diag.max_quat_norm_err < 1e-13);
    }
    // Tip must sag slightly under gravity while remaining strictly inside the 15 mm lumen!
    let tip_id = *world.analysis.indexing.beam_maps[0]
        .free_node_ids
        .last()
        .unwrap();
    let tip_pos = world.state.poses[tip_id].pos;
    assert!(tip_pos.z() < 0.0, "expected gravity sag, tip z={}", tip_pos.z());
    assert!(
        tip_pos.z() > -0.025,
        "catheter tip escaped tube lumen: z={}",
        tip_pos.z()
    );
}

#[test]
fn test_cyclic_tensegrity_6bar_sparse_general_implicit_stepping() {
    // Classic 6-strut cyclic icosahedral tensegrity graph (6 free bodies, 12 cyclic implicit cables, beta_1 = 7)
    let r = 0.08;
    let centers = [
        Vec3::new(r, 0.0, 0.16),
        Vec3::new(-r, 0.0, 0.16),
        Vec3::new(0.0, r, 0.16),
        Vec3::new(0.0, -r, 0.16),
        Vec3::new(0.0, 0.0, 0.16 + r),
        Vec3::new(0.0, 0.0, 0.16 - r),
    ];
    let bodies: Vec<_> = centers
        .iter()
        .enumerate()
        .map(|(i, &p)| make_body(i, p, false, 0.12, [1.5e-4, 1.5e-4, 1.5e-4]))
        .collect();
    let edges = [
        (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 4), (2, 5), (3, 4), (3, 5),
    ];
    let joints: Vec<_> = edges
        .iter()
        .enumerate()
        .map(|(id, &(a, b))| {
            let d = (centers[b] - centers[a]).norm();
            JointEdgeSpec {
                id,
                name: format!("cable_{id}"),
                body_a: a,
                body_b: b,
                joint_type: JointType::CableSpring {
                    anchor_a: Vec3::ZERO,
                    anchor_b: Vec3::ZERO,
                    rest_length: d * 0.96, // slight pretension
                    stiffness: 600.0,
                    damping: 2.0,
                    compression_allowed: true,
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
            }
        })
        .collect();
    let sites: Vec<_> = (0..6)
        .map(|bi| ContactSiteSpec {
            id: bi,
            name: format!("pad_{bi}"),
            attachment: ContactAttachment::RigidBody { body_id: bi },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: 0.02,
            patch_radius: 0.008,
            mode: ContactMode::IsotropicPad {
                mu: 0.7,
                pivot_mu: 0.3,
                static_mu_ratio: 1.1,
                sdf_only: false,
            },
            material_profile_id: MAT_HARD_FLOOR,
            solref_imp: None,
        })
        .collect();

    let graph = RigidFlexGraph {
        name: "tensegrity_6bar".into(),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams: Vec::new(),
        tendons: Vec::new(),
        joints,
        contact_sites: sites,
        symmetry: Default::default(),
        solver_hints: Default::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };
    let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), SolverConfig::default()).unwrap();
    assert_eq!(world.analysis.route, SchurRoute::SparseGeneral);
    assert_eq!(world.analysis.cyclomatic_number, 7);

    for _ in 0..100 {
        let diag = world.step(0.001).expect("tensegrity sparse LDL^T step ok");
        assert!(!diag.failed);
        assert!(diag.max_quat_norm_err < 1e-13);
    }
    // Bottom node settled onto ground without falling through z = 0
    assert!(world.state.poses[5].pos.z() > 0.005);
}

#[test]
fn test_delassus_lcp_stick_and_slip_on_inclined_plane() {
    // Sloped plane tilted by 20 degrees (tan 20 deg ~ 0.364)
    let theta = 20.0_f64.to_radians();
    let normal = Vec3::new(theta.sin(), 0.0, theta.cos()).normalized();
    let mut sdf = WorldSdf::new(MaterialLibrary::standard());
    sdf.add_unbounded(Arc::new(PlaneSdf::new(Vec3::ZERO, normal, MAT_HARD_FLOOR)));

    let run_slope = |mu: f64, method: LcpSolverMethod| -> f64 {
        let down = Vec3::new(theta.cos(), 0.0, -theta.sin());
        // 2-foot flat block aligned with the 20-deg slope (prevents rolling, isolates pure stick vs slide!)
        let mut body = make_body(0, normal * 0.0195, false, 0.5, [2e-3, 2e-3, 2e-3]);
        body.rest_pose.rot = Quat::from_axis_angle(Vec3::EY, theta);
        let pos0 = body.rest_pose.pos;
        let graph = RigidFlexGraph {
            name: "slope_block".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![body],
            beams: Vec::new(),
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: vec![
                ContactSiteSpec {
                    id: 0,
                    name: "foot_front".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 0 },
                    local_pos: Vec3::new(0.04, 0.0, 0.0),
                    local_frame: Quat::IDENTITY,
                    radius: 0.02,
                    patch_radius: 0.0,
                    mode: ContactMode::IsotropicPad {
                        mu,
                        pivot_mu: 0.0,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: None,
                },
                ContactSiteSpec {
                    id: 1,
                    name: "foot_rear".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 0 },
                    local_pos: Vec3::new(-0.04, 0.0, 0.0),
                    local_frame: Quat::IDENTITY,
                    radius: 0.02,
                    patch_radius: 0.0,
                    mode: ContactMode::IsotropicPad {
                        mu,
                        pivot_mu: 0.0,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: None,
                },
            ],
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = SolverConfig {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::DelassusLcp {
                lcp: LcpSolverConfig {
                    method,
                    max_iters: 300,
                    tol: 1e-10,
                    compliance_reg: 1e-6,
                    ..LcpSolverConfig::default()
                },
                dedup_radius: 0.002,
                baumgarte_beta: 0.2,
            },
            ..SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, WorldSdf::new(MaterialLibrary::standard()), cfg).unwrap();
        world.sdf.add_unbounded(Arc::new(PlaneSdf::new(Vec3::ZERO, normal, MAT_HARD_FLOOR)));

        for step in 0..80 {
            let diag = world.step(0.002).expect("Delassus LCP step ok");
            assert!(
                diag.lcp_residual < 1e-7,
                "mu={mu} method={method:?} step={step} iters={} LCP residual = {:e}",
                diag.lcp_iters,
                diag.lcp_residual
            );
        }
        (world.state.poses[0].pos - pos0).dot(down)
    };

    let disp_stick = run_slope(0.80, LcpSolverMethod::AlartCurnierNewton);
    let disp_slide = run_slope(0.05, LcpSolverMethod::BlockPgs);
    assert!(
        disp_stick.abs() < 1e-3,
        "high-friction block should stick on 20-deg slope: disp={disp_stick}"
    );
    assert!(
        disp_slide > 0.015,
        "low-friction block should slide down 20-deg slope: disp={disp_slide}"
    );
}

#[test]
fn test_ipc_sdf_barrier_and_accd_non_tunneling_impact() {
    // Fast downward projectile body approaching z = 0 under IPC log-barrier + conservative ACCD
    let graph = RigidFlexGraph {
        name: "ipc_drop".into(),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies: vec![make_body(0, Vec3::new(0.0, 0.0, 0.04), false, 0.2, [1e-4, 1e-4, 1e-4])],
        beams: Vec::new(),
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: vec![ContactSiteSpec {
            id: 0,
            name: "ball".into(),
            attachment: ContactAttachment::RigidBody { body_id: 0 },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: 0.01,
            patch_radius: 0.0,
            mode: ContactMode::IsotropicPad {
                mu: 0.3,
                pivot_mu: 0.0,
                static_mu_ratio: 1.0,
                sdf_only: true,
            },
            material_profile_id: MAT_HARD_FLOOR,
            solref_imp: None,
        }],
        symmetry: Default::default(),
        solver_hints: Default::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };
    let cfg = SolverConfig {
        integrator: IntegratorKind::DampedNewtonEuler,
        contact: ContactBackend::SdfBarrier {
            barrier: IpcBarrierConfig {
                kind: BarrierKind::IpcLogBarrier,
                stiffness: 5.0e4,
                d_hat: 0.01,
                thickness: 0.01,
                mollifier_eps_x: 1e-5,
            },
            mu: 0.3,
            friction_eps_v: 1e-2,
            use_ccd: true,
        },
        enable_ccd_clamping: true,
        newton_max_iters: 25,
        newton_tol: 1e-8,
        ..SolverConfig::default()
    };
    let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
    // Shoot downward fast enough to tunnel through z = 0 in a single 5 ms step without CCD!
    world.state.vels[0] = Vec6::from_parts(Vec3::new(0.0, 0.0, -15.0), Vec3::ZERO);

    for _ in 0..15 {
        let _diag = world.step(0.005).expect("IPC CCD step ok");
        let phi = world.sdf.query(world.state.poses[0].pos).phi;
        assert!(
            phi > 0.01,
            "IPC + ACCD violated zero-interpenetration invariant: phi={phi} <= r=0.01"
        );
    }
}

#[test]
fn test_rayon_batch_stepping_matches_sequential_bit_for_bit() {
    let p0 = Vec3::new(0.0, 0.0, 0.10);
    let p1 = Vec3::new(0.12, 0.0, 0.10);
    let make_world = |target: f64| {
        let strip = make_curved_steel_strip(0, 0, 1, p0, p1, 6, 0.03, 0.05, 0.002);
        let graph = RigidFlexGraph {
            name: "batch_env".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![
                make_body(0, p0, false, 0.28, [4e-4, 2.5e-4, 2.5e-4]),
                make_body(1, p1, false, 0.28, [4e-4, 2.5e-4, 2.5e-4]),
            ],
            beams: vec![strip],
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let mut w = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), SolverConfig::default()).unwrap();
        w.state.vels[0] = Vec6::from_parts(Vec3::new(target, 0.0, 0.0), Vec3::ZERO);
        w
    };

    let targets = [0.05, 0.10, -0.08, 0.12, 0.0, -0.04, 0.07, 0.02];
    let mut seq_worlds: Vec<_> = targets.iter().map(|&t| make_world(t)).collect();
    let mut par_worlds: Vec<_> = targets.iter().map(|&t| make_world(t)).collect();

    for w in &mut seq_worlds {
        w.step_n(20, 0.0005).unwrap();
    }
    FlexGraphWorld::step_batch_n_rayon(&mut par_worlds, 20, 0.0005).unwrap();

    for (sw, pw) in seq_worlds.iter().zip(par_worlds.iter()) {
        assert_eq!(sw.poses_flat(), pw.poses_flat());
        assert_eq!(sw.vels_flat(), pw.vels_flat());
    }
}

#[test]
fn test_der_nonlinear_damped_newton_and_newmark_convergence() {
    // Fixed-base Nitinol DER cantilever rod under gravity solved with DampedNewtonEuler & NewmarkBeta
    for integrator in [
        IntegratorKind::DampedNewtonEuler,
        IntegratorKind::NewmarkBeta {
            beta: 0.25,
            gamma: 0.50,
        },
        IntegratorKind::Bdf2,
    ] {
        let n_el = 6;
        let dl = 0.03;
        let poses: Vec<Se3> = (0..=n_el)
            .map(|i| Se3::new(Vec3::new(i as f64 * dl, 0.0, 0.05), Quat::IDENTITY))
            .collect();
        let der_beam = BeamEdgeSpec {
            id: 0,
            name: "nitinol_newton".into(),
            body_a: 0,
            clamp_a: Se3::IDENTITY,
            body_b: None,
            clamp_b: None,
            n_elements: n_el,
            formulation: BeamFormulation::DiscreteElasticRod,
            section: BeamSection::Circular { radius: 0.0008 },
            material: BeamMaterial {
                rayleigh_alpha: 0.05,
                rayleigh_beta: 1e-4,
                ..BeamMaterial::NITINOL
            },
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: poses,
            element_lengths: Vec::new(),
            element_transforms: None,
            interior_node_contact_radius: None,
            enable_self_contact: false,
        };
        let graph = RigidFlexGraph {
            name: "der_newton_cantilever".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![make_body(0, Vec3::new(0.0, 0.0, 0.05), true, 1.0, [1e-2; 3])],
            beams: vec![der_beam],
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = SolverConfig {
            integrator,
            contact: ContactBackend::None,
            newton_max_iters: 30,
            newton_tol: 1e-10,
            project_der_hessian_spd: true,
            ..SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
        let tip_id = *world.analysis.indexing.beam_maps[0].free_node_ids.last().unwrap();
        let mut min_tip_z = 0.05_f64;
        for step in 0..25 {
            let diag = world.step(0.002).expect("DER nonlinear Newton step ok");
            assert!(
                diag.residual_inf < 1e-8,
                "{integrator:?} step {step}: DER residual_inf = {:e}",
                diag.residual_inf
            );
            assert!(diag.max_quat_norm_err < 1e-13);
            min_tip_z = min_tip_z.min(world.state.poses[tip_id].pos.z());
        }
        assert!(
            min_tip_z < 0.049,
            "{integrator:?}: expected DER guidewire tip gravity sag below z=0.049, got min_tip_z={min_tip_z}"
        );
    }
}

#[test]
fn test_serpworm_two_segment_kelvin_and_arpc_parity_locomotion() {
    // Minimal 2-segment (3-plate, 2 arched spring-steel ribbons, 2 yaw-coupled tendons, 2 ratchet wheels)
    // exercising SolverConfig::serpworm_kelvin_parity() and serpworm_arpc_parity() + set_all_targets()
    let build_graph = || {
        let p0 = Vec3::new(0.0, 0.0, 0.0155);
        let p1 = Vec3::new(0.12, 0.0, 0.0155);
        let p2 = Vec3::new(0.24, 0.0, 0.0155);
        let b0 = make_body(0, p0, false, 0.28, [4.5e-4, 2.5e-4, 2.5e-4]);
        let b1 = make_body(1, p1, false, 0.28, [4.5e-4, 2.5e-4, 2.5e-4]);
        let b2 = make_body(2, p2, false, 0.28, [4.5e-4, 2.5e-4, 2.5e-4]);
        let s0 = make_curved_steel_strip(0, 0, 1, p0, p1, 6, 0.035, 0.05, 0.002);
        let s1 = make_curved_steel_strip(1, 1, 2, p1, p2, 6, 0.035, 0.05, 0.002);
        let make_tendon = |id: usize, ba: usize, bb: usize| TendonEdgeSpec {
            id,
            name: format!("tendon_{id}"),
            actuator_id: id,
            waypoints: vec![
                TendonWaypoint {
                    body_id: ba,
                    local_pos: Vec3::ZERO,
                    wrap_angle_rad: 0.0,
                },
                TendonWaypoint {
                    body_id: bb,
                    local_pos: Vec3::ZERO,
                    wrap_angle_rad: 0.0,
                },
            ],
            drive_mode: TendonDriveMode::EccentricCrank {
                horn_radius: 0.018,
                guide_dist: 0.045,
            },
            rest_length: 0.12,
            slack: 0.0,
            stiffness: 1800.0,
            damping: 0.5,
            capstan_mu_theta: 0.12,
            stick_slip_enabled: false,
            servo_torque_limit: 1.8,
            servo_speed_limit: 4.5,
            servo_deadband: 0.005,
            angle_limits: [0.0, 2.5],
            solref_imp: None,
        };
        let make_wheel = |id: usize, body_id: usize| ContactSiteSpec {
            id,
            name: format!("wheel_{id}"),
            attachment: ContactAttachment::RigidBody { body_id },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: 0.016,
            patch_radius: 0.006,
            mode: ContactMode::RatchetWheel {
                wheel_mass: 0.025,
                rolling_resistance: 0.015,
                mu_fwd: 0.65,
                mu_lat_ratio: 1.0,
                pivot_mu: 0.2,
                static_mu_ratio: 1.1,
                backlash_rad: 0.0,
                axle_axis_local: Vec3::EY,
            },
            material_profile_id: MAT_HARD_FLOOR,
            solref_imp: None,
        };
        RigidFlexGraph {
            name: "serpworm_2seg_smoke".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![b0, b1, b2],
            beams: vec![s0, s1],
            tendons: vec![make_tendon(0, 0, 1), make_tendon(1, 1, 2)],
            joints: vec![JointEdgeSpec {
                id: 0,
                name: "yaw_0".into(),
                body_a: 0,
                body_b: 1,
                joint_type: JointType::Revolute {
                    axis_local_a: Vec3::EZ,
                    anchor_a: Vec3::new(0.12, 0.0, 0.0),
                    anchor_b: Vec3::ZERO,
                    pos_stiffness: 0.0,
                    pos_damping: 0.0,
                    align_stiffness: 0.0,
                    align_damping: 0.0,
                },
                actuator_id: Some(2),
                kp: 1.2,
                kd: 0.05,
                torque_limit: 0.8,
                speed_limit: 5.0,
                static_friction: 0.01,
                deadband: 0.0,
                limits: Some([-0.6, 0.6]),
                implicit_coupling: false,
                solref_imp: None,
            }],
            contact_sites: vec![make_wheel(0, 0), make_wheel(1, 1), make_wheel(2, 2)],
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        }
    };

    for cfg in [
        SolverConfig::serpworm_kelvin_parity(),
        SolverConfig::serpworm_arpc_parity(),
    ] {
        let mut world = FlexGraphWorld::new(build_graph(), WorldSdf::flat_ground(), cfg).unwrap();
        assert_eq!(world.analysis.route, SchurRoute::TreeChain);
        assert_eq!(world.num_tendon_actuators(), 2);
        assert_eq!(world.num_joint_actuators(), 1);
        assert_eq!(world.num_total_actuators(), 3);

        world.set_all_targets(&[1.2, 0.4, 0.15]);
        for _ in 0..80 {
            let diag = world.step(0.0005).expect("SerpWorm parity step ok");
            assert!(!diag.failed);
            assert!(diag.max_quat_norm_err < 1e-13);
        }
        let budget = world.energy_and_momentum();
        assert!(budget.total_energy.is_finite());
        assert!(world.state.contact_normal_forces.iter().any(|&f| f > 0.1));
    }
}

#[test]
fn test_revolute_non_z_axis_quaternion_double_cover_invariance() {
    // Regression: Revolute joint with non-Z axis must treat q and -q identically (canonical double cover)
    let joint = JointEdgeSpec {
        id: 0,
        name: "pitch_hinge".into(),
        body_a: 0,
        body_b: 1,
        joint_type: JointType::Revolute {
            axis_local_a: Vec3::EY,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            pos_stiffness: 100.0,
            pos_damping: 1.0,
            align_stiffness: 50.0,
            align_damping: 0.5,
        },
        actuator_id: Some(0),
        kp: 10.0,
        kd: 0.2,
        torque_limit: 100.0,
        speed_limit: 0.0,
        static_friction: 0.0,
        deadband: 0.0,
        limits: None,
        implicit_coupling: false,
        solref_imp: None,
    };
    let rest = vec![Se3::IDENTITY, Se3::IDENTITY];
    let vels = vec![Vec6::ZERO, Vec6::ZERO];
    let poses_pos = vec![Se3::IDENTITY, Se3::new(Vec3::ZERO, Quat::IDENTITY)];
    let poses_neg = vec![
        Se3::IDENTITY,
        Se3::new(Vec3::ZERO, Quat::new(0.0, 0.0, 0.0, -1.0)),
    ];

    let mut act_pos = fg_solver::ActuationState::from_specs(&[], std::slice::from_ref(&joint), &rest);
    let mut act_neg = fg_solver::ActuationState::from_specs(&[], std::slice::from_ref(&joint), &rest);
    let ev_pos = fg_solver::evaluate_actuation(
        2,
        &[],
        std::slice::from_ref(&joint),
        &rest,
        &poses_pos,
        &vels,
        &mut act_pos,
        0.0,
        0.0,
        false,
        true,
        false,
    );
    let ev_neg = fg_solver::evaluate_actuation(
        2,
        &[],
        std::slice::from_ref(&joint),
        &rest,
        &poses_neg,
        &vels,
        &mut act_neg,
        0.0,
        0.0,
        false,
        true,
        false,
    );
    assert!(
        act_neg.last_joint_torque[0].abs() < 1e-13,
        "spurious double-cover torque = {}",
        act_neg.last_joint_torque[0]
    );
    assert!((ev_pos.body_wrenches[1] - ev_neg.body_wrenches[1]).max_abs() < 1e-13);
}

#[test]
fn test_damped_newton_and_newmark_with_delassus_lcp_floor_support() {
    // Regression: sphere (r=0.02) dropped onto z=0 under DampedNewtonEuler & NewmarkBeta + DelassusLcp
    for integrator in [
        IntegratorKind::DampedNewtonEuler,
        IntegratorKind::NewmarkBeta {
            beta: 0.25,
            gamma: 0.50,
        },
    ] {
        let graph = RigidFlexGraph {
            name: "lcp_newton_drop".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![make_body(0, Vec3::new(0.0, 0.0, 0.025), false, 0.4, [1e-3; 3])],
            beams: Vec::new(),
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: vec![ContactSiteSpec {
                id: 0,
                name: "sphere".into(),
                attachment: ContactAttachment::RigidBody { body_id: 0 },
                local_pos: Vec3::ZERO,
                local_frame: Quat::IDENTITY,
                radius: 0.02,
                patch_radius: 0.0,
                mode: ContactMode::IsotropicPad {
                    mu: 0.5,
                    pivot_mu: 0.0,
                    static_mu_ratio: 1.0,
                    sdf_only: false,
                },
                material_profile_id: MAT_HARD_FLOOR,
                solref_imp: None,
            }],
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = SolverConfig {
            integrator,
            contact: ContactBackend::DelassusLcp {
                lcp: LcpSolverConfig {
                    method: LcpSolverMethod::AlartCurnierNewton,
                    max_iters: 100,
                    tol: 1e-10,
                    compliance_reg: 1e-6,
                    ..LcpSolverConfig::default()
                },
                dedup_radius: 0.002,
                baumgarte_beta: 0.25,
            },
            ..SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
        let mut saw_contact = false;
        for step in 0..50 {
            let diag = world.step(0.002).expect("Newton + DelassusLcp step ok");
            assert!(
                diag.lcp_residual < 1e-7,
                "{integrator:?} step {step}: lcp_residual={:e}",
                diag.lcp_residual
            );
            if diag.active_site_contacts > 0 {
                saw_contact = true;
            }
        }
        assert!(saw_contact, "{integrator:?}: expected active LCP floor contact");
        let z_final = world.state.poses[0].pos.z();
        assert!(
            z_final > 0.018 && z_final < 0.022,
            "{integrator:?}: sphere fell through or bounced away from z=0.02, z_final={z_final}"
        );
        assert!(world.state.contact_normal_forces[0] > 1.0);
    }
}

#[test]
fn test_accd_per_point_thickness_heterogeneous_radii() {
    // Regression: wheel (r=0.025, z=0.035) + beam interior node (r=0.002, z=0.015) must NOT freeze ccd_toi=0
    let p0 = Vec3::new(0.0, 0.0, 0.035);
    let p1 = Vec3::new(0.10, 0.0, 0.035);
    let mut poses = Vec::new();
    for i in 0..=4 {
        let u = i as f64 / 4.0;
        // Dip interior nodes down to z = 0.015 (safely above r_beam=0.002, but below r_wheel=0.025!)
        let z = if i == 0 || i == 4 { 0.035 } else { 0.015 };
        poses.push(Se3::new(Vec3::new(u * 0.10, 0.0, z), Quat::IDENTITY));
    }
    let mut strip = make_curved_steel_strip(0, 0, 1, p0, p1, 4, 0.0, 0.05, 0.001);
    strip.centerline_rest_poses = poses;
    strip.interior_node_contact_radius = Some(0.002);

    let graph = RigidFlexGraph {
        name: "heterogeneous_accd".into(),
        gravity: Vec3::ZERO,
        bodies: vec![
            make_body(0, p0, false, 0.3, [1e-3; 3]),
            make_body(1, p1, false, 0.3, [1e-3; 3]),
        ],
        beams: vec![strip],
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: vec![ContactSiteSpec {
            id: 0,
            name: "big_wheel".into(),
            attachment: ContactAttachment::RigidBody { body_id: 0 },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: 0.025,
            patch_radius: 0.0,
            mode: ContactMode::IsotropicPad {
                mu: 0.3,
                pivot_mu: 0.0,
                static_mu_ratio: 1.0,
                sdf_only: true,
            },
            material_profile_id: MAT_HARD_FLOOR,
            solref_imp: None,
        }],
        symmetry: Default::default(),
        solver_hints: Default::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };
    let cfg = SolverConfig {
        integrator: IntegratorKind::DampedNewtonEuler,
        contact: ContactBackend::SdfBarrier {
            barrier: IpcBarrierConfig {
                kind: BarrierKind::IpcLogBarrier,
                stiffness: 1.0e3,
                d_hat: 0.005,
                thickness: 0.001,
                mollifier_eps_x: 1e-5,
            },
            mu: 0.1,
            friction_eps_v: 1e-2,
            use_ccd: true,
        },
        enable_ccd_clamping: true,
        ..SolverConfig::default()
    };
    let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
    for v in &mut world.state.vels {
        *v = Vec6::from_parts(Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO);
    }
    let diag = world.step(0.002).expect("heterogeneous ACCD step ok");
    assert!(
        diag.ccd_toi > 0.5,
        "beam node radius was wrongly pooled with wheel radius: ccd_toi={}",
        diag.ccd_toi
    );
    assert!(
        world.state.poses[0].pos.x() > 5e-4,
        "robot froze horizontally due to false ACCD clamp: x={}",
        world.state.poses[0].pos.x()
    );
}

#[test]
fn test_der_free_flight_linear_momentum_conservation() {
    // Regression: 2-body clamped-clamped DER rod in zero-g vacuum must conserve exact linear momentum
    // (including DER end_a_mass and end_b_mass in energy_and_momentum!)
    let n_el = 4;
    let dl = 0.03;
    let poses: Vec<Se3> = (0..=n_el)
        .map(|i| Se3::new(Vec3::new(i as f64 * dl, 0.0, 0.1), Quat::IDENTITY))
        .collect();
    let der_beam = BeamEdgeSpec {
        id: 0,
        name: "der_free_flight".into(),
        body_a: 0,
        clamp_a: Se3::IDENTITY,
        body_b: Some(1),
        clamp_b: Some(Se3::IDENTITY),
        n_elements: n_el,
        formulation: BeamFormulation::DiscreteElasticRod,
        section: BeamSection::Circular { radius: 0.003 },
        material: BeamMaterial {
            rayleigh_alpha: 0.0,
            rayleigh_beta: 0.0,
            ..BeamMaterial::NITINOL
        },
        thin_shell_compat: false,
        prestress_frac: 0.0,
        centerline_rest_poses: poses,
        element_lengths: Vec::new(),
        element_transforms: None,
        interior_node_contact_radius: None,
        enable_self_contact: false,
    };
    let mut b0 = make_body(0, Vec3::new(0.0, 0.0, 0.1), false, 0.15, [5e-4; 3]);
    let mut b1 = make_body(1, Vec3::new(n_el as f64 * dl, 0.0, 0.1), false, 0.15, [5e-4; 3]);
    b0.rayleigh_alpha = 0.0;
    b1.rayleigh_alpha = 0.0;

    let graph = RigidFlexGraph {
        name: "der_2body_momentum".into(),
        gravity: Vec3::ZERO,
        bodies: vec![b0, b1],
        beams: vec![der_beam],
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: Vec::new(),
        symmetry: Default::default(),
        solver_hints: Default::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };
    let cfg = SolverConfig {
        integrator: IntegratorKind::LinearlyImplicitEuler,
        contact: ContactBackend::None,
        world_fixed_inertia: false,
        ..SolverConfig::default()
    };
    let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
    // Give asymmetric initial impulse so the rod translates, rotates, and flexes internally!
    world.state.vels[0] = Vec6::from_parts(Vec3::new(0.4, 0.25, -0.15), Vec3::new(0.0, 0.5, 0.3));
    world.state.vels[1] = Vec6::from_parts(Vec3::new(0.1, -0.2, 0.35), Vec3::new(0.0, -0.4, 0.2));
    let p0 = world.energy_and_momentum().linear_momentum;

    for _ in 0..30 {
        world.step(0.001).expect("DER free-flight step ok");
    }
    let p1 = world.energy_and_momentum().linear_momentum;
    let drift = (p1 - p0).norm();
    assert!(
        drift < 1e-10,
        "DER free-flight linear momentum drift = {drift:e} (p0={p0:?}, p1={p1:?})"
    );
}

#[test]
fn test_clarabel_socp_lcp_inclined_plane_and_stiff_mass_ratio() {
    // 1. 20-degree inclined plane (2-foot flat block aligned with slope to isolate pure stick vs slip):
    //    ClarabelSocp must match AlartCurnierNewton for both stick & slip
    let theta = 20.0_f64.to_radians();
    let normal = Vec3::new(theta.sin(), 0.0, theta.cos()).normalized();
    let slope_down = Vec3::new(theta.cos(), 0.0, -theta.sin()).normalized();

    let run_incline = |mu: f64, method: LcpSolverMethod| -> (Vec3, f64, f64) {
        let mut sdf = WorldSdf::new(MaterialLibrary::standard());
        sdf.add_unbounded(Arc::new(PlaneSdf::new(Vec3::ZERO, normal, MAT_HARD_FLOOR)));
        let r = 0.02;
        let mut body = make_body(0, normal * (r - 5e-4), false, 0.5, [2e-3; 3]);
        body.rest_pose.rot = Quat::from_axis_angle(Vec3::EY, theta);
        let graph = RigidFlexGraph {
            name: "incline_socp".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![body],
            beams: Vec::new(),
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: vec![
                ContactSiteSpec {
                    id: 0,
                    name: "foot_front".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 0 },
                    local_pos: Vec3::new(0.04, 0.0, 0.0),
                    local_frame: Quat::IDENTITY,
                    radius: r,
                    patch_radius: 0.0,
                    mode: ContactMode::IsotropicPad {
                        mu,
                        pivot_mu: 0.0,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: None,
                },
                ContactSiteSpec {
                    id: 1,
                    name: "foot_rear".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 0 },
                    local_pos: Vec3::new(-0.04, 0.0, 0.0),
                    local_frame: Quat::IDENTITY,
                    radius: r,
                    patch_radius: 0.0,
                    mode: ContactMode::IsotropicPad {
                        mu,
                        pivot_mu: 0.0,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: None,
                },
            ],
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = SolverConfig {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::DelassusLcp {
                lcp: LcpSolverConfig {
                    method,
                    max_iters: 120,
                    tol: 1e-10,
                    compliance_reg: 1e-6,
                    ..LcpSolverConfig::default()
                },
                dedup_radius: 0.002,
                baumgarte_beta: 0.2,
            },
            ..SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, sdf, cfg).unwrap();
        let mut max_res = 0.0_f64;
        for _ in 0..40 {
            let d = world.step(0.002).expect("ClarabelSocp incline step ok");
            max_res = max_res.max(d.lcp_residual);
        }
        let v_down = world.state.vels[0].lin().dot(slope_down);
        (world.state.poses[0].pos, v_down, max_res)
    };

    // Stick regime (mu = 0.50 > tan 20 deg ≈ 0.364)
    let (p_socp_stick, v_socp_stick, r_stick) =
        run_incline(0.50, LcpSolverMethod::ClarabelSocp);
    let (p_ac_stick, _v_ac_stick, _) =
        run_incline(0.50, LcpSolverMethod::AlartCurnierNewton);
    assert!(r_stick < 1e-6, "ClarabelSocp stick residual = {r_stick:e}");
    assert!(
        v_socp_stick.abs() < 1e-3,
        "ClarabelSocp failed static stick: v_down = {v_socp_stick}"
    );
    assert!(
        (p_socp_stick - p_ac_stick).norm() < 1e-5,
        "ClarabelSocp vs AlartCurnierNewton stick diff = {:e}",
        (p_socp_stick - p_ac_stick).norm()
    );

    // Slip regime (mu = 0.15 < tan 20 deg)
    let (p_socp_slip, v_socp_slip, r_slip) = run_incline(0.15, LcpSolverMethod::ClarabelSocp);
    let (p_ac_slip, v_ac_slip, _) = run_incline(0.15, LcpSolverMethod::AlartCurnierNewton);
    assert!(r_slip < 1e-6, "ClarabelSocp slip residual = {r_slip:e}");
    assert!(
        v_socp_slip > 0.05,
        "ClarabelSocp expected down-slope slip, got v_down = {v_socp_slip}"
    );
    assert!(
        (v_socp_slip - v_ac_slip).abs() < 1e-4 && (p_socp_slip - p_ac_slip).norm() < 1e-4,
        "ClarabelSocp vs AlartCurnierNewton slip mismatch: v=({v_socp_slip}, {v_ac_slip})"
    );

    // 2. High-mass-ratio (1e4:1) stiff coupled 2-body contact system under ClarabelSocp
    let p0 = Vec3::new(0.0, 0.0, 0.0195);
    let p1 = Vec3::new(0.10, 0.0, 0.0195);
    let strip = make_curved_steel_strip(0, 0, 1, p0, p1, 4, 0.015, 0.05, 0.002);
    let graph_heavy = RigidFlexGraph {
        name: "socp_stiff_mass_ratio".into(),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies: vec![
            make_body(0, p0, false, 100.0, [1.0; 3]),
            make_body(1, p1, false, 0.01, [1e-4; 3]),
        ],
        beams: vec![strip],
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: vec![
            ContactSiteSpec {
                id: 0,
                name: "heavy_pad".into(),
                attachment: ContactAttachment::RigidBody { body_id: 0 },
                local_pos: Vec3::ZERO,
                local_frame: Quat::IDENTITY,
                radius: 0.02,
                patch_radius: 0.0,
                mode: ContactMode::IsotropicPad {
                    mu: 0.4,
                    pivot_mu: 0.0,
                    static_mu_ratio: 1.0,
                    sdf_only: false,
                },
                material_profile_id: MAT_HARD_FLOOR,
                solref_imp: None,
            },
            ContactSiteSpec {
                id: 1,
                name: "light_pad".into(),
                attachment: ContactAttachment::RigidBody { body_id: 1 },
                local_pos: Vec3::ZERO,
                local_frame: Quat::IDENTITY,
                radius: 0.02,
                patch_radius: 0.0,
                mode: ContactMode::IsotropicPad {
                    mu: 0.4,
                    pivot_mu: 0.0,
                    static_mu_ratio: 1.0,
                    sdf_only: false,
                },
                material_profile_id: MAT_HARD_FLOOR,
                solref_imp: None,
            },
        ],
        symmetry: Default::default(),
        solver_hints: Default::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };
    let cfg_heavy = SolverConfig {
        integrator: IntegratorKind::LinearlyImplicitEuler,
        contact: ContactBackend::DelassusLcp {
            lcp: LcpSolverConfig {
                method: LcpSolverMethod::ClarabelSocp,
                max_iters: 120,
                tol: 1e-9,
                compliance_reg: 1e-7,
                ..LcpSolverConfig::default()
            },
            dedup_radius: 0.002,
            baumgarte_beta: 0.2,
        },
        ..SolverConfig::default()
    };
    let mut world_heavy = FlexGraphWorld::new(graph_heavy, WorldSdf::flat_ground(), cfg_heavy).unwrap();
    for step in 0..25 {
        let diag = world_heavy.step(0.001).expect("1e4:1 mass ratio SOCP step ok");
        assert!(
            diag.lcp_residual < 1e-6,
            "1e4:1 mass ratio step {step}: lcp_residual={:e}",
            diag.lcp_residual
        );
    }
    assert!(world_heavy.state.poses[0].pos.z() > 0.018);
    assert!(world_heavy.state.poses[1].pos.z() > 0.015);
}

#[cfg(feature = "parry-mesh")]
#[test]
fn test_trimesh_sdf_and_tight_inclusion_ccd_non_tunneling() {
    // 1. Static TriMesh floor slab auto-mounted via `graph.mesh_colliders` (`body_id: None`)
    // Closed unit slab spanning x,y in [-0.5, 0.5], z in [-0.1, 0.0] (outward CCW triangles)
    let verts = vec![
        Vec3::new(-0.5, -0.5, -0.1), // 0
        Vec3::new(0.5, -0.5, -0.1),  // 1
        Vec3::new(0.5, 0.5, -0.1),   // 2
        Vec3::new(-0.5, 0.5, -0.1),  // 3
        Vec3::new(-0.5, -0.5, 0.0),  // 4
        Vec3::new(0.5, -0.5, 0.0),   // 5
        Vec3::new(0.5, 0.5, 0.0),    // 6
        Vec3::new(-0.5, 0.5, 0.0),   // 7
    ];
    let tris = vec![
        [4, 5, 6], [4, 6, 7], // top (+Z)
        [0, 2, 1], [0, 3, 2], // bottom (-Z)
        [0, 1, 5], [0, 5, 4], // front (-Y)
        [2, 3, 7], [2, 7, 6], // back (+Y)
        [1, 2, 6], [1, 6, 5], // right (+X)
        [3, 0, 4], [3, 4, 7], // left (-X)
    ];

    let graph = RigidFlexGraph {
        name: "trimesh_ti_ccd".into(),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies: vec![make_body(0, Vec3::new(0.0, 0.0, 0.03), false, 0.15, [1e-4; 3])],
        beams: Vec::new(),
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: vec![ContactSiteSpec {
            id: 0,
            name: "ball".into(),
            attachment: ContactAttachment::RigidBody { body_id: 0 },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: 0.01,
            patch_radius: 0.0,
            mode: ContactMode::IsotropicPad {
                mu: 0.2,
                pivot_mu: 0.0,
                static_mu_ratio: 1.0,
                sdf_only: true,
            },
            material_profile_id: MAT_HARD_FLOOR,
            solref_imp: None,
        }],
        symmetry: Default::default(),
        solver_hints: Default::default(),
        diff_params: Vec::new(),
        mesh_colliders: vec![MeshColliderSpec {
            id: 0,
            name: "floor_slab".into(),
            body_id: None,
            local_pose: Se3::IDENTITY,
            geometry: ColliderGeometry::TriMesh {
                vertices: verts,
                indices: tris,
            },
            material_profile_id: MAT_HARD_FLOOR,
            solref_imp: None,
            margin: 0.0,
            collision_group: u32::MAX,
            collision_mask: u32::MAX,
        }],
    };
    let cfg = SolverConfig {
        integrator: IntegratorKind::DampedNewtonEuler,
        contact: ContactBackend::SdfBarrier {
            barrier: IpcBarrierConfig {
                kind: BarrierKind::IpcLogBarrier,
                stiffness: 1.0e4,
                d_hat: 0.008,
                thickness: 0.01,
                mollifier_eps_x: 1e-5,
            },
            mu: 0.1,
            friction_eps_v: 1e-2,
            use_ccd: true,
        },
        enable_ccd_clamping: true,
        use_tight_inclusion_ccd: true,
        newton_max_iters: 30,
        ..SolverConfig::default()
    };
    let mut world =
        FlexGraphWorld::new(graph, WorldSdf::new(MaterialLibrary::standard()), cfg).unwrap();
    // High-speed downward impact (-25 m/s * 0.005 s = -0.125 m >> 0.02 m clearance!)
    world.state.vels[0] = Vec6::from_parts(Vec3::new(0.0, 0.0, -25.0), Vec3::ZERO);
    let mut min_toi = 1.0_f64;
    for _ in 0..8 {
        let diag = world.step(0.005).expect("TriMesh + TI-CCD step ok");
        min_toi = min_toi.min(diag.ccd_toi);
        let z = world.state.poses[0].pos.z();
        let phi = world.sdf.query(world.state.poses[0].pos).phi;
        assert!(
            z > 0.01 && phi > 0.01,
            "Tight Inclusion CCD tunneled through TriMesh slab: z={z}, phi={phi}"
        );
    }
    assert!(min_toi < 1.0, "expected Tight Inclusion CCD clamping (min_toi={min_toi})");

    // 2. Corrugated pipe cavity registered via `register_static_trimesh` + Tight Inclusion CCD
    let pipe = fg_sdf::corrugated_pipe_mesh(
        0.3,
        0.045,
        0.004,
        4.0,
        12,
        12,
        Vec3::ZERO,
        MAT_ACRYLIC_WALL,
    )
    .expect("corrugated pipe mesh ok");
    let graph_pipe = RigidFlexGraph {
        name: "pipe_ti_ccd".into(),
        gravity: Vec3::ZERO,
        bodies: vec![make_body(0, Vec3::new(-0.15, 0.0, 0.0), false, 0.1, [1e-4; 3])],
        beams: Vec::new(),
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: vec![ContactSiteSpec {
            id: 0,
            name: "capsule_probe".into(),
            attachment: ContactAttachment::RigidBody { body_id: 0 },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: 0.012,
            patch_radius: 0.0,
            mode: ContactMode::IsotropicPad {
                mu: 0.2,
                pivot_mu: 0.0,
                static_mu_ratio: 1.0,
                sdf_only: true,
            },
            material_profile_id: MAT_ACRYLIC_WALL,
            solref_imp: None,
        }],
        symmetry: Default::default(),
        solver_hints: Default::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };
    let mut world_pipe = FlexGraphWorld::new(
        graph_pipe,
        WorldSdf::new(MaterialLibrary::standard()),
        SolverConfig {
            integrator: IntegratorKind::DampedNewtonEuler,
            contact: ContactBackend::SdfBarrier {
                barrier: IpcBarrierConfig {
                    kind: BarrierKind::IpcLogBarrier,
                    stiffness: 5.0e3,
                    d_hat: 0.006,
                    thickness: 0.012,
                    mollifier_eps_x: 1e-5,
                },
                mu: 0.1,
                friction_eps_v: 1e-2,
                use_ccd: true,
            },
            enable_ccd_clamping: true,
            use_tight_inclusion_ccd: true,
            newton_max_iters: 25,
            ..SolverConfig::default()
        },
    )
    .unwrap();
    world_pipe.register_static_trimesh(Arc::new(pipe));
    // Launch radially toward pipe wall at +12 m/s (0.036 m/step > 0.029 m wall clearance!)
    world_pipe.state.vels[0] = Vec6::from_parts(Vec3::new(0.0, 12.0, 0.0), Vec3::ZERO);
    for _ in 0..4 {
        world_pipe.step(0.003).expect("Corrugated pipe TI-CCD step ok");
        let phi = world_pipe.sdf.query(world_pipe.state.poses[0].pos).phi;
        assert!(
            phi > 0.012,
            "Particle escaped corrugated pipe wall: phi={phi} <= r=0.012"
        );
    }
}

#[test]
fn test_mujoco_solref_imp_arpc_and_delassus_lcp() {
    let solref_imp = SolRefImp::from_mujoco([0.02, 1.0], [0.9, 0.95, 0.002, 0.5, 2.0]);
    let backends = [
        ContactBackend::Arpc { implicit: true },
        ContactBackend::DelassusLcp {
            lcp: LcpSolverConfig {
                method: LcpSolverMethod::ClarabelSocp,
                max_iters: 120,
                tol: 1e-9,
                compliance_reg: 1e-7,
                ..LcpSolverConfig::default()
            },
            dedup_radius: 0.002,
            baumgarte_beta: 0.2,
        },
    ];

    for contact in backends {
        let graph = RigidFlexGraph {
            name: "mujoco_solref_drop".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![make_body(0, Vec3::new(0.0, 0.0, 0.023), false, 0.5, [1e-3; 3])],
            beams: Vec::new(),
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: vec![ContactSiteSpec {
                id: 0,
                name: "soft_sphere".into(),
                attachment: ContactAttachment::RigidBody { body_id: 0 },
                local_pos: Vec3::ZERO,
                local_frame: Quat::IDENTITY,
                radius: 0.02,
                patch_radius: 0.004,
                mode: ContactMode::IsotropicPad {
                    mu: 0.5,
                    pivot_mu: 0.05,
                    static_mu_ratio: 1.0,
                    sdf_only: false,
                },
                material_profile_id: MAT_HARD_FLOOR,
                solref_imp: Some(solref_imp),
            }],
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = SolverConfig {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: contact.clone(),
            ..SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
        for _ in 0..80 {
            let diag = world.step(0.002).expect("SolRefImp drop step ok");
            assert!(diag.lcp_residual < 1e-6, "lcp_residual={:e}", diag.lcp_residual);
            assert!(world.state.poses[0].pos.z().is_finite());
        }
        let z = world.state.poses[0].pos.z();
        let vz = world.state.vels[0].lin().z();
        let fn_est = world.state.contact_normal_forces[0];
        assert!(
            z > 0.015 && z < 0.022 && vz.abs() < 0.05,
            "SolRefImp settling failed for {contact:?}: z={z}, vz={vz}, fn={fn_est}"
        );
        assert!(
            fn_est > 2.5 && fn_est < 8.0,
            "SolRefImp expected ~mg=4.905N support for {contact:?}, got fn={fn_est}"
        );
    }

    // Part 2: Exact MuJoCo SolRefImp dynamic damping law via `.with_pre_vel_n(vn)` in FlexGraphWorld!
    // Unit-mass sphere (m=1 kg, g=0, r=0.02) at z=0.018 (gap = -0.002 m), initial vn^- = -1.0 m/s, dt=0.001:
    //   SolRefImp::direct(1000, 0)   -> v_n+ = -0.998 m/s
    //   SolRefImp::direct(1000, 500) -> v_n+ = -0.498 m/s
    for (b_val, expected_vz) in [(0.0_f64, -0.998_f64), (500.0_f64, -0.498_f64)] {
        let sri_direct = SolRefImp::direct(1000.0, b_val);
        for method in [
            LcpSolverMethod::BlockPgs,
            LcpSolverMethod::AlartCurnierNewton,
            LcpSolverMethod::ClarabelSocp,
        ] {
            let mut body = make_body(0, Vec3::new(0.0, 0.0, 0.018), false, 1.0, [1.0; 3]);
            body.rayleigh_alpha = 0.0;
            let graph = RigidFlexGraph {
                name: "solref_dyn_damping".into(),
                gravity: Vec3::ZERO,
                bodies: vec![body],
                beams: Vec::new(),
                tendons: Vec::new(),
                joints: Vec::new(),
                contact_sites: vec![ContactSiteSpec {
                    id: 0,
                    name: "probe".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 0 },
                    local_pos: Vec3::ZERO,
                    local_frame: Quat::IDENTITY,
                    radius: 0.02,
                    patch_radius: 0.0,
                    mode: ContactMode::IsotropicPad {
                        mu: 0.5,
                        pivot_mu: 0.0,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: Some(sri_direct),
                }],
                symmetry: Default::default(),
                solver_hints: Default::default(),
                diff_params: Vec::new(),
                mesh_colliders: Vec::new(),
            };
            let cfg = SolverConfig {
                integrator: IntegratorKind::LinearlyImplicitEuler,
                contact: ContactBackend::DelassusLcp {
                    lcp: LcpSolverConfig {
                        method,
                        max_iters: 120,
                        tol: 1e-11,
                        compliance_reg: 0.0,
                        ..LcpSolverConfig::default()
                    },
                    dedup_radius: 0.0,
                    baumgarte_beta: 0.0,
                },
                ..SolverConfig::default()
            };
            let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
            world.state.vels[0] = Vec6::from_parts(Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO);
            let diag = world.step(0.001).expect("SolRefImp dynamic damping step ok");
            assert!(diag.lcp_residual < 1e-8, "{method:?} res={:e}", diag.lcp_residual);
            let vz_post = world.state.vels[0].lin().z();
            assert!(
                (vz_post - expected_vz).abs() < 1e-5,
                "SolRefImp::direct(1000, {b_val}) via {method:?}: expected vz={expected_vz}, got {vz_post}"
            );
        }
    }
}

#[cfg(feature = "parry-mesh")]
#[test]
fn test_corrugated_pipe_mesh_crawler_locomotion() {
    // 2-segment anisotropic-scale flexible continuum worm locomoting along -X inside a 3D
    // corrugated triangle-mesh pipe cavity (`fg_sdf::corrugated_pipe_mesh`, Parry SIMD QBvh +
    // angle-weighted pseudonormals) under both ARPC locomotion and SdfBarrier + Tight Inclusion CCD.
    let pipe_origin = Vec3::new(0.15, 0.0, 0.0);
    let pipe = Arc::new(
        fg_sdf::corrugated_pipe_mesh(
            0.65,
            0.055,
            0.003,
            8.0,
            24,
            16,
            pipe_origin,
            MAT_ACRYLIC_WALL,
        )
        .expect("corrugated_pipe_mesh ok"),
    );

    let build_crawler = |z_init: f64| {
        let p0 = Vec3::new(0.02, 0.0, z_init);
        let p1 = Vec3::new(-0.08, 0.0, z_init);
        let p2 = Vec3::new(-0.18, 0.0, z_init);
        let mut s0 = make_curved_steel_strip(0, 0, 1, p0, p1, 4, 0.015, 0.05, 0.002);
        let mut s1 = make_curved_steel_strip(1, 1, 2, p1, p2, 4, 0.015, 0.05, 0.002);
        s0.interior_node_contact_radius = Some(0.004);
        s1.interior_node_contact_radius = Some(0.004);
        let make_ten = |id: usize, ba: usize, bb: usize| TendonEdgeSpec {
            id,
            name: format!("ten_{id}"),
            actuator_id: id,
            waypoints: vec![
                TendonWaypoint {
                    body_id: ba,
                    local_pos: Vec3::ZERO,
                    wrap_angle_rad: 0.0,
                },
                TendonWaypoint {
                    body_id: bb,
                    local_pos: Vec3::ZERO,
                    wrap_angle_rad: 0.0,
                },
            ],
            drive_mode: TendonDriveMode::Linear,
            rest_length: 0.10,
            slack: 0.0,
            stiffness: 1800.0,
            damping: 0.8,
            capstan_mu_theta: 0.0,
            stick_slip_enabled: false,
            servo_torque_limit: 20.0,
            servo_speed_limit: 0.5,
            servo_deadband: 0.0,
            angle_limits: [0.0, 0.025],
            solref_imp: None,
        };
        let make_scale = |id: usize, body_id: usize| ContactSiteSpec {
            id,
            name: format!("scale_{id}"),
            attachment: ContactAttachment::RigidBody { body_id },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: 0.012,
            patch_radius: 0.004,
            mode: ContactMode::AnisotropicScale {
                mu_fwd: 0.12,
                mu_rev: 0.90,
                mu_lat: 0.50,
                pivot_mu: 0.05,
                static_mu_ratio: 1.0,
                fwd_axis_local: Vec3::new(-1.0, 0.0, 0.0),
            },
            material_profile_id: MAT_ACRYLIC_WALL,
            solref_imp: None,
        };
        RigidFlexGraph {
            name: "corrugated_pipe_crawler".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![
                make_body(0, p0, false, 0.12, [1.5e-4; 3]),
                make_body(1, p1, false, 0.12, [1.5e-4; 3]),
                make_body(2, p2, false, 0.12, [1.5e-4; 3]),
            ],
            beams: vec![s0, s1],
            tendons: vec![make_ten(0, 0, 1), make_ten(1, 1, 2)],
            joints: Vec::new(),
            contact_sites: vec![make_scale(0, 0), make_scale(1, 1), make_scale(2, 2)],
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        }
    };

    // 1. Peristaltic crawling inside corrugated pipe mesh via ARPC
    let mut world = FlexGraphWorld::new(
        build_crawler(-0.036),
        WorldSdf::new(MaterialLibrary::standard()),
        SolverConfig {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::Arpc { implicit: true },
            use_strain_jacobian: true,
            ..SolverConfig::default()
        },
    )
    .unwrap();
    world.register_static_trimesh(Arc::clone(&pipe));

    let com_x0 = (world.state.poses[0].pos.x()
        + world.state.poses[1].pos.x()
        + world.state.poses[2].pos.x())
        / 3.0;
    let s0 = world.sdf.query(world.state.poses[2].pos).arc_s;

    let dt = 0.002;
    let mut saw_pipe_contact = false;
    for step in 0..160 {
        let t = step as f64 * dt;
        // Traveling peristaltic wave propagating from front (-X) to rear (+X) drives locomotion along -X
        let u_front = 0.011 * (1.0 + (2.0 * std::f64::consts::PI * 5.0 * t).sin());
        let u_rear = 0.011 * (1.0 + (2.0 * std::f64::consts::PI * 5.0 * t - 1.5).sin());
        world.set_tendon_targets(&[u_rear, u_front]);
        let diag = world.step(dt).expect("corrugated pipe locomotion step ok");
        assert!(!diag.failed);
        assert!(diag.max_quat_norm_err < 1e-13);
        if diag.active_site_contacts > 0 {
            saw_pipe_contact = true;
        }
        for bi in 0..3 {
            let q = world.sdf.query(world.state.poses[bi].pos);
            assert!(
                q.phi > 0.0,
                "step {step} body {bi} escaped corrugated pipe cavity: phi={}",
                q.phi
            );
        }
    }
    assert!(saw_pipe_contact, "crawler never contacted corrugated pipe wall");
    let com_x1 = (world.state.poses[0].pos.x()
        + world.state.poses[1].pos.x()
        + world.state.poses[2].pos.x())
        / 3.0;
    let s1 = world.sdf.query(world.state.poses[2].pos).arc_s;
    assert!(
        com_x1 < com_x0 - 0.003 && s1 > s0 + 0.002,
        "expected axial locomotion along -X in corrugated pipe: com_x {com_x0} -> {com_x1}, arc_s {s0} -> {s1}"
    );

    // 2. SdfBarrier + Tight Inclusion Point-Triangle CCD inside the same corrugated pipe mesh
    let mut world_ipc = FlexGraphWorld::new(
        build_crawler(-0.028),
        WorldSdf::new(MaterialLibrary::standard()),
        SolverConfig {
            integrator: IntegratorKind::DampedNewtonEuler,
            contact: ContactBackend::SdfBarrier {
                barrier: IpcBarrierConfig {
                    kind: BarrierKind::IpcLogBarrier,
                    stiffness: 2.0e3,
                    d_hat: 0.006,
                    thickness: 0.012,
                    mollifier_eps_x: 1e-5,
                },
                mu: 0.2,
                friction_eps_v: 1e-2,
                use_ccd: true,
            },
            enable_ccd_clamping: true,
            use_tight_inclusion_ccd: true,
            use_strain_jacobian: true,
            newton_max_iters: 20,
            ..SolverConfig::default()
        },
    )
    .unwrap();
    world_ipc.register_static_trimesh(pipe);
    world_ipc.set_tendon_targets(&[0.012, 0.006]);
    for step in 0..10 {
        let diag = world_ipc
            .step(0.002)
            .expect("corrugated pipe SdfBarrier+TI-CCD step ok");
        assert!(!diag.failed);
        for bi in 0..3 {
            let phi = world_ipc.sdf.query(world_ipc.state.poses[bi].pos).phi;
            assert!(
                phi > 0.012,
                "step {step} body {bi} violated SdfBarrier+TI-CCD pipe clearance: phi={phi}"
            );
        }
    }
}

#[test]
fn test_tight_inclusion_ccd_beam_self_contact_non_tunneling() {
    // Two perpendicular flexible beams approaching each other at high speed along Z:
    // Tight Inclusion Edge-Edge CCD (`use_tight_inclusion_ccd: true`) must clamp `ccd_toi < 1.0`
    // and prevent the beam centerlines from passing through each other!
    let p_a0 = Vec3::new(-0.05, 0.0, 0.045);
    let p_a1 = Vec3::new(0.05, 0.0, 0.045);
    let p_b0 = Vec3::new(0.0, -0.05, 0.025);
    let p_b1 = Vec3::new(0.0, 0.05, 0.025);

    let mut beam_a = make_curved_steel_strip(0, 0, 1, p_a0, p_a1, 2, 0.0, 0.05, 0.002);
    beam_a.enable_self_contact = true;
    // Rotate beam B 90 deg around Z so it runs along Y at z = 0.025
    let mut beam_b = make_curved_steel_strip(1, 2, 3, p_b0, p_b1, 2, 0.0, 0.05, 0.002);
    let q_y = Quat::from_axis_angle(Vec3::EZ, 0.5 * std::f64::consts::PI);
    beam_b.clamp_a = Se3::new(Vec3::ZERO, q_y);
    beam_b.clamp_b = Some(Se3::new(Vec3::ZERO, q_y));
    beam_b.centerline_rest_poses = (0..=2)
        .map(|i| {
            let u = i as f64 / 2.0;
            Se3::new(p_b0 + (p_b1 - p_b0) * u, q_y)
        })
        .collect();
    beam_b.enable_self_contact = true;

    let graph = RigidFlexGraph {
        name: "ti_ccd_beam_self_contact".into(),
        gravity: Vec3::ZERO,
        bodies: vec![
            make_body(0, p_a0, false, 0.2, [1e-3; 3]),
            make_body(1, p_a1, false, 0.2, [1e-3; 3]),
            make_body(2, p_b0, false, 0.2, [1e-3; 3]),
            make_body(3, p_b1, false, 0.2, [1e-3; 3]),
        ],
        beams: vec![beam_a, beam_b],
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: Vec::new(),
        symmetry: Default::default(),
        solver_hints: Default::default(),
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };
    let thickness = 0.004;
    let cfg = SolverConfig {
        integrator: IntegratorKind::DampedNewtonEuler,
        contact: ContactBackend::None,
        use_strain_jacobian: true,
        enable_self_contact: true,
        self_contact_barrier: IpcBarrierConfig {
            kind: BarrierKind::IpcLogBarrier,
            stiffness: 2.0e3,
            d_hat: 0.008,
            thickness,
            mollifier_eps_x: 1e-5,
        },
        self_contact_mu: 0.1,
        enable_ccd_clamping: true,
        use_tight_inclusion_ccd: true,
        newton_max_iters: 20,
        ..SolverConfig::default()
    };
    let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();

    // High closing speed: top beam moves -5 m/s in Z, bottom moves +5 m/s in Z
    // (0.05 m closing per 5 ms step >> 0.016 m clearance!)
    let mid_a = world.analysis.indexing.beam_maps[0].free_node_ids[0];
    let mid_b = world.analysis.indexing.beam_maps[1].free_node_ids[0];
    for &nid in &[0usize, 1, mid_a] {
        world.state.vels[nid] = Vec6::from_parts(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO);
    }
    for &nid in &[2usize, 3, mid_b] {
        world.state.vels[nid] = Vec6::from_parts(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
    }

    let mut min_toi = 1.0_f64;
    for step in 0..5 {
        let diag = world.step(0.005).expect("TI-CCD beam self-contact step ok");
        assert!(!diag.failed);
        min_toi = min_toi.min(diag.ccd_toi);
        let z_a = world.state.poses[mid_a].pos.z();
        let z_b = world.state.poses[mid_b].pos.z();
        assert!(
            z_a - z_b > thickness,
            "step {step}: beams tunneled through each other! z_a={z_a}, z_b={z_b}, dz={}",
            z_a - z_b
        );
    }
    assert!(
        min_toi < 1.0,
        "expected Tight Inclusion Edge-Edge CCD clamping (min_toi={min_toi})"
    );
}

#[test]
fn test_tendon_solref_imp_positive_mujoco_timeconst_dampratio_dynamics() {
    // Regression for Bug 4 (#134): Tendon with stiffness=1000, damping=5 and positive MuJoCo
    // SolRefImp([tc, dr], ...) must produce distinct trajectories governed by timeconst & dampratio!
    let run_tendon_rollout = |solref_imp: Option<SolRefImp>| -> f64 {
        let mut b0 = make_body(0, Vec3::ZERO, true, 1.0, [1e-3; 3]);
        let mut b1 = make_body(1, Vec3::new(0.12, 0.0, 0.0), false, 0.5, [1e-3; 3]);
        b0.rayleigh_alpha = 0.0;
        b1.rayleigh_alpha = 0.0;
        let tendon = TendonEdgeSpec {
            id: 0,
            name: "sri_ten".into(),
            actuator_id: 0,
            waypoints: vec![
                TendonWaypoint {
                    body_id: 0,
                    local_pos: Vec3::ZERO,
                    wrap_angle_rad: 0.0,
                },
                TendonWaypoint {
                    body_id: 1,
                    local_pos: Vec3::ZERO,
                    wrap_angle_rad: 0.0,
                },
            ],
            drive_mode: TendonDriveMode::Linear,
            rest_length: 0.10, // initial extension = 0.02 m
            slack: 0.0,
            stiffness: 1000.0,
            damping: 5.0,
            capstan_mu_theta: 0.0,
            stick_slip_enabled: false,
            servo_torque_limit: 50.0,
            servo_speed_limit: 5.0,
            servo_deadband: 0.0,
            angle_limits: [0.0, 0.05],
            solref_imp,
        };
        let graph = RigidFlexGraph {
            name: "tendon_solref_dyn".into(),
            gravity: Vec3::ZERO,
            bodies: vec![b0, b1],
            beams: Vec::new(),
            tendons: vec![tendon],
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = SolverConfig {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::None,
            ..SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
        world.step(0.001).expect("tendon solref step ok");
        world.state.vels[1].lin().x()
    };

    let vx_none = run_tendon_rollout(None);
    let vx_stiff = run_tendon_rollout(Some(SolRefImp::from_mujoco(
        [0.005, 0.5],
        [0.9, 0.95, 0.002, 0.5, 2.0],
    )));
    let vx_soft = run_tendon_rollout(Some(SolRefImp::from_mujoco(
        [0.5, 5.0],
        [0.9, 0.95, 0.002, 0.5, 2.0],
    )));

    // solref_imp == None must preserve exact base_k=1000: a = -(1000*0.02)/0.5 = -40 => vx = -0.04
    assert!(
        (vx_none - (-0.04)).abs() < 1e-12,
        "solref_imp=None violated exact v1 parity: vx_none={vx_none}"
    );
    // Positive MuJoCo mode: [0.005, 0.5] must pull dramatically faster than [0.5, 5.0]!
    assert!(
        vx_stiff < -1.0 && vx_soft > -0.005 && vx_soft < 0.0,
        "positive SolRefImp did not govern tendon dynamics: vx_stiff={vx_stiff}, vx_soft={vx_soft}"
    );
}

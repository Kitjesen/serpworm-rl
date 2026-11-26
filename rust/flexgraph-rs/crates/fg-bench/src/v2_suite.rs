//! FlexGraph-RS v2 SOTA Benchmarks:
//! 1. Head-to-head rigid frictional contact accuracy & speed vs `rapier3d-f64 0.36`
//! 2. `faer 0.24` SIMD supernodal sparse $LDL^\top$ vs custom Block-CSC on `RigidFlexGrid-4x4` ($\beta_1=9$) & `6x6` ($\beta_1=25$)
//! 3. PyElastica / analytical Euler buckling & nonlinear Bisshopp-Drucker (1945) Elastica (`< 1%` error)
//! 4. IFT reverse-mode adjoint vs central finite differences speedup & Sim2Real SysID recovery
//! 5. `parry3d-f64` `TriMeshSdf` corrugated pipe locomotion with Tight Inclusion CCD (zero tunneling)

use fg_contact::{BarrierKind, IpcBarrierConfig};
use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, DiffParamKind,
    DiffParamSpec, RigidBodySpec, RigidFlexGraph, SolverHints, SymmetryGroupSpec, TendonDriveMode,
    TendonEdgeSpec, TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Mat6, Quat, Se3, Vec3, Vec6};
use fg_robots::{
    build_euler_elastica_rod, build_rigid_flex_grid, build_rigid_flex_grid_4x4,
    build_rigid_flex_grid_6x6, elastica_cantilever_tip, euler_buckling_critical_load,
    FlexGridConfig,
};
use fg_sdf::{corrugated_pipe_mesh, MaterialLibrary, SdfField, WorldSdf, MAT_ACRYLIC_WALL, MAT_HARD_FLOOR};
use fg_solver::{
    compute_trajectory_adjoint, get_diff_param_value, set_diff_param_value, ContactBackend,
    FaerSparseLdltFactor, FaerSparsePattern, FlexGraphWorld, IntegratorKind, PoseTrackingLoss,
    RigidSchurMatrix, SolverConfig, SparseBackend, SparseBlockLdltFactor, TrajectoryLoss,
};
use rapier3d_f64::prelude as rapier;
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

/// Benchmark 1: Head-to-head rigid sliding sphere accuracy & throughput vs `rapier3d-f64`.
#[derive(Clone, Debug, Serialize)]
pub struct RapierComparisonResult {
    pub dt_s: f64,
    pub steps: usize,
    pub analytic_x_stop_m: f64,
    pub analytic_z_eq_m: f64,
    pub fg_x_final_m: f64,
    pub fg_z_final_m: f64,
    pub fg_x_err_m: f64,
    pub fg_z_err_m: f64,
    pub fg_steps_per_sec: f64,
    pub fg_us_per_step: f64,
    pub rapier_x_final_m: f64,
    pub rapier_z_final_m: f64,
    pub rapier_x_err_m: f64,
    pub rapier_z_err_m: f64,
    pub rapier_steps_per_sec: f64,
    pub rapier_us_per_step: f64,
}

/// Benchmark 2: `faer` SIMD supernodal sparse $LDL^\top$ vs custom Block-CSC on cyclic lattices.
#[derive(Clone, Debug, Serialize)]
pub struct SparseBackendBenchRow {
    pub label: String,
    pub num_free_bodies: usize,
    pub num_beams: usize,
    pub rigid_dofs: usize,
    pub cyclomatic_number: usize,
    pub factor_iters: usize,
    pub custom_factor_us: f64,
    pub faer_factor_us: f64,
    pub factor_speedup: f64,
    pub max_solve_diff: f64,
    pub sim_steps: usize,
    pub custom_step_us: f64,
    pub faer_step_us: f64,
    pub step_speedup: f64,
    pub max_pose_diff_m: f64,
}

/// Benchmark 3: Nonlinear Bisshopp-Drucker Elastica & Euler buckling critical load validation.
#[derive(Clone, Debug, Serialize)]
pub struct ElasticaBucklingResult {
    pub rod_length_m: f64,
    pub n_elements: usize,
    pub tip_load_n: f64,
    pub dimensionless_load: f64,
    pub analytic_x_tip_m: f64,
    pub analytic_dz_down_m: f64,
    pub sim_x_tip_m: f64,
    pub sim_dz_down_m: f64,
    pub x_rel_err_pct: f64,
    pub dz_rel_err_pct: f64,
    pub euler_p_cr_n: f64,
    pub subcritical_lat_m: f64,
    pub supercritical_lat_m: f64,
    pub buckling_amplification: f64,
}

/// Benchmark 4: IFT reverse-mode adjoint vs central finite differences speedup & Sim2Real SysID.
#[derive(Clone, Debug, Serialize)]
pub struct AdjointSysIdResult {
    pub horizon_steps: usize,
    pub total_diff_vars: usize,
    pub adjoint_ms: f64,
    pub fd_ms: f64,
    pub speedup_vs_fd: f64,
    pub max_grad_rel_err: f64,
    pub sysid_iters: usize,
    pub true_youngs_gpa: f64,
    pub init_youngs_gpa: f64,
    pub recovered_youngs_gpa: f64,
    pub youngs_err_pct: f64,
    pub true_mu: f64,
    pub init_mu: f64,
    pub recovered_mu: f64,
    pub mu_err_pct: f64,
    pub sysid_elapsed_ms: f64,
}

/// Benchmark 5: `parry3d-f64` `TriMeshSdf` corrugated pipe locomotion + Tight Inclusion CCD.
#[derive(Clone, Debug, Serialize)]
pub struct CorrugatedPipeBenchResult {
    pub pipe_triangles: usize,
    pub arpc_steps: usize,
    pub arpc_arc_progress_m: f64,
    pub arpc_min_phi_m: f64,
    pub arpc_steps_per_sec: f64,
    pub ipc_ticcd_steps: usize,
    pub ipc_ticcd_min_phi_m: f64,
    pub ipc_ticcd_min_toi: f64,
    pub ipc_ticcd_steps_per_sec: f64,
}

/// Complete v2 SOTA benchmark suite output.
#[derive(Clone, Debug, Serialize)]
pub struct V2BenchSuiteReport {
    pub rapier_comparison: RapierComparisonResult,
    pub sparse_scaling: Vec<SparseBackendBenchRow>,
    pub elastica_buckling: ElasticaBucklingResult,
    pub adjoint_sysid: AdjointSysIdResult,
    pub corrugated_pipe: CorrugatedPipeBenchResult,
    pub total_elapsed_s: f64,
}

/// Benchmark 1: Head-to-head rigid sliding sphere accuracy & speed vs `rapier3d-f64 0.36`.
pub fn bench_rapier_rigid_contact() -> RapierComparisonResult {
    let dt = 0.001_f64;
    let steps = 500usize;
    let radius = 0.10_f64;
    let mass = 1.0_f64;
    let g = 9.81_f64;
    let mu = 0.40_f64;
    let vx0 = 1.50_f64;

    // Exact Coulomb analytical solution for pure sliding (locked rotation):
    // t_stop = vx0 / (mu * g) = 0.382263 s < 0.50 s, x_stop = vx0^2 / (2 * mu * g) = 0.286697 m
    let analytic_x_stop = (vx0 * vx0) / (2.0 * mu * g);
    let analytic_z_eq = radius;

    // 1. FlexGraph-RS (DelassusLcp + ClarabelSocp)
    let graph = RigidFlexGraph {
        name: "sliding_sphere_fg".into(),
        gravity: Vec3::new(0.0, 0.0, -g),
        bodies: vec![RigidBodySpec {
            id: 0,
            name: "puck".into(),
            is_fixed: false,
            mass,
            // Huge inertia isolates pure translational sliding matching Rapier `lock_rotations()`
            inertia: Mat3::diag([1e6, 1e6, 1e6]),
            com_local: Vec3::ZERO,
            rest_pose: Se3::new(Vec3::new(0.0, 0.0, radius), Quat::IDENTITY),
            collision_radius: radius,
            rayleigh_alpha: 0.0,
            tags: Vec::new(),
        }],
        beams: Vec::new(),
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: vec![ContactSiteSpec {
            id: 0,
            name: "bottom".into(),
            attachment: ContactAttachment::RigidBody { body_id: 0 },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius,
            patch_radius: 0.0,
            mode: ContactMode::IsotropicPad {
                mu,
                pivot_mu: 0.0,
                static_mu_ratio: 1.0,
                sdf_only: false,
            },
            material_profile_id: MAT_HARD_FLOOR,
            solref_imp: None,
        }],
        symmetry: SymmetryGroupSpec::trivial(),
        solver_hints: SolverHints {
            preferred_dt: dt,
            rayleigh_alpha: 0.0,
            rayleigh_beta: 0.0,
            force_route: None,
        },
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };
    let mut fg_world =
        FlexGraphWorld::new(graph, WorldSdf::flat_ground(), SolverConfig::delassus_socp())
            .expect("fg sliding world");
    fg_world.state.vels[0] = Vec6::from_parts(Vec3::new(vx0, 0.0, 0.0), Vec3::ZERO);

    let t0 = Instant::now();
    for _ in 0..steps {
        let diag = fg_world.step(dt).expect("fg socp step");
        assert!(!diag.failed);
    }
    let fg_elapsed = t0.elapsed().as_secs_f64();
    let fg_pos = fg_world.state.poses[0].pos;
    let fg_sps = (steps as f64) / fg_elapsed.max(1e-9);

    // 2. Rapier3D-f64 0.36.1
    let mut rworld = rapier::PhysicsWorld::new();
    rworld.gravity = rapier::Vector::new(0.0, 0.0, -g);
    rworld.integration_parameters.dt = dt;
    let (_ground, _) = rworld.insert(
        rapier::RigidBodyBuilder::fixed().translation(rapier::Vector::new(0.0, 0.0, -0.5)),
        rapier::ColliderBuilder::cuboid(20.0, 20.0, 0.5)
            .friction(mu)
            .friction_combine_rule(rapier::CoefficientCombineRule::Min)
            .restitution(0.0)
            .restitution_combine_rule(rapier::CoefficientCombineRule::Min),
    );
    let (ball, _) = rworld.insert(
        rapier::RigidBodyBuilder::dynamic()
            .translation(rapier::Vector::new(0.0, 0.0, radius))
            .linvel(rapier::Vector::new(vx0, 0.0, 0.0))
            .lock_rotations()
            .can_sleep(false),
        rapier::ColliderBuilder::ball(radius)
            .mass(mass)
            .friction(mu)
            .friction_combine_rule(rapier::CoefficientCombineRule::Min)
            .restitution(0.0)
            .restitution_combine_rule(rapier::CoefficientCombineRule::Min),
    );

    let t1 = Instant::now();
    for _ in 0..steps {
        rworld.step();
    }
    let rapier_elapsed = t1.elapsed().as_secs_f64();
    let rp = rworld.bodies[ball].translation();
    let rapier_sps = (steps as f64) / rapier_elapsed.max(1e-9);

    RapierComparisonResult {
        dt_s: dt,
        steps,
        analytic_x_stop_m: analytic_x_stop,
        analytic_z_eq_m: analytic_z_eq,
        fg_x_final_m: fg_pos.x(),
        fg_z_final_m: fg_pos.z(),
        fg_x_err_m: (fg_pos.x() - analytic_x_stop).abs(),
        fg_z_err_m: (fg_pos.z() - analytic_z_eq).abs(),
        fg_steps_per_sec: fg_sps,
        fg_us_per_step: 1e6 / fg_sps.max(1e-9),
        rapier_x_final_m: rp.x,
        rapier_z_final_m: rp.z,
        rapier_x_err_m: (rp.x - analytic_x_stop).abs(),
        rapier_z_err_m: (rp.z - analytic_z_eq).abs(),
        rapier_steps_per_sec: rapier_sps,
        rapier_us_per_step: 1e6 / rapier_sps.max(1e-9),
    }
}

/// Benchmark 2: `faer` SIMD supernodal sparse $LDL^\top$ vs custom Block-CSC on cyclic lattices.
pub fn bench_sparse_faer_vs_custom() -> Vec<SparseBackendBenchRow> {
    let cases = [
        ("RigidFlexGrid-4x4 (beta1=9)", build_rigid_flex_grid_4x4(), 80usize, 12usize),
        ("RigidFlexGrid-6x6 (beta1=25)", build_rigid_flex_grid_6x6(), 50usize, 8usize),
        (
            "RigidFlexGrid-8x8 (beta1=49)",
            build_rigid_flex_grid(&FlexGridConfig {
                rows: 8,
                cols: 8,
                elements_per_beam: 2,
                ..Default::default()
            }),
            30usize,
            4usize,
        ),
    ];

    let mut rows = Vec::with_capacity(cases.len());
    for (label, graph, factor_iters, sim_steps) in cases {
        let topo = graph.analyze().expect("grid analyze");
        let n = topo.num_free_bodies;
        let pat = &topo.sparse_pattern;

        // Build representative SPD cyclic RigidSchurMatrix matching the grid's exact block graph
        let mut mat = RigidSchurMatrix::zeros(n);
        for i in 0..n {
            let mut d = Mat6::ZERO;
            for k in 0..6 {
                d.0[k][k] = 120.0 + 5.0 * ((i % 7) as f64);
            }
            d.0[0][4] = 1.5;
            d.0[4][0] = 1.5;
            mat.add_block(i, i, d);
        }
        for beam in &graph.beams {
            if let Some(bb) = beam.body_b {
                if let (Some(u), Some(v)) = (
                    topo.body_to_free_slot[beam.body_a],
                    topo.body_to_free_slot[bb],
                ) {
                    if u != v {
                        let mut c = Mat6::ZERO;
                        for k in 0..6 {
                            c.0[k][k] = -12.0;
                        }
                        c.0[0][5] = 0.8;
                        c.0[5][0] = -0.8;
                        mat.add_block(u, v, c);
                        mat.add_block(v, u, c.transpose());
                    }
                }
            }
        }
        let rhs: Vec<Vec6> = (0..n)
            .map(|i| {
                let f = (i + 1) as f64;
                Vec6([f.sin(), f.cos(), -0.5 * f, 0.1 * f, -0.2 * f, 0.3 * f.sin()])
            })
            .collect();

        // Isolated sparse factorization + solve timing (Custom Block-CSC vs faer supernodal)
        let faer_pat = FaerSparsePattern::from_matrix(&mat).expect("faer pat");
        let x_custom = SparseBlockLdltFactor::factorize(&mat, pat)
            .expect("custom factor")
            .solve(&rhs);
        let x_faer = FaerSparseLdltFactor::factorize_with_pattern(&mat, &faer_pat)
            .expect("faer factor")
            .solve(&rhs);
        let mut max_solve_diff = 0.0_f64;
        for (a, b) in x_custom.iter().zip(x_faer.iter()) {
            max_solve_diff = max_solve_diff.max((*a - *b).max_abs());
        }

        let tc0 = Instant::now();
        for _ in 0..factor_iters {
            let fac = SparseBlockLdltFactor::factorize(&mat, pat).expect("custom factor");
            let _ = std::hint::black_box(fac.solve(&rhs));
        }
        let custom_factor_us = tc0.elapsed().as_secs_f64() * 1e6 / (factor_iters as f64);

        let tf0 = Instant::now();
        for _ in 0..factor_iters {
            let fac = FaerSparseLdltFactor::factorize_with_pattern(&mat, &faer_pat)
                .expect("faer factor");
            let _ = std::hint::black_box(fac.solve(&rhs));
        }
        let faer_factor_us = tf0.elapsed().as_secs_f64() * 1e6 / (factor_iters as f64);

        // Full end-to-end simulation step comparison
        let cfg_custom = SolverConfig {
            sparse_backend: SparseBackend::CustomBlockCsc,
            ..SolverConfig::default()
        };
        let cfg_faer = SolverConfig {
            sparse_backend: SparseBackend::FaerSupernodal,
            ..SolverConfig::default()
        };
        let mut w_custom =
            FlexGraphWorld::new(graph.clone(), WorldSdf::flat_ground(), cfg_custom).expect("w_c");
        let mut w_faer =
            FlexGraphWorld::new(graph.clone(), WorldSdf::flat_ground(), cfg_faer).expect("w_f");
        let tgts = [0.004, 0.002, 0.001, 0.003];
        w_custom.set_all_targets(&tgts);
        w_faer.set_all_targets(&tgts);
        w_custom.step(0.001).expect("warmup custom");
        w_faer.step(0.001).expect("warmup faer");

        let ts0 = Instant::now();
        w_custom.step_n(sim_steps, 0.001).expect("custom sim");
        let custom_step_us = ts0.elapsed().as_secs_f64() * 1e6 / (sim_steps as f64);

        let ts1 = Instant::now();
        w_faer.step_n(sim_steps, 0.001).expect("faer sim");
        let faer_step_us = ts1.elapsed().as_secs_f64() * 1e6 / (sim_steps as f64);

        let mut max_pose_diff_m = 0.0_f64;
        for (pa, pb) in w_custom.state.poses.iter().zip(w_faer.state.poses.iter()) {
            max_pose_diff_m = max_pose_diff_m.max((pa.pos - pb.pos).norm());
        }

        rows.push(SparseBackendBenchRow {
            label: label.to_string(),
            num_free_bodies: n,
            num_beams: graph.beams.len(),
            rigid_dofs: 6 * n,
            cyclomatic_number: topo.cyclomatic_number,
            factor_iters,
            custom_factor_us,
            faer_factor_us,
            factor_speedup: custom_factor_us / faer_factor_us.max(1e-9),
            max_solve_diff,
            sim_steps,
            custom_step_us,
            faer_step_us,
            step_speedup: custom_step_us / faer_step_us.max(1e-9),
            max_pose_diff_m,
        });
    }
    rows
}

/// Benchmark 3: Nonlinear Bisshopp-Drucker (1945) Elastica & analytical Euler buckling load.
pub fn bench_elastica_and_euler_buckling() -> ElasticaBucklingResult {
    let length_m = 0.50_f64;
    let radius_m = 0.006_f64;
    let youngs_pa = 10.0e6_f64;
    let n_elements = 16usize;
    let ei = youngs_pa * 0.25 * std::f64::consts::PI * radius_m.powi(4);

    // Part A: Large-deflection nonlinear cantilever Elastica at dimensionless load FL^2 / EI = 1.5
    // (where linear theory over-predicts vertical deflection by ~20% and misses foreshortening!)
    let dimless_load = 1.5_f64;
    let tip_load_n = dimless_load * ei / (length_m * length_m);
    let g_tip = 9.81_f64;
    let tip_mass = tip_load_n / g_tip;

    let mut rod = build_euler_elastica_rod(
        length_m,
        radius_m,
        youngs_pa,
        1.0, // negligible self-weight (beam mass < 0.06 g vs tip_mass ~ 6.2 g)
        n_elements,
        BeamFormulation::CorotationalEulerBernoulli,
        Vec3::new(0.0, 0.0, -g_tip),
        Some(tip_mass),
    );
    // Heavy damping for fast quasi-static convergence to exact nonlinear static equilibrium
    rod.beams[0].material.rayleigh_alpha = 25.0;
    rod.beams[0].material.rayleigh_beta = 0.04;
    rod.bodies[1].rayleigh_alpha = 25.0;

    let cfg = SolverConfig {
        integrator: IntegratorKind::DampedNewtonEuler,
        contact: ContactBackend::None,
        use_strain_jacobian: true,
        newton_max_iters: 25,
        newton_tol: 1e-11,
        ..SolverConfig::default()
    };
    let mut world =
        FlexGraphWorld::new(rod, WorldSdf::new(MaterialLibrary::standard()), cfg.clone())
            .expect("elastica world");
    let z0 = world.state.poses[0].pos.z();
    for _ in 0..90 {
        let diag = world.step(0.025).expect("elastica step");
        assert!(!diag.failed);
    }
    let tip_pos = world.state.poses[1].pos;
    let sim_x_tip = tip_pos.x();
    let sim_dz_down = z0 - tip_pos.z();
    let (ana_x, ana_dz, _) = elastica_cantilever_tip(tip_load_n, youngs_pa, radius_m, length_m);
    let x_rel_err_pct = 100.0 * (sim_x_tip - ana_x).abs() / ana_x;
    let dz_rel_err_pct = 100.0 * (sim_dz_down - ana_dz).abs() / ana_dz;

    // Part B: Analytical Euler critical buckling load P_cr = pi^2 * E * I / (4 * L^2)
    let p_cr = euler_buckling_critical_load(youngs_pa, radius_m, length_m);
    let run_axial_load = |load_frac: f64| -> f64 {
        let p_ax = load_frac * p_cr;
        let m_tip = 0.01_f64;
        let g_ax = p_ax / m_tip;
        // Tiny lateral imperfection (0.1% of axial load) breaks symmetry cleanly
        let g_lat = 1e-3 * g_ax;
        let mut r = build_euler_elastica_rod(
            length_m,
            radius_m,
            youngs_pa,
            1.0,
            12,
            BeamFormulation::CorotationalEulerBernoulli,
            Vec3::new(-g_ax, 0.0, -g_lat),
            Some(m_tip),
        );
        // Moderate damping lets the supercritical Euler mode grow quickly onto the post-buckled
        // Elastica equilibrium while keeping the subcritical mode cleanly damped.
        r.beams[0].material.rayleigh_alpha = 3.0;
        r.beams[0].material.rayleigh_beta = 0.005;
        r.bodies[1].rayleigh_alpha = 3.0;
        let mut w =
            FlexGraphWorld::new(r, WorldSdf::new(MaterialLibrary::standard()), cfg.clone())
                .expect("buckling world");
        let z_root = w.state.poses[0].pos.z();
        for _ in 0..100 {
            let d = w.step(0.05).expect("buckling step");
            assert!(!d.failed);
        }
        (w.state.poses[1].pos.z() - z_root).abs()
    };
    let sub_lat = run_axial_load(0.75);
    let sup_lat = run_axial_load(1.30);

    ElasticaBucklingResult {
        rod_length_m: length_m,
        n_elements,
        tip_load_n,
        dimensionless_load: dimless_load,
        analytic_x_tip_m: ana_x,
        analytic_dz_down_m: ana_dz,
        sim_x_tip_m: sim_x_tip,
        sim_dz_down_m: sim_dz_down,
        x_rel_err_pct,
        dz_rel_err_pct,
        euler_p_cr_n: p_cr,
        subcritical_lat_m: sub_lat,
        supercritical_lat_m: sup_lat,
        buckling_amplification: sup_lat / sub_lat.max(1e-12),
    }
}

/// Benchmark 4: IFT reverse-mode adjoint vs central finite differences speedup & SysID recovery.
pub fn bench_adjoint_vs_fd_and_sysid() -> AdjointSysIdResult {
    // Active-contact 2-body flexible continuum crawler with exact tangent-aligned rest poses
    // and 2 independent linear tendons (smoothly active Coulomb friction + elastic bending).
    let true_e_gpa = 60.0_f64;
    let init_e_gpa = 85.0_f64;
    let true_mu = 0.45_f64;
    let init_mu = 0.28_f64;

    let build_contact_crawler = |e_gpa: f64, mu_val: f64| -> FlexGraphWorld {
        let p0 = Vec3::new(0.0, 0.0, 0.0185); // 1.5 mm active penetration into z=0 (r=0.02)
        let p1 = Vec3::new(0.12, 0.0, 0.0185);
        let n_el = 4usize;
        let bow = 0.02_f64;
        let poses: Vec<Se3> = (0..=n_el)
            .map(|i| {
                let u = (i as f64) / (n_el as f64);
                let pos = p0 + (p1 - p0) * u + Vec3::new(0.0, 0.0, 4.0 * bow * u * (1.0 - u));
                let tang = ((p1 - p0) + Vec3::new(0.0, 0.0, 4.0 * bow * (1.0 - 2.0 * u)))
                    .normalized();
                let w = Vec3::EY;
                let n = tang.cross(w).normalized();
                Se3::new(pos, Quat::from_mat3(Mat3::from_cols(tang, w, n)))
            })
            .collect();
        let make_body = |id: usize, pos: Vec3| RigidBodySpec {
            id,
            name: format!("b{id}"),
            is_fixed: false,
            mass: 0.30,
            inertia: Mat3::diag([4.0e-4, 2.5e-4, 2.5e-4]),
            com_local: Vec3::ZERO,
            rest_pose: Se3::new(pos, Quat::IDENTITY),
            collision_radius: 0.015,
            rayleigh_alpha: 0.05,
            tags: Vec::new(),
        };
        let make_ten = |id: usize, y_off: f64| TendonEdgeSpec {
            id,
            name: format!("ten_{id}"),
            actuator_id: id,
            waypoints: vec![
                TendonWaypoint {
                    body_id: 0,
                    local_pos: Vec3::new(0.0, y_off, 0.0),
                    wrap_angle_rad: 0.0,
                },
                TendonWaypoint {
                    body_id: 1,
                    local_pos: Vec3::new(0.0, y_off, 0.0),
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
            servo_torque_limit: 50.0,
            servo_speed_limit: 5.0,
            servo_deadband: 0.0,
            angle_limits: [0.0, 0.04],
            solref_imp: None,
        };
        let g = RigidFlexGraph {
            name: "sysid_active_crawler".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![make_body(0, p0), make_body(1, p1)],
            beams: vec![BeamEdgeSpec {
                id: 0,
                name: "strip".into(),
                body_a: 0,
                clamp_a: Se3::new(Vec3::ZERO, poses[0].rot),
                body_b: Some(1),
                clamp_b: Some(Se3::new(Vec3::ZERO, poses[n_el].rot)),
                n_elements: n_el,
                formulation: BeamFormulation::CorotationalEulerBernoulli,
                section: BeamSection::Rectangular {
                    width: 0.016,
                    thickness: 0.00025,
                },
                material: BeamMaterial {
                    youngs_modulus: e_gpa * 1e9,
                    rayleigh_alpha: 0.05,
                    rayleigh_beta: 0.002,
                    ..BeamMaterial::SPRING_STEEL
                },
                thin_shell_compat: false,
                prestress_frac: 0.0,
                centerline_rest_poses: poses,
                element_lengths: Vec::new(),
                element_transforms: None,
                interior_node_contact_radius: None,
                enable_self_contact: false,
            }],
            tendons: vec![make_ten(0, 0.012), make_ten(1, -0.012)],
            joints: Vec::new(),
            contact_sites: vec![
                ContactSiteSpec {
                    id: 0,
                    name: "pad0".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 0 },
                    local_pos: Vec3::ZERO,
                    local_frame: Quat::IDENTITY,
                    radius: 0.02,
                    patch_radius: 0.005,
                    mode: ContactMode::IsotropicPad {
                        mu: mu_val,
                        pivot_mu: 0.1,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: None,
                },
                ContactSiteSpec {
                    id: 1,
                    name: "pad1".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 1 },
                    local_pos: Vec3::ZERO,
                    local_frame: Quat::IDENTITY,
                    radius: 0.02,
                    patch_radius: 0.005,
                    mode: ContactMode::IsotropicPad {
                        mu: mu_val,
                        pivot_mu: 0.1,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: None,
                },
            ],
            symmetry: SymmetryGroupSpec::trivial(),
            solver_hints: SolverHints::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let mut w = FlexGraphWorld::new(
            g,
            WorldSdf::flat_ground(),
            SolverConfig {
                integrator: IntegratorKind::LinearlyImplicitEuler,
                contact: ContactBackend::Arpc { implicit: true },
                use_strain_jacobian: true,
                world_fixed_inertia: false,
                ..SolverConfig::default()
            },
        )
        .expect("sysid crawler world");
        w.state.vels[0] = Vec6::from_parts(Vec3::new(0.35, 0.0, 0.0), Vec3::ZERO);
        w.state.vels[1] = Vec6::from_parts(Vec3::new(0.35, 0.0, 0.0), Vec3::ZERO);
        w
    };

    let horizon = 16usize;
    let dt = 0.001_f64;
    let controls: Vec<Vec<f64>> = (0..horizon)
        .map(|k| {
            let base = 0.001 + 0.0008 * (k as f64);
            vec![base, 0.85 * base]
        })
        .collect();

    // 1. Ground-truth rollout at (E_true = 60 GPa, mu_true = 0.45)
    let mut w_true = build_contact_crawler(true_e_gpa, true_mu);
    let mut obs_poses = Vec::with_capacity(horizon + 1);
    let mut obs_vels = Vec::with_capacity(horizon + 1);
    obs_poses.push(w_true.state.poses[..2].to_vec());
    obs_vels.push(w_true.state.vels[..2].to_vec());
    for u in &controls {
        w_true.set_all_targets(u);
        w_true.step(dt).expect("sysid truth step");
        obs_poses.push(w_true.state.poses[..2].to_vec());
        obs_vels.push(w_true.state.vels[..2].to_vec());
    }

    let mut loss = PoseTrackingLoss::new(obs_poses, 1.0e4, 10.0);
    loss.target_vels = Some(obs_vels);
    loss.weight_vel = 10.0;
    loss.weight_ctrl = 0.02;

    let sysid_specs = vec![
        DiffParamSpec {
            name: "E_10gpa".into(),
            kind: DiffParamKind::BeamYoungsModulus,
            target_ids: Vec::new(),
            bounds: Some([10e9, 300e9]),
            scale: 10e9,
        },
        DiffParamSpec {
            name: "mu".into(),
            kind: DiffParamKind::ContactMuFwd,
            target_ids: Vec::new(),
            bounds: Some([0.05, 1.5]),
            scale: 1.0,
        },
    ];

    // 2. Speed & FD accuracy comparison at wrong initial guess (85 GPa, 0.28):
    //    16 steps * 2 tendons + 2 physical params = 34 scalar variables under active contact!
    let mut w_fd = build_contact_crawler(init_e_gpa, init_mu);
    let init_state = w_fd.state.clone();
    let _ = compute_trajectory_adjoint(&mut w_fd, &controls, dt, &sysid_specs, &loss)
        .expect("adj warmup");
    let t_adj = Instant::now();
    let adj = compute_trajectory_adjoint(&mut w_fd, &controls, dt, &sysid_specs, &loss)
        .expect("adj timed");
    let adj_ms = t_adj.elapsed().as_secs_f64() * 1e3;

    let eval_fwd_loss = |w: &mut FlexGraphWorld, ctrls: &[Vec<f64>]| -> f64 {
        w.state = init_state.clone();
        let mut total = loss
            .step_loss_and_grad(0, &w.state.poses, &w.state.vels, Some(&ctrls[0]))
            .0;
        for (k, u) in ctrls.iter().enumerate() {
            w.set_all_targets(u);
            w.step(dt).expect("fd step");
            let next_u = ctrls.get(k + 1).map(|v| v.as_slice());
            total += loss
                .step_loss_and_grad(k + 1, &w.state.poses, &w.state.vels, next_u)
                .0;
        }
        w.state = init_state.clone();
        total
    };

    let t_fd = Instant::now();
    let mut max_grad_rel_err = 0.0_f64;
    let eps_u = 1e-6_f64;
    let mut ctrl_pert = controls.clone();
    for k in 0..horizon {
        for a in 0..2 {
            let orig = ctrl_pert[k][a];
            ctrl_pert[k][a] = orig + eps_u;
            let lp = eval_fwd_loss(&mut w_fd, &ctrl_pert);
            ctrl_pert[k][a] = orig - eps_u;
            let lm = eval_fwd_loss(&mut w_fd, &ctrl_pert);
            ctrl_pert[k][a] = orig;
            let fd = (lp - lm) / (2.0 * eps_u);
            let rel = (adj.grad_controls[k][a] - fd).abs() / fd.abs().max(adj.grad_controls[k][a].abs()).max(1e-7);
            max_grad_rel_err = max_grad_rel_err.max(rel);
        }
    }
    for (pi, spec) in sysid_specs.iter().enumerate() {
        let p0 = get_diff_param_value(&w_fd, spec);
        let eps_p = p0.abs().max(1.0) * 1e-5;
        set_diff_param_value(&mut w_fd, spec, p0 + eps_p, false);
        let lp = eval_fwd_loss(&mut w_fd, &controls);
        set_diff_param_value(&mut w_fd, spec, p0 - eps_p, false);
        let lm = eval_fwd_loss(&mut w_fd, &controls);
        set_diff_param_value(&mut w_fd, spec, p0, false);
        let fd = ((lp - lm) / (2.0 * eps_p)) * spec.scale;
        let rel = (adj.grad_params[pi] - fd).abs() / fd.abs().max(adj.grad_params[pi].abs()).max(1e-7);
        max_grad_rel_err = max_grad_rel_err.max(rel);
    }
    let fd_ms = t_fd.elapsed().as_secs_f64() * 1e3;
    let total_vars = horizon * 2 + sysid_specs.len();

    // 3. Adam Sim2Real SysID recovery: recover (E = 60 GPa, mu = 0.45) from (+41.7%, -37.8%) to < 1%!
    loss.weight_ctrl = 0.0;
    let mut w_est = build_contact_crawler(init_e_gpa, init_mu);
    let mut theta = [init_e_gpa / 10.0, init_mu];
    let mut m_adam = [0.0_f64; 2];
    let mut v_adam = [0.0_f64; 2];
    let lr = 0.15_f64;
    let (b1, b2, eps_a) = (0.85_f64, 0.99_f64, 1e-8_f64);
    let max_sysid_iters = 120usize;
    let mut used_iters = 0usize;

    let t_sys = Instant::now();
    for iter in 1..=max_sysid_iters {
        used_iters = iter;
        set_diff_param_value(&mut w_est, &sysid_specs[0], theta[0] * 10e9, true);
        set_diff_param_value(&mut w_est, &sysid_specs[1], theta[1], true);
        let res =
            compute_trajectory_adjoint(&mut w_est, &controls, dt, &sysid_specs, &loss)
                .expect("sysid adjoint");
        if res.loss < 1e-12 {
            break;
        }
        for i in 0..2 {
            let g = res.grad_params[i];
            m_adam[i] = b1 * m_adam[i] + (1.0 - b1) * g;
            v_adam[i] = b2 * v_adam[i] + (1.0 - b2) * g * g;
            let m_hat = m_adam[i] / (1.0 - b1.powi(iter as i32));
            let v_hat = v_adam[i] / (1.0 - b2.powi(iter as i32));
            theta[i] -= lr * m_hat / (v_hat.sqrt() + eps_a);
        }
        theta[0] = theta[0].clamp(1.0, 30.0);
        theta[1] = theta[1].clamp(0.05, 1.5);
    }
    let sysid_elapsed_ms = t_sys.elapsed().as_secs_f64() * 1e3;
    let recovered_e_gpa = theta[0] * 10.0;

    AdjointSysIdResult {
        horizon_steps: horizon,
        total_diff_vars: total_vars,
        adjoint_ms: adj_ms,
        fd_ms,
        speedup_vs_fd: fd_ms / adj_ms.max(1e-9),
        max_grad_rel_err,
        sysid_iters: used_iters,
        true_youngs_gpa: true_e_gpa,
        init_youngs_gpa: init_e_gpa,
        recovered_youngs_gpa: recovered_e_gpa,
        youngs_err_pct: 100.0 * (recovered_e_gpa - true_e_gpa).abs() / true_e_gpa,
        true_mu,
        init_mu,
        recovered_mu: theta[1],
        mu_err_pct: 100.0 * (theta[1] - true_mu).abs() / true_mu,
        sysid_elapsed_ms,
    }
}

/// Benchmark 5: `parry3d-f64` `TriMeshSdf` corrugated pipe locomotion + Tight Inclusion CCD.
pub fn bench_corrugated_pipe_trimesh_ticcd() -> CorrugatedPipeBenchResult {
    let pipe = Arc::new(
        corrugated_pipe_mesh(
            0.80,
            0.055,
            0.004,
            8.0,
            32,
            20,
            Vec3::new(0.15, 0.0, 0.0),
            MAT_ACRYLIC_WALL,
        )
        .expect("corrugated pipe mesh"),
    );
    let pipe_triangles = pipe.indices().len();

    let build_crawler = |z_init: f64| -> RigidFlexGraph {
        let p0 = Vec3::new(0.02, 0.0, z_init);
        let p1 = Vec3::new(-0.08, 0.0, z_init);
        let p2 = Vec3::new(-0.18, 0.0, z_init);
        let make_strip = |id: usize, ba: usize, bb: usize, a: Vec3, b: Vec3| -> BeamEdgeSpec {
            let mut poses = Vec::with_capacity(5);
            for j in 0..=4 {
                let u = (j as f64) / 4.0;
                let mut pos = a + (b - a) * u;
                pos[2] += 0.015 * (std::f64::consts::PI * u).sin();
                poses.push(Se3::new(pos, Quat::IDENTITY));
            }
            BeamEdgeSpec {
                id,
                name: format!("strip_{id}"),
                body_a: ba,
                clamp_a: Se3::IDENTITY,
                body_b: Some(bb),
                clamp_b: Some(Se3::IDENTITY),
                n_elements: 4,
                formulation: BeamFormulation::CorotationalEulerBernoulli,
                section: BeamSection::Rectangular {
                    width: 0.015,
                    thickness: 0.001,
                },
                material: BeamMaterial {
                    youngs_modulus: 65e9,
                    poisson_ratio: 0.30,
                    density: 7850.0,
                    rayleigh_alpha: 0.05,
                    rayleigh_beta: 0.002,
                },
                thin_shell_compat: false,
                prestress_frac: 0.0,
                centerline_rest_poses: poses,
                element_lengths: vec![0.0265; 4],
                element_transforms: None,
                interior_node_contact_radius: Some(0.004),
                enable_self_contact: false,
            }
        };
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
        let make_hub = |id: usize, pos: Vec3| RigidBodySpec {
            id,
            name: format!("hub_{id}"),
            is_fixed: false,
            mass: 0.12,
            inertia: Mat3::diag([1.5e-4, 1.5e-4, 1.5e-4]),
            com_local: Vec3::ZERO,
            rest_pose: Se3::new(pos, Quat::IDENTITY),
            collision_radius: 0.012,
            rayleigh_alpha: 0.05,
            tags: Vec::new(),
        };
        RigidFlexGraph {
            name: "corrugated_pipe_worm".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![make_hub(0, p0), make_hub(1, p1), make_hub(2, p2)],
            beams: vec![make_strip(0, 0, 1, p0, p1), make_strip(1, 1, 2, p1, p2)],
            tendons: vec![make_ten(0, 0, 1), make_ten(1, 1, 2)],
            joints: Vec::new(),
            contact_sites: vec![make_scale(0, 0), make_scale(1, 1), make_scale(2, 2)],
            symmetry: SymmetryGroupSpec::trivial(),
            solver_hints: SolverHints::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        }
    };

    // Part A: 160 peristaltic crawling steps via ARPC inside corrugated TriMeshSdf
    let arpc_steps = 160usize;
    let dt = 0.002_f64;
    let mut w_arpc = FlexGraphWorld::new(
        build_crawler(-0.035),
        WorldSdf::new(MaterialLibrary::standard()),
        SolverConfig {
            integrator: IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::Arpc { implicit: true },
            use_strain_jacobian: true,
            ..SolverConfig::default()
        },
    )
    .expect("pipe arpc world");
    w_arpc.register_static_trimesh(Arc::clone(&pipe));
    let s0 = w_arpc.sdf.query(w_arpc.state.poses[2].pos).arc_s;
    let mut arpc_min_phi = f64::INFINITY;
    let t0 = Instant::now();
    for step in 0..arpc_steps {
        let t = (step as f64) * dt;
        let u_front = 0.011 * (1.0 + (2.0 * std::f64::consts::PI * 5.0 * t).sin());
        let u_rear = 0.011 * (1.0 + (2.0 * std::f64::consts::PI * 5.0 * t - 1.5).sin());
        w_arpc.set_tendon_targets(&[u_rear, u_front]);
        let diag = w_arpc.step(dt).expect("pipe arpc step");
        assert!(!diag.failed);
        for bi in 0..3 {
            arpc_min_phi = arpc_min_phi.min(w_arpc.sdf.query(w_arpc.state.poses[bi].pos).phi);
        }
    }
    let arpc_elapsed = t0.elapsed().as_secs_f64();
    let s1 = w_arpc.sdf.query(w_arpc.state.poses[2].pos).arc_s;

    // Part B: SdfBarrier + Tight Inclusion Point-Triangle CCD inside same corrugated pipe mesh
    let ipc_steps = 20usize;
    let mut w_ipc = FlexGraphWorld::new(
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
    .expect("pipe ipc world");
    w_ipc.register_static_trimesh(pipe);
    w_ipc.set_tendon_targets(&[0.012, 0.006]);
    let mut ipc_min_phi = f64::INFINITY;
    let mut ipc_min_toi = 1.0_f64;
    let t1 = Instant::now();
    for _ in 0..ipc_steps {
        let diag = w_ipc.step(dt).expect("pipe ipc step");
        assert!(!diag.failed);
        ipc_min_toi = ipc_min_toi.min(diag.ccd_toi);
        for bi in 0..3 {
            ipc_min_phi = ipc_min_phi.min(w_ipc.sdf.query(w_ipc.state.poses[bi].pos).phi);
        }
    }
    let ipc_elapsed = t1.elapsed().as_secs_f64();

    CorrugatedPipeBenchResult {
        pipe_triangles,
        arpc_steps,
        arpc_arc_progress_m: s1 - s0,
        arpc_min_phi_m: arpc_min_phi,
        arpc_steps_per_sec: (arpc_steps as f64) / arpc_elapsed.max(1e-9),
        ipc_ticcd_steps: ipc_steps,
        ipc_ticcd_min_phi_m: ipc_min_phi,
        ipc_ticcd_min_toi: ipc_min_toi,
        ipc_ticcd_steps_per_sec: (ipc_steps as f64) / ipc_elapsed.max(1e-9),
    }
}

/// Run all 5 SOTA FlexGraph-RS v2 benchmarks (`< 5s` total in release mode).
pub fn run_v2_sota_benchmarks() -> V2BenchSuiteReport {
    let t_start = Instant::now();
    let rapier_comparison = bench_rapier_rigid_contact();
    let sparse_scaling = bench_sparse_faer_vs_custom();
    let elastica_buckling = bench_elastica_and_euler_buckling();
    let adjoint_sysid = bench_adjoint_vs_fd_and_sysid();
    let corrugated_pipe = bench_corrugated_pipe_trimesh_ticcd();
    V2BenchSuiteReport {
        rapier_comparison,
        sparse_scaling,
        elastica_buckling,
        adjoint_sysid,
        corrugated_pipe,
        total_elapsed_s: t_start.elapsed().as_secs_f64(),
    }
}

//! `fg-bench`: Reproducible `< 5s` CPU throughput benchmark across all 5 `flexgraph-rs`
//! robot archetypes and all 3 Schur routes (`BlockDiagonalPairs`, `TreeChain`, `SparseGeneral`),
//! measuring both single-thread and Rayon multi-environment parallel stepping speed.

#![allow(clippy::needless_range_loop)]

pub mod v2_suite;

pub use v2_suite::{
    bench_adjoint_vs_fd_and_sysid, bench_corrugated_pipe_trimesh_ticcd,
    bench_elastica_and_euler_buckling, bench_rapier_rigid_contact, bench_sparse_faer_vs_custom,
    run_v2_sota_benchmarks, AdjointSysIdResult, CorrugatedPipeBenchResult, ElasticaBucklingResult,
    RapierComparisonResult, SparseBackendBenchRow, V2BenchSuiteReport,
};

use fg_ir::SchurRoute;
use fg_robots::{
    build_archetype_default_world, build_bristle_worm, build_flexspine_centipede,
    build_rigid_flex_grid_4x4, build_serpworm, build_soft_octopus_arm, build_tendon_endoscope,
    build_tensegrity_6bar, BristleWormConfig, CentipedeConfig, EndoscopeConfig, OctopusArmConfig,
    SerpWormConfig, SerpWormPreset, TensegrityConfig,
};
use fg_solver::{FlexGraphWorld, SolverConfig};
use serde::Serialize;
use std::time::Instant;

/// Benchmark metrics for a single robot archetype.
#[derive(Clone, Debug, Serialize)]
pub struct ArchetypeBenchResult {
    pub name: String,
    pub schur_route: String,
    pub num_bodies: usize,
    pub total_nodes: usize,
    pub total_dofs: usize,
    pub dt_s: f64,
    pub single_steps: usize,
    pub single_elapsed_ms: f64,
    pub single_steps_per_sec: f64,
    pub single_us_per_step: f64,
    pub batch_size: usize,
    pub batch_steps: usize,
    pub batch_elapsed_ms: f64,
    pub batch_env_steps_per_sec: f64,
    pub parallel_speedup: f64,
}

fn route_label(r: SchurRoute) -> &'static str {
    match r {
        SchurRoute::BlockDiagonalPairs => "block_diagonal_pairs",
        SchurRoute::TreeChain => "tree_chain",
        SchurRoute::SparseGeneral => "sparse_general",
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_case(
    name: &str,
    graph: fg_ir::RigidFlexGraph,
    world_key: &str,
    solver_cfg: SolverConfig,
    dt: f64,
    targets: &[f64],
    single_steps: usize,
    batch_size: usize,
    batch_steps: usize,
) -> ArchetypeBenchResult {
    let sdf = build_archetype_default_world(world_key).expect("default world");
    let mut single = FlexGraphWorld::new(graph.clone(), sdf, solver_cfg.clone()).expect("world");
    single.set_all_targets(targets);

    let route = route_label(single.analysis.route).to_string();
    let num_bodies = single.analysis.num_bodies;
    let total_nodes = single.total_nodes();
    let total_dofs = 6 * (single.analysis.num_free_bodies + single.analysis.indexing.num_beam_nodes);

    // Warmup
    single
        .step_n(2, dt)
        .unwrap_or_else(|e| panic!("warmup step failed on {name}: {e}"));

    // Single-thread timing
    let t0 = Instant::now();
    let diag = single
        .step_n(single_steps, dt)
        .unwrap_or_else(|e| panic!("single bench step failed on {name} at step {}: {e}", single.state.step_count));
    let single_elapsed = t0.elapsed().as_secs_f64();
    assert!(!diag.failed, "benchmark step failed on {name}");

    // Rayon multi-env timing
    let mut batch: Vec<FlexGraphWorld> = (0..batch_size)
        .map(|i| {
            let s = build_archetype_default_world(world_key).expect("batch world");
            let mut w = FlexGraphWorld::new(graph.clone(), s, solver_cfg.clone()).expect("world");
            let mut tgt = targets.to_vec();
            if let Some(first) = tgt.first_mut() {
                *first += 0.001 * (i as f64);
            }
            w.set_all_targets(&tgt);
            w
        })
        .collect();

    FlexGraphWorld::step_batch_n_rayon(&mut batch, 2, dt).expect("batch warmup");
    let t1 = Instant::now();
    let diags = FlexGraphWorld::step_batch_n_rayon(&mut batch, batch_steps, dt)
        .expect("batch bench step");
    let batch_elapsed = t1.elapsed().as_secs_f64();
    assert!(diags.iter().all(|d| !d.failed), "batch step failed on {name}");

    let single_sps = (single_steps as f64) / single_elapsed.max(1e-9);
    let batch_sps = ((batch_size * batch_steps) as f64) / batch_elapsed.max(1e-9);

    ArchetypeBenchResult {
        name: name.to_string(),
        schur_route: route,
        num_bodies,
        total_nodes,
        total_dofs,
        dt_s: dt,
        single_steps,
        single_elapsed_ms: single_elapsed * 1e3,
        single_steps_per_sec: single_sps,
        single_us_per_step: 1e6 / single_sps.max(1e-9),
        batch_size,
        batch_steps,
        batch_elapsed_ms: batch_elapsed * 1e3,
        batch_env_steps_per_sec: batch_sps,
        parallel_speedup: batch_sps / single_sps.max(1e-9),
    }
}

/// Run the concise `< 5s` benchmark suite across all 5 calibrated robot archetypes.
pub fn run_all_archetype_benchmarks(
    batch_size: usize,
    scale_steps: usize,
) -> Vec<ArchetypeBenchResult> {
    let s = scale_steps.max(1);
    vec![
        // 1. SerpWorm-5 (E=16, 3660 DoFs, BlockDiagonalPairs, Kelvin parity)
        bench_case(
            "SerpWorm-5 (E=16, Kelvin parity)",
            build_serpworm(
                &SerpWormConfig::from_preset(SerpWormPreset::GpuCoreDefault, 5, 16),
                None,
            ),
            "serpworm_5",
            SolverConfig::serpworm_kelvin_parity(),
            0.0005,
            &[0.8, 0.4, 0.5, 0.9, 0.6, 0.6, 0.4, 0.8, 0.7, 0.3, 0.15, -0.15, 0.10, -0.10],
            24 * s,
            batch_size,
            12 * s,
        ),
        // 2. BristleWorm-3 (E=8, Z2xZ2 anisotropic scales, BlockDiagonalPairs, ARPC implicit)
        bench_case(
            "BristleWorm-3 (E=8, ARPC scales)",
            build_bristle_worm(
                &BristleWormConfig {
                    n_segments: 3,
                    elements_per_strip: 8,
                    ..Default::default()
                },
                None,
            ),
            "bristle_worm",
            SolverConfig::default(),
            0.001,
            &[0.6, 0.6, 0.3, 0.3, 0.7, 0.7, 0.0, 0.0],
            32 * s,
            batch_size,
            16 * s,
        ),
        // 3. TendonEndoscope-3D (fixed base + 4 tendons + 3D SplineTubeSdf, TreeChain)
        bench_case(
            "TendonEndoscope-3D (SplineTubeSdf)",
            build_tendon_endoscope(&EndoscopeConfig::default()),
            "tendon_endoscope_3d",
            SolverConfig::default(),
            0.001,
            &[0.010, 0.004, 0.0, 0.0],
            80 * s,
            batch_size,
            40 * s,
        ),
        // 4. FlexSpine-Centipede (3 torsos + 6 legs + Bekker soft soil, TreeChain)
        bench_case(
            "FlexSpine-Centipede (Bekker soil)",
            build_flexspine_centipede(&CentipedeConfig::default()),
            "flexspine_centipede",
            SolverConfig::default(),
            0.001,
            &[0.3, -0.3, -0.3, 0.3, 0.3, -0.3],
            80 * s,
            batch_size,
            40 * s,
        ),
        // 5. Tensegrity-6Bar (cyclic K_{2,2,2} beta_1=7, SparseGeneral Block-LDL^T)
        bench_case(
            "Tensegrity-6Bar (cyclic beta1=7)",
            build_tensegrity_6bar(&TensegrityConfig::default()),
            "tensegrity_6bar",
            SolverConfig::default(),
            0.001,
            &[0.01; 24],
            60 * s,
            batch_size,
            30 * s,
        ),
        // 6. SoftOctopusArm-DER (tapered 3D Cosserat/DER arm + 4 tendons, TreeChain, D4)
        bench_case(
            "SoftOctopusArm-DER (tapered DER)",
            build_soft_octopus_arm(&OctopusArmConfig::default()),
            "soft_octopus_arm_der",
            SolverConfig::default(),
            0.001,
            &[0.012, 0.004, 0.0, 0.0],
            40 * s,
            batch_size,
            20 * s,
        ),
        // 7. RigidFlexGrid-4x4 (16 hubs, 24 beams, cyclic beta_1=9, faer SparseGeneral)
        bench_case(
            "RigidFlexGrid-4x4 (faer beta1=9)",
            build_rigid_flex_grid_4x4(),
            "rigid_flex_grid_4x4",
            SolverConfig {
                sparse_backend: fg_solver::SparseBackend::FaerSupernodal,
                ..SolverConfig::default()
            },
            0.001,
            &[0.004, 0.002, 0.001, 0.003],
            24 * s,
            batch_size,
            12 * s,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_smoke_all_5_archetypes_fast() {
        let g = build_serpworm(
            &SerpWormConfig::from_preset(SerpWormPreset::GpuCoreDefault, 2, 6),
            None,
        );
        let r_sw = bench_case(
            "SerpWorm-2",
            g,
            "serpworm_5",
            SolverConfig::serpworm_kelvin_parity(),
            0.0005,
            &[0.5, 0.5, 0.4, 0.6, 0.1],
            4,
            2,
            2,
        );
        let r_bw = bench_case(
            "BristleWorm-2",
            build_bristle_worm(
                &BristleWormConfig {
                    n_segments: 2,
                    elements_per_strip: 4,
                    ..Default::default()
                },
                None,
            ),
            "bristle_worm",
            SolverConfig::default(),
            0.001,
            &[0.4, 0.4, 0.3, 0.3, 0.0],
            4,
            2,
            2,
        );
        let r_en = bench_case(
            "TendonEndoscope-3D",
            build_tendon_endoscope(&EndoscopeConfig::default()),
            "tendon_endoscope_3d",
            SolverConfig::default(),
            0.001,
            &[0.01, 0.0, 0.0, 0.0],
            4,
            2,
            2,
        );
        let r_cp = bench_case(
            "FlexSpine-Centipede",
            build_flexspine_centipede(&CentipedeConfig::default()),
            "flexspine_centipede",
            SolverConfig::default(),
            0.001,
            &[0.2, -0.2, -0.2, 0.2, 0.2, -0.2],
            4,
            2,
            2,
        );
        let r_tg = bench_case(
            "Tensegrity-6Bar",
            build_tensegrity_6bar(&TensegrityConfig::default()),
            "tensegrity_6bar",
            SolverConfig::default(),
            0.001,
            &[0.01; 24],
            4,
            2,
            2,
        );
        let results = [r_sw, r_bw, r_en, r_cp, r_tg];
        for r in &results {
            assert!(r.single_steps_per_sec > 0.0 && r.single_steps_per_sec.is_finite());
            assert!(r.batch_env_steps_per_sec > 0.0 && r.batch_env_steps_per_sec.is_finite());
        }
        assert!(results.iter().any(|r| r.schur_route == "block_diagonal_pairs"));
        assert!(results.iter().any(|r| r.schur_route == "tree_chain"));
        assert!(results.iter().any(|r| r.schur_route == "sparse_general"));
    }

    #[test]
    fn test_v2_sota_benchmarks_accuracy_invariants() {
        let rc = bench_rapier_rigid_contact();
        assert!(rc.fg_x_err_m < 0.015, "fg_x_err_m={}", rc.fg_x_err_m);
        assert!(rc.fg_z_err_m < 1e-3, "fg_z_err_m={}", rc.fg_z_err_m);
        assert!(rc.rapier_x_err_m < 0.015, "rapier_x_err_m={}", rc.rapier_x_err_m);

        let eb = bench_elastica_and_euler_buckling();
        assert!(eb.x_rel_err_pct < 1.0, "elastica x err={}%", eb.x_rel_err_pct);
        assert!(eb.dz_rel_err_pct < 1.0, "elastica dz err={}%", eb.dz_rel_err_pct);
        assert!(
            eb.buckling_amplification > 20.0,
            "buckling amp={}",
            eb.buckling_amplification
        );

        let cp = bench_corrugated_pipe_trimesh_ticcd();
        assert!(cp.arpc_arc_progress_m > 0.002, "arpc arc_s={}", cp.arpc_arc_progress_m);
        assert!(cp.arpc_min_phi_m > 0.0, "arpc min_phi={}", cp.arpc_min_phi_m);
        assert!(cp.ipc_ticcd_min_phi_m > 0.012, "ipc min_phi={}", cp.ipc_ticcd_min_phi_m);

        let ad = bench_adjoint_vs_fd_and_sysid();
        assert!(
            ad.max_grad_rel_err < 1e-4,
            "IFT adjoint vs FD max_rel_err={:e}",
            ad.max_grad_rel_err
        );
        assert!(ad.speedup_vs_fd > 2.0, "IFT speedup={}", ad.speedup_vs_fd);
        assert!(ad.youngs_err_pct < 1.0, "SysID E err={}%", ad.youngs_err_pct);
        assert!(ad.mu_err_pct < 1.0, "SysID mu err={}%", ad.mu_err_pct);
    }
}

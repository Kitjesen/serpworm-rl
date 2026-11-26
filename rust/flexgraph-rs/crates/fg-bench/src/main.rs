//! Executable `< 8s` CPU throughput & SOTA v2 benchmark suite for `flexgraph-rs`.
//!
//! Run with:
//! ```bash
//! cargo run --release -p fg-bench
//! ```

use fg_bench::{run_all_archetype_benchmarks, run_v2_sota_benchmarks};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct FullBenchSummary {
    archetypes: Vec<fg_bench::ArchetypeBenchResult>,
    v2_sota: fg_bench::V2BenchSuiteReport,
    total_wall_s: f64,
}

fn main() {
    let n_threads = rayon::current_num_threads();
    let batch_size = n_threads.clamp(8, 16);

    let t_total = Instant::now();
    let archetypes = run_all_archetype_benchmarks(batch_size, 1);
    let v2 = run_v2_sota_benchmarks();
    let total_s = t_total.elapsed().as_secs_f64();

    println!(
        "# flexgraph-rs v2 CPU Throughput & SOTA Benchmark (Rayon threads = {n_threads}, batch = {batch_size})"
    );
    println!();
    println!("## 1. Multi-Robot Archetype Throughput (All 7 Archetypes, All 3 Schur Routes)");
    println!(
        "| Archetype | Schur Route | DoFs | 1-Thread (steps/s) | us/step | Rayon B={batch_size} (env-steps/s) | Parallel Speedup |"
    );
    println!("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |");
    for r in &archetypes {
        println!(
            "| {} | `{}` | {} | {:>10.1} | {:>7.1} | {:>12.1} | {:>6.2}x |",
            r.name,
            r.schur_route,
            r.total_dofs,
            r.single_steps_per_sec,
            r.single_us_per_step,
            r.batch_env_steps_per_sec,
            r.parallel_speedup,
        );
    }

    println!();
    println!("## 2. SOTA v2 Verification & Benchmarks");

    let rc = &v2.rapier_comparison;
    println!();
    println!("### 2.1 Rigid Frictional Contact vs `rapier3d-f64 0.36` (Analytic x_stop = {:.6} m, z_eq = {:.4} m)", rc.analytic_x_stop_m, rc.analytic_z_eq_m);
    println!("| Engine | x_final (m) | |x_err| (m) | z_final (m) | |z_err| (m) | steps/s | us/step |");
    println!("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |");
    println!(
        "| FlexGraph-RS (Clarabel SOCP) | {:.6} | {:.2e} | {:.6} | {:.2e} | {:>9.1} | {:>6.2} |",
        rc.fg_x_final_m,
        rc.fg_x_err_m,
        rc.fg_z_final_m,
        rc.fg_z_err_m,
        rc.fg_steps_per_sec,
        rc.fg_us_per_step
    );
    println!(
        "| Rapier3D-f64 0.36 | {:.6} | {:.2e} | {:.6} | {:.2e} | {:>9.1} | {:>6.2} |",
        rc.rapier_x_final_m,
        rc.rapier_x_err_m,
        rc.rapier_z_final_m,
        rc.rapier_z_err_m,
        rc.rapier_steps_per_sec,
        rc.rapier_us_per_step
    );

    println!();
    println!("### 2.2 `faer 0.24` Supernodal Sparse LDL^T vs Custom Block-CSC");
    println!("| Lattice | Free Bodies | Rigid DoFs | beta_1 | Custom Factor (us) | faer Factor (us) | Factor Speedup | Solve Diff | Step Speedup | Pose Diff (m) |");
    println!("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for s in &v2.sparse_scaling {
        println!(
            "| {} | {} | {} | {} | {:.1} | {:.1} | {:.2}x | {:.2e} | {:.2}x | {:.2e} |",
            s.label,
            s.num_free_bodies,
            s.rigid_dofs,
            s.cyclomatic_number,
            s.custom_factor_us,
            s.faer_factor_us,
            s.factor_speedup,
            s.max_solve_diff,
            s.step_speedup,
            s.max_pose_diff_m
        );
    }

    let eb = &v2.elastica_buckling;
    println!();
    println!("### 2.3 Analytical Bisshopp-Drucker (1945) Elastica & Euler Buckling");
    println!(
        "- Nonlinear cantilever (`FL^2/EI = {:.2}`, `E={}`): x_tip = {:.5} m vs analytic {:.5} m (**{:.3}%** err); dz = {:.5} m vs analytic {:.5} m (**{:.3}%** err)",
        eb.dimensionless_load,
        eb.n_elements,
        eb.sim_x_tip_m,
        eb.analytic_x_tip_m,
        eb.x_rel_err_pct,
        eb.sim_dz_down_m,
        eb.analytic_dz_down_m,
        eb.dz_rel_err_pct
    );
    println!(
        "- Euler buckling (`P_cr = {:.5} N`): subcritical `0.75*P_cr` lat = {:.3e} m vs supercritical `1.30*P_cr` lat = {:.4} m (**{:.1}x** amplification)",
        eb.euler_p_cr_n, eb.subcritical_lat_m, eb.supercritical_lat_m, eb.buckling_amplification
    );

    let ad = &v2.adjoint_sysid;
    println!();
    println!("### 2.4 IFT Reverse-Mode Adjoint vs Central Finite Differences & Sim2Real SysID");
    println!(
        "- {} vars (T={} steps): IFT Adjoint = {:.2} ms vs Central FD = {:.2} ms (**{:.2}x speedup**, max rel grad err = **{:.2e}**)",
        ad.total_diff_vars,
        ad.horizon_steps,
        ad.adjoint_ms,
        ad.fd_ms,
        ad.speedup_vs_fd,
        ad.max_grad_rel_err
    );
    println!(
        "- Adam SysID ({} iters, {:.1} ms): E = {:.2} -> **{:.3} GPa** (true {:.2} GPa, **{:.2}%** err); mu = {:.3} -> **{:.4}** (true {:.3}, **{:.2}%** err)",
        ad.sysid_iters,
        ad.sysid_elapsed_ms,
        ad.init_youngs_gpa,
        ad.recovered_youngs_gpa,
        ad.true_youngs_gpa,
        ad.youngs_err_pct,
        ad.init_mu,
        ad.recovered_mu,
        ad.true_mu,
        ad.mu_err_pct
    );

    let cp = &v2.corrugated_pipe;
    println!();
    println!("### 2.5 `parry3d-f64` TriMeshSdf Corrugated Pipe Locomotion + Tight Inclusion CCD");
    println!(
        "- Mesh ({} tris): ARPC locomotion ({} steps) arc_s progress = **+{:.4} m**, min_phi = {:.4} m (> 0), {:.1} steps/s",
        cp.pipe_triangles,
        cp.arpc_steps,
        cp.arpc_arc_progress_m,
        cp.arpc_min_phi_m,
        cp.arpc_steps_per_sec
    );
    println!(
        "- SdfBarrier + Tight Inclusion CCD ({} steps): min_phi = {:.4} m (> r=0.012 m, zero tunneling), min_toi = {:.4}, {:.1} steps/s",
        cp.ipc_ticcd_steps,
        cp.ipc_ticcd_min_phi_m,
        cp.ipc_ticcd_min_toi,
        cp.ipc_ticcd_steps_per_sec
    );

    println!();
    println!("Total wall-clock time: {total_s:.3} s");
    println!();
    let summary = FullBenchSummary {
        archetypes,
        v2_sota: v2,
        total_wall_s: total_s,
    };
    println!(
        "JSON_SUMMARY={}",
        serde_json::to_string(&summary).expect("serialize summary")
    );
}

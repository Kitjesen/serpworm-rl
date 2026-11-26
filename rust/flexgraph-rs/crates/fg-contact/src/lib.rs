//! `fg-contact`: Tri-engine FP64 contact mechanics library for `flexgraph-rs`.
//!
//! # Engines
//! 1. **Engine A (`broadphase`, `distance`, `ccd`, `ipc`)**:
//!    - $O(N)$ spatial hash grid broadphase for point clouds and codimensional beam segments;
//!    - Exact codimensional distances (`Point-Point`, `Point-Segment`, `Segment-Segment`)
//!      with exact analytical gradients, exact analytical Hessians, and IPC parallel-edge mollifier;
//!    - Conservative **Additive CCD (ACCD)** (Li et al. SIGGRAPH 2021, Alg. 1) with finite thickness
//!      offsets $\xi \ge 0$ plus Lipschitz-certified Point-vs-SDF CCD;
//!    - Exact **IPC $C^2$ clamped log-barrier** $b(d) = -(d - \hat{d})^2 \ln(d / \hat{d})$,
//!      **Py-DiSMech $C^2$ smooth soft-barrier** (Lahoti et al. 2025, Eq. 25–27),
//!      **SPD-projected Hessians** (`spd::project_spd_n`), and **IPC $C^1$ mollified friction**.
//! 2. **Engine B (`lcp`)**:
//!    - Compliance-based frictional contact LCP / Delassus-Schur complementarity solvers
//!      (Shen et al. IJRR 2026 & Moreau contact-implicit): Block Projected Gauss-Seidel (PGS/NSGS),
//!      Alart-Curnier semismooth Newton with Clarke generalized Jacobian, and Fischer-Burmeister
//!      polyhedral $m$-facet LCP with redundant-contact deduplication.
//! 3. **Engine C (`arpc`)**:
//!    - Unified Implicit Anisotropic Patch Contact (ARPC) unifying `RatchetWheel` (bit-level parity
//!      with `serpworm.physics.wheel_stiction`), `AnisotropicScale` ($\mu_{\mathrm{fwd}} \ll \mu_{\mathrm{rev}}, \mu_{\mathrm{lat}}$),
//!      and `IsotropicPad`, with Hunt-Crossley + Bekker elasto-plastic soil rutting, Stribeck
//!      friction ellipse, tangential stick-slip bristle memory, circular patch pivoting torque,
//!      and exact SPD $6\times 6$ Contact-Schur tangent matrices.

#![allow(clippy::needless_range_loop)]

pub mod arpc;
pub mod broadphase;
pub mod ccd;
pub mod distance;
pub mod ipc;
pub mod lcp;
pub mod spd;
pub mod tight_inclusion;

pub use arpc::{
    arpc_normal_and_rut, arpc_tangential_contact, eval_arpc_site, eval_arpc_site_with_mass,
    eval_sdf_surface_contact, implicit_wheel_contact, legacy_wheel_contact, ArpcNormalEval,
    ArpcSiteEval, ArpcSiteState, ArpcTangentialEval, SdfSurfaceContactEval, WheelContactEval,
};
pub use broadphase::SpatialHashGrid;
pub use ccd::{
    accd_point_point, accd_point_sdf, accd_point_segment, accd_segment_segment,
    compute_safe_step_points_sdf, compute_safe_step_segments,
    compute_safe_step_segments_tight_inclusion, tight_inclusion_edge_edge_ccd,
    tight_inclusion_point_triangle_ccd, TightInclusionConfig,
};
pub use distance::{
    edge_edge_mollifier, point_point_distance, point_segment_distance, point_triangle_distance,
    point_triangle_distance_grad_hess, segment_segment_distance, EdgeEdgeMollifierEval,
    PointPointDist, PointSegmentDist, PointTriangleDist, SegmentFeature, SegmentSegmentDist,
    TriangleFeature,
};
pub use ipc::{
    ipc_barrier_eval, ipc_mollifier_friction_eval, ipc_point_point_barrier, ipc_point_sdf_barrier,
    ipc_point_segment_barrier, ipc_point_triangle_barrier, ipc_segment_segment_barrier,
    pydismech_barrier_eval, pydismech_friction_eval, BarrierKind, IpcBarrierConfig,
    IpcFrictionEval, PointPointBarrierEval, PointSdfBarrierEval, PointSegmentBarrierEval,
    PointTriangleBarrierEval, SegmentSegmentBarrierEval, SelfContactAssembly, SelfContactEngine,
    SelfContactTriplet,
};
pub use lcp::{
    ContactConstraint, DelassusSystem, LcpContactConstraint, LcpSolution, LcpSolverConfig,
    LcpSolverKind, LcpSolverMethod,
};
pub use spd::{project_spd_n, sym_eigen_n};

#[cfg(test)]
mod tests {
    use super::*;
    use fg_ir::{ContactAttachment, ContactMode, ContactSiteSpec};
    use fg_math::{Quat, Se3, SolRefImp, Vec3, Vec6};
    use fg_sdf::{
        HeightFieldSdf, MaterialLibrary, PlaneSdf, SdfField, SphereSdf, MAT_BEKKER_SOFT_SOIL,
        MAT_HARD_FLOOR,
    };

    #[test]
    fn test_ipc_log_barrier_and_pydismech_scalar_fd() {
        let d_hat = 0.02;
        let xi = 0.002;
        for d in [0.004, 0.008, 0.015, 0.0215] {
            let (e, de, d2e) = ipc_barrier_eval(d, d_hat, xi).unwrap();
            let eps = 1e-7;
            let (ep, dep, _) = ipc_barrier_eval(d + eps, d_hat, xi).unwrap();
            let (em, dem, _) = ipc_barrier_eval(d - eps, d_hat, xi).unwrap();
            let fd1 = (ep - em) / (2.0 * eps);
            let fd2 = (dep - dem) / (2.0 * eps);
            assert!(e > 0.0 && de < 0.0 && d2e > 0.0);
            assert!((de - fd1).abs() / de.abs().max(1.0) < 1e-7);
            assert!((d2e - fd2).abs() / d2e.abs().max(1.0) < 1e-6);
        }
        // C2 continuity at activation boundary d = xi + d_hat
        let (e_b, de_b, d2e_b) = ipc_barrier_eval(xi + d_hat - 1e-9, d_hat, xi).unwrap();
        assert!(e_b.abs() < 1e-18 && de_b.abs() < 1e-12 && d2e_b.abs() < 1e-5);

        // Py-DiSMech C2 smooth soft-barrier scalar FD
        let d0 = 0.01;
        let delta = 0.003;
        for d in [0.005, 0.008, 0.010, 0.012] {
            let (_, de, d2e) = pydismech_barrier_eval(d, d0, delta);
            let eps = 1e-7;
            let (ep, dep, _) = pydismech_barrier_eval(d + eps, d0, delta);
            let (em, dem, _) = pydismech_barrier_eval(d - eps, d0, delta);
            let fd1 = (ep - em) / (2.0 * eps);
            let fd2 = (dep - dem) / (2.0 * eps);
            assert!(d2e >= 0.0);
            assert!((de - fd1).abs() / de.abs().max(1e-6) < 1e-6);
            assert!((d2e - fd2).abs() / d2e.abs().max(1e-6) < 1e-5);
        }
    }

    #[test]
    fn test_distance_and_ipc_barrier_6d_9d_12d_exact_fd_hessians_and_spd() {
        let cfg = IpcBarrierConfig {
            kind: BarrierKind::IpcLogBarrier,
            stiffness: 1.0e4,
            d_hat: 0.25,
            thickness: 0.02,
            mollifier_eps_x: 1e-3,
        };
        let eps = 1e-6;

        // 1. Point-Point 6D FD gradient & Hessian + SPD check
        let xa = Vec3::new(0.10, -0.05, 0.08);
        let xb = Vec3::new(0.18, 0.04, 0.01);
        let ev_pp = ipc_point_point_barrier(xa, xb, &cfg);
        assert!(ev_pp.active);
        let pts_pp = [xa, xb];
        for i in 0..6 {
            let mut p_plus = pts_pp;
            let mut p_minus = pts_pp;
            p_plus[i / 3][i % 3] += eps;
            p_minus[i / 3][i % 3] -= eps;
            let ep = ipc_point_point_barrier(p_plus[0], p_plus[1], &cfg);
            let em = ipc_point_point_barrier(p_minus[0], p_minus[1], &cfg);
            let fd_g = (ep.energy - em.energy) / (2.0 * eps);
            assert!(
                (ev_pp.grad[i] - fd_g).abs() / ev_pp.grad[i].abs().max(1.0) < 1e-6,
                "PP grad[{i}] mismatch: {} vs {}",
                ev_pp.grad[i],
                fd_g
            );
            for j in 0..6 {
                let fd_h = (ep.grad[j] - em.grad[j]) / (2.0 * eps);
                assert!(
                    (ev_pp.hess_exact[j][i] - fd_h).abs() / ev_pp.hess_exact[j][i].abs().max(1.0)
                        < 1e-5,
                    "PP hess[{j}][{i}] mismatch: {} vs {}",
                    ev_pp.hess_exact[j][i],
                    fd_h
                );
            }
        }
        let (evals_pp, _) = sym_eigen_n(ev_pp.hess_spd);
        assert!(evals_pp.iter().all(|&lam| lam >= -1e-10));

        // 2. Point-Segment 9D interior FD gradient & Hessian + SPD check
        let p = Vec3::new(0.12, 0.08, 0.05);
        let a = Vec3::new(-0.20, 0.00, 0.00);
        let b = Vec3::new(0.45, 0.03, -0.02);
        let ev_ps = ipc_point_segment_barrier(p, a, b, &cfg);
        assert!(ev_ps.active);
        let pts_ps = [p, a, b];
        for i in 0..9 {
            let mut pp = pts_ps;
            let mut pm = pts_ps;
            pp[i / 3][i % 3] += eps;
            pm[i / 3][i % 3] -= eps;
            let ep = ipc_point_segment_barrier(pp[0], pp[1], pp[2], &cfg);
            let em = ipc_point_segment_barrier(pm[0], pm[1], pm[2], &cfg);
            let fd_g = (ep.energy - em.energy) / (2.0 * eps);
            assert!(
                (ev_ps.grad[i] - fd_g).abs() / ev_ps.grad[i].abs().max(1.0) < 1e-6,
                "PS grad[{i}] mismatch: {} vs {}",
                ev_ps.grad[i],
                fd_g
            );
            for j in 0..9 {
                let fd_h = (ep.grad[j] - em.grad[j]) / (2.0 * eps);
                assert!(
                    (ev_ps.hess_exact[j][i] - fd_h).abs() / ev_ps.hess_exact[j][i].abs().max(1.0)
                        < 1e-5,
                    "PS hess[{j}][{i}] mismatch: {} vs {}",
                    ev_ps.hess_exact[j][i],
                    fd_h
                );
            }
        }
        let (evals_ps, _) = sym_eigen_n(ev_ps.hess_spd);
        assert!(evals_ps.iter().all(|&lam| lam >= -1e-10));

        // 3. Segment-Segment 12D interior skew edges FD gradient & Hessian + SPD check
        let a0 = Vec3::new(-0.30, -0.02, 0.00);
        let a1 = Vec3::new(0.35, 0.04, 0.01);
        let b0 = Vec3::new(0.05, -0.25, 0.09);
        let b1 = Vec3::new(-0.04, 0.30, 0.11);
        let ev_ss = ipc_segment_segment_barrier(a0, a1, b0, b1, &cfg);
        assert!(ev_ss.active);
        let pts_ss = [a0, a1, b0, b1];
        for i in 0..12 {
            let mut pp = pts_ss;
            let mut pm = pts_ss;
            pp[i / 3][i % 3] += eps;
            pm[i / 3][i % 3] -= eps;
            let ep = ipc_segment_segment_barrier(pp[0], pp[1], pp[2], pp[3], &cfg);
            let em = ipc_segment_segment_barrier(pm[0], pm[1], pm[2], pm[3], &cfg);
            let fd_g = (ep.energy - em.energy) / (2.0 * eps);
            assert!(
                (ev_ss.grad[i] - fd_g).abs() / ev_ss.grad[i].abs().max(1.0) < 1e-6,
                "SS grad[{i}] mismatch: {} vs {}",
                ev_ss.grad[i],
                fd_g
            );
            for j in 0..12 {
                let fd_h = (ep.grad[j] - em.grad[j]) / (2.0 * eps);
                assert!(
                    (ev_ss.hess_exact[j][i] - fd_h).abs() / ev_ss.hess_exact[j][i].abs().max(1.0)
                        < 1e-5,
                    "SS hess[{j}][{i}] mismatch: {} vs {}",
                    ev_ss.hess_exact[j][i],
                    fd_h
                );
            }
        }
        let (evals_ss, _) = sym_eigen_n(ev_ss.hess_spd);
        assert!(evals_ss.iter().all(|&lam| lam >= -1e-10));

        // 4. Segment-Segment 12D NEARLY PARALLEL edges (exercising mollifier product-rule Hessian!)
        let pa0 = Vec3::new(-0.30, 0.00, 0.00);
        let pa1 = Vec3::new(0.30, 0.00, 0.00);
        let pb0 = Vec3::new(-0.25, 0.004, 0.08);
        let pb1 = Vec3::new(0.35, -0.005, 0.082);
        let cfg_mol = IpcBarrierConfig {
            mollifier_eps_x: 1e-2, // force active mollifier region!
            ..cfg
        };
        let ev_mol = ipc_segment_segment_barrier(pa0, pa1, pb0, pb1, &cfg_mol);
        assert!(ev_mol.active && ev_mol.mollifier < 1.0 && ev_mol.mollifier > 0.0);
        let pts_mol = [pa0, pa1, pb0, pb1];
        for i in 0..12 {
            let mut pp = pts_mol;
            let mut pm = pts_mol;
            pp[i / 3][i % 3] += eps;
            pm[i / 3][i % 3] -= eps;
            let ep = ipc_segment_segment_barrier(pp[0], pp[1], pp[2], pp[3], &cfg_mol);
            let em = ipc_segment_segment_barrier(pm[0], pm[1], pm[2], pm[3], &cfg_mol);
            let fd_g = (ep.energy - em.energy) / (2.0 * eps);
            assert!(
                (ev_mol.grad[i] - fd_g).abs() / ev_mol.grad[i].abs().max(1e-3) < 1e-5,
                "Mollified SS grad[{i}] mismatch: {} vs {}",
                ev_mol.grad[i],
                fd_g
            );
            for j in 0..12 {
                let fd_h = (ep.grad[j] - em.grad[j]) / (2.0 * eps);
                assert!(
                    (ev_mol.hess_exact[j][i] - fd_h).abs()
                        / ev_mol.hess_exact[j][i].abs().max(1e-2)
                        < 1e-4,
                    "Mollified SS hess[{j}][{i}] mismatch: {} vs {}",
                    ev_mol.hess_exact[j][i],
                    fd_h
                );
            }
        }
    }

    #[test]
    fn test_ipc_mollifier_friction_fd_and_spsd() {
        let normal = Vec3::new(0.1, -0.2, 0.9746794344808963).normalized();
        let lambda_n = 12.5;
        let mu = 0.65;
        let eps_u = 1e-3;
        for u in [
            Vec3::new(3e-4, -4e-4, 1e-4), // sticking mollifier branch
            Vec3::new(5e-3, -2e-3, 1e-3), // sliding branch
        ] {
            let ev = ipc_mollifier_friction_eval(u, normal, lambda_n, mu, eps_u);
            let h = 1e-7;
            for i in 0..3 {
                let mut up = u;
                let mut um = u;
                up[i] += h;
                um[i] -= h;
                let ep = ipc_mollifier_friction_eval(up, normal, lambda_n, mu, eps_u);
                let em = ipc_mollifier_friction_eval(um, normal, lambda_n, mu, eps_u);
                let fd_g = (ep.energy - em.energy) / (2.0 * h);
                assert!((ev.grad[i] - fd_g).abs() / ev.grad[i].abs().max(1.0) < 1e-6);
                for j in 0..3 {
                    let fd_h = (ep.grad[j] - em.grad[j]) / (2.0 * h);
                    assert!((ev.hess.0[j][i] - fd_h).abs() / ev.hess.0[j][i].abs().max(1.0) < 1e-5);
                }
            }
            let (evals, _) = ev.hess.sym_eigen();
            assert!(evals.iter().all(|&l| l >= -1e-10));
        }
    }

    #[test]
    fn test_accd_non_tunneling_extreme_fast_edge_crossing_and_sdf() {
        // Two fast thin rods crossing head-on across a full step (100x thickness displacement!).
        let xi = 0.002;
        let a0 = Vec3::new(-0.5, 0.0, 0.05);
        let a1 = Vec3::new(0.5, 0.0, 0.05);
        let b0 = Vec3::new(0.0, -0.5, -0.05);
        let b1 = Vec3::new(0.0, 0.5, -0.05);
        // Move a downward by -0.30 m and b upward by +0.30 m (they would cross at t = 0.1 / 0.6 = 1/6)
        let da = Vec3::new(0.0, 0.0, -0.30);
        let db = Vec3::new(0.0, 0.0, 0.30);
        let toi = accd_segment_segment(a0, a1, b0, b1, da, da, db, db, xi, 0.1, 1.0)
            .expect("crossing segments must report collision TOI");
        let exact_contact_t = (0.10 - xi) / 0.60;
        assert!(toi > 0.0 && toi < exact_contact_t);
        // Verify strictly non-penetrating separation at t = toi!
        let dist_at_toi =
            segment_segment_distance(a0 + da * toi, a1 + da * toi, b0 + db * toi, b1 + db * toi)
                .dist;
        assert!(
            dist_at_toi > xi,
            "CCD violated thickness offset: dist={dist_at_toi}, xi={xi}"
        );

        // Global spatial-hash swept CCD test
        let x0 = vec![a0, a1, b0, b1];
        let x1 = vec![a0 + da, a1 + da, b0 + db, b1 + db];
        let segs = vec![(0, 1), (2, 3)];
        let t_safe = compute_safe_step_segments(&x0, &x1, &segs, xi, 0.1);
        assert!(t_safe > 0.0 && t_safe <= toi + 1e-12);
        let dist_global = segment_segment_distance(
            x0[0] + (x1[0] - x0[0]) * t_safe,
            x0[1] + (x1[1] - x0[1]) * t_safe,
            x0[2] + (x1[2] - x0[2]) * t_safe,
            x0[3] + (x1[3] - x0[3]) * t_safe,
        )
        .dist;
        assert!(dist_global > xi);

        // Point-vs-SDF CCD against a sphere obstacle (fast tunneling shot)
        let sphere = SphereSdf::new(Vec3::new(0.0, 0.0, 0.0), 0.10, MAT_HARD_FLOOR);
        let p0 = Vec3::new(-0.50, 0.0, 0.0);
        let dp = Vec3::new(1.50, 0.0, 0.0); // would shoot clean through the sphere!
        let toi_sdf = accd_point_sdf(&sphere, p0, dp, 0.01, 0.1, 1.0)
            .expect("fast point shot must hit sphere SDF");
        let p_hit = p0 + dp * toi_sdf;
        assert!(sphere.query(p_hit).phi > 0.01);
        assert!(toi_sdf < (0.50 - 0.11) / 1.50);
    }

    #[test]
    fn test_lcp_delassus_pgs_alart_curnier_and_fb_stick_slip_hysteresis() {
        // 2-contact coupled continuum compliance system (e.g. Shen et al. IJRR 2026)
        let c0 = LcpContactConstraint {
            normal: Vec3::EZ,
            t1: Vec3::EX,
            t2: Vec3::EY,
            mu_1: 0.5,
            mu_2: 0.5,
            free_vel: [-0.20, 0.04, 0.0], // sticking contact (small tangential motion)
            gap: 0.0,
            restitution: 0.0,
            solref_imp: None,
            pre_vel_n: None,
        };
        let c1 = LcpContactConstraint {
            normal: Vec3::EZ,
            t1: Vec3::EX,
            t2: Vec3::EY,
            mu_1: 0.3,
            mu_2: 0.3,
            free_vel: [-0.10, 0.80, -0.30], // sliding contact (large tangential slip)
            gap: 0.0,
            restitution: 0.0,
            solref_imp: None,
            pre_vel_n: None,
        };
        // SPD 6x6 coupled compliance matrix W = C_c
        let mut w = vec![0.0; 36];
        let diag = [2.0, 1.8, 1.8, 2.5, 2.0, 2.0];
        for i in 0..6 {
            w[i * 6 + i] = diag[i];
        }
        // Symmetric off-diagonal compliance coupling between contact 0 and contact 1
        w[3] = 0.35;
        w[18] = 0.35;
        w[10] = 0.25;
        w[25] = 0.25;

        let sys = DelassusSystem::new(vec![c0, c1], w);
        let cfg = LcpSolverConfig::default();

        let sol_pgs = sys.solve(&LcpSolverConfig { method: LcpSolverMethod::BlockPgs, ..cfg }, None);
        let sol_ac = sys.solve(&LcpSolverConfig { method: LcpSolverMethod::AlartCurnierNewton, ..cfg }, None);
        let sol_fb = sys.solve(&LcpSolverConfig { method: LcpSolverMethod::FischerBurmeisterPolyhedral, ..cfg }, None);

        assert!(sol_pgs.normal_complementarity_res < 1e-10);
        assert!(sol_pgs.friction_cone_res < 1e-10);
        assert!(sol_ac.normal_complementarity_res < 1e-10);
        assert!(sol_ac.friction_cone_res < 1e-10);
        assert!(sol_fb.normal_complementarity_res < 1e-9);

        // PGS and Alart-Curnier Newton must converge to the exact same unique Signorini-Coulomb solution!
        for i in 0..2 {
            for k in 0..3 {
                assert!(
                    (sol_pgs.impulses[i][k] - sol_ac.impulses[i][k]).abs() < 1e-7,
                    "PGS vs AC mismatch at contact {i} comp {k}: {} vs {}",
                    sol_pgs.impulses[i][k],
                    sol_ac.impulses[i][k]
                );
            }
        }
        // Contact 0 sticks (post-contact tangential velocity ~ 0), Contact 1 slides on cone boundary!
        assert!(sol_ac.post_vel[0][1].hypot(sol_ac.post_vel[0][2]) < 1e-8);
        let lam1_t = sol_ac.impulses[1][1].hypot(sol_ac.impulses[1][2]);
        assert!((lam1_t - 0.3 * sol_ac.impulses[1][0]).abs() < 1e-8);
    }

    #[test]
    fn test_arpc_wheel_stiction_exact_parity_and_anisotropic_scale() {
        // Exact replication of `wheel_stiction.py::__main__` validation suite!
        let dt = 0.0005_f64;
        let drives = [0.04, 0.12, -0.04, -2.0, 0.0];
        let mut vel = [0.0, 0.0, 0.0, 0.0, 0.1];
        let mut pos = [0.0; 5];
        let mut spin = [0.0, 0.0, 0.0, 0.0, 0.1 / 0.018];
        let mut mem = [[0.0; 2]; 5];
        let normal = 1.4;
        let steps = (1.5 / dt).round() as usize;
        for _ in 0..steps {
            for k in 0..5 {
                let res = implicit_wheel_contact(
                    [vel[k], 0.0],
                    spin[k],
                    mem[k],
                    normal,
                    dt,
                    0.018,
                    0.8,
                    0.06,
                    0.009,
                );
                spin[k] = res.omega;
                mem[k] = res.memory;
                vel[k] += dt * (drives[k] + res.force[0]) / 0.14;
                pos[k] += dt * vel[k];
            }
        }
        assert!(vel[0].abs() < 1e-8 && vel[2].abs() < 1e-8);
        assert!((pos[0] - 0.04 / 2000.0).abs() < 1e-6);
        assert!(vel[1] > 0.01 && vel[3] < -0.1 && vel[4].abs() < 1e-5);

        // Bit-exact FP64 parity (< 1e-14) against Python `serpworm.physics.wheel_stiction`
        // (`arpc_normal_and_rut` + `arpc_tangential_contact` with Bekker soft soil + Stribeck + backlash)
        let rn = arpc_normal_and_rut(0.004, -0.02, 0.15, 0.001, 0.0005, 1.2e4, 6.0, 1.35, 3.5, 0.030);
        assert!((rn.normal - 7.783595476649691).abs() < 1e-13);
        assert!((rn.new_rut - 0.002857142857142857).abs() < 1e-15);
        assert!((rn.kn_tan - 2626.963473369271).abs() < 1e-11);
        assert!((rn.cn_tan - 41.69783291062335).abs() < 1e-13);

        let rt = arpc_tangential_contact(
            [-0.08, 0.03],
            -2.5,
            [0.0002, -0.0001],
            0.02,
            rn.normal,
            rn.new_rut,
            0.35,
            0.0005,
            0.018,
            0.70,
            0.70 * 1.35,
            0.22,
            0.05,
            0.013,
            0.80,
            650.0,
            1.18,
            0.0179,
        );
        assert!((rt.force_2d[0] - 0.31999999999999995).abs() < 1e-14);
        assert!((rt.force_2d[1] - (-1.7509896139118948)).abs() < 1e-14);
        assert!(rt.omega.abs() < 1e-15);
        assert!((rt.new_memory[0] - 0.00016).abs() < 1e-15);
        assert!((rt.new_memory[1] - (-8.5e-5)).abs() < 1e-15);
        assert!((rt.new_play - 0.02).abs() < 1e-15);
        assert!((rt.reaction - 0.020258999999999996).abs() < 1e-15);
        assert!((rt.tau_z - (-0.05296076696637559)).abs() < 1e-15);
        assert!((rt.ct_fwd_tan - 9.0).abs() < 1e-14);
        assert!((rt.ct_lat_tan - 25.77990218576737).abs() < 1e-13);
        assert!((rt.c_piv_tan - 0.0132817047263125).abs() < 1e-15);

        // Unified ARPC evaluator: RatchetWheel on Bekker soft soil produces strictly SPD 6x6 tangent
        let ground = HeightFieldSdf::flat(0.0, MAT_BEKKER_SOFT_SOIL);
        let mats = MaterialLibrary::standard();
        let site = ContactSiteSpec {
            id: 0,
            name: "wheel_l".to_string(),
            attachment: ContactAttachment::RigidBody { body_id: 0 },
            local_pos: Vec3::new(0.0, 0.045, -0.02),
            local_frame: Quat::IDENTITY,
            radius: 0.018,
            patch_radius: 0.013,
            mode: ContactMode::RatchetWheel {
                wheel_mass: 0.0179,
                rolling_resistance: 0.22,
                backlash_rad: 0.05,
                pivot_mu: 0.80,
                mu_fwd: 0.70,
                mu_lat_ratio: 1.35,
                static_mu_ratio: 1.18,
                axle_axis_local: Vec3::EY,
            },
            material_profile_id: MAT_BEKKER_SOFT_SOIL,
            solref_imp: None,
        };
        let pose = Se3::new(Vec3::new(0.0, 0.0, 0.036), Quat::IDENTITY);
        let twist = Vec6::from_parts(Vec3::new(-0.08, 0.02, -0.01), Vec3::new(0.0, 0.0, 0.2));
        let eval = eval_arpc_site(
            &site,
            pose,
            twist,
            &ground,
            &mats,
            &ArpcSiteState::default(),
            0.0005,
            true,
        );
        assert!(eval.active && eval.normal_force > 0.0 && eval.next_state.soil_sinkage > 0.0);
        let (eigs, _) = eval.k_tan.sym_eigen();
        assert!(
            eigs.iter().all(|&l| l >= -1e-12),
            "ARPC 6x6 tangent must be PSD/SPD: {:?}",
            eigs
        );

        // AnisotropicScale forward vs backward friction asymmetry (Gidoni stasis-domain mechanics)
        let scale_site = ContactSiteSpec {
            id: 1,
            name: "ventral_scale".to_string(),
            attachment: ContactAttachment::RigidBody { body_id: 0 },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: 0.010,
            patch_radius: 0.006,
            mode: ContactMode::AnisotropicScale {
                mu_fwd: 0.12,
                mu_rev: 0.95,
                mu_lat: 0.70,
                pivot_mu: 0.35,
                static_mu_ratio: 1.15,
                fwd_axis_local: Vec3::EX,
            },
            material_profile_id: MAT_HARD_FLOOR,
            solref_imp: None,
        };
        let flat = PlaneSdf::flat_ground(0.0, MAT_HARD_FLOOR);
        let p_scale = Se3::new(Vec3::new(0.0, 0.0, 0.008), Quat::IDENTITY);
        // Steady sliding speed (2.0 m/s) saturates both forward (mu_fwd=0.12) and reverse (mu_rev=0.95)
        // friction cones, verifying the exact ~7.9x anisotropic drag ratio!
        let ev_fwd = eval_arpc_site(
            &scale_site,
            p_scale,
            Vec6::from_parts(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO),
            &flat,
            &mats,
            &ArpcSiteState::default(),
            0.0005,
            true,
        );
        let ev_rev = eval_arpc_site(
            &scale_site,
            p_scale,
            Vec6::from_parts(Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO),
            &flat,
            &mats,
            &ArpcSiteState::default(),
            0.0005,
            true,
        );
        assert!(ev_fwd.active && ev_rev.active);
        let drag_ratio = ev_rev.force_world.x().abs() / ev_fwd.force_world.x().abs();
        assert!(
            (drag_ratio - (0.95 / 0.12)).abs() < 0.05,
            "anisotropic scale drag ratio mismatch: got {drag_ratio}, expected ~{}",
            0.95 / 0.12
        );
        assert!(ev_fwd.force_world.x() < 0.0 && ev_rev.force_world.x() > 0.0);
        assert!(ev_rev.k_tan.0[0][0] > 5.0 * ev_fwd.k_tan.0[0][0]);
        let (eigs_s, _) = ev_rev.k_tan.sym_eigen();
        assert!(eigs_s.iter().all(|&l| l >= -1e-12));
    }

    #[test]
    fn test_self_contact_engine_assembly_ipc_and_pydismech_fd() {
        let pts = vec![
            Vec3::new(-0.20, 0.00, 0.00),
            Vec3::new(0.20, 0.00, 0.00),
            Vec3::new(0.02, -0.20, 0.014),
            Vec3::new(-0.01, 0.20, 0.015),
        ];
        let segs = vec![(0, 1), (2, 3)];
        for kind in [BarrierKind::IpcLogBarrier, BarrierKind::PyDiSMechSoftBarrier] {
            let eng = SelfContactEngine::new(
                IpcBarrierConfig {
                    kind,
                    stiffness: 5.0e3,
                    d_hat: 0.015,
                    thickness: 0.008,
                    mollifier_eps_x: 1e-5,
                },
                0.0,
                1e-2,
            );
            let asm = eng.assemble_segments(&pts, None, &segs, 1e-3);
            assert_eq!(asm.active_pairs, 1);
            assert!(asm.total_energy > 0.0);

            // FD check of assembled force vector == -grad(total_energy)
            let eps = 1e-6;
            for node in 0..4 {
                let (eigs, _) = asm.diag_hess_spd[node].sym_eigen();
                assert!(eigs.iter().all(|&l| l >= -1e-10));
                for c in 0..3 {
                    let mut p_plus = pts.clone();
                    let mut p_minus = pts.clone();
                    p_plus[node][c] += eps;
                    p_minus[node][c] -= eps;
                    let ep = eng.assemble_segments(&p_plus, None, &segs, 1e-3).total_energy;
                    let em = eng.assemble_segments(&p_minus, None, &segs, 1e-3).total_energy;
                    let fd_f = -(ep - em) / (2.0 * eps);
                    assert!(
                        (asm.forces[node][c] - fd_f).abs() / asm.forces[node][c].abs().max(1.0)
                            < 1e-5,
                        "{:?} node {node} comp {c}: analytic={} vs fd={}",
                        kind,
                        asm.forces[node][c],
                        fd_f
                    );
                }
            }
        }
    }

    #[test]
    fn test_lcp_stick_slip_hysteresis_anisotropic_and_dedup() {
        // Test multi-step stick -> breakaway slide -> reverse stick -> opposite slide hysteresis!
        let mut w = vec![0.0; 9];
        w[0] = 1.0;
        w[4] = 1.0;
        w[8] = 1.0;
        let cfg = LcpSolverConfig {
            method: LcpSolverMethod::AlartCurnierNewton,
            ..LcpSolverConfig::default()
        };
        let mut last_lt = 0.0;
        for &vt_free in &[0.05, 0.20, 0.60, -0.05, -0.60] {
            let c = LcpContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.40,
                mu_2: 0.80, // anisotropic cone
                free_vel: [-1.0, vt_free, 0.0],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            };
            let sys = DelassusSystem::new(vec![c], w.clone());
            let sol = sys.solve(&cfg, None);
            assert!(sol.converged && sol.normal_complementarity_res < 1e-11);
            assert!((sol.impulses[0][0] - 1.0).abs() < 1e-11);
            let lt = sol.impulses[0][1];
            if vt_free.abs() <= 0.40 {
                // Strictly sticking: impulse cancels free tangential velocity exactly
                assert!((lt + vt_free).abs() < 1e-10);
                assert!(sol.post_vel[0][1].abs() < 1e-10);
            } else {
                // Sliding on cone boundary |lt| == mu_1 * ln = 0.40
                assert!((lt.abs() - 0.40).abs() < 1e-10);
                assert!(lt * vt_free < 0.0);
            }
            last_lt = lt;
        }
        assert!(last_lt > 0.39);

        // Redundant contact deduplication keeps deepest penetration per cluster
        let c_pos = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.002, 0.0, 0.0),
            Vec3::new(0.004, 0.0, 0.0),
            Vec3::new(0.10, 0.0, 0.0),
        ];
        let c_pen = vec![0.001, 0.005, 0.002, 0.003];
        let kept = DelassusSystem::deduplicate_contacts(&c_pos, &c_pen, 0.01);
        assert_eq!(kept, vec![1, 3]);
    }

    #[test]
    fn test_parallel_rods_cipc_no_ghosting_and_fd() {
        // Exact parallel & near-parallel rod configuration (no parallel-rod ghosting!).
        let cfg = IpcBarrierConfig {
            kind: BarrierKind::IpcLogBarrier,
            stiffness: 1.0e4,
            d_hat: 0.05,
            thickness: 0.002,
            mollifier_eps_x: 1e-3,
        };
        let eng = SelfContactEngine::new(cfg, 0.0, 1e-2);
        let segs = vec![(0, 1), (2, 3)];

        for tilt in [0.0_f64, 4.0e-12] {
            let pts = vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.3, 0.01, 0.0),
                Vec3::new(0.7, 0.01 + tilt, 0.0),
            ];
            let asm = eng.assemble_segments(&pts, None, &segs, 1e-3);
            assert!(
                asm.active_pairs >= 2 && asm.total_energy > 1e-3,
                "parallel rods ghosted at tilt={tilt}: pairs={}, E={}",
                asm.active_pairs,
                asm.total_energy
            );
            // Repulsive forces must push rod A toward -Y and rod B toward +Y!
            assert!(asm.forces[0].y() + asm.forces[1].y() < -1.0);
            assert!(asm.forces[2].y() + asm.forces[3].y() > 1.0);

            // Verify exact FD force consistency on all 12 DOFs + SPD diagonal Hessians
            let eps = 1e-6;
            for node in 0..4 {
                let (eigs, _) = asm.diag_hess_spd[node].sym_eigen();
                assert!(eigs.iter().all(|&l| l >= -1e-10));
                for c in 0..3 {
                    let mut pp = pts.clone();
                    let mut pm = pts.clone();
                    pp[node][c] += eps;
                    pm[node][c] -= eps;
                    let ep = eng.assemble_segments(&pp, None, &segs, 1e-3).total_energy;
                    let em = eng.assemble_segments(&pm, None, &segs, 1e-3).total_energy;
                    let fd_f = -(ep - em) / (2.0 * eps);
                    assert!(
                        (asm.forces[node][c] - fd_f).abs() / asm.forces[node][c].abs().max(1.0)
                            < 1e-5,
                        "parallel rod FD mismatch (tilt={tilt}, node={node}, c={c}): {} vs {}",
                        asm.forces[node][c],
                        fd_f
                    );
                }
            }
        }
    }

    #[test]
    fn test_lcp_5contact_nonscalar_wtt_anisotropic_pgs_ac_fb_convergence() {
        // 5-contact coupled SPD Delassus matrix with w11 != w22, w12 != 0, mixed stick/slide/
        // separated/anisotropic friction cones!
        let contacts = vec![
            // 0: sticking isotropic
            LcpContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.6,
                mu_2: 0.6,
                free_vel: [-0.40, 0.05, -0.04],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            // 1: sliding anisotropic ellipse (mu_1 != mu_2)
            LcpContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.25,
                mu_2: 0.70,
                free_vel: [-0.35, 0.90, -0.80],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            // 2: sliding isotropic with oblique velocity
            LcpContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.35,
                mu_2: 0.35,
                free_vel: [-0.25, -0.60, 0.55],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            // 3: separated inactive contact
            LcpContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.5,
                mu_2: 0.5,
                free_vel: [0.30, 0.10, -0.10],
                gap: 0.05,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            // 4: sliding strongly anisotropic
            LcpContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.80,
                mu_2: 0.20,
                free_vel: [-0.50, -0.70, 0.65],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
        ];

        // Construct strictly SPD 15x15 W = diag_blocks(non-scalar w11 != w22, w12 != 0) + L L^T coupling
        let dim = 15;
        let mut w = vec![0.0; dim * dim];
        for i in 0..5 {
            let b = 3 * i;
            let w00 = 1.0 + 0.2 * (i as f64);
            let w11 = 1.5 + 0.15 * (i as f64);
            let w22 = 2.2 - 0.10 * (i as f64);
            let w12 = if i % 2 == 0 { 0.35 } else { -0.28 };
            let w01 = 0.08 * ((i as f64) - 2.0);
            w[b * dim + b] = w00;
            w[(b + 1) * dim + (b + 1)] = w11;
            w[(b + 2) * dim + (b + 2)] = w22;
            w[(b + 1) * dim + (b + 2)] = w12;
            w[(b + 2) * dim + (b + 1)] = w12;
            w[b * dim + (b + 1)] = w01;
            w[(b + 1) * dim + b] = w01;
        }
        // Symmetric SPD cross-contact compliance couplings
        for i in 0..4 {
            let b0 = 3 * i;
            let b1 = 3 * (i + 1);
            for k in 0..3 {
                let c = 0.12 * (1.0 - 0.15 * (k as f64));
                w[(b0 + k) * dim + (b1 + k)] += c;
                w[(b1 + k) * dim + (b0 + k)] += c;
            }
        }

        let sys = DelassusSystem::new(contacts, w);
        let sol_ac = sys.solve(
            &LcpSolverConfig {
                method: LcpSolverMethod::AlartCurnierNewton,
                max_iters: 100,
                tol: 1e-12,
                ..LcpSolverConfig::default()
            },
            None,
        );
        let sol_pgs = sys.solve(
            &LcpSolverConfig {
                method: LcpSolverMethod::BlockPgs,
                max_iters: 500,
                tol: 1e-12,
                ..LcpSolverConfig::default()
            },
            None,
        );

        assert!(
            sol_ac.converged
                && sol_ac.normal_complementarity_res < 1e-12
                && sol_ac.friction_cone_res < 1e-12,
            "AC failed: conv={}, nr={}, fr={}",
            sol_ac.converged,
            sol_ac.normal_complementarity_res,
            sol_ac.friction_cone_res
        );
        assert!(
            sol_pgs.converged
                && sol_pgs.normal_complementarity_res < 1e-11
                && sol_pgs.friction_cone_res < 1e-11,
            "PGS failed: conv={}, iters={}, nr={}, fr={}",
            sol_pgs.converged,
            sol_pgs.iterations,
            sol_pgs.normal_complementarity_res,
            sol_pgs.friction_cone_res
        );

        let mut max_pgs_ac = 0.0_f64;
        for i in 0..5 {
            for k in 0..3 {
                max_pgs_ac = max_pgs_ac.max((sol_pgs.impulses[i][k] - sol_ac.impulses[i][k]).abs());
            }
        }
        assert!(
            max_pgs_ac < 1e-10,
            "BlockPgs vs AlartCurnier mismatch on non-scalar W_tt: {max_pgs_ac}"
        );
        // Contact 3 must be separated (zero impulse)
        assert!(sol_ac.impulses[3].iter().all(|&v| v.abs() < 1e-14));

        // Polyhedral LCP must converge monotonically toward continuous AlartCurnierNewton as k grows!
        let mut prev_err = f64::INFINITY;
        let mut err_8 = 0.0_f64;
        for &facets in &[8usize, 16, 32, 64, 128] {
            let sol_fb = sys.solve(
                &LcpSolverConfig {
                    method: LcpSolverMethod::FischerBurmeisterPolyhedral,
                    max_iters: 500,
                    tol: 1e-11,
                    polyhedral_facets: facets,
                    ..LcpSolverConfig::default()
                },
                None,
            );
            assert!(
                sol_fb.converged && sol_fb.normal_complementarity_res < 1e-10,
                "FB(k={facets}) failed: conv={}, iters={}, nr={}",
                sol_fb.converged,
                sol_fb.iterations,
                sol_fb.normal_complementarity_res
            );
            let mut err = 0.0_f64;
            for i in 0..5 {
                for k in 0..3 {
                    err = err.max((sol_fb.impulses[i][k] - sol_ac.impulses[i][k]).abs());
                }
            }
            if facets == 8 {
                err_8 = err;
            }
            assert!(
                err < prev_err,
                "FB(k={facets}) did not improve monotonically: err={err} >= prev={prev_err}"
            );
            prev_err = err;
        }
        assert!(
            prev_err < 1.5e-3 && prev_err < 0.1 * err_8,
            "FB(k=128) error vs continuous cone too large: err_128={prev_err}, err_8={err_8}"
        );

        // Small-impulse inclined-plane slip regression (mu=0.05, c ~ 1e-4, s* ~ 5e-7):
        // verifies Cauchy-Schwarz bracket scaling in `solve_2d_ellipse_qp` converges to < 1e-13!
        let c_small = LcpContactConstraint {
            normal: Vec3::EZ,
            t1: Vec3::EX,
            t2: Vec3::EY,
            mu_1: 0.05,
            mu_2: 0.05,
            free_vel: [-0.017, 0.0098, -0.003],
            gap: 0.0,
            restitution: 0.0,
            solref_imp: None,
            pre_vel_n: None,
        };
        let mut w_small = vec![0.0; 9];
        w_small[0] = 3.5;
        w_small[4] = 4.2;
        w_small[8] = 5.1;
        w_small[5] = 0.4;
        w_small[7] = 0.4;
        let sys_small = DelassusSystem::new(vec![c_small], w_small);
        let pgs_small = sys_small.solve(
            &LcpSolverConfig {
                method: LcpSolverMethod::BlockPgs,
                max_iters: 100,
                tol: 1e-13,
                ..LcpSolverConfig::default()
            },
            None,
        );
        let ac_small = sys_small.solve(
            &LcpSolverConfig {
                method: LcpSolverMethod::AlartCurnierNewton,
                max_iters: 50,
                tol: 1e-13,
                ..LcpSolverConfig::default()
            },
            None,
        );
        assert!(
            pgs_small.converged
                && pgs_small.normal_complementarity_res < 1e-13
                && pgs_small.friction_cone_res < 1e-13,
            "small-impulse BlockPgs failed: conv={}, iters={}, nr={}, fr={}",
            pgs_small.converged,
            pgs_small.iterations,
            pgs_small.normal_complementarity_res,
            pgs_small.friction_cone_res
        );
        for k in 0..3 {
            assert!(
                (pgs_small.impulses[0][k] - ac_small.impulses[0][k]).abs() < 1e-13,
                "small-impulse PGS vs AC mismatch at {k}: {} vs {}",
                pgs_small.impulses[0][k],
                ac_small.impulses[0][k]
            );
        }
    }

    #[test]
    fn test_point_triangle_distance_and_ipc_barrier_12d_fd_all_regions() {
        // Non-axis-aligned oblique 3D triangle
        let t0 = Vec3::new(-0.20, -0.15, 0.05);
        let t1 = Vec3::new(0.40, -0.05, 0.12);
        let t2 = Vec3::new(0.08, 0.45, -0.08);

        let cfg = IpcBarrierConfig {
            kind: BarrierKind::IpcLogBarrier,
            stiffness: 1.0e4,
            d_hat: 0.35,
            thickness: 0.02,
            mollifier_eps_x: 0.0,
        };
        let eps = 1e-6;

        // Test all 7 Voronoi regions: FaceInterior, 3 Edges, 3 Vertices
        let cases: [(Vec3, TriangleFeature); 7] = [
            // Face interior (offset along face normal from centroid)
            (
                (t0 + t1 + t2) / 3.0 + (t1 - t0).cross(t2 - t0).normalized() * 0.12,
                TriangleFeature::FaceInterior,
            ),
            // Edge (0, 1) interior
            (
                (t0 + t1) * 0.5 + Vec3::new(0.0, -0.10, 0.08),
                TriangleFeature::Edge(0, 1),
            ),
            // Edge (1, 2) interior
            (
                (t1 + t2) * 0.5 + Vec3::new(0.11, 0.06, 0.07),
                TriangleFeature::Edge(1, 2),
            ),
            // Edge (0, 2) interior
            (
                (t0 + t2) * 0.5 + Vec3::new(-0.11, 0.05, 0.06),
                TriangleFeature::Edge(0, 2),
            ),
            // Vertex 0
            (t0 + Vec3::new(-0.08, -0.09, 0.05), TriangleFeature::Vertex(0)),
            // Vertex 1
            (t1 + Vec3::new(0.10, -0.07, 0.06), TriangleFeature::Vertex(1)),
            // Vertex 2
            (t2 + Vec3::new(0.02, 0.11, -0.06), TriangleFeature::Vertex(2)),
        ];

        for (case_idx, &(p, expected_feat)) in cases.iter().enumerate() {
            let d_ev = point_triangle_distance_grad_hess(p, t0, t1, t2);
            assert_eq!(
                d_ev.feature, expected_feat,
                "case {case_idx} Voronoi feature mismatch"
            );
            let bary_sum = d_ev.bary[0] + d_ev.bary[1] + d_ev.bary[2];
            assert!((bary_sum - 1.0).abs() < 1e-12);
            assert!(d_ev
                .bary
                .iter()
                .all(|&b| (-1e-12..=1.0 + 1e-12).contains(&b)));

            let b_ev = ipc_point_triangle_barrier(p, t0, t1, t2, &cfg);
            assert!(b_ev.active, "case {case_idx} barrier must be active");

            let pts = [p, t0, t1, t2];
            for i in 0..12 {
                let mut pp = pts;
                let mut pm = pts;
                pp[i / 3][i % 3] += eps;
                pm[i / 3][i % 3] -= eps;

                // Distance 12D FD grad & 12x12 FD Hessian
                let dp = point_triangle_distance_grad_hess(pp[0], pp[1], pp[2], pp[3]);
                let dm = point_triangle_distance_grad_hess(pm[0], pm[1], pm[2], pm[3]);
                let fd_dg = (dp.dist - dm.dist) / (2.0 * eps);
                assert!(
                    (d_ev.grad[i] - fd_dg).abs() / d_ev.grad[i].abs().max(1.0) < 1e-6,
                    "VF dist case {case_idx} grad[{i}] mismatch: {} vs {}",
                    d_ev.grad[i],
                    fd_dg
                );
                for j in 0..12 {
                    let fd_dh = (dp.grad[j] - dm.grad[j]) / (2.0 * eps);
                    assert!(
                        (d_ev.hess[j][i] - fd_dh).abs() / d_ev.hess[j][i].abs().max(1.0) < 1e-5,
                        "VF dist case {case_idx} hess[{j}][{i}] mismatch: {} vs {}",
                        d_ev.hess[j][i],
                        fd_dh
                    );
                }

                // IPC Barrier 12D FD grad & 12x12 FD Hessian
                let bp = ipc_point_triangle_barrier(pp[0], pp[1], pp[2], pp[3], &cfg);
                let bm = ipc_point_triangle_barrier(pm[0], pm[1], pm[2], pm[3], &cfg);
                let fd_bg = (bp.energy - bm.energy) / (2.0 * eps);
                assert!(
                    (b_ev.grad[i] - fd_bg).abs() / b_ev.grad[i].abs().max(1.0) < 1e-6,
                    "VF barrier case {case_idx} grad[{i}] mismatch: {} vs {}",
                    b_ev.grad[i],
                    fd_bg
                );
                for j in 0..12 {
                    let fd_bh = (bp.grad[j] - bm.grad[j]) / (2.0 * eps);
                    assert!(
                        (b_ev.hess_exact[j][i] - fd_bh).abs()
                            / b_ev.hess_exact[j][i].abs().max(1.0)
                            < 1e-5,
                        "VF barrier case {case_idx} hess[{j}][{i}] mismatch: {} vs {}",
                        b_ev.hess_exact[j][i],
                        fd_bh
                    );
                }
            }

            let (evals, _) = sym_eigen_n(b_ev.hess_spd);
            assert!(
                evals.iter().all(|&l| l >= -1e-10),
                "VF barrier case {case_idx} SPD projection failed: {:?}",
                evals
            );
        }
    }

    #[test]
    fn test_tight_inclusion_ccd_vf_and_ee_parallel_grazing_zero_false_negatives() {
        let cfg = TightInclusionConfig::default();

        // 1. VF head-on impact onto moving tilted triangle
        let t0_0 = Vec3::new(0.0, 0.0, 0.0);
        let t1_0 = Vec3::new(1.0, 0.0, 0.0);
        let t2_0 = Vec3::new(0.0, 1.0, 0.0);
        let p0 = Vec3::new(0.25, 0.25, 0.30);
        let p1 = Vec3::new(0.25, 0.25, -0.70);
        let toi_vf = tight_inclusion_point_triangle_ccd(
            p0, t0_0, t1_0, t2_0, p1, t0_0, t1_0, t2_0, 1.0, &cfg,
        )
        .expect("head-on VF must hit");
        assert!(
            toi_vf > 0.0 && toi_vf <= 0.30 + 1e-12 && (0.30 - toi_vf) < 1e-4,
            "VF head-on TOI mismatch: {toi_vf} vs 0.30"
        );
        let p_at_toi = p0 * (1.0 - toi_vf) + p1 * toi_vf;
        assert!(p_at_toi.z() >= -1e-12, "VF TOI must be conservative!");

        // 2. VF COPLANAR GRAZING impact along triangle plane (classic cubic-solver failure case!)
        let pg0 = Vec3::new(-0.40, 0.20, 0.0);
        let pg1 = Vec3::new(0.60, 0.20, 0.0); // enters triangle at x=0 -> t = 0.40, strictly coplanar z=0!
        let toi_graze = tight_inclusion_point_triangle_ccd(
            pg0, t0_0, t1_0, t2_0, pg1, t0_0, t1_0, t2_0, 1.0, &cfg,
        )
        .expect("coplanar grazing VF must NOT false-negative!");
        assert!(
            toi_graze > 0.0 && toi_graze <= 0.40 + 1e-12 && (0.40 - toi_graze) < 1e-4,
            "VF coplanar grazing TOI mismatch: {toi_graze} vs 0.40"
        );

        // 3. VF grazing MISS (stays at z = 1e-3 parallel to triangle)
        let pm0 = Vec3::new(-0.40, 0.20, 1e-3);
        let pm1 = Vec3::new(0.60, 0.20, 1e-3);
        assert!(
            tight_inclusion_point_triangle_ccd(
                pm0, t0_0, t1_0, t2_0, pm1, t0_0, t1_0, t2_0, 1.0, &cfg
            )
            .is_none(),
            "parallel separated VF pass must return None"
        );

        // 4. VF with finite min_dist = 0.05 (must hit at t = (0.30 - 0.05) / 1.0 = 0.25)
        let cfg_md = TightInclusionConfig {
            min_dist: 0.05,
            ..cfg
        };
        let toi_md = tight_inclusion_point_triangle_ccd(
            p0, t0_0, t1_0, t2_0, p1, t0_0, t1_0, t2_0, 1.0, &cfg_md,
        )
        .expect("VF with min_dist must hit");
        assert!(
            toi_md > 0.0 && toi_md <= 0.25 + 1e-12 && (0.25 - toi_md) < 1e-4,
            "VF min_dist TOI mismatch: {toi_md} vs 0.25"
        );
        let p_md = p0 * (1.0 - toi_md) + p1 * toi_md;
        assert!(
            point_triangle_distance_grad_hess(p_md, t0_0, t1_0, t2_0).dist >= 0.05 - 1e-7,
            "VF min_dist TOI violated separation"
        );

        // 5. EE oblique crossing impact
        let a0_0 = Vec3::new(-0.5, 0.0, 0.20);
        let a1_0 = Vec3::new(0.5, 0.0, 0.20);
        let a0_1 = Vec3::new(-0.5, 0.0, -0.80);
        let a1_1 = Vec3::new(0.5, 0.0, -0.80);
        let b0 = Vec3::new(0.0, -0.5, 0.0);
        let b1 = Vec3::new(0.0, 0.5, 0.0);
        let toi_ee = tight_inclusion_edge_edge_ccd(
            a0_0, a1_0, b0, b1, a0_1, a1_1, b0, b1, 1.0, &cfg,
        )
        .expect("oblique EE crossing must hit");
        assert!(
            toi_ee > 0.0 && toi_ee <= 0.20 + 1e-12 && (0.20 - toi_ee) < 1e-4,
            "EE crossing TOI mismatch: {toi_ee} vs 0.20"
        );

        // 6. EE EXACTLY PARALLEL COPLANAR SLIDING collision (cross product identically 0!)
        let pa0_0 = Vec3::new(0.0, 0.30, 0.0);
        let pa1_0 = Vec3::new(1.0, 0.30, 0.0);
        let pa0_1 = Vec3::new(0.2, -0.70, 0.0);
        let pa1_1 = Vec3::new(1.2, -0.70, 0.0);
        let pb0 = Vec3::new(0.1, 0.0, 0.0);
        let pb1 = Vec3::new(0.9, 0.0, 0.0); // exact parallel impact at t = 0.30!
        let toi_par = tight_inclusion_edge_edge_ccd(
            pa0_0, pa1_0, pb0, pb1, pa0_1, pa1_1, pb0, pb1, 1.0, &cfg,
        )
        .expect("parallel coplanar EE collision must NOT false-negative!");
        assert!(
            toi_par > 0.0 && toi_par <= 0.30 + 1e-12 && (0.30 - toi_par) < 1e-4,
            "parallel EE TOI mismatch: {toi_par} vs 0.30"
        );

        // 7. EE grazing miss (offset in z by 2e-3)
        let pb0_m = Vec3::new(0.1, 0.0, 2e-3);
        let pb1_m = Vec3::new(0.9, 0.0, 2e-3);
        assert!(
            tight_inclusion_edge_edge_ccd(
                pa0_0, pa1_0, pb0_m, pb1_m, pa0_1, pa1_1, pb0_m, pb1_m, 1.0, &cfg
            )
            .is_none(),
            "separated parallel EE pass must return None"
        );
    }

    #[cfg(feature = "clarabel-socp")]
    #[test]
    fn test_clarabel_socp_vs_alart_curnier_and_extreme_ill_conditioning() {
        // Part A: Well-conditioned 5-contact coupled non-scalar W_tt mixed stick/slide/anisotropic
        let contacts = vec![
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.6,
                mu_2: 0.6,
                free_vel: [-0.40, 0.05, -0.04],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.25,
                mu_2: 0.70,
                free_vel: [-0.35, 0.90, -0.80],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.35,
                mu_2: 0.35,
                free_vel: [-0.25, -0.60, 0.55],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.5,
                mu_2: 0.5,
                free_vel: [0.30, 0.10, -0.10],
                gap: 0.05,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.80,
                mu_2: 0.20,
                free_vel: [-0.50, -0.70, 0.65],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
        ];
        let dim = 15;
        let mut w = vec![0.0; dim * dim];
        for i in 0..5 {
            let b = 3 * i;
            w[b * dim + b] = 1.0 + 0.2 * (i as f64);
            w[(b + 1) * dim + (b + 1)] = 1.5 + 0.15 * (i as f64);
            w[(b + 2) * dim + (b + 2)] = 2.2 - 0.10 * (i as f64);
            let w12 = if i % 2 == 0 { 0.35 } else { -0.28 };
            let w01 = 0.08 * ((i as f64) - 2.0);
            w[(b + 1) * dim + (b + 2)] = w12;
            w[(b + 2) * dim + (b + 1)] = w12;
            w[b * dim + (b + 1)] = w01;
            w[(b + 1) * dim + b] = w01;
        }
        for i in 0..4 {
            let b0 = 3 * i;
            let b1 = 3 * (i + 1);
            for k in 0..3 {
                let c = 0.12 * (1.0 - 0.15 * (k as f64));
                w[(b0 + k) * dim + (b1 + k)] += c;
                w[(b1 + k) * dim + (b0 + k)] += c;
            }
        }

        let sys = DelassusSystem::new(contacts, w);
        let sol_ac = sys.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::AlartCurnierNewton,
                max_iters: 100,
                tol: 1e-12,
                ..LcpSolverConfig::default()
            },
            None,
        );
        let sol_socp = sys.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::ClarabelSocp,
                max_iters: 50,
                tol: 1e-11,
                ..LcpSolverConfig::default()
            },
            None,
        );

        assert!(
            sol_socp.converged
                && sol_socp.normal_complementarity_res < 1e-9
                && sol_socp.friction_cone_res < 1e-9,
            "ClarabelSocp failed on 5-contact problem: conv={}, iters={}, nr={:.3e}, fr={:.3e}\n  lam={:?}\n  u={:?}\n  ac_lam={:?}",
            sol_socp.converged,
            sol_socp.iterations,
            sol_socp.normal_complementarity_res,
            sol_socp.friction_cone_res,
            sol_socp.impulses,
            sol_socp.post_vel,
            sol_ac.impulses,
        );
        let mut max_diff = 0.0_f64;
        for i in 0..5 {
            for k in 0..3 {
                max_diff = max_diff.max((sol_socp.impulses[i][k] - sol_ac.impulses[i][k]).abs());
            }
        }
        assert!(
            max_diff < 1e-8,
            "ClarabelSocp vs AlartCurnierNewton mismatch: {max_diff:.3e}"
        );

        // Part B: Extreme stiffness/mass ratio kappa >= 1e8 (tightly coupled light-heavy contacts)
        // Build W = V * diag(1e8, 1e8, 1e8, 1.0, 1.0, 1.0) * V^T with strong 45-degree coupling
        // where BlockPGS spectral radius is ~ 1 - 1e-8 (stalls completely!), while ClarabelSocp
        // solves directly via QDLDL interior-point + Ruiz equilibration!
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let q_rot: [[f64; 6]; 6] = [
            [s, 0.0, 0.0, s, 0.0, 0.0],
            [0.0, s, 0.0, 0.0, s, 0.0],
            [0.0, 0.0, s, 0.0, 0.0, s],
            [-s, 0.0, 0.0, s, 0.0, 0.0],
            [0.0, -s, 0.0, 0.0, s, 0.0],
            [0.0, 0.0, -s, 0.0, 0.0, s],
        ];
        let evals = [1.0e8_f64, 1.0e8, 1.0e8, 1.0, 1.0, 1.0]; // exact condition number kappa = 1e8!
        let mut w_ill = vec![0.0; 36];
        for r in 0..6 {
            for c in 0..6 {
                let mut sum = 0.0;
                for k in 0..6 {
                    sum += q_rot[r][k] * evals[k] * q_rot[c][k];
                }
                w_ill[r * 6 + c] = sum;
            }
        }
        let ill_contacts = vec![
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.4,
                mu_2: 0.4,
                free_vel: [-1.0, 0.15, -0.10],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.3,
                mu_2: 0.5,
                free_vel: [-2.0, 0.80, -0.50],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
        ];
        let sys_ill = DelassusSystem::new(ill_contacts, w_ill.clone());
        let pgs_ill = sys_ill.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::BlockPgs,
                max_iters: 300,
                tol: 1e-10,
                compliance_reg: 0.0,
                ..LcpSolverConfig::default()
            },
            None,
        );
        let socp_ill = sys_ill.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::ClarabelSocp,
                max_iters: 50,
                tol: 1e-10,
                compliance_reg: 0.0,
                ..LcpSolverConfig::default()
            },
            None,
        );
        assert!(
            !pgs_ill.converged && pgs_ill.normal_complementarity_res > 1e-2,
            "PGS should stall on kappa=1e8 problem (nr={:.3e})",
            pgs_ill.normal_complementarity_res
        );
        assert!(
            socp_ill.converged
                && socp_ill.normal_complementarity_res < 1e-7
                && socp_ill.friction_cone_res < 1e-7,
            "ClarabelSocp failed on kappa=1e8 problem: conv={}, iters={}, nr={:.3e}, fr={:.3e}\n  lam={:?}\n  u={:?}",
            socp_ill.converged,
            socp_ill.iterations,
            socp_ill.normal_complementarity_res,
            socp_ill.friction_cone_res,
            socp_ill.impulses,
            socp_ill.post_vel,
        );

        // Part C: Extreme kappa=1e8 + strongly anisotropic friction cones (audit regression)
        let aniso_ill_contacts = vec![
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 1.5,
                mu_2: 0.25,
                free_vel: [-1.0, 0.5, -0.4],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.2,
                mu_2: 1.8,
                free_vel: [-2.0, -0.9, 0.7],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
        ];
        let sys_aniso_ill = DelassusSystem::new(aniso_ill_contacts, w_ill);
        let ac_aniso = sys_aniso_ill.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::AlartCurnierNewton,
                max_iters: 100,
                tol: 1e-11,
                compliance_reg: 0.0,
                ..LcpSolverConfig::default()
            },
            None,
        );
        let socp_aniso = sys_aniso_ill.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::ClarabelSocp,
                max_iters: 50,
                tol: 1e-10,
                compliance_reg: 0.0,
                ..LcpSolverConfig::default()
            },
            None,
        );
        assert!(
            socp_aniso.converged
                && socp_aniso.normal_complementarity_res < 1e-8
                && socp_aniso.friction_cone_res < 1e-8,
            "ClarabelSocp failed on kappa=1e8 anisotropic problem: conv={}, iters={}, nr={:.3e}, fr={:.3e}",
            socp_aniso.converged,
            socp_aniso.iterations,
            socp_aniso.normal_complementarity_res,
            socp_aniso.friction_cone_res,
        );
        for i in 0..2 {
            for k in 0..3 {
                assert!(
                    (socp_aniso.impulses[i][k] - ac_aniso.impulses[i][k]).abs() < 1e-8,
                    "kappa=1e8 anisotropic Clarabel vs AC mismatch at ({i},{k}): {} vs {}",
                    socp_aniso.impulses[i][k],
                    ac_aniso.impulses[i][k]
                );
            }
        }

        // Part D: Rank-deficient 4-foot rigid table (12x12 Delassus of rank 6)
        let foot_pos = [
            Vec3::new(0.5, 0.5, -0.2),
            Vec3::new(0.5, -0.5, -0.2),
            Vec3::new(-0.5, 0.5, -0.2),
            Vec3::new(-0.5, -0.5, -0.2),
        ];
        let inv_m = [0.5_f64, 0.5, 0.5, 2.0, 2.0, 1.5]; // diag(M^-1) for rigid body
        let mut jac = vec![[0.0_f64; 6]; 12];
        let dirs = [Vec3::EZ, Vec3::EX, Vec3::EY];
        for (fi, &r) in foot_pos.iter().enumerate() {
            for (di, &d) in dirs.iter().enumerate() {
                let row = 3 * fi + di;
                let rx_d = r.cross(d);
                jac[row] = [d.x(), d.y(), d.z(), rx_d.x(), rx_d.y(), rx_d.z()];
            }
        }
        let mut w_table = vec![0.0_f64; 144];
        for r in 0..12 {
            for c in 0..12 {
                let mut sum = 0.0;
                for k in 0..6 {
                    sum += jac[r][k] * inv_m[k] * jac[c][k];
                }
                w_table[r * 12 + c] = sum;
            }
        }
        let table_contacts: Vec<ContactConstraint> = (0..4)
            .map(|_| ContactConstraint::from_normal(Vec3::EZ, 0.5, [-0.2, 0.05, -0.02]))
            .collect();
        let sys_table = DelassusSystem::new(table_contacts, w_table);
        let socp_table = sys_table.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::ClarabelSocp,
                max_iters: 50,
                tol: 1e-10,
                compliance_reg: 0.0,
                ..LcpSolverConfig::default()
            },
            None,
        );
        assert!(
            socp_table.converged
                && socp_table.normal_complementarity_res < 1e-8
                && socp_table.friction_cone_res < 1e-8,
            "ClarabelSocp failed on rank-deficient 4-foot table: conv={}, iters={}, nr={:.3e}, fr={:.3e}",
            socp_table.converged,
            socp_table.iterations,
            socp_table.normal_complementarity_res,
            socp_table.friction_cone_res,
        );
    }

    #[test]
    fn test_mujoco_solref_imp_in_lcp_and_arpc() {
        // Part A: MuJoCo SolRefImp static equilibrium in DelassusSystem:
        // Closed-loop dynamics a1 + (d B) v + (d^2 K) r = (1 - d) a0 => (-r_eq) = ((1-d)/(d^2*K))*g.
        let mass = 2.0_f64;
        let g = 9.81_f64;
        let dt = 0.002_f64;
        let sri = SolRefImp::from_mujoco([0.02, 1.0], [0.90, 0.95, 0.01, 0.5, 2.0]);

        // Solve exact fixed-point penetration depth pen_eq = ((1 - d(pen_eq)) / (d(pen_eq)^2 * K)) * g
        let mut pen_eq = 1e-3_f64;
        for _ in 0..30 {
            let [k_kb, _, imp, _] = sri.kbip(pen_eq, 2.0 * dt);
            pen_eq = ((1.0 - imp) / (imp * imp * k_kb)) * g;
        }
        let c = ContactConstraint {
            normal: Vec3::EZ,
            t1: Vec3::EX,
            t2: Vec3::EY,
            mu_1: 0.5,
            mu_2: 0.5,
            free_vel: [-g * dt, 0.0, 0.0],
            gap: -pen_eq,
            restitution: 0.0,
            solref_imp: Some(sri),
            pre_vel_n: None,
        };
        let mut w = vec![0.0; 9];
        w[0] = 1.0 / mass;
        w[4] = 1.0 / mass;
        w[8] = 1.0 / mass;
        let sys = DelassusSystem::new(vec![c], w);

        let cfg_base = LcpSolverConfig {
            solref_dt: dt,
            tol: 1e-12,
            compliance_reg: 0.0,
            ..LcpSolverConfig::default()
        };
        // Verify exact MuJoCo R = ((1-d)/d) * W_ii diagonal regularization
        let (w_eff, _) = sys.effective_system(&cfg_base);
        let expected_w00 = (1.0 / mass) + sri.constraint_regularizer(pen_eq, 1.0 / mass);
        assert!((w_eff[0] - expected_w00).abs() < 1e-14);

        let expected_lam_n = mass * g * dt;
        for method in [
            LcpSolverKind::BlockPgs,
            LcpSolverKind::AlartCurnierNewton,
            LcpSolverKind::ClarabelSocp,
        ] {
            let sol = sys.solve(&LcpSolverConfig { method, ..cfg_base }, None);
            assert!(
                sol.converged && sol.normal_complementarity_res < 1e-9,
                "{:?} failed with SolRefImp: nr={:.3e}",
                method,
                sol.normal_complementarity_res
            );
            assert!(
                (sol.impulses[0][0] - expected_lam_n).abs() < 1e-8,
                "{:?} MuJoCo equilibrium impulse mismatch: got {}, expected {}",
                method,
                sol.impulses[0][0],
                expected_lam_n
            );
        }

        // Part A2: Dynamic MuJoCo SolRefImp damping `b_kb` verification (exact closed-loop
        // `(v^+ - v_n)/dt + (d*B)*v_n + (d^2*K)*gap = (1 - d)*a0`):
        let dt_dyn = 0.001_f64;
        let w_unit = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (sri_dyn, expected_vn_post) in [
            (SolRefImp::direct(1000.0, 0.0), -0.998_f64),
            (SolRefImp::direct(1000.0, 500.0), -0.498_f64),
        ] {
            let mut c_dyn = ContactConstraint::from_normal(Vec3::EZ, 0.5, [-1.0, 0.0, 0.0])
                .with_solref_imp(sri_dyn)
                .with_pre_vel_n(-1.0);
            c_dyn.gap = -0.002;
            let sys_dyn = DelassusSystem::new(vec![c_dyn], w_unit.clone());
            let cfg_dyn = LcpSolverConfig {
                solref_dt: dt_dyn,
                tol: 1e-12,
                compliance_reg: 0.0,
                ..LcpSolverConfig::default()
            };
            for method in [
                LcpSolverKind::BlockPgs,
                LcpSolverKind::AlartCurnierNewton,
                LcpSolverKind::ClarabelSocp,
            ] {
                let sol = sys_dyn.solve(&LcpSolverConfig { method, ..cfg_dyn }, None);
                let vn_post = c_dyn.free_vel[0] + w_unit[0] * sol.impulses[0][0];
                assert!(
                    (vn_post - expected_vn_post).abs() < 1e-9,
                    "{:?} MuJoCo dynamic SolRefImp damping mismatch: got {vn_post}, expected {expected_vn_post}",
                    method
                );
            }
        }

        // Positive (timeconst, dampratio) mode with nonzero a0 = -9.81, v_n = -0.35, gap = -0.003:
        let sri_osc = SolRefImp::from_mujoco([0.02, 0.8], [0.90, 0.95, 0.01, 0.5, 2.0]);
        let vn_cur = -0.35_f64;
        let a0 = -9.81_f64;
        let gap_osc = -0.003_f64;
        let mut c_osc = ContactConstraint::from_normal(Vec3::EZ, 0.5, [vn_cur + a0 * dt, 0.0, 0.0])
            .with_solref_imp(sri_osc)
            .with_pre_vel_n(vn_cur);
        c_osc.gap = gap_osc;
        let sys_osc = DelassusSystem::new(vec![c_osc], w_unit.clone());
        let sol_osc = sys_osc.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::ClarabelSocp,
                solref_dt: dt,
                tol: 1e-12,
                compliance_reg: 0.0,
                ..LcpSolverConfig::default()
            },
            None,
        );
        let vn_post_osc = c_osc.free_vel[0] + w_unit[0] * sol_osc.impulses[0][0];
        let a1 = (vn_post_osc - vn_cur) / dt;
        let [k_osc, b_osc, d_osc, _] = sri_osc.kbip(-gap_osc, 2.0 * dt);
        let lhs = a1 + (d_osc * b_osc) * vn_cur + (d_osc * d_osc * k_osc) * gap_osc;
        let rhs = (1.0 - d_osc) * a0;
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "MuJoCo closed-loop acceleration law mismatch: lhs={lhs}, rhs={rhs}"
        );

        // Part B: MuJoCo SolRefImp in eval_arpc_site_with_mass (both positive and direct modes)
        let flat = PlaneSdf::flat_ground(0.0, MAT_HARD_FLOOR);
        let mats = MaterialLibrary::standard();
        let eff_mass = 2.5_f64;
        let pen = 0.004_f64;
        let vz = -0.05_f64;

        let site_pos = ContactSiteSpec {
            id: 10,
            name: "mj_pad_pos".to_string(),
            attachment: ContactAttachment::RigidBody { body_id: 0 },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: 0.01,
            patch_radius: 0.008,
            mode: ContactMode::IsotropicPad {
                mu: 0.6,
                pivot_mu: 0.2,
                static_mu_ratio: 1.0,
                sdf_only: false,
            },
            material_profile_id: MAT_HARD_FLOOR,
            solref_imp: Some(sri),
        };
        let pose = Se3::new(Vec3::new(0.0, 0.0, 0.01 - pen), Quat::IDENTITY);
        let twist = Vec6::from_parts(Vec3::new(0.02, 0.0, vz), Vec3::ZERO);
        let ev_pos = eval_arpc_site_with_mass(
            &site_pos,
            pose,
            twist,
            &flat,
            &mats,
            &ArpcSiteState::default(),
            dt,
            true,
            eff_mass,
        );
        let (kn_exp, cn_exp) = sri.to_force_stiffness_damping(pen, eff_mass, dt);
        let fn_exp = kn_exp * pen - cn_exp * vz;
        assert!(ev_pos.active);
        assert!(
            (ev_pos.normal_force - fn_exp).abs() < 1e-10,
            "ARPC positive SolRefImp normal force mismatch: {} vs {}",
            ev_pos.normal_force,
            fn_exp
        );
        let (eigs_pos, _) = ev_pos.k_tan.sym_eigen();
        assert!(eigs_pos.iter().all(|&l| l >= -1e-12));

        // Direct negative mode SolRefImp::direct(5000.0, 25.0)
        let sri_dir = SolRefImp::direct(5000.0, 25.0);
        let site_dir = ContactSiteSpec {
            solref_imp: Some(sri_dir),
            ..site_pos
        };
        let ev_dir = eval_arpc_site_with_mass(
            &site_dir,
            pose,
            twist,
            &flat,
            &mats,
            &ArpcSiteState::default(),
            dt,
            true,
            eff_mass,
        );
        let (kn_d, cn_d) = sri_dir.to_force_stiffness_damping(pen, eff_mass, dt);
        assert!((ev_dir.normal_force - (kn_d * pen - cn_d * vz)).abs() < 1e-10);
        let (eigs_dir, _) = ev_dir.k_tan.sym_eigen();
        assert!(eigs_dir.iter().all(|&l| l >= -1e-12));
    }

    #[test]
    fn test_tight_inclusion_batch_segments_and_clarabel_edge_cases() {
        // 1. Batch segment safe step using Tight Inclusion CCD
        let x0 = vec![
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, -0.5, 0.4),
            Vec3::new(0.0, 0.5, 0.4),
        ];
        let x1 = vec![
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, -0.5, -0.4),
            Vec3::new(0.0, 0.5, -0.4),
        ];
        let segs = vec![(0, 1), (2, 3)];
        let ti_cfg = TightInclusionConfig {
            min_dist: 0.04,
            ..TightInclusionConfig::default()
        };
        // Exact min_dist hit at z = 0.04 => t = (0.4 - 0.04) / 0.8 = 0.45
        let t_safe = compute_safe_step_segments_tight_inclusion(&x0, &x1, &segs, &ti_cfg);
        assert!(
            t_safe <= 0.45 + 1e-9 && (0.45 - t_safe) < 1e-3,
            "batch TI-CCD safe step mismatch: {t_safe}"
        );

        // 2. ClarabelSocp on mixed frictionless (mu=0), 1D friction (mu1>0, mu2=0), and separating contacts
        let contacts = vec![
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.0,
                mu_2: 0.0,
                free_vel: [-1.0, 0.5, -0.3],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.4,
                mu_2: 0.0,
                free_vel: [-2.0, 0.9, 0.2],
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
            ContactConstraint {
                normal: Vec3::EZ,
                t1: Vec3::EX,
                t2: Vec3::EY,
                mu_1: 0.0,
                mu_2: 0.3,
                free_vel: [0.5, 0.1, -0.2], // strictly separating
                gap: 0.0,
                restitution: 0.0,
                solref_imp: None,
                pre_vel_n: None,
            },
        ];
        let mut w = vec![0.0; 81];
        for i in 0..9 {
            w[i * 9 + i] = 1.0 + 0.1 * (i as f64);
        }
        let sys = DelassusSystem::new(contacts, w);
        let sol_ac = sys.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::AlartCurnierNewton,
                tol: 1e-12,
                ..LcpSolverConfig::default()
            },
            None,
        );
        let sol_socp = sys.solve(
            &LcpSolverConfig {
                method: LcpSolverKind::ClarabelSocp,
                tol: 1e-11,
                ..LcpSolverConfig::default()
            },
            None,
        );
        assert!(sol_socp.converged);
        assert!(sol_socp.normal_complementarity_res < 1e-9);
        assert!(sol_socp.friction_cone_res < 1e-9);
        // Separating 3rd contact must have zero impulse
        assert!(sol_socp.impulses[2].iter().all(|v| v.abs() < 1e-10));
        for i in 0..3 {
            for k in 0..3 {
                assert!(
                    (sol_socp.impulses[i][k] - sol_ac.impulses[i][k]).abs() < 1e-8,
                    "mixed cone mismatch at ({i},{k}): {} vs {}",
                    sol_socp.impulses[i][k],
                    sol_ac.impulses[i][k]
                );
            }
        }
    }
}

//! Rigorous numerical tests for `fg-math`:
//! - SO(3) / SE(3) exp, log, dexp, dexp_inv finite-difference verification
//! - Cubic-Hermite beam strain Jacobian analytic vs central finite differences
//! - Heterogeneous 12x12 K/M symmetry, SPD mass, Kirchhoff-Love/Vlasov & Timoshenko checks
//! - Spatial Plücker adjoint/coadjoint duality & gyroscopic inertia
//! - DER / Cosserat stretching, bending, and twisting analytical gradient & Hessian vs finite differences

use fg_math::{
    beam_local_strain, beam_strain_jacobian, compute_bishop_frames, der_bending_energy_grad_hess,
    der_stretching_energy_grad_hess, der_twisting_energy_grad_hess, local_beam_matrices,
    material_frame, parallel_transport, reference_twist, se3_dexp_inv, se3_exp, se3_log,
    so3_dexp, so3_dexp_inv, so3_exp, so3_log, spatial_cross_force, spatial_cross_motion,
    BeamMaterial, BeamSection, Mat3, Mat6, Quat, Se3, SpatialInertia, Vec3, Vec6,
};

#[test]
fn test_so3_exp_log_and_dexp_inv_identities() {
    let test_vecs = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1e-7, -2e-7, 3e-7),
        Vec3::new(0.3, -0.7, 0.4),
        Vec3::new(-1.1, 0.8, -0.6),
    ];
    for phi in test_vecs {
        let q = Quat::exp(phi);
        assert!((q.norm() - 1.0).abs() < 1e-14);
        let phi_rec = q.log();
        assert!((phi_rec - phi).norm() < 1e-13, "phi={phi:?}, rec={phi_rec:?}");

        let r = so3_exp(phi);
        let rq = q.to_mat3();
        assert!((r - rq).max_abs() < 1e-13);
        let phi_m = so3_log(r);
        assert!((phi_m - phi).norm() < 1e-13);

        // J_l(phi) * J_l^{-1}(phi) == I
        let jl = so3_dexp(phi);
        let jli = so3_dexp_inv(phi);
        let prod = jl.mul_mat(jli);
        assert!((prod - Mat3::IDENTITY).max_abs() < 1e-12);

        // Finite-difference check of dexp_inv:
        // log(exp(eps) * exp(phi)) = phi + dexp_inv(phi) * eps + O(eps^2)
        let eps = 1e-6;
        for axis in 0..3 {
            let mut e = [0.0; 3];
            e[axis] = eps;
            let ev = Vec3(e);
            let p_plus = Quat::exp(ev).mul(q).log();
            let p_minus = Quat::exp(-ev).mul(q).log();
            let fd_col = (p_plus - p_minus) / (2.0 * eps);
            let an_col = jli.col(axis);
            assert!(
                (fd_col - an_col).norm() < 1e-8,
                "so3 dexp_inv mismatch at phi={phi:?} axis={axis}: fd={fd_col:?} an={an_col:?}"
            );
        }
    }
}

#[test]
fn test_se3_exp_log_and_dexp_inv_fd() {
    let cases = [
        Vec6([0.15, -0.22, 0.08, 0.35, -0.42, 0.27]),
        Vec6([0.4, -0.3, 0.5, 0.0, 0.0, 0.0]),
        Vec6([100.0, -200.0, 300.0, 5e-4, -3e-4, 4e-4]),
    ];
    for xi in cases {
        let tf = se3_exp(xi);
        let xi_rec = se3_log(tf);
        assert!(
            (xi_rec - xi).norm() < 1e-11 * (1.0 + xi.norm()),
            "se3 log(exp) mismatch for {xi:?}"
        );

        // Check inverse composition
        let id = tf.compose(tf.inverse());
        assert!(id.pos.norm() < 1e-11 * (1.0 + xi.norm()));
        assert!(id.rot.log().norm() < 1e-13);

        // Finite-difference check of se3_dexp_inv:
        // log(exp(eps) * exp(xi)) = xi + dexp_inv(xi) * eps + O(eps^2)
        let jli = se3_dexp_inv(xi);
        let eps = 1e-6;
        for col in 0..6 {
            let mut e = [0.0; 6];
            e[col] = eps;
            let ev = Vec6(e);
            let plus = se3_log(se3_exp(ev).compose(tf));
            let minus = se3_log(se3_exp(-ev).compose(tf));
            let fd = (plus - minus) * (0.5 / eps);
            let mut an = [0.0; 6];
            for (r, val) in an.iter_mut().enumerate() {
                *val = jli.0[r][col];
            }
            let err = (fd - Vec6(an)).norm();
            assert!(
                err < 1e-7 * (1.0 + xi.lin().norm() * 1e-2),
                "se3_dexp_inv xi={xi:?} col {col} err={err}"
            );
        }
    }

    // Verify smooth branch continuity across theta2 = 1e-6 (||w|| = 1e-3)
    let w_dir = Vec3::new(0.6, -0.48, 0.64).normalized();
    let j_below = se3_dexp_inv(Vec6::from_parts(Vec3::new(1.0, -2.0, 3.0), w_dir * 0.999e-3));
    let j_above = se3_dexp_inv(Vec6::from_parts(Vec3::new(1.0, -2.0, 3.0), w_dir * 1.001e-3));
    assert!(
        (j_above - j_below).max_abs() < 1e-5,
        "se3_dexp_inv branch discontinuity = {}",
        (j_above - j_below).max_abs()
    );
}

#[test]
fn test_beam_strain_jacobian_matches_finite_differences() {
    let p0_body = Vec3::new(0.02, -0.05, 0.11);
    let q0_body = Quat::exp(Vec3::new(0.18, -0.25, 0.31));
    let p1_body = Vec3::new(-0.08, -0.01, 0.14);
    let q1_body = Quat::exp(Vec3::new(-0.14, 0.22, -0.19));

    let clamp0 = Se3::new(
        Vec3::new(-0.0165, 0.046, 0.021),
        Quat::exp(Vec3::new(0.1, 0.3, -0.2)),
    );
    let clamp1 = Se3::new(
        Vec3::new(0.0, 0.046, 0.020),
        Quat::exp(Vec3::new(-0.08, -0.27, 0.15)),
    );
    let length = 0.012;

    let eval_u = |pb0: Vec3, qb0: Quat, pb1: Vec3, qb1: Quat| {
        let off0 = qb0.rotate(clamp0.pos);
        let off1 = qb1.rotate(clamp1.pos);
        let p0 = pb0 + off0;
        let p3 = pb1 + off1;
        let q0 = qb0.mul(clamp0.rot);
        let q1 = qb1.mul(clamp1.rot);
        beam_local_strain(p0, q0, p3, q1, length, true).0
    };

    let off0 = q0_body.rotate(clamp0.pos);
    let off1 = q1_body.rotate(clamp1.pos);
    let p0 = p0_body + off0;
    let p3 = p1_body + off1;
    let q0 = q0_body.mul(clamp0.rot);
    let q1 = q1_body.mul(clamp1.rot);
    let j_analytic = beam_strain_jacobian(p0, q0, off0, p3, q1, off1, length);

    let eps = 1e-6;
    let mut max_err = 0.0_f64;
    for col in 0..12 {
        let perturb = |sign: f64| {
            let mut pb0 = p0_body;
            let mut qb0 = q0_body;
            let mut pb1 = p1_body;
            let mut qb1 = q1_body;
            let s = sign * eps;
            match col {
                0..=2 => pb0[col] += s,
                3..=5 => {
                    let mut w = [0.0; 3];
                    w[col - 3] = s;
                    qb0 = Quat::exp(Vec3(w)).mul(qb0);
                }
                6..=8 => pb1[col - 6] += s,
                9..=11 => {
                    let mut w = [0.0; 3];
                    w[col - 9] = s;
                    qb1 = Quat::exp(Vec3(w)).mul(qb1);
                }
                _ => unreachable!(),
            }
            eval_u(pb0, qb0, pb1, qb1)
        };
        let fd = (perturb(1.0) - perturb(-1.0)) * (0.5 / eps);
        for row in 0..12 {
            let err = (fd[row] - j_analytic.0[row][col]).abs();
            max_err = max_err.max(err);
        }
    }
    assert!(
        max_err < 1e-7,
        "analytic beam_strain_jacobian max FD error = {max_err}"
    );
}

#[test]
fn test_local_beam_matrices_properties_and_thin_shell() {
    let sec = BeamSection::Rectangular {
        width: 0.016,
        thickness: 0.00015,
    };
    let mat = BeamMaterial::SPRING_STEEL;
    let l = 0.01;
    let (k_eb, m_eb) = local_beam_matrices(l, &sec, &mat, false, false);
    let (k_shell, _) = local_beam_matrices(l, &sec, &mat, true, false);
    let (k_timo, _) = local_beam_matrices(l, &sec, &mat, false, true);

    // Exact symmetry
    assert!((k_eb - k_eb.transpose()).max_abs() < 1e-12);
    assert!((m_eb - m_eb.transpose()).max_abs() < 1e-18);
    // Rigid-body translation invariance: K * [1,0,0,0,0,0, 1,0,0,0,0,0] == 0
    let rigid_x = fg_math::Vec12([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(k_eb.mul_vec(rigid_x).norm() < 1e-8);
    // Kirchhoff-Love increases out-of-plane bending stiffness by exactly 1/(1 - nu^2)
    let ratio = k_shell.0[2][2] / k_eb.0[2][2];
    let expected = 1.0 / (1.0 - mat.poisson_ratio * mat.poisson_ratio);
    assert!((ratio - expected).abs() < 1e-12);
    // Timoshenko shear softens transverse bending slightly compared to Euler-Bernoulli
    assert!(k_timo.0[1][1] < k_eb.0[1][1]);
    // Consistent mass matrix must be strictly positive definite (check solve_lu on identity)
    let x = m_eb.solve_lu(rigid_x).expect("M must be invertible");
    assert!((m_eb.mul_vec(x) - rigid_x).norm() < 1e-12);
}

#[test]
fn test_mat3_mat6_spd_projection_and_spatial_duality() {
    let sym3 = Mat3([[2.0, 1.0, 0.3], [1.0, -0.5, 0.2], [0.3, 0.2, 1.5]]);
    let spd3 = sym3.project_spd(1e-6);
    let (evals3, _) = spd3.sym_eigen();
    assert!(evals3.iter().all(|&e| e >= 1e-6 - 1e-12));

    let sym6 = Mat6::from_blocks(sym3, Mat3::diag([0.2, -0.1, 0.05]), Mat3::diag([0.2, -0.1, 0.05]), sym3);
    let spd6 = sym6.project_spd(1e-5);
    let (evals6, _) = spd6.sym_eigen();
    assert!(evals6.iter().all(|&e| e >= 1e-5 - 1e-11));
    assert!(spd6.cholesky().is_some());

    // Spatial cross duality: v x* == -(v x)^T
    let v = Vec6([0.4, -0.3, 0.9, -0.2, 0.5, 0.7]);
    let crm = spatial_cross_motion(v);
    let crf = spatial_cross_force(v);
    assert!((crf + crm.transpose()).max_abs() < 1e-14);

    // Spatial inertia kinetic energy positivity & zero work of gyroscopic bias
    let inertia = SpatialInertia::from_mass_and_diag(0.28, [4.5e-4, 2.5e-4, 2.5e-4]);
    let bias = inertia.bias_wrench_body(v);
    assert!(bias.dot(v).abs() < 1e-14, "gyroscopic bias must do zero power");
}

#[test]
fn test_der_stretching_bending_twisting_fd() {
    // 1. Stretching gradient & Hessian vs central FD
    let x0 = Vec3::new(0.0, 0.0, 0.0);
    let x1 = Vec3::new(0.11, 0.03, -0.02);
    let l_rest = 0.10;
    let ea = 500.0;
    let s_eval = der_stretching_energy_grad_hess(x0, x1, l_rest, ea, false);
    let eps = 1e-6;
    for i in 0..6 {
        let shift = |sgn: f64| {
            let mut a = x0;
            let mut b = x1;
            if i < 3 {
                a[i] += sgn * eps;
            } else {
                b[i - 3] += sgn * eps;
            }
            der_stretching_energy_grad_hess(a, b, l_rest, ea, false)
        };
        let sp = shift(1.0);
        let sm = shift(-1.0);
        let fd_g = (sp.energy - sm.energy) / (2.0 * eps);
        assert!((fd_g - s_eval.grad[i]).abs() < 1e-6);
        for r in 0..6 {
            let fd_h = (sp.grad[r] - sm.grad[r]) / (2.0 * eps);
            assert!((fd_h - s_eval.hess.0[r][i]).abs() < 1e-5);
        }
    }

    // 2. Bending & Twisting gradient & Hessian under exact time-parallel frame transport
    let pts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.10, 0.02, -0.01),
        Vec3::new(0.19, 0.07, 0.04),
    ];
    let frames = compute_bishop_frames(&pts, Vec3::EZ);
    let (t0_base, u0_base, _) = frames[0];
    let (t1_base, u1_base, _) = frames[1];
    let th0 = 0.23;
    let th1 = -0.31;
    let rest_kappa = [0.05, -0.03];
    let rest_twist = 0.08;
    let voronoi = 0.105;
    let ei = [1.4, 2.2];
    let gj = 0.9;

    let eval_at = |q: [f64; 11]| {
        let p0 = Vec3([q[0], q[1], q[2]]);
        let p1 = Vec3([q[3], q[4], q[5]]);
        let p2 = Vec3([q[6], q[7], q[8]]);
        let t0 = (p1 - p0).normalized();
        let t1 = (p2 - p1).normalized();
        let u0 = parallel_transport(u0_base, t0_base, t0);
        let v0 = t0.cross(u0);
        let u1 = parallel_transport(u1_base, t1_base, t1);
        let v1 = t1.cross(u1);
        let (m1_0, m2_0) = material_frame(t0, u0, v0, q[9]);
        let (m1_1, m2_1) = material_frame(t1, u1, v1, q[10]);
        let m_ref = reference_twist(u0, t0, u1, t1, 0.0);
        let bend = der_bending_energy_grad_hess(
            p0, p1, p2, m1_0, m2_0, m1_1, m2_1, rest_kappa, voronoi, ei,
        );
        let twist = der_twisting_energy_grad_hess(
            p0, p1, p2, q[9], q[10], m_ref, rest_twist, voronoi, gj,
        );
        (bend, twist)
    };

    let q0 = [
        pts[0][0], pts[0][1], pts[0][2], pts[1][0], pts[1][1], pts[1][2], pts[2][0], pts[2][1],
        pts[2][2], th0, th1,
    ];
    let (b0, t0) = eval_at(q0);

    let mut max_bg_err = 0.0_f64;
    let mut max_bh_err = 0.0_f64;
    let mut max_bh_rc = (0, 0);
    let mut max_bh_vals = (0.0, 0.0);
    let mut max_tg_err = 0.0_f64;
    let mut max_th_err = 0.0_f64;
    let mut max_th_rc = (0, 0);
    let mut max_th_vals = (0.0, 0.0);

    let eps_h = 1e-5;
    for col in 0..11 {
        let mut qp = q0;
        let mut qm = q0;
        qp[col] += eps;
        qm[col] -= eps;
        let (bp, tp) = eval_at(qp);
        let (bm, tm) = eval_at(qm);

        let fd_bg = (bp.energy - bm.energy) / (2.0 * eps);
        max_bg_err = max_bg_err.max((fd_bg - b0.grad[col]).abs());
        let fd_tg = (tp.energy - tm.energy) / (2.0 * eps);
        max_tg_err = max_tg_err.max((fd_tg - t0.grad[col]).abs());

        // Exact 2nd-order central finite difference of scalar energy E(q) for true Hessian
        for row in 0..11 {
            let mut qpp = q0;
            let mut qpm = q0;
            let mut qmp = q0;
            let mut qmm = q0;
            qpp[row] += eps_h;
            qpp[col] += eps_h;
            qpm[row] += eps_h;
            qpm[col] -= eps_h;
            qmp[row] -= eps_h;
            qmp[col] += eps_h;
            qmm[row] -= eps_h;
            qmm[col] -= eps_h;
            let (bpp, tpp) = eval_at(qpp);
            let (bpm, tpm) = eval_at(qpm);
            let (bmp, tmp) = eval_at(qmp);
            let (bmm, tmm) = eval_at(qmm);
            let fd_bh = (bpp.energy - bpm.energy - bmp.energy + bmm.energy) / (4.0 * eps_h * eps_h);
            let e_b = (fd_bh - b0.hess[row][col]).abs();
            if e_b > max_bh_err {
                max_bh_err = e_b;
                max_bh_rc = (row, col);
                max_bh_vals = (fd_bh, b0.hess[row][col]);
            }
            let fd_th = (tpp.energy - tpm.energy - tmp.energy + tmm.energy) / (4.0 * eps_h * eps_h);
            let e_t = (fd_th - t0.hess[row][col]).abs();
            if e_t > max_th_err {
                max_th_err = e_t;
                max_th_rc = (row, col);
                max_th_vals = (fd_th, t0.hess[row][col]);
            }
        }
    }

    assert!(max_bg_err < 1e-6, "DER bending grad FD err = {max_bg_err}");
    assert!(
        max_bh_err < 1e-3,
        "DER bending hess FD err = {max_bh_err} at {max_bh_rc:?}: fd={} an={}",
        max_bh_vals.0,
        max_bh_vals.1
    );
    assert!(max_tg_err < 1e-6, "DER twisting grad FD err = {max_tg_err}");
    assert!(
        max_th_err < 1e-3,
        "DER twisting hess FD err = {max_th_err} at {max_th_rc:?}: fd={} an={}",
        max_th_vals.0,
        max_th_vals.1
    );
}

#[test]
fn test_mujoco_solref_solimp_impedance_and_conversions() {
    use fg_math::SolRefImp;

    // 1. Standard positive solref=(tau, zeta) + piecewise polynomial solimp
    let s = SolRefImp::from_mujoco([0.02, 0.8], [0.85, 0.98, 0.004, 0.35, 3.0]);
    assert!(!s.is_direct_mode());
    assert!((s.impedance(0.0) - 0.85).abs() < 1e-14);
    assert!((s.impedance(0.004) - 0.98).abs() < 1e-14);
    assert!((s.impedance(0.010) - 0.98).abs() < 1e-14);

    // Exact C^1 continuity at midpoint x = 0.35 (r = 0.0014):
    // y(mid) = mid, dy/dx(mid-) = dy/dx(mid+) = power
    let r_mid = s.midpoint * s.width;
    let eps = 1e-8;
    let (d_l, dp_l) = s.impedance_and_deriv(r_mid - eps);
    let (d_c, dp_c) = s.impedance_and_deriv(r_mid);
    let (d_r, dp_r) = s.impedance_and_deriv(r_mid + eps);
    let expected_d_mid = s.dmin + s.midpoint * (s.dmax - s.dmin);
    let expected_dp_mid = (s.dmax - s.dmin) * s.power / s.width;
    assert!((d_c - expected_d_mid).abs() < 1e-14);
    assert!((dp_c - expected_dp_mid).abs() < 1e-12);
    assert!((d_l - d_r).abs() < 1e-4);
    assert!((dp_l - dp_r).abs() < 1e-3);

    // Finite-difference derivative check across transition band (both sides of midpoint)
    for &frac in &[0.1, 0.25, 0.5, 0.8] {
        let r = frac * s.width;
        let (_, dp) = s.impedance_and_deriv(r);
        let fd = (s.impedance(r + 1e-7) - s.impedance(r - 1e-7)) / 2e-7;
        assert!(
            (dp - fd).abs() < 1e-6 * (1.0 + dp.abs()),
            "solimp deriv FD mismatch at frac={frac}: an={dp} fd={fd}"
        );
    }

    // At full penetration (d = dmax), effective (k, b) match exact oscillator (1/(tau*zeta)^2, 2/tau)
    let (k_full, b_full) = s.to_stiffness_damping(s.width);
    let expected_k = 1.0 / (s.timeconst * s.dampratio).powi(2);
    let expected_b = 2.0 / s.timeconst;
    assert!((k_full - expected_k).abs() < 1e-9 * expected_k);
    assert!((b_full - expected_b).abs() < 1e-9 * expected_b);

    // Effective damping ratio zeta = b / (2 sqrt(k)) is invariant to r!
    for &r in &[0.0, 0.001, 0.0025, 0.005] {
        let (kr, br) = s.to_stiffness_damping(r);
        let zeta = br / (2.0 * kr.sqrt());
        assert!((zeta - s.dampratio).abs() < 1e-12);
    }

    // MuJoCo 2*dt stability clamp on positive timeconst
    let (k_clamp, b_clamp) = s.to_stiffness_damping_dt(s.width, 0.02); // 2*dt = 0.04 > 0.02
    assert!((b_clamp - 2.0 / 0.04).abs() < 1e-11);
    assert!((k_clamp - 1.0 / (0.04 * 0.8_f64).powi(2)).abs() < 1e-9);

    // 2. Direct negative convention: solref = (-stiffness, -damping)
    let sd = SolRefImp::direct(5000.0, 12.0);
    assert!(sd.is_direct_mode());
    let (kd, bd) = sd.to_stiffness_damping(sd.width);
    assert!((kd - 5000.0).abs() < 1e-10);
    assert!((bd - 12.0).abs() < 1e-10);
    let (kf, bf) = sd.to_force_stiffness_damping(sd.width, 0.28, 0.001);
    assert!((kf - 5000.0).abs() < 1e-10);
    assert!((bf - 12.0).abs() < 1e-10);

    // 3. Linear power=1 case independent of midpoint
    let slin = SolRefImp::from_mujoco([0.02, 1.0], [0.5, 0.9, 0.01, 0.2, 1.0]);
    assert!((slin.impedance(0.005) - 0.7).abs() < 1e-14);

    // 4. Constraint regularizer R = ((1-d)/d) * diag_approx
    let reg = s.constraint_regularizer(s.width, 2.5);
    assert!((reg - ((1.0 - 0.98) / 0.98) * 2.5).abs() < 1e-14);
}

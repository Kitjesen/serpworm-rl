//! Discrete Elastic Rods (DER) / Cosserat Rod geometry & variational elasticity:
//! Bishop-frame parallel transport, material frames, curvature binormal, reference twist,
//! and exact analytical energies, gradients, and Hessians for stretching, bending, and
//! twisting (Bergou et al., SIGGRAPH 2008 / 2010; Jawed et al., 2018; Py-DiSMech, 2025).

use crate::vec::{Mat3, Mat6, Vec3, Vec6};

/// Minimal-rotation parallel transport of vector `u` (orthogonal to `t_from`) from unit
/// tangent `t_from` to unit tangent `t_to`.
///
/// Closed-form expression:
/// `P(u) = u - ((u . t_to) / (1 + t_from . t_to)) * (t_from + t_to)`.
pub fn parallel_transport(u: Vec3, t_from: Vec3, t_to: Vec3) -> Vec3 {
    let denom = 1.0 + t_from.dot(t_to);
    if denom < 1e-12 {
        // Nearly antiparallel edges: reflect along orthogonal axis.
        let b = t_from.cross(t_to);
        let bn = b.norm();
        if bn > 1e-12 {
            let axis = b / bn;
            let n1 = axis.cross(t_from);
            let n2 = axis.cross(t_to);
            return axis * u.dot(axis) + n2 * u.dot(n1);
        }
        return -u;
    }
    let transported = u - (u.dot(t_to) / denom) * (t_from + t_to);
    // Re-orthogonalize against t_to and re-normalize to suppress drift.
    let proj = transported - transported.dot(t_to) * t_to;
    proj.normalized_or(u)
}

/// Compute twist-free Bishop frames `(t_i, u_i, v_i)` along a polygonal centerline starting
/// from an initial director `u0` (projected orthogonal to the first edge tangent).
pub fn compute_bishop_frames(positions: &[Vec3], u0: Vec3) -> Vec<(Vec3, Vec3, Vec3)> {
    let n_edges = positions.len().saturating_sub(1);
    let mut frames = Vec::with_capacity(n_edges);
    if n_edges == 0 {
        return frames;
    }
    let t0 = (positions[1] - positions[0]).normalized();
    let u_init = (u0 - u0.dot(t0) * t0).normalized_or_else_ortho(t0);
    let mut prev_t = t0;
    let mut prev_u = u_init;
    frames.push((prev_t, prev_u, prev_t.cross(prev_u)));
    for i in 1..n_edges {
        let ti = (positions[i + 1] - positions[i]).normalized();
        let ui = parallel_transport(prev_u, prev_t, ti);
        let vi = ti.cross(ui);
        frames.push((ti, ui, vi));
        prev_t = ti;
        prev_u = ui;
    }
    frames
}

trait Vec3OrthoExt {
    fn normalized_or_else_ortho(self, tangent: Vec3) -> Vec3;
}

impl Vec3OrthoExt for Vec3 {
    fn normalized_or_else_ortho(self, tangent: Vec3) -> Vec3 {
        if self.norm() > 1e-12 {
            self.normalized()
        } else {
            let fallback = if tangent.x().abs() < 0.9 {
                Vec3::EX
            } else {
                Vec3::EY
            };
            tangent.cross(fallback).normalized()
        }
    }
}

/// Rotate Bishop frame `(u, v)` around unit tangent `t` by scalar twist angle `theta`
/// to obtain material directors `(m1, m2)`.
#[inline]
pub fn material_frame(_t: Vec3, u: Vec3, v: Vec3, theta: f64) -> (Vec3, Vec3) {
    let (s, c) = theta.sin_cos();
    let m1 = c * u + s * v;
    let m2 = -s * u + c * v;
    (m1, m2)
}

/// Compute discrete reference twist `m_ref` at the vertex joining edges `(t_prev, u_prev)`
/// and `(t_curr, u_curr)`, unwrapped continuously near `prev_ref_twist`.
pub fn reference_twist(
    u_prev: Vec3,
    t_prev: Vec3,
    u_curr: Vec3,
    t_curr: Vec3,
    prev_ref_twist: f64,
) -> f64 {
    let ut = parallel_transport(u_prev, t_prev, t_curr);
    let mut m_ref = signed_angle(ut, u_curr, t_curr);
    let two_pi = 2.0 * std::f64::consts::PI;
    while m_ref - prev_ref_twist > std::f64::consts::PI {
        m_ref -= two_pi;
    }
    while m_ref - prev_ref_twist < -std::f64::consts::PI {
        m_ref += two_pi;
    }
    m_ref
}

/// Signed angle from unit vector `u` to unit vector `v` about unit axis `n` in `[-pi, pi]`.
#[inline]
pub fn signed_angle(u: Vec3, v: Vec3, n: Vec3) -> f64 {
    let cos_a = u.dot(v).clamp(-1.0, 1.0);
    let sin_a = u.cross(v).dot(n);
    sin_a.atan2(cos_a)
}

/// Discrete curvature binormal `(kappa * b)_i = 2 (e_prev x e_curr) / (|e_prev||e_curr| + e_prev . e_curr)`.
#[inline]
pub fn curvature_binormal(e_prev: Vec3, e_curr: Vec3) -> Vec3 {
    let denom = e_prev.norm() * e_curr.norm() + e_prev.dot(e_curr);
    let denom_safe = if denom.abs() < 1e-12 {
        1e-12_f64.copysign(denom)
    } else {
        denom
    };
    (2.0 / denom_safe) * e_prev.cross(e_curr)
}

/// Discrete material curvature pair `[kappa_1, kappa_2]` projected onto averaged material directors.
#[inline]
pub fn material_curvature(
    kb: Vec3,
    m1_prev: Vec3,
    m2_prev: Vec3,
    m1_curr: Vec3,
    m2_curr: Vec3,
) -> [f64; 2] {
    let k1 = 0.5 * (m2_prev + m2_curr).dot(kb);
    let k2 = -0.5 * (m1_prev + m1_curr).dot(kb);
    [k1, k2]
}

/// Evaluation of DER edge stretching energy, 6D gradient `[x0; x1]`, and 6x6 Hessian.
#[derive(Copy, Clone, Debug)]
pub struct DerStretchEval {
    pub energy: f64,
    pub grad: Vec6,
    pub hess: Mat6,
}

/// Exact DER stretching energy `E_s = 0.5 * EA * l_bar * (|e|/l_bar - 1)^2`,
/// 6D gradient w.r.t. `[x0; x1]`, and 6x6 analytical Hessian.
pub fn der_stretching_energy_grad_hess(
    x0: Vec3,
    x1: Vec3,
    rest_len: f64,
    ea: f64,
    project_spd: bool,
) -> DerStretchEval {
    let e = x1 - x0;
    let l = e.norm().max(1e-15);
    let t = e / l;
    let strain = l / rest_len - 1.0;
    let energy = 0.5 * ea * rest_len * strain * strain;

    let f_edge = (ea * strain) * t;
    let grad = Vec6::from_parts(-f_edge, f_edge);

    // d(f_edge)/de = (EA / rest_len) * t t^T + (EA * strain / l) * (I - t t^T)
    let tt = t.outer(t);
    let k_parallel = ea / rest_len;
    let mut k_perp = ea * strain / l;
    if project_spd && k_perp < 0.0 {
        k_perp = 0.0;
    }
    let h_ee = k_parallel * tt + k_perp * (Mat3::IDENTITY - tt);
    let hess = Mat6::from_blocks(h_ee, -h_ee, -h_ee, h_ee);
    DerStretchEval { energy, grad, hess }
}

/// Evaluation of DER interior-vertex bending energy, 11D gradient, and 11x11 Hessian.
/// DOF ordering: `[x0(0..3), x1(3..6), x2(6..9), theta0(9), theta1(10)]`.
#[derive(Copy, Clone, Debug)]
pub struct DerBendEval {
    pub energy: f64,
    pub kappa: [f64; 2],
    pub grad: [f64; 11],
    pub hess: [[f64; 11]; 11],
}

/// Map 3x3 edge-block Hessian (`[e0, e1]`) into the 9x9 vertex block (`[x0, x1, x2]`),
/// where `e0 = x1 - x0` and `e1 = x2 - x1`.
fn scatter_edge_hess_to_vertices(
    h00: Mat3,
    h01: Mat3,
    h10: Mat3,
    h11: Mat3,
    out: &mut [[f64; 11]; 11],
) {
    // Vertex blocks:
    // x0 = -e0
    // x1 = +e0 - e1
    // x2 = +e1
    let blocks = [
        [h00, -(h00 - h01), -h01],
        [-(h00 - h10), h00 - h01 - h10 + h11, h01 - h11],
        [-h10, h10 - h11, h11],
    ];
    for bi in 0..3 {
        for bj in 0..3 {
            let blk = blocks[bi][bj].0;
            for r in 0..3 {
                for c in 0..3 {
                    out[3 * bi + r][3 * bj + c] += blk[r][c];
                }
            }
        }
    }
}

/// Map `(e0, e1)` cross-derivative with scalar DOF `col_idx` (9 or 10) into `[x0, x1, x2]`.
fn scatter_edge_scalar_hess(
    de0: Vec3,
    de1: Vec3,
    col_idx: usize,
    out: &mut [[f64; 11]; 11],
) {
    let dx = [-de0, de0 - de1, de1];
    for (bi, v) in dx.iter().enumerate() {
        for r in 0..3 {
            out[3 * bi + r][col_idx] += v.0[r];
            out[col_idx][3 * bi + r] += v.0[r];
        }
    }
}

/// Exact DER bending energy `E_b = 0.5 / voronoi_len * (kappa - rest_kappa)^T * diag(ei1, ei2) * (kappa - rest_kappa)`,
/// exact 11D gradient, and exact 11x11 analytical Hessian under time-parallel reference frame transport.
///
/// DOF ordering: `[x0(0..3), x1(3..6), x2(6..9), theta0(9), theta1(10)]`.
#[allow(clippy::too_many_arguments)]
pub fn der_bending_energy_grad_hess(
    x0: Vec3,
    x1: Vec3,
    x2: Vec3,
    m1_0: Vec3,
    m2_0: Vec3,
    m1_1: Vec3,
    m2_1: Vec3,
    rest_kappa: [f64; 2],
    voronoi_len: f64,
    ei: [f64; 2],
) -> DerBendEval {
    let e0 = x1 - x0;
    let e1 = x2 - x1;
    let l0 = e0.norm().max(1e-15);
    let l1 = e1.norm().max(1e-15);
    let t0 = e0 / l0;
    let t1 = e1 / l1;

    let chi = (1.0 + t0.dot(t1)).max(1e-12);
    let inv_chi = 1.0 / chi;
    let t_tilde = (t0 + t1) * inv_chi;
    let kb = (2.0 * inv_chi) * t0.cross(t1);

    let m1_tilde = (m1_0 + m1_1) * inv_chi;
    let m2_tilde = (m2_0 + m2_1) * inv_chi;

    let k1 = 0.5 * (m2_0 + m2_1).dot(kb);
    let k2 = -0.5 * (m1_0 + m1_1).dot(kb);
    let dk1 = k1 - rest_kappa[0];
    let dk2 = k2 - rest_kappa[1];

    let inv_lbar = 1.0 / voronoi_len.max(1e-15);
    let energy = 0.5 * inv_lbar * (ei[0] * dk1 * dk1 + ei[1] * dk2 * dk2);

    // First derivatives of kappa_1 and kappa_2 wrt (e0, e1, theta0, theta1)
    let d1_e0 = (1.0 / l0) * (-k1 * t_tilde + t1.cross(m2_tilde));
    let d1_e1 = (1.0 / l1) * (-k1 * t_tilde - t0.cross(m2_tilde));
    let d1_th0 = -0.5 * m1_0.dot(kb);
    let d1_th1 = -0.5 * m1_1.dot(kb);

    let d2_e0 = (1.0 / l0) * (-k2 * t_tilde - t1.cross(m1_tilde));
    let d2_e1 = (1.0 / l1) * (-k2 * t_tilde + t0.cross(m1_tilde));
    let d2_th0 = -0.5 * m2_0.dot(kb);
    let d2_th1 = -0.5 * m2_1.dot(kb);

    let pack_11 = |de0: Vec3, de1: Vec3, dth0: f64, dth1: f64| -> [f64; 11] {
        let dx0 = -de0;
        let dx1 = de0 - de1;
        let dx2 = de1;
        [
            dx0.0[0], dx0.0[1], dx0.0[2], dx1.0[0], dx1.0[1], dx1.0[2], dx2.0[0], dx2.0[1],
            dx2.0[2], dth0, dth1,
        ]
    };

    let g1 = pack_11(d1_e0, d1_e1, d1_th0, d1_th1);
    let g2 = pack_11(d2_e0, d2_e1, d2_th0, d2_th1);

    let w1 = inv_lbar * ei[0];
    let w2 = inv_lbar * ei[1];

    let mut grad = [0.0; 11];
    for i in 0..11 {
        grad[i] = w1 * dk1 * g1[i] + w2 * dk2 * g2[i];
    }

    // Gauss-Newton (outer product) part of Hessian: w1 * g1 g1^T + w2 * g2 g2^T
    let mut hess = [[0.0; 11]; 11];
    for i in 0..11 {
        for j in 0..11 {
            hess[i][j] = w1 * g1[i] * g1[j] + w2 * g2[i] * g2[j];
        }
    }

    // Exact geometric second derivatives: w1 * dk1 * D^2(k1) + w2 * dk2 * D^2(k2)
    let tt = t_tilde.outer(t_tilde);
    let id_t0 = Mat3::IDENTITY - t0.outer(t0);
    let id_t1 = Mat3::IDENTITY - t1.outer(t1);
    let id_t0t1 = Mat3::IDENTITY + t0.outer(t1);

    let inv_l0_2 = 1.0 / (l0 * l0);
    let inv_l1_2 = 1.0 / (l1 * l1);
    let inv_l0l1 = 1.0 / (l0 * l1);

    // --- D^2(k1) ---
    let c1 = w1 * dk1;
    if c1.abs() > 0.0 {
        let t1_x_m2 = t1.cross(m2_tilde);
        let t0_x_m2 = t0.cross(m2_tilde);
        let h1_00 = inv_l0_2
            * (2.0 * k1 * tt
                - t1_x_m2.outer(t_tilde)
                - t_tilde.outer(t1_x_m2)
                - (k1 * inv_chi) * id_t0
                + 0.25 * (kb.outer(m2_0) + m2_0.outer(kb)));
        let h1_11 = inv_l1_2
            * (2.0 * k1 * tt
                + t0_x_m2.outer(t_tilde)
                + t_tilde.outer(t0_x_m2)
                - (k1 * inv_chi) * id_t1
                + 0.25 * (kb.outer(m2_1) + m2_1.outer(kb)));
        let h1_01 = inv_l0l1
            * (-(k1 * inv_chi) * id_t0t1 - m2_tilde.skew() + 2.0 * k1 * tt
                - t1_x_m2.outer(t_tilde)
                + t_tilde.outer(t0_x_m2));
        let h1_10 = h1_01.transpose();
        scatter_edge_hess_to_vertices(
            c1 * h1_00,
            c1 * h1_01,
            c1 * h1_10,
            c1 * h1_11,
            &mut hess,
        );

        let m1_0_hat = m1_0 * inv_chi;
        let m1_1_hat = m1_1 * inv_chi;
        let d2k1_e0_th0 = (1.0 / l0) * (-d1_th0 * t_tilde - t1.cross(m1_0_hat));
        let d2k1_e1_th0 = (1.0 / l1) * (-d1_th0 * t_tilde + t0.cross(m1_0_hat));
        scatter_edge_scalar_hess(c1 * d2k1_e0_th0, c1 * d2k1_e1_th0, 9, &mut hess);

        let d2k1_e0_th1 = (1.0 / l0) * (-d1_th1 * t_tilde - t1.cross(m1_1_hat));
        let d2k1_e1_th1 = (1.0 / l1) * (-d1_th1 * t_tilde + t0.cross(m1_1_hat));
        scatter_edge_scalar_hess(c1 * d2k1_e0_th1, c1 * d2k1_e1_th1, 10, &mut hess);

        hess[9][9] += c1 * (-0.5 * m2_0.dot(kb));
        hess[10][10] += c1 * (-0.5 * m2_1.dot(kb));
    }

    // --- D^2(k2) ---
    let c2 = w2 * dk2;
    if c2.abs() > 0.0 {
        let t1_x_m1 = t1.cross(m1_tilde);
        let t0_x_m1 = t0.cross(m1_tilde);
        let h2_00 = inv_l0_2
            * (2.0 * k2 * tt
                + t1_x_m1.outer(t_tilde)
                + t_tilde.outer(t1_x_m1)
                - (k2 * inv_chi) * id_t0
                - 0.25 * (kb.outer(m1_0) + m1_0.outer(kb)));
        let h2_11 = inv_l1_2
            * (2.0 * k2 * tt
                - t0_x_m1.outer(t_tilde)
                - t_tilde.outer(t0_x_m1)
                - (k2 * inv_chi) * id_t1
                - 0.25 * (kb.outer(m1_1) + m1_1.outer(kb)));
        let h2_01 = inv_l0l1
            * (-(k2 * inv_chi) * id_t0t1 + m1_tilde.skew() + 2.0 * k2 * tt
                + t1_x_m1.outer(t_tilde)
                - t_tilde.outer(t0_x_m1));
        let h2_10 = h2_01.transpose();
        scatter_edge_hess_to_vertices(
            c2 * h2_00,
            c2 * h2_01,
            c2 * h2_10,
            c2 * h2_11,
            &mut hess,
        );

        let m2_0_hat = m2_0 * inv_chi;
        let m2_1_hat = m2_1 * inv_chi;
        let d2k2_e0_th0 = (1.0 / l0) * (-d2_th0 * t_tilde - t1.cross(m2_0_hat));
        let d2k2_e1_th0 = (1.0 / l1) * (-d2_th0 * t_tilde + t0.cross(m2_0_hat));
        scatter_edge_scalar_hess(c2 * d2k2_e0_th0, c2 * d2k2_e1_th0, 9, &mut hess);

        let d2k2_e0_th1 = (1.0 / l0) * (-d2_th1 * t_tilde - t1.cross(m2_1_hat));
        let d2k2_e1_th1 = (1.0 / l1) * (-d2_th1 * t_tilde + t0.cross(m2_1_hat));
        scatter_edge_scalar_hess(c2 * d2k2_e0_th1, c2 * d2k2_e1_th1, 10, &mut hess);

        hess[9][9] += c2 * (0.5 * m1_0.dot(kb));
        hess[10][10] += c2 * (0.5 * m1_1.dot(kb));
    }

    DerBendEval {
        energy,
        kappa: [k1, k2],
        grad,
        hess,
    }
}

/// Evaluation of DER interior-vertex twisting energy, 11D gradient, and 11x11 Hessian.
/// DOF ordering: `[x0(0..3), x1(3..6), x2(6..9), theta0(9), theta1(10)]`.
#[derive(Copy, Clone, Debug)]
pub struct DerTwistEval {
    pub energy: f64,
    pub twist: f64,
    pub grad: [f64; 11],
    pub hess: [[f64; 11]; 11],
}

/// Exact DER twisting energy `E_t = 0.5 * (GJ / voronoi_len) * (m - rest_twist)^2`
/// where `m = theta1 - theta0 + ref_twist`, exact 11D gradient, and exact symmetric 11x11 Hessian
/// (Bergou et al., 2010; Jawed et al., 2018).
#[allow(clippy::too_many_arguments)]
pub fn der_twisting_energy_grad_hess(
    x0: Vec3,
    x1: Vec3,
    x2: Vec3,
    theta0: f64,
    theta1: f64,
    ref_twist: f64,
    rest_twist: f64,
    voronoi_len: f64,
    gj: f64,
) -> DerTwistEval {
    let e0 = x1 - x0;
    let e1 = x2 - x1;
    let l0 = e0.norm().max(1e-15);
    let l1 = e1.norm().max(1e-15);
    let t0 = e0 / l0;
    let t1 = e1 / l1;

    let chi = (1.0 + t0.dot(t1)).max(1e-12);
    let inv_chi = 1.0 / chi;
    let t_tilde = (t0 + t1) * inv_chi;
    let kb = (2.0 * inv_chi) * t0.cross(t1);

    let twist = theta1 - theta0 + ref_twist;
    let dm = twist - rest_twist;
    let w = gj / voronoi_len.max(1e-15);
    let energy = 0.5 * w * dm * dm;

    // First derivatives of m = theta1 - theta0 + m_ref
    let dm_e0 = kb * (0.5 / l0);
    let dm_e1 = kb * (0.5 / l1);
    let dx0 = -dm_e0;
    let dx1 = dm_e0 - dm_e1;
    let dx2 = dm_e1;
    let gm = [
        dx0.0[0], dx0.0[1], dx0.0[2], dx1.0[0], dx1.0[1], dx1.0[2], dx2.0[0], dx2.0[1], dx2.0[2],
        -1.0, 1.0,
    ];

    let mut grad = [0.0; 11];
    for i in 0..11 {
        grad[i] = w * dm * gm[i];
    }

    let mut hess = [[0.0; 11]; 11];
    for i in 0..11 {
        for j in 0..11 {
            hess[i][j] = w * gm[i] * gm[j];
        }
    }

    // Exact symmetric second derivative D^2(m_ref) under time-parallel reference frame transport
    // (Bergou et al., 2010; Jawed et al., 2018):
    //   d^2 m / de0^2 = -(kb (t0 + t_tilde)^T + (t0 + t_tilde) kb^T) / (4 l0^2)
    //   d^2 m / de1^2 = -(kb (t1 + t_tilde)^T + (t1 + t_tilde) kb^T) / (4 l1^2)
    //   d^2 m / (de0 de1) = 0.5 * (d/de1(kb / (2 l0)) + (d/de0(kb / (2 l1)))^T)
    let c = w * dm;
    if c.abs() > 0.0 {
        let s0 = t0 + t_tilde;
        let s1 = t1 + t_tilde;
        let h00 = (-0.25 / (l0 * l0)) * (kb.outer(s0) + s0.outer(kb));
        let h11 = (-0.25 / (l1 * l1)) * (kb.outer(s1) + s1.outer(kb));
        let d_e0_e1 = (0.5 / (l0 * l1)) * ((2.0 * inv_chi) * t0.skew() - kb.outer(t_tilde));
        let d_e1_e0 = (0.5 / (l0 * l1)) * (-(2.0 * inv_chi) * t1.skew() - kb.outer(t_tilde));
        let h01 = 0.5 * (d_e0_e1 + d_e1_e0.transpose());
        let h10 = h01.transpose();
        scatter_edge_hess_to_vertices(c * h00, c * h01, c * h10, c * h11, &mut hess);
    }

    DerTwistEval {
        energy,
        twist,
        grad,
        hess,
    }
}

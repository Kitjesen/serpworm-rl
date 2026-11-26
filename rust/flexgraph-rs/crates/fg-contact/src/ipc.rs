//! Codimensional IPC $C^2$ clamped log-barrier (Li et al., SIGGRAPH 2020/2021),
//! Py-DiSMech $C^2$ smooth soft-barrier (Lahoti et al., 2025 Eq. 25–27),
//! IPC $C^1$ mollified friction with exact SPSD Hessians, and full-mesh `SelfContactEngine`.

use crate::broadphase::SpatialHashGrid;
use crate::ccd::compute_safe_step_segments;
use crate::distance::{
    edge_edge_mollifier, point_point_distance, point_segment_distance,
    point_triangle_distance_grad_hess, segment_segment_distance, TriangleFeature,
};
use crate::spd::project_spd_n;
use fg_math::{Mat3, Vec3};
use fg_sdf::SdfField;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Barrier formulation selector for Engine A self-contact and SDF obstacle contact.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarrierKind {
    /// Exact C-IPC $C^2$ clamped log-barrier $b(g) = -(g - \hat{d})^2 \ln(g / \hat{d})$
    /// for $0 < g = d - \xi < \hat{d}$ (guaranteed zero interpenetration with CCD).
    IpcLogBarrier,
    /// Py-DiSMech $C^2$ smooth soft-penalty barrier $E(d) = [\frac{1}{K_1}\ln(1 + e^{K_1(d_0 - d)})]^2$
    /// with $K_1 = 15 / \delta$ (line-search-free implicit assembly).
    PyDiSMechSoftBarrier,
}

/// Configuration parameters for IPC / Py-DiSMech barrier contact.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IpcBarrierConfig {
    pub kind: BarrierKind,
    /// Barrier stiffness $\kappa$ (N/m or scaled energy weight).
    pub stiffness: f64,
    /// Activation gap $\hat{d}$ (IPC) or transition half-width $\delta$ (Py-DiSMech) in meters.
    pub d_hat: f64,
    /// Hard codimensional thickness offset $\xi$ (IPC) or contact diameter $d_0 = 2h$ (Py-DiSMech) in meters.
    pub thickness: f64,
    /// Edge-edge parallel mollifier threshold $\epsilon_x$ (`0.0` disables mollifier).
    pub mollifier_eps_x: f64,
}

impl Default for IpcBarrierConfig {
    fn default() -> Self {
        Self {
            kind: BarrierKind::IpcLogBarrier,
            stiffness: 1.0e4,
            d_hat: 0.005,
            thickness: 0.002,
            mollifier_eps_x: 1.0e-5,
        }
    }
}

/// Exact C-IPC $C^2$ clamped log-barrier:
/// $b(g) = -(g - \hat{d})^2 \ln(g / \hat{d})$ where $g = d - \xi$.
///
/// Returns `Some((b, db/dd, d2b/dd2))` if `g > 0`, or `None` if penetrated (`g <= 0`).
#[inline]
pub fn ipc_barrier_eval(d: f64, d_hat: f64, thickness: f64) -> Option<(f64, f64, f64)> {
    let g = d - thickness;
    if g <= 0.0 {
        return None;
    }
    if g >= d_hat || d_hat <= 0.0 {
        return Some((0.0, 0.0, 0.0));
    }
    let u = g - d_hat; // < 0
    let ln_ratio = (g / d_hat).ln(); // < 0
    let b = -(u * u) * ln_ratio;
    let db = -2.0 * u * ln_ratio - (u * u) / g;
    // Exact second derivative: -2 ln(g/d_hat) - 4(g - d_hat)/g + (g - d_hat)^2 / g^2 > 0
    let d2b = -2.0 * ln_ratio - 4.0 * u / g + (u * u) / (g * g);
    Some((b, db, d2b))
}

/// Py-DiSMech $C^2$ smooth soft-penalty barrier (Lahoti et al. 2025, Eq. 25):
/// $s(d) = \frac{1}{K_1} \ln(1 + e^{K_1 (d_0 - d)})$, $E(d) = s(d)^2$, $K_1 = 15 / \delta$.
///
/// Uses overflow-free softplus evaluation so derivatives are exact $C^\infty$ across the full
/// domain (matching Eq. 25 to `< 1e-7` at `d_0 ± delta` and bit-exact outside `±36 / K_1`).
#[inline]
pub fn pydismech_barrier_eval(d: f64, d0: f64, delta: f64) -> (f64, f64, f64) {
    let delta = delta.max(1e-9);
    let k1 = 15.0 / delta;
    let gap = d0 - d;
    let z = k1 * gap;
    if z < -36.0 {
        return (0.0, 0.0, 0.0);
    }
    if z > 36.0 {
        // E = gap^2, dE/dd = -2 * gap, d2E/dd2 = 2
        return (gap * gap, -2.0 * gap, 2.0);
    }
    // Stable softplus & sigmoid
    let (s, sig) = if z >= 0.0 {
        let e_neg = (-z).exp();
        (gap + e_neg.ln_1p() / k1, 1.0 / (1.0 + e_neg))
    } else {
        let e_pos = z.exp();
        (e_pos.ln_1p() / k1, e_pos / (1.0 + e_pos))
    };
    let energy = s * s;
    let de = -2.0 * s * sig;
    let d2e = 2.0 * (sig * sig + k1 * s * sig * (1.0 - sig));
    (energy, de, d2e)
}

#[inline]
fn eval_scalar_barrier(d: f64, cfg: &IpcBarrierConfig) -> Option<(f64, f64, f64)> {
    match cfg.kind {
        BarrierKind::IpcLogBarrier => {
            let (b, db, d2b) = ipc_barrier_eval(d, cfg.d_hat, cfg.thickness)?;
            Some((cfg.stiffness * b, cfg.stiffness * db, cfg.stiffness * d2b))
        }
        BarrierKind::PyDiSMechSoftBarrier => {
            let (b, db, d2b) = pydismech_barrier_eval(d, cfg.thickness, cfg.d_hat);
            Some((cfg.stiffness * b, cfg.stiffness * db, cfg.stiffness * d2b))
        }
    }
}

/// Evaluate IPC $C^1$ smoothed mollifier friction potential, 3D gradient, and exact SPSD 3x3 Hessian
/// (Li et al., SIGGRAPH 2020, Sec. 5.2).
///
/// - `u_rel`: relative displacement over the step `x_rel - x_rel_prev` (or `dt * v_rel`);
/// - `normal`: unit contact normal;
/// - `lambda_n`: lagged positive normal force magnitude `>= 0`;
/// - `mu`: Coulomb friction coefficient;
/// - `eps_u`: displacement mollifier threshold `eps_v * dt > 0`.
#[derive(Copy, Clone, Debug)]
pub struct IpcFrictionEval {
    pub energy: f64,
    /// Gradient `dD / du_rel` (opposing friction force on the moving point is `-grad`).
    pub grad: Vec3,
    /// Exact analytically symmetric positive semi-definite 3x3 Hessian `d^2 D / du_rel^2`.
    pub hess: Mat3,
}

pub fn ipc_mollifier_friction_eval(
    u_rel: Vec3,
    normal: Vec3,
    lambda_n: f64,
    mu: f64,
    eps_u: f64,
) -> IpcFrictionEval {
    let coeff = mu * lambda_n;
    if coeff <= 0.0 || eps_u <= 0.0 {
        return IpcFrictionEval {
            energy: 0.0,
            grad: Vec3::ZERO,
            hess: Mat3::ZERO,
        };
    }
    let n = normal.normalized_or(Vec3::EZ);
    let proj_t = Mat3::IDENTITY - n.outer(n);
    let ut = proj_t.mul_vec(u_rel);
    let y = ut.norm();

    let (f0, f1_over_y, coeff_uu) = if y < eps_u {
        let inv_e = 1.0 / eps_u;
        let inv_e2 = inv_e * inv_e;
        let f0 = -(y * y * y) * (inv_e2 / 3.0) + (y * y) * inv_e;
        // f1(y) / y = -y / eps_u^2 + 2 / eps_u (smooth at y = 0!)
        let f1_ov_y = -y * inv_e2 + 2.0 * inv_e;
        // (f1'(y) * y - f1(y)) / y^3 = (-2y/e^2 + 2/e)*y - (-y^2/e^2 + 2y/e) = -y^2/e^2 => / y^3 = -1/(y * e^2)
        // Wait: ut * ut^T * (-1/(y*e^2)) = -(u_hat u_hat^T) * (y / e^2) (smooth and zero at y = 0!)
        let c_uu = if y > 1e-15 { -inv_e2 / y } else { 0.0 };
        (f0, f1_ov_y, c_uu)
    } else {
        let f0 = y - eps_u / 3.0;
        let f1_ov_y = 1.0 / y;
        let c_uu = -1.0 / (y * y * y);
        (f0, f1_ov_y, c_uu)
    };

    let energy = coeff * f0;
    let grad = ut * (coeff * f1_over_y);
    let hess = (proj_t * f1_over_y + ut.outer(ut) * coeff_uu) * coeff;
    IpcFrictionEval {
        energy,
        grad,
        hess: hess.symmetrize(),
    }
}

/// Py-DiSMech smooth Coulomb friction force and exact SPD tangential damping Jacobian (Eq. 27):
/// `F_fr = -mu * (2 / (1 + exp(-K2 * ||v_t||)) - 1) * v_t_hat * F_n`, `K2 = 15 / v_thresh`.
pub fn pydismech_friction_eval(
    v_rel: Vec3,
    normal: Vec3,
    fn_mag: f64,
    mu: f64,
    v_thresh: f64,
) -> (Vec3, Mat3) {
    let cap = mu * fn_mag;
    if cap <= 0.0 || v_thresh <= 0.0 {
        return (Vec3::ZERO, Mat3::ZERO);
    }
    let n = normal.normalized_or(Vec3::EZ);
    let proj_t = Mat3::IDENTITY - n.outer(n);
    let vt = proj_t.mul_vec(v_rel);
    let speed = vt.norm();
    let k2 = 15.0 / v_thresh;
    // Notice: 2 / (1 + exp(-K2 * s)) - 1 == tanh(0.5 * K2 * s) !
    let half_k2 = 0.5 * k2;
    let (force, df_dv) = if speed < 1e-12 {
        // Smooth linear limit at zero speed: F = -cap * half_k2 * vt
        let c0 = cap * half_k2;
        (-vt * c0, proj_t * c0)
    } else {
        let th = (half_k2 * speed).tanh();
        let sech2 = (1.0 - th * th).max(0.0);
        let dir = vt / speed;
        let force = -dir * (cap * th);
        // -dF/dv_t = cap * (half_k2 * sech^2 * dir dir^T + (th / speed) * (proj_t - dir dir^T)) (strictly SPSD!)
        let dd = dir.outer(dir);
        let c_par = cap * half_k2 * sech2;
        let c_perp = cap * th / speed;
        let jac = dd * c_par + (proj_t - dd) * c_perp;
        (force, jac.symmetrize())
    };
    (force, df_dv)
}

/// Evaluation output for a Point-vs-SDF barrier contact (`3D` DOF).
#[derive(Copy, Clone, Debug)]
pub struct PointSdfBarrierEval {
    pub active: bool,
    pub dist: f64,
    pub normal: Vec3,
    pub energy: f64,
    pub grad: Vec3,
    pub hess_exact: Mat3,
    pub hess_spd: Mat3,
}

/// Evaluate barrier energy, gradient, exact 3x3 Hessian, and SPD-projected 3x3 Hessian
/// for a point `pos` against an arbitrary [`SdfField`].
pub fn ipc_point_sdf_barrier(
    sdf: &dyn SdfField,
    pos: Vec3,
    cfg: &IpcBarrierConfig,
) -> PointSdfBarrierEval {
    let q = sdf.query(pos);
    let Some((energy, de, d2e)) = eval_scalar_barrier(q.phi, cfg) else {
        // Fallback high-stiffness penalty if already penetrated past thickness offset.
        let pen = (cfg.thickness - q.phi).max(0.0);
        let k = cfg.stiffness * 100.0;
        let nn = q.normal.outer(q.normal);
        return PointSdfBarrierEval {
            active: true,
            dist: q.phi,
            normal: q.normal,
            energy: 0.5 * k * pen * pen,
            grad: -q.normal * (k * pen),
            hess_exact: nn * k,
            hess_spd: nn * k,
        };
    };
    if energy == 0.0 && de == 0.0 && d2e == 0.0 {
        return PointSdfBarrierEval {
            active: false,
            dist: q.phi,
            normal: q.normal,
            energy: 0.0,
            grad: Vec3::ZERO,
            hess_exact: Mat3::ZERO,
            hess_spd: Mat3::ZERO,
        };
    }
    let grad = q.normal * de;
    let hess_exact = q.normal.outer(q.normal) * d2e;
    let hess_spd = hess_exact.project_spd(0.0);
    PointSdfBarrierEval {
        active: true,
        dist: q.phi,
        normal: q.normal,
        energy,
        grad,
        hess_exact,
        hess_spd,
    }
}

/// Evaluation output for a Point-Point barrier contact (`6D` DOF: `[xa, xb]`).
#[derive(Copy, Clone, Debug)]
pub struct PointPointBarrierEval {
    pub active: bool,
    pub dist: f64,
    pub normal: Vec3,
    pub energy: f64,
    pub grad: [f64; 6],
    pub hess_exact: [[f64; 6]; 6],
    pub hess_spd: [[f64; 6]; 6],
}

pub fn ipc_point_point_barrier(
    xa: Vec3,
    xb: Vec3,
    cfg: &IpcBarrierConfig,
) -> PointPointBarrierEval {
    let d_eval = point_point_distance(xa, xb);
    let Some((energy, de, d2e)) = eval_scalar_barrier(d_eval.dist, cfg) else {
        return PointPointBarrierEval {
            active: false,
            dist: d_eval.dist,
            normal: d_eval.normal,
            energy: f64::INFINITY,
            grad: [0.0; 6],
            hess_exact: [[0.0; 6]; 6],
            hess_spd: [[0.0; 6]; 6],
        };
    };
    if energy == 0.0 && de == 0.0 && d2e == 0.0 {
        return PointPointBarrierEval {
            active: false,
            dist: d_eval.dist,
            normal: d_eval.normal,
            energy: 0.0,
            grad: [0.0; 6],
            hess_exact: [[0.0; 6]; 6],
            hess_spd: [[0.0; 6]; 6],
        };
    }
    let mut grad = [0.0; 6];
    let mut hess_exact = [[0.0; 6]; 6];
    for i in 0..6 {
        grad[i] = de * d_eval.grad[i];
        for j in 0..6 {
            hess_exact[i][j] = d2e * d_eval.grad[i] * d_eval.grad[j] + de * d_eval.hess[i][j];
        }
    }
    let hess_spd = project_spd_n(hess_exact, 0.0);
    PointPointBarrierEval {
        active: true,
        dist: d_eval.dist,
        normal: d_eval.normal,
        energy,
        grad,
        hess_exact,
        hess_spd,
    }
}

/// Evaluation output for a Point-Segment barrier contact (`9D` DOF: `[p, a, b]`).
#[derive(Copy, Clone, Debug)]
pub struct PointSegmentBarrierEval {
    pub active: bool,
    pub dist: f64,
    pub t: f64,
    pub normal: Vec3,
    pub energy: f64,
    pub grad: [f64; 9],
    pub hess_exact: [[f64; 9]; 9],
    pub hess_spd: [[f64; 9]; 9],
}

pub fn ipc_point_segment_barrier(
    p: Vec3,
    a: Vec3,
    b: Vec3,
    cfg: &IpcBarrierConfig,
) -> PointSegmentBarrierEval {
    let d_eval = point_segment_distance(p, a, b);
    let Some((energy, de, d2e)) = eval_scalar_barrier(d_eval.dist, cfg) else {
        return PointSegmentBarrierEval {
            active: false,
            dist: d_eval.dist,
            t: d_eval.t,
            normal: d_eval.normal,
            energy: f64::INFINITY,
            grad: [0.0; 9],
            hess_exact: [[0.0; 9]; 9],
            hess_spd: [[0.0; 9]; 9],
        };
    };
    if energy == 0.0 && de == 0.0 && d2e == 0.0 {
        return PointSegmentBarrierEval {
            active: false,
            dist: d_eval.dist,
            t: d_eval.t,
            normal: d_eval.normal,
            energy: 0.0,
            grad: [0.0; 9],
            hess_exact: [[0.0; 9]; 9],
            hess_spd: [[0.0; 9]; 9],
        };
    }
    let mut grad = [0.0; 9];
    let mut hess_exact = [[0.0; 9]; 9];
    for i in 0..9 {
        grad[i] = de * d_eval.grad[i];
        for j in 0..9 {
            hess_exact[i][j] = d2e * d_eval.grad[i] * d_eval.grad[j] + de * d_eval.hess[i][j];
        }
    }
    let hess_spd = project_spd_n(hess_exact, 0.0);
    PointSegmentBarrierEval {
        active: true,
        dist: d_eval.dist,
        t: d_eval.t,
        normal: d_eval.normal,
        energy,
        grad,
        hess_exact,
        hess_spd,
    }
}

/// Evaluation output for a Segment-Segment (Edge-Edge) barrier contact (`12D` DOF: `[a0, a1, b0, b1]`).
#[derive(Copy, Clone, Debug)]
pub struct SegmentSegmentBarrierEval {
    pub active: bool,
    pub dist: f64,
    pub s: f64,
    pub t: f64,
    pub normal: Vec3,
    pub mollifier: f64,
    pub energy: f64,
    pub grad: [f64; 12],
    pub hess_exact: [[f64; 12]; 12],
    pub hess_spd: [[f64; 12]; 12],
}

/// Evaluate Segment-Segment barrier energy `m(c) * B(d)` with exact analytical 12D gradient,
/// exact 12x12 product-rule Hessian, and SPD-projected 12x12 Hessian.
pub fn ipc_segment_segment_barrier(
    a0: Vec3,
    a1: Vec3,
    b0: Vec3,
    b1: Vec3,
    cfg: &IpcBarrierConfig,
) -> SegmentSegmentBarrierEval {
    let d_eval = segment_segment_distance(a0, a1, b0, b1);
    let Some((b_val, db, d2b)) = eval_scalar_barrier(d_eval.dist, cfg) else {
        return SegmentSegmentBarrierEval {
            active: false,
            dist: d_eval.dist,
            s: d_eval.s,
            t: d_eval.t,
            normal: d_eval.normal,
            mollifier: 1.0,
            energy: f64::INFINITY,
            grad: [0.0; 12],
            hess_exact: [[0.0; 12]; 12],
            hess_spd: [[0.0; 12]; 12],
        };
    };
    if b_val == 0.0 && db == 0.0 && d2b == 0.0 {
        return SegmentSegmentBarrierEval {
            active: false,
            dist: d_eval.dist,
            s: d_eval.s,
            t: d_eval.t,
            normal: d_eval.normal,
            mollifier: 1.0,
            energy: 0.0,
            grad: [0.0; 12],
            hess_exact: [[0.0; 12]; 12],
            hess_spd: [[0.0; 12]; 12],
        };
    }

    let mut gb = [0.0; 12];
    let mut hb = [[0.0; 12]; 12];
    for i in 0..12 {
        gb[i] = db * d_eval.grad[i];
        for j in 0..12 {
            hb[i][j] = d2b * d_eval.grad[i] * d_eval.grad[j] + db * d_eval.hess[i][j];
        }
    }

    let mol = edge_edge_mollifier(a0, a1, b0, b1, cfg.mollifier_eps_x);
    let energy = mol.value * b_val;
    let mut grad = [0.0; 12];
    let mut hess_exact = [[0.0; 12]; 12];
    for i in 0..12 {
        grad[i] = mol.value * gb[i] + b_val * mol.grad[i];
        for j in 0..12 {
            hess_exact[i][j] = mol.value * hb[i][j]
                + mol.grad[i] * gb[j]
                + gb[i] * mol.grad[j]
                + b_val * mol.hess[i][j];
        }
    }
    let hess_spd = project_spd_n(hess_exact, 0.0);

    SegmentSegmentBarrierEval {
        active: true,
        dist: d_eval.dist,
        s: d_eval.s,
        t: d_eval.t,
        normal: d_eval.normal,
        mollifier: mol.value,
        energy,
        grad,
        hess_exact,
        hess_spd,
    }
}

/// Evaluation output for a Point-Triangle (Vertex-Face) barrier contact
/// (`12D` DOF ordered `[p, t0, t1, t2]`).
#[derive(Copy, Clone, Debug)]
pub struct PointTriangleBarrierEval {
    pub active: bool,
    pub dist: f64,
    pub bary: [f64; 3],
    pub normal: Vec3,
    pub feature: TriangleFeature,
    pub energy: f64,
    pub grad: [f64; 12],
    pub hess_exact: [[f64; 12]; 12],
    pub hess_spd: [[f64; 12]; 12],
}

/// Evaluate Point-Triangle (Vertex-Face) IPC / Py-DiSMech barrier energy, exact 12D analytical
/// gradient, exact 12x12 analytical Hessian, and SPD-projected 12x12 Hessian.
pub fn ipc_point_triangle_barrier(
    p: Vec3,
    t0: Vec3,
    t1: Vec3,
    t2: Vec3,
    cfg: &IpcBarrierConfig,
) -> PointTriangleBarrierEval {
    let d_eval = point_triangle_distance_grad_hess(p, t0, t1, t2);
    let Some((energy, de, d2e)) = eval_scalar_barrier(d_eval.dist, cfg) else {
        return PointTriangleBarrierEval {
            active: false,
            dist: d_eval.dist,
            bary: d_eval.bary,
            normal: d_eval.normal,
            feature: d_eval.feature,
            energy: f64::INFINITY,
            grad: [0.0; 12],
            hess_exact: [[0.0; 12]; 12],
            hess_spd: [[0.0; 12]; 12],
        };
    };
    if energy == 0.0 && de == 0.0 && d2e == 0.0 {
        return PointTriangleBarrierEval {
            active: false,
            dist: d_eval.dist,
            bary: d_eval.bary,
            normal: d_eval.normal,
            feature: d_eval.feature,
            energy: 0.0,
            grad: [0.0; 12],
            hess_exact: [[0.0; 12]; 12],
            hess_spd: [[0.0; 12]; 12],
        };
    }

    let mut grad = [0.0; 12];
    let mut hess_exact = [[0.0; 12]; 12];
    for i in 0..12 {
        grad[i] = de * d_eval.grad[i];
        for j in 0..12 {
            hess_exact[i][j] = d2e * d_eval.grad[i] * d_eval.grad[j] + de * d_eval.hess[i][j];
        }
    }
    let hess_spd = project_spd_n(hess_exact, 0.0);

    PointTriangleBarrierEval {
        active: true,
        dist: d_eval.dist,
        bary: d_eval.bary,
        normal: d_eval.normal,
        feature: d_eval.feature,
        energy,
        grad,
        hess_exact,
        hess_spd,
    }
}

/// Sparse 3x3 Hessian block triplet `(node_i, node_j, K_ij)`.
#[derive(Copy, Clone, Debug)]
pub struct SelfContactTriplet {
    pub node_i: usize,
    pub node_j: usize,
    pub block: Mat3,
}

/// Assembled global self-contact barrier + friction potential, forces, and SPD Hessians.
#[derive(Clone, Debug)]
pub struct SelfContactAssembly {
    pub active_pairs: usize,
    pub total_energy: f64,
    /// Repulsive + frictional world forces (`-grad E`) per node.
    pub forces: Vec<Vec3>,
    /// Diagonal 3x3 SPD stiffness blocks per node (ready for direct Block-Thomas diagonal injection).
    pub diag_hess_spd: Vec<Mat3>,
    /// Full off-diagonal + diagonal 3x3 SPD Hessian triplets (for exact Newton assembly).
    pub triplets: Vec<SelfContactTriplet>,
}

/// Spatial-hash-accelerated beam/rod self-contact engine supporting both C-IPC log-barrier
/// (with conservative ACCD step clamping) and Py-DiSMech $C^2$ soft-barrier + mollified friction.
#[derive(Clone, Debug)]
pub struct SelfContactEngine {
    pub barrier: IpcBarrierConfig,
    pub mu: f64,
    pub friction_eps_v: f64,
    pub ccd_s_conserv: f64,
}

impl Default for SelfContactEngine {
    fn default() -> Self {
        Self {
            barrier: IpcBarrierConfig::default(),
            mu: 0.25,
            friction_eps_v: 1.0e-2,
            ccd_s_conserv: 0.1,
        }
    }
}

impl SelfContactEngine {
    pub fn new(barrier: IpcBarrierConfig, mu: f64, friction_eps_v: f64) -> Self {
        Self {
            barrier,
            mu: mu.max(0.0),
            friction_eps_v: friction_eps_v.max(1e-6),
            ccd_s_conserv: 0.1,
        }
    }

    /// Compute conservative intersection-free CCD step fraction `t_safe in [0, 1]`
    /// from `x0` to `x1` across all beam segments.
    pub fn ccd_safe_step(&self, x0: &[Vec3], x1: &[Vec3], segments: &[(usize, usize)]) -> f64 {
        compute_safe_step_segments(x0, x1, segments, self.barrier.thickness, self.ccd_s_conserv)
    }

    /// Assemble barrier + lagged mollifier friction energy, forces (`-grad`), diagonal 3x3 SPD
    /// blocks, and full 3x3 SPD Hessian triplets for all active non-adjacent segment pairs.
    pub fn assemble_segments(
        &self,
        points: &[Vec3],
        prev_points: Option<&[Vec3]>,
        segments: &[(usize, usize)],
        dt: f64,
    ) -> SelfContactAssembly {
        let n_pts = points.len();
        let mut forces = vec![Vec3::ZERO; n_pts];
        let mut diag_hess_spd = vec![Mat3::ZERO; n_pts];
        let mut triplets = Vec::new();
        let mut total_energy = 0.0;
        let mut active_pairs = 0usize;

        if segments.len() < 2 {
            return SelfContactAssembly {
                active_pairs,
                total_energy,
                forces,
                diag_hess_spd,
                triplets,
            };
        }

        let mut max_len = 0.0_f64;
        for &(u, v) in segments {
            max_len = max_len.max((points[v] - points[u]).norm());
        }
        let inflate = self.barrier.thickness + self.barrier.d_hat;
        let cell_size = (max_len + 2.0 * inflate).max(0.02);
        let mut grid = SpatialHashGrid::new(cell_size);
        let pairs = grid.find_segment_pairs(points, segments, inflate);

        let eps_u = self.friction_eps_v * dt.max(1e-9);
        // Track deduplicated (vertex_id, segment_id) Point-Segment stencils per C-IPC (Li et al. 2020/2021 Sec. 4.1)
        // so parallel edges (where the EE mollifier smoothly fades to 0) retain exact unmollified endpoint-segment repulsion!
        let mut seen_pe: HashSet<(usize, usize)> = HashSet::with_capacity(pairs.len() * 4);

        for (s0, s1) in pairs {
            let (u0, v0) = segments[s0];
            let (u1, v1) = segments[s1];

            // 1. Mollified Edge-Edge barrier + lagged friction
            let ev = ipc_segment_segment_barrier(
                points[u0],
                points[v0],
                points[u1],
                points[v1],
                &self.barrier,
            );
            if ev.active && ev.energy.is_finite() && ev.mollifier > 0.0 {
                active_pairs += 1;
                total_energy += ev.energy;

                let nodes = [u0, v0, u1, v1];
                for bi in 0..4 {
                    let ni = nodes[bi];
                    forces[ni] -= Vec3::new(
                        ev.grad[bi * 3],
                        ev.grad[bi * 3 + 1],
                        ev.grad[bi * 3 + 2],
                    );
                    for bj in 0..4 {
                        let nj = nodes[bj];
                        let mut blk = [[0.0; 3]; 3];
                        for r in 0..3 {
                            for c in 0..3 {
                                blk[r][c] = ev.hess_spd[bi * 3 + r][bj * 3 + c];
                            }
                        }
                        let m = Mat3(blk);
                        if ni == nj {
                            diag_hess_spd[ni] += m;
                        }
                        triplets.push(SelfContactTriplet {
                            node_i: ni,
                            node_j: nj,
                            block: m,
                        });
                    }
                }

                if self.mu > 0.0 {
                    if let Some(prev) = prev_points {
                        let f_a = Vec3::new(
                            ev.grad[0] + ev.grad[3],
                            ev.grad[1] + ev.grad[4],
                            ev.grad[2] + ev.grad[5],
                        );
                        let lambda_n = (-f_a.dot(ev.normal)).max(0.0);
                        if lambda_n > 0.0 {
                            let w = [1.0 - ev.s, ev.s, -(1.0 - ev.t), -ev.t];
                            let mut u_rel = Vec3::ZERO;
                            for bi in 0..4 {
                                u_rel += (points[nodes[bi]] - prev[nodes[bi]]) * w[bi];
                            }
                            let fr = ipc_mollifier_friction_eval(
                                u_rel,
                                ev.normal,
                                lambda_n,
                                self.mu,
                                eps_u,
                            );
                            total_energy += fr.energy;
                            for bi in 0..4 {
                                let ni = nodes[bi];
                                forces[ni] -= fr.grad * w[bi];
                                for bj in 0..4 {
                                    let nj = nodes[bj];
                                    let m = fr.hess * (w[bi] * w[bj]);
                                    if ni == nj {
                                        diag_hess_spd[ni] += m;
                                    }
                                    triplets.push(SelfContactTriplet {
                                        node_i: ni,
                                        node_j: nj,
                                        block: m,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // 2. Deduplicated unmollified endpoint Point-Segment barriers (eliminates parallel-rod ghosting!)
            let pe_candidates = [(u0, s1), (v0, s1), (u1, s0), (v1, s0)];
            for (vid, sid) in pe_candidates {
                let (ea, eb) = segments[sid];
                if vid == ea || vid == eb || !seen_pe.insert((vid, sid)) {
                    continue;
                }
                let pev = ipc_point_segment_barrier(
                    points[vid],
                    points[ea],
                    points[eb],
                    &self.barrier,
                );
                if !pev.active || !pev.energy.is_finite() {
                    continue;
                }
                active_pairs += 1;
                total_energy += pev.energy;
                let p_nodes = [vid, ea, eb];
                for bi in 0..3 {
                    let ni = p_nodes[bi];
                    forces[ni] -= Vec3::new(
                        pev.grad[bi * 3],
                        pev.grad[bi * 3 + 1],
                        pev.grad[bi * 3 + 2],
                    );
                    for bj in 0..3 {
                        let nj = p_nodes[bj];
                        let mut blk = [[0.0; 3]; 3];
                        for r in 0..3 {
                            for c in 0..3 {
                                blk[r][c] = pev.hess_spd[bi * 3 + r][bj * 3 + c];
                            }
                        }
                        let m = Mat3(blk);
                        if ni == nj {
                            diag_hess_spd[ni] += m;
                        }
                        triplets.push(SelfContactTriplet {
                            node_i: ni,
                            node_j: nj,
                            block: m,
                        });
                    }
                }

                if self.mu > 0.0 {
                    if let Some(prev) = prev_points {
                        let f_p = Vec3::new(pev.grad[0], pev.grad[1], pev.grad[2]);
                        let lambda_n = (-f_p.dot(pev.normal)).max(0.0);
                        if lambda_n > 0.0 {
                            let w = [1.0, -(1.0 - pev.t), -pev.t];
                            let mut u_rel = Vec3::ZERO;
                            for bi in 0..3 {
                                u_rel += (points[p_nodes[bi]] - prev[p_nodes[bi]]) * w[bi];
                            }
                            let fr = ipc_mollifier_friction_eval(
                                u_rel,
                                pev.normal,
                                lambda_n,
                                self.mu,
                                eps_u,
                            );
                            total_energy += fr.energy;
                            for bi in 0..3 {
                                let ni = p_nodes[bi];
                                forces[ni] -= fr.grad * w[bi];
                                for bj in 0..3 {
                                    let nj = p_nodes[bj];
                                    let m = fr.hess * (w[bi] * w[bj]);
                                    if ni == nj {
                                        diag_hess_spd[ni] += m;
                                    }
                                    triplets.push(SelfContactTriplet {
                                        node_i: ni,
                                        node_j: nj,
                                        block: m,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        SelfContactAssembly {
            active_pairs,
            total_energy,
            forces,
            diag_hess_spd,
            triplets,
        }
    }
}

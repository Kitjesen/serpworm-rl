//! Cubic-Hermite corotational 3D beam kinematics, exact analytic strain Jacobian
//! (including midpoint spline-frame derivatives), and heterogeneous 12x12 local
//! stiffness/mass matrices with optional Kirchhoff-Love/Vlasov thin-shell and
//! Timoshenko shear-deformation corrections.
//!
//! Designed for `< 1e-12` FP64 parity with `serpworm.physics.gpu_core` (BeamAdapter lineage).

use crate::lie::{so3_dexp_inv, Quat};
use crate::spatial::plucker_motion_transform;
use crate::vec::{Mat12, Mat3, Vec12, Vec3, Vec6};
use serde::{Deserialize, Serialize};

/// Midpoint corotational frame of a 2-node cubic-Hermite beam element.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CorotationalFrame {
    pub center: Vec3,
    pub rot: Quat,
    pub compressed: bool,
}

/// Beam cross-section geometry.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BeamSection {
    /// Thin rectangular ribbon/strip of width `width` (local y) and thickness `thickness` (local z).
    Rectangular { width: f64, thickness: f64 },
    /// Solid circular rod of radius `radius`.
    Circular { radius: f64 },
    /// Hollow circular tube (e.g. surgical catheter).
    HollowCircular {
        outer_radius: f64,
        inner_radius: f64,
    },
    /// Arbitrary cross-section specified directly by geometric moments.
    General {
        area: f64,
        iy: f64,
        iz: f64,
        j_torsion: f64,
        shear_ky: f64,
        shear_kz: f64,
    },
}

/// Resolved section properties `(area, iy, iz, j_torsion, shear_ky, shear_kz, width_for_vlasov)`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SectionProperties {
    pub area: f64,
    pub iy: f64,
    pub iz: f64,
    pub j_torsion: f64,
    pub shear_ky: f64,
    pub shear_kz: f64,
    pub thin_width: Option<f64>,
}

impl BeamSection {
    pub fn properties(&self) -> SectionProperties {
        match *self {
            BeamSection::Rectangular { width, thickness } => {
                let area = width * thickness;
                let iy = width * thickness.powi(3) / 12.0;
                let iz = thickness * width.powi(3) / 12.0;
                // Exact Saint-Venant torsion constant for rectangle (exact match to gpu_core.py when w >= t)
                let (b_long, b_short) = if width >= thickness {
                    (width, thickness)
                } else {
                    (thickness, width)
                };
                let ratio = b_short / b_long.max(1e-15);
                let j_torsion = b_long
                    * b_short.powi(3)
                    * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio.powi(4) / 12.0));
                SectionProperties {
                    area,
                    iy,
                    iz,
                    j_torsion,
                    shear_ky: 5.0 / 6.0,
                    shear_kz: 5.0 / 6.0,
                    thin_width: Some(width),
                }
            }
            BeamSection::Circular { radius } => {
                let pi = std::f64::consts::PI;
                let r2 = radius * radius;
                let r4 = r2 * r2;
                let i = 0.25 * pi * r4;
                SectionProperties {
                    area: pi * r2,
                    iy: i,
                    iz: i,
                    j_torsion: 0.5 * pi * r4,
                    shear_ky: 0.9,
                    shear_kz: 0.9,
                    thin_width: None,
                }
            }
            BeamSection::HollowCircular {
                outer_radius,
                inner_radius,
            } => {
                let pi = std::f64::consts::PI;
                let ro2 = outer_radius * outer_radius;
                let ri2 = inner_radius * inner_radius;
                let diff4 = ro2 * ro2 - ri2 * ri2;
                let i = 0.25 * pi * diff4;
                SectionProperties {
                    area: pi * (ro2 - ri2),
                    iy: i,
                    iz: i,
                    j_torsion: 0.5 * pi * diff4,
                    shear_ky: 0.5,
                    shear_kz: 0.5,
                    thin_width: None,
                }
            }
            BeamSection::General {
                area,
                iy,
                iz,
                j_torsion,
                shear_ky,
                shear_kz,
            } => SectionProperties {
                area,
                iy,
                iz,
                j_torsion,
                shear_ky,
                shear_kz,
                thin_width: None,
            },
        }
    }
}

/// Linear isotropic elastic material + Rayleigh damping parameters.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BeamMaterial {
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    pub density: f64,
    #[serde(default = "default_rayleigh_alpha")]
    pub rayleigh_alpha: f64,
    #[serde(default = "default_rayleigh_beta")]
    pub rayleigh_beta: f64,
}

fn default_rayleigh_alpha() -> f64 {
    0.05
}

fn default_rayleigh_beta() -> f64 {
    0.002
}

impl BeamMaterial {
    /// Spring steel (65Mn / 301) matching `assets/inputs/parameters.json`.
    pub const SPRING_STEEL: Self = Self {
        youngs_modulus: 200e9,
        poisson_ratio: 0.3,
        density: 7850.0,
        rayleigh_alpha: 0.05,
        rayleigh_beta: 0.002,
    };

    /// Superelastic Nitinol (NiTi) for continuum endoscopes/catheters.
    pub const NITINOL: Self = Self {
        youngs_modulus: 60e9,
        poisson_ratio: 0.33,
        density: 6450.0,
        rayleigh_alpha: 0.1,
        rayleigh_beta: 0.003,
    };

    #[inline]
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }
}

/// Compute the cubic-Hermite midpoint corotational frame for a 2-node beam element.
/// Exact FP64 port of `midpoint(p, q, L)` in `serpworm.physics.gpu_core`.
pub fn cubic_hermite_midpoint(
    p0: Vec3,
    q0: Quat,
    p3: Vec3,
    q1: Quat,
    length: f64,
) -> CorotationalFrame {
    let t0 = q0.rotate(Vec3::EX);
    let t1 = q1.rotate(Vec3::EX);
    let p1 = p0 + (length / 3.0) * t0;
    let p2 = p3 - (length / 3.0) * t1;
    let center = (p0 + 3.0 * p1 + 3.0 * p2 + p3) * 0.125;
    let tangent_raw = -0.75 * p0 - 0.75 * p1 + 0.75 * p2 + 0.75 * p3;
    let t = tangent_raw.normalized();
    let r0 = q0.align_x_to(t);
    let mut r1 = q1.align_x_to(t);
    if r0.dot(r1) < 0.0 {
        r1 = Quat([-r1.0[0], -r1.0[1], -r1.0[2], -r1.0[3]]);
    }
    let avg = Quat([
        r0.0[0] + r1.0[0],
        r0.0[1] + r1.0[1],
        r0.0[2] + r1.0[2],
        r0.0[3] + r1.0[3],
    ])
    .normalized();
    let r = avg.align_x_to(t);
    let compressed = (p1 - p0).dot(p2 - p1) < 0.0 && (p3 - p0).norm() < 0.4 * length;
    if compressed {
        CorotationalFrame {
            center: 0.5 * (p0 + p3),
            rot: q0,
            compressed: true,
        }
    } else {
        CorotationalFrame {
            center,
            rot: r,
            compressed: false,
        }
    }
}

/// Compute local 12D corotational displacement vector `u = [u0_pos, u0_rot, u1_pos, u1_rot]`
/// and midpoint frame. Set `stable_log = true` for exact identity derivative (analytic Jacobian),
/// or `false` for exact `gpu_core.logq` legacy parity.
pub fn beam_local_strain(
    p0: Vec3,
    q0: Quat,
    p3: Vec3,
    q1: Quat,
    length: f64,
    stable_log: bool,
) -> (Vec12, CorotationalFrame) {
    let frame = cubic_hermite_midpoint(p0, q0, p3, q1, length);
    let inv = frame.rot.conj();
    let dp0 = inv.rotate(p0 - frame.center);
    let dp1 = inv.rotate(p3 - frame.center);
    let dq0 = inv.mul(q0);
    let dq1 = inv.mul(q1);
    let phi0 = if stable_log {
        dq0.log()
    } else {
        dq0.log_legacy()
    };
    let phi1 = if stable_log {
        dq1.log()
    } else {
        dq1.log_legacy()
    };
    let u = Vec12::from_halves(Vec6::from_parts(dp0, phi0), Vec6::from_parts(dp1, phi1));
    (u, frame)
}

/// Construct the 12x12 block-diagonal Plücker transform `T` mapping world endpoint twists
/// (about node origins with clamp offsets `offset0`, `offset1`) into the corotational frame.
/// Exact match to `T` in `serpworm.physics.gpu_core.frames()`.
pub fn beam_frame_transform(frame_rot: Quat, offset0: Vec3, offset1: Vec3) -> Mat12 {
    let rt = frame_rot.to_mat3().transpose();
    let t0 = plucker_motion_transform(rt, offset0);
    let t1 = plucker_motion_transform(rt, offset1);
    Mat12::from_blocks6(t0, crate::vec::Mat6::ZERO, crate::vec::Mat6::ZERO, t1)
}

// --- Internal small fixed-size matrix helpers for exact analytic beam_strain_jacobian ---

type Mat3x12 = [[f64; 12]; 3];
type Mat4x12 = [[f64; 12]; 4];

fn mul_3x3_3x12(a: Mat3, b: &Mat3x12) -> Mat3x12 {
    let mut out = [[0.0; 12]; 3];
    for i in 0..3 {
        for j in 0..12 {
            out[i][j] = a.0[i][0] * b[0][j] + a.0[i][1] * b[1][j] + a.0[i][2] * b[2][j];
        }
    }
    out
}

fn mul_4x3_3x12(a: [[f64; 3]; 4], b: &Mat3x12) -> Mat4x12 {
    let mut out = [[0.0; 12]; 4];
    for i in 0..4 {
        for j in 0..12 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mul_3x4_4x12(a: [[f64; 4]; 3], b: &Mat4x12) -> Mat3x12 {
    let mut out = [[0.0; 12]; 3];
    for i in 0..3 {
        for j in 0..12 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

fn mul_4x4_4x12(a: [[f64; 4]; 4], b: &Mat4x12) -> Mat4x12 {
    let mut out = [[0.0; 12]; 4];
    for i in 0..4 {
        for j in 0..12 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

fn qmatrix(q: Quat, left: bool) -> [[f64; 4]; 4] {
    let v = q.vec();
    let w = q.w();
    let sign = if left { 1.0 } else { -1.0 };
    let block = Mat3::diag([w, w, w]) + sign * v.skew();
    [
        [block.0[0][0], block.0[0][1], block.0[0][2], v.0[0]],
        [block.0[1][0], block.0[1][1], block.0[1][2], v.0[1]],
        [block.0[2][0], block.0[2][1], block.0[2][2], v.0[2]],
        [-v.0[0], -v.0[1], -v.0[2], w],
    ]
}

fn omega_map(q: Quat) -> [[f64; 4]; 3] {
    let v = q.vec();
    let w = q.w();
    let blk = Mat3::diag([w, w, w]) + v.skew();
    [
        [2.0 * blk.0[0][0], 2.0 * blk.0[0][1], 2.0 * blk.0[0][2], -2.0 * v.0[0]],
        [2.0 * blk.0[1][0], 2.0 * blk.0[1][1], 2.0 * blk.0[1][2], -2.0 * v.0[1]],
        [2.0 * blk.0[2][0], 2.0 * blk.0[2][1], 2.0 * blk.0[2][2], -2.0 * v.0[2]],
    ]
}

fn dq_from_omega_map(q: Quat) -> [[f64; 3]; 4] {
    let v = q.vec();
    let w = q.w();
    let blk = Mat3::diag([w, w, w]) - v.skew();
    [
        [0.5 * blk.0[0][0], 0.5 * blk.0[0][1], 0.5 * blk.0[0][2]],
        [0.5 * blk.0[1][0], 0.5 * blk.0[1][1], 0.5 * blk.0[1][2]],
        [0.5 * blk.0[2][0], 0.5 * blk.0[2][1], 0.5 * blk.0[2][2]],
        [-0.5 * v.0[0], -0.5 * v.0[1], -0.5 * v.0[2]],
    ]
}

fn unit3_d(v: Vec3, dv: &Mat3x12) -> (Vec3, Mat3x12) {
    let n = v.norm();
    let n_safe = n.max(1e-15);
    let u = v / n_safe;
    let proj = if n > 1e-15 {
        Mat3::IDENTITY - u.outer(u)
    } else {
        Mat3::IDENTITY
    };
    let du = mul_3x3_3x12(proj * (1.0 / n_safe), dv);
    (u, du)
}

fn unit4_d(v: [f64; 4], dv: &Mat4x12) -> (Quat, Mat4x12) {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    let n_safe = n.max(1e-15);
    let u = [v[0] / n_safe, v[1] / n_safe, v[2] / n_safe, v[3] / n_safe];
    let mut du = [[0.0; 12]; 4];
    for i in 0..4 {
        for j in 0..12 {
            let mut sum = 0.0;
            for k in 0..4 {
                let p_ik = (if i == k { 1.0 } else { 0.0 }) - if n > 1e-15 { u[i] * u[k] } else { 0.0 };
                sum += p_ik * dv[k][j];
            }
            du[i][j] = sum / n_safe;
        }
    }
    (Quat(u), du)
}

fn mul_quat_d(a: Quat, da: &Mat4x12, b: Quat, db: &Mat4x12) -> (Quat, Mat4x12) {
    let prod = a.mul(b);
    let qb_r = qmatrix(b, false);
    let qa_l = qmatrix(a, true);
    let p1 = mul_4x4_4x12(qb_r, da);
    let p2 = mul_4x4_4x12(qa_l, db);
    let mut dprod = [[0.0; 12]; 4];
    for i in 0..4 {
        for j in 0..12 {
            dprod[i][j] = p1[i][j] + p2[i][j];
        }
    }
    (prod, dprod)
}

fn align_d(q: Quat, dq: &Mat4x12, t: Vec3, dt: &Mat3x12) -> (Quat, Mat4x12) {
    let a = q.rotate(Vec3::EX);
    let omega_dq = mul_3x4_4x12(omega_map(q), dq);
    let da = mul_3x3_3x12(-a.skew(), &omega_dq);
    let c = a.cross(t);
    let raw = [c.0[0], c.0[1], c.0[2], 1.0 + a.dot(t)];

    let part_a = mul_3x3_3x12(-t.skew(), &da);
    let part_t = mul_3x3_3x12(a.skew(), dt);
    let mut draw = [[0.0; 12]; 4];
    for i in 0..3 {
        for j in 0..12 {
            draw[i][j] = part_a[i][j] + part_t[i][j];
        }
    }
    for j in 0..12 {
        draw[3][j] = t.0[0] * da[0][j]
            + t.0[1] * da[1][j]
            + t.0[2] * da[2][j]
            + a.0[0] * dt[0][j]
            + a.0[1] * dt[1][j]
            + a.0[2] * dt[2][j];
    }
    let (r, dr) = unit4_d(raw, &draw);
    mul_quat_d(r, &dr, q, dq)
}

/// Exact analytic 12x12 strain Jacobian `J = du / d(world endpoint twist)`,
/// including all derivatives of the cubic-Hermite midpoint corotational frame and
/// the $SO(3)$ inverse left Jacobian `dexp^{-1}(phi)`.
///
/// Exact FP64 port of `beam_strain_jacobian(p, q, offset, L)` in `serpworm.physics.gpu_core`.
pub fn beam_strain_jacobian(
    p0: Vec3,
    q0: Quat,
    offset0: Vec3,
    p3: Vec3,
    q1: Quat,
    offset1: Vec3,
    length: f64,
) -> Mat12 {
    // W_0, W_1 (3x12): angular velocity maps for endpoints 0 and 1
    let mut w0 = [[0.0; 12]; 3];
    let mut w1 = [[0.0; 12]; 3];
    for i in 0..3 {
        w0[i][3 + i] = 1.0;
        w1[i][9 + i] = 1.0;
    }

    // P_0, P_1 (3x12): clamp endpoint position velocity maps
    let mut p0_jac = [[0.0; 12]; 3];
    let mut p1_jac = [[0.0; 12]; 3];
    let s0 = -offset0.skew();
    let s1 = -offset1.skew();
    for i in 0..3 {
        p0_jac[i][i] = 1.0;
        p1_jac[i][6 + i] = 1.0;
        for j in 0..3 {
            p0_jac[i][3 + j] = s0.0[i][j];
            p1_jac[i][9 + j] = s1.0[i][j];
        }
    }

    let dq0 = mul_4x3_3x12(dq_from_omega_map(q0), &w0);
    let dq1 = mul_4x3_3x12(dq_from_omega_map(q1), &w1);

    let t0 = q0.rotate(Vec3::EX);
    let t1 = q1.rotate(Vec3::EX);
    let dt0 = mul_3x3_3x12(-t0.skew(), &w0);
    let dt1 = mul_3x3_3x12(-t1.skew(), &w1);

    let p1 = p0 + (length / 3.0) * t0;
    let p2 = p3 - (length / 3.0) * t1;
    let tangent_raw = -0.75 * p0 - 0.75 * p1 + 0.75 * p2 + 0.75 * p3;
    let mut dtr = [[0.0; 12]; 3];
    for i in 0..3 {
        for j in 0..12 {
            dtr[i][j] = 1.5 * (p1_jac[i][j] - p0_jac[i][j])
                - 0.25 * length * (dt0[i][j] + dt1[i][j]);
        }
    }
    let (t, dt) = unit3_d(tangent_raw, &dtr);
    let (r0, dr0) = align_d(q0, &dq0, t, &dt);
    let (r1, dr1) = align_d(q1, &dq1, t, &dt);
    let sign = if r0.dot(r1) < 0.0 { -1.0 } else { 1.0 };

    let sum_r = [
        r0.0[0] + sign * r1.0[0],
        r0.0[1] + sign * r1.0[1],
        r0.0[2] + sign * r1.0[2],
        r0.0[3] + sign * r1.0[3],
    ];
    let mut dsum_r = [[0.0; 12]; 4];
    for i in 0..4 {
        for j in 0..12 {
            dsum_r[i][j] = dr0[i][j] + sign * dr1[i][j];
        }
    }
    let (middle, dmiddle) = unit4_d(sum_r, &dsum_r);
    let (r_align, dr_align) = align_d(middle, &dmiddle, t, &dt);

    let compressed = (p1 - p0).dot(p2 - p1) < 0.0 && (p3 - p0).norm() < 0.4 * length;
    let (r, dr) = if compressed {
        (q0, dq0)
    } else {
        (r_align, dr_align)
    };
    let b_frame = mul_3x4_4x12(omega_map(r), &dr);

    let center = if compressed {
        0.5 * (p0 + p3)
    } else {
        (p0 + 3.0 * p1 + 3.0 * p2 + p3) * 0.125
    };
    let mut c_frame = [[0.0; 12]; 3];
    if compressed {
        for i in 0..3 {
            for j in 0..12 {
                c_frame[i][j] = 0.5 * (p0_jac[i][j] + p1_jac[i][j]);
            }
        }
    } else {
        let term0 = mul_3x3_3x12(-t0.skew(), &w0);
        let term1 = mul_3x3_3x12(t1.skew(), &w1);
        for i in 0..3 {
            for j in 0..12 {
                c_frame[i][j] = 0.5 * (p0_jac[i][j] + p1_jac[i][j])
                    + (length / 8.0) * (term0[i][j] + term1[i][j]);
            }
        }
    }

    let rt = r.to_mat3().transpose();
    let inv_r = r.conj();
    let phi0 = inv_r.mul(q0).log();
    let phi1 = inv_r.mul(q1).log();

    let s_dp0 = (p0 - center).skew();
    let s_dp1 = (p3 - center).skew();
    let sb0 = mul_3x3_3x12(s_dp0, &b_frame);
    let sb1 = mul_3x3_3x12(s_dp1, &b_frame);

    let mut diff_p0 = [[0.0; 12]; 3];
    let mut diff_p1 = [[0.0; 12]; 3];
    let mut diff_w0 = [[0.0; 12]; 3];
    let mut diff_w1 = [[0.0; 12]; 3];
    for i in 0..3 {
        for j in 0..12 {
            diff_p0[i][j] = p0_jac[i][j] - c_frame[i][j] + sb0[i][j];
            diff_p1[i][j] = p1_jac[i][j] - c_frame[i][j] + sb1[i][j];
            diff_w0[i][j] = w0[i][j] - b_frame[i][j];
            diff_w1[i][j] = w1[i][j] - b_frame[i][j];
        }
    }

    let jp0 = mul_3x3_3x12(rt, &diff_p0);
    let jp1 = mul_3x3_3x12(rt, &diff_p1);
    let jr0 = mul_3x3_3x12(so3_dexp_inv(phi0).mul_mat(rt), &diff_w0);
    let jr1 = mul_3x3_3x12(so3_dexp_inv(phi1).mul_mat(rt), &diff_w1);

    let mut out = [[0.0; 12]; 12];
    out[..3].copy_from_slice(&jp0);
    out[3..6].copy_from_slice(&jr0);
    out[6..9].copy_from_slice(&jp1);
    out[9..12].copy_from_slice(&jr1);
    Mat12(out)
}

/// Compute local 12x12 beam stiffness `K` and consistent mass `M` (including rotary inertia).
///
/// - When `thin_shell_compat == true` on a `Rectangular` section, applies Kirchhoff-Love
///   plate flexural stiffening `E_y = E / (1 - nu^2)` to out-of-plane bending `Iy` and Vlasov
///   restrained-warping torsional stiffening matching `serpworm.physics.gpu_core.local_matrices_torch`.
/// - When `timoshenko_shear == true`, applies exact Przemieniecki shear-deformation factors
///   `Phi_y = 12 E Iz / (G A k_y L^2)` and `Phi_z = 12 E_y Iy / (G A k_z L^2)`.
pub fn local_beam_matrices(
    length: f64,
    section: &BeamSection,
    material: &BeamMaterial,
    thin_shell_compat: bool,
    timoshenko_shear: bool,
) -> (Mat12, Mat12) {
    let props = section.properties();
    let l = length;
    let l2 = l * l;
    let l3 = l2 * l;

    let e = material.youngs_modulus;
    let nu = material.poisson_ratio;
    let g = material.shear_modulus();
    let rho = material.density;

    let a = props.area;
    let iy = props.iy;
    let iz = props.iz;

    let (e_y, j_eff) = if thin_shell_compat {
        let e_plate = e / (1.0 - nu * nu);
        let j_vlasov = match props.thin_width {
            Some(w) => props.j_torsion * (1.0 + (w * w) / (12.0 * 0.145 * 0.145 * (1.0 - nu))),
            None => props.j_torsion,
        };
        (e_plate, j_vlasov)
    } else {
        (e, props.j_torsion)
    };

    let (phi_y, phi_z) = if timoshenko_shear {
        (
            12.0 * e * iz / (g * a * props.shear_ky * l2).max(1e-30),
            12.0 * e_y * iy / (g * a * props.shear_kz * l2).max(1e-30),
        )
    } else {
        (0.0, 0.0)
    };

    let mut k = [[0.0; 12]; 12];
    let kd = [
        e * a / l,
        12.0 * e * iz / (l3 * (1.0 + phi_y)),
        12.0 * e_y * iy / (l3 * (1.0 + phi_z)),
        g * j_eff / l,
        (4.0 + phi_z) * e_y * iy / (l * (1.0 + phi_z)),
        (4.0 + phi_y) * e * iz / (l * (1.0 + phi_y)),
    ];
    let kc = [
        -kd[0],
        -kd[1],
        -kd[2],
        -kd[3],
        (2.0 - phi_z) * e_y * iy / (l * (1.0 + phi_z)),
        (2.0 - phi_y) * e * iz / (l * (1.0 + phi_y)),
    ];
    for idx in 0..6 {
        k[idx][idx] = kd[idx];
        k[idx + 6][idx + 6] = kd[idx];
        k[idx + 6][idx] = kc[idx];
        k[idx][idx + 6] = kc[idx];
    }
    let a42 = -6.0 * e_y * iy / (l2 * (1.0 + phi_z));
    let a51 = 6.0 * e * iz / (l2 * (1.0 + phi_y));
    k[4][2] = a42;
    k[2][4] = a42;
    k[5][1] = a51;
    k[1][5] = a51;
    k[10][8] = -a42;
    k[8][10] = -a42;
    k[11][7] = -a51;
    k[7][11] = -a51;
    k[7][5] = -a51;
    k[5][7] = -a51;
    k[8][4] = -a42;
    k[4][8] = -a42;
    k[10][2] = a42;
    k[2][10] = a42;
    k[11][1] = a51;
    k[1][11] = a51;

    let mut m = [[0.0; 12]; 12];
    let md = [
        1.0 / 3.0,
        13.0 / 35.0 + 6.0 * iz / (5.0 * a * l2),
        13.0 / 35.0 + 6.0 * iy / (5.0 * a * l2),
        (iy + iz) / (3.0 * a),
        l2 / 105.0 + 2.0 * iy / (15.0 * a),
        l2 / 105.0 + 2.0 * iz / (15.0 * a),
    ];
    let mc = [
        1.0 / 6.0,
        9.0 / 70.0 - 6.0 * iz / (5.0 * a * l2),
        9.0 / 70.0 - 6.0 * iy / (5.0 * a * l2),
        (iy + iz) / (6.0 * a),
        -l2 / 140.0 - iy / (30.0 * a),
        -l2 / 140.0 - iz / (30.0 * a),
    ];
    for idx in 0..6 {
        m[idx][idx] = md[idx];
        m[idx + 6][idx + 6] = md[idx];
        m[idx + 6][idx] = mc[idx];
        m[idx][idx + 6] = mc[idx];
    }
    let m42 = -11.0 * l / 210.0 - iy / (10.0 * a * l);
    let m51 = 11.0 * l / 210.0 + iz / (10.0 * a * l);
    m[4][2] = m42;
    m[2][4] = m42;
    m[5][1] = m51;
    m[1][5] = m51;
    m[10][8] = -m42;
    m[8][10] = -m42;
    m[11][7] = -m51;
    m[7][11] = -m51;
    let c15 = 13.0 * l / 420.0 - iz / (10.0 * a * l);
    let c24 = -13.0 * l / 420.0 + iy / (10.0 * a * l);
    m[7][5] = c15;
    m[5][7] = c15;
    m[8][4] = c24;
    m[4][8] = c24;
    m[10][2] = -c24;
    m[2][10] = -c24;
    m[11][1] = -c15;
    m[1][11] = -c15;

    let mass_scale = rho * a * l;
    for row in &mut m {
        for v in row.iter_mut() {
            *v *= mass_scale;
        }
    }

    (Mat12(k), Mat12(m))
}

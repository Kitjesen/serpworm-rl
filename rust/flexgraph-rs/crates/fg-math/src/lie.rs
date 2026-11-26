//! Pure-Rust FP64 Lie group operations on $SO(3)$ and $SE(3)$:
//! unit quaternions `[x, y, z, w]`, exponential/logarithmic maps, left/right Jacobians
//! (`dexp`, `dexp_inv`), and exact manifold pose integration.

use crate::vec::{Mat3, Mat6, Vec3, Vec6};
use serde::{Deserialize, Serialize};

/// Unit quaternion stored as `[x, y, z, w]` (vector part `xyz`, scalar part `w`).
/// Matches `serpworm.physics.gpu_core` and `topology.py` layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Quat(pub [f64; 4]);

impl Default for Quat {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::ops::Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Quat::mul(self, rhs)
    }
}

impl Quat {
    pub const IDENTITY: Self = Self([0.0, 0.0, 0.0, 1.0]);

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self([x, y, z, w])
    }

    #[inline]
    pub const fn vec(self) -> Vec3 {
        Vec3([self.0[0], self.0[1], self.0[2]])
    }

    #[inline]
    pub const fn w(self) -> f64 {
        self.0[3]
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1] + self.0[2] * rhs.0[2] + self.0[3] * rhs.0[3]
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n > 1e-15 {
            let inv = 1.0 / n;
            Self([
                self.0[0] * inv,
                self.0[1] * inv,
                self.0[2] * inv,
                self.0[3] * inv,
            ])
        } else {
            Self::IDENTITY
        }
    }

    /// Canonicalize hemisphere so `w >= 0`.
    #[inline]
    pub fn canonical(self) -> Self {
        if self.0[3] < 0.0 {
            Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
        } else {
            self
        }
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], self.0[3]])
    }

    /// Hamilton product `self * rhs`.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        let av = self.vec();
        let aw = self.0[3];
        let bv = rhs.vec();
        let bw = rhs.0[3];
        let v = bv * aw + av * bw + av.cross(bv);
        let w = aw * bw - av.dot(bv);
        Self([v.0[0], v.0[1], v.0[2], w])
    }

    /// Rotate a 3D vector by this unit quaternion: `x + w*(2 q_v x x) + q_v x (2 q_v x x)`.
    #[inline]
    pub fn rotate(self, x: Vec3) -> Vec3 {
        let qv = self.vec();
        let t = 2.0 * qv.cross(x);
        x + self.0[3] * t + qv.cross(t)
    }

    /// Rotate a 3D vector by the inverse (conjugate) quaternion.
    #[inline]
    pub fn inv_rotate(self, x: Vec3) -> Vec3 {
        self.conj().rotate(x)
    }

    /// Convert to 3x3 proper orthogonal rotation matrix `I + 2 w [v]_x + 2 [v]_x^2`.
    #[inline]
    pub fn to_mat3(self) -> Mat3 {
        let s = self.vec().skew();
        Mat3::IDENTITY + (2.0 * self.0[3]) * s + 2.0 * s.mul_mat(s)
    }

    /// Shepperd's method converting a 3x3 proper rotation matrix to canonical unit quaternion (`w >= 0`).
    pub fn from_mat3(r: Mat3) -> Self {
        let m = r.0;
        let tr = m[0][0] + m[1][1] + m[2][2];
        let (x, y, z, w) = if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0;
            (
                (m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
                0.25 * s,
            )
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
            (
                0.25 * s,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[2][1] - m[1][2]) / s,
            )
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
            (
                (m[0][1] + m[1][0]) / s,
                0.25 * s,
                (m[1][2] + m[2][1]) / s,
                (m[0][2] - m[2][0]) / s,
            )
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
            (
                (m[0][2] + m[2][0]) / s,
                (m[1][2] + m[2][1]) / s,
                0.25 * s,
                (m[1][0] - m[0][1]) / s,
            )
        };
        Self([x, y, z, w]).normalized().canonical()
    }

    /// Exponential map `exp(phi)` from rotation vector `phi in so(3)` to unit quaternion.
    /// Uses Taylor expansion near `||phi|| == 0` so derivative is exact at identity.
    #[inline]
    pub fn exp(phi: Vec3) -> Self {
        let half = 0.5 * phi;
        let theta2 = half.norm_sq();
        let (sinc, c) = if theta2 < 1e-12 {
            let t4 = theta2 * theta2;
            (
                1.0 - theta2 / 6.0 + t4 / 120.0,
                1.0 - 0.5 * theta2 + t4 / 24.0,
            )
        } else {
            let theta = theta2.sqrt();
            (theta.sin() / theta, theta.cos())
        };
        let v = half * sinc;
        Self([v.0[0], v.0[1], v.0[2], c]).normalized()
    }

    /// Smooth quaternion logarithm `log(q) in so(3)` with exact `2 I` derivative at identity
    /// (matches `stable_logq` in `serpworm.physics.gpu_core`).
    #[inline]
    pub fn log(self) -> Vec3 {
        let q = self.canonical();
        let v = q.vec();
        let w = q.0[3];
        let n2 = v.norm_sq();
        let ratio = if n2 < 1e-12 {
            let ws = w.max(1e-8);
            let ws3 = ws * ws * ws;
            let ws5 = ws3 * ws * ws;
            2.0 / ws - 2.0 * n2 / (3.0 * ws3) + 2.0 * n2 * n2 / (5.0 * ws5)
        } else {
            let n = n2.sqrt();
            2.0 * n.atan2(w) / n
        };
        v * ratio
    }

    /// Legacy `gpu_core.logq` formula (for bit-exact comparison when `strain_jacobian=false`).
    #[inline]
    pub fn log_legacy(self) -> Vec3 {
        let q = self.canonical();
        let v = q.vec();
        let n = v.norm();
        v * (2.0 * n.atan2(q.0[3]) / n.max(1e-15))
    }

    /// Minimal-angle rotation mapping unit vector `a` to unit vector `b`.
    pub fn from_two_unit_vectors(a: Vec3, b: Vec3) -> Self {
        let c = a.dot(b);
        if c < -1.0 + 1e-12 {
            // Antiparallel: pick any axis orthogonal to `a`.
            let ortho = if a.x().abs() < 0.9 {
                Vec3::EX.cross(a)
            } else {
                Vec3::EY.cross(a)
            }
            .normalized();
            Self([ortho.0[0], ortho.0[1], ortho.0[2], 0.0])
        } else {
            let cross = a.cross(b);
            Self([cross.0[0], cross.0[1], cross.0[2], 1.0 + c]).normalized()
        }
    }

    /// Align the local x-axis of `self` to unit tangent `t` via minimal rotation
    /// (matches `align(q, t)` in `serpworm.physics.gpu_core`).
    #[inline]
    pub fn align_x_to(self, target_tangent: Vec3) -> Self {
        let a = self.rotate(Vec3::EX);
        let r = Self::from_two_unit_vectors(a, target_tangent);
        r.mul(self)
    }

    /// Rotation from axis-angle `(axis_unit, angle_rad)`.
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        Self::exp(axis.normalized() * angle)
    }
}

/// Rodrigues exponential map `exp([phi]_x) in SO(3)`.
pub fn so3_exp(phi: Vec3) -> Mat3 {
    let theta2 = phi.norm_sq();
    let s = phi.skew();
    let (a, b) = if theta2 < 1e-10 {
        let t4 = theta2 * theta2;
        (1.0 - theta2 / 6.0 + t4 / 120.0, 0.5 - theta2 / 24.0 + t4 / 720.0)
    } else {
        let theta = theta2.sqrt();
        (theta.sin() / theta, (1.0 - theta.cos()) / theta2)
    };
    Mat3::IDENTITY + a * s + b * s.mul_mat(s)
}

/// Matrix logarithm `log(R) in so(3)` via canonical unit quaternion.
#[inline]
pub fn so3_log(r: Mat3) -> Vec3 {
    Quat::from_mat3(r).log()
}

/// Left Jacobian of $SO(3)$:
/// `J_l(phi) = I + ((1 - cos(th))/th^2) [phi]_x + ((th - sin(th))/th^3) [phi]_x^2`.
pub fn so3_dexp(phi: Vec3) -> Mat3 {
    let theta2 = phi.norm_sq();
    let s = phi.skew();
    let (c1, c2) = if theta2 < 1e-8 {
        let t4 = theta2 * theta2;
        (
            0.5 - theta2 / 24.0 + t4 / 720.0,
            1.0 / 6.0 - theta2 / 120.0 + t4 / 5040.0,
        )
    } else {
        let theta = theta2.sqrt();
        (
            (1.0 - theta.cos()) / theta2,
            (theta - theta.sin()) / (theta2 * theta),
        )
    };
    Mat3::IDENTITY + c1 * s + c2 * s.mul_mat(s)
}

/// Inverse left Jacobian of $SO(3)$:
/// `dexp^{-1}(phi) = I - 0.5 [phi]_x + ((1 - 0.5*th*cot(0.5*th)) / th^2) [phi]_x^2`.
/// Matches `A` in `beam_strain_jacobian` of `serpworm.physics.gpu_core`.
pub fn so3_dexp_inv(phi: Vec3) -> Mat3 {
    let theta2 = phi.norm_sq();
    let s = phi.skew();
    let coeff = if theta2 < 1e-8 {
        1.0 / 12.0 + theta2 / 720.0 + (theta2 * theta2) / 30240.0
    } else {
        let theta = theta2.sqrt();
        let half = 0.5 * theta;
        (1.0 - half / half.tan()) / theta2
    };
    Mat3::IDENTITY - 0.5 * s + coeff * s.mul_mat(s)
}

/// Right Jacobian of $SO(3)$: `J_r(phi) = J_l(-phi)`.
#[inline]
pub fn so3_right_jacobian(phi: Vec3) -> Mat3 {
    so3_dexp(-phi)
}

/// Inverse right Jacobian of $SO(3)$: `J_r^{-1}(phi) = J_l^{-1}(-phi)`.
#[inline]
pub fn so3_right_jacobian_inv(phi: Vec3) -> Mat3 {
    so3_dexp_inv(-phi)
}

/// Rigid pose in $SE(3)$ represented by position `pos` and unit quaternion `rot`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Se3 {
    pub pos: Vec3,
    pub rot: Quat,
}

impl Se3 {
    pub const IDENTITY: Self = Self {
        pos: Vec3::ZERO,
        rot: Quat::IDENTITY,
    };

    #[inline]
    pub const fn new(pos: Vec3, rot: Quat) -> Self {
        Self { pos, rot }
    }

    /// Construct from `[px, py, pz, qx, qy, qz, qw]`.
    #[inline]
    pub fn from_array7(a: [f64; 7]) -> Self {
        Self {
            pos: Vec3([a[0], a[1], a[2]]),
            rot: Quat([a[3], a[4], a[5], a[6]]).normalized(),
        }
    }

    /// Export as `[px, py, pz, qx, qy, qz, qw]`.
    #[inline]
    pub const fn to_array7(self) -> [f64; 7] {
        [
            self.pos.0[0],
            self.pos.0[1],
            self.pos.0[2],
            self.rot.0[0],
            self.rot.0[1],
            self.rot.0[2],
            self.rot.0[3],
        ]
    }

    /// Compose `self * rhs`.
    #[inline]
    pub fn compose(self, rhs: Self) -> Self {
        Self {
            pos: self.pos + self.rot.rotate(rhs.pos),
            rot: self.rot.mul(rhs.rot).normalized(),
        }
    }

    /// Group inverse `self^{-1}`.
    #[inline]
    pub fn inverse(self) -> Self {
        let r_inv = self.rot.conj();
        Self {
            pos: r_inv.rotate(-self.pos),
            rot: r_inv,
        }
    }

    /// Transform a 3D point from local frame to parent/world frame.
    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        self.pos + self.rot.rotate(p)
    }

    /// Transform a free 3D vector (rotation only).
    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        self.rot.rotate(v)
    }

    /// Exact $SE(3)$ exponential map for twist `xi = [v; omega]`.
    #[inline]
    pub fn exp(xi: Vec6) -> Self {
        se3_exp(xi)
    }

    /// Exact $SE(3)$ logarithmic map returning `[v; omega]`.
    #[inline]
    pub fn log(self) -> Vec6 {
        se3_log(self)
    }

    /// Decoupled world-frame pose update matching `serpworm.physics.gpu_core`:
    /// `p <- p + dt * v`, `q <- unit(exp(dt * omega) * q)`.
    #[inline]
    pub fn integrate_decoupled(self, twist_world: Vec6, dt: f64) -> Self {
        let pos = self.pos + dt * twist_world.lin();
        let dq = Quat::exp(dt * twist_world.ang());
        let rot = dq.mul(self.rot).normalized();
        Self { pos, rot }
    }

    /// Exact right-trivialized Lie-group step `T_{k+1} = T_k * exp(dt * xi_body)`.
    #[inline]
    pub fn integrate_lie_body(self, twist_body: Vec6, dt: f64) -> Self {
        self.compose(se3_exp(dt * twist_body))
    }

    /// Exact left-trivialized Lie-group step `T_{k+1} = exp(dt * xi_spatial) * T_k`.
    #[inline]
    pub fn integrate_lie_spatial(self, twist_spatial: Vec6, dt: f64) -> Self {
        se3_exp(dt * twist_spatial).compose(self)
    }
}

/// Exact exponential map $\exp: \mathfrak{se}(3) \to SE(3)$ for `xi = [v; omega]`.
pub fn se3_exp(xi: Vec6) -> Se3 {
    let v = xi.lin();
    let omega = xi.ang();
    let jl = so3_dexp(omega);
    Se3 {
        pos: jl.mul_vec(v),
        rot: Quat::exp(omega),
    }
}

/// Exact logarithmic map $\log: SE(3) \to \mathfrak{se}(3)$ returning `[v; omega]`.
pub fn se3_log(tf: Se3) -> Vec6 {
    let omega = tf.rot.log();
    let jl_inv = so3_dexp_inv(omega);
    let v = jl_inv.mul_vec(tf.pos);
    Vec6::from_parts(v, omega)
}

/// Auxiliary $Q(\mathbf{v}, \boldsymbol{\omega})$ block for the exact $SE(3)$ left Jacobian
/// (Barfoot, *State Estimation for Robotics*, Eq. 7.86), in `[linear; angular]` ordering:
/// `J_l(xi) = [[J_l(w), Q(v, w)], [0, J_l(w)]]`,
/// `J_l^{-1}(xi) = [[J_l^{-1}(w), -J_l^{-1}(w) Q(v, w) J_l^{-1}(w)], [0, J_l^{-1}(w)]]`.
fn se3_q_matrix(v: Vec3, omega: Vec3) -> Mat3 {
    let theta2 = omega.norm_sq();
    let vx = v.skew();
    let wx = omega.skew();
    let wx2 = wx.mul_mat(wx);
    let (c2, c3, c4) = if theta2 < 1e-6 {
        let t4 = theta2 * theta2;
        (
            1.0 / 6.0 - theta2 / 120.0 + t4 / 5040.0,
            1.0 / 24.0 - theta2 / 720.0 + t4 / 40320.0,
            1.0 / 120.0 - theta2 / 2520.0 + t4 / 120960.0,
        )
    } else {
        let th = theta2.sqrt();
        let th3 = theta2 * th;
        let th4 = theta2 * theta2;
        let th5 = th4 * th;
        let s = th.sin();
        let c = th.cos();
        (
            (th - s) / th3,
            (theta2 + 2.0 * c - 2.0) / (2.0 * th4),
            (2.0 * th - 3.0 * s + th * c) / (2.0 * th5),
        )
    };
    let t1 = 0.5 * vx;
    let t2 = c2 * (wx.mul_mat(vx) + vx.mul_mat(wx) + wx.mul_mat(vx).mul_mat(wx));
    let t3 = c3
        * (wx2.mul_mat(vx) + vx.mul_mat(wx2) - 3.0 * wx.mul_mat(vx).mul_mat(wx));
    let t4 = c4 * (wx.mul_mat(vx).mul_mat(wx2) + wx2.mul_mat(vx).mul_mat(wx));
    t1 + t2 + t3 + t4
}

/// Exact inverse left Jacobian `dexp^{-1}(xi)` on $SE(3)$ (6x6, `[linear; angular]` ordering).
/// Satisfies `log(exp(xi) * exp(eps)) = xi + dexp_inv(xi) * eps + O(||eps||^2)` for left-trivialized
/// perturbations `exp(eps) * exp(xi)` -> `xi + dexp_inv(xi) * eps`.
pub fn se3_dexp_inv(xi: Vec6) -> Mat6 {
    let v = xi.lin();
    let w = xi.ang();
    let jli = so3_dexp_inv(w);
    let q = se3_q_matrix(v, w);
    let upper_right = -(jli.mul_mat(q).mul_mat(jli));
    Mat6::from_blocks(jli, upper_right, Mat3::ZERO, jli)
}

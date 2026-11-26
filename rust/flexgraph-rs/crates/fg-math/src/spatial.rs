//! 6x6 Spatial Plücker Vector Algebra (Featherstone 2008) in `[linear (0..3); angular (3..6)]`
//! ordering, including adjoint/coadjoint transforms, spatial cross products, and rigid
//! spatial inertia tensors.

use crate::lie::{Quat, Se3};
use crate::vec::{Mat3, Mat6, Vec3, Vec6};
use serde::{Deserialize, Serialize};

/// Spatial motion (twist) transform across a rigid transform `T = (R, p)` (Adjoint `Ad_T`):
/// maps a spatial twist `[v; omega]` from child frame to parent frame:
/// `Ad_T = [[R, [p]_x R], [0, R]]` (in `[linear; angular]` order).
#[inline]
pub fn adjoint_se3(tf: Se3) -> Mat6 {
    let r = tf.rot.to_mat3();
    let px_r = tf.pos.skew().mul_mat(r);
    Mat6::from_blocks(r, px_r, Mat3::ZERO, r)
}

/// Spatial force (wrench) transform (dual / transpose of inverse adjoint `Ad_{T^{-1}}^T`):
/// maps a spatial wrench `[f; tau]` from child frame to parent frame:
/// `Ad^*_T = [[R, 0], [[p]_x R, R]]`.
#[inline]
pub fn coadjoint_se3(tf: Se3) -> Mat6 {
    let r = tf.rot.to_mat3();
    let px_r = tf.pos.skew().mul_mat(r);
    Mat6::from_blocks(r, Mat3::ZERO, px_r, r)
}

/// Plücker motion transform mapping a twist expressed at origin `A` in world orientation
/// to a shifted point `A + offset_from` expressed in frame `B` with rotation `r_to_from = R_B^T`:
/// `X_M = [[r_to_from, -r_to_from * [offset_from]_x], [0, r_to_from]]`.
///
/// Matches the 6x6 endpoint block `trans` in `serpworm.physics.gpu_core.frames()`.
#[inline]
pub fn plucker_motion_transform(r_to_from: Mat3, offset_from: Vec3) -> Mat6 {
    let top_right = -(r_to_from.mul_mat(offset_from.skew()));
    Mat6::from_blocks(r_to_from, top_right, Mat3::ZERO, r_to_from)
}

/// Dual Plücker force transform `X_F = X_M^T`, mapping wrenches back to the body origin.
#[inline]
pub fn plucker_force_transform(r_to_from: Mat3, offset_from: Vec3) -> Mat6 {
    plucker_motion_transform(r_to_from, offset_from).transpose()
}

/// Spatial motion cross-product matrix `ad_v = v x` in `[linear; angular]` order:
/// `ad_{[v; w]} [u; eta] = [w x u + v x eta; w x eta]`.
#[inline]
pub fn spatial_cross_motion(v: Vec6) -> Mat6 {
    let vx = v.lin().skew();
    let wx = v.ang().skew();
    Mat6::from_blocks(wx, vx, Mat3::ZERO, wx)
}

/// Spatial force cross-product matrix `v x* = -ad_v^T` in `[linear; angular]` order:
/// `v x* [f; tau] = [w x f; v x f + w x tau]`.
#[inline]
pub fn spatial_cross_force(v: Vec6) -> Mat6 {
    let vx = v.lin().skew();
    let wx = v.ang().skew();
    Mat6::from_blocks(wx, Mat3::ZERO, vx, wx)
}

/// Convert a 3D world force `f_world` applied at lever arm `arm_world = p_contact - p_body`
/// into a 6D spatial wrench `[f_world; arm_world x f_world]`.
#[inline]
pub fn point_force_to_wrench(arm_world: Vec3, f_world: Vec3) -> Vec6 {
    Vec6::from_parts(f_world, arm_world.cross(f_world))
}

/// 6x3 Jacobian columns (returned as 3 rows of 6D vectors, one per world axis `[ex, ey, ez]`)
/// mapping body twist `[v; omega]` to contact point linear velocity `v + omega x arm = v - [arm]_x omega`.
#[inline]
pub fn point_wrench_jacobian(arm_world: Vec3) -> [Vec6; 3] {
    let ax = arm_world.skew();
    // Row i of [I, -[arm]_x] = [e_i; arm x e_i] because (-[arm]_x)^T e_i = [arm]_x e_i = arm x e_i.
    [
        Vec6::from_parts(Vec3::EX, Vec3([0.0, ax.0[1][0], ax.0[2][0]])),
        Vec6::from_parts(Vec3::EY, Vec3([ax.0[0][1], 0.0, ax.0[2][1]])),
        Vec6::from_parts(Vec3::EZ, Vec3([ax.0[0][2], ax.0[1][2], 0.0])),
    ]
}

/// Rigid-body 6x6 spatial inertia tensor in body coordinates (`[linear; angular]` order).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialInertia {
    pub mass: f64,
    pub com: Vec3,
    pub rot_inertia_com: Mat3,
}

impl SpatialInertia {
    /// Construct centered inertia (`com = 0`) from scalar mass and principal inertia diagonal.
    #[inline]
    pub const fn from_mass_and_diag(mass: f64, diag_inertia: [f64; 3]) -> Self {
        Self {
            mass,
            com: Vec3::ZERO,
            rot_inertia_com: Mat3::diag(diag_inertia),
        }
    }

    /// Assemble 6x6 symmetric spatial inertia matrix in body frame:
    /// `I_b = [[m I, -m [c]_x], [m [c]_x, I_c - m [c]_x^2]]`.
    pub fn to_mat6(&self) -> Mat6 {
        let m = self.mass;
        let m_eye = Mat3::diag([m, m, m]);
        let cx = self.com.skew();
        let m_cx = m * cx;
        let rot = self.rot_inertia_com - m * cx.mul_mat(cx);
        Mat6::from_blocks(m_eye, -m_cx, m_cx, rot)
    }

    /// Assemble world-oriented 6x6 spatial inertia `diag(R, R) * I_b * diag(R^T, R^T)`.
    pub fn world_mat6(&self, q: Quat) -> Mat6 {
        let r = q.to_mat3();
        let rt = r.transpose();
        let b = self.to_mat6();
        let b00 = r.mul_mat(b.block(0, 0)).mul_mat(rt);
        let b01 = r.mul_mat(b.block(0, 1)).mul_mat(rt);
        let b10 = r.mul_mat(b.block(1, 0)).mul_mat(rt);
        let b11 = r.mul_mat(b.block(1, 1)).mul_mat(rt);
        Mat6::from_blocks(b00, b01, b10, b11)
    }

    /// Gyroscopic / Coriolis spatial bias wrench `v x* (I_b v)` in body coordinates.
    pub fn bias_wrench_body(&self, twist_body: Vec6) -> Vec6 {
        let momentum = self.to_mat6().mul_vec(twist_body);
        spatial_cross_force(twist_body).mul_vec(momentum)
    }
}

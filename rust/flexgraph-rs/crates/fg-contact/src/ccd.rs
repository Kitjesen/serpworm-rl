//! Conservative **Additive CCD (ACCD)** from C-IPC (Li et al., SIGGRAPH 2021, Algorithm 1)
//! supporting finite thickness offsets $\xi \ge 0$ for Point-Point, Point-Segment,
//! Segment-Segment (Edge-Edge), and Lipschitz-certified Point-vs-SDF trajectories.

use crate::broadphase::SpatialHashGrid;
use crate::distance::segment_segment_dist_sq;
pub use crate::tight_inclusion::{
    tight_inclusion_edge_edge_ccd, tight_inclusion_point_triangle_ccd, TightInclusionConfig,
};
use fg_math::Vec3;
use fg_sdf::SdfField;

const ACCD_MAX_ITERS: usize = 256;

/// Point-Segment squared distance helper for ACCD inner loops.
#[inline]
fn point_segment_dist_sq_fast(p: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = b - a;
    let l2 = ab.norm_sq();
    if l2 <= 1e-24 {
        return (p - a).norm_sq();
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    (p - (a + ab * t)).norm_sq()
}

/// Conservative Additive CCD for a moving Point-Point pair with thickness offset `xi >= 0`.
///
/// Returns `Some(toi)` in `[0, t_max]` if the points approach within the conservative target
/// separation `xi + s_conserv * (d0 - xi)`, or `None` if the full step `t_max` is guaranteed safe.
pub fn accd_point_point(
    mut x0: Vec3,
    mut x1: Vec3,
    p0_in: Vec3,
    p1_in: Vec3,
    xi: f64,
    s_conserv: f64,
    t_max: f64,
) -> Option<f64> {
    let avg = 0.5 * (p0_in + p1_in);
    let p0 = p0_in - avg;
    let p1 = p1_in - avg;
    let l_p = p0.norm() + p1.norm();
    if l_p <= 1e-15 {
        return None;
    }

    let s = s_conserv.clamp(1e-3, 0.9);
    let xi = xi.max(0.0);
    let xi2 = xi * xi;
    let mut d2 = (x0 - x1).norm_sq();
    if d2 <= xi2 {
        return Some(0.0);
    }

    // Cancellation-safe gap (d^2 - xi^2) / (sqrt(d^2) + xi) == d - xi (C-IPC Alg. 1 lines 9-11).
    let mut gap = (d2 - xi2) / (d2.sqrt() + xi);
    let g_target = s * gap;
    let mut t = 0.0;
    let mut t_l = (1.0 - s) * gap / l_p;

    for _ in 0..ACCD_MAX_ITERS {
        x0 += p0 * t_l;
        x1 += p1 * t_l;
        d2 = (x0 - x1).norm_sq();
        gap = (d2 - xi2) / (d2.sqrt() + xi);
        if t > 0.0 && gap < g_target {
            return Some(t);
        }
        t += t_l;
        if t >= t_max {
            return None;
        }
        if gap <= 0.0 {
            return Some(t);
        }
        t_l = 0.9 * gap / l_p;
    }
    Some(t.min(t_max))
}

/// Conservative Additive CCD for a moving Point-Segment pair `(p, (a, b))` with offset `xi >= 0`.
#[allow(clippy::too_many_arguments)]
pub fn accd_point_segment(
    mut p: Vec3,
    mut a: Vec3,
    mut b: Vec3,
    dp_in: Vec3,
    da_in: Vec3,
    db_in: Vec3,
    xi: f64,
    s_conserv: f64,
    t_max: f64,
) -> Option<f64> {
    let avg = (dp_in + da_in + db_in) / 3.0;
    let dp = dp_in - avg;
    let da = da_in - avg;
    let db = db_in - avg;
    let l_p = dp.norm() + da.norm().max(db.norm());
    if l_p <= 1e-15 {
        return None;
    }

    let s = s_conserv.clamp(1e-3, 0.9);
    let xi = xi.max(0.0);
    let xi2 = xi * xi;
    let mut d2 = point_segment_dist_sq_fast(p, a, b);
    if d2 <= xi2 {
        return Some(0.0);
    }

    let mut gap = (d2 - xi2) / (d2.sqrt() + xi);
    let g_target = s * gap;
    let mut t = 0.0;
    let mut t_l = (1.0 - s) * gap / l_p;

    for _ in 0..ACCD_MAX_ITERS {
        p += dp * t_l;
        a += da * t_l;
        b += db * t_l;
        d2 = point_segment_dist_sq_fast(p, a, b);
        gap = (d2 - xi2) / (d2.sqrt() + xi);
        if t > 0.0 && gap < g_target {
            return Some(t);
        }
        t += t_l;
        if t >= t_max {
            return None;
        }
        if gap <= 0.0 {
            return Some(t);
        }
        t_l = 0.9 * gap / l_p;
    }
    Some(t.min(t_max))
}

/// Conservative Additive CCD for a moving Segment-Segment (Edge-Edge) pair `((a0, a1), (b0, b1))`
/// with finite thickness offset `xi >= 0` (C-IPC Li et al. SIGGRAPH 2021, Algorithm 1 verbatim).
#[allow(clippy::too_many_arguments)]
pub fn accd_segment_segment(
    mut a0: Vec3,
    mut a1: Vec3,
    mut b0: Vec3,
    mut b1: Vec3,
    da0_in: Vec3,
    da1_in: Vec3,
    db0_in: Vec3,
    db1_in: Vec3,
    xi: f64,
    s_conserv: f64,
    t_max: f64,
) -> Option<f64> {
    let avg = 0.25 * (da0_in + da1_in + db0_in + db1_in);
    let da0 = da0_in - avg;
    let da1 = da1_in - avg;
    let db0 = db0_in - avg;
    let db1 = db1_in - avg;
    let l_p = da0.norm().max(da1.norm()) + db0.norm().max(db1.norm());
    if l_p <= 1e-15 {
        return None;
    }

    let s = s_conserv.clamp(1e-3, 0.9);
    let xi = xi.max(0.0);
    let xi2 = xi * xi;
    let mut d2 = segment_segment_dist_sq(a0, a1, b0, b1).0;
    if d2 <= xi2 {
        return Some(0.0);
    }

    let mut gap = (d2 - xi2) / (d2.sqrt() + xi);
    let g_target = s * gap;
    let mut t = 0.0;
    let mut t_l = (1.0 - s) * gap / l_p;

    for _ in 0..ACCD_MAX_ITERS {
        a0 += da0 * t_l;
        a1 += da1 * t_l;
        b0 += db0 * t_l;
        b1 += db1 * t_l;
        d2 = segment_segment_dist_sq(a0, a1, b0, b1).0;
        gap = (d2 - xi2) / (d2.sqrt() + xi);
        if t > 0.0 && gap < g_target {
            return Some(t);
        }
        t += t_l;
        if t >= t_max {
            return None;
        }
        if gap <= 0.0 {
            return Some(t);
        }
        t_l = 0.9 * gap / l_p;
    }
    Some(t.min(t_max))
}

/// Conservative Lipschitz-certified Additive CCD for a point moving along `x(t) = x0 + t * dx`
/// against an arbitrary 3D [`SdfField`] with thickness offset `xi >= 0`.
pub fn accd_point_sdf(
    sdf: &dyn SdfField,
    x0: Vec3,
    dx: Vec3,
    xi: f64,
    s_conserv: f64,
    t_max: f64,
) -> Option<f64> {
    let lip_speed = sdf.lipschitz().max(1.0) * dx.norm();
    if lip_speed <= 1e-15 {
        return None;
    }
    let s = s_conserv.clamp(1e-3, 0.9);
    let xi = xi.max(0.0);
    let phi0 = sdf.query(x0).phi;
    let mut gap = phi0 - xi;
    if gap <= 0.0 {
        return Some(0.0);
    }
    let g_target = s * gap;
    let mut t = 0.0;
    let mut t_l = (1.0 - s) * gap / lip_speed;

    for _ in 0..ACCD_MAX_ITERS {
        let x_next = x0 + dx * (t + t_l);
        gap = sdf.query(x_next).phi - xi;
        if t > 0.0 && gap < g_target {
            return Some(t);
        }
        t += t_l;
        if t >= t_max {
            return None;
        }
        if gap <= 0.0 {
            return Some(t);
        }
        t_l = 0.9 * gap / lip_speed;
    }
    Some(t.min(t_max))
}

/// Compute global intersection-free step clamp `t_safe in [0, 1]` across all non-adjacent
/// moving beam segments using swept spatial-hash broadphase + conservative ACCD.
pub fn compute_safe_step_segments(
    x0: &[Vec3],
    x1: &[Vec3],
    segments: &[(usize, usize)],
    thickness: f64,
    s_conserv: f64,
) -> f64 {
    if segments.len() < 2 {
        return 1.0;
    }
    let mut max_len = 0.0_f64;
    let mut max_disp = 0.0_f64;
    for &(u, v) in segments {
        max_len = max_len.max((x0[v] - x0[u]).norm()).max((x1[v] - x1[u]).norm());
        max_disp = max_disp
            .max((x1[u] - x0[u]).norm())
            .max((x1[v] - x0[v]).norm());
    }
    let cell_size = (max_len + max_disp + 2.0 * thickness).max(0.02);
    let mut grid = SpatialHashGrid::new(cell_size);
    let pairs = grid.find_swept_segment_pairs(x0, x1, segments, thickness * 1.1 + 1e-6);

    let mut t_safe = 1.0_f64;
    for (s0, s1) in pairs {
        let (u0, v0) = segments[s0];
        let (u1, v1) = segments[s1];
        if let Some(toi) = accd_segment_segment(
            x0[u0],
            x0[v0],
            x0[u1],
            x0[v1],
            x1[u0] - x0[u0],
            x1[v0] - x0[v0],
            x1[u1] - x0[u1],
            x1[v1] - x0[v1],
            thickness,
            s_conserv,
            t_safe,
        ) {
            if toi < t_safe {
                t_safe = toi;
                if t_safe <= 0.0 {
                    return 0.0;
                }
            }
        }
    }
    t_safe
}

/// Compute global intersection-free step clamp `t_safe in [0, 1]` across all non-adjacent
/// moving beam segments using swept spatial-hash broadphase + provably conservative
/// **Tight Inclusion Edge-Edge CCD** (Wang et al., TOG 2021).
pub fn compute_safe_step_segments_tight_inclusion(
    x0: &[Vec3],
    x1: &[Vec3],
    segments: &[(usize, usize)],
    cfg: &TightInclusionConfig,
) -> f64 {
    if segments.len() < 2 {
        return 1.0;
    }
    let min_dist = cfg.min_dist.max(0.0);
    let mut max_len = 0.0_f64;
    let mut max_disp = 0.0_f64;
    for &(u, v) in segments {
        max_len = max_len.max((x0[v] - x0[u]).norm()).max((x1[v] - x1[u]).norm());
        max_disp = max_disp
            .max((x1[u] - x0[u]).norm())
            .max((x1[v] - x0[v]).norm());
    }
    let cell_size = (max_len + max_disp + 2.0 * min_dist).max(0.02);
    let mut grid = SpatialHashGrid::new(cell_size);
    let pairs = grid.find_swept_segment_pairs(x0, x1, segments, min_dist * 1.1 + 1e-6);

    let mut t_safe = 1.0_f64;
    for (s0, s1) in pairs {
        let (u0, v0) = segments[s0];
        let (u1, v1) = segments[s1];
        if let Some(toi) = tight_inclusion_edge_edge_ccd(
            x0[u0],
            x0[v0],
            x0[u1],
            x0[v1],
            x1[u0],
            x1[v0],
            x1[u1],
            x1[v1],
            t_safe,
            cfg,
        ) {
            if toi < t_safe {
                t_safe = toi;
                if t_safe <= 0.0 {
                    return 0.0;
                }
            }
        }
    }
    t_safe
}

/// Compute global safe step clamp `t_safe in [0, 1]` for a set of moving points against an SDF world.
pub fn compute_safe_step_points_sdf(
    sdf: &dyn SdfField,
    x0: &[Vec3],
    x1: &[Vec3],
    thickness: f64,
    s_conserv: f64,
) -> f64 {
    let mut t_safe = 1.0_f64;
    for (&p0, &p1) in x0.iter().zip(x1.iter()) {
        if let Some(toi) = accd_point_sdf(sdf, p0, p1 - p0, thickness, s_conserv, t_safe) {
            if toi < t_safe {
                t_safe = toi;
                if t_safe <= 0.0 {
                    return 0.0;
                }
            }
        }
    }
    t_safe
}

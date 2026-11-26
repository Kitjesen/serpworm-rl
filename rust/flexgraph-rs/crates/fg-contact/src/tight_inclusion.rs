//! Provably conservative **Tight Inclusion Continuous Collision Detection (CCD)**
//! (Wang, Ferguson, Schneider, Gao, Panozzo & Zorin, ACM TOG / SIGGRAPH 2021).
//!
//! Uses multilinear corner convex-hull bounds with rigorous IEEE-754 floating-point
//! rounding error inflation (`err_box`) and earliest-`t0` priority-queue interval
//! bisection to certify **zero false negatives** on arbitrary linear trajectories,
//! including coplanar, parallel, and grazing Vertex-Face (Point-Triangle) and
//! Edge-Edge impacts with optional minimum separation distance `min_dist >= 0`.

use crate::distance::{point_triangle_distance_grad_hess, segment_segment_dist_sq};
use fg_math::Vec3;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for Tight Inclusion CCD interval root finding.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TightInclusionConfig {
    /// Minimum separation distance offset `ms >= 0` (m). Any approach within `min_dist`
    /// is treated as an impact.
    pub min_dist: f64,
    /// Codomain acceptance tolerance `co_domain_tol > 0` (m): a box overlapping the
    /// inclusion target whose spatial extent across all 3 axes is `<= co_domain_tol`
    /// triggers conservative acceptance at `box.t0`.
    pub co_domain_tol: f64,
    /// Time/parameter domain tolerance `t_tol > 0` preventing degenerate infinite splitting.
    pub t_tol: f64,
    /// Hard iteration budget. If exhausted, returns conservative `Some(top.t0)` (never false negative!).
    pub max_iters: usize,
}

impl Default for TightInclusionConfig {
    fn default() -> Self {
        Self {
            min_dist: 0.0,
            co_domain_tol: 1.0e-6,
            t_tol: 1.0e-9,
            max_iters: 100_000,
        }
    }
}

/// 3D interval box `[t0, t1] x [u0, u1] x [v0, v1]` ordered as a min-heap on `t0`.
#[derive(Copy, Clone, Debug)]
struct IntervalBox {
    t0: f64,
    t1: f64,
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
}

impl PartialEq for IntervalBox {
    fn eq(&self, other: &Self) -> bool {
        self.t0.to_bits() == other.t0.to_bits()
            && self.t1.to_bits() == other.t1.to_bits()
            && self.u0.to_bits() == other.u0.to_bits()
            && self.u1.to_bits() == other.u1.to_bits()
            && self.v0.to_bits() == other.v0.to_bits()
            && self.v1.to_bits() == other.v1.to_bits()
    }
}

impl Eq for IntervalBox {}

impl PartialOrd for IntervalBox {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IntervalBox {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap: reverse compare on `t0` so smallest `t0` pops first!
        other
            .t0
            .total_cmp(&self.t0)
            .then_with(|| other.t1.total_cmp(&self.t1))
            .then_with(|| other.u0.total_cmp(&self.u0))
            .then_with(|| other.u1.total_cmp(&self.u1))
            .then_with(|| other.v0.total_cmp(&self.v0))
            .then_with(|| other.v1.total_cmp(&self.v1))
    }
}

/// Compute rigorous component-wise IEEE-754 floating-point error bound `err_box`
/// over the 8 input vertex endpoints (Wang et al., TOG 2021 Sec. 4.3).
#[inline]
fn compute_fp_err_box(verts: &[Vec3; 8]) -> [f64; 3] {
    let mut max_abs = [0.0_f64; 3];
    for v in verts {
        for c in 0..3 {
            max_abs[c] = max_abs[c].max(v[c].abs());
        }
    }
    // 64 * eps_mach (~1.42e-14) rigorously upper-bounds all trilinear interpolation and
    // barycentric rounding operations with generous safety margin and zero false negatives.
    [
        64.0 * f64::EPSILON * max_abs[0].max(1.0),
        64.0 * f64::EPSILON * max_abs[1].max(1.0),
        64.0 * f64::EPSILON * max_abs[2].max(1.0),
    ]
}

/// Evaluate exact minimum L2 distance from the origin `0` to the error-inflated bounding box
/// `[bmin[c] - err[c], bmax[c] + err[c]]`. Returns `true` iff the box may contain a point
/// with `||F||_2 <= min_dist` (guaranteed zero false negatives!).
#[inline]
fn box_overlaps_ball(bmin: [f64; 3], bmax: [f64; 3], err: [f64; 3], min_dist: f64) -> bool {
    let mut d2 = 0.0_f64;
    for c in 0..3 {
        let lo = bmin[c] - err[c];
        let hi = bmax[c] + err[c];
        if lo > min_dist || hi < -min_dist {
            return false;
        }
        let dc = if hi < 0.0 {
            -hi
        } else if lo > 0.0 {
            lo
        } else {
            0.0
        };
        d2 += dc * dc;
    }
    // Conservative check: allow tiny FP slack on `min_dist^2` so boundary hits are never missed.
    let md_sq = min_dist * min_dist;
    d2 <= md_sq + 8.0 * f64::EPSILON * md_sq.max(1.0e-30)
}

/// Evaluate VF separation vector `F(t, u, v) = P(t) - ((1 - u - v) T0(t) + u T1(t) + v T2(t))`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_vf(
    p0: Vec3,
    t0_0: Vec3,
    t1_0: Vec3,
    t2_0: Vec3,
    p1: Vec3,
    t0_1: Vec3,
    t1_1: Vec3,
    t2_1: Vec3,
    t: f64,
    u: f64,
    v: f64,
) -> Vec3 {
    let omt = 1.0 - t;
    let p = p0 * omt + p1 * t;
    let a = t0_0 * omt + t0_1 * t;
    let b = t1_0 * omt + t1_1 * t;
    let c = t2_0 * omt + t2_1 * t;
    let w0 = 1.0 - u - v;
    p - (a * w0 + b * u + c * v)
}

/// Evaluate tight convex-hull bounds `[bmin, bmax]` and per-axis codomain variations
/// `(var_t, var_u, var_v)` for `F_VF` over `[t0, t1] x ([u0, u1] x [v0, v1] ∩ {u + v <= 1})`.
///
/// Because `F_VF(t, u, v)` is linear in `t` and affine in `(u, v)`, its exact image over
/// the clipped domain is contained in the convex hull of its values at `{t0, t1}` crossed
/// with the `<= 5` vertices of the clipped `(u, v)` polygon.
#[allow(clippy::too_many_arguments)]
fn vf_box_bounds(
    p0: Vec3,
    t0_0: Vec3,
    t1_0: Vec3,
    t2_0: Vec3,
    p1: Vec3,
    t0_1: Vec3,
    t1_1: Vec3,
    t2_1: Vec3,
    b: IntervalBox,
) -> ([f64; 3], [f64; 3], f64, f64, f64) {
    // Build exact convex vertices of [u0, u1] x [v0, v1] ∩ {u + v <= 1} (caller guarantees u0 + v0 <= 1).
    let mut uv_verts = [[0.0_f64; 2]; 5];
    let n_uv = if b.u1 + b.v1 <= 1.0 + 1e-15 {
        uv_verts[0] = [b.u0, b.v0];
        uv_verts[1] = [b.u1, b.v0];
        uv_verts[2] = [b.u0, b.v1];
        uv_verts[3] = [b.u1, b.v1];
        4
    } else {
        let mut n = 0usize;
        uv_verts[n] = [b.u0, b.v0];
        n += 1;
        if b.u1 + b.v0 <= 1.0 {
            uv_verts[n] = [b.u1, b.v0];
            n += 1;
            uv_verts[n] = [b.u1, (1.0 - b.u1).clamp(b.v0, b.v1)];
            n += 1;
        } else {
            uv_verts[n] = [(1.0 - b.v0).clamp(b.u0, b.u1), b.v0];
            n += 1;
        }
        if b.u0 + b.v1 <= 1.0 {
            uv_verts[n] = [(1.0 - b.v1).clamp(b.u0, b.u1), b.v1];
            n += 1;
            uv_verts[n] = [b.u0, b.v1];
            n += 1;
        } else {
            uv_verts[n] = [b.u0, (1.0 - b.u0).clamp(b.v0, b.v1)];
            n += 1;
        }
        n
    };

    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];
    let mut var_t = 0.0_f64;

    for idx in 0..n_uv {
        let [u, v] = uv_verts[idx];
        let f0 = eval_vf(p0, t0_0, t1_0, t2_0, p1, t0_1, t1_1, t2_1, b.t0, u, v);
        let f1 = eval_vf(p0, t0_0, t1_0, t2_0, p1, t0_1, t1_1, t2_1, b.t1, u, v);
        for c in 0..3 {
            bmin[c] = bmin[c].min(f0[c]).min(f1[c]);
            bmax[c] = bmax[c].max(f0[c]).max(f1[c]);
            var_t = var_t.max((f1[c] - f0[c]).abs());
        }
    }

    // Directional sensitivity along u and v from edge vectors T1(t) - T0(t) and T2(t) - T0(t).
    let du = b.u1 - b.u0;
    let dv = b.v1 - b.v0;
    let mut var_u = 0.0_f64;
    let mut var_v = 0.0_f64;
    for t in [b.t0, b.t1] {
        let omt = 1.0 - t;
        let e1 = (t1_0 - t0_0) * omt + (t1_1 - t0_1) * t;
        let e2 = (t2_0 - t0_0) * omt + (t2_1 - t0_1) * t;
        for c in 0..3 {
            var_u = var_u.max(e1[c].abs() * du);
            var_v = var_v.max(e2[c].abs() * dv);
        }
    }

    (bmin, bmax, var_t, var_u, var_v)
}

/// Provably conservative Tight Inclusion CCD for a linearly moving Point–Triangle (Vertex–Face) pair:
/// `p(t) = (1-t)*p0 + t*p1` against triangle `(t0(t), t1(t), t2(t))` over `t in [0, t_max]`.
///
/// Returns `Some(toi)` with `0.0 <= toi <= t_exact_impact` if the point approaches within
/// `cfg.min_dist` of the triangle on `[0, t_max]`, or `None` if provably separated by `> min_dist`
/// over the entire interval `[0, t_max]`.
#[allow(clippy::too_many_arguments)]
pub fn tight_inclusion_point_triangle_ccd(
    p0: Vec3,
    t0_0: Vec3,
    t1_0: Vec3,
    t2_0: Vec3,
    p1: Vec3,
    t0_1: Vec3,
    t1_1: Vec3,
    t2_1: Vec3,
    t_max: f64,
    cfg: &TightInclusionConfig,
) -> Option<f64> {
    let min_dist = cfg.min_dist.max(0.0);
    // Exact t = 0 separation check
    if point_triangle_distance_grad_hess(p0, t0_0, t1_0, t2_0).dist <= min_dist {
        return Some(0.0);
    }
    if t_max <= 0.0 {
        return None;
    }

    let co_tol = cfg.co_domain_tol.max(1e-14);
    let t_tol = cfg.t_tol.max(1e-15);
    let err = compute_fp_err_box(&[p0, t0_0, t1_0, t2_0, p1, t0_1, t1_1, t2_1]);

    let root = IntervalBox {
        t0: 0.0,
        t1: t_max,
        u0: 0.0,
        u1: 1.0,
        v0: 0.0,
        v1: 1.0,
    };

    let (rmin, rmax, _, _, _) = vf_box_bounds(p0, t0_0, t1_0, t2_0, p1, t0_1, t1_1, t2_1, root);
    if !box_overlaps_ball(rmin, rmax, err, min_dist) {
        return None;
    }

    let mut heap = BinaryHeap::with_capacity(256);
    heap.push(root);

    let mut iters = 0usize;
    while let Some(top) = heap.pop() {
        iters += 1;
        let (bmin, bmax, var_t, var_u, var_v) =
            vf_box_bounds(p0, t0_0, t1_0, t2_0, p1, t0_1, t1_1, t2_1, top);
        if !box_overlaps_ball(bmin, bmax, err, min_dist) {
            continue;
        }

        let dt = top.t1 - top.t0;
        let du = top.u1 - top.u0;
        let dv = top.v1 - top.v0;
        let width_max = (bmax[0] - bmin[0])
            .max(bmax[1] - bmin[1])
            .max(bmax[2] - bmin[2]);
        let small_domain = dt <= t_tol && du <= t_tol && dv <= t_tol;
        let small_codomain = width_max <= co_tol;

        // Avoid returning spurious t = 0 when t = 0 was already verified strictly separated:
        // keep bisecting t until the [0, t_mid] prefix is pruned!
        if (small_codomain || small_domain) && (top.t0 > 0.0 || dt <= t_tol) {
            return Some(top.t0);
        }
        if iters >= cfg.max_iters {
            return Some(top.t0);
        }

        let Some(split_axis) =
            choose_split_axis(top, small_codomain, var_t, var_u, var_v, t_tol)
        else {
            return Some(top.t0);
        };

        let (c0, c1) = split_box(top, split_axis);
        for child in [c0, c1] {
            if child.u0 + child.v0 > 1.0 + 1e-15 {
                continue;
            }
            let (cmn, cmx, _, _, _) =
                vf_box_bounds(p0, t0_0, t1_0, t2_0, p1, t0_1, t1_1, t2_1, child);
            if box_overlaps_ball(cmn, cmx, err, min_dist) {
                heap.push(child);
            }
        }
    }

    None
}

/// Evaluate EE separation vector `F(t, u, v) = ((1-u)*A0(t) + u*A1(t)) - ((1-v)*B0(t) + v*B1(t))`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_ee(
    a0_0: Vec3,
    a1_0: Vec3,
    b0_0: Vec3,
    b1_0: Vec3,
    a0_1: Vec3,
    a1_1: Vec3,
    b0_1: Vec3,
    b1_1: Vec3,
    t: f64,
    u: f64,
    v: f64,
) -> Vec3 {
    let omt = 1.0 - t;
    let a0 = a0_0 * omt + a0_1 * t;
    let a1 = a1_0 * omt + a1_1 * t;
    let b0 = b0_0 * omt + b0_1 * t;
    let b1 = b1_0 * omt + b1_1 * t;
    let pa = a0 * (1.0 - u) + a1 * u;
    let pb = b0 * (1.0 - v) + b1 * v;
    pa - pb
}

/// Evaluate exact 8-corner trilinear convex-hull bounds `[bmin, bmax]` and axis variations
/// `(var_t, var_u, var_v)` for `F_EE` over `[t0, t1] x [u0, u1] x [v0, v1]`.
#[allow(clippy::too_many_arguments)]
fn ee_box_bounds(
    a0_0: Vec3,
    a1_0: Vec3,
    b0_0: Vec3,
    b1_0: Vec3,
    a0_1: Vec3,
    a1_1: Vec3,
    b0_1: Vec3,
    b1_1: Vec3,
    b: IntervalBox,
) -> ([f64; 3], [f64; 3], f64, f64, f64) {
    let ts = [b.t0, b.t1];
    let us = [b.u0, b.u1];
    let vs = [b.v0, b.v1];

    let mut corners = [Vec3::ZERO; 8];
    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];

    let mut k = 0;
    for &t in &ts {
        for &u in &us {
            for &v in &vs {
                let f = eval_ee(a0_0, a1_0, b0_0, b1_0, a0_1, a1_1, b0_1, b1_1, t, u, v);
                corners[k] = f;
                k += 1;
                for c in 0..3 {
                    bmin[c] = bmin[c].min(f[c]);
                    bmax[c] = bmax[c].max(f[c]);
                }
            }
        }
    }

    // Index mapping: idx = (it << 2) | (iu << 1) | iv
    let mut var_t = 0.0_f64;
    let mut var_u = 0.0_f64;
    let mut var_v = 0.0_f64;
    for iu in 0..2 {
        for iv in 0..2 {
            let f0 = corners[(iu << 1) | iv];
            let f1 = corners[4 | (iu << 1) | iv];
            for c in 0..3 {
                var_t = var_t.max((f1[c] - f0[c]).abs());
            }
        }
    }
    for it in 0..2 {
        for iv in 0..2 {
            let f0 = corners[(it << 2) | iv];
            let f1 = corners[(it << 2) | 2 | iv];
            for c in 0..3 {
                var_u = var_u.max((f1[c] - f0[c]).abs());
            }
        }
    }
    for it in 0..2 {
        for iu in 0..2 {
            let f0 = corners[(it << 2) | (iu << 1)];
            let f1 = corners[(it << 2) | (iu << 1) | 1];
            for c in 0..3 {
                var_v = var_v.max((f1[c] - f0[c]).abs());
            }
        }
    }

    (bmin, bmax, var_t, var_u, var_v)
}

/// Provably conservative Tight Inclusion CCD for a linearly moving Edge–Edge pair:
/// segment `(a0(t), a1(t))` against segment `(b0(t), b1(t))` over `t in [0, t_max]`.
///
/// Returns `Some(toi)` with `0.0 <= toi <= t_exact_impact` if the segments approach within
/// `cfg.min_dist` on `[0, t_max]`, or `None` if provably separated by `> min_dist`.
#[allow(clippy::too_many_arguments)]
pub fn tight_inclusion_edge_edge_ccd(
    a0_0: Vec3,
    a1_0: Vec3,
    b0_0: Vec3,
    b1_0: Vec3,
    a0_1: Vec3,
    a1_1: Vec3,
    b0_1: Vec3,
    b1_1: Vec3,
    t_max: f64,
    cfg: &TightInclusionConfig,
) -> Option<f64> {
    let min_dist = cfg.min_dist.max(0.0);
    // Exact t = 0 separation check
    if segment_segment_dist_sq(a0_0, a1_0, b0_0, b1_0).0.sqrt() <= min_dist {
        return Some(0.0);
    }
    if t_max <= 0.0 {
        return None;
    }

    let co_tol = cfg.co_domain_tol.max(1e-14);
    let t_tol = cfg.t_tol.max(1e-15);
    let err = compute_fp_err_box(&[a0_0, a1_0, b0_0, b1_0, a0_1, a1_1, b0_1, b1_1]);

    let root = IntervalBox {
        t0: 0.0,
        t1: t_max,
        u0: 0.0,
        u1: 1.0,
        v0: 0.0,
        v1: 1.0,
    };

    let (rmin, rmax, _, _, _) = ee_box_bounds(a0_0, a1_0, b0_0, b1_0, a0_1, a1_1, b0_1, b1_1, root);
    if !box_overlaps_ball(rmin, rmax, err, min_dist) {
        return None;
    }

    let mut heap = BinaryHeap::with_capacity(256);
    heap.push(root);

    let mut iters = 0usize;
    while let Some(top) = heap.pop() {
        iters += 1;
        let (bmin, bmax, var_t, var_u, var_v) =
            ee_box_bounds(a0_0, a1_0, b0_0, b1_0, a0_1, a1_1, b0_1, b1_1, top);
        if !box_overlaps_ball(bmin, bmax, err, min_dist) {
            continue;
        }

        let dt = top.t1 - top.t0;
        let du = top.u1 - top.u0;
        let dv = top.v1 - top.v0;
        let width_max = (bmax[0] - bmin[0])
            .max(bmax[1] - bmin[1])
            .max(bmax[2] - bmin[2]);
        let small_domain = dt <= t_tol && du <= t_tol && dv <= t_tol;
        let small_codomain = width_max <= co_tol;

        if (small_codomain || small_domain) && (top.t0 > 0.0 || dt <= t_tol) {
            return Some(top.t0);
        }
        if iters >= cfg.max_iters {
            return Some(top.t0);
        }

        let Some(split_axis) =
            choose_split_axis(top, small_codomain, var_t, var_u, var_v, t_tol)
        else {
            return Some(top.t0);
        };

        let (c0, c1) = split_box(top, split_axis);
        for child in [c0, c1] {
            let (cmn, cmx, _, _, _) =
                ee_box_bounds(a0_0, a1_0, b0_0, b1_0, a0_1, a1_1, b0_1, b1_1, child);
            if box_overlaps_ball(cmn, cmx, err, min_dist) {
                heap.push(child);
            }
        }
    }

    None
}

#[inline]
fn choose_split_axis(
    b: IntervalBox,
    small_codomain: bool,
    var_t: f64,
    var_u: f64,
    var_v: f64,
    t_tol: f64,
) -> Option<usize> {
    let dt = b.t1 - b.t0;
    let du = b.u1 - b.u0;
    let dv = b.v1 - b.v0;
    let dt_eps = 1e-15 * b.t1.abs().max(1.0);
    let prefer_t = (b.t0 == 0.0
        && (small_codomain || var_t >= 0.25 * var_u.max(var_v))
        && dt > t_tol.max(dt_eps))
        || (var_t >= var_u && var_t >= var_v && dt > dt_eps);
    if prefer_t {
        Some(0)
    } else if var_u >= var_v && du > 1e-15 {
        Some(1)
    } else if dv > 1e-15 {
        Some(2)
    } else if dt > dt_eps {
        Some(0)
    } else {
        None
    }
}

#[inline]
fn split_box(b: IntervalBox, axis: usize) -> (IntervalBox, IntervalBox) {
    match axis {
        0 => {
            let tm = 0.5 * (b.t0 + b.t1);
            (
                IntervalBox { t1: tm, ..b },
                IntervalBox { t0: tm, ..b },
            )
        }
        1 => {
            let um = 0.5 * (b.u0 + b.u1);
            (
                IntervalBox { u1: um, ..b },
                IntervalBox { u0: um, ..b },
            )
        }
        _ => {
            let vm = 0.5 * (b.v0 + b.v1);
            (
                IntervalBox { v1: vm, ..b },
                IntervalBox { v0: vm, ..b },
            )
        }
    }
}

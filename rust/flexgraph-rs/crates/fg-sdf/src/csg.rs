//! CSG boolean & smooth-blend operators (`UnionSdf`, `SmoothUnionSdf`, `DifferenceSdf`,
//! `IntersectionSdf`, `SmoothDifferenceSdf`, `SmoothIntersectionSdf`, `TransformSdf`) and
//! BVH-accelerated [`WorldSdf`] scene composer.

use crate::bvh::{Aabb, Bvh};
use crate::material::{MaterialLibrary, MAT_HARD_FLOOR};
use crate::primitives::PlaneSdf;
use crate::{SdfField, SdfQuery};
use fg_math::{Se3, Vec3};
use std::sync::Arc;

/// Rigidly transformed child SDF (`pose` maps child-local coordinates to world coordinates).
pub struct TransformSdf {
    pub child: Arc<dyn SdfField>,
    pub pose: Se3,
}

impl TransformSdf {
    pub fn new(child: Arc<dyn SdfField>, pose: Se3) -> Self {
        Self { child, pose }
    }
}

impl SdfField for TransformSdf {
    #[inline]
    fn query(&self, pos: Vec3) -> SdfQuery {
        let p_loc = self.pose.rot.inv_rotate(pos - self.pose.pos);
        let mut q = self.child.query(p_loc);
        q.normal = self.pose.rot.rotate(q.normal).normalized_or(Vec3::EZ);
        if q.tangent.norm_sq() > 0.0 {
            q.tangent = self.pose.rot.rotate(q.tangent).normalized();
        }
        q
    }

    fn bounds(&self) -> Option<Aabb> {
        let b = self.child.bounds()?;
        let corners = [
            Vec3::new(b.min.x(), b.min.y(), b.min.z()),
            Vec3::new(b.max.x(), b.min.y(), b.min.z()),
            Vec3::new(b.min.x(), b.max.y(), b.min.z()),
            Vec3::new(b.max.x(), b.max.y(), b.min.z()),
            Vec3::new(b.min.x(), b.min.y(), b.max.z()),
            Vec3::new(b.max.x(), b.min.y(), b.max.z()),
            Vec3::new(b.min.x(), b.max.y(), b.max.z()),
            Vec3::new(b.max.x(), b.max.y(), b.max.z()),
        ];
        let mut out = Aabb::EMPTY;
        for c in corners {
            out = out.union(Aabb::from_point(self.pose.transform_point(c)));
        }
        Some(out)
    }

    fn lipschitz(&self) -> f64 {
        self.child.lipschitz()
    }
}

/// Exact CSG union (`min(phi_i)`) over an arbitrary list of SDF fields.
pub struct UnionSdf {
    pub children: Vec<Arc<dyn SdfField>>,
}

impl UnionSdf {
    pub fn new(children: Vec<Arc<dyn SdfField>>) -> Self {
        Self { children }
    }
}

impl SdfField for UnionSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let mut best = SdfQuery::FAR_EMPTY;
        for c in &self.children {
            let q = c.query(pos);
            if q.phi < best.phi {
                best = q;
            }
        }
        best
    }

    fn bounds(&self) -> Option<Aabb> {
        if self.children.is_empty() {
            return None;
        }
        let mut acc = Aabb::EMPTY;
        for c in &self.children {
            acc = acc.union(c.bounds()?);
        }
        Some(acc)
    }

    fn lipschitz(&self) -> f64 {
        self.children
            .iter()
            .map(|c| c.lipschitz())
            .fold(1.0_f64, f64::max)
    }
}

/// $C^2$ cubic polynomial smooth-union operator (`smin(a, b; k)`):
/// `h = max(k - |a - b|, 0) / k`, `phi = min(a, b) - (k / 6) * h^3`,
/// with exact analytical gradient `(1 - 0.5 h^2) n_min + (0.5 h^2) n_max`!
pub struct SmoothUnionSdf {
    pub a: Arc<dyn SdfField>,
    pub b: Arc<dyn SdfField>,
    pub k: f64,
}

impl SmoothUnionSdf {
    pub fn new(a: Arc<dyn SdfField>, b: Arc<dyn SdfField>, k: f64) -> Self {
        Self {
            a,
            b,
            k: k.max(0.0),
        }
    }
}

impl SdfField for SmoothUnionSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let qa = self.a.query(pos);
        let qb = self.b.query(pos);
        if self.k <= 1e-12 {
            return if qa.phi <= qb.phi { qa } else { qb };
        }
        let (q_min, q_max) = if qa.phi <= qb.phi { (qa, qb) } else { (qb, qa) };
        let diff = q_max.phi - q_min.phi;
        if diff >= self.k {
            return q_min;
        }
        let h = 1.0 - diff / self.k;
        let phi = q_min.phi - (self.k / 6.0) * h * h * h;
        let w_max = 0.5 * h * h;
        let w_min = 1.0 - w_max;
        let grad = q_min.normal * w_min + q_max.normal * w_max;
        SdfQuery {
            phi,
            normal: grad.normalized_or(q_min.normal),
            arc_s: q_min.arc_s,
            tangent: q_min.tangent,
            mat_id: q_min.mat_id,
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        let ba = self.a.bounds()?;
        let bb = self.b.bounds()?;
        Some(ba.union(bb).expand(self.k))
    }

    fn lipschitz(&self) -> f64 {
        self.a.lipschitz().max(self.b.lipschitz())
    }
}

/// Exact CSG subtraction `A \ B`: `phi = max(phi_a, -phi_b)`.
pub struct DifferenceSdf {
    pub a: Arc<dyn SdfField>,
    pub b: Arc<dyn SdfField>,
}

impl DifferenceSdf {
    pub fn new(a: Arc<dyn SdfField>, b: Arc<dyn SdfField>) -> Self {
        Self { a, b }
    }
}

impl SdfField for DifferenceSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let qa = self.a.query(pos);
        let qb = self.b.query(pos);
        let neg_b = -qb.phi;
        if qa.phi >= neg_b {
            qa
        } else {
            SdfQuery {
                phi: neg_b,
                normal: -qb.normal,
                arc_s: qa.arc_s,
                tangent: qa.tangent,
                mat_id: qb.mat_id,
            }
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        self.a.bounds()
    }

    fn lipschitz(&self) -> f64 {
        self.a.lipschitz().max(self.b.lipschitz())
    }
}

/// Exact CSG intersection `A ∩ B`: `phi = max(phi_a, phi_b)`.
pub struct IntersectionSdf {
    pub a: Arc<dyn SdfField>,
    pub b: Arc<dyn SdfField>,
}

impl IntersectionSdf {
    pub fn new(a: Arc<dyn SdfField>, b: Arc<dyn SdfField>) -> Self {
        Self { a, b }
    }
}

impl SdfField for IntersectionSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let qa = self.a.query(pos);
        let qb = self.b.query(pos);
        if qa.phi >= qb.phi {
            qa
        } else {
            qb
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        match (self.a.bounds(), self.b.bounds()) {
            (Some(ba), Some(bb)) => Some(Aabb::new(
                Vec3::new(
                    ba.min.x().max(bb.min.x()),
                    ba.min.y().max(bb.min.y()),
                    ba.min.z().max(bb.min.z()),
                ),
                Vec3::new(
                    ba.max.x().min(bb.max.x()),
                    ba.max.y().min(bb.max.y()),
                    ba.max.z().min(bb.max.z()),
                ),
            )),
            (Some(b), None) | (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    fn lipschitz(&self) -> f64 {
        self.a.lipschitz().max(self.b.lipschitz())
    }
}

/// $C^2$ smooth CSG intersection (`smax(a, b; k) = -smin(-a, -b; k)`).
pub struct SmoothIntersectionSdf {
    pub a: Arc<dyn SdfField>,
    pub b: Arc<dyn SdfField>,
    pub k: f64,
}

impl SmoothIntersectionSdf {
    pub fn new(a: Arc<dyn SdfField>, b: Arc<dyn SdfField>, k: f64) -> Self {
        Self {
            a,
            b,
            k: k.max(0.0),
        }
    }
}

impl SdfField for SmoothIntersectionSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let qa = self.a.query(pos);
        let qb = self.b.query(pos);
        if self.k <= 1e-12 {
            return if qa.phi >= qb.phi { qa } else { qb };
        }
        let (q_max, q_min) = if qa.phi >= qb.phi { (qa, qb) } else { (qb, qa) };
        let diff = q_max.phi - q_min.phi;
        if diff >= self.k {
            return q_max;
        }
        let h = 1.0 - diff / self.k;
        let phi = q_max.phi + (self.k / 6.0) * h * h * h;
        let w_min = 0.5 * h * h;
        let w_max = 1.0 - w_min;
        let grad = q_max.normal * w_max + q_min.normal * w_min;
        SdfQuery {
            phi,
            normal: grad.normalized_or(q_max.normal),
            arc_s: q_max.arc_s,
            tangent: q_max.tangent,
            mat_id: q_max.mat_id,
        }
    }

    fn lipschitz(&self) -> f64 {
        self.a.lipschitz().max(self.b.lipschitz())
    }
}

/// $C^2$ smooth CSG difference (`smax(a, -b; k)`).
pub struct SmoothDifferenceSdf {
    pub a: Arc<dyn SdfField>,
    pub b: Arc<dyn SdfField>,
    pub k: f64,
}

impl SmoothDifferenceSdf {
    pub fn new(a: Arc<dyn SdfField>, b: Arc<dyn SdfField>, k: f64) -> Self {
        Self {
            a,
            b,
            k: k.max(0.0),
        }
    }
}

impl SdfField for SmoothDifferenceSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let qa = self.a.query(pos);
        let mut qb = self.b.query(pos);
        qb.phi = -qb.phi;
        qb.normal = -qb.normal;
        if self.k <= 1e-12 {
            return if qa.phi >= qb.phi { qa } else { qb };
        }
        let (q_max, q_min) = if qa.phi >= qb.phi { (qa, qb) } else { (qb, qa) };
        let diff = q_max.phi - q_min.phi;
        if diff >= self.k {
            return q_max;
        }
        let h = 1.0 - diff / self.k;
        let phi = q_max.phi + (self.k / 6.0) * h * h * h;
        let w_min = 0.5 * h * h;
        let w_max = 1.0 - w_min;
        let grad = q_max.normal * w_max + q_min.normal * w_min;
        SdfQuery {
            phi,
            normal: grad.normalized_or(q_max.normal),
            arc_s: q_max.arc_s,
            tangent: q_max.tangent,
            mat_id: q_max.mat_id,
        }
    }

    fn lipschitz(&self) -> f64 {
        self.a.lipschitz().max(self.b.lipschitz())
    }
}

/// BVH-accelerated composable 3D SDF world combining unbounded terrain/cavities,
/// bounded obstacles/tubes, and a [`MaterialLibrary`].
pub struct WorldSdf {
    pub materials: MaterialLibrary,
    unbounded: Vec<Arc<dyn SdfField>>,
    bounded: Vec<Arc<dyn SdfField>>,
    bounded_aabbs: Vec<Aabb>,
    bvh: Bvh,
}

impl Default for WorldSdf {
    fn default() -> Self {
        Self::flat_ground()
    }
}

impl WorldSdf {
    /// Empty world with the provided material library.
    pub fn new(materials: MaterialLibrary) -> Self {
        Self {
            materials,
            unbounded: Vec::new(),
            bounded: Vec::new(),
            bounded_aabbs: Vec::new(),
            bvh: Bvh::default(),
        }
    }

    /// Standard flat hard ground `z = 0` with the 5-material library.
    pub fn flat_ground() -> Self {
        let mut w = Self::new(MaterialLibrary::standard());
        w.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, MAT_HARD_FLOOR)));
        w
    }

    /// Add an unbounded SDF (e.g. infinite plane, heightfield, or interior cavity tube).
    pub fn add_unbounded(&mut self, sdf: Arc<dyn SdfField>) {
        self.unbounded.push(sdf);
    }

    /// Add a bounded solid SDF (boxes, spheres, cylinders, capsules, solid tubes, voxel grids).
    /// Call [`WorldSdf::rebuild_bvh`] after adding bounded objects (automatically falls back to
    /// unbounded list if `sdf.bounds()` is `None`).
    pub fn add_bounded(&mut self, sdf: Arc<dyn SdfField>) {
        if let Some(aabb) = sdf.bounds() {
            self.bounded.push(sdf);
            self.bounded_aabbs.push(aabb);
        } else {
            self.unbounded.push(sdf);
        }
    }

    /// Rebuild the internal BVH over all bounded scene objects.
    pub fn rebuild_bvh(&mut self) {
        self.bvh = Bvh::build(&self.bounded_aabbs);
    }

    /// Query all active environment surfaces (returns exact minimum-`phi` hit across unbounded + BVH).
    #[inline]
    pub fn query_world(&self, pos: Vec3) -> SdfQuery {
        self.query(pos)
    }

    /// Query all surfaces whose signed distance is below `threshold` (useful for multi-contact
    /// corners, pipe+floor simultaneous contact, and Compliance-LCP contact set construction).
    pub fn query_all_within(&self, pos: Vec3, threshold: f64) -> Vec<SdfQuery> {
        let mut hits = Vec::new();
        for s in &self.unbounded {
            let q = s.query(pos);
            if q.phi <= threshold {
                hits.push(q);
            }
        }
        let pad = threshold.max(0.0);
        if !self.bvh.is_empty() {
            let mut candidates = Vec::new();
            let qbox = Aabb::from_point(pos).expand(pad);
            self.bvh.query_aabb(qbox, &mut candidates);
            for idx in candidates {
                let q = self.bounded[idx].query(pos);
                if q.phi <= threshold {
                    hits.push(q);
                }
            }
        } else {
            for s in &self.bounded {
                let q = s.query(pos);
                if q.phi <= threshold {
                    hits.push(q);
                }
            }
        }
        hits
    }
}

impl SdfField for WorldSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let mut best = SdfQuery::FAR_EMPTY;
        for s in &self.unbounded {
            let q = s.query(pos);
            if q.phi < best.phi {
                best = q;
            }
        }
        if !self.bvh.is_empty() {
            let init_bound_sq = if best.phi > 0.0 {
                best.phi * best.phi
            } else {
                // Already penetrating an unbounded surface; only check primitives whose AABB contains `pos`
                // (or are within `0.0` of `pos`) so deeper interior penetrations are still found.
                0.0
            };
            self.bvh.visit_nearest(pos, init_bound_sq, |idx, bound_d2| {
                let q = self.bounded[idx].query(pos);
                if q.phi < best.phi {
                    best = q;
                    *bound_d2 = if best.phi > 0.0 {
                        best.phi * best.phi
                    } else {
                        0.0
                    };
                }
            });
        } else {
            for s in &self.bounded {
                let q = s.query(pos);
                if q.phi < best.phi {
                    best = q;
                }
            }
        }
        best
    }

    fn lipschitz(&self) -> f64 {
        let l_unb = self
            .unbounded
            .iter()
            .map(|s| s.lipschitz())
            .fold(1.0_f64, f64::max);
        let l_bnd = self
            .bounded
            .iter()
            .map(|s| s.lipschitz())
            .fold(1.0_f64, f64::max);
        l_unb.max(l_bnd)
    }
}

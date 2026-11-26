//! Axis-aligned bounding boxes (AABB) and binary BVH tree with best-first
//! branch-and-bound nearest-primitive pruning.

use fg_math::Vec3;
use serde::{Deserialize, Serialize};

/// Axis-aligned 3D bounding box in FP64.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Empty inverted bounding box ready for union accumulation.
    pub const EMPTY: Self = Self {
        min: Vec3([f64::INFINITY; 3]),
        max: Vec3([f64::NEG_INFINITY; 3]),
    };

    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    #[inline]
    pub fn from_segment(a: Vec3, b: Vec3, pad: f64) -> Self {
        let min = Vec3::new(
            a.x().min(b.x()) - pad,
            a.y().min(b.y()) - pad,
            a.z().min(b.z()) - pad,
        );
        let max = Vec3::new(
            a.x().max(b.x()) + pad,
            a.y().max(b.y()) + pad,
            a.z().max(b.z()) + pad,
        );
        Self { min, max }
    }

    #[inline]
    pub fn expand(self, margin: f64) -> Self {
        let m = Vec3::new(margin, margin, margin);
        Self {
            min: self.min - m,
            max: self.max + m,
        }
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x().min(other.min.x()),
                self.min.y().min(other.min.y()),
                self.min.z().min(other.min.z()),
            ),
            max: Vec3::new(
                self.max.x().max(other.max.x()),
                self.max.y().max(other.max.y()),
                self.max.z().max(other.max.z()),
            ),
        }
    }

    #[inline]
    pub fn center(self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    #[inline]
    pub fn extents(self) -> Vec3 {
        self.max - self.min
    }

    #[inline]
    pub fn longest_axis(self) -> usize {
        let e = self.extents();
        if e.x() >= e.y() && e.x() >= e.z() {
            0
        } else if e.y() >= e.z() {
            1
        } else {
            2
        }
    }

    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.min.x() <= other.max.x()
            && self.max.x() >= other.min.x()
            && self.min.y() <= other.max.y()
            && self.max.y() >= other.min.y()
            && self.min.z() <= other.max.z()
            && self.max.z() >= other.min.z()
    }

    /// Squared Euclidean distance from `p` to this AABB (`0.0` if inside).
    #[inline]
    pub fn min_dist_sq(self, p: Vec3) -> f64 {
        let mut d2 = 0.0;
        for ax in 0..3 {
            let v = p[ax];
            if v < self.min[ax] {
                let d = self.min[ax] - v;
                d2 += d * d;
            } else if v > self.max[ax] {
                let d = v - self.max[ax];
                d2 += d * d;
            }
        }
        d2
    }
}

#[derive(Clone, Debug)]
enum BvhNodeKind {
    Internal { left: usize, right: usize },
    Leaf { start: usize, count: usize },
}

#[derive(Clone, Debug)]
struct BvhNode {
    bounds: Aabb,
    kind: BvhNodeKind,
}

/// Binary Bounding Volume Hierarchy (BVH) accelerating closest-primitive and overlap queries.
#[derive(Clone, Debug, Default)]
pub struct Bvh {
    nodes: Vec<BvhNode>,
    prim_indices: Vec<usize>,
}

impl Bvh {
    const LEAF_SIZE: usize = 4;

    /// Build a BVH over primitive AABBs.
    pub fn build(aabbs: &[Aabb]) -> Self {
        let n = aabbs.len();
        if n == 0 {
            return Self::default();
        }
        let mut prim_indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(2 * n);
        Self::build_recursive(aabbs, &mut prim_indices, 0, n, &mut nodes);
        Self {
            nodes,
            prim_indices,
        }
    }

    fn build_recursive(
        aabbs: &[Aabb],
        prim_indices: &mut [usize],
        start: usize,
        end: usize,
        nodes: &mut Vec<BvhNode>,
    ) -> usize {
        let mut bounds = Aabb::EMPTY;
        let mut centroid_bounds = Aabb::EMPTY;
        for &idx in &prim_indices[start..end] {
            bounds = bounds.union(aabbs[idx]);
            centroid_bounds = centroid_bounds.union(Aabb::from_point(aabbs[idx].center()));
        }

        let count = end - start;
        let node_idx = nodes.len();
        if count <= Self::LEAF_SIZE {
            nodes.push(BvhNode {
                bounds,
                kind: BvhNodeKind::Leaf { start, count },
            });
            return node_idx;
        }

        let axis = centroid_bounds.longest_axis();
        let mid_coord = centroid_bounds.center()[axis];
        let slice = &mut prim_indices[start..end];
        let mut split = 0usize;
        for i in 0..count {
            if aabbs[slice[i]].center()[axis] < mid_coord {
                slice.swap(i, split);
                split += 1;
            }
        }
        if split == 0 || split == count {
            slice.sort_by(|&i, &j| {
                aabbs[i].center()[axis]
                    .partial_cmp(&aabbs[j].center()[axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            split = count / 2;
        }

        // Push placeholder internal node to preserve preorder locality.
        nodes.push(BvhNode {
            bounds,
            kind: BvhNodeKind::Leaf { start, count },
        });
        let left = Self::build_recursive(aabbs, prim_indices, start, start + split, nodes);
        let right = Self::build_recursive(aabbs, prim_indices, start + split, end, nodes);
        nodes[node_idx].kind = BvhNodeKind::Internal { left, right };
        node_idx
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    pub fn root_bounds(&self) -> Option<Aabb> {
        self.nodes.first().map(|n| n.bounds)
    }

    /// Best-first branch-and-bound traversal visiting primitives whose AABB is within
    /// the dynamically tightened `best_dist_sq` upper bound.
    ///
    /// `eval_prim(prim_idx, &mut best_dist_sq)` should evaluate primitive `prim_idx`
    /// and update `*best_dist_sq` whenever a tighter squared distance upper bound is found.
    pub fn visit_nearest<F>(&self, point: Vec3, initial_best_dist_sq: f64, mut eval_prim: F)
    where
        F: FnMut(usize, &mut f64),
    {
        if self.nodes.is_empty() {
            return;
        }
        let mut best_d2 = initial_best_dist_sq;
        let mut stack: Vec<(usize, f64)> = Vec::with_capacity(64);
        let root_d2 = self.nodes[0].bounds.min_dist_sq(point);
        if root_d2 <= best_d2 {
            stack.push((0, root_d2));
        }

        while let Some((node_idx, node_d2)) = stack.pop() {
            if node_d2 > best_d2 {
                continue;
            }
            match self.nodes[node_idx].kind {
                BvhNodeKind::Leaf { start, count } => {
                    for &prim_idx in &self.prim_indices[start..start + count] {
                        eval_prim(prim_idx, &mut best_d2);
                    }
                }
                BvhNodeKind::Internal { left, right } => {
                    let dl = self.nodes[left].bounds.min_dist_sq(point);
                    let dr = self.nodes[right].bounds.min_dist_sq(point);
                    // Push farther child first so closer child is popped first (tightens best_d2 rapidly).
                    if dl <= dr {
                        if dr <= best_d2 {
                            stack.push((right, dr));
                        }
                        if dl <= best_d2 {
                            stack.push((left, dl));
                        }
                    } else {
                        if dl <= best_d2 {
                            stack.push((left, dl));
                        }
                        if dr <= best_d2 {
                            stack.push((right, dr));
                        }
                    }
                }
            }
        }
    }

    /// Collect all primitive indices whose AABB overlaps `query_box`.
    pub fn query_aabb(&self, query_box: Aabb, out: &mut Vec<usize>) {
        if self.nodes.is_empty() || !self.nodes[0].bounds.intersects(query_box) {
            return;
        }
        let mut stack = Vec::with_capacity(64);
        stack.push(0usize);
        while let Some(node_idx) = stack.pop() {
            match self.nodes[node_idx].kind {
                BvhNodeKind::Leaf { start, count } => {
                    out.extend_from_slice(&self.prim_indices[start..start + count]);
                }
                BvhNodeKind::Internal { left, right } => {
                    if self.nodes[left].bounds.intersects(query_box) {
                        stack.push(left);
                    }
                    if self.nodes[right].bounds.intersects(query_box) {
                        stack.push(right);
                    }
                }
            }
        }
    }
}

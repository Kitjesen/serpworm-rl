//! $O(N)$ average-case spatial hash grid broadphase for point clouds and
//! codimensional beam segments (both discrete proximity and swept CCD queries).

use fg_math::Vec3;
use fg_sdf::Aabb;
use std::collections::{HashMap, HashSet};

/// Uniform 3D spatial hash grid for fast $O(N)$ collision candidate generation.
#[derive(Clone, Debug)]
pub struct SpatialHashGrid {
    cell_size: f64,
    inv_cell_size: f64,
    buckets: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialHashGrid {
    /// Create a spatial hash grid with cell width `cell_size` (must be `> 0`).
    pub fn new(cell_size: f64) -> Self {
        let cs = cell_size.max(1e-6);
        Self {
            cell_size: cs,
            inv_cell_size: 1.0 / cs,
            buckets: HashMap::new(),
        }
    }

    #[inline]
    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    #[inline]
    fn cell_coord(&self, x: f64) -> i32 {
        (x * self.inv_cell_size).floor() as i32
    }

    #[inline]
    pub fn clear(&mut self) {
        self.buckets.clear();
    }

    /// Insert item `id` across all grid cells overlapped by `aabb`.
    pub fn insert_aabb(&mut self, id: usize, aabb: Aabb) {
        let ix0 = self.cell_coord(aabb.min.x());
        let iy0 = self.cell_coord(aabb.min.y());
        let iz0 = self.cell_coord(aabb.min.z());
        let ix1 = self.cell_coord(aabb.max.x());
        let iy1 = self.cell_coord(aabb.max.y());
        let iz1 = self.cell_coord(aabb.max.z());
        for ix in ix0..=ix1 {
            for iy in iy0..=iy1 {
                for iz in iz0..=iz1 {
                    self.buckets.entry((ix, iy, iz)).or_default().push(id);
                }
            }
        }
    }

    /// Find all candidate point pairs `(i, j)` (`i < j`, `j - i > min_index_gap`)
    /// whose Euclidean distance could be `<= query_dist`.
    pub fn find_point_pairs(
        &mut self,
        points: &[Vec3],
        query_dist: f64,
        min_index_gap: usize,
    ) -> Vec<(usize, usize)> {
        self.clear();
        let half = 0.5 * query_dist.max(0.0);
        let aabbs: Vec<Aabb> = points
            .iter()
            .map(|&p| Aabb::from_point(p).expand(half))
            .collect();
        for (i, &b) in aabbs.iter().enumerate() {
            self.insert_aabb(i, b);
        }

        let mut seen = HashSet::new();
        let mut pairs = Vec::new();
        let d2_max = query_dist * query_dist;
        for bucket in self.buckets.values() {
            let m = bucket.len();
            for a in 0..m {
                for b in (a + 1)..m {
                    let (i, j) = if bucket[a] < bucket[b] {
                        (bucket[a], bucket[b])
                    } else {
                        (bucket[b], bucket[a])
                    };
                    if j - i <= min_index_gap || !seen.insert((i, j)) {
                        continue;
                    }
                    if (points[i] - points[j]).norm_sq() <= d2_max {
                        pairs.push((i, j));
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }

    /// Find all candidate non-adjacent segment pairs `(s0, s1)` (`s0 < s1`)
    /// whose inflated AABBs overlap (`inflate = 0.5 * (d_hat + thickness)`).
    ///
    /// Segments sharing any endpoint vertex are automatically excluded.
    pub fn find_segment_pairs(
        &mut self,
        points: &[Vec3],
        segments: &[(usize, usize)],
        inflate: f64,
    ) -> Vec<(usize, usize)> {
        self.clear();
        let pad = inflate.max(0.0);
        let aabbs: Vec<Aabb> = segments
            .iter()
            .map(|&(u, v)| Aabb::from_segment(points[u], points[v], pad))
            .collect();
        for (s, &b) in aabbs.iter().enumerate() {
            self.insert_aabb(s, b);
        }
        self.collect_segment_pairs(segments, &aabbs)
    }

    /// Find all candidate non-adjacent segment pairs `(s0, s1)` whose **swept motion**
    /// AABBs from `x0` to `x1` overlap (used by conservative ACCD step clamping).
    pub fn find_swept_segment_pairs(
        &mut self,
        x0: &[Vec3],
        x1: &[Vec3],
        segments: &[(usize, usize)],
        inflate: f64,
    ) -> Vec<(usize, usize)> {
        self.clear();
        let pad = inflate.max(0.0);
        let aabbs: Vec<Aabb> = segments
            .iter()
            .map(|&(u, v)| {
                let b0 = Aabb::from_segment(x0[u], x0[v], pad);
                let b1 = Aabb::from_segment(x1[u], x1[v], pad);
                b0.union(b1)
            })
            .collect();
        for (s, &b) in aabbs.iter().enumerate() {
            self.insert_aabb(s, b);
        }
        self.collect_segment_pairs(segments, &aabbs)
    }

    fn collect_segment_pairs(
        &self,
        segments: &[(usize, usize)],
        aabbs: &[Aabb],
    ) -> Vec<(usize, usize)> {
        let mut seen = HashSet::new();
        let mut pairs = Vec::new();
        for bucket in self.buckets.values() {
            let m = bucket.len();
            for a in 0..m {
                for b in (a + 1)..m {
                    let (i, j) = if bucket[a] < bucket[b] {
                        (bucket[a], bucket[b])
                    } else {
                        (bucket[b], bucket[a])
                    };
                    let (u0, v0) = segments[i];
                    let (u1, v1) = segments[j];
                    // Skip incident segments sharing a node.
                    if u0 == u1 || u0 == v1 || v0 == u1 || v0 == v1 {
                        continue;
                    }
                    if !seen.insert((i, j)) {
                        continue;
                    }
                    if aabbs[i].intersects(aabbs[j]) {
                        pairs.push((i, j));
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }
}

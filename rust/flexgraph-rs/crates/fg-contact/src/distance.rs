//! Exact codimensional primitive distances (`Point-Point`, `Point-Segment`, `Segment-Segment`)
//! with closed-form analytical gradients, closed-form analytical Hessians (`6x6`, `9x9`, `12x12`),
//! and IPC edge-edge parallel mollifier $m(\|e_0 \times e_1\|^2)$.

use fg_math::{Mat3, Vec3};

/// Exact Point-Point Euclidean distance `d = ||xa - xb||` with 6D gradient and 6x6 Hessian.
#[derive(Copy, Clone, Debug)]
pub struct PointPointDist {
    pub dist: f64,
    pub normal: Vec3,
    pub grad: [f64; 6],
    pub hess: [[f64; 6]; 6],
}

/// Evaluate Point-Point distance `d(xa, xb)` and its exact 6D gradient & 6x6 Hessian
/// ordered as `[xa (0..3), xb (3..6)]`.
pub fn point_point_distance(xa: Vec3, xb: Vec3) -> PointPointDist {
    let r = xa - xb;
    let d2 = r.norm_sq();
    let dist = d2.sqrt();
    if dist < 1e-15 {
        return PointPointDist {
            dist: 0.0,
            normal: Vec3::EZ,
            grad: [0.0; 6],
            hess: [[0.0; 6]; 6],
        };
    }
    let n = r / dist;
    let grad = [n.x(), n.y(), n.z(), -n.x(), -n.y(), -n.z()];
    let m = (Mat3::IDENTITY - n.outer(n)) * (1.0 / dist);
    let mut hess = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            let v = m.0[i][j];
            hess[i][j] = v;
            hess[i][j + 3] = -v;
            hess[i + 3][j] = -v;
            hess[i + 3][j + 3] = v;
        }
    }
    PointPointDist {
        dist,
        normal: n,
        grad,
        hess,
    }
}

/// Exact Point-Segment distance `d(p, (a, b))` with 9D gradient and 9x9 Hessian.
#[derive(Copy, Clone, Debug)]
pub struct PointSegmentDist {
    pub dist: f64,
    pub t: f64,
    pub normal: Vec3,
    pub grad: [f64; 9],
    pub hess: [[f64; 9]; 9],
}

/// Evaluate Point-Segment distance `d(p, (a, b))` and its exact 9D gradient & 9x9 Hessian
/// ordered as `[p (0..3), a (3..6), b (6..9)]`.
///
/// In the interior Voronoi region `0 < t < 1`, uses the exact implicit-function identity:
/// `H_d = (1/d) W^T (I - n n^T) W - (L^2 / d) (grad_z t)(grad_z t)^T`.
pub fn point_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> PointSegmentDist {
    let e = b - a;
    let l2 = e.norm_sq();
    if l2 < 1e-24 {
        let pp = point_point_distance(p, a);
        let mut grad = [0.0; 9];
        let mut hess = [[0.0; 9]; 9];
        grad[0..6].copy_from_slice(&pp.grad);
        for i in 0..6 {
            for j in 0..6 {
                hess[i][j] = pp.hess[i][j];
            }
        }
        return PointSegmentDist {
            dist: pp.dist,
            t: 0.0,
            normal: pp.normal,
            grad,
            hess,
        };
    }

    let t_raw = (p - a).dot(e) / l2;
    if t_raw <= 0.0 {
        let pp = point_point_distance(p, a);
        let mut grad = [0.0; 9];
        let mut hess = [[0.0; 9]; 9];
        grad[0..6].copy_from_slice(&pp.grad);
        for i in 0..6 {
            for j in 0..6 {
                hess[i][j] = pp.hess[i][j];
            }
        }
        return PointSegmentDist {
            dist: pp.dist,
            t: 0.0,
            normal: pp.normal,
            grad,
            hess,
        };
    }
    if t_raw >= 1.0 {
        let pp = point_point_distance(p, b);
        let mut grad = [0.0; 9];
        let mut hess = [[0.0; 9]; 9];
        // Map (0->p[0..3], 1->b[6..9])
        grad[0..3].copy_from_slice(&pp.grad[0..3]);
        grad[6..9].copy_from_slice(&pp.grad[3..6]);
        let map = [0usize, 1, 2, 6, 7, 8];
        for i in 0..6 {
            for j in 0..6 {
                hess[map[i]][map[j]] = pp.hess[i][j];
            }
        }
        return PointSegmentDist {
            dist: pp.dist,
            t: 1.0,
            normal: pp.normal,
            grad,
            hess,
        };
    }

    // Strictly interior Voronoi region: 0 < t < 1, r = p - ((1-t)a + t b) perp e.
    let t = t_raw;
    let om_t = 1.0 - t;
    let q = a * om_t + b * t;
    let r = p - q;
    let dist = r.norm();
    if dist < 1e-15 {
        return PointSegmentDist {
            dist: 0.0,
            t,
            normal: Vec3::EZ,
            grad: [0.0; 9],
            hess: [[0.0; 9]; 9],
        };
    }
    let n = r / dist;
    let w = [1.0, -om_t, -t];
    let mut grad = [0.0; 9];
    for blk in 0..3 {
        let wb = w[blk];
        grad[blk * 3] = wb * n.x();
        grad[blk * 3 + 1] = wb * n.y();
        grad[blk * 3 + 2] = wb * n.z();
    }

    // grad_z t = [e, -(1-t)e - r, -t e + r] / L^2
    let inv_l2 = 1.0 / l2;
    let gt_p = e * inv_l2;
    let gt_a = (-e * om_t - r) * inv_l2;
    let gt_b = (-e * t + r) * inv_l2;
    let gt = [
        gt_p.x(),
        gt_p.y(),
        gt_p.z(),
        gt_a.x(),
        gt_a.y(),
        gt_a.z(),
        gt_b.x(),
        gt_b.y(),
        gt_b.z(),
    ];

    let inv_d = 1.0 / dist;
    let p_perp = (Mat3::IDENTITY - n.outer(n)) * inv_d;
    let l2_over_d = l2 * inv_d;

    let mut hess = [[0.0; 9]; 9];
    for bi in 0..3 {
        for bj in 0..3 {
            let ww = w[bi] * w[bj];
            for i in 0..3 {
                let ri = bi * 3 + i;
                for j in 0..3 {
                    let cj = bj * 3 + j;
                    hess[ri][cj] = ww * p_perp.0[i][j] - l2_over_d * gt[ri] * gt[cj];
                }
            }
        }
    }

    PointSegmentDist {
        dist,
        t,
        normal: n,
        grad,
        hess,
    }
}

/// Active Voronoi feature classification for a Segment-Segment pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SegmentFeature {
    PointPoint(usize, usize),
    PointSegment,
    EdgeEdgeInterior,
}

/// Exact Segment-Segment (Edge-Edge) distance `d((a0, a1), (b0, b1))` with 12D gradient
/// and exact 12x12 analytical Hessian.
#[derive(Copy, Clone, Debug)]
pub struct SegmentSegmentDist {
    pub dist: f64,
    pub s: f64,
    pub t: f64,
    pub normal: Vec3,
    pub feature: SegmentFeature,
    pub grad: [f64; 12],
    pub hess: [[f64; 12]; 12],
}

/// Fast squared distance between two 3D line segments `(a0, a1)` and `(b0, b1)`
/// returning `(dist_sq, s, t)` (Ericson / Lumelsky exact 9-region projection).
pub fn segment_segment_dist_sq(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> (f64, f64, f64) {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let (s, t) = if a <= 1e-24 && e <= 1e-24 {
        (0.0, 0.0)
    } else if a <= 1e-24 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= 1e-24 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s = if denom > 1e-24 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    let c1 = a0 + d1 * s;
    let c2 = b0 + d2 * t;
    ((c1 - c2).norm_sq(), s, t)
}

/// Evaluate Segment-Segment distance `d((a0, a1), (b0, b1))` and its exact 12D gradient
/// and 12x12 analytical Hessian ordered as `[a0 (0..3), a1 (3..6), b0 (6..9), b1 (9..12)]`.
pub fn segment_segment_distance(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> SegmentSegmentDist {
    let e0 = a1 - a0;
    let e1 = b1 - b0;
    let (d2, s, t) = segment_segment_dist_sq(a0, a1, b0, b1);

    // Check if closest points lie strictly inside both segments and edges are non-parallel.
    // Use ||e0 x e1||^2 (Lagrange identity) instead of g00*g11 - g01^2 to avoid FP64 cancellation.
    let g00 = e0.dot(e0);
    let g11 = e1.dot(e1);
    let g01 = e0.dot(e1);
    let det = e0.cross(e1).norm_sq();
    let eps_param = 1e-12;
    let interior_s = s > eps_param && s < 1.0 - eps_param;
    let interior_t = t > eps_param && t < 1.0 - eps_param;

    if interior_s && interior_t && det > 1e-20 * g00 * g11 {
        let om_s = 1.0 - s;
        let om_t = 1.0 - t;
        let pa = a0 * om_s + a1 * s;
        let pb = b0 * om_t + b1 * t;
        let r = pa - pb;
        let dist = r.norm();
        if dist < 1e-15 {
            return SegmentSegmentDist {
                dist: 0.0,
                s,
                t,
                normal: Vec3::EZ,
                feature: SegmentFeature::EdgeEdgeInterior,
                grad: [0.0; 12],
                hess: [[0.0; 12]; 12],
            };
        }
        let n = r / dist;
        let w = [om_s, s, -om_t, -t];
        let mut grad = [0.0; 12];
        for blk in 0..4 {
            let wb = w[blk];
            grad[blk * 3] = wb * n.x();
            grad[blk * 3 + 1] = wb * n.y();
            grad[blk * 3 + 2] = wb * n.z();
        }

        // Exact implicit-function Hessian:
        // H_d = (1/d) W^T (I - n n^T) W - (1/d) [u0 u1] G^{-1} [u0 u1]^T
        // where G = [[e0.e0, -e0.e1], [-e0.e1, e1.e1]],
        // u0 = W^T e0 + [-r, +r, 0, 0], u1 = -W^T e1 + [0, 0, +r, -r].
        let mut u0 = [0.0; 12];
        let mut u1 = [0.0; 12];
        for blk in 0..4 {
            let wb = w[blk];
            for c in 0..3 {
                u0[blk * 3 + c] = wb * e0[c];
                u1[blk * 3 + c] = -wb * e1[c];
            }
        }
        for c in 0..3 {
            u0[c] -= r[c];
            u0[3 + c] += r[c];
            u1[6 + c] += r[c];
            u1[9 + c] -= r[c];
        }

        // G^{-1} = (1 / det) [[g11, g01], [g01, g00]]
        let inv_det = 1.0 / det;
        let ig00 = g11 * inv_det;
        let ig01 = g01 * inv_det;
        let ig11 = g00 * inv_det;

        let inv_d = 1.0 / dist;
        let p_perp = (Mat3::IDENTITY - n.outer(n)) * inv_d;

        let mut hess = [[0.0; 12]; 12];
        for bi in 0..4 {
            for bj in 0..4 {
                let ww = w[bi] * w[bj];
                for i in 0..3 {
                    let ri = bi * 3 + i;
                    for j in 0..3 {
                        let cj = bj * 3 + j;
                        let g_quad = u0[ri] * (ig00 * u0[cj] + ig01 * u1[cj])
                            + u1[ri] * (ig01 * u0[cj] + ig11 * u1[cj]);
                        hess[ri][cj] = ww * p_perp.0[i][j] - inv_d * g_quad;
                    }
                }
            }
        }

        return SegmentSegmentDist {
            dist,
            s,
            t,
            normal: n,
            feature: SegmentFeature::EdgeEdgeInterior,
            grad,
            hess,
        };
    }

    // Boundary Voronoi regions: delegate to exact Point-Segment or Point-Point and scatter to 12 DOF.
    if interior_t {
        // Endpoint of segment A (a0 if s<=0.5 else a1) against segment B (b0, b1)
        let idx_a = if s <= 0.5 { 0 } else { 1 };
        let pa = if idx_a == 0 { a0 } else { a1 };
        let ps = point_segment_distance(pa, b0, b1);
        let map = [
            idx_a * 3,
            idx_a * 3 + 1,
            idx_a * 3 + 2,
            6,
            7,
            8,
            9,
            10,
            11,
        ];
        return scatter_ps_to_12(ps, idx_a as f64, ps.t, ps.normal, map);
    }

    if interior_s {
        // Endpoint of segment B (b0 if t<=0.5 else b1) against segment A (a0, a1)
        // Note: point_segment_distance computes normal = (pb - qa)/d = -(qa - pb)/d, so flip sign for A->B normal!
        let idx_b = if t <= 0.5 { 0 } else { 1 };
        let pb = if idx_b == 0 { b0 } else { b1 };
        let ps = point_segment_distance(pb, a0, a1);
        let map = [
            6 + idx_b * 3,
            6 + idx_b * 3 + 1,
            6 + idx_b * 3 + 2,
            0,
            1,
            2,
            3,
            4,
            5,
        ];
        return scatter_ps_to_12(ps, ps.t, idx_b as f64, -ps.normal, map);
    }

    // Endpoint-Endpoint (PP)
    let _ = d2;
    let idx_a = if s <= 0.5 { 0 } else { 1 };
    let idx_b = if t <= 0.5 { 0 } else { 1 };
    let pa = if idx_a == 0 { a0 } else { a1 };
    let pb = if idx_b == 0 { b0 } else { b1 };
    let pp = point_point_distance(pa, pb);
    let map = [
        idx_a * 3,
        idx_a * 3 + 1,
        idx_a * 3 + 2,
        6 + idx_b * 3,
        6 + idx_b * 3 + 1,
        6 + idx_b * 3 + 2,
    ];
    let mut grad = [0.0; 12];
    let mut hess = [[0.0; 12]; 12];
    for i in 0..6 {
        grad[map[i]] = pp.grad[i];
        for j in 0..6 {
            hess[map[i]][map[j]] = pp.hess[i][j];
        }
    }
    SegmentSegmentDist {
        dist: pp.dist,
        s: idx_a as f64,
        t: idx_b as f64,
        normal: pp.normal,
        feature: SegmentFeature::PointPoint(idx_a, idx_b),
        grad,
        hess,
    }
}

fn scatter_ps_to_12(
    ps: PointSegmentDist,
    s: f64,
    t: f64,
    normal_a_from_b: Vec3,
    map: [usize; 9],
) -> SegmentSegmentDist {
    let mut grad = [0.0; 12];
    let mut hess = [[0.0; 12]; 12];
    for i in 0..9 {
        grad[map[i]] = ps.grad[i];
        for j in 0..9 {
            hess[map[i]][map[j]] = ps.hess[i][j];
        }
    }
    SegmentSegmentDist {
        dist: ps.dist,
        s,
        t,
        normal: normal_a_from_b,
        feature: SegmentFeature::PointSegment,
        grad,
        hess,
    }
}

/// Exact IPC edge-edge parallel mollifier evaluation (Li et al. SIGGRAPH 2020, Eq. 11–12).
#[derive(Copy, Clone, Debug)]
pub struct EdgeEdgeMollifierEval {
    pub value: f64,
    pub grad: [f64; 12],
    pub hess: [[f64; 12]; 12],
}

/// Evaluate IPC parallel-edge mollifier `m(c) = -(c/eps_x)^2 + 2(c/eps_x)` for `c = ||e0 x e1||^2 < eps_x`,
/// and `1.0` for `c >= eps_x`, along with exact 12D gradient and 12x12 analytical Hessian.
pub fn edge_edge_mollifier(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3, eps_x: f64) -> EdgeEdgeMollifierEval {
    if eps_x <= 0.0 {
        return EdgeEdgeMollifierEval {
            value: 1.0,
            grad: [0.0; 12],
            hess: [[0.0; 12]; 12],
        };
    }
    let e0 = a1 - a0;
    let e1 = b1 - b0;
    let v = e0.cross(e1);
    let c = v.norm_sq();
    if c >= eps_x {
        return EdgeEdgeMollifierEval {
            value: 1.0,
            grad: [0.0; 12],
            hess: [[0.0; 12]; 12],
        };
    }

    let r = c / eps_x;
    let val = -r * r + 2.0 * r;
    let dm_dc = (2.0 - 2.0 * r) / eps_x;
    let d2m_dc2 = -2.0 / (eps_x * eps_x);

    // Exact derivatives of c = ||e0 x e1||^2 w.r.t. (e0, e1):
    // dc/de0 = 2 (e1 x v), dc/de1 = 2 (v x e0).
    let gc_e0 = 2.0 * e1.cross(v);
    let gc_e1 = 2.0 * v.cross(e0);
    let gc = [
        -gc_e0.x(),
        -gc_e0.y(),
        -gc_e0.z(),
        gc_e0.x(),
        gc_e0.y(),
        gc_e0.z(),
        -gc_e1.x(),
        -gc_e1.y(),
        -gc_e1.z(),
        gc_e1.x(),
        gc_e1.y(),
        gc_e1.z(),
    ];

    let mut grad = [0.0; 12];
    for i in 0..12 {
        grad[i] = dm_dc * gc[i];
    }

    // 6x6 Hessian of c w.r.t. [e0, e1]:
    // H00 = 2 (||e1||^2 I - e1 e1^T)
    // H11 = 2 (||e0||^2 I - e0 e0^T)
    // H01 = 2 (-[v]_x + [e1]_x [e0]_x)
    let h00 = (Mat3::IDENTITY * e1.norm_sq() - e1.outer(e1)) * 2.0;
    let h11 = (Mat3::IDENTITY * e0.norm_sq() - e0.outer(e0)) * 2.0;
    let h01 = (-v.skew() + e1.skew().mul_mat(e0.skew())) * 2.0;
    let h10 = h01.transpose();

    // Chain rule from [e0, e1] to [a0, a1, b0, b1]: e0 = a1 - a0, e1 = b1 - b0.
    let signs = [-1.0, 1.0, -1.0, 1.0];
    let edge_id = [0usize, 0, 1, 1];
    let blocks = [[h00, h01], [h10, h11]];

    let mut hess = [[0.0; 12]; 12];
    for bi in 0..4 {
        for bj in 0..4 {
            let s_ij = signs[bi] * signs[bj];
            let blk = blocks[edge_id[bi]][edge_id[bj]];
            for i in 0..3 {
                let ri = bi * 3 + i;
                for j in 0..3 {
                    let cj = bj * 3 + j;
                    let hc = s_ij * blk.0[i][j];
                    hess[ri][cj] = d2m_dc2 * gc[ri] * gc[cj] + dm_dc * hc;
                }
            }
        }
    }

    EdgeEdgeMollifierEval {
        value: val,
        grad,
        hess,
    }
}

/// Active Voronoi feature classification for a Point-Triangle (Vertex-Face) pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TriangleFeature {
    /// Closest point is triangle vertex `v in {0, 1, 2}` (`t0`, `t1`, or `t2`).
    Vertex(usize),
    /// Closest point lies strictly inside triangle edge `(i, j)`.
    Edge(usize, usize),
    /// Closest point lies strictly inside the triangle face.
    FaceInterior,
}

/// Exact Point-Triangle (Vertex-Face) distance `d(p, (t0, t1, t2))` with Voronoi classification,
/// barycentric coordinates, exact 12D gradient, and exact 12x12 analytical Hessian.
#[derive(Copy, Clone, Debug)]
pub struct PointTriangleDist {
    pub dist: f64,
    /// Barycentric weights `[alpha, beta, gamma]` of closest point `q = alpha*t0 + beta*t1 + gamma*t2`.
    pub bary: [f64; 3],
    /// Unit normal `(p - q) / ||p - q||` pointing from triangle toward `p`.
    pub normal: Vec3,
    pub feature: TriangleFeature,
    /// Exact 12D gradient ordered `[p (0..3), t0 (3..6), t1 (6..9), t2 (9..12)]`.
    pub grad: [f64; 12],
    /// Exact 12x12 analytical Hessian ordered `[p, t0, t1, t2]`.
    pub hess: [[f64; 12]; 12],
}

/// Evaluate exact Point-Triangle distance `d(p, (t0, t1, t2))`, Voronoi region, 12D gradient,
/// and 12x12 analytical Hessian ordered as `[p (0..3), t0 (3..6), t1 (6..9), t2 (9..12)]`.
///
/// Uses Ericson's exact 7-region Voronoi classification:
/// - Vertex regions delegate to [`point_point_distance`];
/// - Edge regions delegate to [`point_segment_distance`];
/// - Strictly interior face region uses the exact implicit-function Hessian:
///   `H_d = (1/d) W^T (I - n n^T) W - (1/d) [u1 u2] G^{-1} [u1 u2]^T`,
///   with `det(G) = ||e1 x e2||^2` (Lagrange identity, zero FP64 cancellation).
pub fn point_triangle_distance_grad_hess(
    p: Vec3,
    t0: Vec3,
    t1: Vec3,
    t2: Vec3,
) -> PointTriangleDist {
    let ab = t1 - t0;
    let ac = t2 - t0;
    let ap = p - t0;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return scatter_pt_vertex(p, t0, 0);
    }

    let bp = p - t1;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return scatter_pt_vertex(p, t1, 1);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return scatter_pt_edge(p, t0, t1, 0, 1);
    }

    let cp = p - t2;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return scatter_pt_vertex(p, t2, 2);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return scatter_pt_edge(p, t0, t2, 0, 2);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return scatter_pt_edge(p, t1, t2, 1, 2);
    }

    // Check triangle non-degeneracy using Lagrange identity det(G) = ||ab x ac||^2.
    let g11 = ab.dot(ab);
    let g12 = ab.dot(ac);
    let g22 = ac.dot(ac);
    let det = ab.cross(ac).norm_sq();
    let denom = va + vb + vc;
    if denom <= 1e-24 || det <= 1e-24 * g11 * g22 {
        // Degenerate sliver triangle: pick closest among the three edges.
        let e01 = scatter_pt_edge(p, t0, t1, 0, 1);
        let e12 = scatter_pt_edge(p, t1, t2, 1, 2);
        let e20 = scatter_pt_edge(p, t2, t0, 2, 0);
        let best = if e01.dist <= e12.dist { e01 } else { e12 };
        return if best.dist <= e20.dist { best } else { e20 };
    }

    let inv_denom = 1.0 / denom;
    let u = vb * inv_denom;
    let v = vc * inv_denom;
    let alpha = 1.0 - u - v;

    let q = t0 * alpha + t1 * u + t2 * v;
    let r = p - q;
    let dist = r.norm();
    if dist < 1e-15 {
        let face_n = ab.cross(ac).normalized_or(Vec3::EZ);
        return PointTriangleDist {
            dist: 0.0,
            bary: [alpha, u, v],
            normal: face_n,
            feature: TriangleFeature::FaceInterior,
            grad: [0.0; 12],
            hess: [[0.0; 12]; 12],
        };
    }

    let n = r / dist;
    let w = [1.0, -alpha, -u, -v];
    let mut grad = [0.0; 12];
    for blk in 0..4 {
        let wb = w[blk];
        grad[blk * 3] = wb * n.x();
        grad[blk * 3 + 1] = wb * n.y();
        grad[blk * 3 + 2] = wb * n.z();
    }

    // Exact implicit-function Hessian:
    // u1 = W^T e1 + [0, -r, +r, 0], u2 = W^T e2 + [0, -r, 0, +r],
    // G = [[e1.e1, e1.e2], [e1.e2, e2.e2]], G^{-1} = (1/det) [[g22, -g12], [-g12, g11]],
    // H_d = (1/d) W^T (I - n n^T) W - (1/d) [u1 u2] G^{-1} [u1 u2]^T.
    let mut u1 = [0.0; 12];
    let mut u2 = [0.0; 12];
    for blk in 0..4 {
        let wb = w[blk];
        for c in 0..3 {
            u1[blk * 3 + c] = wb * ab[c];
            u2[blk * 3 + c] = wb * ac[c];
        }
    }
    for c in 0..3 {
        u1[3 + c] -= r[c];
        u1[6 + c] += r[c];
        u2[3 + c] -= r[c];
        u2[9 + c] += r[c];
    }

    let inv_det = 1.0 / det;
    let ig11 = g22 * inv_det;
    let ig12 = -g12 * inv_det;
    let ig22 = g11 * inv_det;

    let inv_d = 1.0 / dist;
    let p_perp = (Mat3::IDENTITY - n.outer(n)) * inv_d;

    let mut hess = [[0.0; 12]; 12];
    for bi in 0..4 {
        for bj in 0..4 {
            let ww = w[bi] * w[bj];
            for i in 0..3 {
                let ri = bi * 3 + i;
                for j in 0..3 {
                    let cj = bj * 3 + j;
                    let g_quad = u1[ri] * (ig11 * u1[cj] + ig12 * u2[cj])
                        + u2[ri] * (ig12 * u1[cj] + ig22 * u2[cj]);
                    hess[ri][cj] = ww * p_perp.0[i][j] - inv_d * g_quad;
                }
            }
        }
    }

    PointTriangleDist {
        dist,
        bary: [alpha, u, v],
        normal: n,
        feature: TriangleFeature::FaceInterior,
        grad,
        hess,
    }
}

/// Alias for [`point_triangle_distance_grad_hess`] matching `point_*_distance` naming.
#[inline]
pub fn point_triangle_distance(p: Vec3, t0: Vec3, t1: Vec3, t2: Vec3) -> PointTriangleDist {
    point_triangle_distance_grad_hess(p, t0, t1, t2)
}

fn scatter_pt_vertex(p: Vec3, tv: Vec3, v_idx: usize) -> PointTriangleDist {
    let pp = point_point_distance(p, tv);
    let base = 3 + 3 * v_idx;
    let map = [0usize, 1, 2, base, base + 1, base + 2];
    let mut grad = [0.0; 12];
    let mut hess = [[0.0; 12]; 12];
    for i in 0..6 {
        grad[map[i]] = pp.grad[i];
        for j in 0..6 {
            hess[map[i]][map[j]] = pp.hess[i][j];
        }
    }
    let mut bary = [0.0; 3];
    bary[v_idx] = 1.0;
    PointTriangleDist {
        dist: pp.dist,
        bary,
        normal: pp.normal,
        feature: TriangleFeature::Vertex(v_idx),
        grad,
        hess,
    }
}

fn scatter_pt_edge(p: Vec3, va: Vec3, vb: Vec3, idx_a: usize, idx_b: usize) -> PointTriangleDist {
    let ps = point_segment_distance(p, va, vb);
    let ba = 3 + 3 * idx_a;
    let bb = 3 + 3 * idx_b;
    let map = [0usize, 1, 2, ba, ba + 1, ba + 2, bb, bb + 1, bb + 2];
    let mut grad = [0.0; 12];
    let mut hess = [[0.0; 12]; 12];
    for i in 0..9 {
        grad[map[i]] = ps.grad[i];
        for j in 0..9 {
            hess[map[i]][map[j]] = ps.hess[i][j];
        }
    }
    let mut bary = [0.0; 3];
    bary[idx_a] = 1.0 - ps.t;
    bary[idx_b] = ps.t;
    let feature = if ps.t <= 0.0 {
        TriangleFeature::Vertex(idx_a)
    } else if ps.t >= 1.0 {
        TriangleFeature::Vertex(idx_b)
    } else {
        TriangleFeature::Edge(idx_a, idx_b)
    };
    PointTriangleDist {
        dist: ps.dist,
        bary,
        normal: ps.normal,
        feature,
        grad,
        hess,
    }
}

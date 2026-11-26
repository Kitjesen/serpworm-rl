//! `parry3d-f64` SIMD QBvh-accelerated triangle mesh (`TriMeshSdf`), convex hull
//! (`ConvexHullSdf`), minimal Wavefront `.obj` parser, procedural corrugated-pipe /
//! rubble-terrain mesh builders, and swept shape-casting (`ParryShapeCast`).

use crate::bvh::Aabb;
use crate::{SdfField, SdfQuery};
use fg_math::{Se3, Vec3};
use parry3d_f64::math::{Pose, Rotation, Vector};
use parry3d_f64::query::{
    cast_shapes, PointQuery, PointQueryWithLocation, ShapeCastOptions, ShapeCastStatus,
};
use parry3d_f64::shape::{Ball, ConvexPolyhedron, Shape, TriMesh, TrianglePointLocation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

#[inline]
fn to_pvec(v: Vec3) -> Vector {
    Vector::new(v.x(), v.y(), v.z())
}

#[inline]
fn from_pvec(v: Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

#[inline]
fn to_ppose(tf: Se3) -> Pose {
    let q = tf.rot.normalized();
    Pose::from_parts(
        to_pvec(tf.pos),
        Rotation::from_xyzw(q.0[0], q.0[1], q.0[2], q.0[3]),
    )
}

/// Sign convention used by [`TriMeshSdf`] to classify free space (`phi > 0`) vs solid (`phi < 0`).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum MeshSignMode {
    /// Watertight oriented closed solid mesh (outward triangle normals point into free space).
    /// Uses exact Bærentzen–Aanæs (TVCG 2005) angle-weighted pseudonormals for robust sign classification.
    ClosedSolid,
    /// Open oriented surface or interior cavity (e.g. `rubble_terrain_mesh` or `corrugated_pipe_mesh`)
    /// whose triangle normals point into navigable free space (`bounds()` returns `None` so `WorldSdf`
    /// never prunes deep penetrations).
    OpenSurface,
    /// Two-sided thin shell of uniform `half_thickness`: `phi = ||p - q|| - half_thickness`.
    UnsignedShell { half_thickness: f64 },
}

/// Centerline polyline used by pipe/lumen meshes to report continuous `arc_s` and `tangent`.
#[derive(Clone, Debug)]
struct MeshCenterline {
    points: Vec<Vec3>,
    cum_s: Vec<f64>,
}

impl MeshCenterline {
    fn new(points: Vec<Vec3>) -> Option<Self> {
        if points.len() < 2 {
            return None;
        }
        let mut cum_s = Vec::with_capacity(points.len());
        cum_s.push(0.0);
        let mut s = 0.0;
        for w in points.windows(2) {
            s += (w[1] - w[0]).norm();
            cum_s.push(s);
        }
        Some(Self { points, cum_s })
    }

    fn project(&self, p_local: Vec3) -> (f64, Vec3) {
        let mut best_d2 = f64::INFINITY;
        let mut best_s = 0.0;
        let mut best_t = Vec3::EX;
        for i in 0..(self.points.len() - 1) {
            let a = self.points[i];
            let b = self.points[i + 1];
            let ab = b - a;
            let len2 = ab.norm_sq();
            let (t, len, dir) = if len2 > 1e-24 {
                let len = len2.sqrt();
                (((p_local - a).dot(ab) / len2).clamp(0.0, 1.0), len, ab / len)
            } else {
                (0.0, 0.0, Vec3::EX)
            };
            let q = a + ab * t;
            let d2 = (p_local - q).norm_sq();
            if d2 < best_d2 {
                best_d2 = d2;
                best_s = self.cum_s[i] + t * len;
                best_t = dir;
            }
        }
        (best_s, best_t)
    }
}

/// SIMD QBvh-accelerated arbitrary 3D triangle-mesh SDF backed by `parry3d_f64::shape::TriMesh`,
/// with exact angle-weighted pseudonormal sign classification (Bærentzen & Aanæs, TVCG 2005).
#[derive(Clone)]
pub struct TriMeshSdf {
    vertices: Vec<Vec3>,
    indices: Vec<[u32; 3]>,
    mesh: TriMesh,
    pose: Se3,
    sign_mode: MeshSignMode,
    mat_id: usize,
    face_normals: Vec<Vec3>,
    vertex_pseudonormals: Vec<Vec3>,
    edge_pseudonormals: HashMap<(u32, u32), Vec3>,
    local_aabb: Aabb,
    centerline: Option<MeshCenterline>,
}

impl TriMeshSdf {
    /// Construct a `TriMeshSdf` from local-space vertices and oriented triangles.
    ///
    /// Triangle winding `(v0, v1, v2)` defines the outward free-space normal via `(v1 - v0) x (v2 - v0)`.
    pub fn new(
        vertices: Vec<Vec3>,
        indices: Vec<[u32; 3]>,
        pose: Se3,
        sign_mode: MeshSignMode,
        mat_id: usize,
    ) -> Result<Self, String> {
        if indices.is_empty() {
            return Err("TriMeshSdf requires at least one triangle".to_string());
        }
        let n_verts = vertices.len();
        for (ti, &[a, b, c]) in indices.iter().enumerate() {
            if (a as usize) >= n_verts || (b as usize) >= n_verts || (c as usize) >= n_verts {
                return Err(format!(
                    "triangle {ti} index [{a}, {b}, {c}] out of bounds (n_verts={n_verts})"
                ));
            }
        }

        let p_verts: Vec<Vector> = vertices.iter().map(|&v| to_pvec(v)).collect();
        let mesh = TriMesh::new(p_verts, indices.clone())
            .map_err(|e| format!("parry3d TriMesh build failed: {e}"))?;

        let mut local_aabb = Aabb::EMPTY;
        for &v in &vertices {
            local_aabb = local_aabb.union(Aabb::from_point(v));
        }
        if let MeshSignMode::UnsignedShell { half_thickness } = sign_mode {
            local_aabb = local_aabb.expand(half_thickness.max(0.0));
        }

        // Precompute face normals, angle-weighted vertex pseudonormals, and edge pseudonormals.
        let mut face_normals = Vec::with_capacity(indices.len());
        let mut vert_acc = vec![Vec3::ZERO; n_verts];
        let mut edge_acc: HashMap<(u32, u32), Vec3> = HashMap::with_capacity(indices.len() * 3);

        for &[i0, i1, i2] in &indices {
            let p0 = vertices[i0 as usize];
            let p1 = vertices[i1 as usize];
            let p2 = vertices[i2 as usize];
            let e01 = p1 - p0;
            let e02 = p2 - p0;
            let n_raw = e01.cross(e02);
            let n = n_raw.normalized_or(Vec3::EZ);
            face_normals.push(n);

            // Interior angles at vertices 0, 1, 2 for exact Bærentzen-Aanæs angle weighting
            let a0 = angle_between(p1 - p0, p2 - p0);
            let a1 = angle_between(p2 - p1, p0 - p1);
            let a2 = angle_between(p0 - p2, p1 - p2);
            vert_acc[i0 as usize] += n * a0;
            vert_acc[i1 as usize] += n * a1;
            vert_acc[i2 as usize] += n * a2;

            for (u, v) in [(i0, i1), (i1, i2), (i2, i0)] {
                let key = if u < v { (u, v) } else { (v, u) };
                *edge_acc.entry(key).or_insert(Vec3::ZERO) += n;
            }
        }

        let vertex_pseudonormals: Vec<Vec3> = vert_acc
            .into_iter()
            .map(|v| v.normalized_or(Vec3::EZ))
            .collect();
        let edge_pseudonormals: HashMap<(u32, u32), Vec3> = edge_acc
            .into_iter()
            .map(|(k, v)| (k, v.normalized_or(Vec3::EZ)))
            .collect();

        Ok(Self {
            vertices,
            indices,
            mesh,
            pose,
            sign_mode,
            mat_id,
            face_normals,
            vertex_pseudonormals,
            edge_pseudonormals,
            local_aabb,
            centerline: None,
        })
    }

    /// Attach a local-space centerline polyline so `query()` reports continuous `arc_s` and `tangent`.
    pub fn with_centerline(mut self, centerline_pts: Vec<Vec3>) -> Self {
        self.centerline = MeshCenterline::new(centerline_pts);
        self
    }

    /// Minimal Wavefront `.obj` parser supporting `v x y z`, `f a b c ...` (with optional `/vt/vn`,
    /// negative relative indices, and fan triangulation of quads/n-gons).
    pub fn from_obj_str(
        obj: &str,
        scale: Vec3,
        pose: Se3,
        sign_mode: MeshSignMode,
        mat_id: usize,
    ) -> Result<Self, String> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for (line_no, raw_line) in obj.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(head) = parts.next() else {
                continue;
            };
            match head {
                "v" => {
                    let coords: Vec<f64> = parts
                        .take(3)
                        .map(|tok| {
                            tok.parse::<f64>().map_err(|e| {
                                format!("OBJ line {}: invalid vertex float '{tok}': {e}", line_no + 1)
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    if coords.len() < 3 {
                        return Err(format!(
                            "OBJ line {}: expected 3 vertex coordinates",
                            line_no + 1
                        ));
                    }
                    vertices.push(Vec3::new(
                        coords[0] * scale.x(),
                        coords[1] * scale.y(),
                        coords[2] * scale.z(),
                    ));
                }
                "f" => {
                    let mut poly = Vec::new();
                    for tok in parts {
                        let v_str = tok.split('/').next().unwrap_or("");
                        if v_str.is_empty() {
                            continue;
                        }
                        let raw_idx = v_str.parse::<isize>().map_err(|e| {
                            format!("OBJ line {}: invalid face index '{tok}': {e}", line_no + 1)
                        })?;
                        let resolved = if raw_idx > 0 {
                            (raw_idx - 1) as usize
                        } else if raw_idx < 0 {
                            let n = vertices.len() as isize;
                            let idx = n + raw_idx;
                            if idx < 0 {
                                return Err(format!(
                                    "OBJ line {}: negative index {raw_idx} out of range",
                                    line_no + 1
                                ));
                            }
                            idx as usize
                        } else {
                            return Err(format!("OBJ line {}: zero face index", line_no + 1));
                        };
                        poly.push(resolved as u32);
                    }
                    if poly.len() < 3 {
                        return Err(format!(
                            "OBJ line {}: face has fewer than 3 vertices",
                            line_no + 1
                        ));
                    }
                    let flip = scale.x() * scale.y() * scale.z() < 0.0;
                    for k in 1..(poly.len() - 1) {
                        if flip {
                            indices.push([poly[0], poly[k + 1], poly[k]]);
                        } else {
                            indices.push([poly[0], poly[k], poly[k + 1]]);
                        }
                    }
                }
                _ => {}
            }
        }

        Self::new(vertices, indices, pose, sign_mode, mat_id)
    }

    #[inline]
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    #[inline]
    pub fn indices(&self) -> &[[u32; 3]] {
        &self.indices
    }

    #[inline]
    pub fn pose(&self) -> Se3 {
        self.pose
    }

    #[inline]
    pub fn sign_mode(&self) -> MeshSignMode {
        self.sign_mode
    }

    #[inline]
    pub fn mat_id(&self) -> usize {
        self.mat_id
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        self.local_aabb
    }

    #[inline]
    pub fn parry_mesh(&self) -> &TriMesh {
        &self.mesh
    }

    /// Return world-space vertex positions `[t0, t1, t2]` of triangle `tri_idx`.
    #[inline]
    pub fn world_triangle(&self, tri_idx: usize) -> [Vec3; 3] {
        let [a, b, c] = self.indices[tri_idx];
        [
            self.pose.transform_point(self.vertices[a as usize]),
            self.pose.transform_point(self.vertices[b as usize]),
            self.pose.transform_point(self.vertices[c as usize]),
        ]
    }

    /// Query candidate triangle indices overlapping a world-space AABB using Parry's QBvh.
    pub fn triangles_near_aabb(&self, world_aabb: Aabb) -> Vec<usize> {
        let inv = self.pose.inverse();
        let corners = aabb_corners(world_aabb);
        let mut loc = Aabb::EMPTY;
        for c in corners {
            loc = loc.union(Aabb::from_point(inv.transform_point(c)));
        }
        let p_aabb = parry3d_f64::bounding_volume::Aabb::new(to_pvec(loc.min), to_pvec(loc.max));
        self.mesh
            .bvh()
            .intersect_aabb(&p_aabb)
            .map(|id| id as usize)
            .collect()
    }

    fn pseudonormal_at(&self, tri_idx: usize, loc: TrianglePointLocation) -> Vec3 {
        let tri = self.indices[tri_idx];
        match loc {
            TrianglePointLocation::OnVertex(vi) => {
                let vid = tri[(vi as usize).min(2)] as usize;
                self.vertex_pseudonormals[vid]
            }
            TrianglePointLocation::OnEdge(ei, _) => {
                let (u, v) = match ei {
                    0 => (tri[0], tri[1]),
                    1 => (tri[1], tri[2]),
                    _ => (tri[2], tri[0]),
                };
                let key = if u < v { (u, v) } else { (v, u) };
                self.edge_pseudonormals
                    .get(&key)
                    .copied()
                    .unwrap_or(self.face_normals[tri_idx])
            }
            TrianglePointLocation::OnFace(_, _) | TrianglePointLocation::OnSolid => {
                self.face_normals[tri_idx]
            }
        }
    }
}

impl SdfField for TriMeshSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let p_loc = self.pose.rot.inv_rotate(pos - self.pose.pos);
        let (proj, (tri_u32, loc)) = self
            .mesh
            .project_local_point_and_get_location(to_pvec(p_loc), false);
        let tri_idx = (tri_u32 as usize).min(self.indices.len() - 1);
        let q_loc = from_pvec(proj.point);
        let diff = p_loc - q_loc;
        let dist = diff.norm();
        let pseudo = self.pseudonormal_at(tri_idx, loc);

        let (phi, n_loc) = match self.sign_mode {
            MeshSignMode::UnsignedShell { half_thickness } => {
                let n = if dist > 1e-13 { diff / dist } else { pseudo };
                (dist - half_thickness.max(0.0), n)
            }
            MeshSignMode::ClosedSolid | MeshSignMode::OpenSurface => {
                if dist <= 1e-13 {
                    (0.0, pseudo)
                } else if diff.dot(pseudo) >= 0.0 {
                    (dist, diff / dist)
                } else {
                    (-dist, -diff / dist)
                }
            }
        };

        let (arc_s, tangent_world) = match &self.centerline {
            Some(cl) => {
                let (s, t_loc) = cl.project(p_loc);
                (s, self.pose.rot.rotate(t_loc).normalized_or(Vec3::EX))
            }
            None => (0.0, Vec3::ZERO),
        };

        SdfQuery {
            phi,
            normal: self.pose.rot.rotate(n_loc).normalized_or(Vec3::EZ),
            arc_s,
            tangent: tangent_world,
            mat_id: self.mat_id,
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        match self.sign_mode {
            MeshSignMode::OpenSurface => None,
            MeshSignMode::ClosedSolid | MeshSignMode::UnsignedShell { .. } => {
                Some(transform_aabb(self.local_aabb, self.pose))
            }
        }
    }
}

/// Exact convex-hull 3D SDF backed by `parry3d_f64::shape::ConvexPolyhedron`.
#[derive(Clone)]
pub struct ConvexHullSdf {
    poly: ConvexPolyhedron,
    pose: Se3,
    mat_id: usize,
    inverted: bool,
    local_aabb: Aabb,
}

impl ConvexHullSdf {
    /// Compute the 3D convex hull of `points` (local coordinates) via `parry3d-f64`.
    pub fn from_points(
        points: &[Vec3],
        pose: Se3,
        mat_id: usize,
        inverted: bool,
    ) -> Result<Self, String> {
        if points.len() < 4 {
            return Err("ConvexHullSdf requires at least 4 non-coplanar points".to_string());
        }
        let p_pts: Vec<Vector> = points.iter().map(|&v| to_pvec(v)).collect();
        let poly = ConvexPolyhedron::from_convex_hull(&p_pts)
            .ok_or_else(|| "parry3d ConvexPolyhedron::from_convex_hull failed (degenerate input)".to_string())?;
        let mut local_aabb = Aabb::EMPTY;
        for &p in poly.points() {
            local_aabb = local_aabb.union(Aabb::from_point(from_pvec(p)));
        }
        Ok(Self {
            poly,
            pose,
            mat_id,
            inverted,
            local_aabb,
        })
    }

    #[inline]
    pub fn pose(&self) -> Se3 {
        self.pose
    }

    #[inline]
    pub fn mat_id(&self) -> usize {
        self.mat_id
    }

    #[inline]
    pub fn parry_polyhedron(&self) -> &ConvexPolyhedron {
        &self.poly
    }
}

impl SdfField for ConvexHullSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let p_loc = self.pose.rot.inv_rotate(pos - self.pose.pos);
        // `solid = false` projects interior points onto the closest boundary face so we recover
        // exact negative signed distance inside the convex hull!
        let proj = self.poly.project_local_point(to_pvec(p_loc), false);
        let q_loc = from_pvec(proj.point);
        let diff = p_loc - q_loc;
        let dist = diff.norm();

        let (phi_solid, n_solid) = if proj.is_inside {
            let n = if dist > 1e-13 {
                -diff / dist
            } else {
                // Closest face outward normal fallback at exact boundary
                let mut best_n = Vec3::EZ;
                let mut best_gap = f64::INFINITY;
                for face in self.poly.faces() {
                    let v0 = from_pvec(self.poly.points()[self.poly.vertices_adj_to_face()[face.first_vertex_or_edge as usize] as usize]);
                    let fn_dir = from_pvec(face.normal).normalized_or(Vec3::EZ);
                    let gap = (p_loc - v0).dot(fn_dir).abs();
                    if gap < best_gap {
                        best_gap = gap;
                        best_n = fn_dir;
                    }
                }
                best_n
            };
            (-dist, n)
        } else {
            let n = if dist > 1e-13 {
                diff / dist
            } else {
                p_loc.normalized_or(Vec3::EZ)
            };
            (dist, n)
        };

        let (phi, n_loc) = if self.inverted {
            (-phi_solid, -n_solid)
        } else {
            (phi_solid, n_solid)
        };

        SdfQuery {
            phi,
            normal: self.pose.rot.rotate(n_loc).normalized_or(Vec3::EZ),
            arc_s: 0.0,
            tangent: Vec3::ZERO,
            mat_id: self.mat_id,
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        if self.inverted {
            None
        } else {
            Some(transform_aabb(self.local_aabb, self.pose))
        }
    }
}

/// Procedural generator returning `(vertices, indices, centerline_pts)` for a 3D corrugated
/// cylindrical pipe lumen running from `origin` along `-X` for `length` meters with radius
/// `r(s) = r_mean + corrugation_amp * sin(2 * pi * n_waves * s / length)`.
///
/// Triangles are wound so normals point **inward** toward the centerline (navigable cavity).
#[allow(clippy::too_many_arguments)]
pub fn corrugated_pipe_mesh_data(
    length: f64,
    r_mean: f64,
    corrugation_amp: f64,
    n_waves: f64,
    n_axial: usize,
    n_circ: usize,
    origin: Vec3,
) -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<Vec3>) {
    let na = n_axial.max(4);
    let nc = n_circ.max(8);
    let len = length.max(1e-3);
    let rm = r_mean.max(1e-3);

    let mut vertices = Vec::with_capacity((na + 1) * nc);
    let mut centerline = Vec::with_capacity(na + 1);
    for ia in 0..=na {
        let frac = (ia as f64) / (na as f64);
        let s = frac * len;
        let x = origin.x() - s;
        let r = (rm + corrugation_amp * (2.0 * PI * n_waves * frac).sin()).max(1e-4);
        centerline.push(Vec3::new(x, origin.y(), origin.z()));
        for ic in 0..nc {
            let th = 2.0 * PI * (ic as f64) / (nc as f64);
            vertices.push(Vec3::new(
                x,
                origin.y() + r * th.cos(),
                origin.z() + r * th.sin(),
            ));
        }
    }

    let mut indices = Vec::with_capacity(na * nc * 2);
    for ia in 0..na {
        for ic in 0..nc {
            let ic_next = (ic + 1) % nc;
            let v00 = (ia * nc + ic) as u32;
            let v10 = ((ia + 1) * nc + ic) as u32;
            let v11 = ((ia + 1) * nc + ic_next) as u32;
            let v01 = (ia * nc + ic_next) as u32;
            // Winding chosen so (p1 - p0) x (p2 - p0) points INWARD toward the tube centerline!
            indices.push([v00, v11, v10]);
            indices.push([v00, v01, v11]);
        }
    }
    (vertices, indices, centerline)
}

/// Build a BVH-accelerated [`TriMeshSdf`] for a corrugated cylindrical pipe lumen (`OpenSurface` cavity).
#[allow(clippy::too_many_arguments)]
pub fn corrugated_pipe_mesh(
    length: f64,
    r_mean: f64,
    corrugation_amp: f64,
    n_waves: f64,
    n_axial: usize,
    n_circ: usize,
    origin: Vec3,
    mat_id: usize,
) -> Result<TriMeshSdf, String> {
    let (verts, idx, cl) =
        corrugated_pipe_mesh_data(length, r_mean, corrugation_amp, n_waves, n_axial, n_circ, origin);
    Ok(TriMeshSdf::new(verts, idx, Se3::IDENTITY, MeshSignMode::OpenSurface, mat_id)?
        .with_centerline(cl))
}

/// Procedural generator returning `(vertices, indices)` for a deterministic rocky/rubble 3D
/// triangle-mesh terrain surface with upward (`+Z`) normals.
pub fn rubble_terrain_mesh_data(
    origin_xy: [f64; 2],
    size_xy: [f64; 2],
    nx: usize,
    ny: usize,
    roughness_amp: f64,
    seed: u64,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let nx = nx.max(2);
    let ny = ny.max(2);
    let dx = size_xy[0].max(1e-3) / ((nx - 1) as f64);
    let dy = size_xy[1].max(1e-3) / ((ny - 1) as f64);
    let mut vertices = Vec::with_capacity(nx * ny);

    for iy in 0..ny {
        let y = origin_xy[1] + (iy as f64) * dy;
        for ix in 0..nx {
            let x = origin_xy[0] + (ix as f64) * dx;
            // Deterministic hash-seeded multi-octave rocky rubble elevation
            let h = hash_rubble_height(ix as u64, iy as u64, seed);
            let wave = 0.45 * (3.7 * x + 1.1).sin() * (4.1 * y - 0.7).cos()
                + 0.25 * (9.3 * x - 7.1 * y).cos();
            let z = roughness_amp * (0.55 * h + 0.45 * wave);
            vertices.push(Vec3::new(x, y, z));
        }
    }

    let mut indices = Vec::with_capacity((nx - 1) * (ny - 1) * 2);
    for iy in 0..(ny - 1) {
        for ix in 0..(nx - 1) {
            let v00 = (iy * nx + ix) as u32;
            let v10 = (iy * nx + ix + 1) as u32;
            let v01 = ((iy + 1) * nx + ix) as u32;
            let v11 = ((iy + 1) * nx + ix + 1) as u32;
            // Counter-clockwise in XY => outward normal points +Z!
            indices.push([v00, v10, v11]);
            indices.push([v00, v11, v01]);
        }
    }
    (vertices, indices)
}

/// Build a BVH-accelerated [`TriMeshSdf`] for a rocky rubble terrain surface (`OpenSurface`).
pub fn rubble_terrain_mesh(
    origin_xy: [f64; 2],
    size_xy: [f64; 2],
    nx: usize,
    ny: usize,
    roughness_amp: f64,
    seed: u64,
    mat_id: usize,
) -> Result<TriMeshSdf, String> {
    let (verts, idx) = rubble_terrain_mesh_data(origin_xy, size_xy, nx, ny, roughness_amp, seed);
    TriMeshSdf::new(verts, idx, Se3::IDENTITY, MeshSignMode::OpenSurface, mat_id)
}

/// Result of a `parry3d-f64` swept shape-cast TOI query.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParryShapeCastHit {
    pub toi: f64,
    pub witness1_world: Vec3,
    pub witness2_world: Vec3,
    pub normal1_world: Vec3,
    pub normal2_world: Vec3,
    pub penetrating_at_start: bool,
}

/// Swept-shape continuous collision / time-of-impact queries backed by `parry3d_f64::query::cast_shapes`.
pub struct ParryShapeCast;

impl ParryShapeCast {
    /// Generic linear swept shape cast between two posed `parry3d-f64` shapes.
    #[allow(clippy::too_many_arguments)]
    pub fn cast_shapes_generic(
        pose1: Se3,
        vel1: Vec3,
        shape1: &dyn Shape,
        pose2: Se3,
        vel2: Vec3,
        shape2: &dyn Shape,
        max_toi: f64,
        target_distance: f64,
    ) -> Option<ParryShapeCastHit> {
        let p1 = to_ppose(pose1);
        let p2 = to_ppose(pose2);
        let opts = ShapeCastOptions {
            max_time_of_impact: max_toi.max(0.0),
            target_distance: target_distance.max(0.0),
            stop_at_penetration: true,
            compute_impact_geometry_on_penetration: true,
        };
        let hit = cast_shapes(&p1, to_pvec(vel1), shape1, &p2, to_pvec(vel2), shape2, opts)
            .ok()??;
        let t = hit.time_of_impact;
        let p1_t = Se3::new(pose1.pos + vel1 * t, pose1.rot);
        let p2_t = Se3::new(pose2.pos + vel2 * t, pose2.rot);
        Some(ParryShapeCastHit {
            toi: t,
            witness1_world: p1_t.transform_point(from_pvec(hit.witness1)),
            witness2_world: p2_t.transform_point(from_pvec(hit.witness2)),
            normal1_world: p1_t.transform_vector(from_pvec(hit.normal1)).normalized_or(Vec3::EZ),
            normal2_world: p2_t.transform_vector(from_pvec(hit.normal2)).normalized_or(Vec3::EZ),
            penetrating_at_start: hit.status == ShapeCastStatus::PenetratingOrWithinTargetDist,
        })
    }

    /// Cast a moving sphere of radius `radius` along `center0 + t * disp` (`t in [0, max_toi]`)
    /// against a static [`TriMeshSdf`].
    pub fn cast_sphere_trimesh(
        center0: Vec3,
        disp: Vec3,
        radius: f64,
        mesh_sdf: &TriMeshSdf,
        max_toi: f64,
    ) -> Option<ParryShapeCastHit> {
        let ball = Ball::new(radius.max(1e-9));
        Self::cast_shapes_generic(
            Se3::new(center0, fg_math::Quat::IDENTITY),
            disp,
            &ball,
            mesh_sdf.pose(),
            Vec3::ZERO,
            mesh_sdf.parry_mesh(),
            max_toi,
            0.0,
        )
    }

    /// Cast a moving sphere against a static [`ConvexHullSdf`].
    pub fn cast_sphere_convex(
        center0: Vec3,
        disp: Vec3,
        radius: f64,
        hull_sdf: &ConvexHullSdf,
        max_toi: f64,
    ) -> Option<ParryShapeCastHit> {
        let ball = Ball::new(radius.max(1e-9));
        Self::cast_shapes_generic(
            Se3::new(center0, fg_math::Quat::IDENTITY),
            disp,
            &ball,
            hull_sdf.pose(),
            Vec3::ZERO,
            hull_sdf.parry_polyhedron(),
            max_toi,
            0.0,
        )
    }

    /// Cast a moving [`ConvexHullSdf`] along `disp1` against a static [`TriMeshSdf`].
    pub fn cast_convex_trimesh(
        pose1: Se3,
        disp1: Vec3,
        hull: &ConvexHullSdf,
        mesh_sdf: &TriMeshSdf,
        max_toi: f64,
    ) -> Option<ParryShapeCastHit> {
        Self::cast_shapes_generic(
            pose1,
            disp1,
            hull.parry_polyhedron(),
            mesh_sdf.pose(),
            Vec3::ZERO,
            mesh_sdf.parry_mesh(),
            max_toi,
            0.0,
        )
    }
}

#[inline]
fn angle_between(a: Vec3, b: Vec3) -> f64 {
    let la = a.norm();
    let lb = b.norm();
    if la <= 1e-15 || lb <= 1e-15 {
        return 0.0;
    }
    let cos_th = (a.dot(b) / (la * lb)).clamp(-1.0, 1.0);
    cos_th.acos()
}

#[inline]
fn aabb_corners(b: Aabb) -> [Vec3; 8] {
    [
        Vec3::new(b.min.x(), b.min.y(), b.min.z()),
        Vec3::new(b.max.x(), b.min.y(), b.min.z()),
        Vec3::new(b.min.x(), b.max.y(), b.min.z()),
        Vec3::new(b.max.x(), b.max.y(), b.min.z()),
        Vec3::new(b.min.x(), b.min.y(), b.max.z()),
        Vec3::new(b.max.x(), b.min.y(), b.max.z()),
        Vec3::new(b.min.x(), b.max.y(), b.max.z()),
        Vec3::new(b.max.x(), b.max.y(), b.max.z()),
    ]
}

#[inline]
fn transform_aabb(local: Aabb, pose: Se3) -> Aabb {
    let mut out = Aabb::EMPTY;
    for c in aabb_corners(local) {
        out = out.union(Aabb::from_point(pose.transform_point(c)));
    }
    out
}

/// Deterministic SplitMix64 height hash in `[-1, 1]` for procedural rubble terrain.
#[inline]
fn hash_rubble_height(ix: u64, iy: u64, seed: u64) -> f64 {
    let mut z = ix
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(iy.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(seed.wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let unit = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    2.0 * unit - 1.0
}

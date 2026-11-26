//! `fg-sdf`: BVH-accelerated composable FP64 3D Signed Distance Field (SDF) world
//! and surface material library for `flexgraph-rs`.
//!
//! # Sign & Normal Convention
//! - `phi > 0.0`: point lies strictly outside the solid (free space).
//! - `phi == 0.0`: point lies on the solid boundary.
//! - `phi < 0.0`: point penetrates the solid (penetration depth `delta = -phi`).
//! - `normal`: unit outward normal pointing from solid into free space (`grad(phi) / ||grad(phi)||`).
//!   For interior cavities such as `SplineTubeSdf` in `InteriorCavity` mode, `normal` points
//!   inward toward the tube centerline (i.e., into the navigable free-space lumen).

#![allow(clippy::needless_range_loop)]

pub mod bvh;
pub mod csg;
pub mod material;
#[cfg(feature = "parry-mesh")]
pub mod parry_mesh;
pub mod primitives;
pub mod spline_tube;
pub mod voxel;

use fg_math::Vec3;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub use bvh::{Aabb, Bvh};
pub use csg::{
    DifferenceSdf, IntersectionSdf, SmoothDifferenceSdf, SmoothIntersectionSdf, SmoothUnionSdf,
    TransformSdf, UnionSdf, WorldSdf,
};
pub use material::{
    MaterialLibrary, NormalContactModel, SurfaceMaterial, MAT_ACRYLIC_WALL, MAT_BEKKER_SOFT_SOIL,
    MAT_CARPET, MAT_GRAVEL, MAT_HARD_FLOOR,
};
#[cfg(feature = "parry-mesh")]
pub use parry_mesh::{
    corrugated_pipe_mesh, corrugated_pipe_mesh_data, rubble_terrain_mesh,
    rubble_terrain_mesh_data, ConvexHullSdf, MeshSignMode, ParryShapeCast, ParryShapeCastHit,
    TriMeshSdf,
};
pub use primitives::{
    BoxSdf, CapsuleSdf, CylinderSdf, HeightFieldSdf, HeightFieldSource, MaterialZone, PlaneSdf,
    SphereSdf, ZoneShape,
};
pub use spline_tube::{
    CrossSection, FunnelConfig, RadiusProfile, SplineKind, SplineTubeSdf, TubeWallMode,
};
pub use voxel::{VoxelGridSdf, VoxelInterp};

/// Unified FP64 SDF query result returned by every primitive, CSG node, and world scene.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SdfQuery {
    /// Signed distance (m): `> 0` in free space, `< 0` inside solid.
    pub phi: f64,
    /// Unit outward normal pointing into free space.
    pub normal: Vec3,
    /// Centerline arc-length progress coordinate (m) when defined (e.g. `SplineTubeSdf`, `CapsuleSdf`), else `0.0`.
    pub arc_s: f64,
    /// Unit centerline/reference forward tangent when defined, else `Vec3::ZERO`.
    pub tangent: Vec3,
    /// Surface material identifier indexing into [`MaterialLibrary`].
    pub mat_id: usize,
}

impl SdfQuery {
    /// Far-away empty free-space sentinel query.
    pub const FAR_EMPTY: Self = Self {
        phi: 1.0e12,
        normal: Vec3::EZ,
        arc_s: 0.0,
        tangent: Vec3::ZERO,
        mat_id: MAT_HARD_FLOOR,
    };

    /// Positive penetration depth `max(0, -phi)`.
    #[inline]
    pub fn penetration(self) -> f64 {
        (-self.phi).max(0.0)
    }
}

/// Thread-safe FP64 3D Signed Distance Field trait.
pub trait SdfField: Send + Sync {
    /// Evaluate signed distance, outward unit normal, optional arc-progress/tangent, and material id.
    fn query(&self, pos: Vec3) -> SdfQuery;

    /// Optional conservative world-space bounding box of the solid/active influence region.
    /// Unbounded primitives (e.g. infinite planes, heightfields) return `None`.
    fn bounds(&self) -> Option<Aabb> {
        None
    }

    /// Upper bound on the Lipschitz constant of `phi(x)` (`>= 1.0`) used by conservative SDF CCD.
    fn lipschitz(&self) -> f64 {
        1.0
    }

    /// Batch query helper.
    fn query_batch(&self, points: &[Vec3]) -> Vec<SdfQuery> {
        points.iter().map(|&p| self.query(p)).collect()
    }
}

impl<T: SdfField + ?Sized> SdfField for Arc<T> {
    #[inline]
    fn query(&self, pos: Vec3) -> SdfQuery {
        (**self).query(pos)
    }

    #[inline]
    fn bounds(&self) -> Option<Aabb> {
        (**self).bounds()
    }

    #[inline]
    fn lipschitz(&self) -> f64 {
        (**self).lipschitz()
    }
}

impl<T: SdfField + ?Sized> SdfField for Box<T> {
    #[inline]
    fn query(&self, pos: Vec3) -> SdfQuery {
        (**self).query(pos)
    }

    #[inline]
    fn bounds(&self) -> Option<Aabb> {
        (**self).bounds()
    }

    #[inline]
    fn lipschitz(&self) -> f64 {
        (**self).lipschitz()
    }
}

/// Central finite-difference SDF gradient helper (for testing & gradient verification).
pub fn finite_diff_gradient(sdf: &dyn SdfField, pos: Vec3, eps: f64) -> Vec3 {
    let dx = Vec3::new(eps, 0.0, 0.0);
    let dy = Vec3::new(0.0, eps, 0.0);
    let dz = Vec3::new(0.0, 0.0, eps);
    let inv = 0.5 / eps;
    Vec3::new(
        (sdf.query(pos + dx).phi - sdf.query(pos - dx).phi) * inv,
        (sdf.query(pos + dy).phi - sdf.query(pos - dy).phi) * inv,
        (sdf.query(pos + dz).phi - sdf.query(pos - dz).phi) * inv,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use fg_math::{Quat, Se3};

    #[test]
    fn test_primitives_exact_sdf_and_fd_normals() {
        let sphere = SphereSdf::new(Vec3::new(0.2, -0.1, 0.3), 0.15, MAT_HARD_FLOOR);
        let p = Vec3::new(0.42, 0.05, 0.35);
        let q = sphere.query(p);
        let fd = finite_diff_gradient(&sphere, p, 1e-6);
        assert!((q.normal - fd).norm() < 1e-7);
        assert!((q.phi - ((p - sphere.center).norm() - 0.15)).abs() < 1e-13);

        // Oriented rounded box outside and inside
        let pose = Se3::new(
            Vec3::new(0.1, -0.2, 0.4),
            Quat::from_axis_angle(Vec3::new(0.3, -0.5, 0.8), 0.7),
        );
        let bx = BoxSdf::new(pose, Vec3::new(0.2, 0.12, 0.08), 0.01, MAT_ACRYLIC_WALL);
        for pt in [
            Vec3::new(0.45, -0.15, 0.48),
            Vec3::new(0.12, -0.18, 0.41),
            Vec3::new(-0.2, -0.4, 0.2),
        ] {
            let qb = bx.query(pt);
            let fdb = finite_diff_gradient(&bx, pt, 1e-6);
            assert!(
                (qb.normal - fdb.normalized()).norm() < 1e-6,
                "box normal mismatch at {:?}: analytic={:?}, fd={:?}",
                pt,
                qb.normal,
                fdb
            );
            assert!((fdb.norm() - 1.0).abs() < 1e-5);
        }

        // Finite cylinder outside and inside
        let cyl = CylinderSdf::new(pose, 0.10, Some(0.25), MAT_HARD_FLOOR, false);
        for pt in [
            Vec3::new(0.35, -0.10, 0.50),
            Vec3::new(0.11, -0.19, 0.42),
            Vec3::new(0.10, -0.20, 0.90),
        ] {
            let qc = cyl.query(pt);
            let fdc = finite_diff_gradient(&cyl, pt, 1e-6);
            assert!(
                (qc.normal - fdc.normalized()).norm() < 1e-6,
                "cyl normal mismatch: {:?} vs {:?}",
                qc.normal,
                fdc
            );
        }
    }

    #[test]
    fn test_heightfield_analytic_grid_and_zones() {
        let hf = HeightFieldSdf::analytic(0.05, -0.03, 0.004, 8.0, 6.0, 0.0, MAT_HARD_FLOOR)
            .with_zone(MaterialZone {
                shape: ZoneShape::XInterval {
                    x_min: -3.0,
                    x_max: -1.5,
                },
                mat_id: MAT_BEKKER_SOFT_SOIL,
            })
            .with_zone(MaterialZone {
                shape: ZoneShape::XInterval {
                    x_min: -5.0,
                    x_max: -3.0,
                },
                mat_id: MAT_CARPET,
            });
        let p1 = Vec3::new(-0.5, 0.2, 0.05);
        let q1 = hf.query(p1);
        assert_eq!(q1.mat_id, MAT_HARD_FLOOR);
        assert!(q1.phi > 0.0);
        // Check exact slope normal
        let (z_g, up_g) = hf.height_and_normal(p1.x(), p1.y());
        assert!((q1.normal - up_g).norm() < 1e-14);
        assert!((q1.phi - (p1.z() - z_g) * up_g.z()).abs() < 1e-14);

        assert_eq!(
            hf.query(Vec3::new(-2.0, 0.0, 0.02)).mat_id,
            MAT_BEKKER_SOFT_SOIL
        );
        assert_eq!(hf.query(Vec3::new(-3.5, 0.0, 0.02)).mat_id, MAT_CARPET);

        // Discrete grid heightfield from same analytic function
        let grid = HeightFieldSdf::from_grid_fn(
            [-1.0, -1.0],
            0.1,
            0.1,
            21,
            21,
            |x, y| 0.04 * x - 0.02 * y + 0.01 * (2.0 * x).sin() * (2.0 * y).cos(),
            MAT_GRAVEL,
        );
        let pg = Vec3::new(0.13, -0.27, 0.08);
        let qg = grid.query(pg);
        let fdg = finite_diff_gradient(&grid, pg, 1e-6);
        assert!((qg.normal - fdg.normalized()).norm() < 1e-4);
        assert_eq!(qg.mat_id, MAT_GRAVEL);
    }

    #[test]
    fn test_csg_smooth_union_exact_c2_gradient() {
        let a = Arc::new(SphereSdf::new(Vec3::new(-0.15, 0.0, 0.0), 0.25, MAT_HARD_FLOOR));
        let b = Arc::new(SphereSdf::new(Vec3::new(0.15, 0.0, 0.0), 0.25, MAT_CARPET));
        let su = SmoothUnionSdf::new(a.clone(), b.clone(), 0.12);
        // Point inside blend region where |phi_a - phi_b| < k
        let p = Vec3::new(0.03, 0.22, 0.04);
        let q = su.query(p);
        let fd = finite_diff_gradient(&su, p, 1e-6);
        assert!(
            (q.normal - fd.normalized()).norm() < 1e-6,
            "smooth union normal mismatch: {:?} vs {:?}",
            q.normal,
            fd.normalized()
        );
        assert!(q.phi <= a.query(p).phi.min(b.query(p).phi));

        let diff = DifferenceSdf::new(a, b);
        let pd = Vec3::new(-0.30, 0.10, 0.0);
        let qd = diff.query(pd);
        let fdd = finite_diff_gradient(&diff, pd, 1e-6);
        assert!((qd.normal - fdd.normalized()).norm() < 1e-6);
    }

    #[test]
    fn test_spline_tube_3d_bvh_and_rounded_rect() {
        // 3D helical-S curved tube with variable cosine taper and entry funnel
        let ctrl = vec![
            Vec3::new(0.0, 0.0, 0.12),
            Vec3::new(-0.5, 0.0, 0.12),
            Vec3::new(-1.0, 0.35, 0.18),
            Vec3::new(-1.5, 0.65, 0.28),
            Vec3::new(-2.0, 0.65, 0.32),
        ];
        let tube = SplineTubeSdf::new(
            ctrl,
            SplineKind::CubicBSpline,
            RadiusProfile::CosineTaper {
                s_start: 0.4,
                s_end: 1.4,
                r_start: 0.14,
                r_end: 0.10,
            },
            CrossSection::Circular,
            TubeWallMode::InteriorCavity,
            FunnelConfig {
                entry_len: 0.08,
                entry_flare: 0.05,
                exit_len: 0.0,
                exit_flare: 0.0,
            },
            MAT_ACRYLIC_WALL,
            64,
        );

        // Query a point near the middle curve wall
        let mid_c = tube.eval_centerline(0.5);
        let mid_t = tube.eval_tangent(0.5);
        let radial_dir = if mid_t.z().abs() < 0.9 {
            mid_t.cross(Vec3::EZ).normalized()
        } else {
            mid_t.cross(Vec3::EX).normalized()
        };
        let p_in = mid_c + radial_dir * 0.08;
        let q = tube.query(p_in);
        assert!(q.phi > 0.0, "expected inside tube cavity, got phi={}", q.phi);
        assert!(q.arc_s > 0.5 && q.arc_s < tube.total_length());
        assert!((q.tangent.norm() - 1.0).abs() < 1e-12);
        let fd = finite_diff_gradient(&tube, p_in, 1e-6);
        assert!(
            (q.normal - fd.normalized()).norm() < 1e-4,
            "spline tube normal mismatch: analytic={:?}, fd={:?}",
            q.normal,
            fd.normalized()
        );

        // Rounded-rectangular 3D curved duct
        let rect_tube = SplineTubeSdf::new(
            vec![
                Vec3::new(0.0, 0.0, 0.12),
                Vec3::new(-0.8, 0.0, 0.12),
                Vec3::new(-1.4, 0.6, 0.12),
                Vec3::new(-1.4, 1.4, 0.12),
            ],
            SplineKind::CatmullRom { tension: 0.5 },
            RadiusProfile::Constant(0.12),
            CrossSection::RoundedRect {
                half_width: 0.13,
                half_height: 0.11,
                corner_radius: 0.02,
            },
            TubeWallMode::InteriorCavity,
            FunnelConfig::NONE,
            MAT_ACRYLIC_WALL,
            64,
        );
        let pr = rect_tube.eval_centerline(0.35) + Vec3::new(0.0, 0.0, 0.07);
        let qr = rect_tube.query(pr);
        let fdr = finite_diff_gradient(&rect_tube, pr, 1e-6);
        assert!(qr.phi > 0.0);
        assert!(
            (qr.normal - fdr.normalized()).norm() < 1e-4,
            "rounded rect tube normal mismatch: {:?} vs {:?}",
            qr.normal,
            fdr.normalized()
        );
    }

    #[test]
    fn test_voxel_grid_sdf_trilinear_and_tricubic_grad() {
        let sphere = SphereSdf::new(Vec3::new(0.5, 0.5, 0.5), 0.25, MAT_HARD_FLOOR);
        for interp in [VoxelInterp::Trilinear, VoxelInterp::Tricubic] {
            let grid = VoxelGridSdf::bake_from_sdf(
                &sphere,
                Vec3::new(0.0, 0.0, 0.0),
                0.05,
                [21, 21, 21],
                MAT_HARD_FLOOR,
                interp,
            );
            let p = Vec3::new(0.72, 0.58, 0.44);
            let q = grid.query(p);
            let q_ref = sphere.query(p);
            assert!((q.phi - q_ref.phi).abs() < 3e-3);
            let fd = finite_diff_gradient(&grid, p, 1e-6);
            assert!(
                (q.normal - fd.normalized()).norm() < 1e-6,
                "voxel {:?} analytic grad vs FD: {:?} vs {:?}",
                interp,
                q.normal,
                fd.normalized()
            );
        }

        // Also test CapsuleSdf, TransformSdf, SmoothIntersectionSdf, SmoothDifferenceSdf, bend90 & endoscope tubes
        let cap = CapsuleSdf::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.1, MAT_HARD_FLOOR);
        let q_cap = cap.query(Vec3::new(0.25, 0.0, 0.4));
        assert!((q_cap.phi - 0.15).abs() < 1e-13 && (q_cap.arc_s - 0.4).abs() < 1e-13);

        let tf_sphere = TransformSdf::new(
            Arc::new(SphereSdf::new(Vec3::ZERO, 0.2, MAT_CARPET)),
            Se3::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_axis_angle(Vec3::EZ, 0.8)),
        );
        let q_tf = tf_sphere.query(Vec3::new(1.0, 2.0, 3.35));
        assert!((q_tf.phi - 0.15).abs() < 1e-13 && (q_tf.normal - Vec3::EZ).norm() < 1e-12);

        let bend = SplineTubeSdf::bend90_pipe(-1.1, -1.55, 0.50, 0.122, 1.0, 0.75, 0.0, 0.118, MAT_ACRYLIC_WALL);
        let p_mid = bend.eval_centerline(0.5);
        assert!(bend.query(p_mid).phi > 0.11);

        let endo = SplineTubeSdf::curved_endoscope_3d(0.02, MAT_ACRYLIC_WALL);
        let p_endo = endo.eval_centerline(0.6);
        assert!(endo.query(p_endo).phi > 0.019);
    }

    #[test]
    fn test_world_sdf_bvh_composite() {
        let mut world = WorldSdf::new(MaterialLibrary::standard());
        world.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, MAT_HARD_FLOOR)));
        world.add_bounded(Arc::new(SphereSdf::new(
            Vec3::new(1.0, 0.0, 0.3),
            0.2,
            MAT_GRAVEL,
        )));
        world.add_bounded(Arc::new(BoxSdf::axis_aligned(
            Vec3::new(-1.0, 0.0, 0.25),
            Vec3::new(0.2, 0.2, 0.25),
            0.01,
            MAT_ACRYLIC_WALL,
        )));
        world.rebuild_bvh();

        // Near ground
        let qg = world.query(Vec3::new(0.0, 0.0, 0.02));
        assert_eq!(qg.mat_id, MAT_HARD_FLOOR);
        assert!((qg.phi - 0.02).abs() < 1e-14);

        // Near sphere
        let qs = world.query(Vec3::new(1.0, 0.0, 0.55));
        assert_eq!(qs.mat_id, MAT_GRAVEL);
        assert!((qs.phi - 0.05).abs() < 1e-13);

        // Near box: inner half_extent.x = 0.20 + rounding 0.01 = 0.21 => face at x = -0.79
        let qb = world.query(Vec3::new(-0.77, 0.0, 0.25));
        assert_eq!(qb.mat_id, MAT_ACRYLIC_WALL);
        assert!((qb.phi - 0.02).abs() < 1e-13);

        // InteriorCavity SplineTubeSdf added via add_bounded must NEVER be pruned outside its lumen!
        let mut tube_world = WorldSdf::new(MaterialLibrary::standard());
        tube_world.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, MAT_HARD_FLOOR)));
        let cavity_tube = Arc::new(SplineTubeSdf::new(
            vec![Vec3::new(0.0, 0.0, 0.12), Vec3::new(-1.0, 0.0, 0.12)],
            SplineKind::CatmullRom { tension: 0.5 },
            RadiusProfile::Constant(0.10),
            CrossSection::Circular,
            TubeWallMode::InteriorCavity,
            FunnelConfig::NONE,
            MAT_ACRYLIC_WALL,
            16,
        ));
        assert!(cavity_tube.bounds().is_none());
        tube_world.add_bounded(cavity_tube);
        tube_world.rebuild_bvh();
        // Query far outside tube wall at y = 0.26 (penetration = 0.16 into wall, NOT +0.12 from ground!)
        let qt = tube_world.query(Vec3::new(-0.5, 0.26, 0.12));
        assert_eq!(qt.mat_id, MAT_ACRYLIC_WALL);
        assert!(
            (qt.phi - (-0.16)).abs() < 1e-10,
            "InteriorCavity tube was pruned! expected -0.16, got {}",
            qt.phi
        );
    }

    #[cfg(feature = "parry-mesh")]
    #[test]
    fn test_parry_trimesh_obj_convex_corrugated_rubble_and_shapecast() {
        // 1. Minimal Wavefront .obj watertight unit cube centered at origin (half-extent 0.5)
        let cube_obj = r#"
            # unit cube
            v -0.5 -0.5 -0.5
            v  0.5 -0.5 -0.5
            v  0.5  0.5 -0.5
            v -0.5  0.5 -0.5
            v -0.5 -0.5  0.5
            v  0.5 -0.5  0.5
            v  0.5  0.5  0.5
            v -0.5  0.5  0.5
            f 1 4 3 2
            f 5 6 7 8
            f 1 2 6 5
            f 3 4 8 7
            f 2 3 7 6
            f 4 1 5 8
        "#;
        let cube = TriMeshSdf::from_obj_str(
            cube_obj,
            Vec3::new(1.0, 1.0, 1.0),
            Se3::IDENTITY,
            MeshSignMode::ClosedSolid,
            MAT_HARD_FLOOR,
        )
        .expect("OBJ cube must parse");
        assert_eq!(cube.indices().len(), 12);

        // Outside face, outside edge, outside corner, and strictly inside sign check!
        let q_out = cube.query(Vec3::new(0.75, 0.1, -0.2));
        assert!((q_out.phi - 0.25).abs() < 1e-12 && (q_out.normal - Vec3::EX).norm() < 1e-12);
        let q_in = cube.query(Vec3::new(0.30, 0.0, 0.0));
        assert!((q_in.phi - (-0.20)).abs() < 1e-12 && (q_in.normal - Vec3::EX).norm() < 1e-12);
        let q_corner = cube.query(Vec3::new(0.8, 0.9, 0.5));
        let fd_c = finite_diff_gradient(&cube, Vec3::new(0.8, 0.9, 0.5), 1e-6);
        assert!((q_corner.normal - fd_c.normalized()).norm() < 1e-6);

        // 2. ConvexHullSdf from point cloud (with interior points filtered by Parry)
        let hull_pts = vec![
            Vec3::new(-0.2, -0.2, -0.2),
            Vec3::new(0.2, -0.2, -0.2),
            Vec3::new(0.2, 0.2, -0.2),
            Vec3::new(-0.2, 0.2, -0.2),
            Vec3::new(-0.2, -0.2, 0.2),
            Vec3::new(0.2, -0.2, 0.2),
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::new(-0.2, 0.2, 0.2),
            Vec3::new(0.0, 0.0, 0.0), // interior point
        ];
        let hull = ConvexHullSdf::from_points(
            &hull_pts,
            Se3::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY),
            MAT_ACRYLIC_WALL,
            false,
        )
        .expect("convex hull must build");
        assert!((hull.query(Vec3::new(1.35, 0.0, 0.0)).phi - 0.15).abs() < 1e-12);
        assert!((hull.query(Vec3::new(1.05, 0.0, 0.0)).phi - (-0.15)).abs() < 1e-12);

        // 3. Procedural corrugated pipe mesh (navigable cavity, arc_s, inward normal)
        let pipe = corrugated_pipe_mesh(
            1.2,
            0.12,
            0.012,
            6.0,
            48,
            24,
            Vec3::new(0.0, 0.0, 0.12),
            MAT_ACRYLIC_WALL,
        )
        .expect("corrugated pipe mesh must build");
        assert!(pipe.bounds().is_none()); // OpenSurface cavity never pruned by WorldSdf
        let q_pipe_in = pipe.query(Vec3::new(-0.5, 0.0, 0.12));
        assert!(q_pipe_in.phi > 0.09, "centerline must be inside free lumen: {}", q_pipe_in.phi);
        assert!((q_pipe_in.arc_s - 0.5).abs() < 1e-3);
        assert!((q_pipe_in.tangent - Vec3::new(-1.0, 0.0, 0.0)).norm() < 1e-12);
        // Outside corrugated pipe wall -> negative phi (penetrating wall) & inward normal!
        let q_pipe_out = pipe.query(Vec3::new(-0.5, 0.20, 0.12));
        assert!(q_pipe_out.phi < -0.05 && q_pipe_out.normal.y() < -0.7);

        // 4. Procedural rubble terrain mesh (upward +Z free space, negative underground)
        let rubble = rubble_terrain_mesh([-1.0, -1.0], [2.0, 2.0], 16, 16, 0.03, 42, MAT_GRAVEL)
            .expect("rubble terrain must build");
        assert!(rubble.query(Vec3::new(0.0, 0.0, 0.20)).phi > 0.14);
        assert!(rubble.query(Vec3::new(0.0, 0.0, -0.20)).phi < -0.14);

        // 5. ParryShapeCast swept sphere vs TriMesh cube (exact TOI check)
        // Sphere r=0.1 starts at x=1.6 moving -X by -2.0; cube face is at x=0.5 => exact hit at x=0.6 => toi=0.5!
        let hit = ParryShapeCast::cast_sphere_trimesh(
            Vec3::new(1.6, 0.0, 0.0),
            Vec3::new(-2.0, 0.0, 0.0),
            0.1,
            &cube,
            1.0,
        )
        .expect("sphere must hit cube");
        assert!(
            (hit.toi - 0.5).abs() < 1e-6,
            "ParryShapeCast sphere-cube TOI mismatch: got {}",
            hit.toi
        );
        assert!(!hit.penetrating_at_start);
        assert!(
            (hit.witness2_world - Vec3::new(0.5, 0.0, 0.0)).norm() < 1e-5,
            "ParryShapeCast witness2 mismatch: {:?}",
            hit.witness2_world
        );
        assert!(
            (hit.normal2_world - Vec3::EX).norm() < 1e-5,
            "ParryShapeCast normal2 mismatch: {:?}",
            hit.normal2_world
        );

        // Edge & vertex inside/outside pseudonormal sign classification on rotated cube
        let rot_cube = TriMeshSdf::from_obj_str(
            cube_obj,
            Vec3::new(1.0, 1.0, 1.0),
            Se3::new(
                Vec3::new(0.2, -0.3, 0.4),
                Quat::from_axis_angle(Vec3::new(0.3, 0.6, -0.7), 0.9),
            ),
            MeshSignMode::ClosedSolid,
            MAT_HARD_FLOOR,
        )
        .expect("rotated cube must build");
        // Strictly inside near an edge (0.45, 0.45, 0.0) in local coords -> negative phi = -0.05
        let p_in_edge = rot_cube.pose().transform_point(Vec3::new(0.45, 0.45, 0.0));
        let q_in_edge = rot_cube.query(p_in_edge);
        assert!(
            (q_in_edge.phi - (-0.05)).abs() < 1e-11,
            "rotated cube interior-edge phi mismatch: {}",
            q_in_edge.phi
        );
        // Strictly outside near an edge (0.6, 0.6, 0.0) in local coords -> positive phi = 0.1 * sqrt(2)
        let p_out_edge = rot_cube.pose().transform_point(Vec3::new(0.6, 0.6, 0.0));
        let q_out_edge = rot_cube.query(p_out_edge);
        assert!(
            (q_out_edge.phi - (0.1 * std::f64::consts::SQRT_2)).abs() < 1e-11,
            "rotated cube exterior-edge phi mismatch: {}",
            q_out_edge.phi
        );
    }
}

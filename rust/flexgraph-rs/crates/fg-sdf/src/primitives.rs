//! Exact analytical 3D SDF primitives: `PlaneSdf`, `SphereSdf`, `BoxSdf`,
//! `CylinderSdf`, `CapsuleSdf`, and sloped/undulating/grid zoned `HeightFieldSdf`.

use crate::bvh::Aabb;
use crate::{SdfField, SdfQuery};
use fg_math::{Quat, Se3, Vec3};
use serde::{Deserialize, Serialize};

/// Infinite half-space plane `{ x : (x - origin) · normal >= 0 }`.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaneSdf {
    pub origin: Vec3,
    pub normal: Vec3,
    pub mat_id: usize,
}

impl PlaneSdf {
    pub fn new(origin: Vec3, normal: Vec3, mat_id: usize) -> Self {
        Self {
            origin,
            normal: normal.normalized_or(Vec3::EZ),
            mat_id,
        }
    }

    /// Flat horizontal ground plane `z = z0` with outward normal `+Z`.
    pub fn flat_ground(z0: f64, mat_id: usize) -> Self {
        Self {
            origin: Vec3::new(0.0, 0.0, z0),
            normal: Vec3::EZ,
            mat_id,
        }
    }
}

impl SdfField for PlaneSdf {
    #[inline]
    fn query(&self, pos: Vec3) -> SdfQuery {
        SdfQuery {
            phi: (pos - self.origin).dot(self.normal),
            normal: self.normal,
            arc_s: 0.0,
            tangent: Vec3::ZERO,
            mat_id: self.mat_id,
        }
    }
}

/// Solid sphere (or spherical cavity when `inverted = true`).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SphereSdf {
    pub center: Vec3,
    pub radius: f64,
    pub mat_id: usize,
    #[serde(default)]
    pub inverted: bool,
}

impl SphereSdf {
    pub fn new(center: Vec3, radius: f64, mat_id: usize) -> Self {
        Self {
            center,
            radius: radius.max(0.0),
            mat_id,
            inverted: false,
        }
    }

    pub fn cavity(center: Vec3, radius: f64, mat_id: usize) -> Self {
        Self {
            center,
            radius: radius.max(0.0),
            mat_id,
            inverted: true,
        }
    }
}

impl SdfField for SphereSdf {
    #[inline]
    fn query(&self, pos: Vec3) -> SdfQuery {
        let d = pos - self.center;
        let r = d.norm();
        let n_out = if r > 1e-15 { d / r } else { Vec3::EZ };
        let (phi, normal) = if self.inverted {
            (self.radius - r, -n_out)
        } else {
            (r - self.radius, n_out)
        };
        SdfQuery {
            phi,
            normal,
            arc_s: 0.0,
            tangent: Vec3::ZERO,
            mat_id: self.mat_id,
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        if self.inverted {
            None
        } else {
            let r = Vec3::new(self.radius, self.radius, self.radius);
            Some(Aabb::new(self.center - r, self.center + r))
        }
    }
}

/// Oriented 3D box (with optional Minkowski corner rounding radius `rounding >= 0`).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxSdf {
    pub pose: Se3,
    /// Inner half-extents (total physical half-extent along axis `i` is `half_extents[i] + rounding`).
    pub half_extents: Vec3,
    #[serde(default)]
    pub rounding: f64,
    pub mat_id: usize,
}

impl BoxSdf {
    pub fn new(pose: Se3, half_extents: Vec3, rounding: f64, mat_id: usize) -> Self {
        Self {
            pose,
            half_extents: Vec3::new(
                half_extents.x().max(1e-9),
                half_extents.y().max(1e-9),
                half_extents.z().max(1e-9),
            ),
            rounding: rounding.max(0.0),
            mat_id,
        }
    }

    pub fn axis_aligned(center: Vec3, half_extents: Vec3, rounding: f64, mat_id: usize) -> Self {
        Self::new(
            Se3::new(center, Quat::IDENTITY),
            half_extents,
            rounding,
            mat_id,
        )
    }
}

impl SdfField for BoxSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let p_loc = self.pose.rot.inv_rotate(pos - self.pose.pos);
        let qx = p_loc.x().abs() - self.half_extents.x();
        let qy = p_loc.y().abs() - self.half_extents.y();
        let qz = p_loc.z().abs() - self.half_extents.z();

        let ox = qx.max(0.0);
        let oy = qy.max(0.0);
        let oz = qz.max(0.0);
        let out_norm_sq = ox * ox + oy * oy + oz * oz;

        let (phi, n_loc) = if out_norm_sq > 0.0 {
            let out_norm = out_norm_sq.sqrt();
            let sx = if p_loc.x() >= 0.0 { 1.0 } else { -1.0 };
            let sy = if p_loc.y() >= 0.0 { 1.0 } else { -1.0 };
            let sz = if p_loc.z() >= 0.0 { 1.0 } else { -1.0 };
            let n = Vec3::new(sx * ox, sy * oy, sz * oz) / out_norm;
            (out_norm - self.rounding, n)
        } else {
            // Strictly inside inner box: closest face determines exact signed distance and normal.
            let (max_q, n) = if qx >= qy && qx >= qz {
                (qx, Vec3::new(if p_loc.x() >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0))
            } else if qy >= qz {
                (qy, Vec3::new(0.0, if p_loc.y() >= 0.0 { 1.0 } else { -1.0 }, 0.0))
            } else {
                (qz, Vec3::new(0.0, 0.0, if p_loc.z() >= 0.0 { 1.0 } else { -1.0 }))
            };
            (max_q - self.rounding, n)
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
        let r = self.pose.rot.to_mat3();
        let he = Vec3::new(
            self.half_extents.x() + self.rounding,
            self.half_extents.y() + self.rounding,
            self.half_extents.z() + self.rounding,
        );
        let mut world_he = [0.0; 3];
        for i in 0..3 {
            world_he[i] = r.0[i][0].abs() * he.x()
                + r.0[i][1].abs() * he.y()
                + r.0[i][2].abs() * he.z();
        }
        let w = Vec3(world_he);
        Some(Aabb::new(self.pose.pos - w, self.pose.pos + w))
    }
}

/// Oriented cylinder aligned with its local Z axis (finite when `half_height = Some(h)`,
/// infinite when `None`; cavity when `inverted = true`).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CylinderSdf {
    pub pose: Se3,
    pub radius: f64,
    pub half_height: Option<f64>,
    pub mat_id: usize,
    #[serde(default)]
    pub inverted: bool,
}

impl CylinderSdf {
    pub fn new(
        pose: Se3,
        radius: f64,
        half_height: Option<f64>,
        mat_id: usize,
        inverted: bool,
    ) -> Self {
        Self {
            pose,
            radius: radius.max(1e-9),
            half_height: half_height.map(|h| h.max(1e-9)),
            mat_id,
            inverted,
        }
    }
}

impl SdfField for CylinderSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let p_loc = self.pose.rot.inv_rotate(pos - self.pose.pos);
        let rho = (p_loc.x() * p_loc.x() + p_loc.y() * p_loc.y()).sqrt();
        let dr = rho - self.radius;
        let radial_dir = if rho > 1e-15 {
            Vec3::new(p_loc.x() / rho, p_loc.y() / rho, 0.0)
        } else {
            Vec3::EX
        };

        let (phi_solid, n_solid) = match self.half_height {
            None => (dr, radial_dir),
            Some(hh) => {
                let dz = p_loc.z().abs() - hh;
                let sz = if p_loc.z() >= 0.0 { 1.0 } else { -1.0 };
                let or = dr.max(0.0);
                let oz = dz.max(0.0);
                if or > 0.0 && oz > 0.0 {
                    let dist = (or * or + oz * oz).sqrt();
                    let n = (radial_dir * or + Vec3::new(0.0, 0.0, sz * oz)) / dist;
                    (dist, n)
                } else if or > 0.0 {
                    (dr, radial_dir)
                } else if oz > 0.0 {
                    (dz, Vec3::new(0.0, 0.0, sz))
                } else if dr >= dz {
                    (dr, radial_dir)
                } else {
                    (dz, Vec3::new(0.0, 0.0, sz))
                }
            }
        };

        let (phi, n_loc) = if self.inverted {
            (-phi_solid, -n_solid)
        } else {
            (phi_solid, n_solid)
        };

        let tangent = self.pose.rot.rotate(Vec3::EZ);
        SdfQuery {
            phi,
            normal: self.pose.rot.rotate(n_loc).normalized_or(Vec3::EZ),
            arc_s: p_loc.z(),
            tangent,
            mat_id: self.mat_id,
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        if self.inverted {
            return None;
        }
        let hh = self.half_height?;
        let r_mat = self.pose.rot.to_mat3();
        let he = Vec3::new(self.radius, self.radius, hh);
        let mut world_he = [0.0; 3];
        for i in 0..3 {
            world_he[i] = r_mat.0[i][0].abs() * he.x()
                + r_mat.0[i][1].abs() * he.y()
                + r_mat.0[i][2].abs() * he.z();
        }
        let w = Vec3(world_he);
        Some(Aabb::new(self.pose.pos - w, self.pose.pos + w))
    }
}

/// Finite 3D capsule connecting `a` and `b` with radius `radius`.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapsuleSdf {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f64,
    pub mat_id: usize,
}

impl CapsuleSdf {
    pub fn new(a: Vec3, b: Vec3, radius: f64, mat_id: usize) -> Self {
        Self {
            a,
            b,
            radius: radius.max(0.0),
            mat_id,
        }
    }
}

impl SdfField for CapsuleSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let ab = self.b - self.a;
        let len_sq = ab.norm_sq();
        let (t, len, tangent) = if len_sq > 1e-24 {
            let len = len_sq.sqrt();
            let t = ((pos - self.a).dot(ab) / len_sq).clamp(0.0, 1.0);
            (t, len, ab / len)
        } else {
            (0.0, 0.0, Vec3::EX)
        };
        let closest = self.a + ab * t;
        let diff = pos - closest;
        let dist = diff.norm();
        let normal = if dist > 1e-15 {
            diff / dist
        } else if tangent.z().abs() < 0.9 {
            tangent.cross(Vec3::EZ).normalized()
        } else {
            tangent.cross(Vec3::EX).normalized()
        };
        SdfQuery {
            phi: dist - self.radius,
            normal,
            arc_s: t * len,
            tangent,
            mat_id: self.mat_id,
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        Some(Aabb::from_segment(self.a, self.b, self.radius))
    }
}

/// Planar 2D region selector for assigning heterogeneous terrain materials across a heightfield.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum ZoneShape {
    XInterval { x_min: f64, x_max: f64 },
    YInterval { y_min: f64, y_max: f64 },
    Box2D { min_xy: [f64; 2], max_xy: [f64; 2] },
    Circle2D { center_xy: [f64; 2], radius: f64 },
}

impl ZoneShape {
    #[inline]
    pub fn contains(self, x: f64, y: f64) -> bool {
        match self {
            ZoneShape::XInterval { x_min, x_max } => x >= x_min && x <= x_max,
            ZoneShape::YInterval { y_min, y_max } => y >= y_min && y <= y_max,
            ZoneShape::Box2D { min_xy, max_xy } => {
                x >= min_xy[0] && x <= max_xy[0] && y >= min_xy[1] && y <= max_xy[1]
            }
            ZoneShape::Circle2D { center_xy, radius } => {
                let dx = x - center_xy[0];
                let dy = y - center_xy[1];
                dx * dx + dy * dy <= radius * radius
            }
        }
    }
}

/// Spatial material override zone on a heightfield.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialZone {
    pub shape: ZoneShape,
    pub mat_id: usize,
}

/// Height function representation for [`HeightFieldSdf`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum HeightFieldSource {
    /// Exact analytical sloped + sinusoidal undulating surface matching
    /// `serpworm.physics.gpu_core.GPUCore.ground_height_and_normal`:
    /// `z(x, y) = base_z + slope_x * x + slope_y * y + wave_amp * sin(wave_kx * x) * cos(wave_ky * y)`.
    Analytic {
        slope_x: f64,
        slope_y: f64,
        wave_amp: f64,
        wave_kx: f64,
        wave_ky: f64,
        base_z: f64,
    },
    /// Regular 2D bilinear-interpolated heightfield grid stored in row-major `[iy * nx + ix]`.
    Grid {
        origin_xy: [f64; 2],
        dx: f64,
        dy: f64,
        nx: usize,
        ny: usize,
        heights: Vec<f64>,
    },
}

/// Sloped, undulating, or discrete gridded terrain heightfield with multi-zone soil materials.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeightFieldSdf {
    pub source: HeightFieldSource,
    pub default_mat_id: usize,
    #[serde(default)]
    pub zones: Vec<MaterialZone>,
}

impl HeightFieldSdf {
    /// Analytical sloped/undulating ground matching `GPUCore::ground_height_and_normal`.
    pub fn analytic(
        slope_x: f64,
        slope_y: f64,
        wave_amp: f64,
        wave_kx: f64,
        wave_ky: f64,
        base_z: f64,
        default_mat_id: usize,
    ) -> Self {
        Self {
            source: HeightFieldSource::Analytic {
                slope_x,
                slope_y,
                wave_amp,
                wave_kx,
                wave_ky,
                base_z,
            },
            default_mat_id,
            zones: Vec::new(),
        }
    }

    /// Flat horizontal ground `z = z0`.
    pub fn flat(z0: f64, default_mat_id: usize) -> Self {
        Self::analytic(0.0, 0.0, 0.0, 0.0, 0.0, z0, default_mat_id)
    }

    /// Build a discrete grid heightfield by sampling an arbitrary function `f(x, y)`.
    pub fn from_grid_fn<F>(
        origin_xy: [f64; 2],
        dx: f64,
        dy: f64,
        nx: usize,
        ny: usize,
        mut f: F,
        default_mat_id: usize,
    ) -> Self
    where
        F: FnMut(f64, f64) -> f64,
    {
        assert!(nx >= 2 && ny >= 2, "HeightField grid must be at least 2x2");
        assert!(dx > 0.0 && dy > 0.0, "HeightField grid spacing must be > 0");
        let mut heights = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            let y = origin_xy[1] + (iy as f64) * dy;
            for ix in 0..nx {
                let x = origin_xy[0] + (ix as f64) * dx;
                heights.push(f(x, y));
            }
        }
        Self {
            source: HeightFieldSource::Grid {
                origin_xy,
                dx,
                dy,
                nx,
                ny,
                heights,
            },
            default_mat_id,
            zones: Vec::new(),
        }
    }

    /// Attach a spatial material zone (later zones override earlier ones if overlapping).
    pub fn with_zone(mut self, zone: MaterialZone) -> Self {
        self.zones.push(zone);
        self
    }

    /// Resolve surface material ID at planar coordinates `(x, y)`.
    #[inline]
    pub fn material_at(&self, x: f64, y: f64) -> usize {
        for zone in self.zones.iter().rev() {
            if zone.shape.contains(x, y) {
                return zone.mat_id;
            }
        }
        self.default_mat_id
    }

    /// Exact terrain height `z_g(x, y)` and local unit upward normal `up_g(x, y)`.
    pub fn height_and_normal(&self, x: f64, y: f64) -> (f64, Vec3) {
        match &self.source {
            HeightFieldSource::Analytic {
                slope_x,
                slope_y,
                wave_amp,
                wave_kx,
                wave_ky,
                base_z,
            } => {
                let sx = (*wave_kx * x).sin();
                let cx = (*wave_kx * x).cos();
                let sy = (*wave_ky * y).sin();
                let cy = (*wave_ky * y).cos();
                let z_g = *base_z + *slope_x * x + *slope_y * y + *wave_amp * sx * cy;
                let dz_dx = *slope_x + *wave_amp * *wave_kx * cx * cy;
                let dz_dy = *slope_y - *wave_amp * *wave_ky * sx * sy;
                let up = Vec3::new(-dz_dx, -dz_dy, 1.0).normalized();
                (z_g, up)
            }
            HeightFieldSource::Grid {
                origin_xy,
                dx,
                dy,
                nx,
                ny,
                heights,
            } => {
                let fx = ((x - origin_xy[0]) / *dx).clamp(0.0, (*nx - 1) as f64);
                let fy = ((y - origin_xy[1]) / *dy).clamp(0.0, (*ny - 1) as f64);
                let ix = (fx.floor() as usize).min(*nx - 2);
                let iy = (fy.floor() as usize).min(*ny - 2);
                let u = fx - ix as f64;
                let v = fy - iy as f64;

                let h00 = heights[iy * *nx + ix];
                let h10 = heights[iy * *nx + (ix + 1)];
                let h01 = heights[(iy + 1) * *nx + ix];
                let h11 = heights[(iy + 1) * *nx + (ix + 1)];

                let z_g = (1.0 - u) * (1.0 - v) * h00
                    + u * (1.0 - v) * h10
                    + (1.0 - u) * v * h01
                    + u * v * h11;
                let dz_dx = ((1.0 - v) * (h10 - h00) + v * (h11 - h01)) / *dx;
                let dz_dy = ((1.0 - u) * (h01 - h00) + u * (h11 - h10)) / *dy;
                let up = Vec3::new(-dz_dx, -dz_dy, 1.0).normalized();
                (z_g, up)
            }
        }
    }
}

impl SdfField for HeightFieldSdf {
    #[inline]
    fn query(&self, pos: Vec3) -> SdfQuery {
        let (z_g, up_g) = self.height_and_normal(pos.x(), pos.y());
        // Exact tangent-plane signed distance matching `GPUCore`:
        // penetration = (z_g - pos.z) * up_g.z => phi = (pos.z - z_g) * up_g.z.
        let phi = (pos.z() - z_g) * up_g.z();
        SdfQuery {
            phi,
            normal: up_g,
            arc_s: 0.0,
            tangent: Vec3::ZERO,
            mat_id: self.material_at(pos.x(), pos.y()),
        }
    }
}

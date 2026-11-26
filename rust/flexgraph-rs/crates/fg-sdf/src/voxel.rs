//! Regular 3D `VoxelGridSdf` with exact analytical trilinear gradient and
//! Catmull-Rom tricubic interpolation, plus automatic baking from any [`SdfField`].

use crate::bvh::Aabb;
use crate::{SdfField, SdfQuery};
use fg_math::Vec3;
use serde::{Deserialize, Serialize};

/// Spatial interpolation mode for [`VoxelGridSdf`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoxelInterp {
    /// Trilinear value interpolation with exact analytical trilinear gradient.
    Trilinear,
    /// $C^1$ Catmull-Rom tricubic interpolation with exact analytical cubic gradient.
    Tricubic,
}

/// Dense 3D FP64 signed distance grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoxelGridSdf {
    pub origin: Vec3,
    pub voxel_size: f64,
    pub dims: [usize; 3],
    /// Flat row-major storage indexed as `(iz * ny + iy) * nx + ix`.
    pub values: Vec<f64>,
    #[serde(default)]
    pub mat_ids: Option<Vec<usize>>,
    pub default_mat_id: usize,
    pub interp: VoxelInterp,
}

impl VoxelGridSdf {
    /// Construct a voxel grid from precomputed SDF values.
    pub fn new(
        origin: Vec3,
        voxel_size: f64,
        dims: [usize; 3],
        values: Vec<f64>,
        mat_ids: Option<Vec<usize>>,
        default_mat_id: usize,
        interp: VoxelInterp,
    ) -> Self {
        assert!(
            dims[0] >= 2 && dims[1] >= 2 && dims[2] >= 2,
            "VoxelGridSdf requires dims >= [2, 2, 2]"
        );
        assert!(voxel_size > 0.0, "voxel_size must be positive");
        let n = dims[0] * dims[1] * dims[2];
        assert_eq!(values.len(), n, "values length must equal nx*ny*nz");
        if let Some(ref m) = mat_ids {
            assert_eq!(m.len(), n, "mat_ids length must equal nx*ny*nz");
        }
        Self {
            origin,
            voxel_size,
            dims,
            values,
            mat_ids,
            default_mat_id,
            interp,
        }
    }

    /// Bake any [`SdfField`] into a dense 3D voxel grid.
    pub fn bake_from_sdf(
        sdf: &dyn SdfField,
        origin: Vec3,
        voxel_size: f64,
        dims: [usize; 3],
        default_mat_id: usize,
        interp: VoxelInterp,
    ) -> Self {
        let [nx, ny, nz] = dims;
        let mut values = Vec::with_capacity(nx * ny * nz);
        let mut mats = Vec::with_capacity(nx * ny * nz);
        for iz in 0..nz {
            let z = origin.z() + iz as f64 * voxel_size;
            for iy in 0..ny {
                let y = origin.y() + iy as f64 * voxel_size;
                for ix in 0..nx {
                    let x = origin.x() + ix as f64 * voxel_size;
                    let q = sdf.query(Vec3::new(x, y, z));
                    values.push(q.phi);
                    mats.push(q.mat_id);
                }
            }
        }
        Self::new(
            origin,
            voxel_size,
            dims,
            values,
            Some(mats),
            default_mat_id,
            interp,
        )
    }

    #[inline]
    fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (iz * self.dims[1] + iy) * self.dims[0] + ix
    }

    #[inline]
    fn sample_clamped(&self, ix: isize, iy: isize, iz: isize) -> f64 {
        let x = ix.clamp(0, (self.dims[0] - 1) as isize) as usize;
        let y = iy.clamp(0, (self.dims[1] - 1) as isize) as usize;
        let z = iz.clamp(0, (self.dims[2] - 1) as isize) as usize;
        self.values[self.idx(x, y, z)]
    }

    #[inline]
    fn grid_aabb(&self) -> Aabb {
        let max = Vec3::new(
            self.origin.x() + (self.dims[0] - 1) as f64 * self.voxel_size,
            self.origin.y() + (self.dims[1] - 1) as f64 * self.voxel_size,
            self.origin.z() + (self.dims[2] - 1) as f64 * self.voxel_size,
        );
        Aabb::new(self.origin, max)
    }

    fn eval_trilinear(&self, gx: f64, gy: f64, gz: f64) -> (f64, Vec3) {
        let ix = (gx.floor() as usize).min(self.dims[0] - 2);
        let iy = (gy.floor() as usize).min(self.dims[1] - 2);
        let iz = (gz.floor() as usize).min(self.dims[2] - 2);
        let u = gx - ix as f64;
        let v = gy - iy as f64;
        let w = gz - iz as f64;

        let c000 = self.values[self.idx(ix, iy, iz)];
        let c100 = self.values[self.idx(ix + 1, iy, iz)];
        let c010 = self.values[self.idx(ix, iy + 1, iz)];
        let c110 = self.values[self.idx(ix + 1, iy + 1, iz)];
        let c001 = self.values[self.idx(ix, iy, iz + 1)];
        let c101 = self.values[self.idx(ix + 1, iy, iz + 1)];
        let c011 = self.values[self.idx(ix, iy + 1, iz + 1)];
        let c111 = self.values[self.idx(ix + 1, iy + 1, iz + 1)];

        let c00 = c000 * (1.0 - u) + c100 * u;
        let c10 = c010 * (1.0 - u) + c110 * u;
        let c01 = c001 * (1.0 - u) + c101 * u;
        let c11 = c011 * (1.0 - u) + c111 * u;

        let c0 = c00 * (1.0 - v) + c10 * v;
        let c1 = c01 * (1.0 - v) + c11 * v;
        let phi = c0 * (1.0 - w) + c1 * w;

        // Exact analytical trilinear gradient
        let dphi_du = ((c100 - c000) * (1.0 - v) + (c110 - c010) * v) * (1.0 - w)
            + ((c101 - c001) * (1.0 - v) + (c111 - c011) * v) * w;
        let dphi_dv = (c10 - c00) * (1.0 - w) + (c11 - c01) * w;
        let dphi_dw = c1 - c0;

        let inv_h = 1.0 / self.voxel_size;
        (phi, Vec3::new(dphi_du * inv_h, dphi_dv * inv_h, dphi_dw * inv_h))
    }

    fn eval_tricubic(&self, gx: f64, gy: f64, gz: f64) -> (f64, Vec3) {
        let ix = (gx.floor() as isize).clamp(0, (self.dims[0] - 2) as isize);
        let iy = (gy.floor() as isize).clamp(0, (self.dims[1] - 2) as isize);
        let iz = (gz.floor() as isize).clamp(0, (self.dims[2] - 2) as isize);
        let u = gx - ix as f64;
        let v = gy - iy as f64;
        let w = gz - iz as f64;

        let (wx, dwx) = catmull_rom_weights(u);
        let (wy, dwy) = catmull_rom_weights(v);
        let (wz, dwz) = catmull_rom_weights(w);

        let mut phi = 0.0;
        let mut du = 0.0;
        let mut dv = 0.0;
        let mut dw = 0.0;

        for kz in 0..4 {
            let oz = iz + kz as isize - 1;
            for ky in 0..4 {
                let oy = iy + ky as isize - 1;
                let wyz = wy[ky] * wz[kz];
                let dwy_z = dwy[ky] * wz[kz];
                let wy_dz = wy[ky] * dwz[kz];
                for kx in 0..4 {
                    let ox = ix + kx as isize - 1;
                    let val = self.sample_clamped(ox, oy, oz);
                    phi += val * wx[kx] * wyz;
                    du += val * dwx[kx] * wyz;
                    dv += val * wx[kx] * dwy_z;
                    dw += val * wx[kx] * wy_dz;
                }
            }
        }

        let inv_h = 1.0 / self.voxel_size;
        (phi, Vec3::new(du * inv_h, dv * inv_h, dw * inv_h))
    }
}

/// Uniform Catmull-Rom 1D interpolation weights and derivatives for `t in [0, 1]`.
#[inline]
fn catmull_rom_weights(t: f64) -> ([f64; 4], [f64; 4]) {
    let t2 = t * t;
    let t3 = t2 * t;
    let w = [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ];
    let dw = [
        -1.5 * t2 + 2.0 * t - 0.5,
        4.5 * t2 - 5.0 * t,
        -4.5 * t2 + 4.0 * t + 0.5,
        1.5 * t2 - t,
    ];
    (w, dw)
}

impl SdfField for VoxelGridSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let aabb = self.grid_aabb();
        let clamped_pos = Vec3::new(
            pos.x().clamp(aabb.min.x(), aabb.max.x()),
            pos.y().clamp(aabb.min.y(), aabb.max.y()),
            pos.z().clamp(aabb.min.z(), aabb.max.z()),
        );
        let outside = pos - clamped_pos;
        let out_dist = outside.norm();

        let gx = ((clamped_pos.x() - self.origin.x()) / self.voxel_size)
            .clamp(0.0, (self.dims[0] - 1) as f64);
        let gy = ((clamped_pos.y() - self.origin.y()) / self.voxel_size)
            .clamp(0.0, (self.dims[1] - 1) as f64);
        let gz = ((clamped_pos.z() - self.origin.z()) / self.voxel_size)
            .clamp(0.0, (self.dims[2] - 1) as f64);

        let (phi_in, grad) = match self.interp {
            VoxelInterp::Trilinear => self.eval_trilinear(gx, gy, gz),
            VoxelInterp::Tricubic => self.eval_tricubic(gx, gy, gz),
        };

        let mat_id = if let Some(ref mats) = self.mat_ids {
            let ix = (gx.round() as usize).min(self.dims[0] - 1);
            let iy = (gy.round() as usize).min(self.dims[1] - 1);
            let iz = (gz.round() as usize).min(self.dims[2] - 1);
            mats[self.idx(ix, iy, iz)]
        } else {
            self.default_mat_id
        };

        if out_dist > 1e-12 {
            // Smooth exterior extension beyond grid bounds.
            SdfQuery {
                phi: phi_in + out_dist,
                normal: outside / out_dist,
                arc_s: 0.0,
                tangent: Vec3::ZERO,
                mat_id,
            }
        } else {
            SdfQuery {
                phi: phi_in,
                normal: grad.normalized_or(Vec3::EZ),
                arc_s: 0.0,
                tangent: Vec3::ZERO,
                mat_id,
            }
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        Some(self.grid_aabb())
    }
}

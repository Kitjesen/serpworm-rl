//! BVH-accelerated 3D `SplineTubeSdf`: $C^1$ Catmull-Rom or $C^2$ clamped cubic B-spline
//! centerline with variable radius $r(s)$, circular or rounded-rectangular cross-sections
//! in a parallel-transported Bishop frame, smooth entry/exit funnel flares, and Newton
//! closest-point projection.

use crate::bvh::{Aabb, Bvh};
use crate::{SdfField, SdfQuery};
use fg_math::{parallel_transport, Vec3};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Centerline cubic spline family.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SplineKind {
    /// $C^1$ interpolating Catmull-Rom spline (`tension = 0.5` is standard uniform Catmull-Rom).
    CatmullRom { tension: f64 },
    /// $C^2$ clamped uniform cubic B-spline (approximating, strictly $C^2$ smooth everywhere).
    CubicBSpline,
}

/// Tube inner/outer radius variation along arc length `s`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "profile", rename_all = "snake_case")]
pub enum RadiusProfile {
    Constant(f64),
    LinearTaper {
        r_start: f64,
        r_end: f64,
    },
    CosineTaper {
        s_start: f64,
        s_end: f64,
        r_start: f64,
        r_end: f64,
    },
    /// Piecewise-linear `(arc_s, radius)` lookup table sorted by `arc_s`.
    PiecewiseSamples(Vec<(f64, f64)>),
}

impl RadiusProfile {
    /// Evaluate `(radius(s), dr_ds(s), d2r_ds2(s))` at arc length `s` in `[0, total_len]`.
    pub fn eval(&self, s: f64, total_len: f64) -> (f64, f64, f64) {
        match self {
            RadiusProfile::Constant(r) => (*r, 0.0, 0.0),
            RadiusProfile::LinearTaper { r_start, r_end } => {
                let l = total_len.max(1e-9);
                let u = (s / l).clamp(0.0, 1.0);
                let r = *r_start + (*r_end - *r_start) * u;
                let dr = if s > 0.0 && s < l {
                    (*r_end - *r_start) / l
                } else {
                    0.0
                };
                (r, dr, 0.0)
            }
            RadiusProfile::CosineTaper {
                s_start,
                s_end,
                r_start,
                r_end,
            } => {
                let span = (*s_end - *s_start).max(1e-9);
                if s <= *s_start {
                    (*r_start, 0.0, 0.0)
                } else if s >= *s_end {
                    (*r_end, 0.0, 0.0)
                } else {
                    let u = (s - *s_start) / span;
                    let k = PI / span;
                    let r = *r_start + 0.5 * (*r_end - *r_start) * (1.0 - (PI * u).cos());
                    let dr = 0.5 * (*r_end - *r_start) * k * (PI * u).sin();
                    let d2r = 0.5 * (*r_end - *r_start) * k * k * (PI * u).cos();
                    (r, dr, d2r)
                }
            }
            RadiusProfile::PiecewiseSamples(samples) => {
                if samples.is_empty() {
                    return (0.1, 0.0, 0.0);
                }
                if s <= samples[0].0 {
                    return (samples[0].1, 0.0, 0.0);
                }
                for w in samples.windows(2) {
                    if s <= w[1].0 {
                        let ds = (w[1].0 - w[0].0).max(1e-9);
                        let t = (s - w[0].0) / ds;
                        let r = w[0].1 + t * (w[1].1 - w[0].1);
                        let dr = (w[1].1 - w[0].1) / ds;
                        return (r, dr, 0.0);
                    }
                }
                (samples[samples.len() - 1].1, 0.0, 0.0)
            }
        }
    }

    pub fn max_radius(&self) -> f64 {
        match self {
            RadiusProfile::Constant(r) => *r,
            RadiusProfile::LinearTaper { r_start, r_end } => r_start.max(*r_end),
            RadiusProfile::CosineTaper { r_start, r_end, .. } => r_start.max(*r_end),
            RadiusProfile::PiecewiseSamples(samples) => samples
                .iter()
                .map(|&(_, r)| r)
                .fold(0.0_f64, f64::max),
        }
    }
}

/// Cross-sectional shape of the tube normal to the spline tangent.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum CrossSection {
    /// Circular lumen of radius `r(s)`.
    Circular,
    /// Rounded-rectangular duct in the Bishop `(u, v)` frame (`half_width` along `u`,
    /// `half_height` along `v`, inner corner fillet `corner_radius`).
    RoundedRect {
        half_width: f64,
        half_height: f64,
        corner_radius: f64,
    },
}

/// Whether the tube represents an internal navigable cavity (pipe/lumen) or solid obstacle wire.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TubeWallMode {
    /// Free space is INSIDE the tube (`phi > 0` inside lumen, `normal` points inward to centerline).
    InteriorCavity,
    /// Solid cable/obstacle (`phi > 0` outside tube, `normal` points outward).
    SolidExterior,
}

/// Smooth entry and exit funnel flares at the open mouths of a tube.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunnelConfig {
    pub entry_len: f64,
    pub entry_flare: f64,
    pub exit_len: f64,
    pub exit_flare: f64,
}

impl FunnelConfig {
    pub const NONE: Self = Self {
        entry_len: 0.0,
        entry_flare: 0.0,
        exit_len: 0.0,
        exit_flare: 0.0,
    };

    /// Standard SerpWorm acrylic pipe entry funnel (`len = 0.08 m`, `flare = 0.055 m`).
    pub const SERPWORM_ENTRY: Self = Self {
        entry_len: 0.08,
        entry_flare: 0.055,
        exit_len: 0.0,
        exit_flare: 0.0,
    };

    /// Evaluate `(delta_r, d_delta_r/ds, d2_delta_r/ds2)` for entry/exit funnel flares at arc length `s`.
    #[inline]
    pub fn eval(&self, s: f64, total_len: f64) -> (f64, f64, f64) {
        let mut dr_val = 0.0;
        let mut dr_ds = 0.0;
        let mut d2r_ds2 = 0.0;
        if self.entry_len > 1e-9 && self.entry_flare > 0.0 && s < self.entry_len {
            let u = (s / self.entry_len).clamp(0.0, 1.0);
            let k = PI / self.entry_len;
            // Smooth C1 cosine flare from `entry_flare` at s=0 to `0` at s=entry_len.
            dr_val += 0.5 * self.entry_flare * (1.0 + (PI * u).cos());
            if s > 0.0 {
                dr_ds += -0.5 * self.entry_flare * k * (PI * u).sin();
                d2r_ds2 += -0.5 * self.entry_flare * k * k * (PI * u).cos();
            }
        }
        if self.exit_len > 1e-9 && self.exit_flare > 0.0 && s > total_len - self.exit_len {
            let u = ((total_len - s) / self.exit_len).clamp(0.0, 1.0);
            let k = PI / self.exit_len;
            dr_val += 0.5 * self.exit_flare * (1.0 + (PI * u).cos());
            if s < total_len {
                dr_ds += 0.5 * self.exit_flare * k * (PI * u).sin();
                d2r_ds2 += -0.5 * self.exit_flare * k * k * (PI * u).cos();
            }
        }
        (dr_val, dr_ds, d2r_ds2)
    }

    #[inline]
    pub fn max_flare(&self) -> f64 {
        self.entry_flare.max(self.exit_flare).max(0.0)
    }
}

#[derive(Clone, Debug)]
struct SamplePoint {
    u: f64,
    pos: Vec3,
    tangent: Vec3,
    bishop_u: Vec3,
    arc_s: f64,
}

/// BVH-accelerated 3D spline tube SDF (`C1` Catmull-Rom / `C2` cubic B-spline).
#[derive(Clone, Debug)]
pub struct SplineTubeSdf {
    control_points: Vec<Vec3>,
    kind: SplineKind,
    radius_profile: RadiusProfile,
    cross_section: CrossSection,
    wall_mode: TubeWallMode,
    funnel: FunnelConfig,
    mat_id: usize,
    samples: Vec<SamplePoint>,
    bvh: Bvh,
    total_len: f64,
    max_extent: f64,
}

impl SplineTubeSdf {
    /// Construct a 3D BVH-accelerated spline tube from control points.
    ///
    /// At least 2 control points are required. `n_samples` controls the BVH discretization
    /// density (default 64-128); exact spline closest points are refined via Newton iterations.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        control_points: Vec<Vec3>,
        kind: SplineKind,
        radius_profile: RadiusProfile,
        cross_section: CrossSection,
        wall_mode: TubeWallMode,
        funnel: FunnelConfig,
        mat_id: usize,
        n_samples: usize,
    ) -> Self {
        assert!(
            control_points.len() >= 2,
            "SplineTubeSdf requires at least 2 control points"
        );
        let n_segs = n_samples.max(8);

        let max_cross = match cross_section {
            CrossSection::Circular => radius_profile.max_radius() + funnel.max_flare(),
            CrossSection::RoundedRect {
                half_width,
                half_height,
                ..
            } => half_width.hypot(half_height) + funnel.max_flare(),
        };

        let mut samples = Vec::with_capacity(n_segs + 1);
        let mut arc_s = 0.0;
        let mut prev_pos = eval_spline_raw(&control_points, kind, 0.0).0;
        let t0 = eval_spline_raw(&control_points, kind, 0.0)
            .1
            .normalized_or(Vec3::EX);
        let mut prev_t = t0;
        // Initialize Bishop u orthogonal to t0 (prefer world +Z aligned v so rectangular ducts stay upright).
        let mut prev_u = if t0.z().abs() < 0.95 {
            Vec3::EZ.cross(t0).normalized()
        } else {
            Vec3::EY.cross(t0).normalized()
        };

        for i in 0..=n_segs {
            let u = (i as f64) / (n_segs as f64);
            let (pos, d1, _) = eval_spline_raw(&control_points, kind, u);
            if i > 0 {
                arc_s += (pos - prev_pos).norm();
            }
            let tangent = d1.normalized_or(prev_t);
            let bishop_u = if i == 0 {
                prev_u
            } else {
                let transported = parallel_transport(prev_u, prev_t, tangent);
                (transported - tangent * transported.dot(tangent)).normalized_or(prev_u)
            };
            samples.push(SamplePoint {
                u,
                pos,
                tangent,
                bishop_u,
                arc_s,
            });
            prev_pos = pos;
            prev_t = tangent;
            prev_u = bishop_u;
        }

        let pad = max_cross * 1.1 + 0.02;
        let mut seg_aabbs = Vec::with_capacity(n_segs);
        for i in 0..n_segs {
            seg_aabbs.push(Aabb::from_segment(samples[i].pos, samples[i + 1].pos, pad));
        }
        let bvh = Bvh::build(&seg_aabbs);

        Self {
            control_points,
            kind,
            radius_profile,
            cross_section,
            wall_mode,
            funnel,
            mat_id,
            samples,
            bvh,
            total_len: arc_s.max(1e-9),
            max_extent: max_cross,
        }
    }

    /// Convenience constructor for a 90-degree smooth curved circular pipe in XY
    /// (aligned with `pipe_geometry.py::make_bend90_pipe`).
    #[allow(clippy::too_many_arguments)]
    pub fn bend90_pipe(
        x_entry: f64,
        x_bend: f64,
        r_bend: f64,
        r_in: f64,
        bend_sgn: f64,
        l_exit: f64,
        y0: f64,
        z_center: f64,
        mat_id: usize,
    ) -> Self {
        let sgn = if bend_sgn >= 0.0 { 1.0 } else { -1.0 };
        let cy = y0 + sgn * r_bend;
        let mut pts = vec![
            Vec3::new(x_entry, y0, z_center),
            Vec3::new(0.5 * (x_entry + x_bend), y0, z_center),
            Vec3::new(x_bend, y0, z_center),
        ];
        for k in 1..=6 {
            let a = (k as f64 / 6.0) * 0.5 * PI;
            pts.push(Vec3::new(
                x_bend - r_bend * a.sin(),
                cy - sgn * r_bend * a.cos(),
                z_center,
            ));
        }
        pts.push(Vec3::new(x_bend - r_bend, cy + sgn * 0.5 * l_exit, z_center));
        pts.push(Vec3::new(x_bend - r_bend, cy + sgn * l_exit, z_center));
        Self::new(
            pts,
            SplineKind::CatmullRom { tension: 0.5 },
            RadiusProfile::Constant(r_in),
            CrossSection::Circular,
            TubeWallMode::InteriorCavity,
            FunnelConfig::SERPWORM_ENTRY,
            mat_id,
            80,
        )
    }

    /// Convenience constructor for a 3D curved endoscope lumen (e.g. for `TendonEndoscope-3D`).
    pub fn curved_endoscope_3d(radius: f64, mat_id: usize) -> Self {
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.06, 0.0, 0.0),
            Vec3::new(0.12, 0.025, 0.015),
            Vec3::new(0.17, 0.075, 0.045),
            Vec3::new(0.20, 0.135, 0.080),
            Vec3::new(0.21, 0.200, 0.105),
        ];
        Self::new(
            pts,
            SplineKind::CubicBSpline,
            RadiusProfile::Constant(radius),
            CrossSection::Circular,
            TubeWallMode::InteriorCavity,
            FunnelConfig::NONE,
            mat_id,
            72,
        )
    }

    #[inline]
    pub fn total_length(&self) -> f64 {
        self.total_len
    }

    /// Evaluate centerline position at normalized parameter `u in [0, 1]`.
    #[inline]
    pub fn eval_centerline(&self, u: f64) -> Vec3 {
        eval_spline_raw(&self.control_points, self.kind, u.clamp(0.0, 1.0)).0
    }

    /// Evaluate unit centerline tangent at normalized parameter `u in [0, 1]`.
    #[inline]
    pub fn eval_tangent(&self, u: f64) -> Vec3 {
        let (_, d1, _) = eval_spline_raw(&self.control_points, self.kind, u.clamp(0.0, 1.0));
        d1.normalized_or(Vec3::EX)
    }

    /// Interpolate arc length and Bishop frame `(arc_s, bishop_u, bishop_v)` at parameter `u`.
    fn interp_frame(&self, u: f64, tangent: Vec3) -> (f64, Vec3, Vec3) {
        let n_segs = self.samples.len() - 1;
        let scaled = u.clamp(0.0, 1.0) * (n_segs as f64);
        let idx = (scaled.floor() as usize).min(n_segs - 1);
        let frac = scaled - idx as f64;
        let s0 = &self.samples[idx];
        let s1 = &self.samples[idx + 1];
        let arc_s = s0.arc_s + frac * (s1.arc_s - s0.arc_s);
        let u_interp = parallel_transport(s0.bishop_u, s0.tangent, tangent);
        let bishop_u = (u_interp - tangent * u_interp.dot(tangent)).normalized_or(s0.bishop_u);
        let bishop_v = tangent.cross(bishop_u).normalized();
        (arc_s, bishop_u, bishop_v)
    }

    /// Find closest parameter `u* in [0, 1]` on the 3D cubic spline via BVH broadphase + Newton polish.
    /// Returns `(u, pos, unit_tangent, d1, d2)`.
    fn closest_param(&self, pos: Vec3) -> (f64, Vec3, Vec3, Vec3, Vec3) {
        let mut best_d2 = f64::INFINITY;
        let mut best_u = 0.0_f64;

        self.bvh.visit_nearest(pos, f64::INFINITY, |seg_idx, bound_d2| {
            let a = self.samples[seg_idx].pos;
            let b = self.samples[seg_idx + 1].pos;
            let ab = b - a;
            let denom = ab.norm_sq();
            let t = if denom > 1e-24 {
                ((pos - a).dot(ab) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = a + ab * t;
            let d2 = (pos - q).norm_sq();
            if d2 < best_d2 {
                best_d2 = d2;
                best_u = self.samples[seg_idx].u
                    + t * (self.samples[seg_idx + 1].u - self.samples[seg_idx].u);
                *bound_d2 = best_d2;
            }
        });

        // Guarded Newton-Raphson refinement on f(u) = 0.5 ||gamma(u) - pos||^2.
        let mut u = best_u;
        let (mut c, mut d1, mut d2) = eval_spline_raw(&self.control_points, self.kind, u);
        let mut cur_dist_sq = (c - pos).norm_sq();

        for _ in 0..8 {
            let diff = c - pos;
            let g = diff.dot(d1);
            if g.abs() < 1e-14 {
                break;
            }
            let h_gn = d1.norm_sq();
            let h_full = h_gn + diff.dot(d2);
            let h = if h_full > 0.25 * h_gn { h_full } else { h_gn }.max(1e-15);
            let step = (g / h).clamp(-0.15, 0.15);
            let mut alpha = 1.0;
            let mut accepted = false;
            for _ in 0..5 {
                let u_try = (u - alpha * step).clamp(0.0, 1.0);
                let (c_try, d1_try, d2_try) =
                    eval_spline_raw(&self.control_points, self.kind, u_try);
                let dist_sq_try = (c_try - pos).norm_sq();
                if dist_sq_try <= cur_dist_sq {
                    let du = (u_try - u).abs();
                    u = u_try;
                    c = c_try;
                    d1 = d1_try;
                    d2 = d2_try;
                    cur_dist_sq = dist_sq_try;
                    accepted = true;
                    if du < 1e-13 {
                        return (u, c, d1.normalized_or(Vec3::EX), d1, d2);
                    }
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                break;
            }
        }

        (u, c, d1.normalized_or(Vec3::EX), d1, d2)
    }
}

impl SdfField for SplineTubeSdf {
    fn query(&self, pos: Vec3) -> SdfQuery {
        let (u, c, tangent, d1, d2) = self.closest_param(pos);
        let (arc_s, bishop_u, bishop_v) = self.interp_frame(u, tangent);

        let diff = pos - c;
        let axial = diff.dot(tangent);

        // For an open InteriorCavity pipe, points strictly beyond the open entry/exit mouths
        // are in open free space (do not trap the robot with invisible end-caps!).
        if self.wall_mode == TubeWallMode::InteriorCavity
            && ((u <= 1e-12 && axial < 0.0) || (u >= 1.0 - 1e-12 && axial > 0.0))
        {
            return SdfQuery {
                phi: axial.abs() + self.max_extent,
                normal: if axial < 0.0 { -tangent } else { tangent },
                arc_s: if axial < 0.0 {
                    axial
                } else {
                    self.total_len + axial
                },
                tangent,
                mat_id: self.mat_id,
            };
        }

        let (r_base, dr_base, d2r_base) = self.radius_profile.eval(arc_s, self.total_len);
        let (r_fun, dr_fun, d2r_fun) = self.funnel.eval(arc_s, self.total_len);
        let r_loc = (r_base + r_fun).max(1e-4);
        let dr_ds = dr_base + dr_fun;
        let d2r_ds2 = d2r_base + d2r_fun;

        // Exact curvature-corrected arc-length gradient factor:
        // From (c(u) - p)·d1(u) = 0 => grad_p u = d1 / (d1·d1 + (c - p)·d2),
        // and ds/du = ||d1|| => grad_p s = (||d1||^2 / (d1·d1 + (c - p)·d2)) * tangent.
        let d1_sq = d1.norm_sq();
        let denom_s = (d1_sq - diff.dot(d2)).max(0.1 * d1_sq).max(1e-15);
        let alpha_s = d1_sq / denom_s;

        let (phi, normal) = match self.cross_section {
            CrossSection::Circular => {
                // Radial vector perpendicular to centerline (for SolidExterior beyond caps, use full diff).
                let radial = if self.wall_mode == TubeWallMode::InteriorCavity {
                    diff - tangent * axial
                } else {
                    diff
                };
                let rho = radial.norm();
                let e_rho = if rho > 1e-15 {
                    radial / rho
                } else {
                    bishop_u
                };
                let slope_sq = 1.0 + dr_ds * dr_ds;
                let inv_slope = 1.0 / slope_sq.sqrt();
                let d_inv_slope_ds = -(dr_ds * d2r_ds2) / (slope_sq * slope_sq.sqrt());
                match self.wall_mode {
                    TubeWallMode::InteriorCavity => {
                        let raw = r_loc - rho;
                        let phi_in = raw * inv_slope;
                        let coeff_t = (dr_ds * inv_slope + raw * d_inv_slope_ds) * alpha_s;
                        let grad_in = tangent * coeff_t - e_rho * inv_slope;
                        (phi_in, grad_in.normalized_or(-e_rho))
                    }
                    TubeWallMode::SolidExterior => {
                        let raw = rho - r_loc;
                        let phi_out = raw * inv_slope;
                        let coeff_t = (-dr_ds * inv_slope + raw * d_inv_slope_ds) * alpha_s;
                        let grad_out = e_rho * inv_slope + tangent * coeff_t;
                        (phi_out, grad_out.normalized_or(e_rho))
                    }
                }
            }
            CrossSection::RoundedRect {
                half_width,
                half_height,
                corner_radius,
            } => {
                let hw = (half_width + r_fun).max(1e-4);
                let hh = (half_height + r_fun).max(1e-4);
                let rc = corner_radius.clamp(0.0, hw.min(hh) - 1e-5);
                let bx = (hw - rc).max(1e-6);
                let by = (hh - rc).max(1e-6);

                let lx = diff.dot(bishop_u);
                let ly = diff.dot(bishop_v);
                let qx = lx.abs() - bx;
                let qy = ly.abs() - by;
                let ox = qx.max(0.0);
                let oy = qy.max(0.0);
                let out2 = ox * ox + oy * oy;

                // Signed distance to the rounded-rectangle boundary (negative inside, positive outside).
                let (d_rect, n2_x, n2_y) = if out2 > 0.0 {
                    let out = out2.sqrt();
                    let sx = if lx >= 0.0 { 1.0 } else { -1.0 };
                    let sy = if ly >= 0.0 { 1.0 } else { -1.0 };
                    (out - rc, sx * ox / out, sy * oy / out)
                } else if qx >= qy {
                    (qx - rc, if lx >= 0.0 { 1.0 } else { -1.0 }, 0.0)
                } else {
                    (qy - rc, 0.0, if ly >= 0.0 { 1.0 } else { -1.0 })
                };

                let n_out_3d = bishop_u * n2_x + bishop_v * n2_y;
                let slope_sq = 1.0 + dr_fun * dr_fun;
                let inv_slope = 1.0 / slope_sq.sqrt();
                let d_inv_slope_ds = -(dr_fun * d2r_fun) / (slope_sq * slope_sq.sqrt());
                match self.wall_mode {
                    TubeWallMode::InteriorCavity => {
                        let raw = -d_rect;
                        let phi_in = raw * inv_slope;
                        let coeff_t = (dr_fun * inv_slope + raw * d_inv_slope_ds) * alpha_s;
                        let grad_in = tangent * coeff_t - n_out_3d * inv_slope;
                        (phi_in, grad_in.normalized_or(-n_out_3d))
                    }
                    TubeWallMode::SolidExterior => {
                        let raw = d_rect;
                        let phi_out = raw * inv_slope;
                        let coeff_t = (-dr_fun * inv_slope + raw * d_inv_slope_ds) * alpha_s;
                        let grad_out = n_out_3d * inv_slope + tangent * coeff_t;
                        (phi_out, grad_out.normalized_or(n_out_3d))
                    }
                }
            }
        };

        SdfQuery {
            phi,
            normal,
            arc_s,
            tangent,
            mat_id: self.mat_id,
        }
    }

    fn bounds(&self) -> Option<Aabb> {
        match self.wall_mode {
            // An interior cavity represents solid material outside the lumen (unbounded exterior),
            // matching `SphereSdf` and `CylinderSdf` inverted semantics so WorldSdf never prunes
            // deep wall penetrations outside the lumen AABB.
            TubeWallMode::InteriorCavity => None,
            TubeWallMode::SolidExterior => self.bvh.root_bounds(),
        }
    }
}

/// Evaluate `(pos, dpos/du, d2pos/du2)` on a cubic Catmull-Rom or clamped cubic B-spline
/// for global parameter `u in [0, 1]`.
fn eval_spline_raw(pts: &[Vec3], kind: SplineKind, u: f64) -> (Vec3, Vec3, Vec3) {
    let n = pts.len();
    if n == 2 {
        let p = pts[0] + (pts[1] - pts[0]) * u;
        let d1 = pts[1] - pts[0];
        return (p, d1, Vec3::ZERO);
    }

    let n_spans = (n - 1) as f64;
    let s = u.clamp(0.0, 1.0) * n_spans;
    let span = (s.floor() as usize).min(n - 2);
    let t = s - span as f64;

    // Phantom reflected endpoints for clamped boundary tangents.
    let get_pt = |idx: isize| -> Vec3 {
        if idx < 0 {
            pts[0] * 2.0 - pts[1]
        } else if idx >= n as isize {
            pts[n - 1] * 2.0 - pts[n - 2]
        } else {
            pts[idx as usize]
        }
    };

    let p0 = get_pt(span as isize - 1);
    let p1 = get_pt(span as isize);
    let p2 = get_pt(span as isize + 1);
    let p3 = get_pt(span as isize + 2);

    let t2 = t * t;
    let t3 = t2 * t;

    let (pos, d1_t, d2_t) = match kind {
        SplineKind::CatmullRom { tension } => {
            let tau = tension.clamp(0.0, 1.0);
            let m1 = (p2 - p0) * tau;
            let m2 = (p3 - p1) * tau;
            // Cubic Hermite basis
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;

            let dh00 = 6.0 * t2 - 6.0 * t;
            let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
            let dh01 = -6.0 * t2 + 6.0 * t;
            let dh11 = 3.0 * t2 - 2.0 * t;

            let d2h00 = 12.0 * t - 6.0;
            let d2h10 = 6.0 * t - 4.0;
            let d2h01 = -12.0 * t + 6.0;
            let d2h11 = 6.0 * t - 2.0;

            let pos = p1 * h00 + m1 * h10 + p2 * h01 + m2 * h11;
            let d1 = p1 * dh00 + m1 * dh10 + p2 * dh01 + m2 * dh11;
            let d2 = p1 * d2h00 + m1 * d2h10 + p2 * d2h01 + m2 * d2h11;
            (pos, d1, d2)
        }
        SplineKind::CubicBSpline => {
            // Uniform cubic B-spline basis (strictly C2 across all knots).
            let b0 = ((1.0 - t).powi(3)) / 6.0;
            let b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
            let b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
            let b3 = t3 / 6.0;

            let db0 = -0.5 * (1.0 - t).powi(2);
            let db1 = 1.5 * t2 - 2.0 * t;
            let db2 = -1.5 * t2 + t + 0.5;
            let db3 = 0.5 * t2;

            let d2b0 = 1.0 - t;
            let d2b1 = 3.0 * t - 2.0;
            let d2b2 = -3.0 * t + 1.0;
            let d2b3 = t;

            let pos = p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
            let d1 = p0 * db0 + p1 * db1 + p2 * db2 + p3 * db3;
            let d2 = p0 * d2b0 + p1 * d2b1 + p2 * d2b2 + p3 * d2b3;
            (pos, d1, d2)
        }
    };

    // Chain rule from local span parameter `t` to global `u`: dt/du = n_spans.
    (pos, d1_t * n_spans, d2_t * (n_spans * n_spans))
}

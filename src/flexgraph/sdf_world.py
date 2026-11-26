"""Unified Composable Signed Distance Field (SDF) Environment Abstraction.

Single source of truth for ground terrain, 3D curved/variable-radius spline tubes,
analytical obstacles, and voxelized CAD/CT SDFs. Every query returns:
    SDFQuery(phi, normal, arc_s, tangent, mat_id)
where:
  - ``phi < 0`` indicates penetration into solid environment boundary
  - ``normal`` is the outward unit normal pointing into free space
  - ``arc_s`` and ``tangent`` provide curvilinear progression along tubes/corridors
  - ``mat_id`` indexes per-material terrain/wall physics properties (Hunt-Crossley / Bekker)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import NamedTuple

import torch
from torch.nn import functional as F

from flexgraph.math3d import norm, unit


class SDFQuery(NamedTuple):
    phi: torch.Tensor       # (B, M) signed distance (>0 free, <0 penetrating)
    normal: torch.Tensor    # (B, M, 3) outward unit normal into free space
    arc_s: torch.Tensor     # (B, M) curvilinear progress along primary guide/tube (m)
    tangent: torch.Tensor   # (B, M, 3) local unit travel tangent
    mat_id: torch.Tensor    # (B, M) int64 material profile index


@dataclass
class MaterialProfile:
    name: str
    hc_k: float = 1.6e5
    hc_alpha: float = 3.5
    soil_exp: float = 1.5
    unload_ratio: float = 1.0      # >1.0 enables Bekker elasto-plastic rutting
    rut_relax_m: float = 0.025
    rut_lat_k: float = 400.0
    mu_scale: float = 1.0
    rolling_override: float | None = None
    linear_kn: float = 5000.0
    linear_cn: float = 8.0


def default_material_library() -> list[MaterialProfile]:
    return [
        MaterialProfile('hard_floor', hc_k=1.6e5, hc_alpha=3.5, soil_exp=1.5, unload_ratio=1.0),
        MaterialProfile('acrylic_wall', hc_k=8.0e4, hc_alpha=2.5, soil_exp=1.2, unload_ratio=1.0, mu_scale=0.225, linear_kn=3200.0, linear_cn=18.0),
        MaterialProfile('soft_soil', hc_k=3.2e4, hc_alpha=4.5, soil_exp=1.4, unload_ratio=3.5, rut_relax_m=0.025, rut_lat_k=450.0, mu_scale=0.85, rolling_override=0.11),
        MaterialProfile('carpet', hc_k=6.5e4, hc_alpha=4.0, soil_exp=1.5, unload_ratio=1.2, mu_scale=1.10, rolling_override=0.095),
    ]


class SDFPrimitive:
    """Base interface for batched FP64 SDF environment primitives."""

    def query(self, pos: torch.Tensor) -> SDFQuery:
        raise NotImplementedError


class HeightFieldSDF(SDFPrimitive):
    """Exact sloped + sinusoidal undulating ground plane matching GPUCore.ground_height_and_normal."""

    def __init__(
        self,
        slope_x: float = 0.0,
        slope_y: float = 0.0,
        wave_amp_m: float = 0.0,
        wave_kx: float = 0.0,
        wave_ky: float = 0.0,
        z_offset_m: float = 0.0,
        mat_id: int = 0,
    ):
        self.slope_x = float(slope_x)
        self.slope_y = float(slope_y)
        self.wave_amp_m = float(wave_amp_m)
        self.wave_kx = float(wave_kx)
        self.wave_ky = float(wave_ky)
        self.z_offset_m = float(z_offset_m)
        self.mat_id = int(mat_id)

    def query(self, pos: torch.Tensor) -> SDFQuery:
        wx, wy, wz = pos[..., 0], pos[..., 1], pos[..., 2]
        sx, sy, amp, kx, ky = self.slope_x, self.slope_y, self.wave_amp_m, self.wave_kx, self.wave_ky
        if amp != 0.0:
            s_x, c_x = torch.sin(kx * wx), torch.cos(kx * wx)
            s_y, c_y = torch.sin(ky * wy), torch.cos(ky * wy)
            z_g = self.z_offset_m + sx * wx + sy * wy + amp * s_x * c_y
            dz_dx = sx + amp * kx * c_x * c_y
            dz_dy = sy - amp * ky * s_x * s_y
        else:
            z_g = self.z_offset_m + sx * wx + sy * wy
            dz_dx = torch.full_like(wx, sx)
            dz_dy = torch.full_like(wy, sy)
        up = unit(torch.stack((-dz_dx, -dz_dy, torch.ones_like(z_g)), dim=-1))
        phi = (wz - z_g) * up[..., 2]
        tangent = torch.zeros_like(pos)
        tangent[..., 0] = -1.0
        arc_s = -wx
        mat = torch.full(pos.shape[:-1], self.mat_id, device=pos.device, dtype=torch.long)
        return SDFQuery(phi=phi, normal=up, arc_s=arc_s, tangent=tangent, mat_id=mat)


class SplineTubeSDF(SDFPrimitive):
    """Generalized 3D variable-radius curved tube SDF from polyline/cubic-spline centerline.

    Subsumes straight cylinders, 90-deg arcs, S-curves, 180-deg U-turns, tapers, 3D helices,
    and vascular/industrial ducts via a single batched projection onto densely sampled spline segments.
    Inside the tube is free space (phi > 0); wall boundary is phi = 0; penetrating wall is phi < 0.
    """

    def __init__(
        self,
        control_points: list[tuple[float, float, float]] | torch.Tensor,
        radii_m: list[float] | float,
        samples_per_span: int = 16,
        funnel_len_m: float = 0.08,
        funnel_flare_m: float = 0.055,
        mat_id: int = 1,
        active_window_only: bool = True,
    ):
        pts = torch.as_tensor(control_points, dtype=torch.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
            raise ValueError('control_points must have shape (K >= 2, 3)')
        if isinstance(radii_m, (int, float)):
            r_ctrl = torch.full((pts.shape[0],), float(radii_m), dtype=torch.float64)
        else:
            r_ctrl = torch.as_tensor(radii_m, dtype=torch.float64)
            if r_ctrl.shape != (pts.shape[0],):
                raise ValueError('radii_m length must match control_points')

        # Centripetal Catmull-Rom interpolation for C1-smooth 3D centerline
        sampled_p = []
        sampled_r = []
        n_ctrl = pts.shape[0]
        for i in range(n_ctrl - 1):
            p0 = pts[max(i - 1, 0)]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[min(i + 2, n_ctrl - 1)]
            r1, r2 = r_ctrl[i], r_ctrl[i + 1]
            n_sub = max(2, int(samples_per_span))
            ts = torch.linspace(0.0, 1.0, n_sub + 1, dtype=torch.float64)[:-1]
            for t in ts:
                t2 = t * t
                t3 = t2 * t
                pos_t = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                sampled_p.append(pos_t)
                sampled_r.append((1.0 - t) * r1 + t * r2)
        sampled_p.append(pts[-1])
        sampled_r.append(r_ctrl[-1])

        self._pts = torch.stack(sampled_p, dim=0)       # (N_s, 3)
        self._r = torch.stack(sampled_r, dim=0)         # (N_s,)
        diffs = self._pts[1:] - self._pts[:-1]          # (N_s-1, 3)
        seg_lens = torch.linalg.vector_norm(diffs, dim=-1).clamp_min(1e-12)
        self._seg_a = self._pts[:-1]
        self._seg_ab = diffs
        self._seg_len2 = (diffs * diffs).sum(-1).clamp_min(1e-24)
        self._seg_tan = diffs / seg_lens[:, None]
        arc = torch.zeros(self._pts.shape[0], dtype=torch.float64)
        arc[1:] = torch.cumsum(seg_lens, dim=0)
        self._arc = arc
        self._total_len = float(arc[-1].item())
        self.funnel_len_m = float(funnel_len_m)
        self.funnel_flare_m = float(funnel_flare_m)
        self.mat_id = int(mat_id)
        self.active_window_only = bool(active_window_only)

    def query(self, pos: torch.Tensor) -> SDFQuery:
        device, dtype = pos.device, pos.dtype
        seg_a = self._seg_a.to(device=device, dtype=dtype)
        seg_ab = self._seg_ab.to(device=device, dtype=dtype)
        seg_len2 = self._seg_len2.to(device=device, dtype=dtype)
        seg_tan = self._seg_tan.to(device=device, dtype=dtype)
        arc = self._arc.to(device=device, dtype=dtype)
        r_nodes = self._r.to(device=device, dtype=dtype)

        # pos: (B, M, 3) -> closest point over all K segments
        ap = pos[..., None, :] - seg_a[None, None, :, :]                    # (B, M, K, 3)
        u_raw = (ap * seg_ab[None, None, :, :]).sum(-1) / seg_len2[None, None, :]
        u = u_raw.clamp(0.0, 1.0)
        closest = seg_a[None, None, :, :] + u[..., None] * seg_ab[None, None, :, :]
        diff = pos[..., None, :] - closest                                  # (B, M, K, 3)
        dist2 = (diff * diff).sum(-1)
        idx = dist2.argmin(dim=-1)                                          # (B, M)

        # Gather winning segment quantities
        idx3 = idx[..., None, None].expand(*pos.shape[:-1], 1, 3)
        diff_best = diff.gather(-2, idx3).squeeze(-2)                       # (B, M, 3)
        u_best = u.gather(-1, idx[..., None]).squeeze(-1)                   # (B, M)
        tan_best = seg_tan[idx]                                             # (B, M, 3)
        arc_s = arc[idx] + u_best * (arc[idx + 1] - arc[idx])               # (B, M)
        r_local = (1.0 - u_best) * r_nodes[idx] + u_best * r_nodes[idx + 1]

        # Smooth entry/exit funnel flare
        if self.funnel_len_m > 1e-6 and self.funnel_flare_m > 0.0:
            s_in = (1.0 - (arc_s / self.funnel_len_m).clamp(0.0, 1.0))
            s_out = (1.0 - ((self._total_len - arc_s) / self.funnel_len_m).clamp(0.0, 1.0))
            flare = self.funnel_flare_m * (s_in * s_in + s_out * s_out)
            r_local = r_local + flare

        # Radial vector perpendicular to local centerline tangent
        radial_vec = diff_best - (diff_best * tan_best).sum(-1, keepdim=True) * tan_best
        r_dist = norm(radial_vec)[..., 0]
        # Outward normal points INTO free interior space (i.e. back toward centerline: -radial_vec)
        inward_n = -radial_vec / r_dist[..., None].clamp_min(1e-12)
        # Fallback arbitrary orthogonal normal right on exact centerline
        on_axis = r_dist < 1e-11
        if on_axis.any():
            fallback = torch.zeros_like(inward_n)
            fallback[..., 2] = 1.0
            inward_n = torch.where(on_axis[..., None], fallback, inward_n)

        phi = r_local - r_dist
        if self.active_window_only:
            # Outside tube ends (projected beyond start/end), free space phi = +1e3
            u_raw_best = u_raw.gather(-1, idx[..., None]).squeeze(-1)
            before_start = (idx == 0) & (u_raw_best < 0.0)
            after_end = (idx == (seg_a.shape[0] - 1)) & (u_raw_best > 1.0)
            outside = before_start | after_end
            phi = torch.where(outside, torch.full_like(phi, 1e3), phi)

        mat = torch.full(pos.shape[:-1], self.mat_id, device=device, dtype=torch.long)
        return SDFQuery(phi=phi, normal=inward_n, arc_s=arc_s, tangent=tan_best, mat_id=mat)


class BoxObstacleSDF(SDFPrimitive):
    """Axis-aligned solid box obstacle (outside is free phi > 0; inside solid is phi < 0)."""

    def __init__(self, center: tuple[float, float, float], half_extents: tuple[float, float, float], mat_id: int = 1):
        self.center = torch.tensor(center, dtype=torch.float64)
        self.half_extents = torch.tensor(half_extents, dtype=torch.float64)
        self.mat_id = int(mat_id)

    def query(self, pos: torch.Tensor) -> SDFQuery:
        c = self.center.to(device=pos.device, dtype=pos.dtype)
        b = self.half_extents.to(device=pos.device, dtype=pos.dtype)
        rel = pos - c
        q = rel.abs() - b
        outside_vec = q.clamp_min(0.0)
        out_dist = norm(outside_vec)[..., 0]
        in_dist = q.max(dim=-1).values.clamp_max(0.0)
        phi = out_dist + in_dist

        # Outward unit normal
        sgn = torch.where(rel >= 0.0, 1.0, -1.0)
        out_n = sgn * (outside_vec / out_dist[..., None].clamp_min(1e-12))
        max_axis = q.argmax(dim=-1)
        in_n = F.one_hot(max_axis, num_classes=3).to(dtype=pos.dtype) * sgn
        normal = torch.where((out_dist > 1e-12)[..., None], out_n, in_n)
        tangent = torch.zeros_like(pos)
        tangent[..., 0] = -1.0
        arc_s = -pos[..., 0]
        mat = torch.full(pos.shape[:-1], self.mat_id, device=pos.device, dtype=torch.long)
        return SDFQuery(phi=phi, normal=normal, arc_s=arc_s, tangent=tangent, mat_id=mat)


class VoxelGridSDF(SDFPrimitive):
    """3D trilinear-interpolated voxel SDF for arbitrary baked CAD / CT vascular meshes."""

    def __init__(
        self,
        grid_phi: torch.Tensor,  # (Nx, Ny, Nz) FP64 signed distances
        bounds_min: tuple[float, float, float],
        bounds_max: tuple[float, float, float],
        mat_id: int = 1,
    ):
        if grid_phi.ndim != 3:
            raise ValueError('grid_phi must be 3D tensor (Nx, Ny, Nz)')
        self._grid = grid_phi.to(dtype=torch.float64).contiguous()
        self._bmin = torch.tensor(bounds_min, dtype=torch.float64)
        self._bmax = torch.tensor(bounds_max, dtype=torch.float64)
        self.mat_id = int(mat_id)

    def _sample(self, pos: torch.Tensor) -> torch.Tensor:
        device, dtype = pos.device, pos.dtype
        # grid_sample expects (N, C, D, H, W) and coords in [-1, 1] ordered (x, y, z) -> (W, H, D)
        grid = self._grid.to(device=device, dtype=dtype).permute(2, 1, 0)[None, None]  # (1, 1, Nz, Ny, Nx)
        bmin = self._bmin.to(device=device, dtype=dtype)
        bmax = self._bmax.to(device=device, dtype=dtype)
        norm_xyz = (2.0 * (pos - bmin) / (bmax - bmin).clamp_min(1e-12) - 1.0).clamp(-1.0, 1.0)
        B, M, _ = norm_xyz.shape
        g_in = norm_xyz.reshape(1, 1, B, M, 3)
        out = F.grid_sample(grid, g_in, mode='bilinear', padding_mode='border', align_corners=True)
        return out.reshape(B, M)

    def query(self, pos: torch.Tensor) -> SDFQuery:
        phi = self._sample(pos)
        eps = 1e-4
        ex = torch.tensor([eps, 0.0, 0.0], device=pos.device, dtype=pos.dtype)
        ey = torch.tensor([0.0, eps, 0.0], device=pos.device, dtype=pos.dtype)
        ez = torch.tensor([0.0, 0.0, eps], device=pos.device, dtype=pos.dtype)
        gx = (self._sample(pos + ex) - self._sample(pos - ex)) / (2.0 * eps)
        gy = (self._sample(pos + ey) - self._sample(pos - ey)) / (2.0 * eps)
        gz = (self._sample(pos + ez) - self._sample(pos - ez)) / (2.0 * eps)
        normal = unit(torch.stack((gx, gy, gz), dim=-1))
        tangent = torch.zeros_like(pos)
        tangent[..., 0] = -1.0
        arc_s = -pos[..., 0]
        mat = torch.full(pos.shape[:-1], self.mat_id, device=pos.device, dtype=torch.long)
        return SDFQuery(phi=phi, normal=normal, arc_s=arc_s, tangent=tangent, mat_id=mat)


@dataclass
class ComposableSDFWorld:
    """Boolean union of arbitrary SDF primitives + material property table.

    Evaluates ``min_i phi_i(x)`` (exact union of solid boundaries = intersection of free space)
    and tracks the winning primitive's outward normal, curvilinear arc progress, and material ID.
    """

    primitives: list[SDFPrimitive] = field(default_factory=list)
    materials: list[MaterialProfile] = field(default_factory=default_material_library)
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)

    def query(self, pos: torch.Tensor) -> SDFQuery:
        if not self.primitives:
            B, M, _ = pos.shape
            phi = torch.full((B, M), 1e3, device=pos.device, dtype=pos.dtype)
            normal = torch.zeros_like(pos)
            normal[..., 2] = 1.0
            tangent = torch.zeros_like(pos)
            tangent[..., 0] = -1.0
            return SDFQuery(
                phi=phi,
                normal=normal,
                arc_s=-pos[..., 0],
                tangent=tangent,
                mat_id=torch.zeros((B, M), device=pos.device, dtype=torch.long),
            )

        best = self.primitives[0].query(pos)
        if len(self.primitives) == 1:
            return best
        phi, normal, arc_s, tangent, mat_id = best
        for prim in self.primitives[1:]:
            q = prim.query(pos)
            take = q.phi < phi
            phi = torch.where(take, q.phi, phi)
            normal = torch.where(take[..., None], q.normal, normal)
            arc_s = torch.where(take, q.arc_s, arc_s)
            tangent = torch.where(take[..., None], q.tangent, tangent)
            mat_id = torch.where(take, q.mat_id, mat_id)
        return SDFQuery(phi=phi, normal=normal, arc_s=arc_s, tangent=tangent, mat_id=mat_id)

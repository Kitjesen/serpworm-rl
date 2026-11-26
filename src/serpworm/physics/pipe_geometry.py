"""Unified analytical geometry & C1-smooth contact for straight, 90-deg arc, S-curve, and sharp 90-deg L-duct pipes.

Single source of truth shared across:
  1. PyTorch ``GPUCore`` (CPU / ``segment_lu`` reference & exact FP64 parity verification)
  2. CUDA ``fused_physics.cu`` (bit-exact analytical mirror for 1024-env GPU RL)
  3. MuJoCo 3D transparent acrylic pipe / bend / L-duct renderer

``pipe_cfg`` tensor layout ``(B, 12)`` (FP64):
  [0]  pipe_type   : 0=off, 1=straight_cyl, 2=bend90_arc, 3=scurve_double_arc, 4=sharp90_L_duct,
                     5=u_turn_180, 6=taper_reducer, 7=double_L_chicane, 8=triple_wave_long
  [1]  y0          : entry centerline y (m)
  [2]  r_in        : inner radius (types 1..3,5,6,8) or half-width W/2 (types 4,7) (m)
  [3]  x_entry     : entry x coordinate along -X travel axis (m; +0.08 m smooth entry funnel)
  [4]  x_bend      : x_exit (type 1); bend/taper start x (types 2,3,5,6,8) or corner#1 center x (types 4,7) (m)
  [5]  R_bend      : centerline arc radius (types 2,3,5,8), L_taper (type 6), or L_mid between corners (type 7) (m)
  [6]  bend_angle  : bend sweep angle in rad (pi/2 for type 2, 50-75 deg for type 3/8, pi for type 5)
  [7]  bend_sgn    : +1.0 (left turn toward +Y) or -1.0 (right turn toward -Y)
  [8]  L_exit      : straight exit arm length (m)
  [9]  z_center    : centerline height (types 1..3,5,6,8, e.g. 0.118 m) or duct ceiling height H (types 4,7)
  [10] k_wall      : wall normal stiffness (N/m, e.g. 3200.0; 0 disables)
  [11] r_fillet    : inner corner acrylic fillet radius (types 4,7, m) or narrow exit radius r_out (type 6, m)
"""
from __future__ import annotations

import math
import numpy as np
import torch

__all__ = [
    'PIPE_TYPE_OFF',
    'PIPE_TYPE_STRAIGHT',
    'PIPE_TYPE_BEND90',
    'PIPE_TYPE_SCURVE',
    'PIPE_TYPE_L_DUCT',
    'PIPE_TYPE_UTURN',
    'PIPE_TYPE_TAPER',
    'PIPE_TYPE_Z_DUCT',
    'PIPE_TYPE_TRIPLE_WAVE',
    'PIPE_NAMES',
    'make_off_pipe',
    'make_straight_pipe',
    'make_bend90_pipe',
    'make_scurve_pipe',
    'make_right_angle_duct',
    'make_uturn_pipe',
    'make_taper_pipe',
    'make_z_duct',
    'make_triple_wave_pipe',
    'eval_pipe_contact_torch',
    'eval_pipe_membership_torch',
    'eval_pipe_arc_progress_torch',
    'pipe_arc_progress_np',
    'build_pipe_mujoco_geoms',
]

PIPE_TYPE_OFF = 0
PIPE_TYPE_STRAIGHT = 1
PIPE_TYPE_BEND90 = 2
PIPE_TYPE_SCURVE = 3
PIPE_TYPE_L_DUCT = 4
PIPE_TYPE_UTURN = 5
PIPE_TYPE_TAPER = 6
PIPE_TYPE_Z_DUCT = 7
PIPE_TYPE_TRIPLE_WAVE = 8

PIPE_NAMES = {
    PIPE_TYPE_OFF: 'open',
    PIPE_TYPE_STRAIGHT: 'straight',
    PIPE_TYPE_BEND90: 'bend90',
    PIPE_TYPE_SCURVE: 'scurve',
    PIPE_TYPE_L_DUCT: 'right_angle_L',
    PIPE_TYPE_UTURN: 'u_turn_180',
    PIPE_TYPE_TAPER: 'taper_reducer',
    PIPE_TYPE_Z_DUCT: 'double_L_Z',
    PIPE_TYPE_TRIPLE_WAVE: 'triple_wave',
}

_FUNNEL_LEN = 0.08
_FUNNEL_FLARE = 0.055
_C_WALL = 18.0
_MU_WALL = 0.18
_V_REG = 0.015


def make_off_pipe() -> list[float]:
    return [0.0, 0.0, 1e6, 1e6, -1e6, 0.50, 0.0, 1.0, 0.75, 0.118, 0.0, 0.008]


def make_straight_pipe(
    x_entry: float = -1.20,
    x_exit: float = -2.30,
    r_in: float = 0.122,
    y0: float = 0.0,
    z_center: float = 0.118,
    k_wall: float = 3200.0,
) -> list[float]:
    return [1.0, float(y0), float(r_in), float(x_entry), float(x_exit), 0.50, 0.0, 1.0, 0.0, float(z_center), float(k_wall), 0.008]


def make_bend90_pipe(
    x_entry: float = -1.10,
    x_bend: float = -1.55,
    R_bend: float = 0.50,
    r_in: float = 0.122,
    bend_sgn: float = +1.0,
    L_exit: float = 0.75,
    y0: float = 0.0,
    z_center: float = 0.118,
    k_wall: float = 3200.0,
) -> list[float]:
    return [
        2.0, float(y0), float(r_in), float(x_entry), float(x_bend),
        float(R_bend), 0.5 * math.pi, 1.0 if bend_sgn >= 0.0 else -1.0,
        float(L_exit), float(z_center), float(k_wall), 0.008,
    ]


def make_scurve_pipe(
    x_entry: float = -1.10,
    x_bend: float = -1.45,
    R_bend: float = 0.52,
    bend_angle_deg: float = 65.0,
    r_in: float = 0.122,
    bend_sgn: float = +1.0,
    L_exit: float = 0.65,
    y0: float = 0.0,
    z_center: float = 0.118,
    k_wall: float = 3200.0,
) -> list[float]:
    return [
        3.0, float(y0), float(r_in), float(x_entry), float(x_bend),
        float(R_bend), math.radians(float(bend_angle_deg)), 1.0 if bend_sgn >= 0.0 else -1.0,
        float(L_exit), float(z_center), float(k_wall), 0.008,
    ]


def make_right_angle_duct(
    x_entry: float = -1.10,
    x_corner: float = -1.68,
    width: float = 0.256,
    height: float = 0.245,
    bend_sgn: float = +1.0,
    L_exit: float = 0.80,
    y0: float = 0.0,
    r_fillet: float = 0.008,
    k_wall: float = 3200.0,
) -> list[float]:
    return [
        4.0, float(y0), 0.5 * float(width), float(x_entry), float(x_corner),
        0.0, 0.5 * math.pi, 1.0 if bend_sgn >= 0.0 else -1.0,
        float(L_exit), float(height), float(k_wall), float(r_fillet),
    ]


def make_uturn_pipe(
    x_entry: float = -1.05,
    x_bend: float = -1.40,
    R_bend: float = 0.54,
    r_in: float = 0.125,
    bend_sgn: float = +1.0,
    L_exit: float = 1.00,
    y0: float = 0.0,
    z_center: float = 0.118,
    k_wall: float = 3200.0,
) -> list[float]:
    return [
        5.0, float(y0), float(r_in), float(x_entry), float(x_bend),
        float(R_bend), math.pi, 1.0 if bend_sgn >= 0.0 else -1.0,
        float(L_exit), float(z_center), float(k_wall), 0.008,
    ]


def make_taper_pipe(
    x_entry: float = -1.05,
    x_taper_start: float = -1.45,
    L_taper: float = 0.65,
    r_in: float = 0.152,
    r_out: float = 0.112,
    L_exit: float = 0.90,
    y0: float = 0.0,
    z_center: float = 0.118,
    k_wall: float = 3200.0,
) -> list[float]:
    return [
        6.0, float(y0), float(r_in), float(x_entry), float(x_taper_start),
        float(L_taper), 0.0, 1.0,
        float(L_exit), float(z_center), float(k_wall), float(r_out),
    ]


def make_z_duct(
    x_entry: float = -1.05,
    x_corner: float = -1.48,
    L_mid: float = 0.68,
    width: float = 0.264,
    height: float = 0.250,
    bend_sgn: float = +1.0,
    L_exit: float = 0.90,
    y0: float = 0.0,
    r_fillet: float = 0.008,
    k_wall: float = 3200.0,
) -> list[float]:
    return [
        7.0, float(y0), 0.5 * float(width), float(x_entry), float(x_corner),
        float(L_mid), 0.5 * math.pi, 1.0 if bend_sgn >= 0.0 else -1.0,
        float(L_exit), float(height), float(k_wall), float(r_fillet),
    ]


def make_triple_wave_pipe(
    x_entry: float = -1.05,
    x_bend: float = -1.35,
    R_bend: float = 0.54,
    bend_angle_deg: float = 55.0,
    r_in: float = 0.125,
    bend_sgn: float = +1.0,
    L_exit: float = 0.85,
    y0: float = 0.0,
    z_center: float = 0.118,
    k_wall: float = 3200.0,
) -> list[float]:
    return [
        8.0, float(y0), float(r_in), float(x_entry), float(x_bend),
        float(R_bend), math.radians(float(bend_angle_deg)), 1.0 if bend_sgn >= 0.0 else -1.0,
        float(L_exit), float(z_center), float(k_wall), 0.008,
    ]


def _accumulate_surface_contact(
    vel: torch.Tensor,         # (B, M, 3)
    pen: torch.Tensor,         # (B, M)
    normal: torch.Tensor,      # (B, M, 3) unit inward normal
    k_w: torch.Tensor,         # (B, 1)
    c_w: float,
    mu_w: float,
    dt: float,
    implicit: bool,
    f_acc: torch.Tensor,       # (B, M, 3) modified in-place
    k_acc: torch.Tensor,       # (B, M, 3, 3) modified in-place
    fn_acc: torch.Tensor,      # (B, M) modified in-place
) -> None:
    """Exact C1 normal + Coulomb sliding friction + SPD implicit 3x3 tangent (mirrors CUDA)."""
    vn = (vel * normal).sum(-1)                       # velocity along inward normal
    raw_fn = k_w * pen - c_w * vn
    hit = (pen > 0.0) & (raw_fn > 0.0)
    fn = torch.where(hit, raw_fn, torch.zeros_like(raw_fn))
    fn_acc.add_(fn)

    vt = vel - vn[..., None] * normal
    vt_mag = torch.linalg.vector_norm(vt, dim=-1)
    th = torch.tanh(vt_mag / _V_REG)
    inv_vt = torch.where(vt_mag > 1e-12, th / vt_mag.clamp_min(1e-12), torch.full_like(vt_mag, 1.0 / _V_REG))
    f_fric = -(mu_w * fn * inv_vt)[..., None] * vt

    f_acc.add_(fn[..., None] * normal + f_fric)
    if implicit:
        # Backward-Euler linearization: compensate explicit damping on RHS and add SPD (h*c + h^2*k) n n^T + h*ct*(I - n n^T) to LHS
        c_n_eff = torch.where(hit, torch.full_like(fn, c_w), torch.zeros_like(fn))
        c_t_eff = torch.where(hit, (mu_w * fn / _V_REG) * (1.0 - th * th).clamp_min(0.0), torch.zeros_like(fn))
        f_acc.add_((c_n_eff * vn)[..., None] * normal + c_t_eff[..., None] * vt)
        wn = dt * c_n_eff + (dt * dt) * torch.where(hit, k_w.expand_as(fn), torch.zeros_like(fn))
        wt = dt * c_t_eff
        nn = normal[..., :, None] * normal[..., None, :]
        eye3 = torch.eye(3, device=vel.device, dtype=vel.dtype)
        k_acc.add_(wn[..., None, None] * nn + wt[..., None, None] * (eye3 - nn))


def _tensor_any(cond: torch.Tensor) -> bool:
    """Stream-capture-safe equivalent of ``bool(cond.any())`` (zero D2H sync during CUDA graph capture)."""
    if cond.is_cuda and torch.cuda.is_current_stream_capturing():
        return True
    return bool(cond.any())


def eval_pipe_contact_torch(
    pos: torch.Tensor,         # (B, M, 3)
    vel: torch.Tensor,         # (B, M, 3)
    pipe_cfg: torch.Tensor,    # (B, 12)
    node_radius: float,
    dt: float,
    implicit_contact: bool,
    k_scale: float = 1.0,
    c_wall: float = _C_WALL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate exact closed-form contact force, implicit 3x3 stiffness, and normal-force scalar.

    Bit-exact mathematical mirror of ``eval_pipe_contact_cuda`` in ``fused_physics.cu``.
    """
    B, M, _ = pos.shape
    dev, dtype = pos.device, pos.dtype
    f_acc = torch.zeros((B, M, 3), device=dev, dtype=dtype)
    k_acc = torch.zeros((B, M, 3, 3), device=dev, dtype=dtype)
    fn_acc = torch.zeros((B, M), device=dev, dtype=dtype)

    ptype = pipe_cfg[:, 0]
    if not _tensor_any(ptype > 0.5):
        return f_acc, k_acc, fn_acc

    y0 = pipe_cfg[:, 1:2]
    r_in = pipe_cfg[:, 2:3]
    x_en = pipe_cfg[:, 3:4]
    x_b = pipe_cfg[:, 4:5]
    R_b = pipe_cfg[:, 5:6].clamp_min(1e-3)
    ang = pipe_cfg[:, 6:7]
    sgn = torch.where(pipe_cfg[:, 7:8] >= 0.0, 1.0, -1.0)
    L_ex = pipe_cfg[:, 8:9]
    z_c = pipe_cfg[:, 9:10]
    k_w = pipe_cfg[:, 10:11] * float(k_scale)
    r_fil = pipe_cfg[:, 11:12].clamp_min(1e-4)

    px, py, pz = pos[..., 0], pos[..., 1], pos[..., 2]
    zeros = torch.zeros_like(px)
    ones = torch.ones_like(px)
    big = torch.full_like(px, 1e12)

    # Smooth entry funnel flare over [x_en, x_en + _FUNNEL_LEN]
    in_fun = (px <= x_en + _FUNNEL_LEN) & (px > x_en)
    u_fun = ((x_en + _FUNNEL_LEN - px) / _FUNNEL_LEN).clamp(0.0, 1.0)
    r_flare = torch.where(in_fun, _FUNNEL_FLARE * (1.0 - u_fun), zeros)
    r_eff = (r_in + r_flare - float(node_radius)).clamp_min(0.005)

    # --- Types 1, 2, 3, 5, 8: Constant-radius smooth cylindrical pipes (straight / 90-arc / S / U-turn / triple-wave) ---
    is_t1 = (ptype > 0.5) & (ptype < 1.5) & (pipe_cfg[:, 10] > 0.0)
    is_t2 = (ptype > 1.5) & (ptype < 2.5) & (pipe_cfg[:, 10] > 0.0)
    is_t3 = (ptype > 2.5) & (ptype < 3.5) & (pipe_cfg[:, 10] > 0.0)
    is_t5 = (ptype > 4.5) & (ptype < 5.5) & (pipe_cfg[:, 10] > 0.0)
    is_t8 = (ptype > 7.5) & (ptype < 8.5) & (pipe_cfg[:, 10] > 0.0)
    if _tensor_any(is_t1 | is_t2 | is_t3 | is_t5 | is_t8):
        qx = torch.zeros_like(px)
        qy = torch.zeros_like(py)
        active_c = torch.zeros((B, M), device=dev, dtype=torch.bool)
        r_eff_c = r_eff.clone()

        # Type 1: Straight cylinder x in [x_b (x_exit), x_en + funnel]
        if _tensor_any(is_t1):
            act1 = is_t1[:, None] & (px <= x_en + _FUNNEL_LEN) & (px >= x_b)
            qx = torch.where(act1, px, qx)
            qy = torch.where(act1, y0.expand_as(py), qy)
            active_c = active_c | act1

        # Type 2: 90-deg circular arc pipe: Entry straight -> 90 deg arc -> Exit straight along sgn*Y
        if _tensor_any(is_t2):
            cy2 = y0 + sgn * R_b
            ax_a = px.clamp(x_b, x_en + _FUNNEL_LEN)
            ay_a = y0.expand_as(py)
            d2_a = (px - ax_a).square() + (py - ay_a).square()
            ux = px - x_b
            uy = py - cy2
            alpha = torch.atan2(-ux, -sgn * uy).clamp(0.0, 0.5 * math.pi)
            ax_b = x_b - R_b * torch.sin(alpha)
            ay_b = cy2 - sgn * R_b * torch.cos(alpha)
            d2_b = (px - ax_b).square() + (py - ay_b).square()
            ax_c = (x_b - R_b).expand_as(px)
            sy_c = (sgn * (py - cy2)).clamp(zeros, L_ex)
            ay_c = cy2 + sgn * sy_c
            d2_c = (px - ax_c).square() + (py - ay_c).square()

            pick_b = d2_b < d2_a
            bx = torch.where(pick_b, ax_b, ax_a)
            by = torch.where(pick_b, ay_b, ay_a)
            bd2 = torch.where(pick_b, d2_b, d2_a)
            pick_c = d2_c < bd2
            bx = torch.where(pick_c, ax_c, bx)
            by = torch.where(pick_c, ay_c, by)

            past_in = (px > x_en + _FUNNEL_LEN) & (d2_a <= d2_b) & (d2_a <= d2_c)
            past_out = (sgn * (py - cy2) > L_ex) & (d2_c <= d2_a) & (d2_c <= d2_b)
            act2 = is_t2[:, None] & (~past_in) & (~past_out)
            qx = torch.where(act2, bx, qx)
            qy = torch.where(act2, by, qy)
            active_c = active_c | act2

        # Type 3: C1 S-Curve double-arc pipe (Entry -> Arc1(0..phi) -> Arc2(phi..0) -> Exit)
        if _tensor_any(is_t3):
            phi = ang.clamp(0.1, 0.5 * math.pi)
            dx_a = R_b * torch.sin(phi)
            dy_a = R_b * (1.0 - torch.cos(phi))
            sx1 = px.clamp(x_b, x_en + _FUNNEL_LEN)
            sy1 = y0.expand_as(py)
            d1 = (px - sx1).square() + (py - sy1).square()
            c1x = x_b
            c1y = y0 + sgn * R_b
            a1 = torch.atan2(-(px - c1x), -sgn * (py - c1y)).clamp(zeros, phi)
            sx2 = c1x - R_b * torch.sin(a1)
            sy2 = c1y - sgn * R_b * torch.cos(a1)
            d2 = (px - sx2).square() + (py - sy2).square()
            c2x = x_b - 2.0 * dx_a
            c2y = y0 + sgn * (2.0 * dy_a - R_b)
            a2 = torch.atan2(px - c2x, sgn * (py - c2y)).clamp(zeros, phi)
            sx3 = c2x + R_b * torch.sin(a2)
            sy3 = c2y + sgn * R_b * torch.cos(a2)
            d3 = (px - sx3).square() + (py - sy3).square()
            y_ex = y0 + sgn * 2.0 * dy_a
            sx4 = px.clamp(c2x - L_ex, c2x)
            sy4 = y_ex.expand_as(py)
            d4 = (px - sx4).square() + (py - sy4).square()

            bx, by, bd = sx1, sy1, d1
            m2 = d2 < bd
            bx, by, bd = torch.where(m2, sx2, bx), torch.where(m2, sy2, by), torch.where(m2, d2, bd)
            m3 = d3 < bd
            bx, by, bd = torch.where(m3, sx3, bx), torch.where(m3, sy3, by), torch.where(m3, d3, bd)
            m4 = d4 < bd
            bx, by, bd = torch.where(m4, sx4, bx), torch.where(m4, sy4, by), torch.where(m4, d4, bd)

            past_in = (px > x_en + _FUNNEL_LEN) & (d1 <= d2) & (d1 <= d3) & (d1 <= d4)
            past_out = (px < c2x - L_ex) & (d4 <= d1) & (d4 <= d2) & (d4 <= d3)
            act3 = is_t3[:, None] & (~past_in) & (~past_out)
            qx = torch.where(act3, bx, qx)
            qy = torch.where(act3, by, qy)
            active_c = active_c | act3

        # Type 5: 180-deg Hairpin U-Turn pipe (Entry -X -> 180-deg semicircle -> Return +X)
        if _tensor_any(is_t5):
            cy5 = y0 + sgn * R_b
            y_ret = y0 + 2.0 * sgn * R_b
            ax_a = px.clamp(x_b, x_en + _FUNNEL_LEN)
            ay_a = y0.expand_as(py)
            d2_a = (px - ax_a).square() + (py - ay_a).square()
            ux5 = px - x_b
            uy5 = sgn * (py - cy5)
            alpha5 = torch.atan2((-ux5).clamp_min(0.0), -uy5)
            ax_b = x_b - R_b * torch.sin(alpha5)
            ay_b = cy5 - sgn * R_b * torch.cos(alpha5)
            d2_b = (px - ax_b).square() + (py - ay_b).square()
            ax_c = px.clamp(x_b, x_b + L_ex)
            ay_c = y_ret.expand_as(py)
            d2_c = (px - ax_c).square() + (py - ay_c).square()

            pick_b = d2_b < d2_a
            bx = torch.where(pick_b, ax_b, ax_a)
            by = torch.where(pick_b, ay_b, ay_a)
            bd2 = torch.where(pick_b, d2_b, d2_a)
            pick_c = d2_c < bd2
            bx = torch.where(pick_c, ax_c, bx)
            by = torch.where(pick_c, ay_c, by)

            past_in = (px > x_en + _FUNNEL_LEN) & (d2_a <= d2_b) & (d2_a <= d2_c)
            past_out = (px > x_b + L_ex) & (d2_c <= d2_a) & (d2_c <= d2_b)
            act5 = is_t5[:, None] & (~past_in) & (~past_out)
            qx = torch.where(act5, bx, qx)
            qy = torch.where(act5, by, qy)
            # Only apply entry funnel flare on entry arm (not on parallel return arm!)
            rf5 = torch.where(sgn * (py - y0) < R_b, r_flare, zeros)
            r_eff_5 = (r_in + rf5 - float(node_radius)).clamp_min(0.005)
            r_eff_c = torch.where(act5, r_eff_5, r_eff_c)
            active_c = active_c | act5

        # Type 8: Long Triple-Wave Serpentine pipe (Entry -> +phi -> -2*phi -> +phi -> Exit)
        if _tensor_any(is_t8):
            phi = ang.clamp(0.1, 1.35)
            dx_a = R_b * torch.sin(phi)
            dy_a = R_b * (1.0 - torch.cos(phi))
            sx1 = px.clamp(x_b, x_en + _FUNNEL_LEN)
            sy1 = y0.expand_as(py)
            d1 = (px - sx1).square() + (py - sy1).square()

            c1x, c1y = x_b, y0 + sgn * R_b
            a1 = torch.atan2(-(px - c1x), -sgn * (py - c1y)).clamp(zeros, phi)
            sx2 = c1x - R_b * torch.sin(a1)
            sy2 = c1y - sgn * R_b * torch.cos(a1)
            d2 = (px - sx2).square() + (py - sy2).square()

            c2x, c2y = x_b - 2.0 * dx_a, y0 + sgn * (2.0 * dy_a - R_b)
            a2 = torch.atan2(px - c2x, sgn * (py - c2y)).clamp(-phi, phi)
            sx3 = c2x + R_b * torch.sin(a2)
            sy3 = c2y + sgn * R_b * torch.cos(a2)
            d3 = (px - sx3).square() + (py - sy3).square()

            c3x, c3y = x_b - 4.0 * dx_a, y0 + sgn * R_b
            a3 = torch.atan2(-(px - c3x), -sgn * (py - c3y)).clamp(-phi, zeros)
            sx4 = c3x - R_b * torch.sin(a3)
            sy4 = c3y - sgn * R_b * torch.cos(a3)
            d4 = (px - sx4).square() + (py - sy4).square()

            sx5 = px.clamp(c3x - L_ex, c3x)
            sy5 = y0.expand_as(py)
            d5 = (px - sx5).square() + (py - sy5).square()

            bx, by, bd = sx1, sy1, d1
            m2 = d2 < bd
            bx, by, bd = torch.where(m2, sx2, bx), torch.where(m2, sy2, by), torch.where(m2, d2, bd)
            m3 = d3 < bd
            bx, by, bd = torch.where(m3, sx3, bx), torch.where(m3, sy3, by), torch.where(m3, d3, bd)
            m4 = d4 < bd
            bx, by, bd = torch.where(m4, sx4, bx), torch.where(m4, sy4, by), torch.where(m4, d4, bd)
            m5 = d5 < bd
            bx, by, bd = torch.where(m5, sx5, bx), torch.where(m5, sy5, by), torch.where(m5, d5, bd)

            past_in = (px > x_en + _FUNNEL_LEN) & (d1 <= d2) & (d1 <= d3) & (d1 <= d4) & (d1 <= d5)
            past_out = (px < c3x - L_ex) & (d5 <= d1) & (d5 <= d2) & (d5 <= d3) & (d5 <= d4)
            act8 = is_t8[:, None] & (~past_in) & (~past_out)
            qx = torch.where(act8, bx, qx)
            qy = torch.where(act8, by, qy)
            active_c = active_c | act8

        dx = px - qx
        dy = py - qy
        dz = pz - z_c
        dr = torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-24)
        inv_dr = 1.0 / dr.clamp_min(1e-12)
        n_cyl = torch.stack((-dx * inv_dr, -dy * inv_dr, -dz * inv_dr), dim=-1)
        pen_cyl = torch.where(active_c, (dr - r_eff_c).clamp_min(0.0), zeros)
        _accumulate_surface_contact(vel, pen_cyl, n_cyl, k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)

    # --- Type 6: Smooth C1 Conical Taper / Reducer Pipe (wide r_in -> cosine cone L_taper -> narrow r_out) ---
    is_t6 = (ptype > 5.5) & (ptype < 6.5) & (pipe_cfg[:, 10] > 0.0)
    if _tensor_any(is_t6):
        L_tap = R_b.clamp_min(0.10)
        r_out = r_fil.clamp_min(0.080)
        x_end6 = x_b - L_tap - L_ex
        act6 = is_t6[:, None] & (px <= x_en + _FUNNEL_LEN) & (px >= x_end6)
        u_tap = ((x_b - px) / L_tap).clamp(0.0, 1.0)
        in_tap = (px < x_b) & (px > x_b - L_tap)
        r_tap = r_in + 0.5 * (r_out - r_in) * (1.0 - torch.cos(math.pi * u_tap))
        r_loc = torch.where(px >= x_b, r_in + r_flare, torch.where(in_tap, r_tap, r_out.expand_as(px)))
        dr_dx = torch.where(in_tap, -0.5 * (r_out - r_in) * (math.pi / L_tap) * torch.sin(math.pi * u_tap), zeros)
        dy6 = py - y0
        dz6 = pz - z_c
        rho6 = torch.sqrt(dy6 * dy6 + dz6 * dz6 + 1e-24)
        inv_rho6 = 1.0 / rho6.clamp_min(1e-12)
        r_eff6 = (r_loc - float(node_radius)).clamp_min(0.005)
        inv_sl = 1.0 / torch.sqrt(1.0 + dr_dx * dr_dx)
        pen6 = torch.where(act6, ((rho6 - r_eff6) * inv_sl).clamp_min(0.0), zeros)
        n6 = torch.stack((dr_dx * inv_sl, -(dy6 * inv_rho6) * inv_sl, -(dz6 * inv_rho6) * inv_sl), dim=-1)
        _accumulate_surface_contact(vel, pen6, n6, k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)

    # --- Type 4: Sharp 90-deg Right-Angle Transparent Square L-Duct ---
    is_t4 = (ptype > 3.5) & (ptype < 4.5) & (pipe_cfg[:, 10] > 0.0)
    if _tensor_any(is_t4):
        lx = px - x_b
        ly = sgn * (py - y0)
        lx_en = x_en - x_b
        hw_core = (r_in - float(node_radius)).clamp_min(0.005)
        hw = hw_core + r_flare  # flared only near entrance
        act4 = is_t4[:, None] & (lx <= lx_en + _FUNNEL_LEN) & (ly <= L_ex) & (lx >= -hw_core - 0.15) & (ly >= -hw_core - 0.15)

        # 1) Outer dead-end wall: lx < -hw_core -> inward normal (+1, 0, 0)
        pen_end = torch.where(act4, (-hw_core - lx).clamp_min(0.0), zeros)
        n_end = torch.stack((ones, zeros, zeros), dim=-1)
        _accumulate_surface_contact(vel, pen_end, n_end, k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)

        # 2) Outer side wall along entry arm: ly < -hw -> inward normal (0, +sgn, 0)
        pen_out = torch.where(act4, (-hw - ly).clamp_min(0.0), zeros)
        n_out = torch.stack((zeros, sgn.expand_as(py), zeros), dim=-1)
        _accumulate_surface_contact(vel, pen_out, n_out, k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)

        # 3) Inner walls + exact C1 Minkowski offset of sharp 90-deg inner corner fillet
        # Physical inner solid quadrant is {lx >= r_in, ly >= r_in} with fillet center (r_in + r_fil, r_in + r_fil).
        # Offset by sphere node_radius yields an exact circular arc of radius R_cf = r_fil + node_radius!
        cx_f = r_in + r_fil
        cy_f = r_in + r_fil
        R_cf = r_fil + float(node_radius)
        # Entry inner wall region: lx > cx_f
        in_ew = act4 & (lx > cx_f)
        pen_ew = torch.where(in_ew, (ly - hw).clamp_min(0.0), zeros)
        # Exit inner wall region: ly > cy_f and lx <= cx_f
        in_xw = act4 & (ly > cy_f) & (lx <= cx_f)
        pen_xw = torch.where(in_xw, (lx - hw_core).clamp_min(0.0), zeros)
        # Sharp inner corner Minkowski fillet region: lx <= cx_f and ly <= cy_f
        in_cf = act4 & (lx <= cx_f) & (ly <= cy_f) & (lx > r_in - float(node_radius) - r_fil) & (ly > r_in - float(node_radius) - r_fil)
        dfx = lx - cx_f
        dfy = ly - cy_f
        d_f = torch.sqrt(dfx * dfx + dfy * dfy + 1e-24)
        pen_cf = torch.where(in_cf, (R_cf - d_f).clamp_min(0.0), zeros)

        pen_in = torch.where(in_ew, pen_ew, torch.where(in_xw, pen_xw, pen_cf))
        nx_in = torch.where(in_ew, zeros, torch.where(in_xw, -ones, dfx / d_f.clamp_min(1e-12)))
        ny_in = torch.where(in_ew, -sgn.expand_as(py), torch.where(in_xw, zeros, sgn * (dfy / d_f.clamp_min(1e-12))))
        n_in = torch.stack((nx_in, ny_in, zeros), dim=-1)
        _accumulate_surface_contact(vel, pen_in, n_in, k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)

        # 4) Duct ceiling: pz > H_duct - node_radius -> inward normal (0, 0, -1)
        pen_top = torch.where(act4, (pz - (z_c - float(node_radius))).clamp_min(0.0), zeros)
        n_top = torch.stack((zeros, zeros, -ones), dim=-1)
        _accumulate_surface_contact(vel, pen_top, n_top, k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)

    # --- Type 7: Z-Shaped Double 90-deg Sharp Square Chicane (Corner#1 -X->+sgn*Y, Corner#2 +sgn*Y->-X) ---
    is_t7 = (ptype > 6.5) & (ptype < 7.5) & (pipe_cfg[:, 10] > 0.0)
    if _tensor_any(is_t7):
        lx = px - x_b
        ly = sgn * (py - y0)
        lx_en = x_en - x_b
        L_mid = R_b.clamp_min(0.50)
        ly_half = 0.5 * L_mid
        hw_core = (r_in - float(node_radius)).clamp_min(0.005)
        hw = hw_core + r_flare
        cx_f = r_in + r_fil
        cy_f = r_in + r_fil
        R_cf = r_fil + float(node_radius)

        # Lower half (ly <= ly_half): Corner #1
        act7a = is_t7[:, None] & (ly <= ly_half) & (lx <= lx_en + _FUNNEL_LEN) & (lx >= -hw_core - 0.15) & (ly >= -hw_core - 0.15)
        pen_end1 = torch.where(act7a, (-hw_core - lx).clamp_min(0.0), zeros)
        _accumulate_surface_contact(vel, pen_end1, torch.stack((ones, zeros, zeros), dim=-1), k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)
        pen_out1 = torch.where(act7a, (-hw - ly).clamp_min(0.0), zeros)
        _accumulate_surface_contact(vel, pen_out1, torch.stack((zeros, sgn.expand_as(py), zeros), dim=-1), k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)
        in_ew1 = act7a & (lx > cx_f)
        in_xw1 = act7a & (ly > cy_f) & (lx <= cx_f)
        in_cf1 = act7a & (lx <= cx_f) & (ly <= cy_f) & (lx > r_in - float(node_radius) - r_fil) & (ly > r_in - float(node_radius) - r_fil)
        dfx1, dfy1 = lx - cx_f, ly - cy_f
        df1 = torch.sqrt(dfx1 * dfx1 + dfy1 * dfy1 + 1e-24)
        pen_in1 = torch.where(in_ew1, (ly - hw).clamp_min(0.0), torch.where(in_xw1, (lx - hw_core).clamp_min(0.0), torch.where(in_cf1, (R_cf - df1).clamp_min(0.0), zeros)))
        nx_in1 = torch.where(in_ew1, zeros, torch.where(in_xw1, -ones, dfx1 / df1.clamp_min(1e-12)))
        ny_in1 = torch.where(in_ew1, -sgn.expand_as(py), torch.where(in_xw1, zeros, sgn * (dfy1 / df1.clamp_min(1e-12))))
        _accumulate_surface_contact(vel, pen_in1, torch.stack((nx_in1, ny_in1, zeros), dim=-1), k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)

        # Upper half (ly > ly_half): Corner #2 in reflected local frame (ux = -lx, uy = L_mid - ly)
        ux = -lx
        uy = L_mid - ly
        act7b = is_t7[:, None] & (ly > ly_half) & (ux <= L_ex) & (ux >= -hw_core - 0.15) & (uy >= -hw_core - 0.15)
        pen_end2 = torch.where(act7b, (-hw_core - uy).clamp_min(0.0), zeros)
        _accumulate_surface_contact(vel, pen_end2, torch.stack((zeros, -sgn.expand_as(py), zeros), dim=-1), k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)
        pen_out2 = torch.where(act7b, (-hw_core - ux).clamp_min(0.0), zeros)
        _accumulate_surface_contact(vel, pen_out2, torch.stack((-ones, zeros, zeros), dim=-1), k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)
        in_mw2 = act7b & (uy > cy_f)
        in_xw2 = act7b & (ux > cx_f) & (uy <= cy_f)
        in_cf2 = act7b & (ux <= cx_f) & (uy <= cy_f) & (ux > r_in - float(node_radius) - r_fil) & (uy > r_in - float(node_radius) - r_fil)
        dfu2, dfv2 = ux - cx_f, uy - cy_f
        df2 = torch.sqrt(dfu2 * dfu2 + dfv2 * dfv2 + 1e-24)
        pen_in2 = torch.where(in_mw2, (ux - hw_core).clamp_min(0.0), torch.where(in_xw2, (uy - hw_core).clamp_min(0.0), torch.where(in_cf2, (R_cf - df2).clamp_min(0.0), zeros)))
        nx_in2 = torch.where(in_mw2, ones, torch.where(in_xw2, zeros, -dfu2 / df2.clamp_min(1e-12)))
        ny_in2 = torch.where(in_mw2, zeros, torch.where(in_xw2, sgn.expand_as(py), -sgn * (dfv2 / df2.clamp_min(1e-12))))
        _accumulate_surface_contact(vel, pen_in2, torch.stack((nx_in2, ny_in2, zeros), dim=-1), k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)

        pen_top7 = torch.where(act7a | act7b, (pz - (z_c - float(node_radius))).clamp_min(0.0), zeros)
        _accumulate_surface_contact(vel, pen_top7, torch.stack((zeros, zeros, -ones), dim=-1), k_w, c_wall, _MU_WALL, dt, implicit_contact, f_acc, k_acc, fn_acc)

    return f_acc, k_acc, fn_acc


def eval_pipe_membership_torch(
    pos_xy: torch.Tensor,      # (B, M, 2)
    pipe_cfg: torch.Tensor,    # (B, 12)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(in_pipe [B, M], signed_lat_offset [B, M])`` for all 4 analytical pipe types.

    ``signed_lat_offset`` is measured along the local left-right normal ``(ty, -tx)`` so it is
    continuous around 90-deg arcs, S-curves, and sharp 90-deg L-duct corners, and strictly
    antisymmetric under left-right reflection ``(y -> -y, bend_sgn -> -bend_sgn)``.
    """
    B, M, _ = pos_xy.shape
    dev, dtype = pos_xy.device, pos_xy.dtype
    in_pipe = torch.zeros((B, M), device=dev, dtype=torch.bool)
    lat_off = torch.zeros((B, M), device=dev, dtype=dtype)

    ptype = pipe_cfg[:, 0]
    if not bool((ptype > 0.5).any()):
        return in_pipe, lat_off

    y0 = pipe_cfg[:, 1:2]
    r_in = pipe_cfg[:, 2:3]
    x_en = pipe_cfg[:, 3:4]
    x_b = pipe_cfg[:, 4:5]
    R_b = pipe_cfg[:, 5:6].clamp_min(1e-3)
    ang = pipe_cfg[:, 6:7]
    sgn = torch.where(pipe_cfg[:, 7:8] >= 0.0, 1.0, -1.0)
    L_ex = pipe_cfg[:, 8:9]
    active_k = pipe_cfg[:, 10] > 0.0

    px, py = pos_xy[..., 0], pos_xy[..., 1]
    zeros = torch.zeros_like(px)

    is_t1 = (ptype > 0.5) & (ptype < 1.5) & active_k
    if bool(is_t1.any()):
        act1 = is_t1[:, None] & (px <= x_en + _FUNNEL_LEN) & (px >= x_b)
        in_pipe = in_pipe | act1
        lat_off = torch.where(act1, py - y0, lat_off)

    is_t2 = (ptype > 1.5) & (ptype < 2.5) & active_k
    if bool(is_t2.any()):
        cy2 = y0 + sgn * R_b
        ax_a = px.clamp(x_b, x_en + _FUNNEL_LEN)
        ay_a = y0.expand_as(py)
        d2_a = (px - ax_a).square() + (py - ay_a).square()

        alpha = torch.atan2(-(px - x_b), -sgn * (py - cy2)).clamp(0.0, 0.5 * math.pi)
        ax_b = x_b - R_b * torch.sin(alpha)
        ay_b = cy2 - sgn * R_b * torch.cos(alpha)
        d2_b = (px - ax_b).square() + (py - ay_b).square()

        ax_c = (x_b - R_b).expand_as(px)
        ay_c = cy2 + sgn * (sgn * (py - cy2)).clamp(zeros, L_ex)
        d2_c = (px - ax_c).square() + (py - ay_c).square()

        pick_b = d2_b < d2_a
        qx = torch.where(pick_b, ax_b, ax_a)
        qy = torch.where(pick_b, ay_b, ay_a)
        nx = torch.where(pick_b, sgn * torch.sin(alpha), zeros)
        ny = torch.where(pick_b, torch.cos(alpha), torch.ones_like(px))
        bd2 = torch.where(pick_b, d2_b, d2_a)

        pick_c = d2_c < bd2
        qx = torch.where(pick_c, ax_c, qx)
        qy = torch.where(pick_c, ay_c, qy)
        nx = torch.where(pick_c, sgn.expand_as(px), nx)
        ny = torch.where(pick_c, zeros, ny)

        past_in = (px > x_en + _FUNNEL_LEN) & (d2_a <= d2_b) & (d2_a <= d2_c)
        past_out = (sgn * (py - cy2) > L_ex) & (d2_c <= d2_a) & (d2_c <= d2_b)
        act2 = is_t2[:, None] & (~past_in) & (~past_out)
        in_pipe = in_pipe | act2
        lat_off = torch.where(act2, (px - qx) * nx + (py - qy) * ny, lat_off)

    is_t3 = (ptype > 2.5) & (ptype < 3.5) & active_k
    if bool(is_t3.any()):
        phi = ang.clamp(0.1, 0.5 * math.pi)
        dx_a = R_b * torch.sin(phi)
        dy_a = R_b * (1.0 - torch.cos(phi))
        sx1 = px.clamp(x_b, x_en + _FUNNEL_LEN)
        sy1 = y0.expand_as(py)
        d1 = (px - sx1).square() + (py - sy1).square()

        c1x, c1y = x_b, y0 + sgn * R_b
        a1 = torch.atan2(-(px - c1x), -sgn * (py - c1y)).clamp(zeros, phi)
        sx2 = c1x - R_b * torch.sin(a1)
        sy2 = c1y - sgn * R_b * torch.cos(a1)
        d2 = (px - sx2).square() + (py - sy2).square()

        c2x, c2y = x_b - 2.0 * dx_a, y0 + sgn * (2.0 * dy_a - R_b)
        a2 = torch.atan2(px - c2x, sgn * (py - c2y)).clamp(zeros, phi)
        sx3 = c2x + R_b * torch.sin(a2)
        sy3 = c2y + sgn * R_b * torch.cos(a2)
        d3 = (px - sx3).square() + (py - sy3).square()

        y_ex = y0 + sgn * 2.0 * dy_a
        sx4 = px.clamp(c2x - L_ex, c2x)
        sy4 = y_ex.expand_as(py)
        d4 = (px - sx4).square() + (py - sy4).square()

        qx, qy, bd = sx1, sy1, d1
        nx, ny = zeros, torch.ones_like(px)
        m2 = d2 < bd
        qx, qy, bd = torch.where(m2, sx2, qx), torch.where(m2, sy2, qy), torch.where(m2, d2, bd)
        nx, ny = torch.where(m2, sgn * torch.sin(a1), nx), torch.where(m2, torch.cos(a1), ny)
        m3 = d3 < bd
        qx, qy, bd = torch.where(m3, sx3, qx), torch.where(m3, sy3, qy), torch.where(m3, d3, bd)
        nx, ny = torch.where(m3, sgn * torch.sin(a2), nx), torch.where(m3, torch.cos(a2), ny)
        m4 = d4 < bd
        qx, qy, bd = torch.where(m4, sx4, qx), torch.where(m4, sy4, qy), torch.where(m4, d4, bd)
        nx, ny = torch.where(m4, zeros, nx), torch.where(m4, torch.ones_like(px), ny)

        past_in = (px > x_en + _FUNNEL_LEN) & (d1 <= d2) & (d1 <= d3) & (d1 <= d4)
        past_out = (px < c2x - L_ex) & (d4 <= d1) & (d4 <= d2) & (d4 <= d3)
        act3 = is_t3[:, None] & (~past_in) & (~past_out)
        in_pipe = in_pipe | act3
        lat_off = torch.where(act3, (px - qx) * nx + (py - qy) * ny, lat_off)

    is_t4 = (ptype > 3.5) & (ptype < 4.5) & active_k
    if bool(is_t4.any()):
        lx = px - x_b
        ly = sgn * (py - y0)
        lx_en = x_en - x_b
        act4 = is_t4[:, None] & (lx <= lx_en + _FUNNEL_LEN) & (ly <= L_ex) & (lx >= -r_in - 0.10) & (ly >= -r_in - 0.10)
        in_pipe = in_pipe | act4
        lat4 = torch.where(lx > ly, sgn * ly, sgn * lx)
        lat_off = torch.where(act4, lat4, lat_off)

    is_t5 = (ptype > 4.5) & (ptype < 5.5) & active_k
    if bool(is_t5.any()):
        cy5 = y0 + sgn * R_b
        y_ret = y0 + 2.0 * sgn * R_b
        ax_a = px.clamp(x_b, x_en + _FUNNEL_LEN)
        d2_a = (px - ax_a).square() + (py - y0).square()
        ux5 = px - x_b
        uy5 = sgn * (py - cy5)
        alpha5 = torch.atan2((-ux5).clamp_min(0.0), -uy5)
        ax_b = x_b - R_b * torch.sin(alpha5)
        ay_b = cy5 - sgn * R_b * torch.cos(alpha5)
        d2_b = (px - ax_b).square() + (py - ay_b).square()
        ax_c = px.clamp(x_b, x_b + L_ex)
        d2_c = (px - ax_c).square() + (py - y_ret).square()

        pick_b = d2_b < d2_a
        qx = torch.where(pick_b, ax_b, ax_a)
        qy = torch.where(pick_b, ay_b, y0.expand_as(py))
        nx = torch.where(pick_b, sgn * torch.sin(alpha5), zeros)
        ny = torch.where(pick_b, torch.cos(alpha5), torch.ones_like(px))
        bd2 = torch.where(pick_b, d2_b, d2_a)
        pick_c = d2_c < bd2
        qx = torch.where(pick_c, ax_c, qx)
        qy = torch.where(pick_c, y_ret.expand_as(py), qy)
        nx = torch.where(pick_c, zeros, nx)
        ny = torch.where(pick_c, -torch.ones_like(px), ny)

        past_in = (px > x_en + _FUNNEL_LEN) & (d2_a <= d2_b) & (d2_a <= d2_c)
        past_out = (px > x_b + L_ex) & (d2_c <= d2_a) & (d2_c <= d2_b)
        act5 = is_t5[:, None] & (~past_in) & (~past_out)
        in_pipe = in_pipe | act5
        lat_off = torch.where(act5, (px - qx) * nx + (py - qy) * ny, lat_off)

    is_t6 = (ptype > 5.5) & (ptype < 6.5) & active_k
    if bool(is_t6.any()):
        L_tap = R_b.clamp_min(0.10)
        act6 = is_t6[:, None] & (px <= x_en + _FUNNEL_LEN) & (px >= x_b - L_tap - L_ex)
        in_pipe = in_pipe | act6
        lat_off = torch.where(act6, py - y0, lat_off)

    is_t7 = (ptype > 6.5) & (ptype < 7.5) & active_k
    if bool(is_t7.any()):
        lx = px - x_b
        ly = sgn * (py - y0)
        lx_en = x_en - x_b
        L_mid = R_b.clamp_min(0.50)
        ly_half = 0.5 * L_mid
        act7a = is_t7[:, None] & (ly <= ly_half) & (lx <= lx_en + _FUNNEL_LEN) & (lx >= -r_in - 0.10) & (ly >= -r_in - 0.10)
        ux = -lx
        uy = L_mid - ly
        act7b = is_t7[:, None] & (ly > ly_half) & (ux <= L_ex) & (ux >= -r_in - 0.10) & (uy >= -r_in - 0.10)
        act7 = act7a | act7b
        in_pipe = in_pipe | act7
        lat7a = torch.where(lx > ly, sgn * ly, sgn * lx)
        lat7b = torch.where(uy > ux, sgn * (-ux), sgn * (-uy))
        lat_off = torch.where(act7a, lat7a, torch.where(act7b, lat7b, lat_off))

    is_t8 = (ptype > 7.5) & (ptype < 8.5) & active_k
    if bool(is_t8.any()):
        phi = ang.clamp(0.1, 1.35)
        dx_a = R_b * torch.sin(phi)
        dy_a = R_b * (1.0 - torch.cos(phi))
        sx1 = px.clamp(x_b, x_en + _FUNNEL_LEN)
        d1 = (px - sx1).square() + (py - y0).square()
        c1x, c1y = x_b, y0 + sgn * R_b
        a1 = torch.atan2(-(px - c1x), -sgn * (py - c1y)).clamp(zeros, phi)
        sx2, sy2 = c1x - R_b * torch.sin(a1), c1y - sgn * R_b * torch.cos(a1)
        d2 = (px - sx2).square() + (py - sy2).square()
        c2x, c2y = x_b - 2.0 * dx_a, y0 + sgn * (2.0 * dy_a - R_b)
        a2 = torch.atan2(px - c2x, sgn * (py - c2y)).clamp(-phi, phi)
        sx3, sy3 = c2x + R_b * torch.sin(a2), c2y + sgn * R_b * torch.cos(a2)
        d3 = (px - sx3).square() + (py - sy3).square()
        c3x, c3y = x_b - 4.0 * dx_a, y0 + sgn * R_b
        a3 = torch.atan2(-(px - c3x), -sgn * (py - c3y)).clamp(-phi, zeros)
        sx4, sy4 = c3x - R_b * torch.sin(a3), c3y - sgn * R_b * torch.cos(a3)
        d4 = (px - sx4).square() + (py - sy4).square()
        sx5 = px.clamp(c3x - L_ex, c3x)
        d5 = (px - sx5).square() + (py - y0).square()

        qx, qy, bd = sx1, y0.expand_as(py), d1
        nx, ny = zeros, torch.ones_like(px)
        m2 = d2 < bd
        qx, qy, bd = torch.where(m2, sx2, qx), torch.where(m2, sy2, qy), torch.where(m2, d2, bd)
        nx, ny = torch.where(m2, sgn * torch.sin(a1), nx), torch.where(m2, torch.cos(a1), ny)
        m3 = d3 < bd
        qx, qy, bd = torch.where(m3, sx3, qx), torch.where(m3, sy3, qy), torch.where(m3, d3, bd)
        nx, ny = torch.where(m3, sgn * torch.sin(a2), nx), torch.where(m3, torch.cos(a2), ny)
        m4 = d4 < bd
        qx, qy, bd = torch.where(m4, sx4, qx), torch.where(m4, sy4, qy), torch.where(m4, d4, bd)
        nx, ny = torch.where(m4, sgn * torch.sin(a3), nx), torch.where(m4, torch.cos(a3), ny)
        m5 = d5 < bd
        qx, qy, bd = torch.where(m5, sx5, qx), torch.where(m5, y0.expand_as(py), qy), torch.where(m5, d5, bd)
        nx, ny = torch.where(m5, zeros, nx), torch.where(m5, torch.ones_like(px), ny)

        past_in = (px > x_en + _FUNNEL_LEN) & (d1 <= d2) & (d1 <= d3) & (d1 <= d4) & (d1 <= d5)
        past_out = (px < c3x - L_ex) & (d5 <= d1) & (d5 <= d2) & (d5 <= d3) & (d5 <= d4)
        act8 = is_t8[:, None] & (~past_in) & (~past_out)
        in_pipe = in_pipe | act8
        lat_off = torch.where(act8, (px - qx) * nx + (py - qy) * ny, lat_off)

    return in_pipe, lat_off


def eval_pipe_arc_progress_torch(
    pos_xy: torch.Tensor,      # (B, M, 2)
    pipe_cfg: torch.Tensor,    # (B, 12)
) -> torch.Tensor:
    """Return continuous monotonic centerline arc-progress coordinate ``s [B, M]`` (m) for all 4 pipe types.

    Strictly invariant under left-right mirror ``(y -> -y, y0 -> -y0, bend_sgn -> -bend_sgn)`` and
    C1-smooth across all curved junctions and the sharp 90-deg L-duct corner (polar around inner corner).
    """
    B, M, _ = pos_xy.shape
    px, py = pos_xy[..., 0], pos_xy[..., 1]
    s_out = -px.clone()

    ptype = pipe_cfg[:, 0]
    active_k = pipe_cfg[:, 10] > 0.0
    if not bool(((ptype > 0.5) & active_k).any()):
        return s_out

    y0 = pipe_cfg[:, 1:2]
    r_in = pipe_cfg[:, 2:3].clamp_min(0.05)
    x_en = pipe_cfg[:, 3:4]
    x_b = pipe_cfg[:, 4:5]
    R_b = pipe_cfg[:, 5:6].clamp_min(1e-3)
    ang = pipe_cfg[:, 6:7]
    sgn = torch.where(pipe_cfg[:, 7:8] >= 0.0, 1.0, -1.0)
    zeros = torch.zeros_like(px)

    # Type 1: Straight cylinder
    is_t1 = (ptype > 0.5) & (ptype < 1.5) & active_k
    if bool(is_t1.any()):
        s_out = torch.where(is_t1[:, None], x_en - px, s_out)

    # Type 2: 90-deg circular arc
    is_t2 = (ptype > 1.5) & (ptype < 2.5) & active_k
    if bool(is_t2.any()):
        L_in = (x_en - x_b).clamp_min(0.0)
        L_arc = 0.5 * math.pi * R_b
        cy2 = y0 + sgn * R_b
        ax_a = px.clamp_min(x_b)
        d2_a = (px - ax_a).square() + (py - y0).square()
        s_a = x_en - ax_a

        alpha = torch.atan2(-(px - x_b), -sgn * (py - cy2)).clamp(0.0, 0.5 * math.pi)
        ax_b = x_b - R_b * torch.sin(alpha)
        ay_b = cy2 - sgn * R_b * torch.cos(alpha)
        d2_b = (px - ax_b).square() + (py - ay_b).square()
        s_b = L_in + alpha * R_b

        sy_c = (sgn * (py - cy2)).clamp_min(0.0)
        ax_c = x_b - R_b
        ay_c = cy2 + sgn * sy_c
        d2_c = (px - ax_c).square() + (py - ay_c).square()
        s_c = L_in + L_arc + sy_c

        pick_b = d2_b < d2_a
        s2 = torch.where(pick_b, s_b, s_a)
        bd2 = torch.where(pick_b, d2_b, d2_a)
        s2 = torch.where(d2_c < bd2, s_c, s2)
        s_out = torch.where(is_t2[:, None], s2, s_out)

    # Type 3: S-curve double-arc
    is_t3 = (ptype > 2.5) & (ptype < 3.5) & active_k
    if bool(is_t3.any()):
        phi = ang.clamp(0.1, 0.5 * math.pi)
        L_in = (x_en - x_b).clamp_min(0.0)
        L_arc = phi * R_b
        dx_a = R_b * torch.sin(phi)
        dy_a = R_b * (1.0 - torch.cos(phi))

        sx1 = px.clamp_min(x_b)
        d1 = (px - sx1).square() + (py - y0).square()
        s1 = x_en - sx1

        c1x, c1y = x_b, y0 + sgn * R_b
        a1 = torch.atan2(-(px - c1x), -sgn * (py - c1y)).clamp(zeros, phi)
        sx2 = c1x - R_b * torch.sin(a1)
        sy2 = c1y - sgn * R_b * torch.cos(a1)
        d2 = (px - sx2).square() + (py - sy2).square()
        s2 = L_in + a1 * R_b

        c2x, c2y = x_b - 2.0 * dx_a, y0 + sgn * (2.0 * dy_a - R_b)
        a2 = torch.atan2(px - c2x, sgn * (py - c2y)).clamp(zeros, phi)
        sx3 = c2x + R_b * torch.sin(a2)
        sy3 = c2y + sgn * R_b * torch.cos(a2)
        d3 = (px - sx3).square() + (py - sy3).square()
        s3 = L_in + L_arc + (phi - a2) * R_b

        y_ex = y0 + sgn * 2.0 * dy_a
        sx4 = px.clamp_max(c2x)
        d4 = (px - sx4).square() + (py - y_ex).square()
        s4 = L_in + 2.0 * L_arc + (c2x - sx4)

        sc, bd = s1, d1
        m2 = d2 < bd
        sc, bd = torch.where(m2, s2, sc), torch.where(m2, d2, bd)
        m3 = d3 < bd
        sc, bd = torch.where(m3, s3, sc), torch.where(m3, d3, bd)
        sc = torch.where(d4 < bd, s4, sc)
        s_out = torch.where(is_t3[:, None], sc, s_out)

    # Type 4: Sharp 90-deg L-duct (C1 polar coordinate around inner corner (r_in, r_in) -> zero jump!)
    is_t4 = (ptype > 3.5) & (ptype < 4.5) & active_k
    if bool(is_t4.any()):
        lx = px - x_b
        ly = sgn * (py - y0)
        L_in = (x_en - x_b).clamp_min(0.0)
        s_entry = L_in - lx
        s_exit = (L_in - r_in) + 0.5 * math.pi * r_in + (ly - r_in)
        alpha_c = torch.atan2((r_in - lx).clamp_min(0.0), (r_in - ly).clamp_min(0.0))
        s_corner = (L_in - r_in) + alpha_c * r_in
        s4 = torch.where(lx >= r_in, s_entry, torch.where(ly >= r_in, s_exit, s_corner))
        s_out = torch.where(is_t4[:, None], s4, s_out)

    # Type 5: 180-deg Hairpin U-Turn
    is_t5 = (ptype > 4.5) & (ptype < 5.5) & active_k
    if bool(is_t5.any()):
        L_in = (x_en - x_b).clamp_min(0.0)
        L_arc = math.pi * R_b
        cy5 = y0 + sgn * R_b
        y_ret = y0 + 2.0 * sgn * R_b
        ax_a = px.clamp_min(x_b)
        d2_a = (px - ax_a).square() + (py - y0).square()
        s_a = x_en - ax_a

        ux5 = px - x_b
        uy5 = sgn * (py - cy5)
        alpha5 = torch.atan2((-ux5).clamp_min(0.0), -uy5)
        ax_b = x_b - R_b * torch.sin(alpha5)
        ay_b = cy5 - sgn * R_b * torch.cos(alpha5)
        d2_b = (px - ax_b).square() + (py - ay_b).square()
        s_b = L_in + alpha5 * R_b

        ax_c = px.clamp_min(x_b)
        d2_c = (px - ax_c).square() + (py - y_ret).square()
        s_c = L_in + L_arc + (ax_c - x_b)

        pick_b = d2_b < d2_a
        s5 = torch.where(pick_b, s_b, s_a)
        bd2 = torch.where(pick_b, d2_b, d2_a)
        s5 = torch.where(d2_c < bd2, s_c, s5)
        s_out = torch.where(is_t5[:, None], s5, s_out)

    # Type 6: Smooth conical taper / reducer (straight centerline along -X)
    is_t6 = (ptype > 5.5) & (ptype < 6.5) & active_k
    if bool(is_t6.any()):
        s_out = torch.where(is_t6[:, None], x_en - px, s_out)

    # Type 7: Z-shaped double 90-deg sharp chicane (C1 polar around both inner corners!)
    is_t7 = (ptype > 6.5) & (ptype < 7.5) & active_k
    if bool(is_t7.any()):
        lx = px - x_b
        ly = sgn * (py - y0)
        L_in = (x_en - x_b).clamp_min(0.0)
        L_mid = R_b.clamp_min(0.50)
        ly_half = 0.5 * L_mid
        s_c1_end = (L_in - r_in) + 0.5 * math.pi * r_in
        a_c1 = torch.atan2((r_in - lx).clamp_min(0.0), (r_in - ly).clamp_min(0.0))
        s7_lo = torch.where(lx >= r_in, L_in - lx, torch.where(ly >= r_in, s_c1_end + (ly - r_in), (L_in - r_in) + a_c1 * r_in))

        ux = -lx
        uy = L_mid - ly
        s_c2_start = s_c1_end + (L_mid - 2.0 * r_in)
        a_c2 = torch.atan2((r_in - uy).clamp_min(0.0), (r_in - ux).clamp_min(0.0))
        s7_hi = torch.where(
            uy >= r_in,
            s_c1_end + (ly - r_in),
            torch.where(ux >= r_in, s_c2_start + 0.5 * math.pi * r_in + (ux - r_in), s_c2_start + a_c2 * r_in),
        )
        s7 = torch.where(ly <= ly_half, s7_lo, s7_hi)
        s_out = torch.where(is_t7[:, None], s7, s_out)

    # Type 8: Triple-wave long serpentine pipe
    is_t8 = (ptype > 7.5) & (ptype < 8.5) & active_k
    if bool(is_t8.any()):
        phi = ang.clamp(0.1, 1.35)
        L_in = (x_en - x_b).clamp_min(0.0)
        L_arc = phi * R_b
        dx_a = R_b * torch.sin(phi)
        dy_a = R_b * (1.0 - torch.cos(phi))

        sx1 = px.clamp_min(x_b)
        d1 = (px - sx1).square() + (py - y0).square()
        s1 = x_en - sx1

        c1x, c1y = x_b, y0 + sgn * R_b
        a1 = torch.atan2(-(px - c1x), -sgn * (py - c1y)).clamp(zeros, phi)
        sx2, sy2 = c1x - R_b * torch.sin(a1), c1y - sgn * R_b * torch.cos(a1)
        d2 = (px - sx2).square() + (py - sy2).square()
        s2 = L_in + a1 * R_b

        c2x, c2y = x_b - 2.0 * dx_a, y0 + sgn * (2.0 * dy_a - R_b)
        a2 = torch.atan2(px - c2x, sgn * (py - c2y)).clamp(-phi, phi)
        sx3, sy3 = c2x + R_b * torch.sin(a2), c2y + sgn * R_b * torch.cos(a2)
        d3 = (px - sx3).square() + (py - sy3).square()
        s3 = L_in + L_arc + (phi - a2) * R_b

        c3x, c3y = x_b - 4.0 * dx_a, y0 + sgn * R_b
        a3 = torch.atan2(-(px - c3x), -sgn * (py - c3y)).clamp(-phi, zeros)
        sx4, sy4 = c3x - R_b * torch.sin(a3), c3y - sgn * R_b * torch.cos(a3)
        d4 = (px - sx4).square() + (py - sy4).square()
        s4 = L_in + 3.0 * L_arc + (phi + a3) * R_b

        sx5 = px.clamp_max(c3x)
        d5 = (px - sx5).square() + (py - y0).square()
        s5 = L_in + 4.0 * L_arc + (c3x - sx5)

        sc, bd = s1, d1
        m2 = d2 < bd
        sc, bd = torch.where(m2, s2, sc), torch.where(m2, d2, bd)
        m3 = d3 < bd
        sc, bd = torch.where(m3, s3, sc), torch.where(m3, d3, bd)
        m4 = d4 < bd
        sc, bd = torch.where(m4, s4, sc), torch.where(m4, d4, bd)
        sc = torch.where(d5 < bd, s5, sc)
        s_out = torch.where(is_t8[:, None], sc, s_out)

    return s_out


def pipe_arc_progress_np(xy: np.ndarray, pipe_cfg: list[float] | np.ndarray) -> tuple[float, float]:
    """Return (arc_s_m, total_pipe_len_m) along the pipe centerline for telemetry/eval."""
    pc = np.asarray(pipe_cfg, dtype=np.float64)
    pt = int(round(pc[0]))
    x, y = float(xy[0]), float(xy[1])
    y0, x_en, x_b, R_b, ang, sgn, L_ex = pc[1], pc[3], pc[4], pc[5], pc[6], (1.0 if pc[7] >= 0 else -1.0), pc[8]
    if pt == PIPE_TYPE_STRAIGHT:
        tot = max(1e-3, x_en - x_b)
        return float(x_en - x), tot
    if pt == PIPE_TYPE_BEND90:
        L_in = max(0.0, x_en - x_b)
        L_arc = 0.5 * math.pi * R_b
        tot = L_in + L_arc + L_ex
        cy = y0 + sgn * R_b
        if x > x_b and sgn * (y - y0) < R_b:
            return float(x_en - x), tot
        if sgn * (y - cy) > 0.0:
            return float(L_in + L_arc + sgn * (y - cy)), tot
        a = max(0.0, min(0.5 * math.pi, math.atan2(-(x - x_b), -sgn * (y - cy))))
        return float(L_in + a * R_b), tot
    if pt == PIPE_TYPE_SCURVE:
        L_in = max(0.0, x_en - x_b)
        L_arc = ang * R_b
        tot = L_in + 2.0 * L_arc + L_ex
        dx_a = R_b * math.sin(ang)
        dy_a = R_b * (1.0 - math.cos(ang))
        c2x = x_b - 2.0 * dx_a
        if x > x_b:
            return float(x_en - x), tot
        if x < c2x:
            return float(L_in + 2.0 * L_arc + (c2x - x)), tot
        # Inside S arcs: project monotonically via x-span ratio
        frac = max(0.0, min(1.0, (x_b - x) / max(2.0 * dx_a, 1e-6)))
        return float(L_in + frac * 2.0 * L_arc), tot
    if pt == PIPE_TYPE_L_DUCT:
        L_in = max(0.0, x_en - x_b)
        tot = L_in + L_ex
        ly = sgn * (y - y0)
        lx = x - x_b
        if ly <= abs(lx) and lx > 0.0:
            return float(x_en - x), tot
        return float(L_in + max(0.0, ly)), tot
    if pt == PIPE_TYPE_UTURN:
        L_in = max(0.0, x_en - x_b)
        L_arc = math.pi * R_b
        tot = L_in + L_arc + L_ex
        cy = y0 + sgn * R_b
        if x > x_b and sgn * (y - cy) < 0.0:
            return float(x_en - x), tot
        if x > x_b and sgn * (y - cy) >= 0.0:
            return float(L_in + L_arc + (x - x_b)), tot
        a = math.atan2(max(-(x - x_b), 0.0), -sgn * (y - cy))
        return float(L_in + a * R_b), tot
    if pt == PIPE_TYPE_TAPER:
        L_tap = max(R_b, 0.10)
        tot = max(1e-3, (x_en - x_b) + L_tap + L_ex)
        return float(x_en - x), tot
    if pt == PIPE_TYPE_Z_DUCT:
        L_in = max(0.0, x_en - x_b)
        L_mid = max(R_b, 0.50)
        tot = L_in + L_mid + L_ex
        ly = sgn * (y - y0)
        lx = x - x_b
        if ly <= 0.5 * L_mid:
            if ly <= abs(lx) and lx > 0.0:
                return float(x_en - x), tot
            return float(L_in + max(0.0, ly)), tot
        ux = -lx
        uy = L_mid - ly
        if ux <= abs(uy) and uy > 0.0:
            return float(L_in + ly), tot
        return float(L_in + L_mid + max(0.0, ux)), tot
    if pt == PIPE_TYPE_TRIPLE_WAVE:
        phi = max(0.1, min(1.35, ang))
        L_in = max(0.0, x_en - x_b)
        L_arc = phi * R_b
        tot = L_in + 4.0 * L_arc + L_ex
        dx_a = R_b * math.sin(phi)
        c3x = x_b - 4.0 * dx_a
        if x > x_b:
            return float(x_en - x), tot
        if x < c3x:
            return float(L_in + 4.0 * L_arc + (c3x - x)), tot
        frac = max(0.0, min(1.0, (x_b - x) / max(4.0 * dx_a, 1e-6)))
        return float(L_in + frac * 4.0 * L_arc), tot
    return float(-x), 1.0


def build_pipe_mujoco_geoms(pipe_cfg: list[float] | np.ndarray) -> str:
    """Generate crystal-clear MuJoCo transparent acrylic pipe / 90-deg bend / S-curve / L-duct geoms."""
    pc = np.asarray(pipe_cfg, dtype=np.float64)
    pt = int(round(pc[0]))
    if pt == PIPE_TYPE_OFF or pc[10] <= 0.0:
        return ''
    y0, r_in, x_en, x_b, R_b, ang, sgn, L_ex, z_c, _, r_fil = (
        pc[1], pc[2], pc[3], pc[4], pc[5], pc[6], (1.0 if pc[7] >= 0 else -1.0), pc[8], pc[9], pc[10], pc[11]
    )
    geoms: list[str] = []

    def add_cyl_section(p0: np.ndarray, p1: np.ndarray, n_staves: int = 28, r_override: float | None = None) -> None:
        d = p1 - p0
        seg_len = float(np.linalg.norm(d))
        if seg_len < 1e-4:
            return
        mid = 0.5 * (p0 + p1)
        yaw = math.atan2(d[1], d[0])
        tx, ty = math.cos(yaw), math.sin(yaw)
        nx, ny = -ty, tx  # local +Y perpendicular to tube axis in XY plane
        r_use = r_in if r_override is None else float(r_override)
        r_out = r_use + 0.005
        r_mid = 0.5 * (r_use + r_out)
        dth = 2.0 * math.pi / n_staves
        stave_hw = r_mid * math.tan(0.5 * dth) * 1.06
        for k in range(n_staves):
            th = (k + 0.5) * dth
            c_t, s_t = math.cos(th), math.sin(th)
            px = mid[0] + r_mid * c_t * nx
            py = mid[1] + r_mid * c_t * ny
            pz = z_c + r_mid * s_t
            if pz < 0.003:
                continue
            mat = 'mat_acrylic_rim' if abs(s_t) > 0.88 else 'mat_acrylic_tube'
            # Rotation matrix columns: [tube_axis, radial_out, circumferential]
            Rmat = np.array([
                [tx, nx * c_t, -nx * s_t],
                [ty, ny * c_t, -ny * s_t],
                [0.0, s_t, c_t],
            ], dtype=np.float64)
            # Convert Rmat to MuJoCo wxyz quaternion
            tr = Rmat[0, 0] + Rmat[1, 1] + Rmat[2, 2]
            if tr > 0.0:
                s = math.sqrt(tr + 1.0) * 2.0
                qw, qx, qy, qz = 0.25 * s, (Rmat[2, 1] - Rmat[1, 2]) / s, (Rmat[0, 2] - Rmat[2, 0]) / s, (Rmat[1, 0] - Rmat[0, 1]) / s
            elif Rmat[0, 0] > Rmat[1, 1] and Rmat[0, 0] > Rmat[2, 2]:
                s = math.sqrt(1.0 + Rmat[0, 0] - Rmat[1, 1] - Rmat[2, 2]) * 2.0
                qw, qx, qy, qz = (Rmat[2, 1] - Rmat[1, 2]) / s, 0.25 * s, (Rmat[0, 1] + Rmat[1, 0]) / s, (Rmat[0, 2] + Rmat[2, 0]) / s
            elif Rmat[1, 1] > Rmat[2, 2]:
                s = math.sqrt(1.0 + Rmat[1, 1] - Rmat[0, 0] - Rmat[2, 2]) * 2.0
                qw, qx, qy, qz = (Rmat[0, 2] - Rmat[2, 0]) / s, (Rmat[0, 1] + Rmat[1, 0]) / s, 0.25 * s, (Rmat[1, 2] + Rmat[2, 1]) / s
            else:
                s = math.sqrt(1.0 + Rmat[2, 2] - Rmat[0, 0] - Rmat[1, 1]) * 2.0
                qw, qx, qy, qz = (Rmat[1, 0] - Rmat[0, 1]) / s, (Rmat[0, 2] + Rmat[2, 0]) / s, (Rmat[1, 2] + Rmat[2, 1]) / s, 0.25 * s
            geoms.append(
                f'    <geom type="box" pos="{px:.4f} {py:.4f} {pz:.4f}" '
                f'quat="{qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f}" '
                f'size="{0.52 * seg_len:.4f} 0.0025 {stave_hw:.4f}" '
                f'material="{mat}" contype="0" conaffinity="0"/>'
            )

    def add_collar_ring(pt_xy: np.ndarray, yaw: float, r_override: float | None = None) -> None:
        tx, ty = math.cos(yaw), math.sin(yaw)
        nx, ny = -ty, tx
        r_ring = (r_in if r_override is None else float(r_override)) + 0.010
        for k in range(24):
            th = (k + 0.5) * (2.0 * math.pi / 24)
            c_t, s_t = math.cos(th), math.sin(th)
            px = pt_xy[0] + r_ring * c_t * nx
            py = pt_xy[1] + r_ring * c_t * ny
            pz = z_c + r_ring * s_t
            if pz < 0.006:
                continue
            Rmat = np.array([[tx, nx * c_t, -nx * s_t], [ty, ny * c_t, -ny * s_t], [0.0, s_t, c_t]])
            tr = Rmat[0, 0] + Rmat[1, 1] + Rmat[2, 2]
            if tr > 0.0:
                s = math.sqrt(tr + 1.0) * 2.0
                qw, qx, qy, qz = 0.25 * s, (Rmat[2, 1] - Rmat[1, 2]) / s, (Rmat[0, 2] - Rmat[2, 0]) / s, (Rmat[1, 0] - Rmat[0, 1]) / s
            elif Rmat[0, 0] > Rmat[1, 1] and Rmat[0, 0] > Rmat[2, 2]:
                s = math.sqrt(1.0 + Rmat[0, 0] - Rmat[1, 1] - Rmat[2, 2]) * 2.0
                qw, qx, qy, qz = (Rmat[2, 1] - Rmat[1, 2]) / s, 0.25 * s, (Rmat[0, 1] + Rmat[1, 0]) / s, (Rmat[0, 2] + Rmat[2, 0]) / s
            elif Rmat[1, 1] > Rmat[2, 2]:
                s = math.sqrt(1.0 + Rmat[1, 1] - Rmat[0, 0] - Rmat[2, 2]) * 2.0
                qw, qx, qy, qz = (Rmat[0, 2] - Rmat[2, 0]) / s, (Rmat[0, 1] + Rmat[1, 0]) / s, 0.25 * s, (Rmat[1, 2] + Rmat[2, 1]) / s
            else:
                s = math.sqrt(1.0 + Rmat[2, 2] - Rmat[0, 0] - Rmat[1, 1]) * 2.0
                qw, qx, qy, qz = (Rmat[1, 0] - Rmat[0, 1]) / s, (Rmat[0, 2] + Rmat[2, 0]) / s, (Rmat[1, 2] + Rmat[2, 1]) / s, 0.25 * s
            geoms.append(
                f'    <geom type="box" pos="{px:.4f} {py:.4f} {pz:.4f}" '
                f'quat="{qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f}" '
                f'size="0.008 0.0055 0.018" material="mat_alu_ring" contype="0" conaffinity="0"/>'
            )

    if pt == PIPE_TYPE_STRAIGHT:
        p0 = np.array([x_en, y0])
        p1 = np.array([x_b, y0])
        add_cyl_section(p0, p1)
        add_collar_ring(p0, math.pi)
        add_collar_ring(p1, math.pi)
    elif pt == PIPE_TYPE_BEND90:
        p0 = np.array([x_en, y0])
        p1 = np.array([x_b, y0])
        add_cyl_section(p0, p1)
        add_collar_ring(p0, math.pi)
        cy = y0 + sgn * R_b
        n_arc = 12
        pts = [np.array([x_b - R_b * math.sin(a), cy - sgn * R_b * math.cos(a)]) for a in np.linspace(0.0, 0.5 * math.pi, n_arc + 1)]
        for k in range(n_arc):
            add_cyl_section(pts[k], pts[k + 1], n_staves=24)
        p_ex = np.array([x_b - R_b, cy + sgn * L_ex])
        add_cyl_section(pts[-1], p_ex)
        add_collar_ring(p1, math.pi)
        add_collar_ring(pts[-1], 0.5 * math.pi * sgn)
        add_collar_ring(p_ex, 0.5 * math.pi * sgn)
    elif pt == PIPE_TYPE_SCURVE:
        dx_a = R_b * math.sin(ang)
        dy_a = R_b * (1.0 - math.cos(ang))
        p0 = np.array([x_en, y0])
        p1 = np.array([x_b, y0])
        add_cyl_section(p0, p1)
        add_collar_ring(p0, math.pi)
        c1y = y0 + sgn * R_b
        c2x = x_b - 2.0 * dx_a
        c2y = y0 + sgn * (2.0 * dy_a - R_b)
        n_arc = 9
        arc1 = [np.array([x_b - R_b * math.sin(a), c1y - sgn * R_b * math.cos(a)]) for a in np.linspace(0.0, ang, n_arc + 1)]
        for k in range(n_arc):
            add_cyl_section(arc1[k], arc1[k + 1], n_staves=24)
        arc2 = [np.array([c2x + R_b * math.sin(a), c2y + sgn * R_b * math.cos(a)]) for a in np.linspace(ang, 0.0, n_arc + 1)]
        for k in range(n_arc):
            add_cyl_section(arc2[k], arc2[k + 1], n_staves=24)
        p_ex = np.array([c2x - L_ex, y0 + sgn * 2.0 * dy_a])
        add_cyl_section(arc2[-1], p_ex)
        add_collar_ring(arc1[-1], math.pi - sgn * ang)
        add_collar_ring(p_ex, math.pi)
    elif pt == PIPE_TYPE_L_DUCT:
        # Crystal-clear rectangular acrylic L-duct (5 mm thick walls + brushed aluminum frame rails!)
        hw = r_in
        H = z_c
        wt = 0.003
        # 1. Outer dead-end wall at x = x_b - hw, y from y0 - sgn*hw to y0 + sgn*L_ex
        y_lo = min(y0 - sgn * hw, y0 + sgn * L_ex)
        y_hi = max(y0 - sgn * hw, y0 + sgn * L_ex)
        geoms.append(
            f'    <geom type="box" pos="{x_b - hw - wt:.4f} {0.5*(y_lo+y_hi):.4f} {0.5*H:.4f}" '
            f'size="{wt:.4f} {0.5*(y_hi-y_lo):.4f} {0.5*H:.4f}" material="mat_acrylic_rim" contype="0" conaffinity="0"/>'
        )
        # 2. Outer entry side wall at y = y0 - sgn*hw, x from x_b - hw to x_en
        geoms.append(
            f'    <geom type="box" pos="{0.5*(x_b - hw + x_en):.4f} {y0 - sgn*(hw + wt):.4f} {0.5*H:.4f}" '
            f'size="{0.5*(x_en - (x_b - hw)):.4f} {wt:.4f} {0.5*H:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
        )
        # 3. Inner entry side wall at y = y0 + sgn*hw, x from x_b + hw to x_en
        if x_en > x_b + hw:
            geoms.append(
                f'    <geom type="box" pos="{0.5*(x_b + hw + x_en):.4f} {y0 + sgn*(hw + wt):.4f} {0.5*H:.4f}" '
                f'size="{0.5*(x_en - (x_b + hw)):.4f} {wt:.4f} {0.5*H:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
            )
        # 4. Inner exit wall at x = x_b + hw, y from y0 + sgn*hw to y0 + sgn*L_ex
        if L_ex > hw:
            y_il = min(y0 + sgn * hw, y0 + sgn * L_ex)
            y_ih = max(y0 + sgn * hw, y0 + sgn * L_ex)
            geoms.append(
                f'    <geom type="box" pos="{x_b + hw + wt:.4f} {0.5*(y_il+y_ih):.4f} {0.5*H:.4f}" '
                f'size="{wt:.4f} {0.5*(y_ih-y_il):.4f} {0.5*H:.4f}" material="mat_acrylic_rim" contype="0" conaffinity="0"/>'
            )
        # 5. Inner sharp-corner brushed aluminum vertical post (highlighting the 90-deg sharp corner!)
        geoms.append(
            f'    <geom type="cylinder" pos="{x_b + hw + r_fil:.4f} {y0 + sgn*(hw + r_fil):.4f} {0.5*H:.4f}" '
            f'size="{r_fil:.4f} {0.5*H:.4f}" material="mat_alu_ring" contype="0" conaffinity="0"/>'
        )
        # 6. Top transparent acrylic ceiling panels (entry arm + exit arm)
        geoms.append(
            f'    <geom type="box" pos="{0.5*(x_b - hw + x_en):.4f} {y0:.4f} {H + wt:.4f}" '
            f'size="{0.5*(x_en - (x_b - hw)):.4f} {hw:.4f} {wt:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
        )
        if L_ex > hw:
            y_tl = min(y0 + sgn * hw, y0 + sgn * L_ex)
            y_th = max(y0 + sgn * hw, y0 + sgn * L_ex)
            geoms.append(
                f'    <geom type="box" pos="{x_b:.4f} {0.5*(y_tl+y_th):.4f} {H + wt:.4f}" '
                f'size="{hw:.4f} {0.5*(y_th-y_tl):.4f} {wt:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
            )
        # 7. Crisp aluminum edge trim rails along base & top corners for glass readability
        for rz in (0.005, H):
            geoms.append(
                f'    <geom type="box" pos="{x_b - hw - wt:.4f} {0.5*(y_lo+y_hi):.4f} {rz:.4f}" '
                f'size="0.005 {0.5*(y_hi-y_lo):.4f} 0.005" material="mat_alu_ring" contype="0" conaffinity="0"/>'
            )
            geoms.append(
                f'    <geom type="box" pos="{0.5*(x_b - hw + x_en):.4f} {y0 - sgn*(hw + wt):.4f} {rz:.4f}" '
                f'size="{0.5*(x_en - (x_b - hw)):.4f} 0.005 0.005" material="mat_alu_ring" contype="0" conaffinity="0"/>'
            )
    elif pt == PIPE_TYPE_UTURN:
        p0 = np.array([x_en, y0])
        p1 = np.array([x_b, y0])
        add_cyl_section(p0, p1)
        add_collar_ring(p0, math.pi)
        cy = y0 + sgn * R_b
        n_arc = 20
        pts = [np.array([x_b - R_b * math.sin(a), cy - sgn * R_b * math.cos(a)]) for a in np.linspace(0.0, math.pi, n_arc + 1)]
        for k in range(n_arc):
            add_cyl_section(pts[k], pts[k + 1], n_staves=22)
        p_ex = np.array([x_b + L_ex, y0 + 2.0 * sgn * R_b])
        add_cyl_section(pts[-1], p_ex)
        add_collar_ring(p1, math.pi)
        add_collar_ring(pts[n_arc // 2], 0.5 * math.pi * sgn)
        add_collar_ring(pts[-1], 0.0)
        add_collar_ring(p_ex, 0.0)
    elif pt == PIPE_TYPE_TAPER:
        L_tap = max(R_b, 0.10)
        r_out = max(r_fil, 0.080)
        p0 = np.array([x_en, y0])
        p1 = np.array([x_b, y0])
        add_cyl_section(p0, p1, n_staves=24, r_override=r_in)
        add_collar_ring(p0, math.pi, r_override=r_in)
        add_collar_ring(p1, math.pi, r_override=r_in)
        n_tap = 10
        xs_t = np.linspace(x_b, x_b - L_tap, n_tap + 1)
        for k in range(n_tap):
            xm = 0.5 * (xs_t[k] + xs_t[k + 1])
            u = (x_b - xm) / L_tap
            r_k = r_in + 0.5 * (r_out - r_in) * (1.0 - math.cos(math.pi * u))
            add_cyl_section(np.array([xs_t[k], y0]), np.array([xs_t[k + 1], y0]), n_staves=22, r_override=r_k)
        p2 = np.array([x_b - L_tap, y0])
        p3 = np.array([x_b - L_tap - L_ex, y0])
        add_cyl_section(p2, p3, n_staves=24, r_override=r_out)
        add_collar_ring(p2, math.pi, r_override=r_out)
        add_collar_ring(p3, math.pi, r_override=r_out)
    elif pt == PIPE_TYPE_Z_DUCT:
        hw = r_in
        H = z_c
        wt = 0.003
        L_mid = max(R_b, 0.50)
        y2 = y0 + sgn * L_mid
        # Entry outer wall (y = y0 - sgn*hw, x from x_b - hw to x_en)
        geoms.append(
            f'    <geom type="box" pos="{0.5*(x_b - hw + x_en):.4f} {y0 - sgn*(hw + wt):.4f} {0.5*H:.4f}" '
            f'size="{0.5*(x_en - (x_b - hw)):.4f} {wt:.4f} {0.5*H:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
        )
        # Entry inner wall (y = y0 + sgn*hw, x from x_b + hw to x_en)
        if x_en > x_b + hw:
            geoms.append(
                f'    <geom type="box" pos="{0.5*(x_b + hw + x_en):.4f} {y0 + sgn*(hw + wt):.4f} {0.5*H:.4f}" '
                f'size="{0.5*(x_en - (x_b + hw)):.4f} {wt:.4f} {0.5*H:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
            )
        # Middle left wall (x = x_b - hw, y from y0 - sgn*hw to y2 - sgn*hw)
        y_ml, y_mh = min(y0 - sgn * hw, y2 - sgn * hw), max(y0 - sgn * hw, y2 - sgn * hw)
        geoms.append(
            f'    <geom type="box" pos="{x_b - hw - wt:.4f} {0.5*(y_ml+y_mh):.4f} {0.5*H:.4f}" '
            f'size="{wt:.4f} {0.5*(y_mh-y_ml):.4f} {0.5*H:.4f}" material="mat_acrylic_rim" contype="0" conaffinity="0"/>'
        )
        # Middle right wall (x = x_b + hw, y from y0 + sgn*hw to y2 + sgn*hw)
        y_rl, y_rh = min(y0 + sgn * hw, y2 + sgn * hw), max(y0 + sgn * hw, y2 + sgn * hw)
        geoms.append(
            f'    <geom type="box" pos="{x_b + hw + wt:.4f} {0.5*(y_rl+y_rh):.4f} {0.5*H:.4f}" '
            f'size="{wt:.4f} {0.5*(y_rh-y_rl):.4f} {0.5*H:.4f}" material="mat_acrylic_rim" contype="0" conaffinity="0"/>'
        )
        # Exit outer wall (y = y2 + sgn*hw, x from x_b - L_ex to x_b + hw)
        geoms.append(
            f'    <geom type="box" pos="{0.5*(x_b - L_ex + x_b + hw):.4f} {y2 + sgn*(hw + wt):.4f} {0.5*H:.4f}" '
            f'size="{0.5*(L_ex + hw):.4f} {wt:.4f} {0.5*H:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
        )
        # Exit inner wall (y = y2 - sgn*hw, x from x_b - L_ex to x_b - hw)
        if L_ex > hw:
            geoms.append(
                f'    <geom type="box" pos="{0.5*(x_b - L_ex + x_b - hw):.4f} {y2 - sgn*(hw + wt):.4f} {0.5*H:.4f}" '
                f'size="{0.5*(L_ex - hw):.4f} {wt:.4f} {0.5*H:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
            )
        # Both sharp inner-corner aluminum posts
        geoms.append(
            f'    <geom type="cylinder" pos="{x_b + hw + r_fil:.4f} {y0 + sgn*(hw + r_fil):.4f} {0.5*H:.4f}" '
            f'size="{r_fil:.4f} {0.5*H:.4f}" material="mat_alu_ring" contype="0" conaffinity="0"/>'
        )
        geoms.append(
            f'    <geom type="cylinder" pos="{x_b - hw - r_fil:.4f} {y2 - sgn*(hw + r_fil):.4f} {0.5*H:.4f}" '
            f'size="{r_fil:.4f} {0.5*H:.4f}" material="mat_alu_ring" contype="0" conaffinity="0"/>'
        )
        # Transparent ceiling panels (entry, middle, exit)
        geoms.append(
            f'    <geom type="box" pos="{0.5*(x_b - hw + x_en):.4f} {y0:.4f} {H + wt:.4f}" '
            f'size="{0.5*(x_en - (x_b - hw)):.4f} {hw:.4f} {wt:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
        )
        y_cl, y_ch = min(y0 + sgn * hw, y2 - sgn * hw), max(y0 + sgn * hw, y2 - sgn * hw)
        if y_ch > y_cl + 1e-4:
            geoms.append(
                f'    <geom type="box" pos="{x_b:.4f} {0.5*(y_cl+y_ch):.4f} {H + wt:.4f}" '
                f'size="{hw:.4f} {0.5*(y_ch-y_cl):.4f} {wt:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
            )
        geoms.append(
            f'    <geom type="box" pos="{0.5*(x_b - L_ex + x_b + hw):.4f} {y2:.4f} {H + wt:.4f}" '
            f'size="{0.5*(L_ex + hw):.4f} {hw:.4f} {wt:.4f}" material="mat_acrylic_tube" contype="0" conaffinity="0"/>'
        )
    elif pt == PIPE_TYPE_TRIPLE_WAVE:
        phi = max(0.1, min(1.35, ang))
        dx_a = R_b * math.sin(phi)
        dy_a = R_b * (1.0 - math.cos(phi))
        p0 = np.array([x_en, y0])
        p1 = np.array([x_b, y0])
        add_cyl_section(p0, p1)
        add_collar_ring(p0, math.pi)
        c1y = y0 + sgn * R_b
        c2x, c2y = x_b - 2.0 * dx_a, y0 + sgn * (2.0 * dy_a - R_b)
        c3x, c3y = x_b - 4.0 * dx_a, y0 + sgn * R_b
        n1 = 7
        arc1 = [np.array([x_b - R_b * math.sin(a), c1y - sgn * R_b * math.cos(a)]) for a in np.linspace(0.0, phi, n1 + 1)]
        for k in range(n1):
            add_cyl_section(arc1[k], arc1[k + 1], n_staves=20)
        arc2 = [np.array([c2x + R_b * math.sin(a), c2y + sgn * R_b * math.cos(a)]) for a in np.linspace(phi, -phi, 2 * n1 + 1)]
        for k in range(2 * n1):
            add_cyl_section(arc2[k], arc2[k + 1], n_staves=20)
        arc3 = [np.array([c3x - R_b * math.sin(a), c3y - sgn * R_b * math.cos(a)]) for a in np.linspace(-phi, 0.0, n1 + 1)]
        for k in range(n1):
            add_cyl_section(arc3[k], arc3[k + 1], n_staves=20)
        p_ex = np.array([c3x - L_ex, y0])
        add_cyl_section(arc3[-1], p_ex)
        add_collar_ring(arc1[-1], math.pi - sgn * phi)
        add_collar_ring(arc2[-1], math.pi + sgn * phi)
        add_collar_ring(p_ex, math.pi)
    return '\n'.join(geoms)

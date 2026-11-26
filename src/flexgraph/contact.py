"""Unified Implicit Anisotropic Patch Contact (ARPC) & Py-DiSMech C2 Self-Contact Plugin.

Provides two composable FP64 contact operators:
  1. ``eval_anisotropic_patch_contact`` -- unifies three physical contact regimes in one
     branch-minimal tensor kernel returning (wrench_6d, K_tan_6x6, normal, next_state):
       * ``ratchet_wheel``     : one-way clutch wheel with spin inertia, rolling drag & backlash
       * ``anisotropic_scale`` : wheelless directional friction (mu_fwd << mu_rev, mu_lat)
                                 for limbless snakes, bristle earthworms, and inchworms
       * ``isotropic_pad``     : isotropic Coulomb stick-slip + circular patch pivot damping
                                 for legged feet, catheter rims, and beam wall nodes
  2. ``eval_c2_self_contact`` -- exact Py-DiSMech (arXiv:2512.09911, Eq. 25-27) piecewise C2
     smooth soft-barrier self-collision energy, analytical force, and PSD diagonal Hessian block.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from flexgraph.math3d import cross, norm, skew, unit

CONTACT_MODE_RATCHET_WHEEL = 0
CONTACT_MODE_ANISOTROPIC_SCALE = 1
CONTACT_MODE_ISOTROPIC_PAD = 2

CONTACT_MODE_MAP = {
    'ratchet_wheel': CONTACT_MODE_RATCHET_WHEEL,
    'anisotropic_scale': CONTACT_MODE_ANISOTROPIC_SCALE,
    'isotropic_pad': CONTACT_MODE_ISOTROPIC_PAD,
}


@dataclass
class ContactState:
    """Device-resident per-site implicit contact state for batch B and M sites."""

    rut: torch.Tensor          # (B, M) Bekker plastic sinkage depth (m)
    memory: torch.Tensor       # (B, M, 3) world-frame tangential stick-slip elastic deformation (m)
    spin: torch.Tensor         # (B, M) wheel angular velocity (rad/s) for ratchet_wheel mode
    angle: torch.Tensor        # (B, M) integrated wheel rotation angle (rad)
    ratchet_play: torch.Tensor # (B, M) backlash angle buffer in [0, backlash_rad]

    @classmethod
    def zeros(cls, batch: int, n_sites: int, device: torch.device, dtype: torch.dtype) -> 'ContactState':
        return cls(
            rut=torch.zeros((batch, n_sites), device=device, dtype=dtype),
            memory=torch.zeros((batch, n_sites, 3), device=device, dtype=dtype),
            spin=torch.zeros((batch, n_sites), device=device, dtype=dtype),
            angle=torch.zeros((batch, n_sites), device=device, dtype=dtype),
            ratchet_play=torch.zeros((batch, n_sites), device=device, dtype=dtype),
        )


def hunt_crossley_bekker_normal(
    pen: torch.Tensor,
    vn: torch.Tensor,
    v_horiz: torch.Tensor,
    rut: torch.Tensor,
    dt: float,
    hc_k: torch.Tensor,
    hc_alpha: torch.Tensor,
    soil_exp: torch.Tensor,
    unload_ratio: torch.Tensor,
    rut_relax_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-impulse Hunt-Crossley normal force with Bekker elasto-plastic rut hysteresis.

    Returns ``(normal_n, new_rut, kn_tan, cn_tan)`` with guaranteed non-negative ``kn_tan, cn_tan``.
    """
    zeros = torch.zeros_like(pen)
    pen_pos = pen.clamp_min(0.0)
    has_unload = unload_ratio > 1.0
    inv_relax = 1.0 / rut_relax_m.clamp_min(1e-6)
    rut_rel_u = rut * torch.exp(-v_horiz * dt * inv_relax)
    ur_safe = unload_ratio.clamp_min(1.0)
    plastic_frac = 1.0 - 1.0 / ur_safe
    new_rut_u = torch.maximum(rut_rel_u, plastic_frac * pen_pos)
    rut_rel = torch.where(has_unload, rut_rel_u, zeros)
    new_rut = torch.where(has_unload, new_rut_u, zeros)

    in_contact = (pen > rut_rel) & (pen > 0.0)
    pen_safe = torch.where(in_contact, pen, torch.ones_like(pen))
    pen_pm1 = torch.pow(pen_safe, soil_exp - 1.0)
    pen_pm2 = pen_pm1 / pen_safe

    virgin_el = hc_k * pen_safe * pen_pm1
    dpen = (pen_safe - rut_rel).clamp_min(0.0)
    unload_el = unload_ratio * hc_k * dpen * pen_pm1
    is_unload = has_unload & (unload_el < virgin_el)
    f_el = torch.where(in_contact, torch.where(is_unload, unload_el, virgin_el), zeros)
    df_unload = unload_ratio * hc_k * (pen_pm1 + (soil_exp - 1.0) * dpen * pen_pm2)
    df_virgin = soil_exp * hc_k * pen_pm1
    df_el = torch.where(in_contact, torch.where(is_unload, df_unload, df_virgin), zeros)

    damp_raw = 1.0 - hc_alpha * vn
    damp_pos = damp_raw > 0.0
    damp_mult = torch.where(damp_pos, damp_raw, zeros)

    normal = f_el * damp_mult
    active_tan = in_contact & damp_pos
    kn_tan = torch.where(active_tan, df_el * damp_mult, zeros)
    cn_tan = torch.where(active_tan, f_el * hc_alpha, zeros)
    return normal, new_rut, kn_tan, cn_tan


def eval_anisotropic_patch_contact(
    *,
    pen: torch.Tensor,            # (B, M) positive penetration depth (m)
    normal_w: torch.Tensor,       # (B, M, 3) outward unit surface normal
    vel_site: torch.Tensor,       # (B, M, 3) world linear velocity at contact patch
    omega_owner: torch.Tensor,    # (B, M, 3) world angular velocity of owner rigid body
    arm_w: torch.Tensor,          # (B, M, 3) lever arm from owner body CoM to contact patch
    rim_w: torch.Tensor,          # (B, M, 3) wheel rim vector (0 for non-wheel modes)
    axle_w: torch.Tensor,         # (B, M, 3) world axle direction (for ratchet_wheel)
    fwd_hint_w: torch.Tensor,     # (B, M, 3) preferred forward direction (projected onto tangent plane)
    mode_id: torch.Tensor,        # (M,) int64 in {0: ratchet_wheel, 1: anisotropic_scale, 2: isotropic_pad}
    radius_m: torch.Tensor,       # (M,)
    patch_radius_m: torch.Tensor, # (M,)
    mu_fwd: torch.Tensor,         # (B, M)
    mu_rev: torch.Tensor,         # (B, M)
    mu_lat: torch.Tensor,         # (B, M)
    rolling: torch.Tensor,        # (B, M)
    wheel_mass_kg: torch.Tensor,  # (M,)
    backlash_rad: torch.Tensor,   # (B, M)
    static_mu_ratio: torch.Tensor,# (B, M)
    pivot_mu: torch.Tensor,       # (M,)
    hc_k: torch.Tensor,           # (B, M)
    hc_alpha: torch.Tensor,       # (B, M)
    soil_exp: torch.Tensor,       # (B, M)
    unload_ratio: torch.Tensor,   # (B, M)
    rut_relax_m: torch.Tensor,    # (B, M)
    rut_lat_k: torch.Tensor,      # (B, M)
    state: ContactState,
    dt: float,
    use_kelvin_normal: bool = False,
    kelvin_kn: float = 5000.0,
    kelvin_cn: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ContactState]:
    """Unified implicit anisotropic patch contact returning force, torque, SPD 6x6 K_tan, normal, and next_state.

    Returns:
        (force_w (B,M,3), torque_w (B,M,3), K_tan_6x6 (B,M,6,6), normal_n (B,M), next_state)
    """
    device, dtype = pen.device, pen.dtype
    is_wheel = (mode_id == CONTACT_MODE_RATCHET_WHEEL)[None, :]
    is_scale = (mode_id == CONTACT_MODE_ANISOTROPIC_SCALE)[None, :]

    # Tangent plane orthonormal basis (fwd, lat) orthogonal to normal_w
    fwd_wheel = unit(cross(normal_w, axle_w))
    fwd_proj = fwd_hint_w - (fwd_hint_w * normal_w).sum(-1, keepdim=True) * normal_w
    fwd_other = unit(fwd_proj)
    forward = torch.where(is_wheel[..., None], fwd_wheel, fwd_other)
    lateral = unit(cross(normal_w, forward))
    basis = torch.stack((forward, lateral), dim=-2)  # (B, M, 2, 3)

    vn = (vel_site * normal_w).sum(-1)
    v_tan = torch.einsum('bmkc,bmc->bmk', basis, vel_site)  # (B, M, 2)
    mem_loc = torch.einsum('bmkc,bmc->bmk', basis, state.memory)
    v_horiz = torch.hypot(v_tan[..., 0], v_tan[..., 1])

    # 1. Normal channel: Kelvin (parity mode) or Hunt-Crossley + Bekker elasto-plastic rut
    if use_kelvin_normal:
        raw_n = kelvin_kn * pen - kelvin_cn * vn
        active = (pen > 0.0) & (raw_n > 0.0)
        normal = torch.where(active, raw_n, torch.zeros_like(pen))
        kn_tan = torch.where(active, torch.full_like(pen, kelvin_kn), torch.zeros_like(pen))
        cn_tan = torch.where(active, torch.full_like(pen, kelvin_cn), torch.zeros_like(pen))
        new_rut = torch.zeros_like(pen)
    else:
        normal, new_rut, kn_tan, cn_tan = hunt_crossley_bekker_normal(
            pen, vn, v_horiz, state.rut, dt, hc_k, hc_alpha, soil_exp, unload_ratio, rut_relax_m
        )
        active = normal > 0.0

    # 2. Stribeck speed-dependent static-to-kinetic friction multiplier
    v_str = 0.04
    stribeck = 1.0 + (static_mu_ratio - 1.0) * torch.exp(-((v_horiz / v_str) ** 2))
    mu_f_eff = mu_fwd * stribeck
    mu_r_eff = mu_rev * stribeck
    mu_l_eff = mu_lat * stribeck

    # 3. Tangential elastic-plastic stick-slip + optional ratchet wheel spin
    k_t = 2000.0
    c_t = 8.0
    gain = k_t * dt + c_t
    base_f = -k_t * mem_loc - gain * v_tan  # (B, M, 2)

    # Branch A: Ratchet Wheel (spin inertia + rolling resistance + backlash play)
    rad = radius_m[None, :].clamp_min(1e-6)
    inertia = (0.5 * wheel_mass_kg[None, :] * rad * rad).clamp_min(1e-12)
    viscous = 2e-6
    roll_lim = rolling * normal * rad
    # Freewheel omega (clamped >= 0 for one-way clutch, with optional reverse backlash play)
    omega_free = (inertia * state.spin / dt - rad * base_f[..., 0] - roll_lim) / (
        inertia / dt + rad * rad * gain + viscous
    )
    play_avail = (backlash_rad - state.ratchet_play).clamp_min(0.0)
    omega_min = -play_avail / max(dt, 1e-9)
    omega_w = torch.maximum(omega_free, omega_min)
    f_wheel = base_f.clone()
    f_wheel[..., 0] = f_wheel[..., 0] + gain * rad * omega_w

    # Anisotropic Coulomb friction ellipse for wheel
    cap_w_f = (mu_f_eff * normal).clamp_min(1e-15)
    cap_w_l = (mu_l_eff * normal).clamp_min(1e-15)
    ell_w = torch.hypot(f_wheel[..., 0] / cap_w_f, f_wheel[..., 1] / cap_w_l)
    sliding_w = ell_w > 1.0
    scale_w = torch.minimum(torch.ones_like(ell_w), 1.0 / ell_w.clamp_min(1e-15))
    f_wheel = f_wheel * scale_w[..., None]
    omega_w = torch.maximum(
        (inertia * state.spin / dt - rad * f_wheel[..., 0] - roll_lim) / (inertia / dt + viscous),
        omega_min,
    )
    reaction_w = inertia * (omega_w - state.spin) / dt + rad * f_wheel[..., 0]
    slip_w = v_tan.clone()
    slip_w[..., 0] = slip_w[..., 0] - rad * omega_w
    new_play_w = torch.where(
        active,
        (state.ratchet_play - dt * omega_w).clamp(torch.zeros_like(backlash_rad), backlash_rad),
        state.ratchet_play,
    )
    # Implicit tangential stiffness scalar for wheel: low along rolling direction when rolling freely
    rolling_free = active & (omega_w > omega_min + 1e-6)
    k_t0_w = torch.where(active, torch.where(rolling_free, torch.zeros_like(normal), torch.where(sliding_w, 0.25 * gain, torch.full_like(normal, gain))), torch.zeros_like(normal))
    k_t1_w = torch.where(active, torch.where(sliding_w, 0.25 * gain, torch.full_like(normal, gain)), torch.zeros_like(normal))

    # Branch B: Anisotropic Scale / Isotropic Pad (no wheel spin; directional friction mu_fwd vs mu_rev!)
    # If pushing forward along +forward (v_tan[...,0] > 0 => opposing force f[...,0] < 0), use mu_fwd;
    # if resisting backward slip (v_tan[...,0] < 0 => forward thrust f[...,0] > 0), use mu_rev!
    mu_long = torch.where(is_scale, torch.where(base_f[..., 0] <= 0.0, mu_f_eff, mu_r_eff), mu_f_eff)
    cap_s_0 = (mu_long * normal).clamp_min(1e-15)
    cap_s_1 = (mu_l_eff * normal).clamp_min(1e-15)
    ell_s = torch.hypot(base_f[..., 0] / cap_s_0, base_f[..., 1] / cap_s_1)
    sliding_s = ell_s > 1.0
    scale_s = torch.minimum(torch.ones_like(ell_s), 1.0 / ell_s.clamp_min(1e-15))
    f_pad = base_f * scale_s[..., None]
    slip_s = v_tan
    k_t0_s = torch.where(active, torch.where(sliding_s, 0.25 * gain, torch.full_like(normal, gain)), torch.zeros_like(normal))
    k_t1_s = torch.where(active, torch.where(sliding_s, 0.25 * gain, torch.full_like(normal, gain)), torch.zeros_like(normal))

    # Select active branch per site
    tang = torch.where(is_wheel[..., None], f_wheel, f_pad)
    sliding = torch.where(is_wheel, sliding_w, sliding_s)
    slip = torch.where(is_wheel[..., None], slip_w, slip_s)
    k_t0 = torch.where(is_wheel, k_t0_w, k_t0_s)
    k_t1 = torch.where(is_wheel, k_t1_w, k_t1_s)

    # Bekker lateral bulldozing rut wall resistance on soft soil
    rut_lat_f = -rut_lat_k * new_rut * torch.tanh(v_tan[..., 1] / 0.05)
    tang = tang.clone()
    tang[..., 1] = tang[..., 1] + torch.where(active & (new_rut > 0.0), rut_lat_f, torch.zeros_like(rut_lat_f))

    # Update tangential stick-slip memory
    mem_new_loc = torch.where(sliding[..., None], -(tang + c_t * slip) / k_t, mem_loc + dt * slip)
    mem_new_w = torch.where(active[..., None], torch.einsum('bmk,bmkc->bmc', mem_new_loc, basis), torch.zeros_like(state.memory))

    # Wheel spin/angle update
    spin_air = state.spin * torch.clamp(1.0 - dt * viscous / inertia, min=0.0)
    next_spin = torch.where(is_wheel, torch.where(active, omega_w, spin_air), torch.zeros_like(state.spin))
    next_angle = state.angle + dt * next_spin
    next_play = torch.where(is_wheel, new_play_w, torch.zeros_like(state.ratchet_play))

    # World force & torque (including circular patch vertical pivot resistance)
    force_w = torch.einsum('bmk,bmkc->bmc', tang, basis) + normal[..., None] * normal_w
    force_w = torch.where(active[..., None], force_w, torch.zeros_like(force_w))

    # Wheel reaction torque around axle vs direct arm torque for non-wheel sites
    rt = cross(rim_w, force_w)
    offset_only = arm_w - rim_w
    torque_wheel = (
        cross(offset_only, force_w)
        + rt
        - axle_w * (rt * axle_w).sum(-1, keepdim=True)
        + torch.where(active, reaction_w, torch.zeros_like(reaction_w))[..., None] * axle_w
    )
    torque_pad = cross(arm_w, force_w)
    torque_w = torch.where(is_wheel[..., None], torque_wheel, torque_pad)

    # Circular patch vertical pivot damping tau_n = -(2/3) * mu_piv * r_patch * F_n * tanh(omega_n / 0.15)
    omega_n = (omega_owner * normal_w).sum(-1)
    pr = patch_radius_m[None, :]
    piv_coeff = (2.0 / 3.0) * pivot_mu[None, :] * pr * normal
    tau_piv_mag = -piv_coeff * torch.tanh(omega_n / 0.15)
    torque_w = torque_w + torch.where(active & (pr > 0.0), tau_piv_mag, torch.zeros_like(tau_piv_mag))[..., None] * normal_w
    k_piv = torch.where(active & (pr > 0.0), piv_coeff / (0.15 * torch.cosh((omega_n / 0.15).clamp(-10.0, 10.0)) ** 2), torch.zeros_like(normal))

    # Assemble SPD 6x6 Contact-Schur tangent matrix h*(C_tan + h*K_tan) mapped to owner rigid body CoM
    # Force-space 3x3 damping+stiffness: D_f = (dt*kn_tan + cn_tan)*nn^T + k_t0*ff^T + k_t1*ll^T
    k_n_eff = dt * kn_tan + cn_tan
    D_f = (
        k_n_eff[..., None, None] * (normal_w[..., :, None] * normal_w[..., None, :])
        + k_t0[..., None, None] * (forward[..., :, None] * forward[..., None, :])
        + k_t1[..., None, None] * (lateral[..., :, None] * lateral[..., None, :])
    ) * dt  # multiplied by dt for LHS of (M + dt*C + dt^2*K) v_new = rhs
    S_arm = skew(arm_w)
    K_6x6 = torch.zeros((*pen.shape, 6, 6), device=device, dtype=dtype)
    K_6x6[..., :3, :3] = D_f
    K_6x6[..., :3, 3:] = -D_f @ S_arm
    K_6x6[..., 3:, :3] = S_arm @ D_f
    K_6x6[..., 3:, 3:] = -(S_arm @ D_f @ S_arm) + (dt * k_piv)[..., None, None] * (
        normal_w[..., :, None] * normal_w[..., None, :]
    )

    next_state = ContactState(
        rut=new_rut,
        memory=mem_new_w,
        spin=next_spin,
        angle=next_angle,
        ratchet_play=next_play,
    )
    return force_w, torque_w, K_6x6, normal, next_state


# ---------------------------------------------------------------------------
# Py-DiSMech C2 Smooth Soft-Barrier Self-Contact (arXiv:2512.09911, Eq. 25-27)
# ---------------------------------------------------------------------------
def eval_c2_self_contact(
    pos: torch.Tensor,         # (B, N, 3) beam interior node world positions
    beam_ids: torch.Tensor,    # (N,) int64 owning beam index per node (skip same-beam adjacent pairs)
    radius_m: float = 0.002,
    delta_m: float = 0.002,
    stiffness_k: float = 2000.0,
    dt: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact Py-DiSMech piecewise C2 smooth self-collision repulsion + PSD diagonal Hessian blocks.

    For every candidate pair (i, j) from different beams with separation d = ||p_i - p_j|| < 2h + delta:
      - d <= 2h - delta : E = 0.5 * k * (2h - d)^2
      - 2h - delta < d < 2h + delta : E = 0.5 * k * [ (1/K1) log(1 + exp(K1*(2h - d))) ]^2, K1 = 15/delta
      - d >= 2h + delta : E = 0

    Returns:
        (forces (B, N, 3), diag_K_3x3 (B, N, 3, 3), total_energy (B,))
        where ``diag_K_3x3`` is already scaled by ``dt^2`` for direct injection into Block-Thomas diagonal.
    """
    B, N, _ = pos.shape
    device, dtype = pos.device, pos.dtype
    forces = torch.zeros((B, N, 3), device=device, dtype=dtype)
    diag_K = torch.zeros((B, N, 3, 3), device=device, dtype=dtype)
    energy = torch.zeros((B,), device=device, dtype=dtype)
    if N < 2:
        return forces, diag_K, energy

    two_h = 2.0 * float(radius_m)
    cutoff = two_h + float(delta_m)
    inner = max(1e-6, two_h - float(delta_m))
    k1 = 15.0 / max(float(delta_m), 1e-6)

    # Pairwise vector diff[b, i, j] = p_i - p_j; only keep cross-beam pairs (i < j)
    diff = pos[:, :, None, :] - pos[:, None, :, :]  # (B, N, N, 3)
    dist = torch.linalg.vector_norm(diff, dim=-1)   # (B, N, N)
    cross_beam = (beam_ids[:, None] != beam_ids[None, :])[None, :, :]
    triu_mask = torch.triu(torch.ones((N, N), device=device, dtype=torch.bool), diagonal=1)[None, :, :]
    active = cross_beam & triu_mask & (dist < cutoff)
    if not active.any():
        return forces, diag_K, energy

    d_safe = dist.clamp_min(1e-9)
    n_ij = diff / d_safe[..., None]  # unit vector from j -> i

    # Piecewise C2 energy E(d), dE/dd (<= 0), and d2E/dd2 (>= 0)
    gap = two_h - d_safe
    is_hard = d_safe <= inner
    # Softplus region: s = log(1 + exp(k1 * gap)) / k1, sig = sigmoid(k1 * gap)
    z = (k1 * gap).clamp(-40.0, 40.0)
    softp = torch.nn.functional.softplus(z) / k1
    sig = torch.sigmoid(z)

    e_hard = 0.5 * stiffness_k * gap * gap
    de_hard = -stiffness_k * gap
    d2e_hard = torch.full_like(gap, stiffness_k)

    e_soft = 0.5 * stiffness_k * softp * softp
    de_soft = -stiffness_k * softp * sig
    # d2E/dd2 = k * (sig^2 + k1 * softp * sig * (1 - sig)) >= 0 strictly!
    d2e_soft = stiffness_k * (sig * sig + k1 * softp * sig * (1.0 - sig))

    e_pair = torch.where(active, torch.where(is_hard, e_hard, e_soft), torch.zeros_like(gap))
    f_mag = torch.where(active, -torch.where(is_hard, de_hard, de_soft), torch.zeros_like(gap))  # >= 0 repulsive
    k_rad = torch.where(active, torch.where(is_hard, d2e_hard, d2e_soft), torch.zeros_like(gap))

    energy = e_pair.sum(dim=(1, 2))
    f_pair = f_mag[..., None] * n_ij  # repulsive force on i from j
    forces = f_pair.sum(dim=2) - f_pair.sum(dim=1)

    # PSD radial Hessian contribution dt^2 * k_rad * (n_ij n_ij^T) accumulated onto both i and j diagonals
    h_pair = (dt * dt * k_rad)[..., None, None] * (n_ij[..., :, None] * n_ij[..., None, :])
    diag_K = h_pair.sum(dim=2) + h_pair.sum(dim=1)
    return forces, diag_K, energy

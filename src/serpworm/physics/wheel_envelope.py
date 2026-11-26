"""Conservative finite-cylinder separation certificates, not a collision/contact solver.

Every positive plane gap is a lower bound on solid-cylinder separation. Failure
to find a separating plane is inconclusive and remains a failed clearance guard.
The sphere bound is retained; no wheel dimension or required clearance changes.
"""
from __future__ import annotations

import math
import torch

__all__ = ['MODEL', 'RADIUS', 'HALF_WIDTH', 'SPHERE_RADIUS', 'active_pair_indices', 'certified_clearance']

MODEL = 'finite_cylinder_projection_certificate_v1'
RADIUS = .018
HALF_WIDTH = .006
SPHERE_RADIUS = math.hypot(RADIUS, HALF_WIDTH)


def active_pair_indices(pairmask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute active pair indices (deduplicating symmetric pairs onto the upper triangle)."""
    if pairmask.shape != (20, 20) or pairmask.dtype != torch.bool:
        raise ValueError('Expected Boolean pairmask shaped [20,20]')
    effective = torch.triu(pairmask, diagonal=1) if bool(torch.equal(pairmask, pairmask.T)) and not bool(pairmask.diagonal().any()) else pairmask
    idx_i, idx_j = torch.where(effective)
    return idx_i, idx_j


def certified_clearance(
    centers: torch.Tensor,
    wheel_axes: torch.Tensor,
    pairmask: torch.Tensor,
    *,
    pair_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (certified minimum gap, original sphere minimum gap), both [B].

    centers/axes are [B,20,3]; axes are wheel-cylinder axial unit vectors. A true
    entry of the Boolean [20,20] pairmask enables that pair. Passing precomputed
    ``pair_indices`` avoids ``[20,20]`` symmetric duplication and any host sync
    inside CUDA-Graph or high-throughput RL loops.
    """
    if centers.ndim != 3 or centers.shape[1:] != (20, 3) or wheel_axes.shape != centers.shape:
        raise ValueError('Expected centers and wheel axes shaped [batch,20,3]')
    if not centers.is_floating_point() or wheel_axes.dtype != centers.dtype:
        raise TypeError('Wheel centers and axes must share a floating-point dtype')
    if pairmask.shape != (20, 20) or pairmask.dtype != torch.bool:
        raise ValueError('Expected Boolean pairmask shaped [20,20]')
    if wheel_axes.device != centers.device or pairmask.device != centers.device:
        raise ValueError('Wheel geometry and pairmask must be on the same device')

    B = centers.shape[0]
    if pair_indices is None:
        idx_i, idx_j = active_pair_indices(pairmask)
    else:
        idx_i, idx_j = pair_indices

    if idx_i.numel() == 0:
        inf = torch.full((B,), torch.inf, device=centers.device, dtype=centers.dtype)
        return inf, inf.clone()

    norm_axis = torch.linalg.vector_norm(wheel_axes, dim=-1, keepdim=True)
    finite_c = torch.isfinite(centers).all(-1)
    valid = finite_c & torch.isfinite(wheel_axes).all(-1) & ((norm_axis[..., 0] - 1.).abs() <= 1e-5)
    axes = wheel_axes / norm_axis.clamp_min(1e-12)

    # Evaluate only active pairs [B, P_active, 3] instead of dense [B, 20, 20, 3]
    ci = centers[:, idx_i, :]
    cj = centers[:, idx_j, :]
    delta = cj - ci
    a = axes[:, idx_i, :]
    b = axes[:, idx_j, :]
    up = torch.zeros_like(delta)
    up[..., 2] = 1.

    directions = torch.stack((
        a,
        b,
        torch.cross(a, b, dim=-1),
        delta,
        delta - (delta * a).sum(-1, keepdim=True) * a,
        delta - (delta * b).sum(-1, keepdim=True) * b,
        up,
    ), dim=-2)
    lengths = torch.linalg.vector_norm(directions, dim=-1)
    normals = directions / lengths[..., None].clamp_min(1e-12)
    da = (normals * a[..., None, :]).sum(-1).clamp(-1., 1.)
    db = (normals * b[..., None, :]).sum(-1).clamp(-1., 1.)
    sin_a = torch.linalg.vector_norm(torch.cross(normals, a[..., None, :].expand_as(normals), dim=-1), dim=-1)
    sin_b = torch.linalg.vector_norm(torch.cross(normals, b[..., None, :].expand_as(normals), dim=-1), dim=-1)
    support = lambda dot, sine: HALF_WIDTH * dot.abs() + RADIUS * torch.maximum((1 - dot.square()).clamp_min(0.).sqrt(), sine)
    projected = (normals * delta[..., None, :]).sum(-1).abs() - support(da, sin_a) - support(db, sin_b)
    projected = projected.masked_fill(lengths <= 1e-12, -torch.inf).amax(-1)

    sphere = torch.linalg.vector_norm(delta, dim=-1) - 2 * SPHERE_RADIUS
    scale = centers.abs().sum(-1)
    margin = 1e-10 + 32 * torch.finfo(centers.dtype).eps * (scale[:, idx_i] + scale[:, idx_j] + .05)
    certified = torch.maximum(sphere, projected) - margin

    low = torch.finfo(centers.dtype).min
    pairs_valid = valid[:, idx_i] & valid[:, idx_j]
    pairs_finite = finite_c[:, idx_i] & finite_c[:, idx_j]
    certified = torch.where(pairs_valid, certified, low).amin(-1)
    sphere = torch.where(pairs_finite, sphere, low).amin(-1)
    return certified, sphere

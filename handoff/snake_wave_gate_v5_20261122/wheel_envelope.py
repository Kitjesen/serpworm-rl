"""Conservative finite-cylinder separation certificates, not a collision/contact solver.

Every positive plane gap is a lower bound on solid-cylinder separation. Failure
to find a separating plane is inconclusive and remains a failed clearance guard.
The sphere bound is retained; no wheel dimension or required clearance changes.
"""
import math
import torch

MODEL = 'finite_cylinder_projection_certificate_v1'
RADIUS = .018
HALF_WIDTH = .006
SPHERE_RADIUS = math.hypot(RADIUS, HALF_WIDTH)


def certified_clearance(centers, wheel_axes, pairmask):
    """Return (certified minimum gap, original sphere minimum gap), both [B].

    centers/axes are [B,20,3]; axes are wheel-cylinder axial unit vectors. A true
    entry of the Boolean [20,20] pairmask enables that pair. Computation stays on
    the input device. Invalid axes/centers fail closed with a finite negative gap.
    An empty pairmask yields +inf (no enabled pair), never NaN.
    """
    if centers.ndim != 3 or centers.shape[1:] != (20, 3) or wheel_axes.shape != centers.shape:
        raise ValueError('Expected centers and wheel axes shaped [batch,20,3]')
    if not centers.is_floating_point() or wheel_axes.dtype != centers.dtype:
        raise TypeError('Wheel centers and axes must share a floating-point dtype')
    if pairmask.shape != (20, 20) or pairmask.dtype != torch.bool:
        raise ValueError('Expected Boolean pairmask shaped [20,20]')
    if wheel_axes.device != centers.device or pairmask.device != centers.device:
        raise ValueError('Wheel geometry and pairmask must be on the same device')
    norm_axis = torch.linalg.vector_norm(wheel_axes, dim=-1, keepdim=True)
    valid = torch.isfinite(centers).all(-1) & torch.isfinite(wheel_axes).all(-1)
    valid &= (norm_axis[..., 0]-1.).abs() <= 1e-5
    axes = wheel_axes/norm_axis.clamp_min(1e-12)
    delta = centers[:, None, :, :]-centers[:, :, None, :]
    a = axes[:, :, None, :].expand_as(delta)
    b = axes[:, None, :, :].expand_as(delta)
    up = torch.zeros_like(delta); up[..., 2] = 1.
    directions = torch.stack((a, b, torch.cross(a, b, dim=-1), delta,
        delta-(delta*a).sum(-1, keepdim=True)*a,
        delta-(delta*b).sum(-1, keepdim=True)*b, up), dim=-2)
    lengths = torch.linalg.vector_norm(directions, dim=-1)
    normals = directions/lengths[..., None].clamp_min(1e-12)
    da = (normals*a[..., None, :]).sum(-1).clamp(-1., 1.)
    db = (normals*b[..., None, :]).sum(-1).clamp(-1., 1.)
    # Cross products retain sin(theta) near parallel axes where 1-dot^2 loses
    # significance. Taking the larger equivalent expression stays conservative.
    sin_a = torch.linalg.vector_norm(torch.cross(normals, a[..., None, :].expand_as(normals), dim=-1), dim=-1)
    sin_b = torch.linalg.vector_norm(torch.cross(normals, b[..., None, :].expand_as(normals), dim=-1), dim=-1)
    support = lambda dot, sine: HALF_WIDTH*dot.abs()+RADIUS*torch.maximum((1-dot.square()).clamp_min(0.).sqrt(), sine)
    projected = (normals*delta[..., None, :]).sum(-1).abs()-support(da, sin_a)-support(db, sin_b)
    projected = projected.masked_fill(lengths <= 1e-12, -torch.inf).amax(-1)
    sphere = torch.linalg.vector_norm(delta, dim=-1)-2*SPHERE_RADIUS
    # Account conservatively for arithmetic roundoff as well as the requested
    # strict margin. For float64 this is about 1e-10 m at robot-scale positions.
    scale = centers.abs().sum(-1)
    margin = 1e-10+32*torch.finfo(centers.dtype).eps*(scale[:, :, None]+scale[:, None, :]+.05)
    certified = torch.maximum(sphere, projected)-margin
    pairs_valid = valid[:, :, None] & valid[:, None, :]
    low = torch.finfo(centers.dtype).min
    certified = torch.where(pairs_valid, certified, low)
    sphere = torch.where(torch.isfinite(centers).all(-1)[:, :, None] & torch.isfinite(centers).all(-1)[:, None, :], sphere, low)
    certified = certified.masked_fill(~pairmask[None], torch.inf).amin((1, 2))
    sphere = sphere.masked_fill(~pairmask[None], torch.inf).amin((1, 2))
    return certified, sphere

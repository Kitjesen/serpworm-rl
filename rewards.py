"""Velocity rewards, with frozen legacy arithmetic for controlled comparison."""
import torch


def velocity_reward(version, velocity, command, action_cost, slip, work):
    ev = velocity[:, :2]-command[:, :2]
    ew = velocity[:, 2]-command[:, 2]
    if version == 'legacy':
        tracking = (-ev.square().sum(-1)/.006**2).exp()+.5*(-(ew/.025).square()).exp()
    elif version == 'tracking_v2':
        # One combined score: no independent yaw bonus for stationary motion.
        # Stop commands remain valid; no forced oscillation or pose reward.
        scale_x = command[:, 0].abs().clamp_min(.004)
        error = (ev[:, 0]/scale_x).square()+.25*(ev[:, 1]/.004).square()+.5*(ew/.02).square()
        tracking = (1-error).clamp(0, 1)
    else:
        raise ValueError(version)
    # Keep the same physical penalties in both arms.
    return .1*(tracking-.02*action_cost-.02*slip/.01)-.01*work

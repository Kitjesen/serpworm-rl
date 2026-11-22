"""Velocity rewards; these are candidate engineering choices, not calibration.

sustain_v4: independent velocity errors use a fixed physical speed scale so
increasing the command does not flatten the reward for reachable progress.
With h(u)=hypot(u, 0.002)-0.002 and rho(e)=hypot(1,e)-1,
    gx = (h(command_vx)-h(vx-command_vx))/0.04   when command_vx != 0
         -h(vx)/0.008                          when command_vx == 0
    z = gx - 0.5*rho((vy-command_vy)/0.005)
           - 0.5*rho((wz-command_wz)/0.03) - P
    F(z) = 1+z                 for z >= -0.5
           0.5*exp(2*(z+0.5))  otherwise
    reward = policy_dt * F(z).
F is C1 and positive mathematically (very poor states can underflow to zero
numerically). Its linear productive branch avoids a convex softplus bonus for
short bursts. This is a per-time reward, with no angle-amplitude or gait bonus.
Physical failure penalties and episodic reset semantics remain in the env.

continuous_v3: sx=max(0.008 m/s, 0.2*abs(command_vx)), sy=0.005 m/s,
sw=0.03 rad/s; e=(v-command)/[sx,sy,sw], rho(e)=sqrt(1+e**2)-1.
C=rho(ex)+0.5*rho(ey)+0.5*rho(ew).
P=0.01*action_cost+0.005*acceleration_cost+0.005*slip/0.01+0.002*work.
r=0.1/(1+C+P/0.1), for the nonnegative physical costs supplied by the env.
Thus normal-step rewards are positive and at most 0.1; physical termination
penalties remain the environment's responsibility.  No amplitude bonus is used.

continuous_v2: e=(v-command)/[0.008 m/s, 0.005 m/s, 0.03 rad/s],
rho(e)=sqrt(1+e**2)-1, tracking=1-sum([1, .5, .5]*rho(e)).
All three axes count even when their commands are zero.  The tracking score
has no clipping plateau and no unconditional speed or angle-amplitude bonus.
The work argument is the existing yaw-work proxy, not full robot energy.
"""
import math
import torch


def velocity_reward(version, velocity, command, action_cost, slip, work, acceleration_cost=None, policy_dt=.2):
    """Return one reward per environment.

    velocity/command: [batch, 3], vx/vy in m/s and wz in rad/s.
    action_cost: mean squared normalized *applied target change* over this macro
        action. sustain_v4 rescales by (0.1/policy_dt)^2 to a 0.1 s rate basis.
    acceleration_cost: mean squared second difference of normalized requested
        actions; rescaled by (0.1/policy_dt)^4 to the same 0.1 s basis. None is 0.
    slip: nonnegative supported-wheel slip magnitude, m/s.
    work: nonnegative integrated yaw-work proxy over this macro action, J;
        work/policy_dt is average power in W, not full-robot electrical power.
    policy_dt: macro action duration in seconds, finite and strictly positive.

    Costs must be nonnegative. Legacy versions retain their original fixed
    0.1 s formulas for numerical comparison; policy_dt applies to sustain_v4.
    """
    if version == 'sustain_v4':
        policy_dt = float(policy_dt)
        if not math.isfinite(policy_dt) or policy_dt <= 0:
            raise ValueError('policy_dt must be finite and positive')
        h = lambda u: torch.hypot(u, torch.full_like(u, .002))-.002
        moving_gain = (h(command[:, 0])-h(velocity[:, 0]-command[:, 0]))/.04
        # Stop is an explicit task command. Its tighter physical speed scale
        # distinguishes steady residual creep from a successful stop. Both
        # modes share this rule; command-dependent returns are not evidence
        # that one mode outperforms another under different command mixtures.
        gx = torch.where(command[:, 0] == 0., -h(velocity[:, 0])/.008, moving_gain)
        yz_error = (velocity[:, 1:]-command[:, 1:])/velocity.new_tensor([.005, .03])
        yz_cost = .5*(torch.hypot(torch.ones_like(yz_error), yz_error)-1).sum(-1)
        dt_ratio = .1/policy_dt
        rate_cost = action_cost*dt_ratio**2
        acceleration = (acceleration_cost if acceleration_cost is not None else 0.)*dt_ratio**4
        penalty = .1*rate_cost+.005*acceleration+.05*slip/.01+.002*work/policy_dt
        z = gx-yz_cost-penalty
        # torch.where evaluates both branches. Bound the unused exponential to
        # prevent overflow/NaN gradients for a large positive progress score.
        tail = .5*torch.exp(2*(z+.5).clamp_max(0.))
        return policy_dt*torch.where(z >= -.5, 1+z, tail)
    if version == 'continuous_v3':
        scales = torch.stack(((.2*command[:, 0].abs()).clamp_min(.008),
                              torch.full_like(command[:, 0], .005),
                              torch.full_like(command[:, 0], .03)), -1)
        error = (velocity-command)/scales
        cost = ((torch.hypot(torch.ones_like(error), error)-1)*velocity.new_tensor([1., .5, .5])).sum(-1)
        penalty = .01*action_cost+.005*(acceleration_cost if acceleration_cost is not None else 0.)+.005*slip/.01+.002*work
        return .1/(1+cost+penalty/.1)
    if version == 'continuous_v2':
        scales = velocity.new_tensor([.008, .005, .03])
        weights = velocity.new_tensor([1., .5, .5])
        error = (velocity-command)/scales
        cost = torch.hypot(torch.ones_like(error), error)-1
        tracking = 1-(weights*cost).sum(-1)
        return .1*tracking-.01*action_cost-.005*(acceleration_cost if acceleration_cost is not None else 0.)-.005*slip/.01-.002*work
    ev = velocity[:, :2]-command[:, :2]
    ew = velocity[:, 2]-command[:, 2]
    if version == 'direct9d_v1':
        # Commanded axes are scored separately.  For a nonzero command, zero
        # velocity receives zero progress score rather than a positive baseline;
        # for a stop command, all three velocity components must be near zero.
        scales = velocity.new_tensor([.004, .004, .02])
        error = (velocity-command).square()/scales.square()
        active = command.abs() > velocity.new_tensor([5e-4, 5e-4, 2e-3])
        axis_score = (1-error).clamp(-1., 1.)
        active_count = active.sum(-1)
        moving_score = (axis_score*active).sum(-1)/active_count.clamp_min(1)
        stop_error = error.mean(-1)
        stop_score = (1-stop_error).clamp(-1., 1.)
        tracking = torch.where(active_count > 0, moving_score, stop_score)
        # Keep the first diagnostic version deliberately light on penalties so
        # the policy can discover a sustained motion cycle.
        return .1*tracking-.01*action_cost-.005*(acceleration_cost if acceleration_cost is not None else 0.)-.005*slip/.01-.002*work
    if version == 'legacy':
        tracking = (-ev.square().sum(-1)/.006**2).exp()+.5*(-(ew/.025).square()).exp()
    elif version == 'tracking_v2':
        # One combined score: no independent yaw bonus for stationary motion.
        # Stop commands remain valid; no forced oscillation or pose reward.
        scale_x = command[:, 0].abs().clamp_min(.004)
        error = (ev[:, 0]/scale_x).square()+.25*(ev[:, 1]/.004).square()+.5*(ew/.02).square()
        tracking = (1-error).clamp(0, 1)
    elif version == 'tracking_v3':
        if acceleration_cost is None:
            raise ValueError('tracking_v3 needs the requested-action second difference')
        scale_x = command[:, 0].abs().clamp_min(.004)
        error = (ev[:, 0]/scale_x).square()+.5*(ev[:, 1]/.004).square()+(ew/.02).square()
        tracking = 1/(1+error)
        # Requested actions, before the servo filter: filtering must not hide chatter.
        return .1*(tracking-.05*action_cost-.025*acceleration_cost-.02*slip/.01)-.01*work
    else:
        raise ValueError(version)
    # Keep the same physical penalties in both arms.
    return .1*(tracking-.02*action_cost-.02*slip/.01)-.01*work

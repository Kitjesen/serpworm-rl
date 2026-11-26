"""All online environment arithmetic stays on the physics device.

Python dispatch, initialization, file IO and boundary diagnostics remain on CPU.
This is a tensor API, deliberately avoiding Gym/SB3's NumPy step interface.
"""
import math
import torch
from gpu_core import GPUCore, matrix
from matsuoka_torch import Matsuoka

LIMIT = math.radians(20)
STATE = ('x', 'v', 'servo', 'target', 'yaw', 'spin', 'angle', 'memory', 'tension', 'support')


class TensorEnv:
    def __init__(self, kind='direct', n=2, seed=7, device='cuda', dt=.00025, noise=True):
        if kind not in ('direct', 'liu', 'liu_foc'):
            raise ValueError(kind)
        if device != 'cuda':
            raise ValueError('Physics training requires CUDA; CPU is only a numerical oracle.')
        self.kind, self.n, self.noise = kind, n, noise
        self.c = GPUCore(5, n, device=device, dt=dt, yaw_motor_speed=2., strain_jacobian=True)
        self.c.capture(4)
        for _ in range(25):
            self.c.advance_steps(round(.02/dt))
        # One startup diagnostic, not per-step host traffic.
        if bool(self.c.failed) or not bool(torch.isfinite(self.c.x).all()):
            raise RuntimeError('Passive settling failed')
        self.initial = {k: getattr(self.c, k).clone() for k in STATE}
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.device, self.dtype = self.c.device, self.c.dtype
        self.cpg = Matsuoka(n, device, self.dtype, c=.75 if kind == 'liu_foc' else 0.)
        self.targets = self.zeros(n, 4)
        self.previous_actions = self.zeros(n, 4)
        self.cmd = self.zeros(n, 3)
        self.goals = self.zeros(n, 2)
        self.count = torch.zeros(n, device=device, dtype=torch.long)
        self.filtered_velocity = self.zeros(n, 3)
        self.visual_previous = self.zeros(n, 10, 3)
        self.visual_filtered = self.zeros(n, 2)
        self.prev_theta = self.zeros(n)
        self.delayed = self.zeros(n, 17)
        self.previous_angles = self.zeros(n, 14)
        self.dq = self.zeros(n, 14)
        self.sensor = self.zeros(n, 31)
        self.history = self.zeros(n, 10, 38)
        self.obs_size = 380 if kind == 'direct' else 14
        self.angle_scale = self.tensor([math.pi]*10+[LIMIT]*4)
        self.noise_scale = self.tensor([.002]*14+[.01]*3)
        self.command_scale = self.tensor([.01, .01, .03])
        self.frame_flip = self.tensor([-1., -1., 1.])
        self.no_reset = torch.zeros(n, device=device, dtype=torch.bool)
        owners = torch.arange(10, device=device).repeat_interleave(2)
        self.wheel_pair = owners[:, None] != owners[None, :]
        self.obs = self.reset(torch.ones_like(self.no_reset))

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def zeros(self, *shape):
        return torch.zeros(*shape, device=self.device, dtype=self.dtype)

    def random(self, *shape):
        return torch.rand(*shape, generator=self.rng, device=self.device, dtype=self.dtype)

    def normal(self, shape):
        return torch.randn(shape, generator=self.rng, device=self.device, dtype=self.dtype)

    def geometry(self):
        x, v = self.c.x[:, :10], self.c.v[:, :10]
        R = matrix(x[..., 3:])
        relative = R[:, 1:9:2].transpose(-1, -2) @ R[:, 2:10:2]
        q = torch.atan2(relative[..., 1, 0], relative[..., 0, 0])
        forward = -R[:, 4, :2, 0]
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
        return x, v, R, q, forward, left

    def sensor_update(self, reset):
        x, v, R, q, f, l = self.geometry()
        angles = torch.cat((self.c.servo, q), -1)
        gyro = torch.einsum('bij,bj->bi', R[:, 4].transpose(-1, -2), v[:, 4, 3:])*self.frame_flip
        raw = torch.cat((angles, gyro), -1)
        if self.noise:
            raw = raw+self.normal(raw.shape)*self.noise_scale
        packet = torch.where(reset[:, None], raw, self.delayed)
        self.delayed.copy_(raw)
        previous = torch.where(reset[:, None], raw[:, :14], self.previous_angles)
        old_dq = torch.where(reset[:, None], 0., self.dq)
        self.dq.copy_(old_dq+.25*((packet[:, :14]-previous)/.02-old_dq))
        self.previous_angles.copy_(packet[:, :14])
        self.sensor.copy_(torch.cat((packet[:, :14]/self.angle_scale, self.dq/2, packet[:, 14:]), -1))
        return x, v, R, q, f, l

    def packet(self, reset):
        if self.kind == 'direct':
            frame = torch.cat((self.sensor, self.targets/LIMIT, self.cmd/self.command_scale), -1)
            shifted = torch.cat((self.history[:, 1:], frame[:, None]), 1)
            self.history.copy_(torch.where(reset[:, None, None], frame[:, None], shifted))
            return self.history.flatten(1).float().clone()
        pos = self.c.x[:, :10, :3].clone()
        if self.noise:
            pos += self.normal(pos.shape)*.0002
        measured = (pos[:, 9, :2]-self.visual_previous[:, 9, :2])/.1
        self.visual_filtered.add_(.15*(measured-self.visual_filtered))
        self.visual_previous.copy_(pos)
        delta = self.goals-pos[:, 9, :2]
        distance = delta.norm(dim=-1)
        direction = delta/distance[:, None].clamp_min(1e-9)
        vel = self.visual_filtered
        vg = (vel*direction).sum(-1)
        theta = torch.atan2(direction[:, 0]*vel[:, 1]-direction[:, 1]*vel[:, 0], vg)
        theta = torch.where(vel.norm(dim=-1) > .0005, theta, self.prev_theta)
        difference = theta-self.prev_theta
        dtheta = torch.atan2(difference.sin(), difference.cos())/.1
        axis = pos[:, 1::2, :2]-pos[:, 0::2, :2]
        heading = torch.atan2(axis[..., 1], axis[..., 0]).flip(-1)
        centers = ((pos[:, 1::2, :2]+pos[:, 0::2, :2])/2).flip(1)
        dh = heading[:, 1:]-heading[:, :-1]
        curvature = torch.atan2(dh.sin(), dh.cos())/(centers[:, 1:]-centers[:, :-1]).norm(dim=-1).clamp_min(.05)
        vg, theta, dtheta = [torch.where(reset, 0., a) for a in (vg, theta, dtheta)]
        self.visual_filtered.copy_(torch.where(reset[:, None], 0., self.visual_filtered))
        self.prev_theta.copy_(theta)
        return torch.cat((distance[:, None]/.1, vg[:, None]/.01, theta[:, None]/math.pi,
                          dtheta[:, None].clamp(-10, 10)/10, curvature/5, self.previous_actions,
                          torch.ones_like(distance[:, None]), torch.zeros_like(distance[:, None])), -1).float()

    def reset(self, mask):
        """Masked reset; no nonzero()/item() or per-environment Python branching."""
        for name in STATE:
            value = getattr(self.c, name)
            value.copy_(torch.where(mask.reshape(self.n, *([1]*(value.ndim-1))), self.initial[name], value))
        self.cpg.reset(mask)
        for a in (self.targets, self.previous_actions, self.filtered_velocity):
            a.copy_(torch.where(mask[:, None], 0., a))
        self.count.copy_(torch.where(mask, 0, self.count))
        vx = torch.floor(self.random(self.n)*3)*.004
        wz = (torch.floor(self.random(self.n)*3)-1)*.02
        command = torch.stack((vx, torch.zeros_like(vx), torch.where(vx == 0, 0., wz)), -1)
        self.cmd.copy_(torch.where(mask[:, None], command, self.cmd))
        # Preserve ongoing sensor state: reset must not consume an extra sample.
        saved_sensor = [a.clone() for a in (self.delayed, self.previous_angles, self.dq, self.sensor)]
        x, v, R, q, f, l = self.sensor_update(mask)
        for a, saved in zip((self.delayed, self.previous_angles, self.dq, self.sensor), saved_sensor):
            a.copy_(torch.where(mask[:, None], a, saved))
        angle = torch.where(self.random(self.n) < .5, -1., 1.)*(.35+.2*self.random(self.n))
        goal = x[:, 9, :2]+.08*(angle.cos()[:, None]*f+angle.sin()[:, None]*l)
        self.goals.copy_(torch.where(mask[:, None], goal, self.goals))
        self.visual_previous.copy_(torch.where(mask[:, None, None], x[:, :, :3], self.visual_previous))
        self.prev_theta.copy_(torch.where(mask, 0., self.prev_theta))
        self.visual_filtered.copy_(torch.where(mask[:, None], 0., self.visual_filtered))
        # packet advances camera/history; restore untouched environments after it.
        saved = [a.clone() for a in (self.history, self.visual_previous, self.visual_filtered, self.prev_theta)]
        obs = self.packet(mask)
        for a, old in zip((self.history, self.visual_previous, self.visual_filtered, self.prev_theta), saved):
            a.copy_(torch.where(mask.reshape(self.n, *([1]*(a.ndim-1))), a, old))
        return obs

    @torch.no_grad()
    def step(self, action):
        action = action.to(dtype=self.dtype).clamp(-1, 1)
        c = self.c
        x0 = c.x[:, :10, :3].clone()
        old = self.targets.clone()
        work = self.zeros(self.n)
        raw_peak = self.zeros(self.n)
        for _ in range(5):
            if self.kind == 'direct':
                desired = LIMIT*action
            else:
                signal = self.cpg.step(3*action)
                raw_peak = torch.maximum(raw_peak, signal.abs().amax(-1))
                desired = math.radians(15)*signal.flip(-1).clamp(-1, 1)
            self.targets.add_((desired-self.targets).clamp(-.02, .02))
            c.target.zero_()
            c.yaw.copy_(self.targets)
            c.advance_steps(round(.02/c.dt))
            self.count.add_(1)
            x, v, R, q, f, l = self.sensor_update(self.no_reset)
            rate = ((v[:, 2:10:2, 3:]-v[:, 1:9:2, 3:])*R[:, 1:9:2, :, 2]).sum(-1)
            torque = (2*(self.targets-q)-.03*rate).clamp(-.5, .5)
            cap = .5*torch.where(torque*rate > 0, (1-rate.abs()/2).clamp_min(0), 1.)
            torque = torch.maximum(torch.minimum(torque, cap), -cap)
            work += (torque*rate).abs().sum(-1)*.02
        velocity = (x[:, 4, :2]-x0[:, 4, :2])/.1
        measured = torch.stack(((velocity*f).sum(-1), (velocity*l).sum(-1), v[:, 4, 5]), -1)
        self.filtered_velocity.add_((1-math.exp(-.1))*(measured-self.filtered_velocity))
        active = c.support > 1e-6
        slip = (c.last_slip.norm(dim=-1)*active).sum((1, 2))/active.sum((1, 2)).clamp_min(1)
        action_cost = ((self.targets-old)/LIMIT).square().mean(-1)
        distance = (self.goals-x[:, 9, :2]).norm(dim=-1)
        if self.kind == 'direct':
            ev = self.filtered_velocity[:, :2]-self.cmd[:, :2]
            ew = self.filtered_velocity[:, 2]-self.cmd[:, 2]
            reward = .1*((-ev.square().sum(-1)/.006**2).exp()+.5*(-(ew/.025).square()).exp()-.02*action_cost-.02*slip/.01)-.01*work
            success = torch.zeros_like(active[:, 0, 0])
        else:
            distance0 = (self.goals-x0[:, 9, :2]).norm(dim=-1)
            success = distance < .01
            reward = (distance0-distance)/.08-.002*action_cost-.002*slip/.01-.01*work+2*success
        fail = (q.abs().amax(-1) > math.radians(26)) | (R[:, :, 2, 2].amin(-1) < .7) | (x[:, :, 2].amin(-1) < .075)
        centers = x[:, :, None, :3]+torch.einsum('bpij,psj->bpsi', R, c.mounts)
        centers = centers.reshape(self.n, 20, 3)
        distances = (centers[:, :, None]-centers[:, None, :]).norm(dim=-1)
        clearance = distances.masked_fill(~self.wheel_pair, torch.inf).amin((1, 2))-2*math.hypot(.018, .006)
        fail |= clearance < .002
        reward -= 2*fail
        self.previous_actions.copy_(action)
        terminal_obs = self.packet(self.no_reset)
        terminated = fail | success
        truncated = (self.count >= 1000) & ~terminated
        done = terminated | truncated
        metrics = torch.stack((slip, work, clearance, q.abs().amax(-1), distance, fail.to(self.dtype), success.to(self.dtype), raw_peak), -1)
        reset_obs = self.reset(done)
        self.obs = torch.where(done[:, None], reset_obs, terminal_obs)
        return self.obs, reward.float(), terminated, truncated, terminal_obs, metrics

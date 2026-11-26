"""All online environment arithmetic stays on the physics device.

Python dispatch, initialization, file IO and boundary diagnostics remain on CPU.
This is a tensor API, deliberately avoiding Gym/SB3's NumPy step interface.
"""
import math
import torch
from serpworm.envs.rewards import velocity_reward
from serpworm.physics.gpu_core import GPUCore, matrix
from serpworm.physics.wheel_envelope import certified_clearance

SERVO_LIMIT = math.radians(180)
STATE = ('x', 'v', 'servo', 'target', 'yaw', 'spin', 'angle', 'memory', 'tension', 'support')
ACTION_SIZES = {'snake': 4, 'worm': 5, 'mixed': 9}
PRIVILEGED_SIZE = 64


class TensorEnv:
    def __init__(self, kind='direct', n=2, seed=7, device='cuda', dt=.00025, noise=True, reward_version='sustain_v4', plate_solver='legacy', control_profile='sustain_v4', command_switching=True, servo_neutral_deg=90., fixed_vx_mm_s=None, mode='mixed', vx_min_mm_s=5., vx_max_mm_s=200., stop_fraction=.1, episode_seconds=20., strain_jacobian=True):
        if kind != 'direct':
            raise ValueError(kind)
        if device not in ('cuda', 'cpu'):
            raise ValueError('Use cuda for training or cpu for the numerical oracle')
        if not math.isfinite(dt) or dt <= 0 or not math.isclose(round(.02/dt)*dt, .02, abs_tol=1e-12):
            raise ValueError('Physics dt must exactly divide the 0.02 s sensor interval')
        if device == 'cuda' and round(.02/dt) % 4:
            raise ValueError('CUDA sensor interval must contain whole four-step graph replays')
        if not math.isfinite(episode_seconds) or episode_seconds <= 0 or not math.isclose(episode_seconds/.02, round(episode_seconds/.02)):
            raise ValueError('Episode duration must be a positive multiple of 0.02 s')
        self.episode_seconds = float(episode_seconds)
        self.episode_sensor_steps = round(episode_seconds/.02)
        self.kind, self.n, self.noise = kind, n, noise
        if control_profile != 'sustain_v4':
            raise ValueError('sustain_v4 is the only profile in this experiment')
        if reward_version != 'sustain_v4':
            raise ValueError(reward_version)
        if mode not in ACTION_SIZES:
            raise ValueError(mode)
        if not 0 <= vx_min_mm_s < vx_max_mm_s or not 0 <= stop_fraction < 1:
            raise ValueError('Invalid continuous command range or stop fraction')
        self.mode, self.action_size = mode, ACTION_SIZES[mode]
        self.strain_jacobian = bool(strain_jacobian)
        self.vx_range = (vx_min_mm_s*1e-3, vx_max_mm_s*1e-3)
        self.stop_fraction = stop_fraction
        self.reward_version = reward_version
        self.control_profile = control_profile
        # One probability/transition per held action. Sensors and motors still run
        # at 50 Hz; this is temporal exploration, not an external noise process.
        self.sensor_dt = .02
        self.steps_per_action = 10
        self.policy_dt = self.sensor_dt*self.steps_per_action
        self.limit = math.radians(45)
        self.target_gain = 1-math.exp(-.02/.2)
        self.servo_limit = SERVO_LIMIT
        self.servo_gain = 1-math.exp(-.02/.2)
        self.command_switch_steps = 250 if command_switching else 0
        if not 0.0 <= float(servo_neutral_deg) <= 180.0:
            raise ValueError('servo_neutral_deg must be within [0,180]')
        self.servo_neutral_deg = float(servo_neutral_deg)
        self.fixed_vx_m_s = None if fixed_vx_mm_s is None else float(fixed_vx_mm_s) * 1e-3
        # Settle one prototype instance first, then broadcast to all n parallel
        # environments so startup stays ~1 s even at n=2048 with identical physics.
        settler = GPUCore(5, 1, device=device, dt=dt, yaw_motor_speed=2., strain_jacobian=self.strain_jacobian, plate_solver=plate_solver)
        if device == 'cuda':
            settler.capture(4)
        for _ in range(25):
            settler.advance_steps(round(.02/dt))
        settler.target.fill_(math.radians(self.servo_neutral_deg))
        for _ in range(150):
            settler.advance_steps(round(.02/dt))
        if bool(settler.failed) or not bool(torch.isfinite(settler.x).all()):
            raise RuntimeError('Neutral settling failed')
        settled_single = {k: getattr(settler, k).clone() for k in STATE}
        del settler
        self.c = GPUCore(5, n, device=device, dt=dt, yaw_motor_speed=2., strain_jacobian=self.strain_jacobian, plate_solver=plate_solver)
        for k, value in settled_single.items():
            getattr(self.c, k).copy_(value.expand(n, *value.shape[1:]))
        if device == 'cuda':
            self.c.capture(4)
        self.initial = {k: getattr(self.c, k).clone() for k in STATE}
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.device, self.dtype = self.c.device, self.c.dtype
        self.targets = self.zeros(n, 4)
        self.servo_neutral = self.tensor([math.radians(self.servo_neutral_deg)]*10).expand(n, -1).clone()
        self.servo_targets = self.servo_neutral.clone()
        self.previous_actions = self.zeros(n, 9)
        self.previous_action_delta = self.zeros(n, 9)
        self.cmd = self.zeros(n, 3)
        self.goals = self.zeros(n, 2)
        self.count = torch.zeros(n, device=device, dtype=torch.long)
        self.filtered_velocity = self.zeros(n, 3)
        self.last_body_velocity = self.zeros(n, 3)
        self.visual_previous = self.zeros(n, 10, 3)
        self.visual_filtered = self.zeros(n, 2)
        self.prev_theta = self.zeros(n)
        self.delayed = self.zeros(n, 17)
        self.previous_angles = self.zeros(n, 14)
        self.dq = self.zeros(n, 14)
        self.sensor = self.zeros(n, 31)
        # 31 sensor values + 10 applied worm targets + 4 yaw targets
        # + 9 previous actions + 3 command values.
        self.frame_size = 57
        self.history = self.zeros(n, 10, self.frame_size)  # 1.8 s span at 5 Hz
        self.obs_size = 10*self.frame_size if kind == 'direct' else 14
        self.privileged_size = PRIVILEGED_SIZE
        self.critic_obs_size = self.obs_size + self.privileged_size
        self.angle_scale = self.tensor([math.pi]*10+[self.limit]*4)
        self.noise_scale = self.tensor([.002]*14+[.01]*3)
        self.command_scale = self.tensor([max(.04, self.vx_range[1]), .01, .20])
        self.frame_flip = self.tensor([-1., -1., 1.])
        self.no_reset = torch.zeros(n, device=device, dtype=torch.bool)
        owners = torch.arange(10, device=device).repeat_interleave(2)
        self.wheel_pair = owners[:, None] != owners[None, :]
        self.obs = self.reset(torch.ones_like(self.no_reset))
        self.critic_obs = self.build_critic_obs(self.obs)

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def zeros(self, *shape):
        return torch.zeros(*shape, device=self.device, dtype=self.dtype)

    def random(self, *shape):
        return torch.rand(*shape, generator=self.rng, device=self.device, dtype=self.dtype)

    def sample_command(self):
        if self.fixed_vx_m_s is not None:
            vx = self.tensor([self.fixed_vx_m_s]).expand(self.n)
            return torch.stack((vx, torch.zeros_like(vx), torch.zeros_like(vx)), -1)
        low, high = self.vx_range
        vx = low+(high-low)*self.random(self.n)
        stop = self.random(self.n) < self.stop_fraction
        return torch.stack((torch.where(stop, 0., vx), torch.zeros_like(vx),
                            torch.zeros_like(vx)), -1)

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
            pieces = [self.sensor]
            pieces.append(self.servo_targets/self.servo_limit)
            pieces.append(self.targets/self.limit)
            pieces.append(self.previous_actions)
            frame = torch.cat((*pieces, self.cmd/self.command_scale), -1)
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
        neutral = self.servo_neutral
        self.servo_targets.copy_(torch.where(mask[:, None], neutral, self.servo_targets))
        self.c.target.copy_(torch.where(mask[:, None], neutral, self.c.target))
        for a in (self.targets, self.previous_actions, self.previous_action_delta, self.filtered_velocity, self.last_body_velocity):
            a.copy_(torch.where(mask[:, None], 0., a))
        self.count.copy_(torch.where(mask, 0, self.count))
        command = self.sample_command()
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

    def privileged_state(self):
        """Ground-truth physical state for asymmetric Critic training only (never fed to Actor)."""
        x = self.c.x[:, :10]
        R = matrix(x[..., 3:])
        gap = ((x[:, 1::2, :3]-x[:, 0::2, :3])*(-R[:, 0::2, :, 0])).sum(-1)
        vel_err = (self.filtered_velocity-self.cmd)/self.command_scale
        return torch.cat((
            self.last_body_velocity/self.command_scale,
            self.filtered_velocity/self.command_scale,
            vel_err,
            (x[:, :, 2]-.133)/.05,
            R[:, :, 2, 2],
            (gap-.1175)/.04,
            self.c.tension.reshape(self.n, 20)/10.,
            self.c.support.mean(-1)/5.,
        ), -1).float()

    def build_critic_obs(self, obs):
        return torch.cat((obs, self.privileged_state()), -1)

    def map_action(self, action):
        """Map policy actions to five signed contractions and four yaw joints.

        - snake (4D): four yaw joints, five contractions held at neutral.
        - worm  (5D): five paired segment contractions, four yaw joints held at 0.
        - mixed (9D): five paired contractions + four independent yaw joints.
        """
        if action.shape[-1] != self.action_size:
            raise ValueError(f'{self.mode} expects {self.action_size} effective policy actions')
        if self.mode == 'snake':
            action = torch.cat((torch.zeros_like(action[..., :1]).expand(*action.shape[:-1], 5), action), -1)
        elif self.mode == 'worm':
            action = torch.cat((action, torch.zeros_like(action[..., :1]).expand(*action.shape[:-1], 4)), -1)
        contraction = action[..., :5]
        yaw = action[..., 5:9]
        c = contraction.repeat_interleave(2, -1)
        positive_span = self.servo_limit - self.servo_neutral
        negative_span = self.servo_neutral
        servo_desired = self.servo_neutral + c.clamp_min(0.) * positive_span + c.clamp_max(0.) * negative_span
        servo_desired = servo_desired.clamp(0., self.servo_limit)
        yaw_desired = yaw.clamp(-1., 1.) * self.limit
        physical = torch.cat((contraction, yaw), -1)
        return servo_desired, yaw_desired, physical

    @staticmethod
    def body_velocity(before, after, dt=.1):
        """Plate-centroid translation and head-to-tail heading rate, not one plate's rocking."""
        axis0 = before[:, 9, :2]-before[:, 0, :2]
        axis1 = after[:, 9, :2]-after[:, 0, :2]
        axis0 = axis0/axis0.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        axis1 = axis1/axis1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        forward = axis0+axis1
        forward = forward/forward.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
        velocity = (after[:, :, :2].mean(1)-before[:, :, :2].mean(1))/dt
        turn = torch.atan2(axis0[:, 0]*axis1[:, 1]-axis0[:, 1]*axis1[:, 0], (axis0*axis1).sum(-1))/dt
        return torch.stack(((velocity*forward).sum(-1), (velocity*left).sum(-1), turn), -1)

    @torch.no_grad()
    def step(self, action):
        action = action.to(dtype=self.dtype).clamp(-1, 1)
        c = self.c
        x0 = c.x[:, :10, :3].clone()
        old_yaw = self.targets.clone()
        old_servo = self.servo_targets.clone()
        servo_desired, yaw_desired, physical_action = self.map_action(action)
        action_delta = physical_action-self.previous_actions
        second_difference = action_delta-self.previous_action_delta
        if self.mode == 'snake':
            active_second = second_difference[:, 5:]
        elif self.mode == 'worm':
            active_second = second_difference[:, :5]
        else:
            active_second = second_difference
        acceleration_cost = active_second.square().mean(-1)
        work = self.zeros(self.n)
        raw_peak = self.zeros(self.n)
        diagnostics = []
        for _ in range(self.steps_per_action):
            desired = yaw_desired
            self.servo_targets.add_((self.servo_gain*(servo_desired-self.servo_targets)).clamp(-.02, .02))
            c.target.copy_(self.servo_targets)
            self.targets.add_((self.target_gain*(desired-self.targets)).clamp(-.02, .02))
            c.yaw.copy_(self.targets)
            c.advance_steps(round(.02/c.dt))
            self.count.add_(1)
            x, v, R, q, f, l = self.sensor_update(self.no_reset)
            rate = ((v[:, 2:10:2, 3:]-v[:, 1:9:2, 3:])*R[:, 1:9:2, :, 2]).sum(-1)
            torque = (2*(self.targets-q)-.03*rate).clamp(-.5, .5)
            cap = .5*torch.where(torque*rate > 0, (1-rate.abs()/2).clamp_min(0), 1.)
            torque = torch.maximum(torch.minimum(torque, cap), -cap)
            work += (torque*rate).abs().sum(-1)*.02
            # Torque is sampled from the final actual physics substep; not its interval peak.
            diagnostics.append(torch.stack((desired, self.targets, q, rate,
                c.last_yaw_torque, c.last_yaw_torque_cap), -2).clone())
        velocity = (x[:, 4, :2]-x0[:, 4, :2])/self.policy_dt
        measured = torch.stack(((velocity*f).sum(-1), (velocity*l).sum(-1), v[:, 4, 5]), -1)
        self.last_point_velocity = measured.clone()
        self.last_body_velocity = self.body_velocity(x0, x, self.policy_dt)
        measured = self.last_body_velocity
        tau = .2
        self.filtered_velocity.add_((1-math.exp(-self.policy_dt/tau))*(measured-self.filtered_velocity))
        active = c.support > 1e-6
        slip = (c.last_slip.norm(dim=-1)*active).sum((1, 2))/active.sum((1, 2)).clamp_min(1)
        physical_delta = torch.cat(((self.servo_targets-old_servo)/self.servo_limit,
                                    (self.targets-old_yaw)/self.limit), -1)
        if self.mode == 'snake':
            active_delta = physical_delta[:, 10:]
        elif self.mode == 'worm':
            active_delta = physical_delta[:, :10]
        else:
            active_delta = physical_delta
        action_cost = active_delta.square().mean(-1)
        distance = (self.goals-x[:, 9, :2]).norm(dim=-1)
        if self.kind == 'direct':
            reward = velocity_reward(self.reward_version, self.filtered_velocity, self.cmd,
                                     action_cost, slip, work, acceleration_cost, policy_dt=self.policy_dt)
            success = torch.zeros_like(active[:, 0, 0])
        else:
            distance0 = (self.goals-x0[:, 9, :2]).norm(dim=-1)
            success = distance < .01
            reward = (distance0-distance)/.08-.002*action_cost-.002*slip/.01-.01*work+2*success
        joint_fail_limit = self.limit+math.radians(1)
        guard_reasons = dict(joint_limit=q.abs().amax(-1) > joint_fail_limit,
            tilt=R[:, :, 2, 2].amin(-1) < .7, height=x[:, :, 2].amin(-1) < .075)
        centers = x[:, :, None, :3]+torch.einsum('bpij,psj->bpsi', R, c.mounts)
        centers = centers.reshape(self.n, 20, 3)
        wheel_axes = R[..., :, 1].repeat_interleave(2, dim=1)
        clearance, sphere_clearance = certified_clearance(centers, wheel_axes, self.wheel_pair)
        guard_reasons['wheel_clearance'] = clearance < .002
        fail = torch.stack(tuple(guard_reasons.values())).any(0)
        reward -= 2*fail
        self.previous_actions.copy_(physical_action)
        self.previous_action_delta.copy_(action_delta)
        terminated = fail | success
        truncated = (self.count >= self.episode_sensor_steps) & ~terminated
        done = terminated | truncated
        self.last_command = self.cmd.clone()
        self.last_guard_reasons = {name: value.clone() for name, value in guard_reasons.items()}
        self.last_sphere_clearance = sphere_clearance.clone()
        gap = ((x[:, 1::2, :3]-x[:, 0::2, :3])*(-R[:, 0::2, :, 0])).sum(-1)
        self.last_physics_diagnostics = dict(tension=c.tension.clone(), support=c.support.clone(),
            spin=c.spin.clone(), slip=c.last_slip.clone(), gap=gap.clone(),
            stop_force=50000*(.075-gap).clamp_min(0))
        if self.command_switch_steps:
            change = (self.count.remainder(self.command_switch_steps) == 0) & ~done
            command = self.sample_command()
            self.cmd.copy_(torch.where(change[:, None], command, self.cmd))
        terminal_obs = self.packet(self.no_reset)
        terminal_critic_obs = self.build_critic_obs(terminal_obs)
        metrics = torch.stack((slip, work, clearance, q.abs().amax(-1), distance, fail.to(self.dtype), success.to(self.dtype), raw_peak), -1)
        self.last_velocity = self.filtered_velocity.clone()
        self.last_plate_positions = c.x[:, :10, :3].clone()
        self.last_diagnostics = torch.stack(diagnostics)
        self.last_servo_diagnostics = torch.stack((servo_desired, self.servo_targets, c.servo), -2).clone()
        self.last_action_rate_cost = action_cost.clone()
        self.last_action_acceleration_cost = acceleration_cost.clone()
        self.last_terminal_critic_obs = terminal_critic_obs
        reset_obs = self.reset(done)
        reset_critic_obs = self.build_critic_obs(reset_obs)
        self.obs = torch.where(done[:, None], reset_obs, terminal_obs)
        self.critic_obs = torch.where(done[:, None], reset_critic_obs, terminal_critic_obs)
        return self.obs, reward.float(), terminated, truncated, terminal_obs, metrics

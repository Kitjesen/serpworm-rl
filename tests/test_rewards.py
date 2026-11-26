"""Numerical reward contracts. Run on CPU and, when present, CUDA.

Example: python test_rewards.py --device auto --out reward_contract.json
These checks validate objective arithmetic, not learned motion or physics.
"""
import argparse
import json
import math
from pathlib import Path
import unittest

import torch

from serpworm.envs.rewards import UnifiedRewardConfig, locomotion_reward, velocity_reward


class RewardContracts(unittest.TestCase):
    device = 'cpu'

    def tensor(self, values, requires_grad=False):
        return torch.tensor(values, dtype=torch.float64, device=self.device,
                            requires_grad=requires_grad)

    def reward(self, vx, cx, *, vy=0., wz=0., cy=0., cw=0., action=0.,
               acceleration=0., slip=0., work=0., dt=.2, version='sustain_v4'):
        velocity = self.tensor([[vx, vy, wz]])
        command = self.tensor([[cx, cy, cw]])
        args = [self.tensor([value]) for value in (action, slip, work, acceleration)]
        return velocity_reward(version, velocity, command, *args, policy_dt=dt)[0]

    def test_target_is_unique_peak_and_progress_is_ordered(self):
        for command in (.005, .02, .05, .1, .2):
            velocities = [-command, 0., command*.25, command*.5, command*.75, command]
            scores = [float(self.reward(v, command)) for v in velocities]
            self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])), scores)
            target = scores[-1]
            self.assertLess(float(self.reward(command+.001, command)), target)
            self.assertLess(float(self.reward(command-.001, command)), target)
            self.assertLess(float(self.reward(2*command, command)), target)
            self.assertAlmostEqual(float(self.reward(2*command, command)), scores[1], places=12)

    def test_forward_gradient_remains_visible_at_high_command(self):
        command = self.tensor([[.2, 0., 0.]])
        zero = self.tensor([0.])
        slopes = {}
        for version in ('continuous_v3', 'sustain_v4'):
            velocity = self.tensor([[0., 0., 0.]], requires_grad=True)
            score = velocity_reward(version, velocity, command, zero, zero, zero,
                                    zero, policy_dt=.1)
            slopes[version] = float(torch.autograd.grad(score.sum(), velocity)[0][0, 0])
        self.assertGreater(slopes['sustain_v4'], 25*slopes['continuous_v3'])
        self.assertAlmostEqual(slopes['sustain_v4'], 2.499875009374219, places=8)

    def test_analytic_velocity_gradient_matches_finite_difference(self):
        for vector in ([.01, .001, -.005], [-.08, .002, .01], [.199, -.003, .02]):
            velocity = self.tensor([vector], requires_grad=True)
            command, zero = self.tensor([[.2, 0., 0.]]), self.tensor([0.])
            fun = lambda v: velocity_reward('sustain_v4', v, command, zero, zero, zero, zero)
            grad = torch.autograd.grad(fun(velocity).sum(), velocity)[0]
            for axis in range(3):
                plus, minus = velocity.detach().clone(), velocity.detach().clone()
                plus[0, axis] += 1e-7
                minus[0, axis] -= 1e-7
                numerical = (fun(plus)-fun(minus))/(2e-7)
                torch.testing.assert_close(grad[0, axis], numerical[0], rtol=2e-5, atol=1e-7)

    def test_c1_positive_extension_at_join(self):
        # Zero-command/zero-velocity gives z=-0.1*action at dt=.1.
        command, zero = self.tensor([[0., 0., 0.]]), self.tensor([0.])
        values, slopes = [], []
        for action in (5.-1e-7, 5., 5.+1e-7):
            cost = self.tensor([action], requires_grad=True)
            score = velocity_reward('sustain_v4', command, command, cost, zero,
                                    zero, zero, policy_dt=.1)
            values.append(float(score.detach()))
            slopes.append(float(torch.autograd.grad(score.sum(), cost)[0]))
        for value in values:
            self.assertAlmostEqual(value, .05, places=8)
        for slope in slopes:
            self.assertAlmostEqual(slope, -.01, places=8)

    def test_stop_and_uncommanded_axes(self):
        stop = float(self.reward(0., 0.))
        for args in (dict(vx=.005), dict(vx=-.005), dict(vx=0., vy=.002), dict(vx=0., wz=.01)):
            self.assertLess(float(self.reward(cx=0., **args)), stop)
        forward = float(self.reward(.02, .02))
        for args in (dict(vy=.001), dict(vy=-.001), dict(wz=.01), dict(wz=-.01)):
            self.assertLess(float(self.reward(.02, .02, **args)), forward)
        # Optional nonzero lateral/yaw commands also have their own minima.
        matching = float(self.reward(.02, .02, vy=.003, cy=.003, wz=.01, cw=.01))
        self.assertAlmostEqual(matching, forward, places=12)

    def test_stop_penalty_distinguishes_residual_creep(self):
        standstill = float(self.reward(0., 0., dt=.1))
        stopped = float(self.reward(.00412, 0., dt=.1))
        old_stopped = float(self.reward(.00412, 0., dt=.1, version='continuous_v3'))
        self.assertLess(stopped, old_stopped)
        self.assertLess(stopped, .75*standstill)
        self.assertAlmostEqual(float(self.reward(-.00412, 0., dt=.1)), stopped, places=12)
        stop_values = [float(self.reward(v, 0.)) for v in (0., .001, .002, .00412, .01, .02)]
        self.assertTrue(all(a > b for a, b in zip(stop_values, stop_values[1:])))

    def test_nonnegative_finite_rewards_over_large_physical_errors(self):
        for dtype in (torch.float32, torch.float64):
            vx = torch.linspace(-10., 10., 401, dtype=dtype, device=self.device)
            velocity = torch.stack((vx, .3*vx, 2*vx), -1)
            command = torch.zeros_like(velocity)
            command[:, 0] = .2
            cost = torch.full_like(vx, .05)
            score = velocity_reward('sustain_v4', velocity, command, cost, cost, cost, cost)
            self.assertTrue(bool(torch.isfinite(score).all()))
            self.assertTrue(bool((score >= 0).all()))

    def test_large_positive_unused_exponential_does_not_overflow(self):
        velocity = self.tensor([[100., 0., 0.]], requires_grad=True)
        command, zero = self.tensor([[100., 0., 0.]]), self.tensor([0.])
        score = velocity_reward('sustain_v4', velocity, command, zero, zero, zero, zero)
        gradient = torch.autograd.grad(score.sum(), velocity)[0]
        self.assertTrue(bool(torch.isfinite(score).all() & torch.isfinite(gradient).all()))

    def test_costs_are_monotonic_and_have_no_amplitude_bonus(self):
        reference = float(self.reward(.02, .2))
        for key in ('action', 'acceleration', 'slip', 'work'):
            scores = [float(self.reward(.02, .2, **{key: v})) for v in (0., .001, .01, .1)]
            self.assertAlmostEqual(scores[0], reference, places=12)
            self.assertTrue(all(a > b for a, b in zip(scores, scores[1:])), (key, scores))
        # No angle, gait phase or mode argument exists. Matching physical states
        # and costs therefore use the same formula for snake and mixed policies.

    def test_equal_distance_burst_does_not_beat_continuous_progress(self):
        dt = .2
        stationary = 100*float(self.reward(0., .2, dt=dt))
        sustained = 100*float(self.reward(.02, .2, dt=dt))
        burst_equal_distance = 10*float(self.reward(.2, .2, dt=dt))+90*float(self.reward(0., .2, dt=dt))
        short_kick = 5*float(self.reward(.1, .2, dt=dt))+95*float(self.reward(0., .2, dt=dt))
        self.assertGreater(sustained, burst_equal_distance)
        self.assertGreater(sustained, short_kick)
        self.assertGreater(short_kick, stationary)
        self.assertAlmostEqual(sustained, 29.999444490907877, places=9)

    def test_failure_does_not_profit_over_same_prefix_safe_continuation(self):
        # This tests the current terminal -2 convention, not future reset
        # exploitation or the physical feasibility of avoiding a failure.
        dt, gamma = .2, .99
        future = [float(self.reward(v, .2, dt=dt)) for v in (0., .01, .03, .02)]
        safe_return = sum(gamma**i*r for i, r in enumerate(future))
        failed_return = future[0]-2.
        self.assertGreater(safe_return, failed_return)
        self.assertGreaterEqual(min(future), 0.)

    def test_macro_dt_integrates_same_physical_cost_rates(self):
        # Under the finite-difference contract: squared delta scales as dt^2,
        # squared second difference as dt^4, integrated work as dt.
        integrals = []
        for dt in (.05, .1, .2, .5):
            reward = self.reward(.02, .2, vy=.001, wz=.002,
                                 action=.008*(dt/.1)**2,
                                 acceleration=.04*(dt/.1)**4,
                                 slip=.002, work=.03*dt, dt=dt)
            integrals.append(float(reward)*20/dt)
        for value in integrals[1:]:
            self.assertAlmostEqual(value, integrals[0], places=11)

    def test_missing_acceleration_is_zero(self):
        velocity, command, zero = self.tensor([[.02, 0., 0.]]), self.tensor([[.2, 0., 0.]]), self.tensor([0.])
        a = velocity_reward('sustain_v4', velocity, command, zero, zero, zero)
        b = velocity_reward('sustain_v4', velocity, command, zero, zero, zero, zero)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_dt_rejected(self):
        for dt in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                self.reward(0., .2, dt=dt)

    def test_old_v3_formula_unchanged(self):
        score = self.reward(0., .2, version='continuous_v3')
        self.assertAlmostEqual(float(score), .1/math.sqrt(26.), places=12)
        self.assertAlmostEqual(float(self.reward(.2, .2, version='continuous_v3')), .1, places=12)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable on this host')
    def test_cpu_cuda_parity(self):
        generator = torch.Generator().manual_seed(312)
        velocity = torch.randn(257, 3, generator=generator, dtype=torch.float64)*.02
        command = torch.zeros_like(velocity)
        command[:, 0] = .005+.195*torch.rand(257, generator=generator, dtype=torch.float64)
        costs = [torch.rand(257, generator=generator, dtype=torch.float64)*.01 for _ in range(4)]
        cpu = velocity_reward('sustain_v4', velocity, command, *costs, policy_dt=.2)
        cuda = velocity_reward('sustain_v4', velocity.cuda(), command.cuda(),
                               *(value.cuda() for value in costs), policy_dt=.2)
        torch.testing.assert_close(cpu, cuda.cpu(), rtol=1e-12, atol=1e-12)

    def _locomotion(self, vx=0.03, vy=0.0, wz=0.0, vx_cmd=0.04, wz_cmd=0.0, stop=0.0,
                    mode=2, rate=0.0, acc=0.0, slip=0.0, work_rate=0.0, clearance=0.01,
                    yaw_act=0.2, ten_act=0.2, failed=False, dt=0.1, config=None):
        vel = self.tensor([[vx, vy, wz]])
        cmd = self.tensor([[vx_cmd, wz_cmd, stop]])
        mid = torch.tensor([mode], dtype=torch.long, device=self.device)
        fail = torch.tensor([failed], dtype=torch.bool, device=self.device)
        scalars = [self.tensor([v]) for v in (rate, acc, slip, work_rate, clearance, yaw_act, ten_act)]
        return locomotion_reward(vel, cmd, mid, *scalars, fail, policy_dt=dt, config=config)

    def test_locomotion_reward_tracks_speed_stop_barrier_and_synergy(self):
        # Forward progress increases reward toward commanded speed
        r0, _ = self._locomotion(vx=0.0, vx_cmd=0.04, mode=0)
        r_mid, _ = self._locomotion(vx=0.02, vx_cmd=0.04, mode=0)
        r_tgt, _ = self._locomotion(vx=0.04, vx_cmd=0.04, mode=0)
        self.assertLess(float(r0), float(r_mid))
        self.assertLess(float(r_mid), float(r_tgt))

        # Stop lock rewards zero velocity strictly over creep
        r_stop0, _ = self._locomotion(vx=0.0, wz=0.0, vx_cmd=0.0, stop=1.0, mode=0)
        r_creep, _ = self._locomotion(vx=0.01, wz=0.0, vx_cmd=0.0, stop=1.0, mode=0)
        self.assertGreater(float(r_stop0), float(r_creep))

        # Soft clearance barrier penalizes low wheel clearance and failure drops sharply
        r_safe, terms_safe = self._locomotion(clearance=0.01)
        r_tight, terms_tight = self._locomotion(clearance=0.003)
        r_fail, _ = self._locomotion(clearance=0.001, failed=True)
        self.assertAlmostEqual(float(terms_safe['r_barrier']), 0.0, places=12)
        self.assertLess(float(terms_tight['r_barrier']), 0.0)
        self.assertGreater(float(r_safe), float(r_tight))
        self.assertGreater(float(r_tight), float(r_fail))

        # Mixed synergy is gated by forward progress and active only in mixed mode
        _, terms_mix_move = self._locomotion(vx=0.04, vx_cmd=0.04, mode=2, yaw_act=0.2, ten_act=0.2)
        _, terms_mix_still = self._locomotion(vx=0.0, vx_cmd=0.04, mode=2, yaw_act=0.2, ten_act=0.2)
        _, terms_snake = self._locomotion(vx=0.04, vx_cmd=0.04, mode=0, yaw_act=0.2, ten_act=0.2)
        self.assertGreater(float(terms_mix_move['r_synergy']), 0.0)
        self.assertAlmostEqual(float(terms_mix_still['r_synergy']), 0.0, places=12)
        self.assertAlmostEqual(float(terms_snake['r_synergy']), 0.0, places=12)

        # Custom UnifiedRewardConfig overrides weights cleanly
        r_custom, _ = self._locomotion(config=UnifiedRewardConfig(alive_bonus=0.5))
        self.assertGreater(float(r_custom), float(r_safe))

        with self.assertRaises(ValueError):
            self._locomotion(dt=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but not available')
    devices = ['cpu', 'cuda'] if args.device == 'auto' and torch.cuda.is_available() else [
        'cpu' if args.device == 'auto' else args.device]
    reports = []
    for device in devices:
        case = type(f'RewardContracts_{device}', (RewardContracts,), {'device': device})
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(case))
        reports.append(dict(device=device, tests=result.testsRun, failures=len(result.failures),
                            errors=len(result.errors), skipped=len(result.skipped), passed=result.wasSuccessful()))
    payload = dict(scope='Numerical reward contracts, not learned locomotion or physical validation',
                   torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                   devices=reports, passed=all(r['passed'] for r in reports))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload), flush=True)
    raise SystemExit(0 if payload['passed'] else 1)


if __name__ == '__main__':
    main()

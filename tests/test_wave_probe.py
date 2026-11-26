"""Probe contracts on synthetic tensors; no physics or learned-motion claims."""
import math
import hashlib
from pathlib import Path
import tempfile
import unittest
import torch

from serpworm.probes.fixed_waves import (Case, actions_from_angles, actual_yaw, cases,
                                         requested_angles, summarize_trace, freeze_sources)
from serpworm.envs.rewards import velocity_reward


def synthetic(seconds=40., mode='snake'):
    dt = .2
    count = round(seconds/dt)
    t = torch.arange(count, dtype=torch.float64)*dt
    td = torch.arange(count*10, dtype=torch.float64)*.02
    yaw = torch.zeros(count*10, 1, 6, 4, dtype=torch.float64)
    signal = (math.radians(20)*torch.sin(2*math.pi*td/4))[:, None].expand(-1, 4)
    for field in (0, 1, 2): yaw[:, 0, field] = signal
    servo = torch.full((count, 1, 3, 10), math.pi/2, dtype=torch.float64)
    if mode == 'mixed':
        motion = math.radians(30)*torch.sin(2*math.pi*t/4)
        servo[:, 0] += motion[:, None, None]
    velocity = torch.zeros(count, 1, 3, dtype=torch.float64); velocity[:, 0, 0] = .01
    initial = torch.zeros(1, 10, 3, dtype=torch.float64)
    initial[0, :, 0] = torch.linspace(0., 1., 10)
    initial[..., 2] = .1
    positions = initial[None].expand(count, -1, -1, -1).clone()
    positions[..., 0] += ((t+.2)*.01)[:, None, None]
    metrics = torch.zeros(count, 1, 8, dtype=torch.float64); metrics[..., 2] = .03
    trace = dict(policy_dt_s=dt, sensor_dt_s=.02, time_start_s=t,
        active=torch.ones(count, 1, dtype=torch.bool), terminated=torch.zeros(count, 1, dtype=torch.bool),
        truncated=torch.zeros(count, 1, dtype=torch.bool), body_velocity=velocity,
        filtered_velocity=velocity.clone(), initial_plate_positions_m=initial,
        plate_positions_m=positions, yaw_diagnostics=yaw, servo_diagnostics=servo,
        actions=torch.zeros(count, 1, 9, dtype=torch.float64), metrics=metrics,
        tension=torch.ones(count, 1, 5, 4, dtype=torch.float64),
        normal_force=torch.ones(count, 1, 10, 2, dtype=torch.float64),
        gap=torch.full((count, 1, 5), .1, dtype=torch.float64),
        stop_force=torch.zeros(count, 1, 5, dtype=torch.float64),
        action_cost=torch.zeros(count, 1, dtype=torch.float64),
        acceleration_cost=torch.zeros(count, 1, dtype=torch.float64),
        guard_reasons={'joint_limit': torch.zeros(count, 1, dtype=torch.bool)})
    trace['truncated'][-1] = True
    return trace


class WaveProbeTests(unittest.TestCase):
    def test_four_joints_not_six_fields(self):
        diagnostic = torch.zeros(10, 3, 6, 4)
        diagnostic[:, :, 2, :] = torch.tensor([10., 20., 30., 40.])
        diagnostic[:, :, 4, :] = .5  # Torque must never become a 28.65-degree angle.
        actual = actual_yaw(diagnostic)
        self.assertEqual(actual.shape, (10, 3, 4))
        torch.testing.assert_close(actual[0, 0], torch.tensor([10., 20., 30., 40.]))

    def test_rows_are_independent_and_full_range_mapping(self):
        selected = [Case('zero', 'static', 0.), Case('middle', 'static', 90.), Case('high', 'static', 180.)]
        servo, yaw = requested_angles(3., selected)
        torch.testing.assert_close(servo[:, 0], torch.tensor([0., 90., 180.], dtype=torch.float64))
        act = actions_from_angles(servo, yaw, torch.tensor([0., 90., 180.]))
        self.assertTrue(torch.equal(act, torch.zeros_like(act)))
        commands = torch.tensor([[0.]*10, [180.]*10], dtype=torch.float64)
        act = actions_from_angles(commands, torch.tensor([[-45.]*4, [45.]*4]), 90.)
        torch.testing.assert_close(act[0], -torch.ones(9, dtype=torch.float64))
        torch.testing.assert_close(act[1], torch.ones(9, dtype=torch.float64))
        # Regression for old target[:, arange(n)] writing every actuator column.
        target = torch.full((10, 10), 90.)
        index = torch.arange(10); target[index, index] = 180.
        self.assertTrue(torch.equal((target == 180.).sum(1), torch.ones(10, dtype=torch.long)))

    def test_wave_period_phase_and_reachable_rate(self):
        row = Case('test', 'mixed', 90., 45., 90., 10., 90.)
        servo0, yaw0 = requested_angles(3., [row], ramp_s=0.)
        servo1, yaw1 = requested_angles(13., [row], ramp_s=0.)
        torch.testing.assert_close(servo0, servo1)
        torch.testing.assert_close(yaw0, yaw1)
        _, shifted = requested_angles(5.5, [row], ramp_s=0.)
        self.assertAlmostEqual(float(yaw0[0, 1]), float(shifted[0, 0]), places=10)
        eps = 1e-5
        before, _ = requested_angles(5.-eps, [row], ramp_s=0.)
        after, _ = requested_angles(5.+eps, [row], ramp_s=0.)
        speed = float(((after-before)/(2*eps)).abs().max())*math.pi/180
        self.assertLessEqual(speed, 1.)  # +/-90 degrees, T=10 s respects 1 rad/s target rate.
        self.assertAlmostEqual(speed, 2*math.pi*math.pi/2/10, places=6)

    def test_ramp_starts_neutral_and_case_names_unique(self):
        collection = cases()
        self.assertEqual(len(collection), len({row.name for row in collection}))
        servo, yaw = requested_angles(0., collection)
        torch.testing.assert_close(servo[:, 0], torch.tensor([row.neutral_deg for row in collection], dtype=torch.float64))
        self.assertTrue(torch.equal(yaw, torch.zeros_like(yaw)))

    def test_steady_first_episode_can_pass_without_stitching(self):
        trace = synthetic()
        report = summarize_trace(trace, 0, Case('wave', 'snake', 90., 35., 0., 4.), 40.)
        self.assertTrue(report['sustained_yaw_forward_gate'])
        self.assertTrue(report['first_episode_complete'])
        self.assertAlmostEqual(report['tail']['summary']['mean_vx_mm_s'], 10.)
        self.assertEqual(len(report['tail']['summary']['actual_yaw_pp_deg']), 4)
        self.assertAlmostEqual(report['tail']['summary']['world_initial_forward_displacement_mm'], 200., places=5)

    def test_short_or_startup_only_case_cannot_pass(self):
        short = summarize_trace(synthetic(10.), 0, Case('short', 'snake', 90., 35., 0., 4.), 10.)
        self.assertFalse(short['sustained_yaw_forward_gate'])
        self.assertFalse(short['duration_sufficient_for_gate'])
        trace = synthetic(); trace['body_velocity'][100:, :, :] = 0.
        trace['filtered_velocity'][100:, :, :] = 0.
        trace['yaw_diagnostics'][1000:, :, 2, :] = 0.
        frozen = summarize_trace(trace, 0, Case('frozen', 'snake', 90., 35., 0., 4.), 40.)
        self.assertFalse(frozen['sustained_yaw_forward_gate'])

    def test_body_projection_without_world_progress_cannot_pass(self):
        trace = synthetic()
        trace['plate_positions_m'][:] = trace['initial_plate_positions_m']
        report = summarize_trace(trace, 0, Case('no_net_world', 'snake', 90., 35., 0., 4.), 40.)
        self.assertFalse(report['positive_world_forward_progress'])
        self.assertFalse(report['sustained_yaw_forward_gate'])

    def test_post_reset_samples_and_force_peaks_excluded(self):
        trace = synthetic()
        trace['terminated'][49] = True
        trace['active'][50:] = False
        trace['body_velocity'][50:] = 1000.
        trace['tension'][50:] = 1e9
        trace['guard_reasons']['joint_limit'][49] = True
        report = summarize_trace(trace, 0, Case('reset', 'snake', 90., 35., 0., 4.), 40.)
        self.assertFalse(report['sustained_yaw_forward_gate'])
        self.assertFalse(report['first_episode_complete'])
        self.assertEqual(report['valid_duration_s'], 10.)
        self.assertAlmostEqual(report['whole']['summary']['mean_vx_mm_s'], 10.)
        self.assertEqual(report['whole']['summary']['max_tension_n'], 1.)
        self.assertTrue(report['guards']['joint_limit'])

    def test_mixed_requires_two_independent_rope_pairs(self):
        trace = synthetic(mode='mixed')
        report = summarize_trace(trace, 0, Case('mixed', 'mixed', 90., 35., 60., 8.), 40.)
        self.assertTrue(report['sustained_mixed_forward_gate'])
        trace['servo_diagnostics'][:, :, 2, 2:] = math.pi/2
        one = summarize_trace(trace, 0, Case('one_pair', 'mixed', 90., 35., 60., 8.), 40.)
        self.assertFalse(one['sustained_mixed_forward_gate'])
        self.assertEqual(one['tail']['summary']['active_rope_segment_count'], 1)

    def test_pure_snake_reward_uses_four_active_channels(self):
        trace = synthetic()
        trace['action_cost'].fill_(.02)
        trace['acceleration_cost'].fill_(.12)
        report = summarize_trace(trace, 0, Case('snake', 'snake', 90., 35., 0., 4.), 40.)
        v = torch.tensor([[.01, 0., 0.]], dtype=torch.float64)
        command = torch.tensor([[.02, 0., 0.]], dtype=torch.float64)
        expected = velocity_reward('sustain_v4', v, command,
            torch.tensor([.02*14/4]), torch.zeros(1), torch.zeros(1),
            torch.tensor([.12*9/4]), policy_dt=.2)
        self.assertAlmostEqual(report['old_reward_tail_replay']['20']['reward_per_second'], float(expected[0])/.2, places=7)
        self.assertEqual(report['replay_cost_mode'], 'snake_active4')

    def test_sources_are_archived_and_concurrent_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'root'; root.mkdir(); (root/'inputs').mkdir()
            (root/'core.py').write_bytes(b'physics-v1')
            (root/'inputs'/'topology.bin').write_bytes(b'five-segments')
            manifest = dict(source_sha256={'core.py': hashlib.sha256(b'physics-v1').hexdigest()},
                input_sha256={'topology.bin': hashlib.sha256(b'five-segments').hexdigest()})
            out = Path(directory)/'out'; out.mkdir()
            freeze_sources(root, out, manifest)
            self.assertEqual((out/'sources'/'core.py').read_bytes(), b'physics-v1')
            self.assertEqual((out/'sources'/'inputs'/'topology.bin').read_bytes(), b'five-segments')
            (root/'core.py').write_bytes(b'physics-v2')
            other = Path(directory)/'other'; other.mkdir()
            with self.assertRaises(RuntimeError): freeze_sources(root, other, manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)

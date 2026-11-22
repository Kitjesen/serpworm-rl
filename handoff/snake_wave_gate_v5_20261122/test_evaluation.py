"""CPU-only tests of the evaluator's evidence gates; no physics/GPU is started."""
import math
import unittest

import torch

from evaluate_gpu import CASE_NAMES, SPEEDS_MM_S, aggregate, summarize


def synthetic_traces(dt=.2, seconds=20.):
    steps, n = round(seconds/dt), 8
    sensor_dt = .02
    repeats = round(dt/sensor_dt)
    times = torch.arange(steps, dtype=torch.float64)*dt
    sensor_times = (torch.arange(steps*repeats, dtype=torch.float64)+1)*sensor_dt
    command = torch.zeros(steps, n, 3, dtype=torch.float64)
    command[:, :, 0] = torch.tensor(SPEEDS_MM_S, dtype=torch.float64)*1e-3
    command[times >= 10.-1e-8, 7, 0] = 0.
    velocity = command.clone()
    yaw = torch.zeros(steps*repeats, n, 6, 4, dtype=torch.float64)
    wave = math.radians(10.)*torch.sin(2*math.pi*.5*sensor_times)
    for field in range(3):
        yaw[:, :, field, :] = wave[:, None, None]
    yaw[:, :, 5] = .5
    servo = torch.zeros(steps, n, 3, 10, dtype=torch.float64)
    servo_wave = math.radians(10.)*torch.sin(2*math.pi*.5*(times+dt))
    for field in range(3):
        servo[:, :, field, :] = math.pi/2+servo_wave[:, None, None]
    # Stops remain stationary with a constant posture, including the commanded
    # stop after 10 s; position history stores end-of-interval plate centroids.
    yaw[:, 0, :3] = 0.
    servo[:, 0] = math.pi/2
    yaw[sensor_times > 10.+1e-8, 7, :3] = 0.
    servo[times >= 10.-1e-8, 7] = math.pi/2
    positions = velocity.cumsum(0)[:, :, None, :].expand(-1, -1, 10, -1).clone()*dt
    metrics = torch.zeros(steps, n, 8, dtype=torch.float64)
    metrics[:, :, 2] = .02
    truncated = torch.zeros(steps, n, dtype=torch.bool)
    if math.isclose(seconds, 20.):
        truncated[-1] = True
    return dict(time_start_s=times, time_end_s=times+dt,
                policy_dt_s=dt, diagnostics_dt_s=sensor_dt,
                command=command, body_velocity=velocity,
                filtered_velocity=velocity.clone(), metrics=metrics,
                actions=torch.zeros(steps, n, 9, dtype=torch.float64),
                diagnostics_50hz=yaw, servo_diagnostics_policy_hz=servo,
                plate_positions_m=positions,
                initial_plate_positions_m=torch.zeros(n, 10, 3, dtype=torch.float64),
                active=torch.ones(steps, n, dtype=torch.bool),
                terminated=torch.zeros(steps, n, dtype=torch.bool), truncated=truncated,
                case_names=CASE_NAMES)


class SustainedEvaluationTests(unittest.TestCase):
    def test_twenty_second_final_truncation_is_valid(self):
        traces = synthetic_traces()
        results = summarize(traces, 20., 'mixed')
        totals = aggregate(results)
        self.assertEqual(totals['moving_sustained_case_count'], 6)
        self.assertEqual(totals['mixed_rope_sustained_case_count'], 6)
        self.assertEqual(totals['moving_complete_window_count'], 6)
        self.assertEqual(totals['moving_guard_failures'], 0)
        self.assertEqual(totals['all_stop_mean_abs_vx_mm_s'], 0.)
        for row in results:
            self.assertTrue(row['time_limit_reached'])
            self.assertTrue(row['first_episode_complete'])
            self.assertTrue(row['window_complete'])
            self.assertAlmostEqual(row['valid_duration_s'], 20.)
            self.assertAlmostEqual(row['first_truncation_time_s'], 20.)
        self.assertEqual(len(results[2]['tail_windows']), 5)
        self.assertAlmostEqual(results[2]['summary']['signed_forward_velocity_integral_mm'], 200.)
        self.assertAlmostEqual(results[2]['summary']['world_centroid_displacement_mm'][0], 200.)
        self.assertEqual(results[7]['summary_window_start_s'], 15.)

    def test_startup_motion_does_not_pass_sustained_screen(self):
        traces = synthetic_traces()
        lane = 2
        after_startup = traces['time_start_s'] >= 6.
        traces['body_velocity'][after_startup, lane] = 0.
        traces['filtered_velocity'][after_startup, lane] = 0.
        traces['diagnostics_50hz'][round(6./.02):, lane, :3] = 0.
        traces['servo_diagnostics_policy_hz'][after_startup, lane] = math.pi/2
        results = summarize(traces, 20., 'mixed')
        self.assertFalse(results[lane]['sustained_motion_screen'])
        self.assertEqual(results[lane]['freeze_window_count'], 5)
        self.assertEqual(results[lane]['summary']['mean_vx_mm_s'], 0.)
        self.assertEqual(aggregate(results)['moving_sustained_case_count'], 5)

    def test_masked_reset_never_stitches_new_episode_motion(self):
        traces = synthetic_traces()
        lane, fail_step = 3, 9
        traces['terminated'][fail_step, lane] = True
        traces['active'][fail_step+1:, lane] = False
        traces['body_velocity'][fail_step+1:, lane, 0] = 9.
        traces['filtered_velocity'][fail_step+1:, lane, 0] = 9.
        results = summarize(traces, 20., 'mixed')
        item = results[lane]
        self.assertTrue(item['failed'])
        self.assertFalse(item['time_limit_reached'])
        self.assertFalse(item['first_episode_complete'])
        self.assertFalse(item['window_complete'])
        self.assertIsNone(item['summary'])
        self.assertAlmostEqual(item['first_termination_time_s'], 2.)
        self.assertAlmostEqual(item['valid_duration_s'], 2.)
        self.assertAlmostEqual(item['windows_2s'][0]['summary']['mean_vx_mm_s'], 50.)
        self.assertIsNone(item['windows_2s'][1]['summary'])
        totals = aggregate(results)
        self.assertEqual(totals['moving_guard_failures'], 1)
        self.assertEqual(totals['moving_sustained_case_count'], 5)
        self.assertIsNone(totals['moving_tail_mean_vx_mm_s'])

    def test_stop_uses_absolute_velocity_not_signed_cancellation(self):
        traces = synthetic_traces()
        # Twenty-five policy samples in the stop tail: choose alternating pairs
        # plus a zero sample, making the signed mean exactly zero.
        pattern = torch.tensor([.002, -.002]*12+[0.], dtype=torch.float64)
        traces['body_velocity'][-25:, 0, 0] = pattern
        results = summarize(traces, 20., 'mixed')
        stop = results[0]['summary']
        self.assertAlmostEqual(stop['mean_vx_mm_s'], 0., places=10)
        self.assertAlmostEqual(stop['mean_abs_vx_mm_s'], 1.92)
        self.assertAlmostEqual(aggregate(results)['all_stop_mean_abs_vx_mm_s'], .96)

    def test_real_time_windows_work_at_five_and_ten_hz(self):
        totals = []
        for dt in (.1, .2):
            results = summarize(synthetic_traces(dt=dt), 20., 'mixed')
            totals.append(aggregate(results))
            self.assertEqual(results[2]['valid_steps'], round(20./dt))
            self.assertAlmostEqual(results[2]['summary_valid_duration_s'], 10.)
            self.assertEqual(len(results[2]['tail_windows']), 5)
            for window in results[2]['tail_windows']:
                self.assertAlmostEqual(window['valid_duration_s'], 2.)
                self.assertEqual(len(window['policy_step_indices']), round(2./dt))
            self.assertAlmostEqual(results[2]['summary']['signed_forward_velocity_integral_mm'], 200.)
        self.assertAlmostEqual(totals[0]['moving_tail_mean_vx_mm_s'], totals[1]['moving_tail_mean_vx_mm_s'])
        self.assertEqual(totals[0]['moving_sustained_case_count'], totals[1]['moving_sustained_case_count'])

    def test_stochastic_motion_never_counts_as_deterministic_acceptance(self):
        results = summarize(synthetic_traces(), 20., 'mixed', stochastic=True)
        totals = aggregate(results, stochastic=True)
        self.assertEqual(totals['observed_sustained_case_count'], 6)
        self.assertEqual(totals['moving_sustained_case_count'], 0)
        self.assertEqual(totals['mixed_rope_sustained_case_count'], 0)
        self.assertFalse(totals['acceptance_eligible'])
        self.assertFalse(results[2]['deterministic_sustained_motion_screen'])

    def test_one_duplicated_rope_pair_is_not_two_independent_channels(self):
        traces = synthetic_traces()
        traces['servo_diagnostics_policy_hz'][:, :, :, 2:] = math.pi/2
        results = summarize(traces, 20., 'mixed')
        row = results[2]
        self.assertEqual(row['summary']['actual_servo_active_channel_count'], 2)
        self.assertEqual(row['summary']['actual_rope_active_segment_count'], 1)
        self.assertTrue(row['sustained_motion_screen'])
        self.assertFalse(row['mixed_rope_motion_screen'])

    def test_missing_last_sample_invalidates_complete_tail(self):
        traces = synthetic_traces()
        traces['active'][-1, 2] = False
        results = summarize(traces, 20., 'mixed')
        self.assertFalse(results[2]['first_episode_complete'])
        self.assertFalse(results[2]['window_complete'])
        self.assertFalse(results[2]['sustained_motion_screen'])
        self.assertIsNone(aggregate(results)['moving_vx_rmse_mm_s'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

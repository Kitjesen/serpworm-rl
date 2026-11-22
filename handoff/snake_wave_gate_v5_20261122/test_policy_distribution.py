"""Synthetic contracts only; no CUDA physics and no training."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import torch
from probe_policy_distribution import lanes, choose_actions, configure_noise, summarize, verified_config


def synthetic():
    t = torch.arange(100, dtype=torch.float64)*.2
    td = torch.arange(1000, dtype=torch.float64)*.02
    diag = torch.zeros(1000, 8, 6, 4, dtype=torch.float64)
    wave = math.radians(20)*torch.sin(2*math.pi*td/4)
    diag[:, :, 2] = wave[:, None, None]
    diag[:, :, 0] = 2*wave[:, None, None]
    diag[:, :, 1] = 1.5*wave[:, None, None]
    diag[:, :, 5] = .5  # The old wrong-axis bug would expose this as 28.65 degrees.
    initial = torch.zeros(8, 10, 3, dtype=torch.float64)
    initial[:, :, 0] = torch.linspace(0., 1., 10)
    positions = initial[None].expand(100, -1, -1, -1).clone()
    positions[:, :, :, 0] += ((t+.2)*.01)[:, None, None]
    velocity = torch.zeros(100, 8, 3, dtype=torch.float64); velocity[..., 0] = .01
    metric = torch.zeros(100, 8, 8, dtype=torch.float64); metric[..., 2] = .03
    return dict(time_s=t, policy_dt_s=.2, sensor_dt_s=.02,
        active=torch.ones(100, 8, dtype=torch.bool), terminated=torch.zeros(100, 8, dtype=torch.bool),
        body_velocity=velocity, yaw_diagnostics=diag, servo_diagnostics=torch.zeros(100, 8, 3, 10),
        actions=torch.zeros(100, 8, 4), metrics=metric, initial_plate_positions_m=initial, plate_positions_m=positions)


class DistributionTests(unittest.TestCase):
    def test_eight_lanes_and_independent_action_selection(self):
        definitions = lanes(); self.assertEqual(len(definitions), 8)
        self.assertEqual({(r['command_mm_s'],r['controller'],r['sensor_noise']) for r in definitions},
            {(s,c,n) for s in (20.,200.) for c in ('mean','sample') for n in (False,True)})
        active = torch.ones(8, dtype=torch.bool); active[-1] = False
        result = choose_actions(torch.ones(8, 4), torch.full((8, 4), 2.), definitions, active)
        for lane, row in enumerate(definitions):
            expected = 0. if not active[lane] else (2. if row['controller']=='sample' else 1.)
            self.assertTrue(torch.equal(result[lane], torch.full((4,), expected)))

    def test_initial_noise_does_not_leak_into_disabled_lanes(self):
        class Env:
            device='cpu'; dtype=torch.float64
            noise_scale=torch.tensor([.002]*14+[.01]*3, dtype=torch.float64)
            calls=0
            def reset(self, mask):
                self.calls += 1
                self.initial_packet = torch.ones(8,17)*self.noise_scale
                return self.initial_packet
        env = Env(); obs = configure_noise(env, lanes())
        self.assertEqual(env.calls, 1); self.assertTrue(env.noise)
        for lane, row in enumerate(lanes()):
            self.assertEqual(bool(obs[lane].any()), row['sensor_noise'])

    def test_requested_applied_actual_keep_four_joints_and_world_delta(self):
        result = summarize(synthetic(), lanes())
        for lane in result:
            self.assertEqual(len(lane['tail_windows_2s']), 5)
            for window in lane['tail_windows_2s']:
                self.assertTrue(window['complete'])
                self.assertEqual(len(window['actual_yaw_pp_deg']), 4)
                self.assertAlmostEqual(window['mean_vx_mm_s'], 10.)
                self.assertAlmostEqual(window['world_initial_forward_displacement_mm'], 20., places=4)
                self.assertGreater(window['requested_yaw_pp_deg'][0], window['applied_yaw_pp_deg'][0])
                self.assertGreater(window['applied_yaw_pp_deg'][0], window['actual_yaw_pp_deg'][0])

    def test_first_reset_is_final_even_if_new_episode_moves_fast(self):
        trace = synthetic(); trace['terminated'][49, 0] = True; trace['active'][50:,0] = False
        trace['body_velocity'][50:, 0] = 1000.
        result = summarize(trace, lanes())[0]
        self.assertTrue(result['failed']); self.assertFalse(result['first_episode_complete'])
        self.assertEqual(result['valid_duration_s'], 10.)
        self.assertTrue(all(w['valid_duration_s']==0 for w in result['tail_windows_2s']))

    def test_frozen_source_hash_mismatch_blocks_import(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source'; root.mkdir(); (root/'inputs').mkdir()
            run=Path(d)/'run'; run.mkdir(); checkpoint=run/'checkpoint.pt'
            names=('gpu_core.py','tensor_env.py','rewards.py','train_gpu.py','wheel_stiction.py')
            for name in names: (root/name).write_bytes(b'original')
            (root/'inputs'/'topology.npz').write_bytes(b'topology')
            config=dict(source_sha256={n:hashlib.sha256(b'original').hexdigest() for n in names},
                input_sha256={'topology.npz':hashlib.sha256(b'topology').hexdigest()})
            (run/'config.json').write_text(json.dumps(config))
            verified_config(root, checkpoint)
            (root/'tensor_env.py').write_bytes(b'new environment')
            with self.assertRaises(RuntimeError): verified_config(root, checkpoint)


if __name__ == '__main__': unittest.main(verbosity=2)

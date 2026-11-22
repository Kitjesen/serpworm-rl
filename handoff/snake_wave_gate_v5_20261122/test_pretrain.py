"""Synthetic numerical BC contracts; fixtures are never physical gate evidence."""
import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from pretrain_actor import prepare, cycle_split, fit_actor, manifest, angle_rmse_deg, PHYSICS_SOURCES
from train_gpu import ActorCritic, load_bc_actor, require_initial_motion


class PretrainContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_bc_contract_')
        self.root = Path(self.temp.name)
        for name in PHYSICS_SOURCES:
            (self.root/name).write_text('# Synthetic test fixture only\n')
        (self.root/'inputs').mkdir()
        (self.root/'inputs'/'synthetic_parameters.json').write_text('{}')
        sources, inputs = manifest(self.root)
        t = torch.arange(150, dtype=torch.float64)*.2
        phase = t.float()*2*math.pi/6
        obs = torch.zeros(len(t), 570)
        obs[:, 0], obs[:, 1] = phase.sin(), phase.cos()
        obs[:, -3] = .1
        action = torch.zeros(len(t), 9)
        action[:, -4:] = torch.stack([.6*(phase+j*.6).sin() for j in range(4)], -1)
        self.data = dict(schema='fixed_wave_dataset_v1', source_sha256=sources,
            input_sha256=inputs, device='cuda', physics_dt_s=.00025, policy_dt_s=.2, history_span_s=1.8,
            command_scale=[.2, .01, .2], cases={'synthetic': dict(mode='snake',
                actor_observations=obs, teacher_actions=action, time_s=t, period_s=6.,
                servo_neutral_deg=90., command_vx_mm_s=20., summary=dict(gate_passed=True,
                    reward_replay_better_than_stationary=True, complete=True, valid_duration_s=30.))})

    def tearDown(self):
        self.temp.cleanup()

    def prep(self, data=None, mode='snake', allow_cpu=False):
        return prepare(data or self.data, 'synthetic', mode, allow_cpu, self.root)

    def test_gate_rejects_failed_short_and_reward_inferior_cases(self):
        for key in ('gate_passed', 'reward_replay_better_than_stationary', 'complete'):
            d = copy.deepcopy(self.data)
            d['cases']['synthetic']['summary'][key] = False
            with self.assertRaises(ValueError): self.prep(d)
        d = copy.deepcopy(self.data)
        d['cases']['synthetic']['period_s'] = 12.
        with self.assertRaises(ValueError): self.prep(d)

    def test_no_observation_truth_channels_or_manifest_drift(self):
        d = copy.deepcopy(self.data)
        d['cases']['synthetic']['actor_observations'] = torch.zeros(150, 571)
        with self.assertRaises(ValueError): self.prep(d)
        d = copy.deepcopy(self.data)
        d['source_sha256']['tensor_env.py'] = 'wrong'
        with self.assertRaises(ValueError): self.prep(d)
        d = copy.deepcopy(self.data)
        d['cases']['synthetic']['actor_observations'][:, -3] = .5
        with self.assertRaises(ValueError): self.prep(d)

    def test_mode_dimensions_and_neutral_mapping(self):
        obs, action, _, _, proof = self.prep()
        self.assertEqual(tuple(action.shape), (150, 4))
        d = copy.deepcopy(self.data)
        d['cases']['synthetic']['mode'] = 'mixed'
        _, action, _, _, proof = self.prep(d, mode='mixed')
        self.assertEqual(tuple(action.shape), (150, 9))
        x = torch.zeros(1, 9)
        y = x.clone(); y[:, 0] = 1.; y[:, 5] = 1.
        rmse = angle_rmse_deg(x, y, 'mixed', 45.)
        self.assertAlmostEqual(rmse[0], 135.)
        self.assertAlmostEqual(rmse[5], 45.)
        with self.assertRaises(ValueError): self.prep(d)

    def test_split_is_complete_cycles_with_history_purge(self):
        _, _, train, validation, proof = self.prep()
        time = self.data['cases']['synthetic']['time_s']
        self.assertEqual(int(train.sum()), 60)
        self.assertEqual(int(validation.sum()), 60)
        self.assertLess(float(time[train][-1]), 12.)
        self.assertGreaterEqual(float(time[validation][0]), 18.)
        self.assertGreater(float(time[validation][0]-time[train][-1]), 1.8)
        self.assertFalse(bool((train & validation).any()))

    def test_cpu_oracle_never_authorizes_gpu_warmstart(self):
        d = copy.deepcopy(self.data); d['device'] = 'cpu'
        with self.assertRaises(ValueError): self.prep(d)
        self.assertFalse(self.prep(d, allow_cpu=True)[-1]['gpu_warmstart_eligible'])

    def test_finite_optimization_improves_heldout_cycle(self):
        obs, target, train, validation, _ = self.prep()
        model = ActorCritic(570, 4)
        report = fit_actor(model, obs, target, train, validation, epochs=180, batch_size=60, lr=1e-3)
        self.assertTrue(report['heldout_improved'])
        self.assertLess(report['best_validation_action_mse'], report['initial_validation_action_mse']*.1)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.actor.parameters()))

    def test_warmstart_actor_only_and_strict_proof(self):
        _, _, _, _, proof = self.prep()
        teacher, student = ActorCritic(570, 4), ActorCritic(570, 4)
        before = copy.deepcopy(student.critic.state_dict())
        path = self.root/'synthetic_actor.pt'
        saved = dict(schema='bc_actor_v1', actor=teacher.actor.state_dict(), proof=proof,
            dataset_sha256='synthetic fixture; not real evidence', report={'heldout_improved': True})
        torch.save(saved, path)
        args = SimpleNamespace(mode='snake', servo_neutral_deg=90., vx_max_mm_s=200., initial_std=.65)
        load_bc_actor(student, path, args, self.root)
        self.assertTrue(all(torch.equal(v, student.actor.state_dict()[k]) for k, v in teacher.actor.state_dict().items()))
        self.assertTrue(all(torch.equal(v, student.critic.state_dict()[k]) for k, v in before.items()))
        torch.testing.assert_close(student.logstd.exp(), torch.full((4,), .65))
        for key, value in [('dataset_device', 'cpu'), ('obs_size', 571), ('servo_neutral_deg', 45.), ('history_span_s', .9)]:
            invalid = copy.deepcopy(saved); invalid['proof'][key] = value
            torch.save(invalid, path)
            with self.assertRaises(ValueError): load_bc_actor(student, path, args, self.root)

    def test_initial_gate_requires_deterministic_motion_before_ppo(self):
        row = dict(case='straight20', command_vx_mm_s=20.,
            failed=False, window_complete=True, first_episode_complete=True,
            deterministic_sustained_motion_screen=True, mixed_rope_motion_screen=True)
        require_initial_motion({'results': [row]}, 'snake', 20.)
        require_initial_motion({'results': [row]}, 'mixed', 20.)
        with self.assertRaises(RuntimeError):
            require_initial_motion({'results': [{**row, 'deterministic_sustained_motion_screen': False}]}, 'snake', 20.)
        with self.assertRaises(RuntimeError):
            require_initial_motion({'results': [{**row, 'mixed_rope_motion_screen': False}]}, 'mixed', 20.)
        with self.assertRaises(RuntimeError): require_initial_motion({'results': [row]}, 'mixed', 50.)


if __name__ == '__main__':
    unittest.main(verbosity=2)

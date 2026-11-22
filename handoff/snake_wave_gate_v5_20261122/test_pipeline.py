"""CPU-only orchestration contracts; subprocesses and CUDA are mocked."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_pipeline as pipeline


def fixture():
    data = dict(schema='fixed_wave_dataset_v1', device='cuda', physics_dt_s=.00025, cases={})
    report = dict(complete=True, results=[])
    for name, mode, kind, neutral, speed in (
            ('snake_good', 'snake', 'snake', 45., 8.),
            ('mixed_good', 'mixed', 'mixed', 90., 10.),
            ('full_faster', 'mixed', 'full', 90., 14.)):
        data['cases'][name] = dict(mode=mode, servo_neutral_deg=neutral,
            summary=dict(gate_passed=True, reward_replay_better_than_stationary=True, complete=True, case_kind=kind))
        report['results'].append(dict(case=dict(name=name), first_episode_complete=True,
            tail=dict(complete=True, summary=dict(mean_vx_mm_s=speed))))
    return data, report


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)/'output'
        self.args = SimpleNamespace(out=str(self.out), rollouts=128, probe_seconds=40., bc_epochs=400)

    def test_selects_only_complete_reward_improving_wave_and_ranks_speed(self):
        data, report = fixture()
        chosen = pipeline.candidates(data, report)
        self.assertEqual(chosen['snake']['case'], 'snake_good')
        self.assertEqual(chosen['mixed']['case'], 'full_faster')
        data['cases']['full_faster']['summary']['reward_replay_better_than_stationary'] = False
        self.assertEqual(pipeline.candidates(data, report)['mixed']['case'], 'mixed_good')
        data['cases']['mixed_good']['summary']['complete'] = False
        self.assertIsNone(pipeline.candidates(data, report)['mixed'])

    def test_rejects_cpu_wrong_dt_and_incomplete_probe(self):
        data, report = fixture()
        for change in ({'device': 'cpu'}, {'physics_dt_s': .001}, {'schema': 'unknown'}):
            with self.assertRaises(ValueError):
                pipeline.candidates({**data, **change}, report)
        with self.assertRaises(ValueError):
            pipeline.candidates(data, {**report, 'complete': False})

    def test_filters_missing_bad_speed_bad_neutral_and_wrong_kind(self):
        data, report = fixture()
        for speed in (float('nan'), -2., None):
            altered = copy.deepcopy(report)
            altered['results'][0]['tail']['summary']['mean_vx_mm_s'] = speed
            self.assertIsNone(pipeline.candidates(data, altered)['snake'])
        for neutral in (float('nan'), -1., 181.):
            altered = copy.deepcopy(data)
            altered['cases']['snake_good']['servo_neutral_deg'] = neutral
            self.assertIsNone(pipeline.candidates(altered, report)['snake'])
        data['cases']['snake_good']['summary']['case_kind'] = 'pose'
        self.assertIsNone(pipeline.candidates(data, report)['snake'])

    def test_no_cuda_creates_nothing_and_starts_nothing(self):
        with mock.patch('torch.cuda.is_available', return_value=False), mock.patch.object(pipeline.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                pipeline.run(self.args)
        self.assertFalse(self.out.exists())
        spawn.assert_not_called()

    def test_existing_output_is_not_overwritten(self):
        self.out.mkdir()
        sentinel = self.out/'sentinel'; sentinel.write_text('keep')
        with mock.patch.object(pipeline, 'require_cuda'), mock.patch.object(pipeline, 'no_active_source_jobs'), mock.patch.object(pipeline.subprocess, 'Popen') as spawn:
            with self.assertRaises(FileExistsError):
                pipeline.run(self.args)
        self.assertEqual(sentinel.read_text(), 'keep')
        spawn.assert_not_called()

    def test_invalid_budget_rejected_before_output(self):
        with mock.patch.object(pipeline, 'require_cuda'):
            for count in (0, 129):
                self.args.rollouts = count
                with self.assertRaises(ValueError): pipeline.run(self.args)
        self.assertFalse(self.out.exists())

    def exercise(self, data=None, report=None, failed=None):
        data0, report0 = fixture()
        data, report = data or data0, report or report0
        events, commands = [], []
        failed = failed or set()

        def spawn(command, **kwargs):
            name = Path(command[2]).name
            destination = Path(command[command.index('--out')+1])
            label = destination.name
            commands.append(command); events.append(('start', label))
            destination.mkdir()
            if name == 'probe_fixed_waves.py':
                (destination/'summary.json').write_text(json.dumps(report))
            elif name == 'pretrain_actor.py' and label not in failed:
                (destination/'actor_checkpoint.pt').write_bytes(b'mocked trusted checkpoint')
            status = 'needs_diagnosis' if label in failed else ('pilot_complete' if name == 'train_gpu.py' else 'complete')
            (destination/'status.json').write_text(json.dumps({'status': status}))

            def wait():
                events.append(('wait', label))
                return 1 if label in failed else 0
            return SimpleNamespace(pid=2000+len(commands), wait=wait)

        with mock.patch.object(pipeline, 'require_cuda'), mock.patch.object(pipeline, 'no_active_source_jobs'), \
             mock.patch.object(pipeline.subprocess, 'Popen', side_effect=spawn), mock.patch('torch.load', return_value=data):
            result = pipeline.run(self.args)
        return result, commands, events

    def test_full_pipeline_is_sequential_and_has_finite_budget(self):
        result, commands, events = self.exercise()
        self.assertEqual(result['maximum_training_samples'], 131072)
        training = [c for c in commands if Path(c[2]).name == 'train_gpu.py']
        self.assertEqual(len(training), 2)
        self.assertLess(events.index(('wait', 'snake_ppo')), events.index(('start', 'mixed_bc')))
        for command in training:
            self.assertIn('--eval-initial', command)
            self.assertEqual(command[command.index('--initial-std')+1], '.65')
            self.assertEqual(command[command.index('--rollouts')+1], '128')
            self.assertEqual(command[command.index('--envs')+1], '8')
        self.assertEqual(training[0][training[0].index('--servo-neutral-deg')+1], '45.0')
        self.assertTrue(all('pid' in j and 'exit_code' in j for j in result['jobs']))
        self.assertTrue((self.out/'pipeline.lock').exists())
        self.assertEqual(json.loads((self.out/'pipeline.json').read_text())['status'], 'finished_review_required')

    def test_failed_probe_never_starts_bc_or_ppo(self):
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            self.exercise(failed={'probe'})
        saved = json.loads((self.out/'pipeline.json').read_text())
        self.assertEqual(len(saved['jobs']), 1)
        self.assertEqual(saved['status'], 'failed')

    def test_failed_bc_skips_ppo_and_continues_other_mode(self):
        result, commands, _ = self.exercise(failed={'snake_bc'})
        training = [c for c in commands if Path(c[2]).name == 'train_gpu.py']
        self.assertEqual(len(training), 1)
        self.assertEqual(training[0][training[0].index('--mode')+1], 'mixed')
        self.assertEqual(result['modes']['snake']['status'], 'needs_actor_diagnosis')

    def test_no_teacher_does_not_force_training(self):
        data, report = fixture()
        for case in data['cases'].values(): case['summary']['gate_passed'] = False
        result, commands, _ = self.exercise(data, report)
        self.assertEqual(len(commands), 1)
        self.assertEqual(result['status'], 'needs_diagnosis')
        self.assertTrue(all(x['status'] == 'needs_physics_diagnosis' for x in result['modes'].values()))

    def test_one_mode_without_teacher_does_not_block_the_other(self):
        data, report = fixture()
        data['cases']['snake_good']['summary']['gate_passed'] = False
        result, commands, _ = self.exercise(data, report)
        self.assertEqual(result['modes']['snake']['status'], 'needs_physics_diagnosis')
        self.assertEqual(result['modes']['mixed']['status'], 'pilot_complete')


if __name__ == '__main__':
    unittest.main(verbosity=2)

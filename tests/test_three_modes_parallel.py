"""Contracts for three-mode (snake/worm/mixed) control, asymmetric critic, obs normalizer, and parallel startup."""
import math
import unittest
import torch

from serpworm.envs.tensor_env import ACTION_SIZES, PRIVILEGED_SIZE, TensorEnv
from serpworm.probes.fixed_waves import Case, cases, summarize_trace
from serpworm.rl.evaluate import aggregate, summarize
from serpworm.rl.train_ppo import ActorCritic, RunningMeanStd, parse_args
from serpworm import pipeline


class ThreeModeAndParallelTests(unittest.TestCase):
    def test_running_mean_std_tracks_statistics_and_roundtrips(self):
        norm = RunningMeanStd(4, device='cpu')
        rng = torch.Generator().manual_seed(42)
        a = torch.randn(500, 4, generator=rng) * 3.0 + torch.tensor([1.0, -2.0, 5.0, 0.5])
        b = torch.randn(500, 4, generator=rng) * 3.0 + torch.tensor([1.0, -2.0, 5.0, 0.5])
        norm.update(a)
        norm.update(b)
        combined = torch.cat((a, b), 0).double()
        torch.testing.assert_close(norm.mean, combined.mean(0), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(norm.var, combined.var(0, unbiased=False), atol=1e-2, rtol=1e-2)
        restored = RunningMeanStd(4, device='cpu')
        restored.load_state_dict(norm.state_dict())
        out1 = norm.normalize(a[:5])
        out2 = restored.normalize(a[:5])
        torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)

    def test_asymmetric_actor_critic_all_three_modes(self):
        torch.manual_seed(7)
        obs = torch.randn(8, 570)
        cobs = torch.randn(8, 570 + PRIVILEGED_SIZE)
        for mode, act_dim in ACTION_SIZES.items():
            model = ActorCritic(570, act_dim, critic_obs_size=570 + PRIVILEGED_SIZE)
            with torch.no_grad():
                act, latent, oldlog, val = model.sample(obs, cobs)
            self.assertEqual(tuple(act.shape), (8, act_dim))
            self.assertEqual(tuple(val.shape), (8,))
            logp, ent, val2 = model.evaluate_latent(obs, latent, cobs)
            torch.testing.assert_close((logp - oldlog).exp(), torch.ones_like(logp), atol=2e-6, rtol=0.0)
            torch.testing.assert_close(val2, val, atol=1e-6, rtol=0.0)
            # Actor output must remain independent of privileged critic state.
            torch.testing.assert_close(model.action(obs), model.actor(obs).tanh())
            loss = -logp.mean() + 0.5 * val2.square().mean() - 0.002 * ent.mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_worm_evaluation_accepts_pure_peristalsis_without_yaw(self):
        from tests.test_evaluation import synthetic_traces
        traces = synthetic_traces()
        # Zero out all yaw joints (pure worm has no lateral undulation) while keeping rope motion.
        traces['diagnostics_50hz'][:, :, :3, :] = 0.0
        results = summarize(traces, 20., 'worm')
        totals = aggregate(results)
        self.assertEqual(totals['moving_sustained_case_count'], 6)
        self.assertEqual(totals['mixed_rope_sustained_case_count'], 6)
        # Disabling 4 of 5 segments must fail the >=2 active rope segments screen.
        traces['servo_diagnostics_policy_hz'][:, :, :, 2:] = math.pi / 2
        results_one = summarize(traces, 20., 'worm')
        self.assertFalse(results_one[2]['sustained_motion_screen'])

    def test_worm_fixed_wave_cases_and_gate(self):
        from tests.test_wave_probe import synthetic
        all_cases = cases()
        worm_cases = [c for c in all_cases if c.kind == 'worm']
        self.assertGreaterEqual(len(worm_cases), 6)
        self.assertTrue(all(c.yaw_amplitude_deg == 0.0 and c.rope_amplitude_deg > 0.0 for c in worm_cases))
        trace = synthetic(mode='mixed')
        trace['yaw_diagnostics'][:, :, :3, :] = 0.0
        report = summarize_trace(trace, 0, Case('worm_test', 'worm', 90., 0., 60., 8., 72.), 40.)
        self.assertTrue(report['sustained_worm_forward_gate'])
        self.assertFalse(report['sustained_yaw_forward_gate'])
        self.assertEqual(report['replay_cost_mode'], 'worm_active5')

    def test_pipeline_candidates_supports_all_three_modes(self):
        from tests.test_pipeline import fixture
        data, report = fixture()
        data['cases']['worm_good'] = dict(mode='worm', servo_neutral_deg=90.,
            summary=dict(gate_passed=True, reward_replay_better_than_stationary=True, complete=True, case_kind='worm'))
        report['results'].append(dict(case=dict(name='worm_good'), first_episode_complete=True,
            tail=dict(complete=True, summary=dict(mean_vx_mm_s=9.5))))
        chosen = pipeline.candidates(data, report)
        self.assertEqual(set(chosen.keys()), {'snake', 'worm', 'mixed'})
        self.assertEqual(chosen['worm']['case'], 'worm_good')

    def test_parallel_from_scratch_cli_args(self):
        args = parse_args([
            '--out', 'unused_out', '--mode', 'worm', '--envs', '2048', '--horizon', '16',
            '--batch-size', '4096', '--dt', '0.0005', '--plate-solver', 'segment_lu',
            '--no-strain-jacobian', '--from-scratch', '--asymmetric-critic', '--normalize-obs',
            '--target-kl', '0.01', '--early-stop-stale', '0',
        ])
        self.assertEqual(args.mode, 'worm')
        self.assertEqual(args.envs, 2048)
        self.assertEqual(args.plate_solver, 'segment_lu')
        self.assertFalse(args.strain_jacobian)
        self.assertTrue(args.from_scratch and args.asymmetric_critic and args.normalize_obs)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable on this host')
    def test_tensor_env_worm_mode_and_privileged_obs_on_cuda(self):
        env = TensorEnv(n=4, mode='worm', device='cuda', dt=.0005, noise=False,
                        plate_solver='segment_lu', strain_jacobian=False)
        self.assertEqual(env.action_size, 5)
        self.assertEqual(env.critic_obs_size, 570 + PRIVILEGED_SIZE)
        self.assertEqual(tuple(env.critic_obs.shape), (4, 570 + PRIVILEGED_SIZE))
        act = torch.full((4, 5), 0.3, device='cuda')
        obs, rew, term, trunc, term_obs, metrics = env.step(act)
        self.assertTrue(torch.isfinite(obs).all() and torch.isfinite(env.critic_obs).all())
        self.assertTrue(torch.isfinite(env.last_terminal_critic_obs).all())
        # Pure worm must keep yaw joint targets identically zero.
        self.assertTrue(torch.equal(env.targets, torch.zeros_like(env.targets)))
        del env
        torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main(verbosity=2)

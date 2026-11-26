"""Numerical PPO tests on CPU or CUDA, without constructing the physical env."""
import argparse
import math
import unittest
import torch
from serpworm.rl.train_ppo import (ActorCritic, advantages, temporal_config, parse_args,
                                   evaluation_improved, DEFAULT_GAMMA, DEFAULT_LAMBDA)


def run(device='cpu'):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    checks = []
    for action_size in (4, 9):
        model = ActorCritic(570, action_size).to(device)
        obs = torch.randn(12, 570, device=device)
        with torch.no_grad():
            action, latent, oldlog, value = model.sample(obs)
        assert action.shape == latent.shape == (12, action_size)
        assert oldlog.shape == value.shape == (12,)
        assert action.abs().max() <= 1
        assert model.action(obs).shape == action.shape
        current, entropy, prediction = model.evaluate_latent(obs, latent)
        torch.testing.assert_close((current-oldlog).exp(), torch.ones_like(current), atol=2e-6, rtol=0.)
        extreme = torch.tensor([-20., -8., -2., 0., 2., 8., 20.], device=device)[:, None].expand(-1, action_size)
        extreme_obs = obs[:7]
        with torch.no_grad():
            old_extreme = model.log_probability(model.distribution(extreme_obs), extreme)
        new_extreme, extreme_entropy, _ = model.evaluate_latent(extreme_obs, extreme)
        assert torch.isfinite(new_extreme).all()
        torch.testing.assert_close((new_extreme-old_extreme).exp(), torch.ones_like(new_extreme), atol=2e-6, rtol=0.)
        loss = -.01*new_extreme.mean()-.002*extreme_entropy.mean()+.5*prediction.square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append(f'action_{action_size}: finite gradients, bounded mapping, ratios incl latent +/-20')
    reward = torch.tensor([[1., 1., 1.], [3., 3., 3.]], device=device)
    value = torch.zeros_like(reward)
    next_value = torch.full_like(reward, 2.)
    term = torch.tensor([[True, False, False], [False, False, False]], device=device)
    trunc = torch.tensor([[False, True, False], [False, False, False]], device=device)
    adv, ret = advantages(reward, value, next_value, term, trunc, gamma=.9, lam=.8)
    expected = torch.tensor([[1., 2.8, 2.8+.9*.8*4.8], [4.8, 4.8, 4.8]], device=device)
    torch.testing.assert_close(adv, expected)
    torch.testing.assert_close(ret, expected)
    checks.append('GAE: termination no bootstrap; timeout bootstrap; both cut cross-reset trace')
    args = parse_args(['--out', 'unused_test_output'])
    timing = temporal_config(.2, args.horizon)
    assert args.envs == 8 and args.horizon == 64
    assert math.isclose(timing['rollout_duration_per_env_s'], 12.8)
    assert timing['policy_hz'] == 5 and timing['episode_duration_s'] == 20
    assert args.gamma == DEFAULT_GAMMA == .995**2 and args.gae_lambda == DEFAULT_LAMBDA == .97
    try:
        temporal_config(.1, 64)
    except ValueError:
        pass
    else:
        raise AssertionError('Old 10 Hz timing must be rejected')
    checks.append('Frozen temporal config: 5 Hz, 12.8 s rollout, 20 s episode')
    baseline = dict(guards=0, sustained_cases=0, tail_vx_mm_s=0., rmse_mm_s=100., stop_abs_vx_mm_s=0.)
    assert not evaluation_improved({**baseline, 'rmse_mm_s': 50.}, baseline)
    moving = {**baseline, 'sustained_cases': 2, 'tail_vx_mm_s': 10.}
    assert evaluation_improved(moving, baseline)
    assert not evaluation_improved({**moving, 'stop_abs_vx_mm_s': 2.}, baseline)
    assert not evaluation_improved({**moving, 'sustained_cases': 1, 'tail_vx_mm_s': 100.}, moving)
    checks.append('Evaluation ranking rejects startup-only RMSE gain and worsened stopping')
    return {'passed': True, 'device': device, 'checks': checks}


class PPOContractTests(unittest.TestCase):
    def test_ppo_contracts_cpu(self):
        self.assertTrue(run('cpu')['passed'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable on this host')
    def test_ppo_contracts_cuda(self):
        self.assertTrue(run('cuda')['passed'])


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    print(json.dumps(run(parser.parse_args().device), indent=2))

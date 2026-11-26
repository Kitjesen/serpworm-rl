"""Numerical parity, time-limit GAE, masked reset and CUDA training smoke checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from matsuoka_torch import Matsuoka
from train_gpu import advantages, ActorCritic
from tensor_env import TensorEnv, STATE

ROOT = Path(__file__).resolve().parent


def checks(device):
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('numpy_cpg', ROOT/'reference/pilot_cpu_interface/matsuoka.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    report = {'device': device, 'checks': {}}
    for c in (0., .75):
        a, b = ref.Matsuoka(2, c=c), Matsuoka(2, device=device, c=c)
        error = 0.
        for k in range(500):
            alpha = np.stack([np.full(4, .6), .5*np.sin(k*.02+np.arange(4))])
            x = a.step(alpha)
            y = b.step(torch.tensor(alpha, device=device)).cpu().numpy()
            error = max(error, float(np.max(np.abs(x-y))))
        assert error < 1e-10, error
        report['checks'][f'cpg_c{c}_10s_max_abs_error'] = error
    # A timeout bootstraps final V=10, a termination does not. No episode mixing.
    reward = torch.tensor([[1., 1.], [2., 2.]], device=device)
    value = torch.zeros_like(reward)
    next_value = torch.full_like(value, 10.)
    term = torch.tensor([[False, True], [True, False]], device=device)
    trunc = torch.tensor([[True, False], [False, True]], device=device)
    adv, ret = advantages(reward, value, next_value, term, trunc, gamma=.9)
    assert torch.allclose(adv, torch.tensor([[10., 1.], [2., 11.]], device=device))
    report['checks']['timeout_bootstrap_and_no_cross_episode_gae'] = True
    if device == 'cuda':
        for kind in ('direct', 'liu', 'liu_foc'):
            env = TensorEnv(kind, n=2, noise=False)
            for k in range(4):
                actions = torch.full((2, 4), .2 if k%2 else -.2, device=device)
                obs, reward, term, trunc, final, metrics = env.step(actions)
                assert obs.is_cuda and reward.is_cuda and metrics.is_cuda
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                assert not env.c.failed and not term.any(), metrics
            saved = {name: getattr(env.c, name)[1].clone() for name in STATE}
            sensor = {name: getattr(env, name)[1].clone() for name in ('history','previous_angles','dq','sensor','delayed','visual_previous','visual_filtered','prev_theta','targets','previous_actions','cmd','goals')}
            env.reset(torch.tensor([True, False], device=device))
            assert all(torch.equal(old, getattr(env.c, name)[1]) for name, old in saved.items())
            assert all(torch.equal(old, getattr(env, name)[1]) for name, old in sensor.items())
            # A real environment observation feeds actor, critic, loss, gradients and Adam on GPU.
            model = ActorCritic(env.obs_size).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            before = model.actor[0].weight.detach().clone()
            action, logp, entropy, value = model(obs)
            loss = -logp.mean()+value.square().mean()-.002*entropy.mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            assert not torch.equal(before, model.actor[0].weight)
            assert all(p.is_cuda and torch.isfinite(p).all() for p in model.parameters())
            report['checks'][kind] = {'simulated_s': .4, 'gpu_observation_reward_policy_update': True,
                                     'masked_reset_preserves_other_environment': True,
                                     'mean_slip_m_s': float(metrics[:, 0].mean())}
            del env, model, optimizer
            torch.cuda.empty_cache()
    report['scope'] = 'Port checks only; not calibrated mechanics, long-horizon stability or policy convergence.'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--out', default='results/migration_checks.json')
    args = parser.parse_args()
    start = time.monotonic()
    report = checks(args.device)
    report['wall_s'] = time.monotonic()-start
    report['passed'] = True
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))

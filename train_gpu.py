"""Native PyTorch PPO: CUDA environment, actor/critic, rollout, GAE and updates.

Host synchronization only at rollout/report boundaries. No SB3/NumPy bridge.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from tensor_env import TensorEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        def net(output):
            return nn.Sequential(nn.Linear(obs_size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, output))
        self.actor, self.critic = net(4), net(1)
        self.logstd = nn.Parameter(torch.full((4,), -.8))
        for network in (self.actor, self.critic):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, 2**.5)
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.)

    def forward(self, obs, action=None):
        mean = self.actor(obs)
        dist = torch.distributions.Normal(mean, self.logstd.exp().expand_as(mean), validate_args=False)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(obs).squeeze(-1)


def advantages(reward, value, next_value, terminated, truncated, gamma=.995, lam=.95):
    """Bootstrap time limits; do not carry GAE across either kind of reset."""
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta = reward[t]+gamma*(~terminated[t])*next_value[t]-value[t]
        running = delta+gamma*lam*(~(terminated[t] | truncated[t]))*running
        advantage[t] = running
    return advantage, advantage+value


def write(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'status.json', {'status': 'initializing', 'kind': args.kind})
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        env = TensorEnv(args.kind, args.envs, args.seed)
        model = ActorCritic(env.obs_size).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
        write(out/'config.json', {**vars(args), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
              'online_numeric_device': 'cuda', 'physics_dtype': 'float64', 'policy_dtype': 'float32',
              'cpu_scope': 'initialization, Python dispatch, boundary checks, serialization',
              'physics': 'uncalibrated candidate; unchanged split coupling; no full self-contact',
              'algorithm': 'PPO fixed Kf=1; no option critic', 'dt_s': .00025, 'policy_hz': 10,
              'rollout_equivalence': 'same task/reward as prior pilot; torch RNG differs; no identical training trajectory claim'})
        shape = (args.horizon, args.envs)
        def buffer(*tail, dtype=torch.float32):
            return torch.empty(*shape, *tail, device='cuda', dtype=dtype)
        obsbuf, actionbuf = buffer(env.obs_size), buffer(4)
        logbuf, rewbuf, valbuf, nextbuf = [buffer() for _ in range(4)]
        termbuf, truncbuf = buffer(dtype=torch.bool), buffer(dtype=torch.bool)
        obs = env.obs
        updates, samples, completed, successes, failures = 0, 0, 0, 0, 0
        write(out/'status.json', {'status': 'collecting_first_rollout', 'kind': args.kind, 'samples': 0})
        for rollout in range(1, args.rollouts+1):
            t0 = time.monotonic()
            metric_sum = torch.zeros(8, device='cuda', dtype=torch.float64)
            done_counts = torch.zeros(3, device='cuda', dtype=torch.long)
            with torch.no_grad():
                for t in range(args.horizon):
                    action, logp, _, value = model(obs)
                    obsbuf[t], actionbuf[t], logbuf[t], valbuf[t] = obs, action, logp, value
                    obs, reward, terminated, truncated, terminal_obs, metrics = env.step(action)
                    rewbuf[t], termbuf[t], truncbuf[t] = reward, terminated, truncated
                    nextbuf[t] = model.critic(terminal_obs).squeeze(-1)
                    metric_sum += metrics.sum(0)
                    done_counts += torch.stack(((terminated | truncated).sum(), (metrics[:, 6] > 0).sum(), (metrics[:, 5] > 0).sum()))
            # Block invalid rollouts before any learning. Synchronization is deliberate here.
            if bool(env.c.failed) or not bool(torch.isfinite(obsbuf).all() & torch.isfinite(rewbuf).all() & torch.isfinite(obs).all() & torch.isfinite(nextbuf).all()):
                raise RuntimeError('Invalid physics/observations; rollout not used for learning')
            torch.cuda.synchronize()
            rollout_s = time.monotonic()-t0
            adv, returns = advantages(rewbuf, valbuf, nextbuf, termbuf, truncbuf)
            flatobs, flataction = obsbuf.flatten(0, 1), actionbuf.flatten(0, 1)
            oldlog, oldvalue = logbuf.flatten(), valbuf.flatten()
            adv, returns = adv.flatten(), returns.flatten()
            adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            total = args.horizon*args.envs
            t1 = time.monotonic()
            for _ in range(args.epochs):
                order = torch.randperm(total, device='cuda')
                for index in order.split(args.batch_size):
                    _, logp, entropy, value = model(flatobs[index], flataction[index])
                    ratio = (logp-oldlog[index]).exp()
                    policy_loss = torch.maximum(-adv[index]*ratio, -adv[index]*ratio.clamp(.8, 1.2)).mean()
                    loss = policy_loss+.5*(value-returns[index]).square().mean()-.002*entropy.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), .5)
                    optimizer.step()
                    updates += 1
            torch.cuda.synchronize()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise RuntimeError('Nonfinite policy parameters')
            optimize_s = time.monotonic()-t1
            samples += total
            counts = done_counts.cpu().tolist()
            completed += counts[0]; successes += counts[1]; failures += counts[2]
            metrics = (metric_sum/total).cpu().tolist()
            report = dict(status='training', kind=args.kind, rollouts=rollout, samples=samples,
                          optimizer_steps=updates, epochs=rollout*args.epochs, rollout_wall_s=rollout_s,
                          samples_per_wall_s=total/rollout_s, optimizer_wall_s=optimize_s,
                          mean_reward=float(rewbuf.mean()), completed_episodes=completed,
                          successes=successes, physical_guard_failures=failures,
                          mean_slip_m_s=metrics[0], mean_goal_distance_m=metrics[4], elapsed_s=time.monotonic()-start)
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'obs_size': env.obs_size,
                          'config': vars(args), 'report': report, 'scope': 'pilot, not convergence or hardware validation'}
            torch.save(checkpoint, out/'checkpoint.tmp.pt')
            (out/'checkpoint.tmp.pt').replace(out/'checkpoint.pt')
            write(out/'status.json', report)
            with (out/'learning.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(report)+'\n')
            print(json.dumps(report), flush=True)
            if time.monotonic()-start >= args.seconds:
                break
        write(out/'status.json', {**report, 'status': 'pilot_complete', 'converged': False})
    except BaseException as e:
        write(out/'status.json', dict(status='failed', error=repr(e), elapsed_s=time.monotonic()-start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['direct', 'liu', 'liu_foc'], default='direct')
    parser.add_argument('--out', required=True)
    parser.add_argument('--envs', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--seconds', type=float, default=1800.)
    args = parser.parse_args()
    if min(args.envs, args.horizon, args.rollouts, args.epochs, args.batch_size) < 1 or args.seconds <= 0:
        parser.error('Counts and duration must be positive')
    train(args)

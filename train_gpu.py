"""Native PyTorch PPO: CUDA environment, actor/critic, rollout, GAE and updates.

Host synchronization only at rollout/report boundaries. No SB3/NumPy bridge.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
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


def restore_training(path, model, optimizer, args, obs_size):
    """Continue weights and Adam; physical episodes restart, not bitwise replay."""
    data = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    for key in ('kind', 'reward', 'control_profile', 'envs', 'horizon', 'seed', 'plate_solver'):
        default = 'legacy' if key in ('reward', 'control_profile', 'plate_solver') else None
        if data['config'].get(key, default) != getattr(args, key):
            raise ValueError(f'Resume configuration mismatch: {key}')
    report = data['report']
    if data['obs_size'] != obs_size or args.rollouts <= report['rollouts']:
        raise ValueError('Resume needs identical observations and a larger total rollout budget')
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    return report


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'status.json', {'status': 'initializing', 'kind': args.kind})
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        env = TensorEnv(args.kind, args.envs, args.seed, reward_version=args.reward, plate_solver=args.plate_solver, control_profile=args.control_profile)
        model = ActorCritic(env.obs_size).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
        previous = restore_training(args.resume, model, optimizer, args, env.obs_size) if args.resume else {}
        first_rollout = previous.get('rollouts', 0)
        if args.resume:
            torch.manual_seed(args.seed+first_rollout)
            env.rng.manual_seed(args.seed+first_rollout)
            env.obs = env.reset(torch.ones_like(env.no_reset))
        write(out/'config.json', {**vars(args), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'resume_sha256': hashlib.sha256(Path(args.resume).read_bytes()).hexdigest() if args.resume else None,
              'resume_scope': 'weights and Adam restored; physical episodes and rollout buffers restarted',
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
              'online_numeric_device': 'cuda', 'physics_dtype': 'float64', 'policy_dtype': 'float32',
              'cpu_scope': 'initialization, Python dispatch, boundary checks, serialization',
              'physics': 'uncalibrated candidate; unchanged split coupling; no full self-contact',
              'algorithm': 'direct PPO without CPG' if args.kind == 'direct' else 'PPO fixed Kf=1; no option critic', 'dt_s': .00025, 'policy_hz': 10,
              'reward_scope': 'reward selection applies only to direct velocity task; CPG goal reward unchanged',
              'control_profile': args.control_profile, 'joint_limit_deg': env.limit*180/3.141592653589793,
              'observation_size': env.obs_size, 'command_switch_steps': env.command_switch_steps,
              'reward_velocity': 'plate centroid and whole-body heading' if args.reward == 'tracking_v3' else 'middle plate',
              'rollout_equivalence': 'Compare explicit profiles; smooth_v3 changes actuator filter, angle range, action history and command switching.'})
        (out/'sources').mkdir()
        for source in Path(__file__).parent.glob('*.py'):
            shutil.copy2(source, out/'sources'/source.name)
        shape = (args.horizon, args.envs)
        def buffer(*tail, dtype=torch.float32):
            return torch.empty(*shape, *tail, device='cuda', dtype=dtype)
        obsbuf, actionbuf = buffer(env.obs_size), buffer(4)
        logbuf, rewbuf, valbuf, nextbuf = [buffer() for _ in range(4)]
        termbuf, truncbuf = buffer(dtype=torch.bool), buffer(dtype=torch.bool)
        obs = env.obs
        updates, samples, completed, successes, failures = [previous.get(key, 0) for key in
            ('optimizer_steps', 'samples', 'completed_episodes', 'successes', 'physical_guard_failures')]
        write(out/'status.json', {'status': 'collecting_first_rollout', 'kind': args.kind, 'samples': samples, 'rollouts': first_rollout})
        for rollout in range(first_rollout+1, args.rollouts+1):
            t0 = time.monotonic()
            metric_sum = torch.zeros(8, device='cuda', dtype=torch.float64)
            done_counts = torch.zeros(3, device='cuda', dtype=torch.long)
            error_sum = torch.zeros(3, device='cuda', dtype=torch.float64)
            action_stats = torch.zeros(4, device='cuda', dtype=torch.float64)
            with torch.no_grad():
                for t in range(args.horizon):
                    action, logp, _, value = model(obs)
                    command = env.cmd.clone()
                    obsbuf[t], actionbuf[t], logbuf[t], valbuf[t] = obs, action, logp, value
                    obs, reward, terminated, truncated, terminal_obs, metrics = env.step(action)
                    rewbuf[t], termbuf[t], truncbuf[t] = reward, terminated, truncated
                    nextbuf[t] = model.critic(terminal_obs).squeeze(-1)
                    metric_sum += metrics.sum(0)
                    error_sum += (env.last_velocity-command).square().sum(0)
                    action_stats += torch.stack((env.last_action_rate_cost.sum(), env.last_action_acceleration_cost.sum(),
                        (action.abs()>1).double().mean(-1).sum(), env.last_diagnostics[:, :, 4].abs().amax((0, 2)).sum()))
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
                          reward_version=args.reward, requested_rollouts=args.rollouts,
                          optimizer_steps=updates, epochs=rollout*args.epochs, rollout_wall_s=rollout_s,
                          samples_per_wall_s=total/rollout_s, optimizer_wall_s=optimize_s,
                          mean_reward=float(rewbuf.mean()), completed_episodes=completed,
                          successes=successes, physical_guard_failures=failures,
                          mean_slip_m_s=metrics[0], mean_goal_distance_m=metrics[4], elapsed_s=time.monotonic()-start)
            ar, aa, clipped, torque = (action_stats/total).cpu().tolist()
            report.update(control_profile=args.control_profile, action_change_cost=ar, action_second_difference_cost=aa,
                          raw_action_clipping_fraction=clipped, mean_sampled_peak_torque_nm=torque,
                          policy_std=model.logstd.detach().exp().cpu().tolist())
            if args.kind == 'direct':
                report.pop('mean_goal_distance_m')
                report['training_velocity_rmse'] = (error_sum/total).sqrt().cpu().tolist()
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'obs_size': env.obs_size,
                          'config': vars(args), 'report': report, 'scope': 'pilot, not convergence or hardware validation'}
            torch.save(checkpoint, out/'checkpoint.tmp.pt')
            (out/'checkpoint.tmp.pt').replace(out/'checkpoint.pt')
            if args.save_every and (rollout % args.save_every == 0 or rollout == args.rollouts):
                shutil.copy2(out/'checkpoint.pt', out/f'checkpoint_r{rollout:04d}.pt')
            write(out/'status.json', report)
            with (out/'learning.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(report)+'\n')
            print(json.dumps(report), flush=True)
            if args.eval_every and rollout % args.eval_every == 0:
                evaluation_dir = out/f'evaluation_r{rollout:04d}'
                write(out/'status.json', {**report, 'status': 'evaluating_checkpoint', 'evaluation_dir': str(evaluation_dir)})
                with (out/f'evaluation_r{rollout:04d}.log').open('w') as log:
                    subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('evaluate_gpu.py')),
                                    str(out/'checkpoint.pt'), '--out', str(evaluation_dir)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            if args.seconds and time.monotonic()-start >= args.seconds:
                break
        write(out/'status.json', {**report, 'status': 'pilot_complete', 'converged': False,
                                 'stop_reason': 'rollout_limit' if rollout == args.rollouts else 'wall_time_limit'})
    except BaseException as e:
        write(out/'status.json', dict(status='failed', error=repr(e), elapsed_s=time.monotonic()-start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['direct', 'liu', 'liu_foc'], default='direct')
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', help='Trusted checkpoint; --rollouts is the cumulative target, episodes restart')
    parser.add_argument('--envs', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--rollouts', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--seconds', type=float, default=0., help='Wall limit; zero runs the requested finite rollout budget')
    parser.add_argument('--reward', choices=['legacy', 'tracking_v2', 'tracking_v3'], default='legacy')
    parser.add_argument('--control-profile', choices=['legacy', 'smooth_v3'], default='legacy')
    parser.add_argument('--plate-solver', choices=['legacy', 'segment_lu'], default='legacy', help='Linear plate solve only; segment_lu assumes the existing split coupling')
    parser.add_argument('--save-every', type=int, default=32, help='Retain a milestone checkpoint; zero disables extra copies')
    parser.add_argument('--eval-every', type=int, default=0, help='Run held-out deterministic checks at this interval; zero disables')
    args = parser.parse_args()
    if min(args.envs, args.horizon, args.rollouts, args.epochs, args.batch_size) < 1 or args.seconds < 0 or min(args.save_every, args.eval_every) < 0:
        parser.error('Counts must be positive; time limit and save interval nonnegative')
    train(args)

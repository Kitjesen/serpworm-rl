"""Finite direct PPO experiment with exact stored-latent bounded-action ratios."""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F

POLICY_DT = .2
DEFAULT_GAMMA = .995**2
DEFAULT_LAMBDA = .97


class ActorCritic(nn.Module):
    def __init__(self, obs_size, action_size=9, initial_std=.65):
        super().__init__()
        def net(output):
            return nn.Sequential(nn.Linear(obs_size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, output))
        self.actor, self.critic = net(action_size), net(1)
        if not .05 <= initial_std <= 1.5:
            raise ValueError('initial_std must be within [.05,1.5]')
        self.logstd = nn.Parameter(torch.full((action_size,), math.log(initial_std)))
        for network in (self.actor, self.critic):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, 2**.5)
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.)

    def distribution(self, obs):
        mean = self.actor(obs)
        std = self.logstd.clamp(-3., .5).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std, validate_args=False)

    @staticmethod
    def log_probability(dist, latent):
        # log(1-tanh(z)^2), stable even when float32 tanh(z) rounds to +/-1.
        log_jacobian = 2*(math.log(2.)-latent-F.softplus(-2*latent))
        return (dist.log_prob(latent)-log_jacobian).sum(-1)

    def sample(self, obs):
        dist = self.distribution(obs)
        latent = dist.rsample()
        return latent.tanh(), latent, self.log_probability(dist, latent), self.critic(obs).squeeze(-1)

    def evaluate_latent(self, obs, latent):
        dist = self.distribution(obs)
        logp = self.log_probability(dist, latent)
        # Fresh reparameterized sample: transformed-action differential entropy.
        entropy_mc = -self.log_probability(dist, dist.rsample())
        return logp, entropy_mc, self.critic(obs).squeeze(-1)

    def action(self, obs):
        return self.actor(obs).tanh()


def advantages(reward, value, next_value, terminated, truncated, gamma=DEFAULT_GAMMA, lam=DEFAULT_LAMBDA):
    """Bootstrap time limits; neither kind of reset carries future GAE."""
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


def temporal_config(policy_dt, horizon):
    if not math.isclose(float(policy_dt), POLICY_DT, rel_tol=0., abs_tol=1e-12):
        raise ValueError('sustain_v4 requires a 0.2 s policy step')
    return dict(policy_dt_s=POLICY_DT, policy_hz=1/POLICY_DT, sensor_dt_s=.02,
                episode_duration_s=20., rollout_duration_per_env_s=horizon*POLICY_DT)


def gradient_norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(values).sum().sqrt()


def evaluation_score(aggregate):
    keys = ('moving_tail_mean_vx_mm_s', 'moving_vx_rmse_mm_s', 'all_stop_mean_abs_vx_mm_s')
    if any(aggregate.get(k) is None or not math.isfinite(float(aggregate[k])) for k in keys):
        return None
    return dict(guards=int(aggregate['moving_guard_failures']+aggregate['stop_guard_failures']),
                sustained_cases=int(aggregate['moving_sustained_case_count']),
                tail_vx_mm_s=float(aggregate[keys[0]]), rmse_mm_s=float(aggregate[keys[1]]),
                stop_abs_vx_mm_s=float(aggregate[keys[2]]))


def evaluation_improved(candidate, best):
    """Scheduling diagnostic; no reward sum or startup displacement as success."""
    if candidate is None:
        return False
    if best is None:
        return True  # Establish the baseline, including a stationary one.
    if candidate['guards'] != best['guards']:
        return candidate['guards'] < best['guards']
    if candidate['stop_abs_vx_mm_s'] > best['stop_abs_vx_mm_s']+1.:
        return False
    if candidate['sustained_cases'] != best['sustained_cases']:
        return candidate['sustained_cases'] > best['sustained_cases']
    if candidate['tail_vx_mm_s'] > best['tail_vx_mm_s']+.25:
        return candidate['rmse_mm_s'] <= best['rmse_mm_s']+1.
    # RMSE alone cannot promote a static policy with a new startup transient.
    return (candidate['sustained_cases'] > 0 and candidate['tail_vx_mm_s'] >= best['tail_vx_mm_s']-.25
            and candidate['rmse_mm_s'] < best['rmse_mm_s']-.25)


def load_bc_actor(model, path, args, source_root):
    """Transfer only a project BC actor after its CUDA teacher passed the gate."""
    path, source_root = Path(path), Path(source_root)
    saved = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    proof = saved.get('proof', {})
    gate = proof.get('wave_gate', {})
    if saved.get('schema') != 'bc_actor_v1' or saved.get('report', {}).get('heldout_improved') is not True:
        raise ValueError('Use a passed project behavior-cloning checkpoint, not a PPO/unknown pickle')
    if proof.get('dataset_device') != 'cuda' or proof.get('gpu_warmstart_eligible') is not True:
        raise ValueError('CPU oracle BC cannot authorize GPU PPO; first validate a complete CUDA wave')
    if any(gate.get(k) is not True for k in ('gate_passed', 'reward_replay_better_than_stationary', 'complete')):
        raise ValueError('Missing passed complete-wave / matched-rest reward proof')
    if proof.get('mode') != args.mode or proof.get('obs_size') != 570 or proof.get('action_size') != (4 if args.mode == 'snake' else 9):
        raise ValueError('BC mode or observation/action layout mismatch')
    if proof.get('policy_dt_s') != .2 or proof.get('history_span_s') != 1.8 or proof.get('physics_dt_s') != .00025:
        raise ValueError('BC observation history / policy timing mismatch')
    if not math.isclose(float(proof.get('servo_neutral_deg', -1)), args.servo_neutral_deg):
        raise ValueError('Teacher tendon-neutral configuration differs from the training environment')
    if proof.get('command_scale') != [max(.04, args.vx_max_mm_s*.001), .01, .2]:
        raise ValueError('BC command normalization differs; no implicit weight migration is allowed')
    for name in ('gpu_core.py', 'wheel_stiction.py', 'tensor_env.py', 'rewards.py', 'wheel_envelope.py'):
        actual = hashlib.sha256((source_root/name).read_bytes()).hexdigest()
        if proof.get('source_sha256', {}).get(name) != actual:
            raise ValueError('BC teacher physical/control source mismatch: '+name)
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source_root/'inputs').glob('*') if p.is_file()}
    if proof.get('input_sha256') != inputs:
        raise ValueError('BC teacher physical input manifest mismatch')
    model.actor.load_state_dict(saved['actor'], strict=True)
    with torch.no_grad():
        model.logstd.fill_(math.log(args.initial_std))
    return dict(checkpoint=str(path.resolve()), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        dataset_sha256=saved['dataset_sha256'], teacher_case=proof['case'],
        teacher_command_vx_mm_s=proof['teacher_command_vx_mm_s'], proof=proof,
        scope='Passed fixed-wave teacher -> supervised BC actor only -> PPO fine tuning; critic/Adam/episodes reset')


def require_initial_motion(summary, mode, teacher_command):
    matches = [r for r in summary['results'] if r.get('case') != 'straight100_then_stop'
               and r.get('command_vx_mm_s') == teacher_command]
    if not matches or not any(r.get('failed') is False and r.get('window_complete') is True and
            r.get('first_episode_complete') is True and r.get('deterministic_sustained_motion_screen') is True and
            (mode != 'mixed' or r.get('mixed_rope_motion_screen') is True) for r in matches):
        raise RuntimeError('BC closed-loop deterministic motion failed; no PPO samples may be collected')


def train(args):
    from tensor_env import TensorEnv
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'status.json', {'status': 'initializing', 'kind': args.kind})
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        if not args.warmstart_actor or not args.eval_initial:
            raise ValueError('wave_gate_v5 requires --warmstart-actor from a passed CUDA wave and --eval-initial')
        source_root = Path(__file__).parent
        model = ActorCritic(570, 4 if args.mode == 'snake' else 9, args.initial_std).cuda()
        warmstart = load_bc_actor(model, args.warmstart_actor, args, source_root)
        env = TensorEnv(args.kind, args.envs, args.seed, reward_version=args.reward, plate_solver=args.plate_solver,
            control_profile=args.control_profile, command_switching=args.command_switching,
            servo_neutral_deg=args.servo_neutral_deg, mode=args.mode, vx_min_mm_s=args.vx_min_mm_s,
            vx_max_mm_s=args.vx_max_mm_s, stop_fraction=args.stop_fraction)
        timing = temporal_config(env.policy_dt, args.horizon)
        if env.obs_size != 570 or env.action_size != (4 if args.mode == 'snake' else 9):
            raise ValueError('Unexpected observation or effective-action layout')
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
        configuration = {**vars(args), **timing, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.glob('*.py')},
            'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source_root/'inputs').glob('*') if p.is_file()},
            'online_numeric_device': 'cuda', 'physics_dtype': 'float64', 'policy_dtype': 'float32',
            'cpu_scope': 'initialization, Python dispatch, boundary checks, serialization',
            'physics': 'uncalibrated candidate; unchanged split coupling; no full self-contact',
            'algorithm': 'Direct PPO fine tuning of a supervised BC actor; no CPG; teacher/BC/PPO outcomes reported separately',
            'initialization': 'BC actor only; new critic/Adam and physical episodes',
            'warmstart': warmstart, 'history_span_s': 1.8,
            'dt_s': .00025, 'reward_scope': 'sustain_v4; see frozen rewards.py and tensor_env.py',
            'command_normalization': env.command_scale.cpu().tolist(),
            'joint_limit_deg': env.limit*180/math.pi, 'observation_size': env.obs_size,
            'wheel_clearance_guard': '2 mm minimum certified finite-cylinder separation; conservative sphere or separating-plane lower bound',
            'command_switch_steps': env.command_switch_steps,
            'reward_velocity': 'plate centroid and whole-body heading; actor has no ground-truth velocity',
            'rollout_equivalence': 'Stored pre-tanh latent; stable squash Jacobian; unchanged latent in PPO ratios',
            'entropy': 'Fresh rsample Monte Carlo estimate of tanh-squashed action differential entropy',
            'entropy_coefficient': .002, 'effective_logstd_bounds': [-3., .5],
            'policy_action_size': env.action_size, 'physical_action_size': 9,
            'actuator_mapping': 'snake: four independent yaw, neutral tendons; mixed: five paired tendons and four yaw',
            'servo_target_limit_deg': 180, 'yaw_target_limit_deg': 45,
            'command_range_m_s': list(env.vx_range), 'command_range_mm_s': [args.vx_min_mm_s, args.vx_max_mm_s],
            'command_sampling': 'Continuous uniform vx 5..200 mm/s plus stop mixture, independently per environment every 5 s and reset; range follows CLI',
            'yaw_command_range_rad_s': [0., 0.], 'lateral_command_m_s': 0.,
            'checkpoint_resume_scope': 'Weights, Adam and RNG retained; no physical-episode continuation interface',
            'reset_state': '0.5 s passive + 3 s physical settling at neutral; then snapshot'}
        write(out/'config.json', configuration)
        (out/'sources').mkdir()
        for source in source_root.glob('*.py'):
            shutil.copy2(source, out/'sources'/source.name)
        shutil.copytree(source_root/'inputs', out/'sources'/'inputs')
        shape = (args.horizon, args.envs)
        def buffer(*tail, dtype=torch.float32):
            return torch.empty(*shape, *tail, device='cuda', dtype=dtype)
        obsbuf, latentbuf, actionbuf, meanbuf = buffer(env.obs_size), buffer(env.action_size), buffer(env.action_size), buffer(env.action_size)
        logbuf, rewbuf, valbuf, nextbuf = [buffer() for _ in range(4)]
        termbuf, truncbuf = buffer(dtype=torch.bool), buffer(dtype=torch.bool)
        obs = env.obs
        updates = samples = completed = failures = stale_evaluations = 0
        best_evaluation = None
        stop_reason = 'rollout_limit'
        initial_checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), obs_size=env.obs_size,
            config=vars(args), report=dict(rollouts=0, samples=0, stage='BC_before_PPO'),
            source_sha256=configuration['source_sha256'], scope='BC initialization, not an RL result')
        torch.save(initial_checkpoint, out/'checkpoint_initial.pt')
        initial_dir = out/'evaluation_initial_bc'
        write(out/'status.json', {'status': 'evaluating_initial_bc', 'samples': 0, 'mode': args.mode})
        with (out/'evaluation_initial_bc.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(source_root/'evaluate_gpu.py'), str(out/'checkpoint_initial.pt'),
                '--out', str(initial_dir)], stdout=log, stderr=subprocess.STDOUT, check=True)
        initial_summary = json.loads((initial_dir/'summary.json').read_text())
        require_initial_motion(initial_summary, args.mode, warmstart['teacher_command_vx_mm_s'])
        best_evaluation = evaluation_score(initial_summary['aggregate'])
        write(out/'initial_motion_gate.json', dict(passed=True, stage='BC_before_PPO', samples=0,
            teacher_command_vx_mm_s=warmstart['teacher_command_vx_mm_s'], aggregate=initial_summary['aggregate'],
            scope='BC closed-loop gate, not PPO success or deployment validation'))
        write(out/'status.json', {'status': 'collecting_first_rollout', 'kind': args.kind, 'samples': 0})
        for rollout in range(1, args.rollouts+1):
            t0 = time.monotonic()
            metric_sum = torch.zeros(8, device='cuda', dtype=torch.float64)
            done_counts = torch.zeros(2, device='cuda', dtype=torch.long)
            error_sum = torch.zeros(3, device='cuda', dtype=torch.float64)
            action_stats = torch.zeros(4, device='cuda', dtype=torch.float64)
            command_min, command_max = torch.tensor(float('inf'), device='cuda'), torch.tensor(float('-inf'), device='cuda')
            stop_samples = torch.zeros((), device='cuda', dtype=torch.long)
            with torch.no_grad():
                for t in range(args.horizon):
                    action, latent, logp, value = model.sample(obs)
                    meanbuf[t] = model.action(obs)
                    command = env.cmd.clone()
                    command_min = torch.minimum(command_min, command[:, 0].min())
                    command_max = torch.maximum(command_max, command[:, 0].max())
                    stop_samples += (command[:, 0] == 0).sum()
                    obsbuf[t], latentbuf[t], actionbuf[t], logbuf[t], valbuf[t] = obs, latent, action, logp, value
                    obs, reward, terminated, truncated, terminal_obs, metrics = env.step(action)
                    rewbuf[t], termbuf[t], truncbuf[t] = reward, terminated, truncated
                    nextbuf[t] = model.critic(terminal_obs).squeeze(-1)
                    metric_sum += metrics.sum(0)
                    error_sum += (env.last_velocity-command).square().sum(0)
                    action_stats += torch.stack((env.last_action_rate_cost.sum(), env.last_action_acceleration_cost.sum(),
                        (action.abs()>=.999999).double().mean(-1).sum(), env.last_diagnostics[:, :, 4].abs().amax((0, 2)).sum()))
                    done_counts += torch.stack(((terminated | truncated).sum(), (metrics[:, 5] > 0).sum()))
                    if (t+1) % 16 == 0:
                        write(out/'status.json', dict(status='collecting_rollout', mode=args.mode,
                            completed_rollouts=rollout-1, collecting_rollout=rollout,
                            completed_training_samples=samples, collected_steps=t+1,
                            horizon=args.horizon, envs=args.envs,
                            elapsed_s=time.monotonic()-start, policy_dt_s=POLICY_DT))
            if bool(env.c.failed) or not all(bool(torch.isfinite(a).all()) for a in (obsbuf, latentbuf, logbuf, rewbuf, obs, nextbuf)):
                raise RuntimeError('Invalid physics/observations; rollout not used for learning')
            torch.cuda.synchronize()
            rollout_s = time.monotonic()-t0
            adv, returns = advantages(rewbuf, valbuf, nextbuf, termbuf, truncbuf, args.gamma, args.gae_lambda)
            flatobs, flatlatent = obsbuf.flatten(0, 1), latentbuf.flatten(0, 1)
            oldlog, oldvalue, returns = logbuf.flatten(), valbuf.flatten(), returns.flatten()
            adv = adv.flatten()
            adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            total = args.horizon*args.envs
            update_stats = torch.zeros(8, device='cuda')
            update_count = 0
            t1 = time.monotonic()
            for _ in range(args.epochs):
                for index in torch.randperm(total, device='cuda').split(args.batch_size):
                    logp, entropy_mc, value = model.evaluate_latent(flatobs[index], flatlatent[index])
                    logratio = logp-oldlog[index]
                    ratio = logratio.exp()
                    policy_loss = torch.maximum(-adv[index]*ratio, -adv[index]*ratio.clamp(.8, 1.2)).mean()
                    value_loss = .5*(value-returns[index]).square().mean()
                    loss = policy_loss+value_loss-.002*entropy_mc.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_norm = gradient_norm([*model.actor.parameters(), model.logstd])
                    critic_norm = gradient_norm(model.critic.parameters())
                    nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    update_stats += torch.stack((((ratio-1)-logratio).mean(), ((ratio-1).abs()>.2).float().mean(),
                        actor_norm, critic_norm, entropy_mc.mean(), policy_loss, value_loss, loss)).detach()
                    updates += 1
                    update_count += 1
            torch.cuda.synchronize()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()) or not bool(torch.isfinite(update_stats).all()):
                raise RuntimeError('Nonfinite policy/update diagnostics')
            optimize_s = time.monotonic()-t1
            samples += total
            counts = done_counts.cpu().tolist()
            completed += counts[0]
            failures += counts[1]
            metrics = (metric_sum/total).cpu().tolist()
            ar, aa, clipped, torque = (action_stats/total).cpu().tolist()
            kl, cf, ag, cg, ent, pl, vl, loss_mean = (update_stats/update_count).cpu().tolist()
            return_variance = float(returns.var(unbiased=False))
            explained = None if return_variance < 1e-12 else 1-float((returns-oldvalue).var(unbiased=False))/return_variance
            # Statistics on stochastic-rollout states, not a deterministic rollout.
            amplitude = lambda values: dict(min_per_action=values.amin((0, 1)).cpu().tolist(),
                max_per_action=values.amax((0, 1)).cpu().tolist(), mean_abs_per_action=values.abs().mean((0, 1)).cpu().tolist(),
                mean_temporal_peak_to_peak_per_action=(values.amax(0)-values.amin(0)).mean(0).cpu().tolist())
            report = dict(status='training', kind=args.kind, mode=args.mode, rollouts=rollout, samples=samples,
                reward_version=args.reward, requested_rollouts=args.rollouts, optimizer_steps=updates, epochs=rollout*args.epochs,
                rollout_wall_s=rollout_s, samples_per_wall_s=total/rollout_s, optimizer_wall_s=optimize_s,
                mean_reward=float(rewbuf.mean()), completed_episodes=completed, physical_guard_failures=failures,
                mean_slip_m_s=metrics[0], elapsed_s=time.monotonic()-start, policy_dt_s=POLICY_DT,
                control_profile=args.control_profile, action_change_cost=ar, action_second_difference_cost=aa,
                action_edge_fraction=clipped, mean_sampled_peak_torque_nm=torque,
                logstd=model.logstd.detach().cpu().tolist(), policy_std=model.logstd.detach().clamp(-3., .5).exp().cpu().tolist(),
                command_min_mm_s=float(command_min)*1000, command_max_mm_s=float(command_max)*1000,
                stop_sample_fraction=float(stop_samples)/total, policy_action_size=env.action_size,
                training_velocity_rmse=(error_sum/total).sqrt().cpu().tolist(), approx_kl=kl, clip_fraction=cf,
                explained_variance_pre_update=explained, actor_grad_norm_pre_clip=ag, critic_grad_norm_pre_clip=cg,
                squashed_action_entropy_mc=ent, policy_loss=pl, value_loss=vl, total_loss=loss_mean,
                sampled_action_stats=amplitude(actionbuf), deterministic_action_on_training_states=amplitude(meanbuf),
                action_stats_scope='Training rollout; ranges can include resets; not deterministic locomotion evaluation')
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'obs_size': env.obs_size,
                'config': vars(args), 'report': report, 'source_sha256': configuration['source_sha256'],
                'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
                'environment_rng_state': env.rng.get_state(), 'best_evaluation': best_evaluation,
                'stale_evaluations': stale_evaluations, 'scope': 'Pilot; no convergence claim. Physics episode state not serialized.'}
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
                    subprocess.run([sys.executable, '-u', str(source_root/'evaluate_gpu.py'), str(out/'checkpoint.pt'),
                                    '--out', str(evaluation_dir)], stdout=log, stderr=subprocess.STDOUT, check=True)
                aggregate = json.loads((evaluation_dir/'summary.json').read_text())['aggregate']
                candidate = evaluation_score(aggregate)
                if evaluation_improved(candidate, best_evaluation):
                    best_evaluation, stale_evaluations = candidate, 0
                else:
                    stale_evaluations += 1
                write(out/'evaluation_progress.json', dict(rollout=rollout, aggregate=aggregate,
                    best_evaluation=best_evaluation, successive_unimproved_evaluations=stale_evaluations,
                    caveat='Scheduling only; tail motion, not startup displacement. Not convergence or deployment success.'))
                if stale_evaluations >= 3:
                    stop_reason = 'diagnosis_required_after_three_unimproved_evaluations'
                    break
            if args.seconds and time.monotonic()-start >= args.seconds:
                stop_reason = 'wall_time_limit'
                break
        write(out/'status.json', {**report, 'status': 'needs_diagnosis' if stop_reason.startswith('diagnosis') else 'pilot_complete',
                                 'converged': False, 'stop_reason': stop_reason})
    except BaseException as error:
        write(out/'status.json', dict(status='failed', error=repr(error), elapsed_s=time.monotonic()-start))
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['direct'], default='direct')
    parser.add_argument('--out', required=True)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--rollouts', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--gae-lambda', type=float, default=DEFAULT_LAMBDA)
    parser.add_argument('--reward', choices=['sustain_v4'], default='sustain_v4')
    parser.add_argument('--control-profile', choices=['sustain_v4'], default='sustain_v4')
    parser.add_argument('--plate-solver', choices=['legacy'], default='legacy')
    parser.add_argument('--save-every', type=int, default=8)
    parser.add_argument('--eval-every', type=int, default=8)
    parser.add_argument('--command-switching', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--servo-neutral-deg', type=float, default=90.)
    parser.add_argument('--mode', choices=['snake', 'mixed'], default='mixed')
    parser.add_argument('--vx-min-mm-s', type=float, default=5.)
    parser.add_argument('--vx-max-mm-s', type=float, default=200.)
    parser.add_argument('--stop-fraction', type=float, default=.1)
    parser.add_argument('--warmstart-actor', help='Project BC actor from a complete passed CUDA wave')
    parser.add_argument('--initial-std', type=float, default=.65)
    parser.add_argument('--eval-initial', action='store_true', help='Required: verify 20 s deterministic BC motion before PPO sampling')
    args = parser.parse_args(argv)
    if min(args.envs, args.horizon, args.rollouts, args.epochs, args.batch_size) < 1 or args.seconds < 0 or min(args.save_every, args.eval_every) < 0:
        parser.error('Counts must be positive; time limit and save interval nonnegative')
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        parser.error('Invalid GAE discount/trace values')
    if not 0 <= args.vx_min_mm_s < args.vx_max_mm_s or not 0 <= args.stop_fraction < 1:
        parser.error('Invalid velocity range or stop fraction')
    if not .05 <= args.initial_std <= 1.5:
        parser.error('Initial standard deviation must be within [.05,1.5]')
    return args


if __name__ == '__main__':
    train(parse_args())

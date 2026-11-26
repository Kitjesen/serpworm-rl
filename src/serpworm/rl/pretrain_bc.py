"""Clone a passed fixed-wave demonstrator; this is supervised learning, not PPO."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from serpworm.paths import PHYSICS_SOURCES, PKG_ROOT, collect_input_hashes, collect_source_hashes
from serpworm.rl.train_ppo import ActorCritic, write

ROOT = PKG_ROOT


def manifest(root=ROOT, inputs_dir=None):
    return (collect_source_hashes(root, PHYSICS_SOURCES), collect_input_hashes(root, inputs_dir))


def cycle_split(times, period, dt=.2):
    """Hold out final whole cycles, with an entire intervening cycle purged."""
    if times.ndim != 1 or len(times) < 3 or not torch.isfinite(times).all():
        raise ValueError('A finite one-dimensional physical time array is required')
    if not math.isfinite(period) or period <= 1.8 or not math.isclose(period/dt, round(period/dt), abs_tol=1e-6):
        raise ValueError('Period must exceed the observation history and be a multiple of policy_dt')
    if not torch.allclose(times, torch.arange(len(times), dtype=times.dtype)*dt, atol=1e-6, rtol=0):
        raise ValueError('Time must describe one continuous episode starting at zero')
    complete_cycles = math.floor((float(times[-1])+dt)/period+1e-7)
    if complete_cycles < 3:
        raise ValueError('At least three complete physical cycles are required for a purged split')
    validation_cycles = max(1, math.ceil(complete_cycles/3))
    first_validation = complete_cycles-validation_cycles
    train_end = (first_validation-1)*period
    validation_start = first_validation*period
    train = times < train_end-1e-7
    validation = (times >= validation_start-1e-7) & (times < complete_cycles*period-1e-7)
    if not train.any() or not validation.any():
        raise ValueError('Empty cycle split')
    return train, validation, dict(complete_cycles=complete_cycles, train_end_s=train_end,
        validation_start_s=validation_start, validation_end_s=complete_cycles*period,
        purged_interval_s=[train_end, validation_start], split='Whole cycles; one full cycle purged; no shuffled-frame split')


def prepare(data, case_name, mode, allow_cpu_oracle=False, root=ROOT):
    if data.get('schema') != 'fixed_wave_dataset_v1' or mode not in ('snake', 'worm', 'mixed'):
        raise ValueError('Unsupported dataset schema or mode')
    if data.get('device') not in ('cuda', 'cpu') or (data['device'] != 'cuda' and not allow_cpu_oracle):
        raise ValueError('CPU oracle data require --allow-cpu-oracle and cannot authorize GPU warmstart')
    if not math.isclose(float(data.get('policy_dt_s', -1)), .2) or not math.isclose(float(data.get('history_span_s', -1)), 1.8):
        raise ValueError('Observation history / policy timing mismatch')
    physics_dt = float(data.get('physics_dt_s', float('nan')))
    if not math.isfinite(physics_dt) or physics_dt <= 0 or (data['device'] == 'cuda' and not math.isclose(physics_dt, .00025)):
        raise ValueError('CUDA teacher must use the training physics dt=0.00025; CPU oracle dt must be explicit')
    sources, inputs = manifest(root)
    if any(data.get('source_sha256', {}).get(k) != v for k, v in sources.items()) or data.get('input_sha256') != inputs:
        raise ValueError('Physics/control source or input manifest mismatch')
    case = data.get('cases', {}).get(case_name)
    if not case or case.get('mode') != mode:
        raise ValueError('Case not found or mode mismatch')
    summary = case.get('summary', {})
    required = ('gate_passed', 'reward_replay_better_than_stationary', 'complete')
    if any(summary.get(key) is not True for key in required):
        raise ValueError('Only a complete passed wave with better reward than matched rest may be cloned')
    obs = torch.as_tensor(case['actor_observations']).detach().cpu().float()
    action = torch.as_tensor(case['teacher_actions']).detach().cpu().float()
    times = torch.as_tensor(case['time_s']).detach().cpu().double()
    if obs.ndim != 2 or obs.shape[1] != 570 or action.shape != (len(obs), 9) or times.shape != (len(obs),):
        raise ValueError('Actor inputs must stay [T,570], physical actions [T,9], time [T]')
    if not torch.isfinite(obs).all() or not torch.isfinite(action).all() or action.abs().max() > 1+1e-6:
        raise ValueError('Nonfinite observations/actions or unbounded teacher commands')
    if not math.isclose(float(summary.get('valid_duration_s', -1)), len(times)*.2, abs_tol=1e-6):
        raise ValueError('Complete-wave duration does not match the physical data')
    neutral = float(case.get('servo_neutral_deg', float('nan')))
    if not 0 <= neutral <= 180:
        raise ValueError('A finite tendon neutral angle is required')
    if mode == 'snake' and action[:, :5].abs().max() > 1e-6:
        raise ValueError('A pure-snake teacher must hold tendons at neutral')
    if mode == 'worm' and action[:, 5:].abs().max() > 1e-6:
        raise ValueError('A pure-worm teacher must hold yaw joints at zero')
    command = float(case.get('command_vx_mm_s', 20.))
    command_scale = data.get('command_scale', [.2, .01, .2])
    expected = obs.new_tensor([command*.001/command_scale[0], 0., 0.])
    if not torch.allclose(obs[:, -3:], expected.expand(len(obs), -1), atol=1e-5, rtol=0):
        raise ValueError('Teacher task command and actor observation disagree')
    train, validation, split = cycle_split(times, float(case['period_s']))
    target_action = action[:, -4:] if mode == 'snake' else (action[:, :5] if mode == 'worm' else action)
    action_size = 4 if mode == 'snake' else (5 if mode == 'worm' else 9)
    return obs, target_action, train, validation, dict(
        case=case_name, mode=mode, dataset_device=data['device'], gpu_warmstart_eligible=data['device']=='cuda',
        source_sha256=sources, input_sha256=inputs, physics_dt_s=physics_dt, policy_dt_s=.2, history_span_s=1.8,
        obs_size=570, action_size=action_size, servo_neutral_deg=neutral,
        teacher_command_vx_mm_s=command, command_scale=command_scale, period_s=float(case['period_s']),
        wave_gate=summary, split=split, scope='Fixed-wave behavior cloning only; not PPO or learned locomotion acceptance')


def angle_rmse_deg(prediction, target, mode, neutral):
    def angles(a):
        if mode == 'snake':
            return a*45.
        if mode == 'worm':
            return neutral + a[:, :5].clamp_min(0)*(180.-neutral) + a[:, :5].clamp_max(0)*neutral
        worm = neutral + a[:, :5].clamp_min(0)*(180.-neutral) + a[:, :5].clamp_max(0)*neutral
        return torch.cat((worm, a[:, 5:]*45.), -1)
    return (angles(prediction)-angles(target)).square().mean(0).sqrt().tolist()


def fit_actor(model, obs, target, train_mask, validation_mask, epochs=400, batch_size=128, lr=3e-4):
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
    train_ids = torch.where(train_mask)[0]
    with torch.no_grad():
        initial = float((model.action(obs[validation_mask])-target[validation_mask]).square().mean())
    best, state = initial, copy.deepcopy(model.actor.state_dict())
    curve = []
    for epoch in range(1, epochs+1):
        order = train_ids[torch.randperm(len(train_ids), device=obs.device)]
        for ids in order.split(batch_size):
            loss = (model.action(obs[ids])-target[ids]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            validation_loss = float((model.action(obs[validation_mask])-target[validation_mask]).square().mean())
        if validation_loss < best:
            best, state = validation_loss, copy.deepcopy(model.actor.state_dict())
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            curve.append(dict(epoch=epoch, validation_action_mse=validation_loss, best_validation_action_mse=best))
    model.actor.load_state_dict(state)
    return dict(initial_validation_action_mse=initial, best_validation_action_mse=best,
                heldout_improved=best < initial*.99-1e-9, curve=curve)


def main(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    path, out = Path(args.data).resolve(), Path(args.out)
    data = torch.load(path, map_location='cpu', weights_only=True)
    obs, actions, train, validation, proof = prepare(data, args.case, args.mode, args.allow_cpu_oracle)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'status.json', {'status': 'behavior_cloning', 'scope': 'Supervised fixed-wave initialization, not PPO'})
    try:
        device = torch.device(args.device)
        model = ActorCritic(570, proof['action_size']).to(device)
        obs, actions, train, validation = [a.to(device) for a in (obs, actions, train, validation)]
        report = fit_actor(model, obs, actions, train, validation, args.epochs, args.batch_size)
        with torch.no_grad():
            report['heldout_target_angle_rmse_deg'] = angle_rmse_deg(model.action(obs[validation]), actions[validation], args.mode, proof['servo_neutral_deg'])
            model.logstd.fill_(math.log(args.initial_std))
        report.update(status='behavior_cloning_complete' if report['heldout_improved'] else 'needs_diagnosis',
                      train_frames=int(train.sum()), validation_frames=int(validation.sum()), **proof)
        config = {**vars(args), **proof, 'dataset_path': str(path), 'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        write(out/'config.json', config)
        write(out/'status.json', report)
        if not report['heldout_improved']:
            raise RuntimeError('Held-out imitation did not improve; no warmstart checkpoint produced')
        checkpoint = dict(schema='bc_actor_v1', actor=model.actor.state_dict(), proof=proof,
            dataset_sha256=config['dataset_sha256'], report=report,
            scope='Behavior cloning initialization; requires deterministic closed-loop evaluation before PPO')
        torch.save(checkpoint, out/'actor_checkpoint.pt')
        print(json.dumps(report, indent=2))
    except BaseException as error:
        write(out/'error.json', {'status': 'failed', 'error': repr(error)})
        raise


def cli_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--mode', choices=['snake', 'worm', 'mixed'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--allow-cpu-oracle', action='store_true')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--initial-std', type=float, default=.65)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args(argv)
    if a.device == 'auto':
        a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if min(a.epochs, a.batch_size) < 1 or not .05 <= a.initial_std <= 1.5:
        p.error('Invalid epochs, batch size or initial standard deviation')
    main(a)


if __name__ == '__main__':
    cli_main()

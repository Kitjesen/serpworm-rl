"""Exact frozen-source mean/sample and observation-noise diagnostic; no training."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import torch


def lanes():
    return [dict(command_mm_s=speed, controller=controller, sensor_noise=noise)
            for speed in (20., 200.) for controller in ('mean', 'sample') for noise in (False, True)]


def configure_noise(env, definitions):
    enabled = torch.tensor([row['sensor_noise'] for row in definitions], device=env.device, dtype=env.dtype)
    env.noise_scale = env.noise_scale.clone()[None, :]*enabled[:, None]
    env.noise = True
    return env.reset(torch.ones(len(definitions), device=env.device, dtype=torch.bool))


def choose_actions(mean, sample, definitions, active):
    use_sample = torch.tensor([row['controller'] == 'sample' for row in definitions], device=mean.device)
    return torch.where(active[:, None], torch.where(use_sample[:, None], sample, mean), 0.)


def summarize(trace, definitions):
    dt, sensor_dt = trace['policy_dt_s'], trace['sensor_dt_s']
    repeat, seconds = round(dt/sensor_dt), len(trace['time_s'])*dt
    results = []
    for lane, definition in enumerate(definitions):
        valid = trace['active'][:, lane]
        failed = bool((trace['terminated'][:, lane] & valid).any())
        windows = []
        for start in [max(0., seconds-10.)+2*i for i in range(math.ceil(min(10., seconds)/2))]:
            end = min(seconds, start+2.)
            selected = valid & (trace['time_s'] >= start-1e-8) & (trace['time_s'] < end-1e-8)
            record = dict(start_s=start, end_s=end, valid_duration_s=float(selected.sum())*dt,
                complete=bool(selected.sum() == round((end-start)/dt)))
            if selected.any():
                indices = torch.where(selected)[0]
                initial = trace['initial_plate_positions_m'][lane]
                before = initial if int(indices[0]) == 0 else trace['plate_positions_m'][int(indices[0])-1, lane]
                delta = trace['plate_positions_m'][int(indices[-1]), lane].mean(0)-before.mean(0)
                forward = initial[9, :2]-initial[0, :2]; forward /= forward.norm().clamp_min(1e-12)
                yaw = trace['yaw_diagnostics'][:, lane][selected.repeat_interleave(repeat)]
                if yaw.shape[-2:] != (6, 4): raise ValueError('Yaw diagnostic shape must be six fields by four joints')
                servo = trace['servo_diagnostics'][selected, lane]
                velocity = trace['body_velocity'][selected, lane]
                record.update(mean_vx_mm_s=float(velocity[:, 0].mean())*1000,
                    forward_integral_mm=float(velocity[:, 0].sum())*dt*1000,
                    mean_abs_vx_mm_s=float(velocity[:, 0].abs().mean())*1000,
                    mean_slip_m_s=float(trace['metrics'][selected, lane, 0].mean()),
                    min_clearance_m=float(trace['metrics'][selected, lane, 2].min()),
                    world_centroid_displacement_mm=(delta*1000).tolist(),
                    world_initial_forward_displacement_mm=float(delta[:2]@forward)*1000,
                    physical_guard_count=int((trace['metrics'][selected, lane, 5] > 0).sum()))
                for label, field in (('requested', 0), ('applied', 1), ('actual', 2)):
                    record[label+'_yaw_pp_deg']=((yaw[:, field].amax(0)-yaw[:, field].amin(0))*180/math.pi).tolist()
                    record[label+'_servo_pp_deg']=((servo[:, field].amax(0)-servo[:, field].amin(0))*180/math.pi).tolist()
                record['action_pp']=(trace['actions'][selected, lane].amax(0)-trace['actions'][selected, lane].amin(0)).tolist()
            windows.append(record)
        results.append(dict(**definition, first_episode_complete=bool(valid.all()) and not failed,
            failed=failed, valid_duration_s=float(valid.sum())*dt, tail_windows_2s=windows,
            scope='Stochastic sampled movement is exploration evidence, not deterministic policy success.'))
    return results


def verified_config(root, checkpoint):
    config = json.loads((checkpoint.parent/'config.json').read_text(encoding='utf-8'))
    names = ('gpu_core.py', 'tensor_env.py', 'rewards.py', 'train_gpu.py', 'wheel_stiction.py')
    checked = {}
    for name in names:
        digest = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if digest != config.get('source_sha256', {}).get(name): raise RuntimeError(f'Frozen source mismatch: {name}')
        checked[name] = digest
    if not config.get('input_sha256'): raise RuntimeError('Missing input hash manifest')
    for name, expected in config['input_sha256'].items():
        if hashlib.sha256((root/'inputs'/name).read_bytes()).hexdigest() != expected: raise RuntimeError(f'Input mismatch: {name}')
    return config, checked


def run(args):
    root, checkpoint = Path(args.source_dir).resolve(), Path(args.checkpoint).resolve()
    config, source = verified_config(root, checkpoint)
    # Run this script in its own process. The frozen directory takes precedence;
    # no new experiment environment/controller is imported into this baseline.
    sys.path.insert(0, str(root))
    modules = {name: importlib.import_module(name) for name in ('gpu_core', 'rewards', 'wheel_stiction', 'tensor_env', 'train_gpu')}
    if any(Path(module.__file__).resolve().parent != root for module in modules.values()): raise RuntimeError('Cached module from a different source directory')
    data = torch.load(checkpoint, map_location='cuda', weights_only=False)  # Trusted project checkpoints only.
    for key in ('kind', 'mode', 'reward', 'control_profile', 'servo_neutral_deg'):
        if data['config'][key] != config[key]: raise RuntimeError(f'Checkpoint/config mismatch: {key}')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    definitions = lanes()
    env = modules['tensor_env'].TensorEnv('direct', n=8, seed=args.seed, noise=False,
        reward_version=config['reward'], control_profile=config['control_profile'], plate_solver=config['plate_solver'],
        command_switching=False, servo_neutral_deg=config['servo_neutral_deg'], mode=config['mode'],
        vx_min_mm_s=config['vx_min_mm_s'], vx_max_mm_s=config['vx_max_mm_s'], stop_fraction=config['stop_fraction'])
    obs = configure_noise(env, definitions)
    env.cmd.zero_(); env.cmd[:, 0] = env.tensor([row['command_mm_s']*1e-3 for row in definitions])
    env.history[:, :, -3:] = (env.cmd/env.command_scale)[:, None]
    obs = env.history.flatten(1).float().clone()
    model = modules['train_gpu'].ActorCritic(env.obs_size, env.action_size).cuda()
    model.load_state_dict(data['model']); model.eval()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    metadata = dict(**vars(args), source_sha256=source, input_sha256=config['input_sha256'],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_rollout=data['report']['rollouts'],
        physics_dt_s=env.c.dt, mode=config['mode'], policy_dt_s=env.policy_dt, sensor_dt_s=env.sensor_dt, definitions=definitions,
        scope='Exact frozen checkpoint distribution diagnostic, not new training or an r32 claim unless checkpoint metadata says r32')
    (out/'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out/'training_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    snapshot = out/'sources'; snapshot.mkdir(); (snapshot/'inputs').mkdir()
    for name in source: (snapshot/name).write_bytes((root/name).read_bytes())
    for name in config['input_sha256']: (snapshot/'inputs'/name).write_bytes((root/'inputs'/name).read_bytes())
    active = torch.ones(8, device='cuda', dtype=torch.bool)
    rows = {}; initial = env.c.x[:, :10, :3].clone()
    with torch.no_grad():
        for step in range(round(args.seconds/env.policy_dt)):
            action = choose_actions(model.action(obs), model.sample(obs)[0], definitions, active)
            obs, reward, term, trunc, _, metrics = env.step(action)
            record = dict(actions=action, active=active.clone(), terminated=term, truncated=trunc,
                body_velocity=env.last_body_velocity, filtered_velocity=env.last_velocity, command=env.last_command,
                yaw_diagnostics=env.last_diagnostics, servo_diagnostics=env.last_servo_diagnostics,
                plate_positions_m=env.last_plate_positions, metrics=metrics)
            for key, value in record.items(): rows.setdefault(key, []).append(value.clone())
            active &= ~(term | trunc)
            if bool(env.c.failed): raise RuntimeError('Physics failed; diagnostic result is invalid')
    trace = {key: (torch.cat(values) if key == 'yaw_diagnostics' else torch.stack(values)).cpu() for key, values in rows.items()}
    trace.update(time_s=torch.arange(len(trace['active']), dtype=torch.float64)*env.policy_dt,
        policy_dt_s=env.policy_dt, sensor_dt_s=env.sensor_dt, initial_plate_positions_m=initial.cpu(), definitions=definitions)
    torch.save(trace, out/'traces.pt')
    (out/'summary.json').write_text(json.dumps(dict(results=summarize(trace, definitions), metadata=metadata), indent=2), encoding='utf-8')
    (out/'status.json').write_text(json.dumps(dict(status='complete', no_training=True)), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--checkpoint', required=True, help='Trusted checkpoint from this project only')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 20. or not math.isclose(args.seconds/.2, round(args.seconds/.2)):
        parser.error('Use a duration in (0,20] seconds divisible by 0.2; never concatenate reset episodes')
    run(args)

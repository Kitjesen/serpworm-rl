"""First-episode direct-PPO evaluation with explicit sustained-motion windows.

The default controller is the deterministic actor mean.  Stochastic evaluation
is an exploration diagnostic only and cannot satisfy deterministic acceptance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from tensor_env import TensorEnv
from train_gpu import ActorCritic, write


POLICY_DT = .2
SENSOR_DT = .02
EPISODE_SECONDS = 20.
STOP_TIME = 10.
WINDOW_SECONDS = 2.
CASE_NAMES = ['stop', 'straight5', 'straight20', 'straight50', 'straight100',
              'straight150', 'straight200', 'straight100_then_stop']
SPEEDS_MM_S = [0., 5., 20., 50., 100., 150., 200., 100.]
YAW_FIELDS = ['requested_target_rad', 'applied_target_rad', 'actual_angle_rad',
              'actual_rate_rad_s', 'last_substep_torque_nm', 'last_substep_torque_cap_nm']
SERVO_FIELDS = ['requested_target_rad', 'applied_target_rad', 'actual_angle_rad']


def verify_sources(checkpoint):
    """Reject evaluator/source/input drift relative to the saved run manifest."""
    config = json.loads((checkpoint.parent/'config.json').read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parent
    required = ['train_gpu.py', 'tensor_env.py', 'rewards.py', 'gpu_core.py',
                'wheel_stiction.py', 'wheel_envelope.py', 'evaluate_gpu.py']
    checked = {}
    for name in required:
        expected = config.get('source_sha256', {}).get(name)
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if not expected or actual != expected:
            raise RuntimeError(f'Source manifest mismatch: {name}')
        checked[name] = actual
    inputs = config.get('input_sha256', {})
    if not inputs:
        raise RuntimeError('Training config has no input manifest')
    for name, expected in inputs.items():
        actual = hashlib.sha256((root/'inputs'/name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Input manifest mismatch: {name}')
        checked['inputs/'+name] = actual
    return config, checked


def amplitude_summary(yaw, servo):
    """Yaw is sampled at 50 Hz; servo and actions at the recorded policy rate."""
    deg = 180/math.pi
    result = {}
    for name, field in [('requested', 0), ('applied', 1), ('actual', 2)]:
        result[name+'_yaw_pp_deg'] = ((yaw[:, field].amax(0)-yaw[:, field].amin(0))*deg).tolist()
        result[name+'_servo_pp_deg'] = ((servo[:, field].amax(0)-servo[:, field].amin(0))*deg).tolist()
    result['yaw_tracking_rmse_deg'] = ((yaw[:, 1]-yaw[:, 2]).square().mean(0).sqrt()*deg).tolist()
    result['servo_tracking_rmse_deg'] = ((servo[:, 1]-servo[:, 2]).square().mean(0).sqrt()*deg).tolist()
    result['actual_yaw_active_channel_count'] = sum(value >= 5. for value in result['actual_yaw_pp_deg'])
    result['actual_servo_active_channel_count'] = sum(value >= 5. for value in result['actual_servo_pp_deg'])
    # Adjacent rope servos receive the same segment action.  Count paired units
    # once, requiring both measured servo channels to exceed the threshold.
    pair_pp = [min(result['actual_servo_pp_deg'][i:i+2]) for i in range(0, 10, 2)]
    result['actual_rope_pair_pp_deg'] = pair_pp
    result['actual_rope_active_segment_count'] = sum(value >= 5. for value in pair_pp)
    return result


def window_summary(traces, lane, start, end):
    """Statistics on one physical interval, always masked to the first episode."""
    times, dt = traces['time_start_s'], traces['policy_dt_s']
    repeat = round(dt/traces['diagnostics_dt_s'])
    requested = (times >= start-1e-8) & (times < end-1e-8)
    selected = requested & traces['active'][:, lane]
    expected_steps = round((end-start)/dt)
    indices = torch.where(selected)[0]
    complete = bool(requested.sum() == expected_steps) and bool(selected.sum() == expected_steps) and expected_steps > 0
    result = dict(start_s=start, end_s=end, expected_duration_s=end-start,
                  valid_duration_s=float(selected.sum())*dt, complete=complete,
                  policy_step_indices=indices.tolist(), summary=None)
    if not bool(selected.any()):
        return result
    raw = traces['body_velocity'][selected, lane]
    filtered = traces['filtered_velocity'][selected, lane]
    command = traces['command'][selected, lane]
    metrics = traces['metrics'][selected, lane]
    yaw = traces['diagnostics_50hz'][:, lane][selected.repeat_interleave(repeat)]
    servo = traces['servo_diagnostics_policy_hz'][selected, lane]
    actions = traces['actions'][selected, lane]
    first, last = int(indices[0]), int(indices[-1])
    before = traces['initial_plate_positions_m'][lane] if first == 0 else traces['plate_positions_m'][first-1, lane]
    after = traces['plate_positions_m'][last, lane]
    world_delta = after.mean(0)-before.mean(0)
    summary = dict(
        mean_vx_mm_s=float(raw[:, 0].mean())*1000,
        mean_abs_vx_mm_s=float(raw[:, 0].abs().mean())*1000,
        body_velocity_rmse_m_s_m_s_rad_s=(raw-command).square().mean(0).sqrt().tolist(),
        filtered_velocity_rmse_m_s_m_s_rad_s=(filtered-command).square().mean(0).sqrt().tolist(),
        mean_body_velocity_m_s_m_s_rad_s=raw.mean(0).tolist(),
        mean_filtered_velocity_m_s_m_s_rad_s=filtered.mean(0).tolist(),
        signed_forward_velocity_integral_mm=float(raw[:, 0].sum())*dt*1000,
        world_centroid_displacement_mm=(world_delta*1000).tolist(),
        world_centroid_planar_distance_mm=float(world_delta[:2].norm())*1000,
        mean_slip_m_s=float(metrics[:, 0].mean()),
        yaw_work_proxy_j=float(metrics[:, 1].sum()),
        min_wheel_clearance_m=float(metrics[:, 2].min()),
        sampled_yaw_torque_saturation_fraction=float(((yaw[:, 4].abs() >= .99*yaw[:, 5]) & (yaw[:, 5] > 1e-6)).double().mean()),
        requested_action_pp=(actions.amax(0)-actions.amin(0)).tolist(),
        **amplitude_summary(yaw, servo))
    # This is an engineering screen for continued motion, not target-speed
    # tracking, a required biological waveform, or hardware validation.
    summary['forward_motion_present'] = summary['mean_vx_mm_s'] >= 1.
    summary['yaw_motion_present'] = summary['actual_yaw_active_channel_count'] >= 2
    summary['rope_motion_present'] = summary['actual_rope_active_segment_count'] >= 2
    summary['frozen_pose_detected'] = max(summary['actual_yaw_pp_deg']+summary['actual_servo_pp_deg']) < 1.
    summary['below_sustained_motion_screen'] = not (summary['forward_motion_present'] and summary['yaw_motion_present'])
    result['summary'] = summary
    return result


def summarize(traces, seconds, mode='mixed', stochastic=False):
    times, dt = traces['time_start_s'], traces['policy_dt_s']
    summaries = []
    for lane, name in enumerate(CASE_NAMES):
        valid = traces['active'][:, lane]
        terminated = valid & traces['terminated'][:, lane]
        truncated = valid & traces['truncated'][:, lane]
        stopping = lane in (0, 7)
        start = max(0., seconds-(5. if stopping else 10.))
        if lane == 7:
            start = max(STOP_TIME, start)
        # A short user-requested diagnostic might never enter the stop phase.
        tail = window_summary(traces, lane, min(start, seconds), seconds)
        windows = [window_summary(traces, lane, begin, min(begin+WINDOW_SECONDS, seconds))
                   for begin in (i*WINDOW_SECONDS for i in range(math.ceil(seconds/WINDOW_SECONDS)))]
        # The tail is independently tiled in two-second windows; the final
        # one-second stop window is retained when the tail is five seconds.
        tail_windows = [window_summary(traces, lane, start+i*WINDOW_SECONDS,
                                      min(start+(i+1)*WINDOW_SECONDS, seconds))
                        for i in range(max(0, math.ceil((seconds-start)/WINDOW_SECONDS)))]
        failed = bool(terminated.any())
        complete_episode = int(valid.sum()) == len(times)
        screen = (not stopping and not failed and complete_episode and tail['complete'] and
                  bool(tail_windows) and all(window['complete'] and
                      not window['summary']['below_sustained_motion_screen'] for window in tail_windows))
        rope_screen = bool(screen and mode == 'mixed' and all(window['summary']['rope_motion_present'] for window in tail_windows))
        item = dict(case=name, command_vx_mm_s=SPEEDS_MM_S[lane],
                    stop_time_s=STOP_TIME if lane == 7 else None,
                    valid_steps=int(valid.sum()), valid_duration_s=float(valid.sum())*dt,
                    failed=failed, time_limit_reached=bool(truncated.any()),
                    first_episode_complete=complete_episode,
                    summary_window_start_s=start, summary_window_end_s=seconds,
                    summary_valid_duration_s=tail['valid_duration_s'], window_complete=tail['complete'],
                    summary=tail['summary'], windows_2s=windows, tail_windows=tail_windows,
                    sustained_motion_screen=bool(screen),
                    deterministic_sustained_motion_screen=bool(screen and not stochastic),
                    mixed_rope_motion_screen=rope_screen,
                    freeze_window_count=sum(window['summary'] is not None and window['summary']['frozen_pose_detected'] for window in tail_windows),
                    below_sustained_motion_window_count=sum(window['summary'] is None or window['summary']['below_sustained_motion_screen'] for window in tail_windows))
        if failed:
            item['first_termination_time_s'] = float(times[terminated][0])+dt
        if bool(truncated.any()):
            item['first_truncation_time_s'] = float(times[truncated][0])+dt
        summaries.append(item)
    return summaries


def aggregate(results, stochastic=False):
    """Keep existing trainer fields; deterministic sustained counts cannot use noise."""
    moving, stopping = results[1:7], [results[0], results[7]]
    moving_complete = all(row['window_complete'] for row in moving)
    stopping_complete = all(row['window_complete'] for row in stopping)
    observed_count = sum(row['sustained_motion_screen'] for row in moving)
    return dict(
        moving_vx_rmse_mm_s=sum(row['summary']['body_velocity_rmse_m_s_m_s_rad_s'][0]*1000 for row in moving)/6 if moving_complete else None,
        moving_tail_mean_vx_mm_s=sum(row['summary']['mean_vx_mm_s'] for row in moving)/6 if moving_complete else None,
        moving_guard_failures=sum(row['failed'] for row in moving),
        moving_complete_window_count=sum(row['window_complete'] for row in moving),
        moving_sustained_case_count=0 if stochastic else observed_count,
        observed_sustained_case_count=observed_count,
        moving_frozen_tail_window_count=sum(row['freeze_window_count'] for row in moving),
        mixed_rope_sustained_case_count=0 if stochastic else sum(row['mixed_rope_motion_screen'] for row in moving),
        all_stop_mean_abs_vx_mm_s=sum(row['summary']['mean_abs_vx_mm_s'] for row in stopping)/2 if stopping_complete else None,
        stop_guard_failures=sum(row['failed'] for row in stopping),
        stop_complete_window_count=sum(row['window_complete'] for row in stopping),
        acceptance_eligible=not stochastic,
        aggregation='Equal-weight complete tail windows. Sustained count screens persistence only, not target tracking or convergence.')


def evaluate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'status.json', {'status': 'verifying_sources'})
    try:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        checkpoint = Path(args.checkpoint).resolve()
        manifest, checked = verify_sources(checkpoint)
        # Only trusted checkpoints generated by this project may be supplied.
        data = torch.load(checkpoint, map_location='cuda', weights_only=False)
        config = data['config']
        for key in ('kind', 'mode', 'reward', 'control_profile', 'servo_neutral_deg',
                    'plate_solver', 'vx_min_mm_s', 'vx_max_mm_s', 'stop_fraction'):
            if config[key] != manifest[key]:
                raise RuntimeError(f'Checkpoint/config mismatch: {key}')
        if config['kind'] != 'direct' or config['reward'] != 'sustain_v4' or config['control_profile'] != 'sustain_v4':
            raise ValueError('This evaluator accepts sustain_v4 direct policies only')
        env = TensorEnv('direct', n=8, seed=args.seed, noise=False, reward_version=config['reward'],
                        control_profile=config['control_profile'], plate_solver=config['plate_solver'],
                        command_switching=False, servo_neutral_deg=config['servo_neutral_deg'],
                        fixed_vx_mm_s=None, mode=config['mode'],
                        vx_min_mm_s=config['vx_min_mm_s'], vx_max_mm_s=config['vx_max_mm_s'],
                        stop_fraction=config['stop_fraction'])
        if not (math.isclose(env.policy_dt, POLICY_DT) and math.isclose(env.sensor_dt, SENSOR_DT)
                and env.steps_per_action == round(POLICY_DT/SENSOR_DT)):
            raise RuntimeError('Evaluator/environment timing mismatch')
        if env.obs_size != data['obs_size'] or env.obs_size != 570:
            raise RuntimeError('Checkpoint observation width does not match evaluator')
        if manifest.get('policy_action_size', env.action_size) != env.action_size:
            raise RuntimeError('Checkpoint action width does not match evaluator')
        steps = round(args.seconds/env.policy_dt)
        seconds = steps*env.policy_dt
        controller = 'stochastic_exploration_only' if args.stochastic else 'deterministic_actor_mean'
        write(out/'config.json', {**vars(args), 'actual_seconds': seconds,
              'policy_dt_s': env.policy_dt, 'diagnostics_dt_s': env.sensor_dt,
              'steps_per_action': env.steps_per_action, 'mode': config['mode'],
              'reward': config['reward'], 'control_profile': config['control_profile'],
              'controller': controller, 'acceptance_eligible': not args.stochastic,
              'sensor_noise': False, 'command_switching': False, 'cases': CASE_NAMES,
              'stop_time_s': STOP_TIME, 'evaluation_speed_grid_mm_s': SPEEDS_MM_S,
              'training_command_sampling': manifest.get('command_sampling'),
              'verified_source_sha256': checked,
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'checkpoint_rollout': data.get('report', {}).get('rollouts'),
              'sustained_screen': 'Each moving tail 2 s window: mean vx >=1 mm/s and at least 2 actual yaw channels with PP >=5 deg; full first episode and no guard termination.',
              'mixed_rope_screen': 'Additionally report >=2 independent paired-tendon segments with both servo PP >=5 deg in every tail window.',
              'scope': 'First physical episode only; 20 s final truncation is legitimate; stochastic passing is not deterministic acceptance.'})
        model = ActorCritic(env.obs_size, env.action_size).cuda()
        model.load_state_dict(data['model'])
        model.eval()
        command = env.zeros(8, 3)
        command[:, 0] = env.tensor(SPEEDS_MM_S)*1e-3
        env.cmd.copy_(command)
        env.history[:, :, -3:] = (command/env.command_scale)[:, None]
        active = torch.ones(8, device='cuda', dtype=torch.bool)
        initial_positions = env.c.x[:, :10, :3].clone()
        rows = {name: [] for name in ['actions', 'command', 'active', 'terminated', 'truncated',
                    'reward', 'filtered_velocity', 'body_velocity', 'metrics', 'plate_positions_m',
                    'diagnostics_50hz', 'servo_diagnostics_policy_hz']}
        write(out/'status.json', {'status': 'evaluating', 'seconds': seconds, 'cases': 8, 'controller': controller})
        with torch.no_grad():
            for step in range(steps):
                if step*env.policy_dt >= STOP_TIME-1e-8:
                    command[7].zero_()
                env.cmd.copy_(command)
                env.history[:, -1, -3:] = command/env.command_scale
                obs = env.history.flatten(1).float().clone()
                action = model.sample(obs)[0] if args.stochastic else model.action(obs)
                action[~active] = 0.
                _, reward, terminated, truncated, _, metrics = env.step(action)
                current = {'actions': action, 'command': env.last_command, 'active': active,
                    'terminated': terminated, 'truncated': truncated, 'reward': reward,
                    'filtered_velocity': env.last_velocity, 'body_velocity': env.last_body_velocity,
                    'metrics': metrics, 'plate_positions_m': env.last_plate_positions,
                    'diagnostics_50hz': env.last_diagnostics,
                    'servo_diagnostics_policy_hz': env.last_servo_diagnostics}
                for name, value in current.items():
                    rows[name].append(value.clone())
                # The last pre-reset 20 s sample remains valid.  Reset outputs
                # cannot reactivate a terminated/truncated first episode.
                active &= ~(terminated | truncated)
                if (step+1) % 20 == 0:
                    write(out/'status.json', {'status': 'evaluating', 'completed_steps': step+1,
                          'elapsed_simulation_s': (step+1)*env.policy_dt, 'seconds': seconds,
                          'cases': 8, 'controller': controller})
        if bool(env.c.failed):
            raise RuntimeError('Physics reported nonfinite state')
        traces = {name: (torch.cat(values) if name == 'diagnostics_50hz' else torch.stack(values)).cpu()
                  for name, values in rows.items()}
        for name, values in traces.items():
            if values.is_floating_point() and not bool(torch.isfinite(values).all()):
                raise RuntimeError(f'Nonfinite trace: {name}')
        traces.update(time_start_s=torch.arange(steps, dtype=torch.float64)*env.policy_dt,
                      time_end_s=(torch.arange(steps, dtype=torch.float64)+1)*env.policy_dt,
                      diagnostics_time_end_s=(torch.arange(steps*env.steps_per_action, dtype=torch.float64)+1)*env.sensor_dt,
                      initial_plate_positions_m=initial_positions.cpu(), case_names=CASE_NAMES,
                      diagnostic_fields=YAW_FIELDS, servo_diagnostic_fields=SERVO_FIELDS,
                      policy_dt_s=env.policy_dt, diagnostics_dt_s=env.sensor_dt,
                      controller=controller, stochastic=args.stochastic,
                      first_episode_mask_semantics='active is captured before step; final truncating transition is included; later episodes are excluded')
        torch.save(traces, out/'traces.pt')
        results = summarize(traces, seconds, config['mode'], args.stochastic)
        write(out/'summary.json', {'mode': config['mode'], 'controller': controller,
              'acceptance_eligible': not args.stochastic, 'results': results,
              'aggregate': aggregate(results, args.stochastic),
              'scope': 'Fixed friction; no observation noise; first episode only; persistence screens are not comprehensive success.',
              'limitations': ['Not hardware validation or robust multi-seed success.',
                  'Stochastic evaluation measures exploration, never deterministic deployment acceptance.',
                  'Signed forward velocity integral differs from world-frame net displacement; both are reported.',
                  'Energy is a yaw actuator work proxy, excluding tendon work and electrical losses.',
                  'Post-reset samples are saved with active=false and excluded from all statistics.',
                  'The mixed rope screen counts five paired segment controls, not ten duplicated commands.']})
        write(out/'status.json', {'status': 'evaluation_complete', 'cases': 8, 'seconds': seconds,
                                'controller': controller, 'acceptance_eligible': not args.stochastic})
    except BaseException as error:
        write(out/'status.json', {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--stochastic', action='store_true', help='Exploration diagnostic only; cannot count as deterministic acceptance')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= EPISODE_SECONDS:
        parser.error('Duration must be positive and at most the 20 s first-episode limit')
    if round(args.seconds/POLICY_DT) < 1 or not math.isclose(args.seconds/POLICY_DT, round(args.seconds/POLICY_DT), abs_tol=1e-7):
        parser.error('Duration must be a multiple of the 0.2 s policy interval')
    evaluate(args)

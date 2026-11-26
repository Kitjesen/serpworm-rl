"""First-episode fixed-wave reachability, not PPO or a learned gait."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from serpworm.envs.rewards import velocity_reward
from serpworm.paths import PKG_ROOT, collect_input_hashes, collect_source_hashes, freeze_sources


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    neutral_deg: float
    yaw_amplitude_deg: float = 0.
    rope_amplitude_deg: float = 0.
    period_s: float = 6.
    phase_step_deg: float = 90.


def cases():
    rows = []
    for neutral in (0., 45., 90.):
        rows += [Case(f'neutral{neutral:g}', 'static', neutral),
                 Case(f'pose35_neutral{neutral:g}', 'pose', neutral, 35.)]
        # Keep a known moving baseline while expanding toward full-angle tests.
        for amplitude, period in ((25., 4.), (29.25, 4.), (35., 6.), (45., 8.)):
            for sign in (1, -1):
                rows.append(Case(f'snake_a{amplitude:g}_t{period:g}_p{sign*90:+d}_n{neutral:g}',
                                 'snake', neutral, amplitude, 0., period, sign*90.))
    for kind, rope, yaw, period in (('mixed', 60., 35., 8.), ('full', 90., 45., 10.)):
        for sign in (1, -1):
            rows.append(Case(f'{kind}_r{rope:g}_a{yaw:g}_t{period:g}_p{sign*90:+d}_n90',
                             kind, 90., yaw, rope, period, sign*90.))
    for neutral, rope, yaw, period in ((45., 45., 29.25, 6.), (60., 60., 35., 8.)):
        for sign in (1, -1):
            rows.append(Case(f'mixed_r{rope:g}_a{yaw:g}_t{period:g}_p{sign*90:+d}_n{neutral:g}',
                             'mixed', neutral, yaw, rope, period, sign*90.))
    # Pure peristaltic worm waves (zero yaw, travelling contraction waves across 5 segments).
    for neutral, rope, period in ((90., 45., 6.), (90., 60., 8.), (90., 90., 10.), (45., 45., 6.), (60., 60., 8.)):
        for sign in (1, -1):
            rows.append(Case(f'worm_r{rope:g}_t{period:g}_p{sign*72:+d}_n{neutral:g}',
                             'worm', neutral, 0., rope, period, sign*72.))
    return rows


def requested_angles(time_s, selected, device='cpu', dtype=torch.float64, ramp_s=2.):
    """One independent row per case, angles in degrees; no shared-column writes."""
    if ramp_s < 0:
        raise ValueError('ramp_s must be nonnegative')
    u = min(1., max(0., time_s/ramp_s)) if ramp_s else 1.
    ramp = u*u*(3-2*u)
    servo, yaw = [], []
    for row in selected:
        phase = 2*math.pi*time_s/row.period_s
        dy = math.radians(row.phase_step_deg)
        ys = [row.yaw_amplitude_deg*ramp*math.sin(phase+j*dy) for j in range(4)]
        if row.kind == 'pose':
            ys = [row.yaw_amplitude_deg*ramp*s for s in (1., -1., 1., -1.)]
        rs = [row.neutral_deg+row.rope_amplitude_deg*ramp*math.sin(phase+j*dy) for j in range(5)]
        if any(not -1e-9 <= value <= 180+1e-9 for value in rs):
            raise ValueError(f'Out-of-range rope request in {row.name}')
        servo.append([value for value in rs for _ in range(2)])
        yaw.append(ys)
    return torch.tensor(servo, device=device, dtype=dtype), torch.tensor(yaw, device=device, dtype=dtype)


def actions_from_angles(servo_deg, yaw_deg, neutral_deg):
    """Inverse of signed neutral-centred map, including neutral 0/180 degrees."""
    if servo_deg.shape[-1] != 10 or yaw_deg.shape[-1] != 4:
        raise ValueError('Expected ten rope and four yaw requests')
    neutral = torch.as_tensor(neutral_deg, device=servo_deg.device, dtype=servo_deg.dtype)
    while neutral.ndim < servo_deg.ndim:
        neutral = neutral.unsqueeze(-1)
    delta = servo_deg-neutral
    contraction = torch.where(delta >= 0, delta/(180-neutral).clamp_min(1e-12),
                              delta/neutral.clamp_min(1e-12))
    if not torch.allclose(contraction[..., ::2], contraction[..., 1::2]):
        raise ValueError('This probe uses paired left/right tendon commands')
    action = torch.cat((contraction[..., ::2], yaw_deg/45.), -1)
    if not bool(torch.isfinite(action).all()) or bool((action.abs() > 1+1e-9).any()):
        raise ValueError('Unreachable mapped angle request')
    return action.clamp(-1., 1.)


def actual_yaw(diagnostic):
    """Keep four joints; never reinterpret the six diagnostic fields as joints."""
    if diagnostic.shape[-2:] != (6, 4):
        raise ValueError('Expected (..., 6 fields, 4 joints) yaw diagnostics')
    return diagnostic[..., 2, :]


def summarize_trace(trace, lane, case, seconds):
    """All statistics exclude samples after a first reset, including reward replay."""
    dt = float(trace['policy_dt_s'])
    times = trace['time_start_s']
    valid = trace['active'][:, lane]
    valid_episode = bool(valid.all()) and not bool((trace['terminated'][:, lane] & valid).any())
    repeat = round(dt/float(trace['sensor_dt_s']))
    initial = trace['initial_plate_positions_m'][lane]
    initial_axis = initial[9, :2]-initial[0, :2]
    initial_axis = initial_axis/initial_axis.norm().clamp_min(1e-12)
    tail_duration = max(20., 2*case.period_s) if case.kind in ('snake', 'worm', 'mixed', 'full') else 20.
    tail_start = max(2., seconds-tail_duration)

    def window(start, end):
        wanted = (times >= start-1e-8) & (times < end-1e-8)
        selected = wanted & valid
        indices = torch.where(selected)[0]
        complete = bool(wanted.sum() == round((end-start)/dt)) and bool(selected.sum() == wanted.sum()) and bool(selected.any())
        result = dict(start_s=start, end_s=end, complete=complete, valid_duration_s=float(selected.sum())*dt)
        if not len(indices):
            return {**result, 'summary': None}
        velocity = trace['body_velocity'][selected, lane]
        yaw = actual_yaw(trace['yaw_diagnostics'][:, lane])[selected.repeat_interleave(repeat)]
        servo = trace['servo_diagnostics'][selected, lane]
        pp = lambda value: (value.amax(0)-value.amin(0))*(180/math.pi)
        ypp, spp = pp(yaw), pp(servo[:, 2])
        pair_pp = spp.reshape(5, 2).amin(-1)
        before = initial if int(indices[0]) == 0 else trace['plate_positions_m'][int(indices[0])-1, lane]
        after = trace['plate_positions_m'][int(indices[-1]), lane]
        delta = after.mean(0)-before.mean(0)
        result['summary'] = dict(mean_vx_mm_s=float(velocity[:, 0].mean())*1000,
            mean_abs_vx_mm_s=float(velocity[:, 0].abs().mean())*1000,
            mean_vy_mm_s=float(velocity[:, 1].mean())*1000,
            mean_wz_rad_s=float(velocity[:, 2].mean()),
            forward_integral_mm=float(velocity[:, 0].sum())*dt*1000,
            backward_sample_fraction=float((velocity[:, 0] < -1e-6).double().mean()),
            world_centroid_displacement_mm=(delta*1000).tolist(),
            world_initial_forward_displacement_mm=float(delta[:2]@initial_axis)*1000,
            actual_yaw_min_deg=(yaw.amin(0)*180/math.pi).tolist(),
            actual_yaw_max_deg=(yaw.amax(0)*180/math.pi).tolist(),
            actual_yaw_pp_deg=ypp.tolist(), actual_servo_pp_deg=spp.tolist(),
            actual_servo_min_deg=(servo[:, 2].amin(0)*180/math.pi).tolist(),
            actual_servo_max_deg=(servo[:, 2].amax(0)*180/math.pi).tolist(),
            requested_yaw_pp_deg=pp(trace['yaw_diagnostics'][:, lane, 0][selected.repeat_interleave(repeat)]).tolist(),
            applied_yaw_pp_deg=pp(trace['yaw_diagnostics'][:, lane, 1][selected.repeat_interleave(repeat)]).tolist(),
            requested_servo_pp_deg=pp(servo[:, 0]).tolist(), applied_servo_pp_deg=pp(servo[:, 1]).tolist(),
            active_yaw_count=int((ypp >= 5).sum()), active_rope_segment_count=int((pair_pp >= 5).sum()),
            min_wheel_clearance_m=float(trace['metrics'][selected, lane, 2].min()),
            mean_slip_m_s=float(trace['metrics'][selected, lane, 0].mean()),
            yaw_work_proxy_j=float(trace['metrics'][selected, lane, 1].sum()),
            max_tension_n=float(trace['tension'][selected, lane].max()),
            max_wheel_normal_n=float(trace['normal_force'][selected, lane].max()),
            min_gap_m=float(trace['gap'][selected, lane].min()),
            max_stop_force_n=float(trace['stop_force'][selected, lane].max()))
        return result

    windows = [window(i*2., min((i+1)*2., seconds)) for i in range(math.ceil(seconds/2.))]
    tail_windows = [window(tail_start+i*2., min(tail_start+(i+1)*2., seconds))
                    for i in range(max(0, math.ceil((seconds-tail_start)/2.)))]
    whole, tail = window(0., seconds), window(tail_start, seconds)
    enough_time = seconds >= 20. and seconds-tail_start >= tail_duration-1e-8
    world_forward = tail['summary'] is not None and tail['summary']['world_initial_forward_displacement_mm'] > 0.
    passed = (case.kind in ('snake', 'mixed', 'full') and enough_time and valid_episode and world_forward
              and bool(tail_windows) and all(w['complete'] and w['summary']['mean_vx_mm_s'] >= 1.
                  and w['summary']['active_yaw_count'] >= 2 for w in tail_windows))
    worm_passed = (case.kind == 'worm' and enough_time and valid_episode and world_forward
                   and bool(tail_windows) and all(w['complete'] and w['summary']['mean_vx_mm_s'] >= 1.
                       and w['summary']['active_rope_segment_count'] >= 2 for w in tail_windows))
    rope_passed = passed and case.kind in ('mixed', 'full') and all(w['summary']['active_rope_segment_count'] >= 2 for w in tail_windows)
    selected = valid & (times >= tail_start-1e-8)
    replay = {}
    if selected.any():
        # Batch execution uses the 9D map. Single-modality training averages costs
        # over active channels rather than neutral held channels.
        if case.kind == 'snake':
            rate_scale, acceleration_scale = 14/4, 9/4
        elif case.kind == 'worm':
            rate_scale, acceleration_scale = 14/10, 9/5
        else:
            rate_scale, acceleration_scale = 1., 1.
        for speed in (5., 20., 50., 100., 200.):
            velocity = trace['filtered_velocity'][selected, lane]
            command = torch.zeros_like(velocity); command[:, 0] = speed*1e-3
            reward = velocity_reward('sustain_v4', velocity, command,
                trace['action_cost'][selected, lane]*rate_scale, trace['metrics'][selected, lane, 0],
                trace['metrics'][selected, lane, 1], trace['acceleration_cost'][selected, lane]*acceleration_scale, policy_dt=dt)
            replay[str(int(speed))] = dict(reward_sum=float(reward.sum()),
                reward_per_second=float(reward.sum())/(float(selected.sum())*dt))
    guard = {key: bool((value[:, lane] & valid).any()) for key, value in trace.get('guard_reasons', {}).items()}
    return dict(case=asdict(case), first_episode_complete=valid_episode,
        valid_duration_s=float(valid.sum())*dt, first_episode_only=True,
        tail_start_s=tail_start, required_tail_duration_s=tail_duration,
        duration_sufficient_for_gate=enough_time, positive_world_forward_progress=bool(world_forward),
        sustained_yaw_forward_gate=bool(passed),
        sustained_worm_forward_gate=bool(worm_passed),
        sustained_mixed_forward_gate=bool(rope_passed), guards=guard,
        whole=whole, tail=tail, windows_2s=windows, tail_windows_2s=tail_windows,
        old_reward_tail_replay=replay,
        replay_cost_mode='snake_active4' if case.kind == 'snake' else ('worm_active5' if case.kind == 'worm' else 'mixed_paired5_yaw4'))


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def run(args):
    from serpworm.envs.tensor_env import TensorEnv
    root = PKG_ROOT
    selected = cases()
    if args.suite != 'all':
        selected = [row for row in selected if row.kind in ({'static', 'pose'} if args.suite == 'static' else {args.suite})]
    if args.cases:
        unknown = set(args.cases)-{r.name for r in selected}
        if unknown: raise ValueError(f'Unknown or excluded cases: {sorted(unknown)}')
        selected = [row for row in selected if row.name in args.cases]
    if args.indices:
        selected = [selected[index] for index in args.indices]
    if not selected: raise ValueError('No cases selected')
    if len(selected) != len({row.name for row in selected}): raise ValueError('Duplicate cases selected')
    # Matching static controls are required before approving any teacher data.
    for neutral in sorted({row.neutral_deg for row in selected}):
        control = Case(f'neutral{neutral:g}', 'static', neutral)
        if control.name not in {row.name for row in selected}: selected.append(control)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = dict(**vars(args), selected_cases=[asdict(c) for c in selected],
        device_scope='CPU numerical oracle only; no GPU throughput or training proof' if args.device == 'cpu' else 'CUDA physics probe; no PPO or hardware validation',
        wheel_guard='2 mm certified finite-cylinder separation; sphere bound retained and separating-plane certificate refines false rejections',
        source_sha256=collect_source_hashes(root),
        input_sha256=collect_input_hashes(root),
        phase_convention='request sin(omega*t + j*delta), j ascending in topology; phase velocity dj/dt=-omega/delta. No assumed biological head label.',
        forward_definition='body vx uses normalized vector from plate 0 toward plate 9, matching TensorEnv.body_velocity; no hardware head label assumed',
        force_diagnostics='Sampled at each policy boundary, not substep peak. Prefer pre-reset snapshots supplied by environment.',
        gate='At least 20 s tail or last 2 periods, positive world displacement projected on initial forward axis, each 2 s mean body vx >=1 mm/s and >=2 actual yaw PP>=5 deg, full first episode, no guard; mixed also >=2 rope segments PP>=5 deg. This screens forward locomotion, not precise straight-line tracking.')
    manifest['command_scale'] = [.2, .01, .20]
    write(out/'config.json', manifest)
    freeze_sources(root, out, manifest)
    summaries, teacher_cases = [], {}
    groups = []
    for neutral in sorted({c.neutral_deg for c in selected}):
        rows = [c for c in selected if c.neutral_deg == neutral]
        groups.extend(rows[i:i+args.batch_size] for i in range(0, len(rows), args.batch_size))
    for batch_index, group in enumerate(groups):
        batchdir = out/f'batch_{batch_index:03d}'; batchdir.mkdir()
        env = TensorEnv(n=len(group), mode='mixed', noise=False, device=args.device, dt=args.dt,
            command_switching=False, servo_neutral_deg=group[0].neutral_deg,
            episode_seconds=max(20., args.seconds))
        env.cmd.zero_(); env.cmd[:, 0] = args.command_mm_s*1e-3
        env.history[:, :, -3:] = (env.cmd/env.command_scale)[:, None]
        env.obs = env.history.flatten(1).float().clone()
        initial = env.c.x[:, :10, :3].clone()
        active = torch.ones(len(group), dtype=torch.bool, device=env.device)
        rows, guard_rows = {}, {}
        start = time.monotonic()
        for step in range(round(args.seconds/env.policy_dt)):
            actor_observation = env.obs.clone()
            servo, yaw = requested_angles(step*env.policy_dt, group, env.device, env.dtype)
            action = actions_from_angles(servo, yaw, group[0].neutral_deg)
            _, reward, terminated, truncated, _, metrics = env.step(action)
            raw = env.last_physics_diagnostics
            forces = dict(tension=raw['tension'], normal_force=raw['support'],
                wheel_speed=raw['spin'], wheel_slip=raw['slip'], gap=raw['gap'], stop_force=raw['stop_force'])
            data = dict(actions=action, actor_observations=actor_observation, active=active.clone(), terminated=terminated, truncated=truncated,
                body_velocity=env.last_body_velocity, filtered_velocity=env.last_velocity,
                plate_positions_m=env.last_plate_positions, yaw_diagnostics=env.last_diagnostics,
                servo_diagnostics=env.last_servo_diagnostics, metrics=metrics,
                sphere_clearance=env.last_sphere_clearance,
                action_cost=env.last_action_rate_cost, acceleration_cost=env.last_action_acceleration_cost,
                **forces)
            for key, value in data.items(): rows.setdefault(key, []).append(value.clone())
            for key, value in env.last_guard_reasons.items(): guard_rows.setdefault(key, []).append(value.clone())
            active &= ~(terminated | truncated)
            if bool(env.c.failed): raise RuntimeError('Nonfinite physics; no gate result is valid')
            if (step+1) % 10 == 0:
                write(out/'status.json', dict(status='running', batch=batch_index,
                    cases=[c.name for c in group], integrated_s=(step+1)*env.policy_dt,
                    batch_wall_s=time.monotonic()-start))
        trace = {key: (torch.cat(values) if key == 'yaw_diagnostics' else torch.stack(values)).cpu() for key, values in rows.items()}
        trace.update(policy_dt_s=env.policy_dt, sensor_dt_s=env.sensor_dt,
            time_start_s=torch.arange(len(trace['active']), dtype=torch.float64)*env.policy_dt,
            initial_plate_positions_m=initial.cpu(), cases=[asdict(c) for c in group],
            guard_reasons={key: torch.stack(values).cpu() for key, values in guard_rows.items()},
            initial_plate9_minus_plate0_world_m=(initial[:, 9, :3]-initial[:, 0, :3]).cpu(),
            force_sample_time='interval_end_pre_reset')
        torch.save(trace, batchdir/'traces.pt')
        report = [summarize_trace(trace, lane, case, args.seconds) for lane, case in enumerate(group)]
        for lane, (case, item) in enumerate(zip(group, report)):
            if case.kind == 'snake':
                mode, gate = 'snake', item['sustained_yaw_forward_gate']
            elif case.kind == 'worm':
                mode, gate = 'worm', item['sustained_worm_forward_gate']
            else:
                mode, gate = 'mixed', item['sustained_mixed_forward_gate']
            teacher_cases[case.name] = dict(mode=mode,
                actor_observations=trace['actor_observations'][:, lane],
                teacher_actions=trace['actions'][:, lane], time_s=trace['time_start_s'],
                period_s=case.period_s, servo_neutral_deg=case.neutral_deg,
                command_vx_mm_s=args.command_mm_s,
                summary=dict(gate_passed=gate, reward_replay_better_than_stationary=False,
                    complete=item['first_episode_complete'], valid_duration_s=item['valid_duration_s'],
                    case_kind=case.kind, matched_command_vx_mm_s=args.command_mm_s,
                    gate_scope='First-episode persistent forward motion; not a learned policy or hardware validation'))
        summaries.extend(report)
        write(batchdir/'summary.json', report)
        write(out/'summary.json', dict(results=summaries, complete=False))
        del env
        if args.device == 'cuda': torch.cuda.empty_cache()
    by_name = {item['case']['name']: item for item in summaries}
    for name, item in by_name.items():
        reference = by_name[f"neutral{item['case']['neutral_deg']:g}"]
        scores = item['old_reward_tail_replay']
        controls = reference['old_reward_tail_replay']
        comparison = {speed: scores[speed]['reward_per_second'] > controls[speed]['reward_per_second']+1e-9
                      for speed in scores.keys() & controls.keys()}
        key = str(int(args.command_mm_s))
        teacher_cases[name]['summary']['reward_replay_better_than_stationary'] = bool(comparison.get(key, False))
        item['reward_replay_better_than_same_neutral_static'] = comparison
    dataset = dict(schema='fixed_wave_dataset_v1', source_sha256=manifest['source_sha256'],
        physics_source_sha256={k: manifest['source_sha256'][k] for k in ('physics/gpu_core.py', 'physics/wheel_stiction.py')},
        input_sha256=manifest['input_sha256'], physics_dt_s=args.dt, policy_dt_s=.2, history_span_s=1.8,
        command_scale=manifest['command_scale'],
        device=args.device, device_scope=manifest['device_scope'], cases=teacher_cases,
        teacher_scope='Fixed waves provide candidate motion data, not learned PPO results; only gate-passing and reward-improving complete cases are eligible.')
    torch.save(dataset, out/'teacher_data.pt')
    write(out/'summary.json', dict(results=summaries, complete=True,
        comparisons='Static/pose cases and waves share each neutral group; compare old_reward_tail_replay at matching command only.',
        scope=manifest['device_scope']))
    write(out/'status.json', dict(status='complete', cases=len(summaries)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--dt', type=float, default=.00025)
    parser.add_argument('--seconds', type=float, default=40.)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--command-mm-s', type=float, choices=(5., 20., 50., 100., 200.), default=20.)
    parser.add_argument('--suite', choices=('all', 'static', 'snake', 'worm', 'mixed', 'full'), default='all')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--list-cases', action='store_true')
    args = parser.parse_args(argv)
    if args.list_cases:
        print(json.dumps([dict(index=i, **asdict(c)) for i, c in enumerate(cases())], indent=2))
    else:
        if not args.out or not math.isfinite(args.seconds) or args.seconds <= 0 or not math.isclose(args.seconds/.2, round(args.seconds/.2)):
            parser.error('Require out and a positive duration divisible by 0.2 s')
        if not 1 <= args.batch_size <= 8 or not 0 < args.dt <= .001:
            parser.error('Batch must be 1..8 and dt must be in (0,.001]')
        try:
            run(args)
        except BaseException as error:
            out = Path(args.out)
            if out.is_dir():
                write(out/'status.json', dict(status='failed', error=repr(error), scope='No passed motion claim'))
            raise


if __name__ == '__main__':
    main()

"""Finite evaluation on CUDA; host conversion only after each complete case.

No training, and no implicit claim of hardware or robust tracking performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import math
import torch
from tensor_env import TensorEnv
from train_gpu import ActorCritic, write


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--seconds', type=float, default=12.)
    p.add_argument('--out', required=True)
    p.add_argument('--cases', nargs='+', choices=['stop', 'straight', 'left', 'right', 'stop_after_forward'])
    args = p.parse_args()
    if args.seconds <= 0:
        p.error('Duration must be positive')
    torch.set_num_threads(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    # Only load a checkpoint from this trusted project, never an untrusted pickle.
    data = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    kind = data['config']['kind']
    plate_solver = data['config'].get('plate_solver', 'legacy')
    profile = data['config'].get('control_profile', 'legacy')
    write(out/'config.json', {**vars(args), 'kind': kind, 'seed': 123, 'sensor_noise': False, 'plate_solver': plate_solver,
                             'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()})
    n = 3 if kind == 'direct' else 2
    env = TensorEnv(kind, n=n, seed=123, noise=False, reward_version=data['config'].get('reward', 'legacy'), plate_solver=plate_solver,
                    control_profile=profile, command_switching=False)
    model = ActorCritic(env.obs_size).cuda()
    model.load_state_dict(data['model']); model.eval()
    cases = [('stop', 0., 0.), ('straight', .004, 0.), ('left', .004, .02), ('right', .004, -.02)] if kind == 'direct' else [('straight', 0., 0.), ('left', 0., .45), ('right', 0., -.45)]
    if kind == 'direct':
        cases.append(('stop_after_forward', .004, 0.))
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
        if not cases:
            raise ValueError('No requested cases apply to this controller')
    allmask = torch.ones(n, device='cuda', dtype=torch.bool)
    summaries, traces = [], {}
    try:
        for name, speed, turn in cases:
            write(out/'status.json', {'status': 'evaluating', 'case': name})
            env.reset(allmask)
            command = env.tensor([speed, 0., turn])
            if kind == 'direct':
                env.cmd.copy_(command.expand(n, -1))
            else:
                x, _, _, _, forward, left = env.geometry()
                env.goals.copy_(x[:, 9, :2]+.08*(math.cos(turn)*forward+math.sin(turn)*left))
            obs = env.packet(allmask)
            active = allmask.clone()
            rows, masks, positions, diagnostics, commands, body_velocities = [], [], [], [], [], []
            with torch.no_grad():
                for k in range(round(args.seconds/.1)):
                    if name == 'stop_after_forward' and k == round(args.seconds/.2):
                        command.zero_()
                        env.cmd.zero_()
                        env.history[:, -1, -3:].zero_()
                        obs = env.history.flatten(1).float().clone()
                    action = model.actor(obs)
                    action[1].zero_()
                    if n == 3:
                        # Same physical 15 deg baseline across 20/25 deg policy ranges.
                        moving = speed and not (name == 'stop_after_forward' and k >= round(args.seconds/.2))
                        action[2] = math.radians(15)/env.limit*torch.sin(env.tensor(2*math.pi*k*.1/4)-(3-torch.arange(4, device='cuda'))*math.pi/2) if moving else 0.
                    obs, reward, term, trunc, terminal_obs, metrics = env.step(action)
                    # The tensor environment retains pre-reset diagnostics below.
                    rows.append(torch.cat((reward[:, None], env.last_velocity, metrics), -1))
                    masks.append(active.clone())
                    positions.append(env.last_plate_positions.clone())
                    diagnostics.append(env.last_diagnostics.clone())
                    commands.append(env.last_command.clone())
                    body_velocities.append(env.last_body_velocity.clone())
                    active &= ~(term | trunc)
            if bool(env.c.failed):
                raise RuntimeError('Invalid physics during evaluation')
            record, mask = torch.stack(rows), torch.stack(masks)
            if not bool(torch.isfinite(record).all()):
                raise RuntimeError('Nonfinite evaluation')
            diagnostic = torch.cat(diagnostics).cpu()
            command_trace, body_velocity = torch.stack(commands).cpu(), torch.stack(body_velocities).cpu()
            traces[name] = {'record': record.cpu(), 'active': mask.cpu(), 'plate_positions_m': torch.stack(positions).cpu(),
                            'diagnostics_50hz': diagnostic, 'command': command_trace, 'body_velocity': body_velocity,
                            'diagnostic_fields': ['requested_target_rad', 'applied_target_rad', 'actual_angle_rad',
                                                  'actual_rate_rad_s', 'last_substep_torque_nm', 'last_substep_torque_cap_nm']}
            for i in range(n):
                rr = record[:, i][mask[:, i]]
                controller = 'learned' if i == 0 else ('zero_action' if i == 1 and n == 3 else 'fixed_wave' if i == 2 else 'constant_tonic')
                item = {'case': name, 'controller': controller, 'samples': len(rr),
                        'failed': bool((rr[:, 9] > 0).any()), 'success': bool((rr[:, 10] > 0).any()),
                        'mean_velocity': rr[:, 1:4].mean(0).cpu().tolist(),
                        'mean_slip_m_s': float(rr[:, 4].mean()), 'work_proxy_j': float(rr[:, 5].sum()),
                        'min_clearance_m': float(rr[:, 6].min()), 'peak_joint_deg': float(rr[:, 7].max()*180/math.pi)}
                if kind == 'direct':
                    valid = mask[:, i].cpu()
                    target = command_trace[:, i][valid]
                    item['velocity_rmse'] = ((rr[:, 1:4].cpu()-target).square().mean(0).sqrt()).tolist()
                    item['body_velocity_rmse'] = ((body_velocity[:, i][valid]-target).square().mean(0).sqrt()).tolist()
                    item['mean_body_velocity'] = body_velocity[:, i][valid].mean(0).tolist()
                    d = diagnostic[:, i][valid.repeat_interleave(5)]
                    item['target_tracking_rmse_deg'] = float((d[:, 1]-d[:, 2]).square().mean().sqrt()*180/math.pi)
                    item['requested_target_peak_deg'] = float(d[:, 0].abs().max()*180/math.pi)
                    item['applied_target_peak_deg'] = float(d[:, 1].abs().max()*180/math.pi)
                    item['sampled_torque_saturation_fraction'] = float(((d[:, 4].abs() >= .99*d[:, 5]) & (d[:, 5] > 1e-6)).double().mean())
                    if name == 'stop_after_forward':
                        stop_mask = valid & (command_trace[:, i, 0] == 0)
                        item['stop_phase_mean_body_velocity'] = body_velocity[:, i][stop_mask].mean(0).tolist() if stop_mask.any() else None
                    # Finite-window frequency diagnostic, not a steady-state claim.
                    q = d[min(200, len(d)//2):, 2]
                    if len(q) >= 20:
                        t = torch.arange(len(q), dtype=q.dtype);t -= t.mean()
                        q = q-q.mean(0)-t[:, None]*(t[:, None]*q).sum(0)/t.square().sum()
                        power = torch.fft.rfft(q*torch.hann_window(len(q), periodic=False, dtype=q.dtype)[:, None], dim=0).abs().square()
                        frequency = torch.fft.rfftfreq(len(q), .02)
                        item['actual_angle_dominant_hz'] = frequency[power[1:].argmax(0)+1].tolist()
                        item['actual_angle_power_fraction_ge_2hz'] = (power[frequency >= 2].sum(0)/power[1:].sum(0).clamp_min(1e-20)).tolist()
                else:
                    item['final_goal_distance_m'] = float(rr[-1, 8])
                summaries.append(item)
            torch.save(traces, out/'traces.pt')
            write(out/'summary.json', {'kind': kind, 'results': summaries, 'scope': 'single seed, fixed friction, no noise, finite pilot evaluation; different tasks cannot be ranked together'})
        write(out/'status.json', {'status': 'evaluation_complete', 'cases': len(cases)})
    except BaseException as e:
        write(out/'status.json', {'status': 'failed', 'error': repr(e)})
        raise

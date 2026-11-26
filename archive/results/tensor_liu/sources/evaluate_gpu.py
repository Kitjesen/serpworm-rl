"""Finite evaluation on CUDA; host conversion only after each complete case.

No training, and no implicit claim of hardware or robust tracking performance.
"""
import argparse
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
    args = p.parse_args()
    if args.seconds <= 0:
        p.error('Duration must be positive')
    torch.set_num_threads(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    # Only load a checkpoint from this trusted project, never an untrusted pickle.
    data = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    kind = data['config']['kind']
    n = 3 if kind == 'direct' else 2
    env = TensorEnv(kind, n=n, seed=123, noise=False)
    model = ActorCritic(env.obs_size).cuda()
    model.load_state_dict(data['model']); model.eval()
    cases = [('stop', 0., 0.), ('straight', .004, 0.), ('left', .004, .02), ('right', .004, -.02)] if kind == 'direct' else [('straight', 0., 0.), ('left', 0., .45), ('right', 0., -.45)]
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
            rows, masks, positions = [], [], []
            with torch.no_grad():
                for k in range(round(args.seconds/.1)):
                    action = model.actor(obs)
                    action[1].zero_()
                    if n == 3:
                        action[2] = .75*torch.sin(env.tensor(2*math.pi*k*.1/4)-(3-torch.arange(4, device='cuda'))*math.pi/2) if speed else 0.
                    obs, reward, term, trunc, terminal_obs, metrics = env.step(action)
                    # The tensor environment retains pre-reset diagnostics below.
                    rows.append(torch.cat((reward[:, None], env.last_velocity, metrics), -1))
                    masks.append(active.clone())
                    positions.append(env.last_plate_positions.clone())
                    active &= ~(term | trunc)
            if bool(env.c.failed):
                raise RuntimeError('Invalid physics during evaluation')
            record, mask = torch.stack(rows), torch.stack(masks)
            if not bool(torch.isfinite(record).all()):
                raise RuntimeError('Nonfinite evaluation')
            traces[name] = {'record': record.cpu(), 'active': mask.cpu(), 'plate_positions_m': torch.stack(positions).cpu()}
            for i in range(n):
                rr = record[:, i][mask[:, i]]
                controller = 'learned' if i == 0 else ('zero_action' if i == 1 and n == 3 else 'fixed_wave' if i == 2 else 'constant_tonic')
                item = {'case': name, 'controller': controller, 'samples': len(rr),
                        'failed': bool((rr[:, 9] > 0).any()), 'success': bool((rr[:, 10] > 0).any()),
                        'mean_velocity': rr[:, 1:4].mean(0).cpu().tolist(),
                        'mean_slip_m_s': float(rr[:, 4].mean()), 'work_proxy_j': float(rr[:, 5].sum()),
                        'min_clearance_m': float(rr[:, 6].min()), 'peak_joint_deg': float(rr[:, 7].max()*180/math.pi)}
                if kind == 'direct':
                    item['velocity_rmse'] = ((rr[:, 1:4]-command).square().mean(0).sqrt()).cpu().tolist()
                else:
                    item['final_goal_distance_m'] = float(rr[-1, 8])
                summaries.append(item)
            torch.save(traces, out/'traces.pt')
            write(out/'summary.json', {'kind': kind, 'results': summaries, 'scope': 'single seed, fixed friction, no noise, finite pilot evaluation; different tasks cannot be ranked together'})
        write(out/'status.json', {'status': 'evaluation_complete', 'cases': len(cases)})
    except BaseException as e:
        write(out/'status.json', {'status': 'failed', 'error': repr(e)})
        raise

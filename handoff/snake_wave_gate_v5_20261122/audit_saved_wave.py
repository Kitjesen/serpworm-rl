"""Recompute archived motion and replay v4 reward; does not simulate new motion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from rewards import velocity_reward


def audit(source):
    data = torch.load(source, map_location='cpu', weights_only=False)['straight20']
    dt = .1
    count, lanes = data['active'].shape
    assert lanes == 3 and count == 120
    assert data['diagnostics_50hz'].shape == (count*5, 3, 6, 4)
    assert data['diagnostic_fields'][2] == 'actual_angle_rad'
    rows = []
    # Archived evaluator explicitly names these lanes. Only the pure-yaw wave
    # and zero-action control have reconstructible requested tendon commands.
    for lane, name in ((1, 'zero_action'), (2, 'fixed_yaw_wave')):
        velocity = data['body_velocity'][:, lane].double()
        mask = data['active'][:, lane].bool()
        tail = mask & (torch.arange(count)*dt >= 6.)
        assert int(tail.sum()) == 60
        yaw = data['diagnostics_50hz'][:, lane].double()
        assert yaw.shape[-2:] == (6, 4)
        samples = yaw.reshape(count, 5, 6, 4)
        requested = samples[:, -1, 0]/math.radians(45)
        applied = samples[:, -1, 1]/math.radians(45)
        delta = requested-torch.cat((torch.zeros_like(requested[:1]), requested[:-1]))
        second = delta-torch.cat((torch.zeros_like(delta[:1]), delta[:-1]))
        applied_delta = applied-torch.cat((torch.zeros_like(applied[:1]), applied[:-1]))
        # Mixed env averages over 14 applied channels / 9 requested channels;
        # its ten tendon targets and five tendon actions are constant here.
        action_cost = applied_delta.square().sum(-1)/14
        acceleration_cost = second.square().sum(-1)/9
        recorded = data['record'][:, lane].double()
        slip, work = recorded[:, 4], recorded[:, 5]
        filtered = torch.zeros_like(velocity)
        previous = torch.zeros(3, dtype=velocity.dtype)
        for t in range(count):
            previous = previous+(1-math.exp(-dt/.2))*(velocity[t]-previous)
            filtered[t] = previous
        replays = {}
        for speed in (5., 20., 50., 100., 200.):
            command = torch.zeros_like(velocity)
            command[:, 0] = speed/1000
            reward = velocity_reward('sustain_v4', filtered, command,
                action_cost, slip, work, acceleration_cost, policy_dt=dt)
            replays[str(int(speed))] = dict(tail_reward_per_s=float(reward[tail].sum())/6.,
                tail_reward_sum=float(reward[tail].sum()))
        windows = []
        for begin in (6., 8., 10.):
            selected = mask & (torch.arange(count)*dt >= begin) & (torch.arange(count)*dt < begin+2.)
            windows.append(dict(start_s=begin, end_s=begin+2., mean_vx_mm_s=float(velocity[selected, 0].mean())*1000))
        actual = samples[tail, :, 2, :].reshape(-1, 4)
        rows.append(dict(controller=name, tail_mean_vx_mm_s=float(velocity[tail, 0].mean())*1000,
            tail_min_vx_mm_s=float(velocity[tail, 0].min())*1000,
            tail_forward_integral_mm=float(velocity[tail, 0].sum())*dt*1000,
            actual_yaw_pp_deg=((actual.amax(0)-actual.amin(0))*180/math.pi).tolist(),
            tail_windows=windows, reward_replay=replays,
            any_guard_failure=bool((recorded[mask, 9] > 0).any())))
    reward_better = {key: rows[1]['reward_replay'][key]['tail_reward_per_s'] > rows[0]['reward_replay'][key]['tail_reward_per_s']
                     for key in rows[0]['reward_replay']}
    return dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_kind='archived 12 s CUDA trace; not new v5 simulation', results=rows,
        fixed_wave_reward_better_than_zero=reward_better,
        scope='Reanalysis only. Same physics sources, older initialization and 10 Hz action cadence.',
        limitations=['No new 40 s validation.', 'Replay uses archived sampled slip/yaw work and reconstructed known pure-yaw commands.',
                    'v4 reward is replayed on filtered archived body velocity; PPO did not generate the fixed wave.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    result = audit(Path(args.source))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

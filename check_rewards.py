"""Frozen reward equivalence and explicit motion/stop ranking checks."""
import argparse
import json
from pathlib import Path
import torch
from rewards import velocity_reward


def check(device):
    torch.manual_seed(5)
    velocity = torch.randn(32, 3, device=device, dtype=torch.float64)*.01
    command = torch.randn_like(velocity)*.01
    costs = [torch.rand(32, device=device, dtype=torch.float64)*.01 for _ in range(3)]
    ev, ew = velocity[:, :2]-command[:, :2], velocity[:, 2]-command[:, 2]
    expected = .1*((-ev.square().sum(-1)/.006**2).exp()+.5*(-(ew/.025).square()).exp()-.02*costs[0]-.02*costs[1]/.01)-.01*costs[2]
    actual = velocity_reward('legacy', velocity, command, *costs)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    vel = torch.tensor([[0,0,0],[.002,0,0],[.004,0,0],[.008,0,0],[-.004,0,0],[.004,.004,0],[.004,0,.02]], device=device, dtype=torch.float64)
    cmd = torch.zeros_like(vel);cmd[:,0]=.004
    zero = torch.zeros(7, device=device, dtype=torch.float64)
    new = velocity_reward('tracking_v2', vel, cmd, zero, zero, zero)
    torch.testing.assert_close(new, torch.tensor([0,.075,.1,0,0,.075,.05], device=device, dtype=torch.float64))
    # A stop command rewards stopping; turn commands reward the requested sign.
    stop = velocity_reward('tracking_v2', vel, torch.zeros_like(cmd), zero, zero, zero)
    assert stop[0] > stop[2]
    cmd[:,2]=.02
    left = velocity_reward('tracking_v2', vel, cmd, zero, zero, zero)
    cmd[:,2]=-.02
    right = velocity_reward('tracking_v2', vel, cmd, zero, zero, zero)
    assert left[-1] > right[-1]
    old_rest = float(.1*(torch.exp(torch.tensor(-(.004/.006)**2))+.5))
    return {'passed':True, 'device':device, 'legacy_equivalence':True,
            'new_tracking_scores_without_penalties':new.cpu().tolist(),
            'legacy_stationary_score_4mm_s':old_rest, 'legacy_perfect_score':.15,
            'new_stationary_score_4mm_s':0., 'new_perfect_score':.1,
            'scope':'Reward arithmetic and ranking only; causal control benefit needs matched training/evaluation.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--out',default='results/reward_checks.json');args=p.parse_args()
    report=check(args.device);out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

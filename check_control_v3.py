"""Reward ordering, actuator passband, body-frame and optional CUDA reset checks."""
import argparse
import json
import math
from pathlib import Path
import torch
from rewards import velocity_reward
from tensor_env import TensorEnv, STATE
from train_gpu import ActorCritic, restore_training


def check_resume(path):
    data=torch.load(path,map_location='cpu',weights_only=False)
    args=argparse.Namespace(**data['config']);args.rollouts=data['report']['rollouts']+1
    model=ActorCritic(data['obs_size']);optimizer=torch.optim.Adam(model.parameters())
    report=restore_training(path,model,optimizer,args,data['obs_size'])
    assert report==data['report']
    assert all(torch.equal(value,data['model'][key]) for key,value in model.state_dict().items())
    restored=optimizer.state_dict()
    assert restored['param_groups']==data['optimizer']['param_groups']
    for key,value in restored['state'].items():
        for field,tensor in value.items():
            assert torch.equal(tensor,data['optimizer']['state'][key][field])
    args.reward='invalid'
    try:
        restore_training(path,model,optimizer,args,data['obs_size'])
    except ValueError:
        return True
    raise AssertionError('Incompatible reward checkpoint accepted')


def checks(device):
    torch.set_num_threads(1)
    dtype = torch.float64
    vel = torch.tensor([[.004,0,0], [0,0,0], [.008,0,0], [.012,0,0], [.02,0,0]], dtype=dtype)
    cmd = torch.tensor([[.004,0,0]], dtype=dtype).expand_as(vel)
    zero = torch.zeros(len(vel), dtype=dtype)
    score = velocity_reward('tracking_v3', vel, cmd, zero, zero, zero, zero)
    assert score[0] > score[1] == score[2] > score[3] > score[4] > 0
    stop = velocity_reward('tracking_v3', vel, torch.zeros_like(cmd), zero, zero, zero, zero)
    assert stop[1] > stop[0] > stop[2] > stop[3]
    turn = torch.tensor([[.004,0,.02], [.004,0,0], [.004,0,-.02]], dtype=dtype)
    target = turn[:1].expand_as(turn); z = zero[:3]
    turn_score = velocity_reward('tracking_v3', turn, target, z, z, z, z)
    assert turn_score[0] > turn_score[1] > turn_score[2]
    smooth = velocity_reward('tracking_v3', vel[:1], cmd[:1], zero[:1], zero[:1], zero[:1], zero[:1])
    chatter = velocity_reward('tracking_v3', vel[:1], cmd[:1], torch.ones(1)*4, zero[:1], zero[:1], torch.ones(1)*16)
    assert smooth > chatter+.05
    # A 4 s wave should pass; 5 Hz alternating setpoints should be attenuated.
    ratios = []
    for frequency in (.25, 5.):
        applied = 0.; values = []; references = []
        for k in range(1000):
            sample_time = (k//5)*.1
            desired = math.radians(15)*(math.sin(2*math.pi*frequency*sample_time) if frequency < 5 else (-1)**(k//5))
            applied += max(-.02, min(.02, (1-math.exp(-.02/.2))*(desired-applied)))
            if k >= 500:
                values.append(applied); references.append(desired)
        ratio = float(torch.tensor(values).square().mean().sqrt()/torch.tensor(references).square().mean().sqrt())
        ratios.append(ratio)
    assert ratios[0] > .85 and ratios[1] < .3, ratios
    # A rigid translation along the head direction has the commanded sign.
    before = torch.zeros(1,10,3,dtype=dtype);before[0,:,0] = -torch.arange(10)*.1
    after = before.clone();after[:,:,0] -= .0004
    torch.testing.assert_close(TensorEnv.body_velocity(before, after), torch.tensor([[.004,0,0]],dtype=dtype))
    after = before.clone(); center=before.mean(1,keepdim=True)
    angle=.002; rotation=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]],dtype=dtype)
    after[:,:,:2]=(before[:,:,:2]-center[:,:,:2])@rotation.T+center[:,:,:2]
    torch.testing.assert_close(TensorEnv.body_velocity(before,after),torch.tensor([[0,0,.02]],dtype=dtype),atol=1e-12,rtol=1e-10)
    report=dict(reward_scores=score.tolist(), turn_scores=turn_score.tolist(), filter_rms_gain_025_and_5hz=ratios,
                body_frame_checks=True, device=device)
    if device == 'cuda':
        env=TensorEnv('direct',n=2,noise=False,reward_version='tracking_v3',control_profile='smooth_v3')
        assert env.obs_size==420
        env.count.fill_(245); env.cmd[:,0]=.006
        obs,reward,term,trunc,final,metrics=env.step(torch.full((2,4),.1,device='cuda'))
        assert torch.all(env.cmd[:,0] != .006) and torch.all(env.last_command[:,0] == .006)
        torch.testing.assert_close(obs.reshape(2,10,42)[:,-1,-3:],(env.cmd/env.command_scale).float())
        assert env.last_diagnostics.shape==(5,2,6,4) and torch.isfinite(env.last_diagnostics).all()
        names=('previous_actions','previous_action_delta','targets','history','cmd')
        saved={name:getattr(env,name)[1].clone() for name in names}
        physical={name:getattr(env.c,name)[1].clone() for name in STATE}
        env.reset(torch.tensor([True,False],device='cuda'))
        assert all(torch.equal(value,getattr(env,name)[1]) for name,value in saved.items())
        assert all(torch.equal(value,getattr(env.c,name)[1]) for name,value in physical.items())
        env.count.fill_(995)
        obs,reward,term,trunc,final,metrics=env.step(torch.zeros(2,4,device='cuda'))
        assert trunc.all() and not term.any() and (env.count==0).all()
        model=ActorCritic(420).cuda(); optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        before_weight=model.actor[0].weight.detach().clone()
        action,logp,entropy,value=model(obs)
        loss=-logp.mean()+value.square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
        assert not torch.equal(before_weight,model.actor[0].weight)
        report.update(masked_reset=True, command_switch=True, timeout=True, gpu_update=True)
    report['passed']=True
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True)
    p.add_argument('--resume-checkpoint',help='Optional trusted project checkpoint for weights/Adam/counter restoration check')
    args=p.parse_args();report=checks(args.device);out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    if args.resume_checkpoint:report['resume_weights_adam_counters']=check_resume(args.resume_checkpoint)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

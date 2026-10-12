"""Finite paired old-policy filter diagnostic and loaded 15/20/25-degree wave checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from tensor_env import TensorEnv
from train_gpu import ActorCritic, write


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    env=TensorEnv('direct',n=8,seed=123,noise=False,reward_version='tracking_v2',command_switching=False)
    data=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    model=ActorCritic(380).cuda();model.load_state_dict(data['model']);model.eval()
    # First four environments preserve the old actor's 20-degree normalization.
    env.limit=env.tensor([[20],[20],[20],[20],[20],[25],[25],[25]])*math.pi/180
    gain=1-math.exp(-.02/.2)
    env.target_gain=env.tensor([[1],[gain],[1],[gain],[1],[gain],[gain],[gain]])
    env.cmd.copy_(env.tensor([[.004,0,0],[.004,0,0],[0,0,0],[0,0,0],[.004,0,0],[.004,0,0],[.004,0,0],[.004,0,0]]))
    obs=env.packet(torch.ones(8,device='cuda',dtype=torch.bool))
    labels=['r96_straight_raw','r96_straight_filtered','r96_stop_raw','r96_stop_filtered',
            'fixed15_raw','fixed15_filtered','fixed20_filtered','fixed25_filtered']
    rows=[];body=[];point=[];masks=[];active=torch.ones(8,device='cuda',dtype=torch.bool)
    position0=env.c.x[:,:10,:3].mean(1).clone();positions=[]
    with torch.no_grad():
        for k in range(80):
            action=model.actor(obs)
            wave=torch.sin(env.tensor(2*math.pi*k*.1/4)-(3-torch.arange(4,device='cuda'))*math.pi/2)
            action[4:]=env.tensor([[15/20],[15/25],[20/25],[1.]])*wave
            obs,reward,term,trunc,_,metrics=env.step(action)
            rows.append(env.last_diagnostics.clone());body.append(env.last_body_velocity.clone());point.append(env.last_velocity.clone())
            positions.append(env.last_plate_positions.mean(1).clone());masks.append(active.clone());active &= ~(term|trunc)
            if (k+1)%20==0:print(json.dumps({'simulated_s':(k+1)*.1}),flush=True)
    diagnostics=torch.cat(rows).cpu().numpy();velocity=torch.stack(body).cpu().numpy();valid=torch.stack(masks).cpu().numpy()
    position=torch.stack(positions).cpu().numpy();position0=position0.cpu().numpy()
    np.savez_compressed(out/'diagnostics.npz',diagnostics=diagnostics,body_velocity=velocity,
                        legacy_filtered_velocity=torch.stack(point).cpu().numpy(),valid=valid,position=position,position0=position0)
    results=[]
    for i,label in enumerate(labels):
        d=diagnostics[:,i][np.repeat(valid[:,i],5)];n=len(d);q=d[n//2:,2]
        t=np.arange(len(q))*.02
        qd=q-np.stack([np.polyval(np.polyfit(t,q[:,j],1),t) for j in range(4)],1)
        power=abs(np.fft.rfft(qd*np.hanning(len(t))[:,None],axis=0))**2;frequency=np.fft.rfftfreq(len(t),.02)
        error=(velocity[:,i]-np.array([0 if 'stop' in label else .004,0,0]))[valid[:,i]]
        item=dict(case=label,completed_8s=bool(active[i]),body_rmse=np.sqrt((error**2).mean(0)).tolist(),
                  mean_body_velocity=velocity[:,i][valid[:,i]].mean(0).tolist(),
                  forward_cm=float(-(position[valid[:,i],i][-1,0]-position0[i,0])*100),
                  requested_peak_deg=float(abs(d[:,0]).max()*180/math.pi),applied_peak_deg=float(abs(d[:,1]).max()*180/math.pi),
                  actual_peak_deg=float(abs(d[:,2]).max()*180/math.pi),
                  target_rmse_deg=float(np.sqrt(((d[:,1]-d[:,2])**2).mean())*180/math.pi),
                  actual_pp_deg=(np.ptp(q,axis=0)*180/math.pi).tolist(),
                  high_frequency_power_fraction=(power[frequency>=2].sum(0)/np.maximum(power[1:].sum(0),1e-20)).tolist(),
                  actual_dominant_hz=frequency[power[1:].argmax(0)+1].tolist(),
                  sampled_torque_saturation_fraction=float(((abs(d[:,4])>=.99*d[:,5])&(d[:,5]>1e-6)).mean()))
        results.append(item)
    report=dict(results=results,checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                physics_failed=bool(env.c.failed),profile25_wave_passed=bool(active[7]),
                scope='8 s frozen-physics diagnostic; same old actor for filter pairs, no retraining, no hardware safety claim')
    write(out/'report.json',report);print(json.dumps(report),flush=True)

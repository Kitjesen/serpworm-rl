"""Finite post-training evaluation job. Does not schedule a future chat turn."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from train import Env,write

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('kind',choices=['direct','liu','liu_foc']);p.add_argument('--wait',action='store_true');a=p.parse_args()
run=root/('pilot_'+a.kind)
deadline=time.monotonic()+2400
while a.wait:
    status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    if status.get('status')=='failed':raise RuntimeError('Training failed: '+status.get('error',''))
    if status.get('status')=='pilot_complete':break
    if time.monotonic()>deadline:raise TimeoutError('Training did not finish within evaluation dependency budget')
    time.sleep(10)
out=run/'evaluation';out.mkdir(exist_ok=False)
torch.set_num_threads(1)
write(out/'status.json',dict(status='initializing'))
try:
    n=3 if a.kind=='direct' else 2
    env=Env(a.kind,n,123);env.noise=False
    model=PPO.load(run/'policy.zip',device='cpu')
    cases=[('stop',[0,0,0]),('straight',[.004,0,0]),('left',[.004,0,.02]),('right',[.004,0,-.02])] if a.kind=='direct' else [('straight',[0,0,0]),('left',[0,0,.45]),('right',[0,0,-.45])]
    records=[];poses=[];summaries=[]
    for name,command in cases:
        env.override=np.array(command);obs=env.reset();active=np.ones(n,bool)
        rows=[]
        for k in range(120):
            actions=model.predict(obs,deterministic=True)[0]
            actions[1]=0
            if n==3:actions[2]=0 if command[0]==0 else .75*np.sin(2*np.pi*k*.1/4-(3-np.arange(4))*np.pi/2)
            obs,r,done,infos=env.step(actions)
            for i in range(n):
                if active[i]:
                    row=dict(case=name,env=i,controller='learned' if i==0 else ('zero_action' if i==1 and n==3 else 'fixed_wave' if i==2 else 'constant_tonic'),reward=float(r[i]),**{key:value for key,value in infos[i].items() if key not in ('terminal_observation','episode')})
                    records.append(row);rows.append(row)
            if k%5==0 and active[0] and not done[0]:poses.append((name,(k+1)*.1,env.c.x[0].cpu().numpy().copy()))
            active&=~done
            if k%20==0:write(out/'status.json',dict(status='evaluating',case=name,physical_s=(k+1)*.1))
            if not active.any():break
        for i in range(n):
            rr=[r for r in rows if r['env']==i]
            velocity=np.array([r['velocity'] for r in rr])
            summary=dict(case=name,controller=rr[0]['controller'],samples=len(rr),failed=any(r['failed'] for r in rr),success=any(r['success'] for r in rr),mean_velocity=velocity.mean(0).tolist(),mean_slip_m_s=float(np.mean([r['slip_m_s'] for r in rr])),peak_joint_deg=float(np.max(np.abs([r['joint_deg'] for r in rr]))),work_proxy_j=sum(r['yaw_work_proxy_j'] for r in rr))
            if a.kind=='direct':summary['velocity_rmse']=np.sqrt(np.mean((velocity-command)**2,axis=0)).tolist()
            else:summary['final_goal_distance_m']=rr[-1]['goal_distance_m']
            summaries.append(summary)
    write(out/'records.json',records);write(out/'summary.json',dict(results=summaries,scope='single seed, fixed mu=.8, no observation noise, 12s per case; pilot comparison, not robustness or convergence proof',cross_branch_comparison='direct velocity and Liu visual-goal tasks differ; their rewards/success rates must not be ranked against each other'))
    if poses:np.savez_compressed(out/'learned_poses.npz',case=np.array([p[0] for p in poses]),time_s=[p[1] for p in poses],poses=np.array([p[2] for p in poses]))
    write(out/'status.json',dict(status='evaluation_complete',cases=len(cases)))
except BaseException as e:
    write(out/'status.json',dict(status='failed',error=repr(e)));raise

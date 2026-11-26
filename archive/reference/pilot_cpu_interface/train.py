"""Independent GPU pilots: direct sensor velocity policy; Liu-inspired visual goals.

Only four inter-segment yaw actuators move. Fixed-frequency Liu stage uses PPO
on tonic inputs; it is NOT the later option-critic frequency-learning stage.
"""
import argparse
from collections import deque
import hashlib
import json
import math
import shutil
from pathlib import Path
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from gpu_core import GPUCore, matrix
from matsuoka import Matsuoka

ROOT=Path(__file__).resolve().parent
LIMIT=math.radians(20)
STATE=('x','v','servo','target','yaw','spin','angle','memory','tension','support')


def write(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


class Env(VecEnv):
    def __init__(self,kind,n=2,seed=7,dt=.00025):
        self.kind,self.render_mode=kind,None
        self.rng=np.random.default_rng(seed)
        self.c=GPUCore(5,n,dt=dt,yaw_motor_speed=2.,strain_jacobian=True)
        self.c.capture(4)
        for _ in range(25): self.c.advance_steps(round(.02/dt))
        torch.cuda.synchronize()
        if bool(self.c.failed) or not bool(torch.isfinite(self.c.x).all()): raise RuntimeError('Passive settling failed')
        self.initial={k:getattr(self.c,k).clone() for k in STATE}
        self.cpg=Matsuoka(n,c=.75 if kind=='liu_foc' else 0.)
        self.count=np.zeros(n,int);self.targets=np.zeros((n,4));self.actions=np.zeros((n,4))
        self.previous_actions=np.zeros((n,4));self.cmd=np.zeros((n,3));self.goals=np.zeros((n,2))
        self.total=np.zeros(n);self.filtered_velocity=np.zeros((n,3));self.prev_theta=np.zeros(n)
        self.visual_previous=np.zeros((n,10,3));self.visual_filtered=np.zeros((n,2))
        self.delayed=[deque(maxlen=2) for _ in range(n)]
        self.previous_angles=np.zeros((n,14));self.dq=np.zeros((n,14));self.sensor=np.zeros((n,31))
        self.history=[deque(maxlen=10) for _ in range(n)]
        self.frame_size=38 if kind=='direct' else 14
        self.obs_size=380 if kind=='direct' else 14
        self.obs=np.zeros((n,self.obs_size),np.float32)
        self.override=None;self.noise=True;self.record=False;self.records=[]
        super().__init__(n,gym.spaces.Box(-np.inf,np.inf,(self.obs_size,),np.float32),gym.spaces.Box(-1,1,(4,),np.float32))

    def geometry(self):
        x=self.c.x[:,:10].cpu().numpy();v=self.c.v[:,:10].cpu().numpy()
        R=matrix(self.c.x[:,:10,3:]).cpu().numpy()
        relative=R[:,1:9:2].transpose(0,1,3,2)@R[:,2:10:2]
        q=np.arctan2(relative[...,1,0],relative[...,0,0])
        forward=-R[:,4,:2,0];forward/=np.linalg.norm(forward,axis=1)[:,None]
        left=np.c_[-forward[:,1],forward[:,0]]
        return x,v,R,q,forward,left

    def sensor_update(self,reset=()):
        x,v,R,q,f,l=self.geometry()
        angles=np.c_[self.c.servo.cpu().numpy(),q]
        gyro=np.einsum('bij,bj->bi',R[:,4].transpose(0,2,1),v[:,4,3:])*[-1,-1,1]
        raw=np.c_[angles,gyro]
        if self.noise: raw+=self.rng.normal(size=raw.shape)*np.r_[np.full(14,.002),np.full(3,.01)]
        for i in range(self.num_envs):
            if i in reset:
                self.delayed[i].clear();self.delayed[i].append(raw[i].copy());self.previous_angles[i]=raw[i,:14];self.dq[i]=0
            self.delayed[i].append(raw[i].copy());packet=self.delayed[i][0]
            self.dq[i]+=.25*((packet[:14]-self.previous_angles[i])/.02-self.dq[i])
            self.previous_angles[i]=packet[:14]
            self.sensor[i]=np.r_[packet[:14]/np.r_[np.full(10,np.pi),np.full(4,LIMIT)],self.dq[i]/2,packet[14:]]
        return x,v,R,q,f,l

    def packet(self,reset=()):
        if self.kind=='direct':
            # Pure function of encoder/gyro packets, command, applied command history.
            frames=np.c_[self.sensor,self.targets/LIMIT,self.cmd/[.01,.01,.03]]
            assert frames.shape[1]==38
            for i in range(self.num_envs):
                if i in reset:
                    self.history[i].clear();self.history[i].extend([frames[i].copy() for _ in range(10)])
                else:self.history[i].append(frames[i].copy())
                self.obs[i]=np.concatenate(self.history[i])
        else:
            # Explicit simulated camera measurement, not an onboard IMU substitute.
            x,_,_,_,_,_=self.geometry();pos=x[:,:,:3].copy()
            if self.noise: pos+=self.rng.normal(0,.0002,pos.shape)
            measured=(pos[:,9,:2]-self.visual_previous[:,9,:2])/.1
            self.visual_filtered+=.15*(measured-self.visual_filtered)
            self.visual_previous=pos.copy()
            delta=self.goals-pos[:,9,:2];distance=np.linalg.norm(delta,axis=1)
            direction=delta/np.maximum(distance[:,None],1e-9)
            velocity=self.visual_filtered
            vg=(velocity*direction).sum(1)
            theta=np.arctan2(direction[:,0]*velocity[:,1]-direction[:,1]*velocity[:,0],vg)
            # Velocity direction is undefined near rest: hold last defined angle.
            theta=np.where(np.linalg.norm(velocity,axis=1)>.0005,theta,self.prev_theta)
            dtheta=np.arctan2(np.sin(theta-self.prev_theta),np.cos(theta-self.prev_theta))/.1
            segment_axis=pos[:,1::2,:2]-pos[:,0::2,:2]
            heading=np.arctan2(segment_axis[...,1],segment_axis[...,0])[:,::-1]
            centers=((pos[:,1::2,:2]+pos[:,0::2,:2])/2)[:,::-1]
            bending=np.arctan2(np.sin(np.diff(heading,axis=1)),np.cos(np.diff(heading,axis=1)))
            length=np.linalg.norm(np.diff(centers,axis=1),axis=-1).clip(.05)
            curvature=bending/length
            for i in reset: vg[i]=0;theta[i]=0;dtheta[i]=0;self.visual_filtered[i]=0
            self.prev_theta=theta.copy()
            self.obs=np.c_[distance/.1,vg/.01,theta/np.pi,np.clip(dtheta,-10,10)/10,curvature/5,self.previous_actions,np.ones(self.num_envs),np.zeros(self.num_envs)].astype(np.float32)
        if not np.isfinite(self.obs).all(): raise RuntimeError('Nonfinite observation')
        return self.obs.copy()

    def reset_indices(self,indices):
        for i in indices:
            for k in STATE:getattr(self.c,k)[i].copy_(self.initial[k][i])
            self.count[i]=0;self.total[i]=0;self.targets[i]=0;self.previous_actions[i]=0;self.filtered_velocity[i]=0
            self.cmd[i]=self.override if self.override is not None else [self.rng.choice([0.,.004,.008]),0.,self.rng.choice([-.02,0.,.02])]
            if self.cmd[i,0]==0:self.cmd[i,2]=0
            self.c.wheel_mu[i]=.8  # First pilot fixed contact, equal across controllers.
        self.cpg.reset(indices)
        x,v,R,q,f,l=self.sensor_update(indices)
        for i in indices:
            angle=self.rng.choice([-1,1])*self.rng.uniform(.35,.55)
            if self.override is not None:angle=float(self.override[2])
            self.goals[i]=x[i,9,:2]+.08*(math.cos(angle)*f[i]+math.sin(angle)*l[i])
            self.visual_previous[i]=x[i,:,:3];self.prev_theta[i]=0;self.visual_filtered[i]=0
        self.packet(indices)

    def reset(self):
        self.reset_indices(list(range(self.num_envs)));return self.obs.copy()

    def step_async(self,actions):
        actions=np.asarray(actions,float)
        if actions.shape!=(self.num_envs,4) or not np.isfinite(actions).all():raise ValueError('Four finite yaw/tonic actions per environment required')
        self.actions=np.clip(actions,-1,1)

    def step_wait(self):
        c=self.c;x0,_,_,_,_,_=self.geometry();old=self.targets.copy();raw_peak=np.zeros(self.num_envs)
        work=np.zeros(self.num_envs)
        for _ in range(5):
            if self.kind=='direct': desired=LIMIT*self.actions
            else:
                signal=self.cpg.step(3*self.actions)
                raw_peak=np.maximum(raw_peak,np.max(abs(signal),axis=1))
                # Head-to-tail CPG -> reversed core joint order. Hardware sign is
                # a migration convention; paper pressures are not servo angles.
                desired=math.radians(15)*np.clip(signal[:,::-1],-1,1)
            self.targets+=np.clip(desired-self.targets,-.02,.02)
            c.target.zero_();c.yaw.copy_(c.tensor(self.targets));c.advance_steps(round(.02/c.dt));self.count+=1
            x,v,R,q,f,l=self.sensor_update()
            relrate=np.sum((v[:,2:10:2,3:]-v[:,1:9:2,3:])*R[:,1:9:2,:,2],axis=-1)
            torque=np.clip(2*(self.targets-q)-.03*relrate,-.5,.5)
            cap=.5*np.where(torque*relrate>0,np.maximum(0,1-abs(relrate)/2),1)
            torque=np.clip(torque,-cap,cap)
            work+=np.sum(abs(torque*relrate),axis=1)*.02
        if bool(c.failed) or not bool(torch.isfinite(c.x).all()): raise RuntimeError('Invalid physics: abort rather than reward numerical failure')
        velocity=(x[:,4,:2]-x0[:,4,:2])/.1
        measured=np.c_[np.sum(velocity*f,axis=1),np.sum(velocity*l,axis=1),v[:,4,5]]
        self.filtered_velocity+=(1-np.exp(-.1))* (measured-self.filtered_velocity)
        slip=c.last_slip.norm(dim=-1);active=c.support>1e-6
        slip=((slip*active).sum((1,2))/active.sum((1,2)).clamp_min(1)).cpu().numpy()
        action_cost=np.mean(((self.targets-old)/LIMIT)**2,axis=1)
        if self.kind=='direct':
            ev=self.filtered_velocity[:,:2]-self.cmd[:,:2];ew=self.filtered_velocity[:,2]-self.cmd[:,2]
            reward=.1*(np.exp(-np.sum(ev**2,axis=1)/.006**2)+.5*np.exp(-(ew/.025)**2)-.02*action_cost-.02*slip/.01)-.01*work
            success=np.zeros(self.num_envs,bool)
        else:
            distance0=np.linalg.norm(self.goals-x0[:,9,:2],axis=1)
            distance=np.linalg.norm(self.goals-x[:,9,:2],axis=1)
            reward=(distance0-distance)/.08-.002*action_cost-.002*slip/.01-.01*work
            success=distance<.01;reward+=2*success
        fail=(np.max(abs(q),axis=1)>math.radians(26))|(R[:,:,2,2].min(1)<.7)|(x[:,:,2].min(1)<.075)
        # Conservative sphere guard across distinct plate-owned wheels, not collision response.
        centers=x[:,:,:3,None]+np.einsum('bpij,bpsj->bpis',R,c.mounts.cpu().numpy()[None].repeat(self.num_envs,0))
        centers=centers.transpose(0,1,3,2).reshape(self.num_envs,20,3)
        distances=np.linalg.norm(centers[:,:,None]-centers[:,None,:],axis=-1)
        mask=np.repeat(np.arange(10),2)[:,None]!=np.repeat(np.arange(10),2)[None,:]
        clearance=distances[:,mask].min(1)-2*math.hypot(.018,.006)
        fail|=clearance<.002
        reward-=2*fail;self.total+=reward
        self.previous_actions=self.actions.copy()
        obs=self.packet();timeout=self.count>=1000;done=fail|timeout|success
        infos=[]
        for i in range(self.num_envs):
            info=dict(kind=self.kind,physical_s=float(self.count[i]*.02),command=self.cmd[i].tolist(),velocity=self.filtered_velocity[i].tolist(),joint_deg=np.rad2deg(q[i]).tolist(),target_deg=np.rad2deg(self.targets[i]).tolist(),wheel_clearance_mm=float(clearance[i]*1000),slip_m_s=float(slip[i]),yaw_work_proxy_j=float(work[i]),failed=bool(fail[i]),success=bool(success[i]),cpg_raw_peak=float(raw_peak[i]))
            if self.kind!='direct':info['goal_distance_m']=float(distance[i])
            if done[i]:info.update(terminal_observation=obs[i].copy(),**{'TimeLimit.truncated':bool(timeout[i] and not fail[i] and not success[i]),'episode':dict(r=float(self.total[i]),l=int(self.count[i]//5))})
            infos.append(info)
            if self.record:self.records.append({k:v for k,v in info.items() if k not in ('terminal_observation','episode')})
        if done.any():self.reset_indices(np.flatnonzero(done).tolist())
        return self.obs.copy(),reward.astype(np.float32),done,infos

    def close(self):pass
    def get_attr(self,name,indices=None):return [getattr(self,name) for _ in self._get_indices(indices)]
    def set_attr(self,name,value,indices=None):setattr(self,name,value)
    def env_method(self,name,*args,indices=None,**kwargs):return [getattr(self,name)(*args,**kwargs) for _ in self._get_indices(indices)]
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]


def run(args):
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(1)
    write(out/'status.json',dict(status='initializing',kind=args.kind))
    contract=dict(kind=args.kind,algorithm='PPO; fixed Kf=1 stage (not full PPOC)' if args.kind!='direct' else 'direct joint PPO without CPG',physics='GPU corrected strain Jacobian, corotational beam candidate, implicit wheel contact, ideal ratchet; not fully validated',imu='plate4 local frame rotated diag(-1,-1,1); no accelerometer or CPG observation in direct actor',direct_observation='38 channels x10 frames at10Hz; encoder14,difference14,gyro3,last applied targets4,commands3',liu_observation='simulated visual goal distance, velocity projection, motion-goal angle/rate, FOUR INTERSEGMENT EFFECTIVE CURVATURES (not paper soft-segment curvature), previous alpha4,Kf1,termination0',liu_differences=['steel robot and servo angle mapping','fixed-frequency training stage only','eight base channels use equivalent intersegment curvature','adapted 80mm goal distance/10mm radius and progress reward','coupling index convention explicit in matsuoka.py; not checked against author code'],task='body-frame vx/vy/yaw rate; vy=0 initial curriculum' if args.kind=='direct' else 'single visual goal, not same task as velocity branch',tendon_targets='all ten fixed at zero, no active peristalsis',joint_limit_deg=20,policy_hz=10,control_hz=50,dt=args.dt,friction=.8,rolling=.06,batch=args.n,seed=args.seed,requested_steps=args.steps,max_training_wall_s=args.seconds,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},status='pilot only; no convergence/hardware claim')
    write(out/'contract.json',contract)
    (out/'sources').mkdir()
    for source in ROOT.glob('*.py'):shutil.copy2(source,out/'sources'/source.name)
    try:
        env=Env(args.kind,args.n,args.seed,args.dt);obs=env.reset()
        if args.check:
            env.record=True
            for k in range(40):
                a=np.tile(.5*np.sin(2*np.pi*k*.1/4-np.arange(4)*np.pi/2),(args.n,1)) if args.kind=='direct' else np.zeros((args.n,4))
                obs,reward,done,infos=env.step(a)
                if any(i['failed'] for i in infos):raise RuntimeError('Physical guard failed in smoke test')
            write(out/'smoke.json',env.records)
            write(out/'status.json',dict(status='smoke_passed',kind=args.kind,elapsed_s=time.monotonic()-started,physics_steps=round(4/args.dt),final=infos[0]))
            return
        model=PPO('MlpPolicy',env,n_steps=64,batch_size=64,n_epochs=4,learning_rate=3e-4,gamma=.995,gae_lambda=.95,ent_coef=.002,target_kl=.03,device='cpu',seed=args.seed,policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128]),log_std_init=-.8),verbose=1)
        checkpoint=out/'checkpoint.zip'
        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls%8==0:
                    info=self.locals['infos'][0]
                    write(out/'status.json',dict(status='training',kind=args.kind,steps=self.num_timesteps,elapsed_s=time.monotonic()-started,last={k:v for k,v in info.items() if k not in ('terminal_observation','episode')}))
                return time.monotonic()-started<args.seconds
            def _on_rollout_start(self):
                if self.num_timesteps:
                    model.save(out/'checkpoint.tmp.zip');(out/'checkpoint.tmp.zip').replace(checkpoint)
                    write(out/'checkpoint.json',dict(steps=self.num_timesteps,elapsed_s=time.monotonic()-started))
        model.learn(args.steps,callback=Progress());model.save(out/'policy.zip')
        write(out/'status.json',dict(status='pilot_complete',kind=args.kind,steps=model.num_timesteps,optimizer_updates=model._n_updates,elapsed_s=time.monotonic()-started,converged=False,evaluation_pending=True))
    except BaseException as e:
        write(out/'status.json',dict(status='failed',kind=args.kind,error=repr(e),elapsed_s=time.monotonic()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['direct','liu','liu_foc'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=2);p.add_argument('--seed',type=int,default=7);p.add_argument('--steps',type=int,default=2048);p.add_argument('--seconds',type=int,default=1800);p.add_argument('--dt',type=float,default=.00025);p.add_argument('--check',action='store_true')
    run(p.parse_args())

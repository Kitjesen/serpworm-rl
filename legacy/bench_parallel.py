"""Short batch-size timing with unchanged mechanics; optionally pause our worker.

CPU worker suspension does not immediately drain its CUDA queue. Wait for idle,
then benchmark, and always resume in finally. Never terminate a training process.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import torch
from gpu_core import GPUCore


def write(path, data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--pause-pid',type=int,default=0)
    p.add_argument('--pause-eval-pid',type=int,default=0,help='An evaluation child of the verified training worker')
    p.add_argument('--batches',type=int,nargs='+',default=[8,16,32,64,8])
    p.add_argument('--steps',type=int,default=128);p.add_argument('--warmup',type=int,default=64)
    p.add_argument('--rollout-envs',type=int,default=0,help='Also run one independent 64-step PPO rollout with this many environments')
    p.add_argument('--out',default='results/batch_scaling.json');args=p.parse_args()
    if min(args.batches)<=0 or args.steps<=0 or args.steps%4 or args.warmup<0 or args.warmup%4 or args.rollout_envs<0:
        p.error('Positive batches and steps required; steps/warmup must be multiples of 4')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'starting','results':[], 'scope':'Same corrected beam physics, 5 segments, dt=0.25ms; core-only timing, not full PPO throughput or physics validation.'}
    paused=[]
    started=time.monotonic()
    try:
        if args.pause_pid:
            cmd=Path(f'/proc/{args.pause_pid}/cmdline').read_bytes().replace(b'\0',b' ')
            if b'/reward_ablation_20261005/train_gpu.py' not in cmd or b'long_tracking_v2_seed7' not in cmd:
                raise RuntimeError('Refusing to suspend an unrelated process')
            os.kill(args.pause_pid,signal.SIGSTOP);paused.append(args.pause_pid)
            if args.pause_eval_pid:
                proc=Path(f'/proc/{args.pause_eval_pid}')
                child_cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ')
                parent=next(line.split()[1] for line in (proc/'status').read_text().splitlines() if line.startswith('PPid:'))
                if int(parent)!=args.pause_pid or b'/reward_ablation_20261005/evaluate_gpu.py' not in child_cmd:
                    raise RuntimeError('Refusing to suspend an unrelated evaluation process')
                os.kill(args.pause_eval_pid,signal.SIGSTOP);paused.append(args.pause_eval_pid)
            result['status']='waiting_for_queued_cuda_work';write(out,result)
            idle=0;deadline=time.monotonic()+180
            while idle < 2:
                utilization=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
                idle=idle+1 if int(utilization.strip().splitlines()[0])==0 else 0
                if time.monotonic()>deadline:raise TimeoutError('Other CUDA work remains active; aborting isolated benchmark')
                time.sleep(2)
        torch.set_num_threads(1)
        result['gpu']=torch.cuda.get_device_name()
        result['physics_sha256']=hashlib.sha256(Path(__file__).with_name('gpu_core.py').read_bytes()).hexdigest()
        result['status']='benchmarking';write(out,result)
        for batch in args.batches:
            result['current_batch']=batch;write(out,result)
            c=GPUCore(5,batch,dt=.00025,yaw_motor_speed=2.,strain_jacobian=True)
            c.yaw.copy_(c.tensor([.05,-.05,.05,-.05]).expand(batch,-1))
            c.capture(4);c.advance_steps(args.warmup);torch.cuda.synchronize()
            names=('x','v','servo','target','yaw','spin','angle','memory','tension','support','failed')
            state={name:getattr(c,name).clone() for name in names}
            timings=[]
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                for name,value in state.items():getattr(c,name).copy_(value)
                torch.cuda.synchronize();start=time.perf_counter()
                c.advance_steps(args.steps);torch.cuda.synchronize()
                timings.append(time.perf_counter()-start)
                assert not bool(c.failed) and bool(torch.isfinite(c.x).all())
            seconds=statistics.median(timings)/args.steps
            free,total=torch.cuda.mem_get_info()
            row={'batch':batch,'steps_per_repeat':args.steps,'warmup_steps':args.warmup,'repeat_wall_s':timings,'median_batch_step_ms':seconds*1000,
                 'core_equivalent_policy_samples_per_s':batch/(400*seconds),
                 'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                 'reserved_mib':torch.cuda.memory_reserved()/1024**2,'device_used_mib':(total-free)/1024**2}
            result['results'].append(row);write(out,result);print(json.dumps(row),flush=True)
            del c,state;gc.collect();torch.cuda.empty_cache()
        if args.rollout_envs:
            rollout_out=out.parent/(out.stem+'_rollout')
            result['status']='full_rollout';result['rollout_out']=str(rollout_out);write(out,result)
            with rollout_out.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('train_gpu.py')),
                    '--kind','direct','--reward','tracking_v2','--envs',str(args.rollout_envs),
                    '--horizon','64','--rollouts','1','--seconds','0','--seed','7','--out',str(rollout_out)],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            result['full_rollout']=json.loads((rollout_out/'status.json').read_text())
        result['status']='complete'
    except BaseException as e:
        result['status']='failed';result['error']=repr(e);raise
    finally:
        if paused:
            for pid in reversed(paused):
                try:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass
            result['training_resumed']=True
        result['wall_s']=time.monotonic()-started;write(out,result)

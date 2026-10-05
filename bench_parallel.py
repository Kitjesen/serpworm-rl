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
import time
import torch
from gpu_core import GPUCore


def write(path, data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--pause-pid',type=int,default=0)
    p.add_argument('--out',default='results/batch_scaling.json');args=p.parse_args()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'starting','results':[], 'scope':'Same corrected beam physics, 5 segments, dt=0.25ms; core-only timing, not full PPO throughput or physics validation.'}
    paused=False
    started=time.monotonic()
    try:
        if args.pause_pid:
            cmd=Path(f'/proc/{args.pause_pid}/cmdline').read_bytes().replace(b'\0',b' ')
            if b'/reward_ablation_20261005/train_gpu.py' not in cmd or b'long_tracking_v2_seed7' not in cmd:
                raise RuntimeError('Refusing to suspend an unrelated process')
            os.kill(args.pause_pid,signal.SIGSTOP);paused=True
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
        for batch in (8,16,32,64,8):
            c=GPUCore(5,batch,dt=.00025,yaw_motor_speed=2.,strain_jacobian=True)
            c.yaw.copy_(c.tensor([.05,-.05,.05,-.05]).expand(batch,-1))
            c.capture(4);c.advance_steps(64);torch.cuda.synchronize()
            names=('x','v','servo','target','yaw','spin','angle','memory','tension','support','failed')
            state={name:getattr(c,name).clone() for name in names}
            timings=[]
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                for name,value in state.items():getattr(c,name).copy_(value)
                torch.cuda.synchronize();start=time.perf_counter()
                c.advance_steps(128);torch.cuda.synchronize()
                timings.append(time.perf_counter()-start)
                assert not bool(c.failed) and bool(torch.isfinite(c.x).all())
            seconds=statistics.median(timings)/128
            row={'batch':batch,'steps_per_repeat':128,'repeat_wall_s':timings,'median_batch_step_ms':seconds*1000,
                 'core_equivalent_policy_samples_per_s':batch/(400*seconds),
                 'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
            result['results'].append(row);write(out,result);print(json.dumps(row),flush=True)
            del c,state;gc.collect();torch.cuda.empty_cache()
        result['status']='complete'
    except BaseException as e:
        result['status']='failed';result['error']=repr(e);raise
    finally:
        if paused:
            os.kill(args.pause_pid,signal.SIGCONT)
            result['training_resumed']=True
        result['wall_s']=time.monotonic()-started;write(out,result)

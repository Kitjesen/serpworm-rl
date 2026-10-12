"""Incremental regression for the optional plate solve; does not change mechanics.

Reuses the historical 5 s / 1 s-delay backward contraction pulse. This is a
same-version, same-input old/new solve comparison, not renewed hardware validation.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from gpu_core import GPUCore, matrix
from tensor_env import TensorEnv


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def commands(core, t):
    # Historical run_worm_wave.py: head-to-tail internal indices 4,3,2,1,0.
    age = t-(4-torch.arange(5, device=core.device, dtype=core.dtype))
    phase = age.clamp_min(0).remainder(5.)
    def blend(u):
        u = u.clamp(0., 1.)
        return u**3*(10-15*u+6*u*u)
    pulse = .8*torch.where(phase < 1., blend(phase),
                          torch.where(phase < 1.2, 1., 1-blend(phase-1.2)))*(age >= 0.)
    yaw = .15*math.sin(min(t, .5)*math.pi)*torch.sin(
        core.tensor(2*math.pi*t/2)-torch.arange(4, device=core.device)*math.pi/2)
    core.target.zero_(); core.yaw.zero_()
    core.target[1] = pulse.repeat_interleave(2)
    core.target[3] = core.target[1]
    core.yaw[2] = yaw; core.yaw[3] = yaw


def dynamic_check(seconds, device, progress):
    old = GPUCore(5, 4, device=device, strain_jacobian=True, plate_solver='legacy')
    new = GPUCore(5, 4, device=device, strain_jacobian=True, plate_solver='segment_lu')
    for core in (old, new):
        if device == 'cuda': core.capture(4)
        core.advance_steps(2000)  # Same 0.5 s passive initialization as TensorEnv.
    names = ['position_m', 'rotation_matrix', 'linear_velocity_m_s', 'angular_velocity_rad_s',
             'tension_n', 'support_n', 'wheel_spin_rad_s', 'servo_rad']
    maxima = torch.zeros(len(names), device=old.device, dtype=old.dtype)
    peak_tension = torch.zeros((), device=old.device, dtype=old.dtype)
    frames = round(seconds/.02)
    started = time.perf_counter()
    for frame in range(frames):
        for sub in range(10):
            t = frame*.02+(sub+1)*.002
            for core in (old, new):
                commands(core, t); core.advance_steps(8)
        differences = [(old.x[...,:3]-new.x[...,:3]).abs().max(),
                       (matrix(old.x[...,3:])-matrix(new.x[...,3:])).abs().max(),
                       (old.v[...,:3]-new.v[...,:3]).abs().max(),
                       (old.v[...,3:]-new.v[...,3:]).abs().max()]
        differences += [(getattr(old, k)-getattr(new, k)).abs().max()
                        for k in ('tension', 'support', 'spin', 'servo')]
        maxima = torch.maximum(maxima, torch.stack(differences))
        peak_tension = torch.maximum(peak_tension, new.tension.abs().max())
        if (frame+1) % 50 == 0:
            progress((frame+1)*.02, maxima.cpu().tolist())
    limits = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-3, 1e-3, 1e-7]
    errors = dict(zip(names, maxima.cpu().tolist()))
    finite = bool(torch.isfinite(old.x).all() & torch.isfinite(new.x).all() & torch.isfinite(maxima).all())
    passed = finite and not bool(old.failed | new.failed) and all(x <= y for x, y in zip(errors.values(), limits))
    return dict(passed=passed, cases=['passive', 'backward_wave', 'snake_wave', 'combined'],
                simulated_s=frames*.02, passive_initialization_s=.5,
                max_abs_difference=errors, absolute_tolerances=dict(zip(names, limits)),
                peak_tension_n=float(peak_tension), wall_s=time.perf_counter()-started,
                scope='Same current mechanics with only the linear solve changed; diagnostic waves, not learned policies.')


def speed_check(batch, steps):
    rows = {}
    for solver in ('legacy', 'segment_lu'):
        core = GPUCore(5, batch, strain_jacobian=True, plate_solver=solver)
        core.yaw.copy_(core.tensor([.05, -.05, .05, -.05]))
        core.capture(4); core.advance_steps(64)
        names = ('x','v','servo','target','yaw','spin','angle','memory','tension','support','failed')
        state = {name:getattr(core, name).clone() for name in names}
        measured = []
        for _ in range(3):
            for name, value in state.items(): getattr(core, name).copy_(value)
            torch.cuda.synchronize(); begin = time.perf_counter()
            core.advance_steps(steps); torch.cuda.synchronize()
            measured.append(time.perf_counter()-begin)
            assert not bool(core.failed) and bool(torch.isfinite(core.x).all())
        per_step = statistics.median(measured)/steps
        rows[solver] = dict(repeat_wall_s=measured, batch_physics_step_ms=per_step*1000,
                            equivalent_policy_samples_s=batch/(400*per_step))
        del core, state; gc.collect(); torch.cuda.empty_cache()
    return dict(batch=batch, physical_steps_per_repeat=steps, solvers=rows,
                speedup=rows['legacy']['batch_physics_step_ms']/rows['segment_lu']['batch_physics_step_ms'])


def rollout_check(steps):
    records = {}; rows = {}
    for solver in ('legacy', 'segment_lu'):
        env = TensorEnv('direct', n=8, seed=7, noise=False, reward_version='tracking_v2', plate_solver=solver)
        observations=[]; rewards=[]; metrics=[]; ends=[]
        actions = .5*torch.sin(torch.arange(steps, device='cuda')[:,None]*.4+torch.arange(4, device='cuda'))
        torch.cuda.synchronize(); begin=time.perf_counter()
        for action in actions:
            obs, reward, term, trunc, final, metric = env.step(action.expand(8,-1))
            observations.append(obs); rewards.append(reward); metrics.append(metric); ends.append(torch.stack((term,trunc)))
        torch.cuda.synchronize(); wall=time.perf_counter()-begin
        assert not bool(env.c.failed)
        records[solver]=[torch.stack(v) for v in (observations,rewards,metrics,ends)]
        rows[solver]=dict(samples=8*steps, rollout_wall_s=wall, samples_s=8*steps/wall)
        del env; gc.collect(); torch.cuda.empty_cache()
    a,b=records['legacy'],records['segment_lu']
    differences=[float((x-y).abs().max()) for x,y in zip(a[:3],b[:3])]
    same_ends=bool(torch.equal(a[3],b[3]))
    finite=all(bool(torch.isfinite(x).all()) for x in a[:3]+b[:3])
    return dict(passed=finite and same_ends and max(differences)<1e-4,
                obs_reward_metric_max_abs=differences, identical_termination_and_timeout=same_ends,
                solvers=rows, speedup=rows['legacy']['rollout_wall_s']/rows['segment_lu']['rollout_wall_s'],
                scope='Full sensor/reward/reset path with scripted identical actions; no PPO update and no 20 s timeout coverage.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--batches', type=int, nargs='*', default=[8,64])
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--env-steps', type=int, default=8)
    p.add_argument('--out', required=True)
    args=p.parse_args(); path=Path(args.out)
    if path.exists(): p.error('Refusing to overwrite existing evidence')
    if args.seconds<=0 or abs(args.seconds/.02-round(args.seconds/.02))>1e-8 or args.steps<=0 or args.steps%4 or args.env_steps<0 or any(x<1 for x in args.batches):
        p.error('Positive duration in multiples of .02 s; steps multiple of 4; positive batches; nonnegative env-steps required')
    torch.set_num_threads(1); path.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='dynamic_regression',device=args.device,
                source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('gpu_core.py','tensor_env.py','check_plate_solver.py')})
    def progress(seconds, errors):
        report['progress_s']=seconds; report['progress_errors']=errors; write(path,report)
    write(path,report)
    try:
        report['dynamic']=dynamic_check(args.seconds,args.device,progress)
        write(path,report)
        if not report['dynamic']['passed']: raise AssertionError('Dynamic regression tolerance failed; no speed claims')
        if args.device=='cuda':
            gc.collect(); torch.cuda.empty_cache(); report['timings']=[]
            for batch in args.batches:
                report['status']='core_timing'; report['current_batch']=batch; write(path,report)
                report['timings'].append(speed_check(batch,args.steps)); write(path,report)
            if args.env_steps:
                report['status']='environment_rollout';write(path,report)
                report['rollout']=rollout_check(args.env_steps)
                if not report['rollout']['passed']: raise AssertionError('Environment outputs diverged')
        report['status']='complete'
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:
        write(path,report)

"""CUDA fixed-wave gate -> supervised actor initialization -> finite direct PPO."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from serpworm.paths import PKG_ROOT, REPO_ROOT, SCRIPTS_DIR, collect_input_hashes, collect_source_hashes

ROOT = REPO_ROOT
MAX_SAMPLES = 131072


def write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable: no probe, output directory or training was started')


def no_active_source_jobs(root=SCRIPTS_DIR):
    proc = Path('/proc')
    if not proc.is_dir():
        return
    scripts = {'run_pipeline.py', 'probe_fixed_waves.py', 'pretrain_actor.py', 'train_ppo.py', 'evaluate.py'}
    for process in proc.glob('[0-9]*'):
        if process.name == str(os.getpid()):
            continue
        try:
            args = (process/'cmdline').read_bytes().decode().split('\0')
            cwd = (process/'cwd').resolve()
            if any(Path(a).name in scripts and (cwd in (REPO_ROOT, root) or Path(a).parent == root) for a in args if a):
                raise RuntimeError(f'Existing source job PID {process.name}; refuse a duplicate pipeline')
        except (OSError, UnicodeError):
            continue


MODE_KINDS = (('snake', {'snake'}), ('worm', {'worm'}), ('mixed', {'mixed', 'full'}))


def candidates(data, report, modes=None):
    if data.get('schema') != 'fixed_wave_dataset_v1' or data.get('device') != 'cuda' or report.get('complete') is not True:
        raise ValueError('A complete CUDA fixed-wave dataset is required before any actor training')
    if not math.isclose(float(data.get('physics_dt_s', float('nan'))), .00025):
        raise ValueError('CUDA teacher physics dt must match training dt=0.00025')
    results = {row['case']['name']: row for row in report.get('results', [])}
    present_modes = {case.get('mode') for case in data.get('cases', {}).values()}
    active_modes = tuple(modes) if modes is not None else tuple(m for m, _ in MODE_KINDS if m in ('snake', 'mixed') or m in present_modes)
    selected = {}
    for mode, kinds in MODE_KINDS:
        if mode not in active_modes:
            continue
        ranked = []
        for name, case in data.get('cases', {}).items():
            summary, row = case.get('summary', {}), results.get(name, {})
            if case.get('mode') != mode or summary.get('case_kind') not in kinds:
                continue
            if not all(summary.get(k) is True for k in ('gate_passed', 'reward_replay_better_than_stationary', 'complete')):
                continue
            tail = row.get('tail', {})
            speed = tail.get('summary', {}).get('mean_vx_mm_s') if tail.get('summary') else None
            neutral = case.get('servo_neutral_deg')
            if row.get('first_episode_complete') is not True or tail.get('complete') is not True:
                continue
            if not isinstance(speed, (float, int)) or not math.isfinite(speed) or speed < 1.:
                continue
            if not isinstance(neutral, (float, int)) or not math.isfinite(neutral) or not 0 <= neutral <= 180:
                continue
            ranked.append(dict(case=name, mode=mode, servo_neutral_deg=float(neutral), tail_vx_mm_s=float(speed)))
        selected[mode] = max(ranked, key=lambda x: x['tail_vx_mm_s']) if ranked else None
    return selected


def training_command(scripts_dir, mode, neutral, checkpoint, out, rollouts):
    return [sys.executable, '-u', str(scripts_dir/'train_ppo.py'), '--warmstart-actor', str(checkpoint),
        '--eval-initial', '--initial-std', '.65', '--servo-neutral-deg', str(neutral), '--mode', mode,
        '--out', str(out), '--rollouts', str(rollouts), '--envs', '8', '--horizon', '64',
        '--batch-size', '128', '--epochs', '4', '--save-every', '8', '--eval-every', '8',
        '--vx-min-mm-s', '5', '--vx-max-mm-s', '200']


def run(args):
    require_cuda()  # Deliberately before mkdir/lock/Popen: a CPU preparation is not a CUDA run.
    if not 1 <= args.rollouts <= 128 or args.bc_epochs < 1 or not math.isfinite(args.probe_seconds) or args.probe_seconds < 40:
        raise ValueError('Use 1..128 rollouts, positive BC epochs, and a probe of at least 40 seconds')
    no_active_source_jobs()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'pipeline.lock').mkdir()
    manifest = dict(status='initializing', source_root=str(PKG_ROOT),
        source_sha256=collect_source_hashes(PKG_ROOT),
        input_sha256=collect_input_hashes(PKG_ROOT),
        maximum_training_samples=args.rollouts*512*2, stage_sample_cap=MAX_SAMPLES,
        budget_scope='PPO interaction samples; fixed-wave data and supervised epochs are separate',
        maximum_rollouts_per_mode=args.rollouts, probe_seconds=args.probe_seconds,
        scope='Fixed waves and supervised initialization are not learned PPO motion; training performs an initial deterministic gate', jobs=[], modes={})
    write(out/'pipeline.json', manifest)

    def start(name, command, child_out):
        job = dict(name=name, command=command, out=str(child_out), status='starting')
        manifest['jobs'].append(job)
        try:
            with (out/(name+'.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            job.update(pid=process.pid, status='running')
        except OSError as error:
            job.update(status='failed_to_start', error=repr(error))
            process = None
        write(out/'pipeline.json', manifest)
        return process, job

    def finish(process, job):
        if process is None:
            return False
        code = process.wait()
        job.update(exit_code=code, status='completed' if code == 0 else 'failed')
        status = Path(job['out'])/'status.json'
        if status.exists():
            job['child_status'] = json.loads(status.read_text())
        write(out/'pipeline.json', manifest)
        return code == 0

    try:
        probe = out/'probe'
        manifest['status'] = 'probing_fixed_waves'
        process, job = start('probe', [sys.executable, '-u', str(SCRIPTS_DIR/'probe_fixed_waves.py'), '--device', 'cuda',
            '--seconds', str(args.probe_seconds), '--out', str(probe)], probe)
        if not finish(process, job):
            raise RuntimeError('Fixed-wave probe failed; actor training was not started')
        import torch
        chosen = candidates(torch.load(probe/'teacher_data.pt', map_location='cpu', weights_only=True),
                            json.loads((probe/'summary.json').read_text()))
        attempted_training = 0
        for mode, case in chosen.items():
            if case is None:
                manifest['modes'][mode] = dict(status='needs_physics_diagnosis', reason='No complete wave passed persistent-forward and reward-improvement gates')
                continue
            manifest['modes'][mode] = {**case, 'status': 'behavior_cloning'}
            bc = out/(mode+'_bc')
            process, job = start(mode+'_bc', [sys.executable, '-u', str(SCRIPTS_DIR/'pretrain_actor.py'), '--data', str(probe/'teacher_data.pt'),
                '--case', case['case'], '--mode', mode, '--out', str(bc), '--device', 'cuda', '--epochs', str(args.bc_epochs)], bc)
            if not finish(process, job) or not (bc/'actor_checkpoint.pt').is_file():
                manifest['modes'][mode]['status'] = 'needs_actor_diagnosis'
                continue
            destination = out/(mode+'_ppo')
            process, job = start(mode+'_ppo', training_command(SCRIPTS_DIR, mode, case['servo_neutral_deg'], bc/'actor_checkpoint.pt', destination, args.rollouts), destination)
            manifest['modes'][mode]['status'] = 'initial_closed_loop_gate_then_training' if process else 'failed_to_start'
            # One training parent plus its evaluator fits the tested GPU budget;
            # finish this mode before allocating the next mode's environment.
            attempted_training += 1
            manifest['status'] = 'waiting_for_'+mode
            write(out/'pipeline.json', manifest)
            finish(process, job)
            manifest['modes'][mode]['status'] = job.get('child_status', {}).get('status', job['status'])
        manifest['status'] = 'finished_review_required' if attempted_training else 'needs_diagnosis'
        write(out/'pipeline.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        write(out/'pipeline.json', manifest)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rollouts', type=int, default=128)
    parser.add_argument('--probe-seconds', type=float, default=40.)
    parser.add_argument('--bc-epochs', type=int, default=400)
    print(json.dumps(run(parser.parse_args(argv)), indent=2))


if __name__ == '__main__':
    main()

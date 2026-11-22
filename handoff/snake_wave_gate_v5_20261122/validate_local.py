"""Record reproducible local numerical checks; never start physics or PPO."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parent


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}


def main():
    before = hashes()
    logs = ROOT/'validation_logs'
    logs.mkdir(exist_ok=True)
    checks = []
    commands = [
        ['test_ppo.py', '--device', 'cpu'],
        ['test_rewards.py', '--device', 'cpu'],
        ['test_evaluation.py'], ['test_wave_probe.py'], ['test_pretrain.py'],
        ['test_pipeline.py'], ['test_policy_distribution.py'], ['test_wheel_envelope.py'],
    ]
    for args in commands:
        start = time.monotonic()
        result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        logfile = logs/(Path(args[0]).stem+'.log')
        logfile.write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        checks.append(dict(command=[sys.executable, *args], exit_code=result.returncode,
                           wall_s=time.monotonic()-start, log=str(logfile)))
        print(json.dumps(checks[-1]), flush=True)
    unchanged = hashes() == before
    passed = unchanged and all(row['exit_code'] == 0 for row in checks)
    report = dict(status='local_checks_passed_gpu_pending' if passed else 'local_checks_failed',
        local_numerical_checks_passed=passed, source_unchanged_during_checks=unchanged,
        torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
        source_sha256=before,
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'inputs').glob('*') if p.is_file()},
        checks=checks, gpu_fixed_wave_gate='not_run', real_teacher_bc='not_run', new_ppo='not_started',
        remote_compute='Not checked by this numerical test runner; see README for the dated connectivity finding',
        scope='Numerical/synthetic checks only. CPU smoke and archived CUDA reanalysis are separately labelled; no new sustained-motion success claim.')
    temporary = ROOT/'validation.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(ROOT/'validation.json')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

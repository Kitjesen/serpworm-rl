"""Record reproducible local numerical contract checks without starting PPO training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serpworm.paths import PKG_ROOT, collect_input_hashes, collect_source_hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPO_ROOT / 'runs' / 'local_validation' / 'validation.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args(argv)

    before = collect_source_hashes(PKG_ROOT)
    out = args.out.resolve()
    logs = out.parent / 'logs'
    logs.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_pp = env.get('PYTHONPATH', '')
    extra_pp = os.pathsep.join((str(SRC_DIR), str(REPO_ROOT)))
    env['PYTHONPATH'] = extra_pp + (os.pathsep + existing_pp if existing_pp else '')

    tests_dir = REPO_ROOT / 'tests'
    commands = [
        [str(tests_dir / 'test_ppo.py'), '--device', args.device],
        [str(tests_dir / 'test_rewards.py'), '--device', args.device],
        [str(tests_dir / 'test_evaluation.py')],
        [str(tests_dir / 'test_wave_probe.py')],
        [str(tests_dir / 'test_pretrain.py')],
        [str(tests_dir / 'test_pipeline.py')],
        [str(tests_dir / 'test_policy_distribution.py')],
        [str(tests_dir / 'test_wheel_envelope.py')],
        [str(tests_dir / 'test_package_smoke.py')],
        [str(tests_dir / 'test_three_modes_parallel.py')],
        [str(tests_dir / 'test_unified_architecture.py')],
        [str(tests_dir / 'test_topology_and_acceleration.py')],
    ]
    checks = []
    for cmd_args in commands:
        start = time.monotonic()
        result = subprocess.run([sys.executable, *cmd_args], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        logfile = logs / (Path(cmd_args[0]).stem + '.log')
        logfile.write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
        checks.append(dict(
            command=[Path(cmd_args[0]).name, *cmd_args[1:]],
            exit_code=result.returncode,
            wall_s=time.monotonic() - start,
            log=logfile.relative_to(out.parent).as_posix(),
        ))
        print(json.dumps(checks[-1]), flush=True)

    unchanged = collect_source_hashes(PKG_ROOT) == before
    passed = unchanged and all(row['exit_code'] == 0 for row in checks)
    report = dict(
        status='local_checks_passed' if passed else 'local_checks_failed',
        local_numerical_checks_passed=passed,
        source_unchanged_during_checks=unchanged,
        device=args.device,
        torch=str(torch.__version__),
        cuda_available=torch.cuda.is_available(),
        source_sha256=before,
        input_sha256=collect_input_hashes(PKG_ROOT),
        checks=checks,
        scope='Contract and package smoke checks only; not full 40 s wave-gate or PPO convergence validation.',
    )
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(out)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

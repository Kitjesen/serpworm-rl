"""Finite sequential pilots after successful CUDA migration checks."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
if __name__ == '__main__':
    result = json.loads((root/'results/migration_checks.json').read_text())
    if not result.get('passed') or result.get('device') != 'cuda':
        raise SystemExit('Run check_migration.py on CUDA successfully first')
    for kind in ('direct', 'liu', 'liu_foc'):
        command = [sys.executable, '-u', str(root/'train_gpu.py'), '--kind', kind,
                   '--envs', '2', '--rollouts', '8', '--seconds', '1800',
                   '--out', str(root/'results'/('tensor_'+kind))]
        with (root/('tensor_'+kind+'.log')).open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)

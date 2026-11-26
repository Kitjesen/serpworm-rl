"""Pretrain a direct-action actor via purged-cycle behavior cloning from a passed CUDA wave."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.rl.pretrain_bc import cli_main

if __name__ == '__main__':
    cli_main()

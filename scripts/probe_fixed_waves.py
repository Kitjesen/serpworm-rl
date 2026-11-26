"""Sweep fixed snake/mixed/full waves and export teacher datasets."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.probes.fixed_waves import main

if __name__ == '__main__':
    main()

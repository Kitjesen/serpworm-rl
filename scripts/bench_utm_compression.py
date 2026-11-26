#!/usr/bin/env python3
"""CLI wrapper for 1:1 UTM compression experiment vs FP64 simulation & MuJoCo reproduction."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.probes.bench_utm_compression import main

if __name__ == '__main__':
    main()

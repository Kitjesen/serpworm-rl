#!/usr/bin/env python3
"""CLI wrapper for unified 3-mode asymmetric PPO training."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.rl.train_unified import main

if __name__ == '__main__':
    main()

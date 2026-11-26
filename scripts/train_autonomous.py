#!/usr/bin/env python3
"""CLI wrapper for pure-proprioceptive autonomous phase-transition PPO training (v6-v8)."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.rl.train_autonomous import main

if __name__ == '__main__':
    main()

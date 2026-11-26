"""Run frozen-source mean/sample x sensor-noise diagnostic against an archived checkpoint."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.probes.policy_distribution import main

if __name__ == '__main__':
    main()

"""Audit rigid five-module wheel clearance and finite-cylinder separation certificates."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.probes.wheel_geometry import main

if __name__ == '__main__':
    main()

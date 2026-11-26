#!/usr/bin/env python3
"""Lightweight CPU summary & JSON export tool for all 4 FlexGraph robot archetypes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flexgraph import (
    build_bristle_worm_graph,
    build_flexspine_crawler_graph,
    build_serpworm_graph,
    build_tendon_endoscope_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description='Inspect and optionally export FlexGraph robot archetypes.')
    parser.add_argument('--export-dir', type=Path, default=None, help='Optional directory to write JSON robot specs')
    args = parser.parse_args()

    graphs = [
        build_serpworm_graph(n_segments=5, elements_per_strip=16),
        build_serpworm_graph(n_segments=3, elements_per_strip=16),
        build_serpworm_graph(n_segments=7, elements_per_strip=16),
        build_bristle_worm_graph(n_segments=5, elements_per_strip=10),
        build_tendon_endoscope_graph(n_sections=4, elements_per_section=8, free_cantilever_tip=False),
        build_tendon_endoscope_graph(n_sections=4, elements_per_section=8, free_cantilever_tip=True),
        build_flexspine_crawler_graph(n_segments=3, elements_per_spine=8),
    ]

    rows = []
    for g in graphs:
        sp = g.analyze_sparsity()
        contact_modes = sorted({c.contact_mode for c in g.contact_sites})
        rows.append(
            {
                'name': g.name,
                'bodies': f'{sp.n_free_bodies} free / {sp.n_fixed_bodies} fixed',
                'beams': f'{sp.n_beams} ({sp.n_cantilever_beams} cantilever)',
                'dofs': sp.total_dofs,
                'actuators': len(g.actuators),
                'route': sp.recommended_route,
                'contact_modes': contact_modes,
                'symmetry': f'{g.symmetry.name} (|G|={g.symmetry.order})' if g.symmetry else 'none',
            }
        )
        if args.export_dir is not None:
            g.save_json(args.export_dir / f'{g.name}.json')

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

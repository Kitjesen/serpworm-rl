#!/usr/bin/env python3
"""Export a trained SerpWorm .pt checkpoint (v5/v6/v7) to TorchScript .ts + deploy_meta.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serpworm.deploy.policy_runtime import export_checkpoint_to_torchscript

DEFAULT_CHECKPOINT_CANDIDATES = (
    REPO_ROOT / 'runs' / 'v6_1_65d_elec' / 'best.pt',
    REPO_ROOT / 'runs' / 'v6b_65d_qerr' / 'best.pt',
    REPO_ROOT / 'runs' / 'v6a_51d_pure' / 'best.pt',
    REPO_ROOT / 'runs' / 'unified_v5_hispeed_14d' / 'best.pt',
    REPO_ROOT / 'runs' / 'unified_v3_150r' / 'best.pt',
)


def resolve_default_checkpoint() -> Path:
    for cand in DEFAULT_CHECKPOINT_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        'No local checkpoint found in runs/. Pass --checkpoint <path/to/best.pt> explicitly.'
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Export a SerpWorm v5/v6/v7 .pt checkpoint to frozen TorchScript (.ts) + deploy_meta.json.'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='Path to trained .pt checkpoint (defaults to best local v6/v5 checkpoint).',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help='Output directory for policy_actor.ts and deploy_meta.json (defaults to <ckpt_dir>/deploy_export).',
    )
    parser.add_argument('--device', type=str, default='cpu', help='Export & verification device (default: cpu).')
    parser.add_argument('--verify-samples', type=int, default=16, help='Number of random parity checks.')
    parser.add_argument('--atol', type=float, default=1e-5, help='Maximum allowed absolute parity error.')
    args = parser.parse_args(argv)

    ckpt_path = args.checkpoint if args.checkpoint is not None else resolve_default_checkpoint()
    out_dir = args.out_dir if args.out_dir is not None else (ckpt_path.parent / 'deploy_export')

    report = export_checkpoint_to_torchscript(
        ckpt_path,
        out_dir,
        device=args.device,
        verify_samples=args.verify_samples,
        atol=args.atol,
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

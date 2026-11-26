#!/usr/bin/env python3
"""CLI entrypoint for FlexGraph multi-morphology asymmetric PPO + HIM-DreamWaQ training.

Parses CLI flags (including ``--help``) using stdlib ``argparse`` BEFORE importing PyTorch
so ``--help`` and argument validation return instantaneously.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

BUILTIN_ARCHETYPES: tuple[str, ...] = (
    'serpworm_N3',
    'serpworm_N5',
    'bristle_worm',
    'tendon_endoscope',
    'flexspine_crawler',
)


def _fast_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='FlexGraph Multi-Morphology Asymmetric PPO + HIM-DreamWaQ Trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--morphologies',
        type=str,
        default='serpworm_N3,bristle_worm,flexspine_crawler',
        help=f'Comma-separated archetypes from {list(BUILTIN_ARCHETYPES)} or "all".',
    )
    p.add_argument('--schedule', choices=('round_robin', 'mixture'), default='round_robin')
    p.add_argument('--out-dir', type=str, default='runs/flexgraph_multi_v1')
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint (.pt) to resume from.')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--backend', choices=('auto', 'python', 'rust'), default='auto')
    p.add_argument('--envs', type=int, default=64, help='Batch environments per morphology.')
    p.add_argument('--horizon', type=int, default=24, help='Policy steps per rollout.')
    p.add_argument('--rollouts', type=int, default=100, help='Total PPO rollout iterations.')
    p.add_argument('--epochs', type=int, default=4, help='PPO epochs per rollout.')
    p.add_argument('--minibatches', type=int, default=4, help='PPO minibatches per epoch.')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min-lr', type=float, default=5e-5)
    p.add_argument('--max-lr', type=float, default=1e-3)
    p.add_argument('--target-kl', type=float, default=0.012)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--entropy-coef', type=float, default=0.005)
    p.add_argument('--aux-coef', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-attn-layers', type=int, default=2)
    p.add_argument('--history-len', type=int, default=16)
    p.add_argument('--dt', type=float, default=0.001)
    p.add_argument('--substeps', type=int, default=10, help='Physics substeps per policy step.')
    p.add_argument('--max-episode-steps', type=int, default=200)
    p.add_argument('--elements-per-beam', type=int, default=None, help='Override beam element count for speed/fidelity.')
    p.add_argument('--equivariant', action='store_true', help='Enable exact FiniteGroupEquivariantWrapper.')
    p.add_argument('--symmetry-reg', type=float, default=0.15, help='Group symmetry regularization weight (0 disables).')
    p.add_argument('--fault-prob-start', type=float, default=0.0)
    p.add_argument('--fault-prob-end', type=float, default=0.0, help='End probability of actuator fault curriculum.')
    p.add_argument('--fault-max-actuators', type=int, default=1)
    p.add_argument('--amp-bf16', action='store_true', help='Enable CUDA BF16 autocast for policy forward/backward.')
    p.add_argument('--no-obs-norm', action='store_true', help='Disable TokenRunningNorm.')
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--eval-steps', type=int, default=20)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument(
        '--bench-backend',
        action='store_true',
        help='Run backend FFI parity/benchmark diagnostic on the first morphology and exit.',
    )
    return p


if __name__ == '__main__':
    # Parse immediately before heavy PyTorch imports so --help is instant (<0.1s)
    _fast_arg_parser().parse_args()
    SRC = Path(__file__).resolve().parents[1] / 'src'
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from flexgraph.trainer import main

    main()

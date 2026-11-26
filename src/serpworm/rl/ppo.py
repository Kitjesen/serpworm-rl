"""Shared PPO math primitives and atomic serialization helpers."""
from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import torch

__all__ = ['DEFAULT_GAMMA', 'DEFAULT_LAMBDA', 'advantages', 'gradient_norm', 'write_json']

DEFAULT_GAMMA = 0.995 ** 2
DEFAULT_LAMBDA = 0.97


def advantages(
    reward: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float = DEFAULT_GAMMA,
    lam: float = DEFAULT_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    Bootstraps time-limit truncations via ``next_value``, does not bootstrap
    physical terminations, and cuts the eligibility trace across both reset types.
    """
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * (~terminated[t]) * next_value[t] - value[t]
        running = delta + gamma * lam * (~(terminated[t] | truncated[t])) * running
        advantage[t] = running
    return advantage, advantage + value


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Compute the global L2 norm of parameter gradients."""
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(values).sum().sqrt()


def write_json(path: str | Path, payload: dict) -> None:
    """Atomically write strict JSON (disallowing NaN/Infinity)."""
    target = Path(path)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(target)

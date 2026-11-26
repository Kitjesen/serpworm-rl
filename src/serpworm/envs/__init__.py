"""CUDA tensor RL environments, reward objectives, and CPG oscillators."""
from __future__ import annotations

from serpworm.envs.matsuoka import Matsuoka
from serpworm.envs.rewards import UnifiedRewardConfig, locomotion_reward, velocity_reward
from serpworm.envs.tensor_env import STATE, TensorEnv
from serpworm.envs.unified_env import MODE_NAMES, STATE_KEYS, UnifiedSerpWormEnv, mirror_rollout_batch

__all__ = [
    'MODE_NAMES',
    'STATE',
    'STATE_KEYS',
    'Matsuoka',
    'TensorEnv',
    'UnifiedRewardConfig',
    'UnifiedSerpWormEnv',
    'locomotion_reward',
    'mirror_rollout_batch',
    'velocity_reward',
]

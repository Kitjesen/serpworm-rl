"""SerpWorm-RL: CUDA rigid-flexible five-segment steel-strip worm/snake robot simulation and RL."""
from __future__ import annotations

from serpworm.config import DomainRandConfig, HardwareConfig, TaskConfig, UnifiedTrainConfig
from serpworm.paths import ASSETS_DIR, INPUTS_DIR, PKG_ROOT, REPO_ROOT, SCRIPTS_DIR

__version__ = '0.5.0'
__all__ = [
    '__version__',
    'ASSETS_DIR',
    'INPUTS_DIR',
    'PKG_ROOT',
    'REPO_ROOT',
    'SCRIPTS_DIR',
    'HardwareConfig',
    'DomainRandConfig',
    'TaskConfig',
    'UnifiedTrainConfig',
    'GPUCore',
    'build_topology',
    'certified_clearance',
    'is_fused_cuda_available',
    'TensorEnv',
    'UnifiedSerpWormEnv',
    'mirror_rollout_batch',
    'Matsuoka',
    'UnifiedRewardConfig',
    'locomotion_reward',
    'velocity_reward',
    'RunningNorm',
    'ActorCritic',
    'TemporalActorCritic',
    'advantages',
    'train_unified',
    'evaluate_unified',
]

_LAZY = {
    'GPUCore': ('serpworm.physics.gpu_core', 'GPUCore'),
    'build_topology': ('serpworm.physics.topology', 'build_topology'),
    'certified_clearance': ('serpworm.physics.wheel_envelope', 'certified_clearance'),
    'is_fused_cuda_available': ('serpworm.physics.fused_cuda', 'is_available'),
    'TensorEnv': ('serpworm.envs.tensor_env', 'TensorEnv'),
    'UnifiedSerpWormEnv': ('serpworm.envs.unified_env', 'UnifiedSerpWormEnv'),
    'mirror_rollout_batch': ('serpworm.envs.unified_env', 'mirror_rollout_batch'),
    'Matsuoka': ('serpworm.envs.matsuoka', 'Matsuoka'),
    'UnifiedRewardConfig': ('serpworm.envs.rewards', 'UnifiedRewardConfig'),
    'locomotion_reward': ('serpworm.envs.rewards', 'locomotion_reward'),
    'velocity_reward': ('serpworm.envs.rewards', 'velocity_reward'),
    'RunningNorm': ('serpworm.rl.models', 'RunningNorm'),
    'ActorCritic': ('serpworm.rl.models', 'ActorCritic'),
    'TemporalActorCritic': ('serpworm.rl.models', 'TemporalActorCritic'),
    'advantages': ('serpworm.rl.ppo', 'advantages'),
    'train_unified': ('serpworm.rl.train_unified', 'train_unified'),
    'evaluate_unified': ('serpworm.rl.eval_unified', 'evaluate_unified'),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        module = __import__(mod_name, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

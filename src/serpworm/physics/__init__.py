"""Device-resident beam/tendon/wheel physics core, parametric topology, fused CUDA, and safety certificates."""
from __future__ import annotations

from serpworm.physics.fused_cuda import get_extension as get_fused_extension, is_available as is_fused_cuda_available
from serpworm.physics.gpu_core import GPUCore, matrix
from serpworm.physics.topology import build_topology
from serpworm.physics.wheel_envelope import active_pair_indices, certified_clearance
from serpworm.physics.wheel_stiction import contact as implicit_wheel_contact

__all__ = [
    'GPUCore',
    'matrix',
    'build_topology',
    'active_pair_indices',
    'certified_clearance',
    'implicit_wheel_contact',
    'is_fused_cuda_available',
    'get_fused_extension',
]

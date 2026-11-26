"""JIT loader for the lossless FP64 fused CUDA physics extension (V3)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import torch

__all__ = ['is_available', 'get_extension']

_EXTENSION = None
_LOAD_ERROR: Exception | None = None


def is_available() -> bool:
    return torch.cuda.is_available() and (shutil.which('nvcc') is not None or bool(os.environ.get('CUDA_HOME')))


def get_extension():
    """Compile once per kernel source + active GPU arch hash and cache in build directory."""
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise RuntimeError(f'Fused CUDA physics extension unavailable: {_LOAD_ERROR}') from _LOAD_ERROR
    try:
        from torch.utils.cpp_extension import load
        if not os.environ.get('TORCH_CUDA_ARCH_LIST') and torch.cuda.is_available():
            maj, min_ = torch.cuda.get_device_capability()
            os.environ['TORCH_CUDA_ARCH_LIST'] = f'{maj}.{min_}'
        src = Path(__file__).resolve().parent / 'csrc' / 'fused_physics.cu'
        arch_tag = os.environ.get('TORCH_CUDA_ARCH_LIST', '')
        hasher = hashlib.sha256(src.read_bytes())
        hasher.update(arch_tag.encode('utf-8'))
        digest = hasher.hexdigest()[:10]
        name = f'serpworm_fused_{digest}'
        build_dir = Path(os.environ.get('SERPWORM_CUDA_CACHE', Path.home() / '.cache' / 'serpworm_cuda')) / name
        build_dir.mkdir(parents=True, exist_ok=True)
        _EXTENSION = load(
            name=name,
            sources=[str(src)],
            build_directory=str(build_dir),
            extra_cuda_cflags=['-O3'],
            verbose=False,
        )
        return _EXTENSION
    except Exception as exc:
        _LOAD_ERROR = exc
        raise RuntimeError(f'Failed to build fused CUDA physics extension: {exc}') from exc

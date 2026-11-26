"""Central repository paths and reproducible source/input hash manifests."""
from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = [
    'PKG_ROOT',
    'REPO_ROOT',
    'SCRIPTS_DIR',
    'ASSETS_DIR',
    'INPUTS_DIR',
    'PHYSICS_SOURCES',
    'EVAL_SOURCES',
    'UNIFIED_SOURCES',
    'sha256_file',
    'resolve_inputs_dir',
    'collect_source_hashes',
    'collect_input_hashes',
    'freeze_sources',
]

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parents[1]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
ASSETS_DIR = REPO_ROOT / 'assets'
INPUTS_DIR = ASSETS_DIR / 'inputs'

PHYSICS_SOURCES = (
    'physics/gpu_core.py',
    'physics/wheel_stiction.py',
    'physics/wheel_envelope.py',
    'envs/tensor_env.py',
    'envs/rewards.py',
)

EVAL_SOURCES = (
    *PHYSICS_SOURCES,
    'rl/models.py',
    'rl/ppo.py',
    'rl/train_ppo.py',
    'rl/evaluate.py',
)

UNIFIED_SOURCES = (
    'config.py',
    'physics/topology.py',
    'physics/gpu_core.py',
    'physics/fused_cuda.py',
    'physics/wheel_stiction.py',
    'physics/wheel_envelope.py',
    'envs/rewards.py',
    'envs/unified_env.py',
    'rl/models.py',
    'rl/ppo.py',
    'rl/train_unified.py',
    'rl/eval_unified.py',
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_inputs_dir(source_root: Path = PKG_ROOT, inputs_dir: Path | None = None) -> Path:
    if inputs_dir is not None:
        return Path(inputs_dir)
    local = Path(source_root) / 'inputs'
    if local.is_dir():
        return local
    return INPUTS_DIR


def collect_source_hashes(source_root: Path = PKG_ROOT, names: tuple[str, ...] | None = None) -> dict[str, str]:
    root = Path(source_root)
    if names is not None:
        return {name: sha256_file(root / name) for name in names}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob('*.py'))
        if '__pycache__' not in path.parts
    }


def collect_input_hashes(source_root: Path = PKG_ROOT, inputs_dir: Path | None = None) -> dict[str, str]:
    directory = resolve_inputs_dir(source_root, inputs_dir)
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.glob('*'))
        if path.is_file()
    }


def freeze_sources(source_root: Path, out: Path, manifest: dict, inputs_dir: Path | None = None) -> None:
    """Archive the exact bytes behind each recorded source and input hash."""
    root = Path(source_root)
    in_dir = resolve_inputs_dir(root, inputs_dir)
    snapshot = Path(out) / 'sources'
    snapshot.mkdir(exist_ok=True)
    for name, digest in manifest['source_sha256'].items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'Source changed while freezing: {name}')
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name, digest in manifest['input_sha256'].items():
        data = (in_dir / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'Source changed while freezing: inputs/{name}')
        target = snapshot / 'inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

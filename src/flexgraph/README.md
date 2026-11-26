# `flexgraph` — Generic Graph-Compiled Rigid-Flexible Robotics Framework

Independent, zero-regression generalization of the `SerpWorm-RL` bipartite physics engine, unified anisotropic contact, composable SDF environments, and morphology-agnostic proprioceptive control stack.

## Module Layout

| Module | Layer | Responsibility |
| :--- | :---: | :--- |
| [`math3d.py`](math3d.py) | Math | FP64 SO(3)/SE(3) quaternions, midpoint spline corotational frames, heterogeneous rectangular/circular/custom 12×12 beam stiffness & mass matrices |
| [`graph.py`](graph.py) | Layer 1 | Declarative `RigidFlexGraph` IR, JSON round-trip, automatic sparsity router (`block_diagonal_pairs` / `tree_chain` / `sparse_general`), symmetry group generators ($\mathbb{Z}_2$, $\mathbb{Z}_2\times\mathbb{Z}_2$, $D_4$), and 4 built-in robot builders |
| [`sdf_world.py`](sdf_world.py) | Layer 2 | `ComposableSDFWorld` (`HeightFieldSDF`, 3D variable-radius `SplineTubeSDF`, `BoxObstacleSDF`, trilinear `VoxelGridSDF`) + material property library |
| [`contact.py`](contact.py) | Layer 2 | Unified FP64 `AnisotropicPatchContact` (`ratchet_wheel`, `anisotropic_scale`, `isotropic_pad`, Hunt–Crossley + Bekker rut + Stribeck + pivot) and `Py-DiSMech` $C^2$ smooth self-contact |
| [`solver.py`](solver.py) | Layer 3 | Lossless FP64 `GraphSchurSolver`: batched 6×6 Block-Thomas (clamped-clamped & free cantilever) + adaptive rigid Schur solve + Dirichlet fixed bases |
| [`policy.py`](policy.py) | Layer 4 | Morphology-agnostic `TokenizedHIMDreamWaQ` (variable actuator count $N$ + fault masking) and `FiniteGroupEquivariantWrapper` (exact Reynolds averaging for arbitrary finite groups) |
| [`env.py`](env.py) | Layers 4–5 | Closed-loop `FlexGraphEnv` (entity tokens with universal `q_err`, live fault injection, SDF arc progress) + `UniversalSafetyGuard` + `RobotManifest` export |

## Quick Start

```python
import torch
from flexgraph import (
    ComposableSDFWorld,
    FiniteGroupEquivariantWrapper,
    FlexGraphEnv,
    HeightFieldSDF,
    TokenizedHIMDreamWaQ,
    build_bristle_worm_graph,
    build_tendon_endoscope_graph,
)

policy = TokenizedHIMDreamWaQ().eval()

# Run the exact same policy across completely different robot morphologies:
for graph in (build_bristle_worm_graph(n_segments=3), build_tendon_endoscope_graph(n_sections=3)):
    env = FlexGraphEnv(graph, ComposableSDFWorld([HeightFieldSDF()]), substeps_per_policy=10)
    tok, glob, mask = env.reset()
    with torch.no_grad():
        act = policy(tok, glob, mask).action_mean
    _, reward, info = env.step(act)
```

## Verification

```bash
pytest tests/test_flexgraph.py -v
```

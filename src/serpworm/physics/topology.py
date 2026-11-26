"""SOFA-free parametric topology generator for N-segment pre-curved steel-strip robots.

Reproduces the exact parabolic pre-curved ribbon geometry, clamp hole frames,
tendon anchors/guides, and passive wheel mounts of ``assets/inputs/topology_{1,2,5}.npz``
directly from ``parameters.json``, while supporting arbitrary ``count`` and
``elements_per_strip`` (e.g. ``E = 6, 8, 12, 16, 32``) for mesh-convergence and
GPU acceleration studies without requiring a SOFA installation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from serpworm.paths import INPUTS_DIR

__all__ = ['WHEEL_HOLES_M', 'WHEEL_DROP_M', 'WHEEL_RADIUS_M', 'build_topology']

WHEEL_RADIUS_M = 0.018
WHEEL_DROP_M = np.array([0.0, 0.0, -0.045], dtype=np.float64)
WHEEL_HOLES_M = np.array(
    [
        [
            [-0.014, 0.0617601286221, -0.0689776402648],
            [-0.014, -0.0617601286221, -0.0689776402648],
        ],
        [
            [-0.0005, 0.0617601286221, -0.0700229145182],
            [-0.0005, -0.0617601286221, -0.0700229145182],
        ],
    ],
    dtype=np.float64,
)


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 proper rotation matrix to unit quaternion [x, y, z, w] with w >= 0."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return q


def build_topology(
    count: int = 5,
    elements_per_strip: int = 16,
    parameters: dict | str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Generate parametric robot topology arrays without SOFA.

    Parameters
    ----------
    count : int
        Number of body segments (each has 2 rigid plates and 8 pre-curved steel strips).
    elements_per_strip : int
        Number of corotational beam elements per steel strip (>= 2).
    parameters : dict | Path | None
        Parsed ``parameters.json`` dictionary or path (defaults to ``INPUTS_DIR / 'parameters.json'``).
    """
    if count < 1:
        raise ValueError('count must be >= 1')
    if elements_per_strip < 2:
        raise ValueError('elements_per_strip must be >= 2')

    if parameters is None:
        params = json.loads((INPUTS_DIR / 'parameters.json').read_text(encoding='utf-8'))
    elif isinstance(parameters, (str, Path)):
        params = json.loads(Path(parameters).read_text(encoding='utf-8'))
    else:
        params = parameters

    wheel_mounts = WHEEL_HOLES_M + WHEEL_DROP_M
    z0 = -float(wheel_mounts[0, 0, 2]) + WHEEL_RADIUS_M + 0.001
    poses: list[list[float]] = [
        [-i * 0.1755 - j * 0.1175, 0.0, z0 + j * 0.0010452742534, 0.0, 0.0, 0.0, 1.0]
        for i in range(count)
        for j in (0, 1)
    ]

    front_pairs = params['front_clamp_hole_pairs_m']
    back_pairs = params['back_clamp_hole_pairs_m']
    bow = float(params['stress_free_bow_m'])

    edges: list[list[int]] = []
    lengths: list[float] = []
    strip_nodes: list[list[int]] = []
    transforms = np.zeros((8 * count, elements_per_strip, 2, 7), dtype=np.float64)
    transforms[..., 6] = 1.0

    us = np.linspace(0.0, 1.0, elements_per_strip + 1)
    for seg in range(count):
        for strip in range(8):
            sid = seg * 8 + strip
            front = np.mean(front_pairs[strip], axis=0)
            back = np.mean(back_pairs[strip], axis=0)
            a = np.asarray(poses[2 * seg][:3], dtype=np.float64) + front
            b = np.asarray(poses[2 * seg + 1][:3], dtype=np.float64) + back

            radial = np.array([0.0, front[1], front[2]], dtype=np.float64)
            radial /= np.linalg.norm(radial)
            side = np.subtract(*front_pairs[strip]).astype(np.float64)
            side /= np.linalg.norm(side)

            nodes: list[int] = []
            frames: list[list[float]] = []
            for j, u in enumerate(us):
                point = a + u * (b - a) + 4.0 * bow * u * (1.0 - u) * radial
                tangent = b - a + 4.0 * bow * (1.0 - 2.0 * u) * radial
                tangent /= np.linalg.norm(tangent)
                width_axis = side - np.dot(side, tangent) * tangent
                width_axis /= np.linalg.norm(width_axis)
                normal_axis = np.cross(tangent, width_axis)
                q = _mat_to_quat_xyzw(np.column_stack((tangent, width_axis, normal_axis)))
                frame = [*point.tolist(), *q.tolist()]
                frames.append(frame)
                if j == 0:
                    nodes.append(2 * seg)
                elif j == elements_per_strip:
                    nodes.append(2 * seg + 1)
                else:
                    nodes.append(len(poses))
                    poses.append(frame)

            for j in range(elements_per_strip):
                edges.append([nodes[j], nodes[j + 1]])
                p0 = np.asarray(frames[j][:3], dtype=np.float64)
                p1 = np.asarray(frames[j + 1][:3], dtype=np.float64)
                lengths.append(float(np.linalg.norm(p1 - p0)))

            transforms[sid, 0, 0] = [*front.tolist(), *frames[0][3:]]
            transforms[sid, elements_per_strip - 1, 1] = [*back.tolist(), *frames[-1][3:]]
            strip_nodes.append(nodes)

    anchors = np.array(
        [[-0.0165, y, zz] for y in (0.0285, -0.0285) for zz in (0.039, -0.039)],
        dtype=np.float64,
    )
    guides = anchors.copy()
    guides[:, 0] = 0.0

    return {
        'rest': np.asarray(poses, dtype=np.float64),
        'edges': np.asarray(edges, dtype=np.int32),
        'lengths': np.asarray(lengths, dtype=np.float64),
        'transforms': transforms,
        'strip_nodes': np.asarray(strip_nodes, dtype=np.int64),
        'wheel_mounts': wheel_mounts,
        'anchors': anchors,
        'guides': guides,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Generate SOFA-free parametric topology .npz')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--elements', type=int, default=16)
    parser.add_argument('--parameters', type=Path, default=INPUTS_DIR / 'parameters.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    topo = build_topology(count=args.count, elements_per_strip=args.elements, parameters=args.parameters)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **topo)
    print(f"Saved topology (count={args.count}, E={args.elements}, D={topo['rest'].shape[0]}) -> {args.out}")


if __name__ == '__main__':
    main()

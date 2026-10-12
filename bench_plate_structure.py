"""Check and time the existing five-segment Schur system without changing physics.

Run: python bench_plate_structure.py --device cpu --batch 2 --out results/plate_structure_cpu.json
CUDA timings cover this linear solve only, not end-to-end environment sampling.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from gpu_core import GPUCore


def collect_system(core):
    """Capture the real matrices passed by GPUCore.solve(), preserving its solve."""
    calls = []
    original = torch.linalg.solve_ex

    def record(a, b, **kwargs):
        calls.append((a.detach().clone(), b.detach().clone()))
        return original(a, b, **kwargs)

    with patch.object(torch.linalg, 'solve_ex', record):
        core.physics()
    assert calls, 'GPUCore no longer calls solve_ex; update this benchmark.'
    return torch.cat([a for a, _ in calls]), torch.cat([b for _, b in calls])


def solve_chunked(a, b):
    chunks = [torch.linalg.solve_ex(a[i:i+8], b[i:i+8], check_errors=False)
              for i in range(0, len(a), 8)]
    return torch.cat([x.result for x in chunks]), torch.cat([x.info for x in chunks])


def solve_cholesky(a, b):
    factor, info = torch.linalg.cholesky_ex(a, check_errors=False)
    return torch.cholesky_solve(b, factor), info


def solve_segment_cholesky(a, b):
    batch = len(a)
    blocks = a.reshape(batch, 5, 12, 5, 12).permute(0, 1, 3, 2, 4)
    ids = torch.arange(5, device=a.device)
    local = blocks[:, ids, ids].reshape(batch*5, 12, 12)
    x, info = solve_cholesky(local, b.reshape(batch*5, 12, 1))
    return x.reshape(batch, 60, 1), info


def solve_segment_lu(a, b):
    batch = len(a)
    blocks = a.reshape(batch, 5, 12, 5, 12).permute(0, 1, 3, 2, 4)
    ids = torch.arange(5, device=a.device)
    local = blocks[:, ids, ids].reshape(batch*5, 12, 12)
    x, info = torch.linalg.solve_ex(local, b.reshape(batch*5, 12, 1), check_errors=False)
    return x.reshape(batch, 60, 1), info


def verify(a, b, x, info, reference):
    residual = torch.linalg.vector_norm(a@x-b, dim=(-2, -1))
    scale = (torch.linalg.matrix_norm(a)*torch.linalg.vector_norm(x, dim=(-2, -1))
             + torch.linalg.vector_norm(b, dim=(-2, -1))).clamp_min(1e-30)
    difference = torch.linalg.vector_norm(x-reference, dim=(-2, -1))
    relative = difference/torch.linalg.vector_norm(reference, dim=(-2, -1)).clamp_min(1e-30)
    report = dict(factorization_failures=int((info != 0).sum()),
                  finite=bool(torch.isfinite(x).all()),
                  max_abs_velocity_difference=float((x-reference).abs().max()),
                  max_relative_velocity_difference=float(relative.max()),
                  max_normalized_linear_residual=float((residual/scale).max()))
    report['accepted'] = (report['factorization_failures'] == 0 and report['finite']
                          and report['max_relative_velocity_difference'] < 1e-9
                          and report['max_normalized_linear_residual'] < 1e-10)
    report['criteria'] = 'finite; factorization info=0; relative solution difference<1e-9; backward error<1e-10'
    return report


def timing(fn, a, b, repeats, cuda):
    for _ in range(3):
        fn(a, b)
    if cuda:
        torch.cuda.synchronize()
    wall = []
    device = []
    for _ in range(repeats):
        if cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        start = time.perf_counter()
        fn(a, b)
        if cuda:
            end.record()
            end.synchronize()
            device.append(begin.elapsed_time(end))
        wall.append((time.perf_counter()-start)*1000)
    return dict(wall_median_ms=statistics.median(wall),
                gpu_median_ms=statistics.median(device) if device else None,
                repeats=repeats)


def graph_check(fn, a, b, reference, repeats):
    # Capture is tested separately: a successful eager solve does not prove support.
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(a, b)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            x, info = fn(a, b)
        graph.replay()
        torch.cuda.synchronize()
        checked = verify(a, b, x, info, reference)
        measured = timing(lambda _a, _b: graph.replay(), a, b, repeats, True)
        return dict(supported=True, validation=checked, timing=measured)
    except RuntimeError as exc:
        return dict(supported=False, error=str(exc))


def run(device, batch, repeats, state_steps):
    if batch < 1 or repeats < 1 or state_steps < 0:
        raise ValueError('batch/repeats must be positive; state_steps must be nonnegative')
    torch.set_num_threads(1)
    core = GPUCore(count=5, batch=batch, device=device, strain_jacobian=True)
    pairs, counts = torch.unique(core.nodes[:, [0, -1]], dim=0, return_counts=True)
    expected = torch.arange(10, device=core.device).reshape(5, 2)
    assert torch.equal(pairs, expected) and bool((counts == 8).all()), (pairs, counts)
    core.target.fill_(.1)
    core.yaw.copy_(core.tensor([.03, -.03, .03, -.03]))
    for _ in range(state_steps):
        core.physics()
    a, b = collect_system(core)
    assert a.shape == (batch, 60, 60) and b.shape == (batch, 60, 1)
    blocks = a.reshape(batch, 5, 12, 5, 12).permute(0, 1, 3, 2, 4)
    mask = ~torch.eye(5, device=a.device, dtype=torch.bool)
    off_diagonal = float(blocks[:, mask].abs().max())
    assert off_diagonal == 0., 'Inter-segment implicit coupling now exists: decomposition is invalid.'
    ids = torch.arange(5, device=a.device)
    local = blocks[:, ids, ids].detach().cpu()
    relative_asymmetry = float(torch.linalg.matrix_norm(local-local.mT).div(torch.linalg.matrix_norm(local)).max())
    symmetric = .5*(local+local.mT)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    diagonal_scale = symmetric.diagonal(dim1=-2, dim2=-1).rsqrt()
    scaled = symmetric*diagonal_scale[..., :, None]*diagonal_scale[..., None, :]
    scaled_eigenvalues = torch.linalg.eigvalsh(scaled)
    reference, info = solve_chunked(a, b)
    report = dict(device=device, torch_version=torch.__version__, dtype=str(a.dtype), batch=batch,
                  gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                  state_steps_before_snapshot=state_steps, dt_s=core.dt,
                  endpoint_pairs=pairs.cpu().tolist(), strips_per_pair=counts.cpu().tolist(),
                  inter_segment_block_max_abs=off_diagonal, relative_asymmetry=relative_asymmetry,
                  min_local_eigenvalue=float(eigenvalues.min()),
                  max_local_condition_raw=float((eigenvalues[..., -1]/eigenvalues[..., 0]).max()),
                  max_local_condition_diagonal_scaled=float((scaled_eigenvalues[..., -1]/scaled_eigenvalues[..., 0]).max()),
                  note='Raw condition and velocity norms mix SI translational/rotational units. Frozen-system microbenchmark only; explicit inter-segment coupling is unchanged. No symmetrization is applied to solve candidates; Cholesky uses the lower triangle and is checked against the original matrix.',
                  candidates={})
    report['reference_validation'] = verify(a, b, reference, info, reference)
    for name, fn in [('chunked_60_lu', solve_chunked), ('batch_60_cholesky', solve_cholesky),
                     ('batch_12_cholesky', solve_segment_cholesky), ('batch_12_lu', solve_segment_lu)]:
        x, info = fn(a, b)
        entry = dict(validation=verify(a, b, x, info, reference),
                     eager=timing(fn, a, b, repeats, device == 'cuda'))
        entry['cuda_graph'] = graph_check(fn, a, b, reference, repeats) if device == 'cuda' else dict(tested=False)
        report['candidates'][name] = entry
    report['passed'] = all(entry['validation']['accepted'] for entry in report['candidates'].values())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--state-steps', type=int, default=8)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {path}')
    result = run(args.device, args.batch, args.repeats, args.state_steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

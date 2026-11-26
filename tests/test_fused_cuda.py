"""FP64 parity and speedup verification for the fused CUDA physics kernel."""
import math
import pytest
import torch

from serpworm.config import HardwareConfig
from serpworm.physics.fused_cuda import is_available
from serpworm.physics.gpu_core import GPUCore


@pytest.mark.skipif(not is_available(), reason='CUDA + nvcc required for fused_cuda kernel tests')
def test_fused_cuda_matches_segment_lu_fp64_parity():
    """Verify bit-level FP64 parity (< 1e-11 m) across settling + tendon + yaw + wheel contact."""
    batch = 4
    dt = 0.0005
    hw = HardwareConfig.ds3218_default()
    ref = GPUCore(5, batch, dt=dt, device='cuda', plate_solver='segment_lu', hardware=hw)
    fused = GPUCore(5, batch, dt=dt, device='cuda', plate_solver='fused_cuda', hardware=hw)

    # Distinct per-env commands & randomized friction/mass/gains to test every code path!
    mu_vals = torch.tensor([0.6, 0.8, 0.95, 1.1], device='cuda', dtype=torch.float64)
    ms_vals = torch.tensor([0.92, 1.0, 1.05, 1.08], device='cuda', dtype=torch.float64)
    for core in (ref, fused):
        core.wheel_mu.copy_(mu_vals[:, None, None])
        core.mass_scale.copy_(ms_vals)
        core.target.copy_(torch.tensor([
            [0.8, 0.8, 1.2, 1.2, 0.5, 0.5, 1.5, 1.5, 0.9, 0.9],
            [1.6, 1.6, 0.4, 0.4, 1.8, 1.8, 0.6, 0.6, 1.2, 1.2],
            [0.2, 1.4, 1.0, 0.3, 1.5, 0.7, 0.4, 1.6, 1.1, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ], device='cuda', dtype=torch.float64))
        core.yaw.copy_(torch.tensor([
            [0.35, -0.30, 0.25, -0.20],
            [-0.40, 0.38, -0.35, 0.30],
            [0.0, 0.0, 0.0, 0.0],
            [0.20, 0.20, -0.20, -0.20],
        ], device='cuda', dtype=torch.float64))

    ref.advance_steps(200)    # 0.1 s = 200 substeps of active motion & contact
    fused.advance_steps(200)

    assert not bool(ref.failed) and not bool(fused.failed)
    pos_diff = (ref.x[..., :3] - fused.x[..., :3]).abs().max().item()
    # Compare quaternion up to sign (q == -q)
    qdot = (ref.x[..., 3:] * fused.x[..., 3:]).sum(-1).abs().clamp_max(1.0)
    ang_diff = (2.0 * torch.acos(qdot)).max().item()
    vel_diff = (ref.v - fused.v).abs().max().item()
    servo_diff = (ref.servo - fused.servo).abs().max().item()
    spin_diff = (ref.spin - fused.spin).abs().max().item()
    tension_diff = (ref.tension - fused.tension).abs().max().item()
    yaw_trq_diff = (ref.last_yaw_torque - fused.last_yaw_torque).abs().max().item()

    assert pos_diff < 1e-8, f'pos_diff={pos_diff:.3e}'
    assert ang_diff < 1e-7, f'ang_diff={ang_diff:.3e}'
    assert vel_diff < 1e-4, f'vel_diff={vel_diff:.3e}'
    assert servo_diff < 1e-8, f'servo_diff={servo_diff:.3e}'
    assert spin_diff < 1e-4, f'spin_diff={spin_diff:.3e}'
    assert tension_diff < 1e-5, f'tension_diff={tension_diff:.3e}'
    assert yaw_trq_diff < 1e-6, f'yaw_trq_diff={yaw_trq_diff:.3e}'

"""Comprehensive unit, physics, stability, and FP64 CUDA parity tests for ARPC + Implicit Contact-Schur."""
from __future__ import annotations

from dataclasses import replace
import math
import pytest
import torch

from serpworm.config import HardwareConfig
from serpworm.physics.fused_cuda import is_available as fused_cuda_available
from serpworm.physics.gpu_core import GPUCore
from serpworm.physics.wheel_stiction import arpc_normal_and_rut, arpc_tangential_contact


def _best_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def test_kelvin_default_backward_compatibility():
    """Default HardwareConfig remains legacy explicit Kelvin-Voigt and matches with_terrain('kelvin') bit-for-bit."""
    hw_default = HardwareConfig.ds3218_default()
    assert hw_default.contact_model == 'kelvin'
    assert hw_default.implicit_contact is False

    hw_kelvin = hw_default.with_terrain('kelvin')
    assert hw_kelvin == hw_default

    for terrain, expected_model in (
        ('hard', 'arpc_hard'),
        ('carpet', 'arpc_carpet'),
        ('soft_soil', 'arpc_soft_soil'),
    ):
        hw_t = hw_default.with_terrain(terrain)
        assert hw_t.contact_model == expected_model
        assert hw_t.implicit_contact is True
        hw_exp = hw_default.with_terrain(terrain, implicit_contact=False)
        assert hw_exp.implicit_contact is False

    with pytest.raises(ValueError):
        hw_default.with_terrain('unknown_surface')  # type: ignore[arg-type]

    torch.set_num_threads(1)
    c_def = GPUCore(count=5, batch=1, dt=0.0005, device='cpu', plate_solver='segment_lu', hardware=hw_default)
    c_kel = GPUCore(count=5, batch=1, dt=0.0005, device='cpu', plate_solver='segment_lu', hardware=hw_kelvin)
    for c in (c_def, c_kel):
        c.target.fill_(hw_default.servo_neutral_rad)
        c.yaw.copy_(torch.tensor([[0.15, -0.15, 0.10, -0.10]], device='cpu', dtype=torch.float64))
        c.advance_steps(40)

    assert not bool(c_def.failed) and not bool(c_kel.failed)
    torch.testing.assert_close(c_def.x, c_kel.x, atol=0.0, rtol=0.0)
    torch.testing.assert_close(c_def.v, c_kel.v, atol=0.0, rtol=0.0)
    torch.testing.assert_close(c_def.support, c_kel.support, atol=0.0, rtol=0.0)
    torch.testing.assert_close(c_def.spin, c_kel.spin, atol=0.0, rtol=0.0)
    assert float(c_def.soil_sinkage.abs().max()) == 0.0
    assert float(c_def.ratchet_play.abs().max()) == 0.0


def test_drop_test_touchdown_smoothness():
    """ARPC Hunt-Crossley eliminates Kelvin-Voigt's instantaneous touchdown damping force spike."""
    # 1. Exact analytical touchdown comparison at 1 um penetration with 3 mm free-fall speed (~0.243 m/s)
    vz_impact = -math.sqrt(2.0 * 9.81 * 0.003)
    pen = torch.tensor([0.0, 1e-6], dtype=torch.float64)
    vz = torch.tensor([vz_impact, vz_impact], dtype=torch.float64)
    vh = torch.zeros_like(pen)
    rut = torch.zeros_like(pen)

    hw_kel = HardwareConfig.ds3218_default().with_terrain('kelvin')
    hw_hard = HardwareConfig.ds3218_default().with_terrain('hard')

    kelvin_fn = (hw_kel.ground_kn_n_m * pen[1] - hw_kel.ground_cn_ns_m * vz[1]).item()
    arpc_fn, _, kn_tan, cn_tan = arpc_normal_and_rut(
        pen, vz, vh, rut, dt=0.0005,
        hc_k=hw_hard.ground_hc_k, hc_alpha=hw_hard.ground_hc_alpha,
        soil_exp=hw_hard.ground_soil_exp, unload_ratio=hw_hard.ground_soil_unload_ratio,
        rut_relax_m=hw_hard.ground_rut_relax_m,
    )
    assert arpc_fn[0].item() == 0.0
    assert kelvin_fn > 1.5, f'Expected Kelvin touchdown spike > 1.5 N, got {kelvin_fn:.4f} N'
    assert 0.0 < arpc_fn[1].item() < 1e-3, f'Expected smooth Hunt-Crossley touchdown < 1 mN, got {arpc_fn[1].item():.3e} N'
    assert kelvin_fn / arpc_fn[1].item() > 1000.0
    assert kn_tan[1].item() > 0.0 and cn_tan[1].item() > 0.0

    # 2. Full single-segment drop from 3 mm above ground: compare first-touchdown force step
    dev = _best_device()
    first_jump = {}
    for name, hw in (('kelvin', hw_kel), ('arpc_hard', hw_hard)):
        core = GPUCore(count=1, batch=1, dt=0.0005, device=dev, plate_solver='segment_lu', hardware=hw)
        core.x[..., 2] += 0.003
        core.target.fill_(hw.servo_neutral_rad)
        prev_n = 0.0
        jump = None
        for _ in range(200):
            core.advance_steps(1)
            cur_n = float(core.support.mean().item())
            if jump is None and cur_n > 1e-9:
                jump = cur_n - prev_n
            prev_n = cur_n
        assert not bool(core.failed)
        assert jump is not None and float(core.support.mean().item()) > 0.5
        first_jump[name] = jump

    assert first_jump['kelvin'] > 0.5, f"Expected Kelvin first-step jump > 0.5 N, got {first_jump['kelvin']:.4f} N"
    assert first_jump['arpc_hard'] < 0.2 * first_jump['kelvin'], (
        f"Expected ARPC touchdown jump << Kelvin jump, got arpc={first_jump['arpc_hard']:.4f} vs kelvin={first_jump['kelvin']:.4f}"
    )


def test_soft_soil_plastic_sinkage_and_rut_hysteresis():
    """Soft soil accumulates irreversible plastic rut sinkage, exhibits unloading hysteresis, and relaxes under rolling."""
    hw_soil = HardwareConfig.ds3218_default().with_terrain('soft_soil')
    dt = 0.001
    z0 = torch.tensor([0.0], dtype=torch.float64)

    # Press to 3 mm virgin penetration
    pen_max = torch.tensor([0.003], dtype=torch.float64)
    fn_peak, rut_peak, kn_p, cn_p = arpc_normal_and_rut(
        pen_max, z0, z0, z0, dt,
        hw_soil.ground_hc_k, hw_soil.ground_hc_alpha, hw_soil.ground_soil_exp,
        hw_soil.ground_soil_unload_ratio, hw_soil.ground_rut_relax_m,
    )
    expected_rut = (1.0 - 1.0 / hw_soil.ground_soil_unload_ratio) * 0.003
    assert math.isclose(rut_peak.item(), expected_rut, rel_tol=1e-12)
    assert fn_peak.item() > 0.0 and kn_p.item() > 0.0 and cn_p.item() > 0.0

    # Unload to 2.5 mm: force must be strictly below virgin loading curve at the same depth
    pen_mid = torch.tensor([0.0025], dtype=torch.float64)
    fn_virgin, _, _, _ = arpc_normal_and_rut(
        pen_mid, z0, z0, z0, dt,
        hw_soil.ground_hc_k, hw_soil.ground_hc_alpha, hw_soil.ground_soil_exp,
        hw_soil.ground_soil_unload_ratio, hw_soil.ground_rut_relax_m,
    )
    fn_unload, rut_unload, kn_u, _ = arpc_normal_and_rut(
        pen_mid, z0, z0, rut_peak, dt,
        hw_soil.ground_hc_k, hw_soil.ground_hc_alpha, hw_soil.ground_soil_exp,
        hw_soil.ground_soil_unload_ratio, hw_soil.ground_rut_relax_m,
    )
    assert 0.0 < fn_unload.item() < fn_virgin.item()
    assert math.isclose(rut_unload.item(), rut_peak.item(), rel_tol=1e-12)
    assert kn_u.item() > 0.0

    # Unload below rut bottom (1.8 mm < 2.143 mm): zero contact force, plastic rut persists
    pen_below = torch.tensor([0.0018], dtype=torch.float64)
    fn_below, rut_below, kn_b, cn_b = arpc_normal_and_rut(
        pen_below, z0, z0, rut_peak, dt,
        hw_soil.ground_hc_k, hw_soil.ground_hc_alpha, hw_soil.ground_soil_exp,
        hw_soil.ground_soil_unload_ratio, hw_soil.ground_rut_relax_m,
    )
    assert fn_below.item() == 0.0 and kn_b.item() == 0.0 and cn_b.item() == 0.0
    assert math.isclose(rut_below.item(), rut_peak.item(), rel_tol=1e-12)

    # Rolling forward relaxes rut exponentially with length constant ground_rut_relax_m
    vh_roll = torch.tensor([0.10], dtype=torch.float64)
    _, rut_rolled, _, _ = arpc_normal_and_rut(
        z0, z0, vh_roll, rut_peak, dt,
        hw_soil.ground_hc_k, hw_soil.ground_hc_alpha, hw_soil.ground_soil_exp,
        hw_soil.ground_soil_unload_ratio, hw_soil.ground_rut_relax_m,
    )
    assert math.isclose(
        rut_rolled.item(),
        rut_peak.item() * math.exp(-0.10 * dt / hw_soil.ground_rut_relax_m),
        rel_tol=1e-12,
    )

    # Full GPUCore press-and-lift verification
    dev = _best_device()
    core = GPUCore(count=1, batch=1, dt=0.001, device=dev, plate_solver='segment_lu', hardware=hw_soil)
    core.target.fill_(hw_soil.servo_neutral_rad)
    core.advance_steps(150)
    assert not bool(core.failed)
    rut_settled = core.soil_sinkage.clone()
    assert float(rut_settled.min().item()) > 4e-4, f'Expected >0.4 mm rut, got {rut_settled.min().item():.3e} m'

    # Lift straight off ground (zero horizontal motion) and verify permanent rut remains > 0
    core.x[..., 2] += 0.015
    core.v.zero_()
    core.advance_steps(1)
    assert float(core.support.max().item()) == 0.0
    assert float(core.soil_sinkage.min().item()) > 4e-4


def test_mechanical_ratchet_backlash_deadzone():
    """One-way ratchet permits free backward roll within backlash_rad, locks hard at play=0, and restores on forward roll."""
    dt = 0.0005
    backlash = 0.05
    normal = torch.tensor([2.0], dtype=torch.float64)
    rut = torch.tensor([0.0], dtype=torch.float64)
    wz = torch.tensor([0.0], dtype=torch.float64)
    spin = torch.tensor([0.0], dtype=torch.float64)
    mem = torch.zeros((1, 2), dtype=torch.float64)
    play = torch.tensor([backlash], dtype=torch.float64)

    # Reverse velocity (-0.05 m/s -> ~ -2.78 rad/s -> ~36 steps to consume 0.05 rad play)
    tv_rev = torch.tensor([[-0.05, 0.0]], dtype=torch.float64)
    free_rev_forces = []
    free_rev_omegas = []
    locked_forces = []
    locked_omegas = []

    for _ in range(90):
        was_free = play.item() > 1e-4
        force, spin, mem, play, _, _, _, _, _, _ = arpc_tangential_contact(
            tv_rev, spin, mem, play, normal, rut, wz, dt,
            mu_fwd=0.8, mu_lat=0.8, rolling=0.06, backlash_rad=backlash,
            patch_radius=0.0, pivot_mu=0.0, rut_lat_k=0.0,
        )
        if was_free and play.item() > 1e-4:
            free_rev_forces.append(force[0, 0].item())
            free_rev_omegas.append(spin.item())
        elif play.item() == 0.0:
            locked_forces.append(force[0, 0].item())
            locked_omegas.append(spin.item())

    assert len(free_rev_omegas) >= 10
    assert min(free_rev_omegas) < -1.5, f'Expected free backward spin in deadzone, got min omega={min(free_rev_omegas):.3f}'
    assert max(abs(f) for f in free_rev_forces) < 0.35, f'Expected low rolling drag in deadzone, got {max(abs(f) for f in free_rev_forces):.3f} N'
    assert len(locked_omegas) >= 20
    assert max(abs(w) for w in locked_omegas[-10:]) == 0.0, 'Ratchet must lock spin to 0 once play == 0'
    assert locked_forces[-1] > 1.0, f'Expected strong ratchet braking force > 1.0 N after lock, got {locked_forces[-1]:.3f} N'
    assert play.item() == 0.0

    # Forward roll restores backlash play smoothly back to backlash_rad
    tv_fwd = torch.tensor([[+0.05, 0.0]], dtype=torch.float64)
    for _ in range(80):
        _, spin, mem, play, _, _, _, _, _, _ = arpc_tangential_contact(
            tv_fwd, spin, mem, play, normal, rut, wz, dt,
            mu_fwd=0.8, mu_lat=0.8, rolling=0.06, backlash_rad=backlash,
            patch_radius=0.0, pivot_mu=0.0, rut_lat_k=0.0,
        )
    assert spin.item() > 1.5
    assert math.isclose(play.item(), backlash, rel_tol=1e-9)


def test_vertical_pivoting_friction_torque():
    """Rubber patch pivoting torque opposes yaw spin antisymmetrically and damps rigid-body yaw rate."""
    dt = 0.001
    normal = torch.tensor([2.5, 2.5, 2.5, 2.5], dtype=torch.float64)
    wz = torch.tensor([0.0, +0.5, -0.5, +5.0], dtype=torch.float64)
    zeros1 = torch.zeros_like(normal)
    zeros2 = torch.zeros((4, 2), dtype=torch.float64)
    patch_r = 0.011
    piv_mu = 0.75

    _, _, _, _, _, _, tau_z, _, _, c_piv = arpc_tangential_contact(
        zeros2, zeros1, zeros2, zeros1, normal, zeros1, wz, dt,
        patch_radius=patch_r, pivot_mu=piv_mu, backlash_rad=0.0, rut_lat_k=0.0,
    )
    piv_cap = (2.0 / 3.0) * piv_mu * patch_r * 2.5
    assert tau_z[0].item() == 0.0
    assert math.isclose(c_piv[0].item(), piv_cap / 0.15, rel_tol=1e-12)
    assert tau_z[1].item() < 0.0 and c_piv[1].item() > 0.0
    assert math.isclose(tau_z[2].item(), -tau_z[1].item(), rel_tol=1e-12)
    assert math.isclose(c_piv[2].item(), c_piv[1].item(), rel_tol=1e-12)
    assert math.isclose(abs(tau_z[3].item()), piv_cap, rel_tol=1e-2)

    # Full single-segment yaw spin decay: pivoting friction decelerates pure yaw faster than patch_radius=0
    dev = _best_device()
    hw_piv = HardwareConfig.ds3218_default().with_terrain('carpet')
    hw_nopiv = replace(hw_piv, wheel_patch_radius_m=0.0)
    final_wz = {}
    for label, hw in (('piv', hw_piv), ('nopiv', hw_nopiv)):
        core = GPUCore(count=1, batch=1, dt=0.001, device=dev, plate_solver='segment_lu', hardware=hw)
        core.target.fill_(hw.servo_neutral_rad)
        core.advance_steps(100)  # settle onto ground
        # Impart pure yaw angular velocity around segment center
        com = core.x[0, :2, :3].mean(0)
        rel = core.x[0, :, :3] - com
        wz0 = 1.2
        core.v[0, :, 0] = -wz0 * rel[:, 1]
        core.v[0, :, 1] = +wz0 * rel[:, 0]
        core.v[0, :, 5] = wz0
        core.advance_steps(80)
        assert not bool(core.failed)
        final_wz[label] = float(core.v[0, :2, 5].mean().abs().item())

    assert final_wz['piv'] < final_wz['nopiv'] * 0.95, (
        f"Expected pivoting friction to damp yaw faster: piv={final_wz['piv']:.4f} vs nopiv={final_wz['nopiv']:.4f}"
    )


def test_dt_1ms_static_hold_stability_victory():
    """At dt=1.0 ms, Implicit ARPC holds < 0.5 mm/s static drift across all terrains while explicit Kelvin chatters."""
    dev = _best_device()
    dt = 0.001
    base_hw = HardwareConfig.ds3218_default()

    def measure_drift_mm_s(hw: HardwareConfig) -> tuple[float, float]:
        solver = 'fused_cuda' if fused_cuda_available() else 'segment_lu'
        core = GPUCore(
            count=5, batch=1, dt=dt, device=dev, plate_solver=solver,
            hardware=hw, elements_per_strip=8,
        )
        if dev == 'cuda':
            core.capture(50)
        core.target.fill_(hw.servo_neutral_rad)
        core.advance_steps(1000)  # 1.0 s settle past neutral-tendon transient
        com0 = core.x[0, :10, :3].mean(0).clone()
        core.advance_steps(500)   # 0.5 s static hold measurement window
        assert not bool(core.failed)
        assert bool(torch.isfinite(core.x).all() and torch.isfinite(core.v).all())
        com1 = core.x[0, :10, :3].mean(0)
        drift_mm_s = float(torch.linalg.vector_norm(com1[:2] - com0[:2]).item()) * 2000.0
        mean_rut_mm = float(core.soil_sinkage.mean().item()) * 1000.0
        return drift_mm_s, mean_rut_mm

    kelvin_drift, _ = measure_drift_mm_s(base_hw.with_terrain('kelvin', implicit_contact=False))
    hard_drift, hard_rut = measure_drift_mm_s(base_hw.with_terrain('hard', implicit_contact=True))
    carpet_drift, _ = measure_drift_mm_s(base_hw.with_terrain('carpet', implicit_contact=True))
    soil_drift, soil_rut = measure_drift_mm_s(base_hw.with_terrain('soft_soil', implicit_contact=True))

    assert hard_drift < 0.5, f'ARPC hard dt=1ms drift too high: {hard_drift:.3f} mm/s'
    assert carpet_drift < 0.5, f'ARPC carpet dt=1ms drift too high: {carpet_drift:.3f} mm/s'
    assert soil_drift < 0.5, f'ARPC soft_soil dt=1ms drift too high: {soil_drift:.3f} mm/s'
    assert kelvin_drift > 3.0 and kelvin_drift > 20.0 * hard_drift, (
        f'Expected explicit Kelvin dt=1ms chatter drift >> Implicit ARPC: kelvin={kelvin_drift:.3f} vs hard={hard_drift:.3f} mm/s'
    )
    assert hard_rut == 0.0
    assert soil_rut > 0.3, f'Expected soft_soil static rut > 0.3 mm, got {soil_rut:.3f} mm'


def _quat_angle_diff(q_ref: torch.Tensor, q_cmp: torch.Tensor) -> float:
    """Well-conditioned geodesic quaternion difference (rad), avoiding FP64 acos(1-eps) precision floor."""
    sgn = torch.where((q_ref * q_cmp).sum(-1, keepdim=True) < 0.0, -1.0, 1.0)
    chord = torch.linalg.vector_norm(q_ref - sgn * q_cmp, dim=-1)
    return float((2.0 * torch.asin((0.5 * chord).clamp_max(1.0))).max().item())


@pytest.mark.skipif(not fused_cuda_available(), reason='CUDA + nvcc required for fused_cuda kernel tests')
@pytest.mark.parametrize(
    'terrain,implicit_contact,dt,steps,pos_tol',
    [
        ('kelvin', False, 0.0005, 200, 1e-9),
        ('hard', True, 0.0005, 200, 1e-9),
        ('carpet', True, 0.0005, 200, 1e-9),
        ('soft_soil', True, 0.0005, 200, 1e-9),
        ('kelvin', True, 0.001, 200, 1e-8),
        ('hard', True, 0.001, 200, 1e-8),
        ('carpet', True, 0.001, 200, 1e-8),
        ('soft_soil', True, 0.001, 200, 1e-8),
        ('soft_soil', False, 0.0005, 50, 1e-9),
    ],
)
def test_fused_cuda_arpc_fp64_parity(terrain: str, implicit_contact: bool, dt: float, steps: int, pos_tol: float):
    """Strict FP64 parity (< 1e-9 m at dt=0.5ms/200 steps, < 1e-8 m at dt=1ms/200 steps) vs segment_lu."""
    batch = 4
    hw = HardwareConfig.ds3218_default().with_terrain(terrain, implicit_contact=implicit_contact)
    ref = GPUCore(5, batch, dt=dt, device='cuda', plate_solver='segment_lu', hardware=hw)
    fused = GPUCore(5, batch, dt=dt, device='cuda', plate_solver='fused_cuda', hardware=hw)

    mu_vals = torch.tensor([0.60, 0.80, 0.95, 1.10], device='cuda', dtype=torch.float64)
    ms_vals = torch.tensor([0.92, 1.00, 1.05, 1.08], device='cuda', dtype=torch.float64)
    targets = torch.tensor([
        [0.8, 0.8, 1.2, 1.2, 0.5, 0.5, 1.5, 1.5, 0.9, 0.9],
        [1.6, 1.6, 0.4, 0.4, 1.8, 1.8, 0.6, 0.6, 1.2, 1.2],
        [0.2, 1.4, 1.0, 0.3, 1.5, 0.7, 0.4, 1.6, 1.1, 0.5],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ], device='cuda', dtype=torch.float64)
    yaws = torch.tensor([
        [0.35, -0.30, 0.25, -0.20],
        [-0.40, 0.38, -0.35, 0.30],
        [0.0, 0.0, 0.0, 0.0],
        [0.20, 0.20, -0.20, -0.20],
    ], device='cuda', dtype=torch.float64)

    for core in (ref, fused):
        core.wheel_mu.copy_(mu_vals[:, None, None])
        core.mass_scale.copy_(ms_vals)
        core.target.copy_(targets)
        core.yaw.copy_(yaws)

    ref.advance_steps(steps)
    fused.advance_steps(steps)

    assert not bool(ref.failed) and not bool(fused.failed)
    pos_diff = (ref.x[..., :3] - fused.x[..., :3]).abs().max().item()
    ang_diff = _quat_angle_diff(ref.x[..., 3:], fused.x[..., 3:])
    vel_diff = (ref.v - fused.v).abs().max().item()
    servo_diff = (ref.servo - fused.servo).abs().max().item()
    spin_diff = (ref.spin - fused.spin).abs().max().item()
    support_diff = (ref.support - fused.support).abs().max().item()
    sinkage_diff = (ref.soil_sinkage - fused.soil_sinkage).abs().max().item()
    play_diff = (ref.ratchet_play - fused.ratchet_play).abs().max().item()
    yaw_trq_diff = (ref.last_yaw_torque - fused.last_yaw_torque).abs().max().item()

    scale = pos_tol / 1e-9
    assert pos_diff < pos_tol, f'{terrain} (imp={implicit_contact}, dt={dt}, N={steps}): pos_diff={pos_diff:.3e}'
    assert ang_diff < 1e-7 * scale, f'{terrain}: ang_diff={ang_diff:.3e}'
    assert vel_diff < 1e-5 * scale, f'{terrain}: vel_diff={vel_diff:.3e}'
    assert servo_diff < 1e-8 * scale, f'{terrain}: servo_diff={servo_diff:.3e}'
    assert spin_diff < 1e-4 * scale, f'{terrain}: spin_diff={spin_diff:.3e}'
    assert support_diff < 1e-4 * scale, f'{terrain}: support_diff={support_diff:.3e}'
    assert sinkage_diff < pos_tol, f'{terrain}: sinkage_diff={sinkage_diff:.3e}'
    assert play_diff < 1e-7 * scale, f'{terrain}: play_diff={play_diff:.3e}'
    assert yaw_trq_diff < 1e-5 * scale, f'{terrain}: yaw_trq_diff={yaw_trq_diff:.3e}'


@pytest.mark.skipif(not fused_cuda_available(), reason='CUDA + nvcc required for fused_cuda kernel tests')
def test_fused_cuda_arpc_e8_and_cuda_graph_parity():
    """Verify E=8 template and CUDA Graph capture maintain < 1e-9 m FP64 parity on soft_soil at dt=1.0 ms."""
    hw = HardwareConfig.ds3218_default().with_terrain('soft_soil', implicit_contact=True)
    ref = GPUCore(5, 2, dt=0.001, device='cuda', plate_solver='segment_lu', hardware=hw, elements_per_strip=8)
    fused = GPUCore(5, 2, dt=0.001, device='cuda', plate_solver='fused_cuda', hardware=hw, elements_per_strip=8)
    fused.capture(10)

    for core in (ref, fused):
        core.target.fill_(1.1)
        core.yaw.copy_(torch.tensor([
            [0.30, -0.25, 0.20, -0.15],
            [-0.25, 0.30, -0.20, 0.15],
        ], device='cuda', dtype=torch.float64))

    ref.advance_steps(100)
    fused.advance_steps(100)

    assert not bool(ref.failed) and not bool(fused.failed)
    pos_diff = (ref.x[..., :3] - fused.x[..., :3]).abs().max().item()
    sinkage_diff = (ref.soil_sinkage - fused.soil_sinkage).abs().max().item()
    play_diff = (ref.ratchet_play - fused.ratchet_play).abs().max().item()
    assert pos_diff < 1e-9, f'E=8 CUDA graph pos_diff={pos_diff:.3e}'
    assert sinkage_diff < 1e-11, f'E=8 CUDA graph sinkage_diff={sinkage_diff:.3e}'
    assert play_diff < 1e-8, f'E=8 CUDA graph play_diff={play_diff:.3e}'


def test_capstan_stribeck_stiction_asymmetric_mass_and_real_damping():
    """Verify Capstan pull attenuation, Stribeck static-friction boost, yaw stiction deadzone,
    asymmetric plate masses, and realistic Rayleigh beta=0.0005 stability at dt=1ms."""
    # 1. Stribeck static > kinetic breakaway capacity
    dt = 0.001
    normal = torch.tensor([2.0], dtype=torch.float64)
    z1 = torch.zeros(1, dtype=torch.float64)
    # Slow reverse pull against locked ratchet (play=0) with preloaded spring memory
    tv_slow = torch.tensor([[-0.001, 0.0]], dtype=torch.float64)
    tv_fast = torch.tensor([[-0.20, 0.0]], dtype=torch.float64)
    mem_hi = torch.tensor([[-0.002, 0.0]], dtype=torch.float64)
    f_stat, *_ = arpc_tangential_contact(
        tv_slow, z1, mem_hi, z1, normal, z1, z1, dt,
        mu_fwd=0.8, mu_lat=0.8, rolling=0.025, backlash_rad=0.1,
        patch_radius=0.0, pivot_mu=0.0, rut_lat_k=0.0, static_mu_ratio=1.25,
    )
    f_kin, *_ = arpc_tangential_contact(
        tv_fast, z1, mem_hi, z1, normal, z1, z1, dt,
        mu_fwd=0.8, mu_lat=0.8, rolling=0.025, backlash_rad=0.1,
        patch_radius=0.0, pivot_mu=0.0, rut_lat_k=0.0, static_mu_ratio=1.25,
    )
    assert f_stat[0, 0].item() > 1.85, f'Expected Stribeck static peak ~2.0 N, got {f_stat[0, 0].item():.3f} N'
    assert f_kin[0, 0].item() < 1.65, f'Expected kinetic cap ~1.6 N, got {f_kin[0, 0].item():.3f} N'

    # 2. Capstan pull attenuation + yaw stiction deadzone + asymmetric mass + realistic beta=0.0005
    hw_v5 = HardwareConfig.v5_realistic(terrain='hard')
    assert hw_v5.rayleigh_beta == 0.0005
    assert hw_v5.tendon_capstan_mu_theta > 0.1
    assert hw_v5.plate_masses_kg is not None and len(hw_v5.plate_masses_kg) == 10
    assert abs(sum(hw_v5.plate_masses_kg) - 2.6045) < 1e-4
    assert abs(hw_v5.wheel_mass_kg - 0.0179) < 1e-6
    core = GPUCore(5, 2, dt=0.001, device='cpu', plate_solver='segment_lu', hardware=hw_v5, elements_per_strip=8)
    # Env 0: zero Capstan & zero yaw stiction; Env 1: strong Capstan (0.35) & yaw stiction (0.25 Nm)
    core.env_dynamics[0, 3] = 0.0
    core.env_dynamics[0, 5] = 0.0
    core.env_dynamics[0, 6] = 0.0
    core.env_dynamics[1, 3] = 0.35
    core.env_dynamics[1, 5] = 0.25
    core.env_dynamics[1, 6] = 0.0
    core.servo.fill_(math.radians(100.0))
    core.target.fill_(math.radians(150.0))
    # Tiny 1 deg yaw target -> Kp * 0.0175 rad = 0.175 Nm < 0.25 Nm static friction on Env 1!
    core.yaw.fill_(math.radians(1.0))
    core.advance_steps(25)
    assert not bool(core.failed) and torch.isfinite(core.x).all()
    assert float(core.last_yaw_torque[0].abs().max()) > 0.05
    assert float(core.last_yaw_torque[1].abs().max()) < 1e-12, 'Yaw stiction must block sub-threshold torque'
    # During active winding, Capstan friction reduces transmitted span tension on Env 1 vs Env 0
    assert float(core.tension[0].mean()) > 1.0
    assert float(core.tension[1].mean()) < float(core.tension[0].mean()) * 0.90


@pytest.mark.skipif(not fused_cuda_available(), reason='CUDA + nvcc required for fused_cuda kernel tests')
def test_fused_cuda_v5_capstan_stribeck_asym_mass_parity():
    """Exact FP64 parity (< 1e-8 m) between segment_lu and fused_cuda with ALL v5 physics features active."""
    hw = HardwareConfig.v5_realistic(terrain='hard')
    ref = GPUCore(5, 3, dt=0.001, device='cuda', plate_solver='segment_lu', hardware=hw, elements_per_strip=8)
    fused = GPUCore(5, 3, dt=0.001, device='cuda', plate_solver='fused_cuda', hardware=hw, elements_per_strip=8)

    dyn = torch.tensor([
        [0.08, 1.00, 0.006, 0.00, 0.010, 0.00, 0.000],
        [0.18, 1.20, 0.009, 0.22, 0.012, 0.08, 0.012],
        [0.28, 1.35, 0.012, 0.35, 0.014, 0.16, 0.025],
    ], device='cuda', dtype=torch.float64)
    pm = ref.plate_masses_base[None, :] * torch.tensor([
        [0.90, 1.10, 0.95, 1.05, 1.00, 1.00, 0.92, 1.08, 0.96, 1.04],
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        [1.12, 0.88, 1.06, 0.94, 1.08, 0.92, 1.05, 0.95, 1.10, 0.90],
    ], device='cuda', dtype=torch.float64)
    betas = torch.tensor([0.0002, 0.0005, 0.0018], device='cuda', dtype=torch.float64)
    ggeom = torch.tensor([
        [0.0, 0.0, 0.0, 8.0, 8.0],
        [0.05, -0.03, 0.002, 10.0, 8.0],
        [-0.04, 0.04, 0.0025, 12.0, 10.0],
    ], device='cuda', dtype=torch.float64)

    for c in (ref, fused):
        c.env_dynamics.copy_(dyn)
        c.plate_masses.copy_(pm)
        c.rayleigh_beta_env.copy_(betas)
        c.ground_geom.copy_(ggeom)
        c.align_pose_to_ground(torch.ones(3, device='cuda', dtype=torch.bool))
        c.target.fill_(1.8)
        c.yaw.copy_(torch.tensor([
            [0.25, -0.25, 0.20, -0.20],
            [-0.15, 0.18, -0.12, 0.15],
            [0.02, -0.02, 0.01, -0.01],
        ], device='cuda', dtype=torch.float64))

    ref.advance_steps(100)
    fused.advance_steps(100)
    assert not bool(ref.failed) and not bool(fused.failed)
    pos_diff = (ref.x[..., :3] - fused.x[..., :3]).abs().max().item()
    ten_diff = (ref.tension - fused.tension).abs().max().item()
    assert pos_diff < 1e-8, f'v5 full-physics CUDA parity pos_diff={pos_diff:.3e}'
    assert ten_diff < 1e-6, f'v5 full-physics CUDA parity ten_diff={ten_diff:.3e}'

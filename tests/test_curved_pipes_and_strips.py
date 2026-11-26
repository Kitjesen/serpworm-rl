"""Unit tests for spring-steel realism upgrades and 4-type curved/right-angle pipe physics."""
from __future__ import annotations

from dataclasses import replace
import torch

from serpworm.config import HardwareConfig
from serpworm.physics.gpu_core import GPUCore
from serpworm.physics.pipe_geometry import (
    make_bend90_pipe,
    make_right_angle_duct,
    make_scurve_pipe,
    make_straight_pipe,
)


def test_strip_thin_shell_and_prestress_and_width():
    hw_base = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
    hw_shell = replace(hw_base, strip_thin_shell_compat=True)
    hw_wide = replace(hw_base, strip_width_m=0.020)
    hw_pre = replace(hw_base, strip_prestress_frac=0.5)

    c0 = GPUCore(1, 1, dt=0.001, device='cpu', plate_solver='segment_lu', hardware=hw_base, elements_per_strip=6)
    cs = GPUCore(1, 1, dt=0.001, device='cpu', plate_solver='segment_lu', hardware=hw_shell, elements_per_strip=6)
    cw = GPUCore(1, 1, dt=0.001, device='cpu', plate_solver='segment_lu', hardware=hw_wide, elements_per_strip=6)
    cp = GPUCore(1, 1, dt=0.001, device='cpu', plate_solver='segment_lu', hardware=hw_pre, elements_per_strip=6)

    # Thin-shell Kirchhoff-Love modulus stiffens out-of-plane Iy bending by 1/(1-nu^2) ~ 1.0989x
    ratio_iy = float((cs.K[0, 0, 2, 2] / c0.K[0, 0, 2, 2]).item())
    assert abs(ratio_iy - 1.0 / (1.0 - 0.09)) < 1e-6
    # Wider 20 mm strip increases axial EA by 20/16 = 1.25x and lateral Iz by (20/16)^3 = 1.953125x
    assert abs(float((cw.K[0, 0, 0, 0] / c0.K[0, 0, 0, 0]).item()) - 1.25) < 1e-6
    assert abs(float((cw.K[0, 0, 1, 1] / c0.K[0, 0, 1, 1]).item()) - (1.25 ** 3)) < 1e-6
    # Assembly prestress shifts rest_local toward flat straight strip (nonzero initial elastic force!)
    assert not torch.allclose(cp.rest_local, c0.rest_local)
    for _ in range(5):
        cp.physics()
    assert torch.isfinite(cp.x).all() and torch.isfinite(cp.v).all()


def test_strip_ground_and_all_four_pipe_types_cpu():
    hw = replace(
        HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard'),
        strip_node_contact=True,
    )
    c = GPUCore(5, 4, dt=0.001, device='cpu', plate_solver='segment_lu', hardware=hw, elements_per_strip=6)
    c.pipe_cfg[0] = torch.tensor(make_straight_pipe(x_entry=1.0, x_exit=-3.0, r_in=0.112), dtype=torch.float64)
    c.pipe_cfg[1] = torch.tensor(make_bend90_pipe(x_entry=1.0, x_bend=-0.50, R_bend=0.50, r_in=0.118, bend_sgn=+1.0), dtype=torch.float64)
    c.pipe_cfg[2] = torch.tensor(make_scurve_pipe(x_entry=1.0, x_bend=-0.45, R_bend=0.52, r_in=0.118, bend_sgn=-1.0), dtype=torch.float64)
    c.pipe_cfg[3] = torch.tensor(make_right_angle_duct(x_entry=1.0, x_corner=-0.75, width=0.250, bend_sgn=+1.0), dtype=torch.float64)
    c.rayleigh_beta_env[:] = torch.tensor([0.0003, 0.0008, 0.0015, 0.0022], dtype=torch.float64)

    # Nudge robots laterally so plates & steel strips engage every pipe type + push Env 0 downward to test strip-ground contact
    c.x[0, :, 1] += 0.045
    c.x[0, :, 2] -= 0.055  # lowers bottom strip nodes below z=0 to verify ground protection!
    c.x[1, :, 1] -= 0.045
    c.x[2, :, 1] += 0.045
    c.x[3, :, 0] -= 0.060  # nose presses into sharp 90-deg dead-end wall!

    for _ in range(5):
        c.physics()

    assert torch.isfinite(c.x).all() and torch.isfinite(c.v).all()
    for b in range(4):
        tot_w = float(c.last_wall_force[b].sum().item() + c.last_strip_wall_force[b].sum().item())
        assert tot_w > 1.0, f'Expected active pipe wall force in env {b}, got {tot_w}'
    # Strip ground contact prevents bottom strips from sinking deeply below z=0
    assert float(c.last_strip_min_z[0].min().item()) > -0.025


def test_autonomous_v7_curved_pipes_env_and_symmetry():
    from serpworm.config import DomainRandConfig
    from serpworm.envs.autonomous_env import AutonomousPhaseEnv, AutonomousTaskConfig, mirror_autonomous_batch
    from serpworm.physics.pipe_geometry import eval_pipe_membership_torch

    hw = replace(
        HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard'),
        strip_thin_shell_compat=True,
        strip_node_contact=True,
    )
    task = AutonomousTaskConfig(
        elements_per_strip=6,
        physics_dt_s=0.001,
        include_q_err=True,
        curved_pipes=True,
        history_len=8,
    )
    env = AutonomousPhaseEnv(
        n=6, hardware=hw, task=task, dr=DomainRandConfig(enabled=True),
        device='cpu', plate_solver='segment_lu', seed=17, noise=False, settle_ticks=6,
    )
    # Assign all 6 regimes explicitly and verify step, EstNet targets, strip telemetry, and mirror symmetry
    for rid in range(6):
        env.forced_regime = rid
        mask = torch.zeros(6, dtype=torch.bool)
        mask[rid] = True
        env.reset(mask)
    env.forced_regime = None
    assert sorted(set(env.c.pipe_cfg[:, 0].round().to(torch.long).tolist())) == [0, 1, 2, 3, 4]

    act = torch.zeros((6, 14), dtype=torch.float32)
    nobs, npriv, nest, rew, term, trunc, tobs, tpriv, info = env.step(act)
    assert not bool(term.any()), f"Unexpected guard failure: {env.last_guard_reasons}"
    assert nobs.shape == (6, 8, 65)
    assert npriv.shape == (6, 108)
    assert nest.shape == (6, 7)
    assert torch.isfinite(nobs).all() and torch.isfinite(npriv).all() and torch.isfinite(rew).all()
    assert 'min_strip_z_m' in info and 'strip_wall_mean_n' in info and 'pipe_type' in info
    assert float(info['min_strip_z_m'].min().item()) > 0.01

    # Verify exact antisymmetry of pipe lateral offset & invariance of arc progress under (y -> -y, bend_sgn -> -bend_sgn)
    from serpworm.physics.pipe_geometry import eval_pipe_arc_progress_torch
    pos_xy = torch.tensor([[[-1.20, 0.04], [-1.55, 0.20], [-1.75, 0.55]]], dtype=torch.float64)
    pos_xy_m = pos_xy.clone(); pos_xy_m[..., 1] *= -1.0
    for pc in (
        make_bend90_pipe(x_entry=-1.0, x_bend=-1.40, R_bend=0.50, bend_sgn=+1.0),
        make_scurve_pipe(x_entry=-1.0, x_bend=-1.40, R_bend=0.52, bend_sgn=+1.0),
        make_right_angle_duct(x_entry=-1.0, x_corner=-1.50, width=0.256, bend_sgn=+1.0),
    ):
        pc_t = torch.tensor([pc], dtype=torch.float64)
        pc_m = pc_t.clone(); pc_m[:, 1] *= -1.0; pc_m[:, 7] *= -1.0
        in1, lat1 = eval_pipe_membership_torch(pos_xy, pc_t)
        in2, lat2 = eval_pipe_membership_torch(pos_xy_m, pc_m)
        s1 = eval_pipe_arc_progress_torch(pos_xy, pc_t)
        s2 = eval_pipe_arc_progress_torch(pos_xy_m, pc_m)
        assert torch.equal(in1, in2)
        assert torch.allclose(lat1, -lat2, atol=1e-12)
        assert torch.allclose(s1, s2, atol=1e-12)
        assert bool((s1[0, 1] > s1[0, 0]) and (s1[0, 2] > s1[0, 1]))

    m1 = mirror_autonomous_batch(nobs, npriv, nest, act, include_q_err=True)
    m2 = mirror_autonomous_batch(*m1, include_q_err=True)
    for orig, rt in zip((nobs, npriv, nest, act), m2):
        assert torch.allclose(orig, rt, atol=1e-6)


def test_v8_diverse_8pipes_and_gravel_grass_terrains():
    from serpworm.config import DomainRandConfig
    from serpworm.envs.autonomous_env import AutonomousPhaseEnv, AutonomousTaskConfig
    from serpworm.physics.pipe_geometry import (
        eval_pipe_arc_progress_torch,
        eval_pipe_contact_torch,
        eval_pipe_membership_torch,
        make_taper_pipe,
        make_triple_wave_pipe,
        make_uturn_pipe,
        make_z_duct,
    )

    # 1. Centerline zero-force + exact mirror symmetry across all 4 new v8 pipe types (5..8)
    new_pipes = [
        (make_uturn_pipe(x_entry=-1.0, x_bend=-1.35, R_bend=0.54, bend_sgn=+1.0, L_exit=0.90),
         [(-1.20, 0.0), (-1.89, 0.54), (-1.10, 1.08)]),
        (make_taper_pipe(x_entry=-1.0, x_taper_start=-1.35, L_taper=0.60, r_in=0.150, r_out=0.112, L_exit=0.80),
         [(-1.15, 0.0), (-1.65, 0.0), (-2.30, 0.0)]),
        (make_z_duct(x_entry=-1.0, x_corner=-1.40, L_mid=0.68, width=0.264, bend_sgn=+1.0, L_exit=0.80),
         [(-1.20, 0.0), (-1.40, 0.34), (-1.85, 0.68)]),
        (make_triple_wave_pipe(x_entry=-1.0, x_bend=-1.30, R_bend=0.54, bend_angle_deg=55.0, bend_sgn=+1.0, L_exit=0.70),
         [(-1.15, 0.0), (-2.18, 0.46), (-3.15, 0.0)]),
    ]
    for pc, pts in new_pipes:
        pc_t = torch.tensor([pc], dtype=torch.float64)
        pc_m = pc_t.clone(); pc_m[:, 1] *= -1.0; pc_m[:, 7] *= -1.0
        pt3 = torch.tensor([[[x, y, 0.118] for x, y in pts]], dtype=torch.float64)
        _, _, fn = eval_pipe_contact_torch(pt3, torch.zeros_like(pt3), pc_t, node_radius=0.074, dt=0.001, implicit_contact=True)
        assert float(fn.max().item()) < 1e-12, f'Expected 0 centerline force for pipe type {pc[0]}, got {fn}'

        pt_off = torch.tensor([[[x, y + 0.03] for x, y in pts]], dtype=torch.float64)
        pt_off_m = pt_off.clone(); pt_off_m[..., 1] *= -1.0
        in1, lat1 = eval_pipe_membership_torch(pt_off, pc_t)
        in2, lat2 = eval_pipe_membership_torch(pt_off_m, pc_m)
        s1 = eval_pipe_arc_progress_torch(pt_off, pc_t)
        s2 = eval_pipe_arc_progress_torch(pt_off_m, pc_m)
        assert torch.equal(in1, in2)
        assert torch.allclose(lat1, -lat2, atol=1e-12)
        assert torch.allclose(s1, s2, atol=1e-12)
        assert bool((s1[0, 1] > s1[0, 0]) and (s1[0, 2] > s1[0, 1]))

    # 2. Full 10-regime diverse_v8 env including gravel (3) and grass (4) terrains
    hw = replace(
        HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard'),
        strip_thin_shell_compat=True,
        strip_node_contact=True,
    )
    task = AutonomousTaskConfig(
        elements_per_strip=6, physics_dt_s=0.001, include_q_err=True, diverse_v8=True, history_len=8,
    )
    env = AutonomousPhaseEnv(
        n=11, hardware=hw, task=task, dr=DomainRandConfig(enabled=True),
        device='cpu', plate_solver='segment_lu', seed=23, noise=False, settle_ticks=6,
    )
    for rid in range(10):
        env.forced_regime = rid
        m = torch.zeros(11, dtype=torch.bool); m[rid] = True
        env.reset(m)
    env.forced_regime = 9
    m10 = torch.zeros(11, dtype=torch.bool); m10[10] = True
    env.reset(m10)
    env.forced_regime = None
    # Force slots 9 and 10 to gravel (3) and grass (4) explicitly to verify bumpy micro-terrain stability
    env._apply_terrain_ids(torch.tensor([i == 9 for i in range(11)]), torch.full((11,), 3, dtype=torch.long))
    env._apply_terrain_ids(torch.tensor([i == 10 for i in range(11)]), torch.full((11,), 4, dtype=torch.long))
    assert sorted(set(env.c.pipe_cfg[:, 0].round().to(torch.long).tolist())) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    act = torch.zeros((11, 14), dtype=torch.float32)
    nobs, npriv, nest, rew, term, trunc, tobs, tpriv, info = env.step(act)
    assert not bool(term.any()), f"Unexpected guard failure in v8 env: {env.last_guard_reasons}"
    assert nobs.shape == (11, 8, 65) and npriv.shape == (11, 108) and nest.shape == (11, 7)
    assert torch.isfinite(nobs).all() and torch.isfinite(npriv).all() and torch.isfinite(rew).all()

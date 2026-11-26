"""Comprehensive verification suite for the generic FlexGraph rigid-flexible robotics framework.

Tests run purely on CPU in FP64/FP32 (~5-10 s total, zero GPU usage) and verify:
  1. RigidFlexGraph IR serialization & automatic sparsity routing across 4 robot archetypes
  2. Machine-precision (< 1e-10) FP64 parity against serpworm.physics.gpu_core.GPUCore
  3. Stable physical simulation of 4 heterogeneous robots in the exact same GraphSchurSolver
     (SerpWorm-N, wheelless BristleWorm, 3D TendonEndoscope with fixed base & cantilever tip,
      and compliant-spine FlexSpine-Crawler)
  4. ComposableSDFWorld (HeightField, 3D SplineTube, BoxObstacle, VoxelGridSDF),
     AnisotropicPatchContact (ratchet_wheel / anisotropic_scale / isotropic_pad + Bekker rut),
     and Py-DiSMech C2 self-contact exact gradient & PSD Hessian checks
  5. Morphology-agnostic TokenizedHIMDreamWaQ (variable N=4..20, failure masking, aux losses)
     and exact FiniteGroupEquivariantWrapper across Z2, Z2xZ2 (order 4), and D4 (order 8)
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import torch

from flexgraph import (
    BoxObstacleSDF,
    ComposableSDFWorld,
    FiniteGroupEquivariantWrapper,
    FlexGraphEnv,
    GraphSchurSolver,
    HeightFieldSDF,
    RigidFlexGraph,
    SafetyConfig,
    SplineTubeSDF,
    TokenizedHIMDreamWaQ,
    TokenizedPolicyConfig,
    UniversalSafetyGuard,
    VoxelGridSDF,
    build_bristle_worm_graph,
    build_flexspine_crawler_graph,
    build_serpworm_graph,
    build_tendon_endoscope_graph,
    default_material_library,
    eval_c2_self_contact,
    make_d4_radial_symmetry,
    make_z2_bilateral_symmetry,
    make_z2xz2_bidirectional_symmetry,
)
from flexgraph.policy import apply_signed_perm


# ---------------------------------------------------------------------------
# 1. Graph IR Serialization & Sparsity Auto-Routing
# ---------------------------------------------------------------------------
def test_graph_ir_serialization_and_routing(tmp_path: Path):
    g5 = build_serpworm_graph(n_segments=5, elements_per_strip=16)
    sp5 = g5.analyze_sparsity()
    assert sp5.n_bodies == 10
    assert sp5.n_beams == 40
    assert sp5.n_beam_interior_nodes == 600
    assert sp5.total_dofs == 3660
    assert sp5.recommended_route == 'block_diagonal_pairs'

    # JSON round-trip preserves exact topology & metadata
    jp = g5.save_json(tmp_path / 'serpworm5.json')
    g5_rt = RigidFlexGraph.load_json(jp)
    assert g5_rt.analyze_sparsity() == sp5
    assert len(g5_rt.tendons) == len(g5.tendons) == 20
    assert len(g5_rt.contact_sites) == 20

    # Variable-length SerpWorm N=3 and N=7
    assert build_serpworm_graph(3, 16).analyze_sparsity().total_dofs == 6 * (6 + 24 * 15)
    assert build_serpworm_graph(7, 16).analyze_sparsity().total_dofs == 6 * (14 + 56 * 15)

    # Fixed-base 3D tendon endoscope & cantilever tip
    endo = build_tendon_endoscope_graph(n_sections=4, elements_per_section=6)
    sp_e = endo.analyze_sparsity()
    assert sp_e.n_fixed_bodies == 1
    assert sp_e.recommended_route == 'tree_chain'
    assert endo.symmetry is not None and endo.symmetry.order == 8

    endo_cant = build_tendon_endoscope_graph(n_sections=3, elements_per_section=6, free_cantilever_tip=True)
    assert endo_cant.analyze_sparsity().n_cantilever_beams == 1


# ---------------------------------------------------------------------------
# 2. Exact FP64 Machine-Precision Parity vs serpworm.physics.gpu_core.GPUCore
# ---------------------------------------------------------------------------
def test_fp64_exact_parity_against_serpworm_gpucore():
    from serpworm.config import HardwareConfig
    from serpworm.physics.gpu_core import GPUCore

    hw = HardwareConfig.bench_exp128_calibrated()
    # Compare pure 5-segment beam + crank-tendon + yaw-joint + stop-gap dynamics (ground_contact=False)
    ref = GPUCore(
        count=3, batch=1, dt=0.0005, device='cpu', dtype=torch.float64,
        plate_solver='segment_lu', ground_contact=False, hardware=hw, elements_per_strip=8,
    )
    g = build_serpworm_graph(n_segments=3, elements_per_strip=8, hardware=hw)
    # Remove contact sites so both run identical ground_contact=False physics
    g.contact_sites = []
    sim = GraphSchurSolver(g, world=ComposableSDFWorld(primitives=[]), batch=1, dt=0.0005, device='cpu')

    # Drive both with asymmetric tendons + yaw targets for 15 substeps
    ref.target[:] = torch.tensor([[0.9, 0.3, 0.5, 1.1, 0.7, 0.2]], dtype=torch.float64)
    ref.yaw[:] = torch.tensor([[0.25, -0.30]], dtype=torch.float64)
    tgt_fg = torch.cat((ref.target, ref.yaw), dim=-1)

    for _ in range(15):
        ref.physics()
        sim.step(tgt_fg, steps=1)

    pos_err = float((ref.x[..., :3] - sim.x[..., :3]).abs().max().item())
    plate_vel_err = float((ref.v[:, :6] - sim.v[:, :6]).abs().max().item())
    vel_err = float((ref.v - sim.v).abs().max().item())
    ten_err = float((ref.tension.reshape(1, -1) - sim.tendon_tensions).abs().max().item())
    assert not bool(sim.failed.item())
    assert pos_err < 1e-11, f'FP64 position parity mismatch: {pos_err}'
    assert plate_vel_err < 1e-8, f'FP64 plate velocity parity mismatch: {plate_vel_err}'
    assert vel_err < 1e-7, f'FP64 beam-node velocity parity mismatch: {vel_err}'
    assert ten_err < 1e-12, f'FP64 tendon tension parity mismatch: {ten_err}'


def test_fp64_parity_with_ground_contact_and_ratchet_wheels():
    from serpworm.config import HardwareConfig
    from serpworm.physics.gpu_core import GPUCore

    hw = HardwareConfig.ds3218_default()
    ref = GPUCore(
        count=2, batch=1, dt=0.0005, device='cpu', dtype=torch.float64,
        plate_solver='segment_lu', ground_contact=True, hardware=hw, elements_per_strip=6,
    )
    g = build_serpworm_graph(n_segments=2, elements_per_strip=6, hardware=hw)
    sim = GraphSchurSolver(
        g,
        world=ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)]),
        batch=1,
        dt=0.0005,
        device='cpu',
        implicit_contact=False,
        use_kelvin_normal=True,
    )

    ref.target[:] = torch.tensor([[0.6, 0.6, 0.4, 0.8]], dtype=torch.float64)
    ref.yaw[:] = torch.tensor([[0.15]], dtype=torch.float64)
    tgt_fg = torch.cat((ref.target, ref.yaw), dim=-1)

    for _ in range(12):
        ref.physics()
        sim.step(tgt_fg, steps=1)

    pos_err = float((ref.x[..., :3] - sim.x[..., :3]).abs().max().item())
    assert not bool(sim.failed.item())
    assert pos_err < 1e-10, f'FP64 ground-contact parity mismatch: {pos_err}'


# ---------------------------------------------------------------------------
# 3. Four Heterogeneous Robot Archetypes in the Exact Same Solver
# ---------------------------------------------------------------------------
def test_archetype_bristle_worm_anisotropic_scale_locomotion():
    """Wheelless crawler using anisotropic_scale friction advances forward under symmetric wave actuation."""
    g = build_bristle_worm_graph(n_segments=2, elements_per_strip=6, mu_fwd=0.10, mu_rev=0.90, mu_lat=0.90)
    world = ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)])
    sim = GraphSchurSolver(g, world=world, batch=1, dt=0.001, device='cpu', implicit_contact=True)

    # Settle on ground then apply peristaltic contraction wave
    sim.step(steps=10)
    x0 = float(sim.x[0, :sim.P, 0].mean().item())
    for k in range(30):
        ph = 2.0 * math.pi * (k / 20.0)
        c0 = 0.7 * (0.5 - 0.5 * math.cos(ph))
        c1 = 0.7 * (0.5 - 0.5 * math.cos(ph - 1.2))
        sim.step(torch.tensor([c0, c0, c1, c1, 0.0], dtype=torch.float64), steps=1)

    assert not bool(sim.failed.item())
    assert torch.isfinite(sim.x).all()
    # Zero wheel spin in anisotropic_scale mode & positive normal ground support
    assert float(sim.contact_state.spin.abs().max().item()) == 0.0
    assert float(sim.last_site_normal.sum().item()) > 1.0
    # Body stayed above ground without collapse
    assert float(sim.x[0, :sim.P, 2].min().item()) > 0.04
    assert abs(float(sim.x[0, :sim.P, 0].mean().item()) - x0) > 1e-4


def test_archetype_tendon_endoscope_3d_in_curved_spline_tube():
    """Fixed-base 3D 4-tendon nitinol catheter bends along pulled tendon & obeys 3D SplineTube wall."""
    g = build_tendon_endoscope_graph(n_sections=3, elements_per_section=5, free_cantilever_tip=False)
    tube = SplineTubeSDF(
        control_points=[(0.0, 0.0, 0.12), (0.09, 0.0, 0.12), (0.18, 0.03, 0.125)],
        radii_m=[0.025, 0.022, 0.020],
        samples_per_span=8,
        funnel_len_m=0.0,
        funnel_flare_m=0.0,
        mat_id=1,
    )
    world = ComposableSDFWorld(primitives=[tube])
    sim = GraphSchurSolver(g, world=world, batch=1, dt=0.001, device='cpu', implicit_contact=True)

    # Pull +Y tendon (actuator 0) -> tip must deflect toward +Y while base stays strictly pinned at 0
    sim.step(torch.tensor([0.012, 0.0, 0.0, 0.0], dtype=torch.float64), steps=20)
    assert not bool(sim.failed.item())
    assert torch.isfinite(sim.x).all()
    base_drift = float((sim.x[0, 0, :3] - sim.rest[0, 0, :3]).abs().max().item())
    tip_y = float(sim.x[0, sim.P - 1, 1].item())
    assert base_drift == 0.0
    assert tip_y > 0.001, f'Expected +Y deflection under +Y tendon pull, got tip_y={tip_y}'

    # Also test free-cantilever tip beam (body_b == -1)
    g_cant = build_tendon_endoscope_graph(n_sections=2, elements_per_section=5, free_cantilever_tip=True)
    sim_cant = GraphSchurSolver(g_cant, world=ComposableSDFWorld(primitives=[]), batch=1, dt=0.001, device='cpu')
    sim_cant.step(torch.tensor([0.008, 0.0, 0.0, 0.0], dtype=torch.float64), steps=15)
    assert not bool(sim_cant.failed.item())
    assert torch.isfinite(sim_cant.x).all()
    # Free cantilever tip droops under gravity and bends cleanly
    assert float(sim_cant.x[0, -1, 2].item()) < 0.12


def test_archetype_flexspine_crawler_with_c2_self_contact():
    """Compliant leaf-spring spine + revolute hip legs + isotropic_pad feet + Py-DiSMech C2 self-contact."""
    g = build_flexspine_crawler_graph(n_segments=2, elements_per_spine=5)
    world = ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=2)])  # soft soil with Bekker rutting!
    sim = GraphSchurSolver(
        g, world=world, batch=1, dt=0.001, device='cpu', implicit_contact=True, enable_self_contact=True
    )
    sim.step(torch.tensor([0.3, -0.3, -0.3, 0.3], dtype=torch.float64), steps=20)
    assert not bool(sim.failed.item())
    assert torch.isfinite(sim.x).all()
    # Soft soil material (mat_id=2, unload_ratio=3.5) should record positive Bekker plastic rut
    assert float(sim.contact_state.rut.max().item()) > 0.0


# ---------------------------------------------------------------------------
# 4. ComposableSDFWorld & Py-DiSMech C2 Self-Contact Math Verification
# ---------------------------------------------------------------------------
def test_composable_sdf_primitives_and_voxel_grid():
    hf = HeightFieldSDF(slope_x=0.05, slope_y=-0.02, wave_amp_m=0.003, wave_kx=4.0, wave_ky=3.0)
    box = BoxObstacleSDF(center=(0.0, 0.0, 0.50), half_extents=(0.10, 0.10, 0.10), mat_id=1)
    world = ComposableSDFWorld(primitives=[hf, box])

    pts = torch.tensor([[[0.0, 0.0, 0.05], [0.0, 0.0, 0.38], [0.0, 0.0, 0.45]]], dtype=torch.float64)
    q = world.query(pts)
    # Point 0 near floor (mat 0, phi ~0.05); Point 1 just outside box bottom (phi ~0.02); Point 2 inside box (phi < 0)
    assert int(q.mat_id[0, 0].item()) == 0 and float(q.phi[0, 0].item()) > 0.03
    assert int(q.mat_id[0, 1].item()) == 1 and abs(float(q.phi[0, 1].item()) - 0.02) < 1e-6
    assert int(q.mat_id[0, 2].item()) == 1 and float(q.phi[0, 2].item()) < -0.04

    # VoxelGridSDF trilinear accuracy on analytic sphere of radius 0.2 centered at origin
    xs = torch.linspace(-0.5, 0.5, 21, dtype=torch.float64)
    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing='ij')
    sphere_phi = torch.sqrt(gx * gx + gy * gy + gz * gz) - 0.2
    vox = VoxelGridSDF(sphere_phi, (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), mat_id=1)
    qv = vox.query(torch.tensor([[[0.30, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64))
    assert abs(float(qv.phi[0, 0].item()) - 0.10) < 0.01
    assert float(qv.phi[0, 1].item()) < -0.15
    assert float(qv.normal[0, 0, 0].item()) > 0.95


def test_c2_self_contact_exact_gradient_and_psd_hessian():
    """Verify Py-DiSMech C2 smooth barrier force == -dE/dp via finite differences and PSD Hessian."""
    pos = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0042, 0.0005, 0.0], [0.050, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    beam_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    f, K_diag, E0 = eval_c2_self_contact(pos, beam_ids, radius_m=0.002, delta_m=0.002, stiffness_k=2000.0, dt=0.001)

    # Node 2 is far away (> 2h+delta=0.006) -> strictly zero force & stiffness
    assert float(f[0, 2].abs().max().item()) == 0.0
    assert float(K_diag[0, 2].abs().max().item()) == 0.0
    # Newton's 3rd law on interacting pair (0, 1)
    assert float((f[0, 0] + f[0, 1]).abs().max().item()) < 1e-12

    # Finite-difference check: F_i = -dE / dp_i in the C2 smooth transition zone
    eps = 1e-7
    for axis in range(3):
        p_plus = pos.clone()
        p_minus = pos.clone()
        p_plus[0, 0, axis] += eps
        p_minus[0, 0, axis] -= eps
        _, _, ep = eval_c2_self_contact(p_plus, beam_ids, radius_m=0.002, delta_m=0.002, stiffness_k=2000.0, dt=0.001)
        _, _, em = eval_c2_self_contact(p_minus, beam_ids, radius_m=0.002, delta_m=0.002, stiffness_k=2000.0, dt=0.001)
        fd_force = -float((ep - em).item()) / (2.0 * eps)
        assert abs(fd_force - float(f[0, 0, axis].item())) < 1e-5

    # PSD Hessian eigenvalues >= -1e-15
    eigs = torch.linalg.eigvalsh(K_diag[0, 0])
    assert float(eigs.min().item()) >= -1e-15


# ---------------------------------------------------------------------------
# 5. Morphology-Agnostic TokenizedHIMDreamWaQ & Universal Finite-Group Equivariance
# ---------------------------------------------------------------------------
def test_tokenized_policy_variable_morphology_and_fault_masking():
    torch.manual_seed(0)
    policy = TokenizedHIMDreamWaQ(TokenizedPolicyConfig(d_tok=8, d_glob=9, d_model=32, n_heads=4, n_attn_layers=2))
    policy.eval()

    # Single weight set runs seamlessly across N = 4 (endoscope), 8 (3-seg), 14 (5-seg), 20 (7-seg)!
    for n_act in (4, 8, 14, 20):
        tokens = torch.randn(2, 16, n_act, 8)
        glob = torch.randn(2, 16, 9)
        out = policy(tokens, glob)
        assert out.action_mean.shape == (2, n_act)
        assert out.est_pred.shape == (2, 7)
        assert torch.isfinite(out.action_mean).all()

    # Live actuator failure masking strictly zeros disabled actuator while preserving valid actions
    tokens = torch.randn(2, 16, 14, 8)
    glob = torch.randn(2, 16, 9)
    mask = torch.ones((2, 14), dtype=torch.bool)
    mask[0, 3] = False
    mask[1, 10] = False
    out_m = policy(tokens, glob, active_mask=mask)
    assert float(out_m.action_mean[0, 3].item()) == 0.0
    assert float(out_m.action_mean[1, 10].item()) == 0.0
    assert torch.isfinite(out_m.action_mean).all()

    # Auxiliary losses (EstNet + beta-VAE + HIM InfoNCE + 3-step Sinkhorn SwAV) backprop cleanly
    policy.train()
    losses = policy.compute_aux_losses(
        tokens, glob, next_glob_sens=torch.randn(2, 6), est_target=torch.randn(2, 7), active_mask=mask
    )
    losses['total_aux'].backward()
    assert torch.isfinite(losses['total_aux'])


@pytest.mark.parametrize(
    'group_fn,n_act,glob_signs',
    [
        (lambda: make_z2_bilateral_symmetry(5), 14, [[1.0] * 9, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]]),
        (lambda: make_z2xz2_bidirectional_symmetry(4), 11, None),
        (lambda: make_d4_radial_symmetry(4), 4, None),
    ],
)
def test_universal_finite_group_reynolds_equivariance(group_fn, n_act: int, glob_signs):
    """Exact algebraic equivariance pi_eq(L_g o) == K_g pi_eq(o) across Z2, Z2xZ2 (|G|=4), and D4 (|G|=8)."""
    torch.manual_seed(42)
    group = group_fn()
    base = TokenizedHIMDreamWaQ(TokenizedPolicyConfig(d_tok=8, d_glob=9, d_model=32, n_heads=4, n_attn_layers=1)).double()
    base.eval()
    eq = FiniteGroupEquivariantWrapper(base, group, glob_signs=glob_signs).double()
    eq.eval()

    tokens = torch.randn(3, 16, n_act, 8, dtype=torch.float64)
    glob = torch.randn(3, 16, 9, dtype=torch.float64)
    out0 = eq(tokens, glob)

    # Verify EVERY group element g in G satisfies pi_eq(L_g o) == K_g pi_eq(o) to machine precision!
    for g_idx in range(group.order):
        tg, gg, _ = eq.transform_inputs(tokens, glob, g_idx)
        out_g = eq(tg, gg)
        expected_act = apply_signed_perm(
            out0.action_mean, eq.act_perms[g_idx], eq.act_signs[g_idx].double(), dim=-1
        )
        err = float((out_g.action_mean - expected_act).abs().max().item())
        assert err < 1e-12, f'Group {group.name} element {g_idx} equivariance error={err}'


def test_closed_loop_single_policy_across_heterogeneous_envs_and_safety_guard():
    """One shared TokenizedHIMDreamWaQ policy drives both SerpWorm-N2 and 3D TendonEndoscope closed-loop."""
    torch.manual_seed(7)
    policy = TokenizedHIMDreamWaQ(TokenizedPolicyConfig(d_tok=8, d_glob=9, d_model=32, n_heads=4, n_attn_layers=1))
    policy.eval()

    # Env A: 2-segment SerpWorm on hard floor with 1 jammed actuator (actuator 2 disabled)
    env_worm = FlexGraphEnv(
        build_serpworm_graph(n_segments=2, elements_per_strip=6),
        world=ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)]),
        batch=1,
        dt=0.001,
        substeps_per_policy=5,
        history_len=8,
    )
    mask_fault = torch.tensor([[True, True, False, True, True]])
    env_worm.set_failed_actuators(mask_fault)
    tok, glob, mask = env_worm.reset()
    for _ in range(3):
        with torch.no_grad():
            act = policy(tok, glob, mask).action_mean
        (tok, glob, mask), rew, info = env_worm.step(act)
        assert not info['failed']
        assert torch.isfinite(rew).all()
        assert float(env_worm.u_prev[0, 2].item()) == 0.0  # jammed actuator stayed frozen

    # Env B: 3D fixed-base 4-tendon endoscope driven by the EXACT SAME policy weights!
    env_endo = FlexGraphEnv(
        build_tendon_endoscope_graph(n_sections=2, elements_per_section=5),
        world=ComposableSDFWorld(primitives=[]),
        batch=1,
        dt=0.001,
        substeps_per_policy=5,
        history_len=8,
    )
    tok_e, glob_e, mask_e = env_endo.reset()
    for _ in range(3):
        with torch.no_grad():
            act_e = policy(tok_e, glob_e, mask_e).action_mean
        (tok_e, glob_e, mask_e), rew_e, info_e = env_endo.step(act_e)
        assert not info_e['failed']
        assert torch.isfinite(rew_e).all()

    manifest = env_endo.export_manifest()
    assert manifest['n_actuators'] == 4 and manifest['symmetry_group'] == 'D4_radial_4tendon'

    # UniversalSafetyGuard: rate-limit + tilt E-stop latch
    guard = UniversalSafetyGuard(SafetyConfig(max_tilt_rad=1.0, max_cmd_delta=0.25))
    prev = torch.zeros((1, 4))
    safe_u = guard.filter_action(torch.ones((1, 4)), prev, grav_body=torch.tensor([[0.0, 0.0, -1.0]]))
    assert float(safe_u.max().item()) <= 0.25 + 1e-6
    assert not guard.estop_latched
    # Exceed tilt -> latches E-stop and zeros command permanently until reset
    zero_u = guard.filter_action(torch.ones((1, 4)), safe_u, grav_body=torch.tensor([[1.0, 0.0, 0.0]]))
    assert guard.estop_latched and guard.estop_reason == 'tilt_exceeded'
    assert float(zero_u.abs().max().item()) == 0.0


"""Comprehensive unit & integration tests for v8 HIMDreamWaQActorCritic and GroupEquivariantPolicyWrapper."""
from __future__ import annotations

from pathlib import Path
import time

import pytest
import torch

from serpworm.config import DomainRandConfig, HardwareConfig
from serpworm.deploy.policy_runtime import load_checkpoint_model_and_meta
from serpworm.envs.autonomous_env import AutonomousPhaseEnv, AutonomousTaskConfig, mirror_autonomous_batch
from serpworm.envs.unified_env import mirror_rollout_batch
from serpworm.rl.him_dreamwaq import (
    GroupEquivariantPolicyWrapper,
    HIMDreamWaQActorCritic,
    HIMDreamWaQConfig,
    adaboot_prob,
    build_action_mirror_spec,
    build_est_mirror_spec,
    build_obs_mirror_spec,
    mirror_tensor_by_spec,
    sinkhorn_knopp,
)
from serpworm.rl.models import ProprioceptiveEstimatorActorCritic, TemporalActorCritic

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Multi-Scale Causal TCN & HIMDreamWaQ PPO / Aux-Loss Verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('history_len', [1, 4, 8, 16, 20])
@pytest.mark.parametrize('frame_size', [51, 65])
def test_multiscale_causal_tcn_shapes_and_strict_causality(history_len: int, frame_size: int):
    torch.manual_seed(0)
    model = HIMDreamWaQActorCritic(
        frame_size=frame_size,
        history_len=history_len,
        privileged_size=108,
        est_size=7,
        action_size=14,
    ).eval()
    x = torch.randn(2, history_len, frame_size)
    with torch.no_grad():
        feat = model.context_tcn(x.transpose(1, 2))
    assert feat.shape == (2, 48, model.t_out)
    assert torch.isfinite(feat).all()

    if history_len >= 6:
        # Strict causality: perturbing the last frame must NOT alter earlier strided time slots
        x_pert = x.clone()
        x_pert[:, -1, :] += 10.0
        with torch.no_grad():
            f0 = model.context_tcn(x.transpose(1, 2))
            f1 = model.context_tcn(x_pert.transpose(1, 2))
        assert torch.allclose(f0[:, :, :-1], f1[:, :, :-1], atol=1e-6)
        assert float((f0[:, :, -1] - f1[:, :, -1]).abs().max()) > 1e-4


@pytest.mark.parametrize('frame_size,priv_size', [(51, 102), (65, 108)])
def test_him_dreamwaq_ppo_interface_and_all_gradients(frame_size: int, priv_size: int):
    torch.manual_seed(17)
    model = HIMDreamWaQActorCritic(
        frame_size=frame_size,
        history_len=16,
        privileged_size=priv_size,
        est_size=7,
        action_size=14,
    )
    bsz = 12
    obs = torch.randn(bsz, 16, frame_size)
    next_obs_seq = torch.randn(bsz, 16, frame_size)
    next_obs_frame = next_obs_seq[:, -1, :]
    priv = torch.randn(bsz, priv_size)
    est_tgt = torch.randn(bsz, 7)

    model.obs_norm.update(obs[:, -1, :])
    model.priv_norm.update(priv)

    # Rollout sampling interface
    act, lat, lp, val = model.sample(obs, priv)
    assert act.shape == (bsz, 14) and lat.shape == (bsz, 14)
    assert lp.shape == (bsz,) and val.shape == (bsz,)
    assert float(act.detach().abs().max().item()) <= 1.0

    # Deterministic eval consistency
    model.eval()
    with torch.no_grad():
        a1, e1 = model.action_with_estimate(obs)
        a2 = model.action(obs)
        a3, e3, z_mu, z_gated = model.action_with_context(obs)
    assert torch.equal(a1, a2) and torch.equal(a1, a3) and torch.equal(e1, e3)
    assert z_mu.shape == (bsz, 16) and z_gated.shape == (bsz, 16)
    assert (e1[:, 3] >= 0.0).all() and (e1[:, 3] <= 1.0).all()
    assert (e1[:, 4] >= -1.0).all() and (e1[:, 4] <= 1.0).all()
    assert (e1[:, 5:7] >= 0.0).all() and (e1[:, 5:7] <= 1.0).all()

    # Combined PPO + DreamWaQ beta-VAE + HIM contrastive backward pass
    model.train()
    lp2, ent, v2, det_mu, est_pred = model.evaluate_full(obs, priv, lat)
    assert lp2.shape == (bsz,) and ent.shape == (bsz,) and v2.shape == (bsz,)
    assert det_mu.shape == (bsz, 14) and est_pred.shape == (bsz, 7)

    aux_loss_3d, metrics_3d = model.compute_aux_losses(obs, next_obs_seq, act.detach(), est_tgt)
    aux_loss_2d, _ = model.compute_aux_losses(obs, next_obs_frame, act.detach(), est_tgt)
    assert torch.isfinite(aux_loss_3d) and torch.isfinite(aux_loss_2d)
    for k in ('est_loss', 'rec_loss', 'kl_loss', 'infonce_loss', 'swav_loss', 'him_loss'):
        assert k in metrics_3d and torch.isfinite(metrics_3d[k])
    assert float(metrics_3d['kl_loss']) >= -1e-6

    total_loss = -lp2.mean() - 0.01 * ent.mean() + 0.5 * v2.square().mean() + aux_loss_3d
    total_loss.backward()

    # Every trainable parameter must receive a finite non-None gradient; EMA stays frozen
    for name, p in model.named_parameters():
        if name.startswith('him_tgt_ema.'):
            assert not p.requires_grad and p.grad is None
        else:
            assert p.requires_grad
            assert p.grad is not None, f'Missing gradient on {name}'
            assert torch.isfinite(p.grad).all(), f'Non-finite gradient on {name}'

    # EMA update must move shadow weights smoothly toward online target weights
    before_ema = [p.clone() for p in model.him_tgt_ema.parameters()]
    with torch.no_grad():
        for p in model.him_tgt_proj.parameters():
            p.add_(0.25)
    model.update_ema_target(momentum=0.9)
    diffs = [float((a - b).abs().max()) for a, b in zip(before_ema, model.him_tgt_ema.parameters())]
    assert max(diffs) > 1e-3

    # Edge cases: B=1, use_ema_target=False, swav_coef=0.0
    loss_b1, _ = model.compute_aux_losses(obs[:1], next_obs_frame[:1], act[:1].detach(), est_tgt[:1])
    loss_no_ema, m_no_swav = model.compute_aux_losses(
        obs, next_obs_frame, act.detach(), est_tgt, use_ema_target=False, swav_coef=0.0
    )
    assert torch.isfinite(loss_b1) and torch.isfinite(loss_no_ema)
    assert float(m_no_swav['swav_loss']) == 0.0

    # Parameter summary sanity
    ps = model.parameter_summary()
    assert ps['deploy_params'] > 0 and ps['critic_params'] > 0 and ps['aux_params'] > 0
    assert ps['deploy_params'] < ps['total_trainable_params']


def test_him_dreamwaq_bf16_autocast_stability():
    torch.manual_seed(23)
    model = HIMDreamWaQActorCritic(frame_size=65, history_len=16, privileged_size=108)
    obs = torch.randn(16, 16, 65)
    nobs = torch.randn(16, 65)
    priv = torch.randn(16, 108)
    est = torch.randn(16, 7)

    with torch.amp.autocast('cpu', dtype=torch.bfloat16):
        act, lat, lp, val = model.sample(obs, priv)
        lp2, ent, v2, mu, est_p = model.evaluate_full(obs, priv, lat)
        aux, metrics = model.compute_aux_losses(obs, nobs, act.detach(), est)
        loss = -lp2.mean() + 0.5 * v2.square().mean() + aux

    assert all(torch.isfinite(t).all() for t in (act, lat, lp, val, lp2, ent, v2, mu, est_p, aux))
    loss.backward()
    assert all(
        (p.grad is not None and torch.isfinite(p.grad).all())
        for n, p in model.named_parameters()
        if not n.startswith('him_tgt_ema.')
    )


def test_adaboot_prob_and_sinkhorn_knopp():
    # Stable high rewards -> high bootstrap probability
    p_stable = adaboot_prob(torch.full((32,), 10.0) + 0.1 * torch.randn(32))
    # Noisy early rewards -> low bootstrap probability
    p_noisy = adaboot_prob(torch.tensor([-5.0, 8.0, -3.0, 9.0, -2.0, 7.0]))
    assert 0.85 < p_stable <= 1.0
    assert 0.0 <= p_noisy < 0.25
    assert adaboot_prob([1.0]) == 0.0

    scores = torch.randn(16, 12) * 5.0
    q = sinkhorn_knopp(scores, eps=0.05, n_iters=3)
    assert q.shape == (16, 12) and torch.isfinite(q).all()
    assert (q >= 0.0).all()
    assert torch.allclose(q.sum(dim=-1), torch.ones(16), atol=1e-4)


# ---------------------------------------------------------------------------
# 2. Exact Z2 Mirror Spec Parity & GroupEquivariantPolicyWrapper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('include_q_err,frame_size', [(False, 51), (True, 65)])
def test_mirror_spec_matches_autonomous_env_exactly(include_q_err: bool, frame_size: int):
    torch.manual_seed(3)
    obs = torch.randn(8, 16, frame_size)
    priv = torch.randn(8, 108)
    est = torch.randn(8, 7)
    act = torch.randn(8, 14)

    ref_obs, _, ref_est, ref_act = mirror_autonomous_batch(obs, priv, est, act, include_q_err=include_q_err)

    obs_p, obs_s = build_obs_mirror_spec(frame_size, include_q_err=include_q_err)
    act_p, act_s = build_action_mirror_spec(14)
    est_s = build_est_mirror_spec(7)

    got_obs = mirror_tensor_by_spec(obs, obs_p, obs_s)
    got_act = mirror_tensor_by_spec(act, act_p, act_s)
    got_est = est * est_s

    assert torch.equal(got_obs, ref_obs)
    assert torch.equal(got_act, ref_act)
    assert torch.equal(got_est, ref_est)
    # Exact involution g(g(x)) == x
    assert torch.equal(mirror_tensor_by_spec(got_obs, obs_p, obs_s), obs)
    assert torch.equal(mirror_tensor_by_spec(got_act, act_p, act_s), act)


@pytest.mark.parametrize('pairing,frame_size,action_size', [('independent_10d', 54, 14), ('paired_5d', 49, 9)])
def test_mirror_spec_matches_unified_env_exactly(pairing: str, frame_size: int, action_size: int):
    torch.manual_seed(5)
    obs = torch.randn(6, 16, frame_size)
    priv = torch.randn(6, 101)
    act = torch.randn(6, action_size)

    ref_obs, _, ref_act = mirror_rollout_batch(obs, priv, act, tendon_pairing=pairing)  # type: ignore[arg-type]
    obs_p, obs_s = build_obs_mirror_spec(frame_size, action_size=action_size)
    act_p, act_s = build_action_mirror_spec(action_size)

    assert torch.equal(mirror_tensor_by_spec(obs, obs_p, obs_s), ref_obs)
    assert torch.equal(mirror_tensor_by_spec(act, act_p, act_s), ref_act)


@pytest.mark.parametrize('model_kind,frame_size,action_size', [
    ('v6_est', 51, 14),
    ('v6_est', 65, 14),
    ('v5_temp', 54, 14),
    ('v5_temp', 49, 9),
    ('v8_him', 51, 14),
    ('v8_him', 65, 14),
])
def test_group_equivariant_wrapper_exact_zero_error_all_architectures(
    model_kind: str, frame_size: int, action_size: int
):
    torch.manual_seed(42)
    if model_kind == 'v6_est':
        base = ProprioceptiveEstimatorActorCritic(frame_size=frame_size, history_len=16, privileged_size=108, action_size=action_size).eval()
    elif model_kind == 'v5_temp':
        base = TemporalActorCritic(frame_size=frame_size, history_len=16, privileged_size=101, action_size=action_size).eval()
    else:
        base = HIMDreamWaQActorCritic(frame_size=frame_size, history_len=16, privileged_size=108, action_size=action_size).eval()

    eq_exact = GroupEquivariantPolicyWrapper(base, batched=False).eval()
    eq_batch = GroupEquivariantPolicyWrapper(base, batched=True).eval()

    obs = torch.randn(8, 16, frame_size)
    m_obs = eq_exact.mirror_obs(obs)

    with torch.no_grad():
        # Raw random-init policy is strictly asymmetric
        raw_err = float((base.action(m_obs) - eq_exact.mirror_action(base.action(obs))).abs().max().item())
        assert raw_err > 1e-3

        # Wrapped (batched=False) is BIT-EXACT 0.0 equivariant
        eq_a = eq_exact.action(obs)
        eq_a_m = eq_exact.action(m_obs)
        assert float((eq_a_m - eq_exact.mirror_action(eq_a)).abs().max().item()) == 0.0

        # Wrapped (batched=True) is < 1e-5 equivariant
        eqb_a = eq_batch.action(obs)
        eqb_a_m = eq_batch.action(m_obs)
        assert float((eqb_a_m - eq_batch.mirror_action(eqb_a)).abs().max().item()) < 1e-5

        if eq_exact.has_estimator:
            a1, e1 = eq_exact.action_with_estimate(obs)
            a1m, e1m = eq_exact.action_with_estimate(m_obs)
            assert float((a1m - eq_exact.mirror_action(a1)).abs().max().item()) == 0.0
            assert float((e1m - eq_exact.mirror_est(e1)).abs().max().item()) == 0.0

        if eq_exact.has_context:
            ac, ec, zc, zgc = eq_exact.action_with_context(obs)
            acm, ecm, zcm, zgcm = eq_exact.action_with_context(m_obs)
            assert float((acm - eq_exact.mirror_action(ac)).abs().max().item()) == 0.0
            assert float((ecm - eq_exact.mirror_est(ec)).abs().max().item()) == 0.0
            assert float((zcm - zc).abs().max().item()) == 0.0
            assert float((zgcm - zgc).abs().max().item()) == 0.0

        # Mittal et al. (2403.04359 Sec. III): at exact reflection-symmetric neutral/stop state L_g(o) == o,
        # all odd-parity yaw commands & lateral estimates MUST be identically 0.0
        sym_obs = 0.5 * (obs + m_obs)
        assert torch.allclose(sym_obs, eq_exact.mirror_obs(sym_obs), atol=1e-6)
        sym_stop_obs = sym_obs.clone()
        sym_stop_obs[:, :, -3] = 0.0  # cmd_vx = 0
        sym_stop_obs[:, :, -2] = 0.0  # cmd_wz = 0
        sym_stop_obs[:, :, -1] = 1.0  # stop = 1
        sym_act = eq_exact.action(sym_stop_obs)
        yaw_slice = slice(10, 14) if action_size == 14 else slice(5, 9)
        assert float(sym_act[:, yaw_slice].abs().max().item()) < 1e-6
        # Pure Reynolds mode (saddle_break_tol=0.0) zeroes yaw on ANY symmetric state (even moving)
        eq_pure = GroupEquivariantPolicyWrapper(base, batched=False, saddle_break_tol=0.0).eval()
        assert float(eq_pure.action(sym_obs)[:, yaw_slice].abs().max().item()) < 1e-6
        if eq_exact.has_estimator:
            sym_est = eq_exact.estimate(sym_obs)
            assert float(sym_est[:, [1, 2, 4]].abs().max().item()) < 1e-6


@pytest.mark.parametrize('ckpt_rel', [
    'runs/v6_1_65d_elec/best.pt',
    'runs/v6a_51d_pure/best.pt',
    'runs/unified_v5_hispeed_14d/best.pt',
])
def test_group_equivariant_wrapper_on_real_checkpoints(ckpt_rel: str):
    ckpt_path = REPO_ROOT / ckpt_rel
    if not ckpt_path.is_file():
        pytest.skip(f'Checkpoint not present locally: {ckpt_path}')
    model, meta = load_checkpoint_model_and_meta(ckpt_path, device='cpu')
    wrapper = GroupEquivariantPolicyWrapper(model, batched=False).eval()

    gen = torch.Generator().manual_seed(99)
    obs = torch.randn((16, meta.history_len, meta.frame_size), generator=gen) * 0.5
    m_obs = wrapper.mirror_obs(obs)

    with torch.inference_mode():
        raw_err = float((model.action(m_obs) - wrapper.mirror_action(model.action(obs))).abs().max().item())
        eq_err = float((wrapper.action(m_obs) - wrapper.mirror_action(wrapper.action(obs))).abs().max().item())
        assert raw_err > 1e-4
        assert eq_err == 0.0
        if wrapper.has_estimator:
            _, raw_e = model.action_with_estimate(obs)
            _, raw_em = model.action_with_estimate(m_obs)
            raw_est_err = float((raw_em - wrapper.mirror_est(raw_e)).abs().max().item())
            _, eq_e = wrapper.action_with_estimate(obs)
            _, eq_em = wrapper.action_with_estimate(m_obs)
            eq_est_err = float((eq_em - wrapper.mirror_est(eq_e)).abs().max().item())
            assert raw_est_err > 1e-5
            assert eq_est_err == 0.0


# ---------------------------------------------------------------------------
# 3. CPU Latency Parity & AutonomousPhaseEnv Integration Smoke
# ---------------------------------------------------------------------------

def test_him_dreamwaq_cpu_single_frame_latency():
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = HIMDreamWaQActorCritic(frame_size=65, history_len=16, privileged_size=108).eval()
        x = torch.randn(1, 16, 65)
        with torch.inference_mode():
            for _ in range(25):
                model.action_with_estimate(x)
            samples = []
            for _ in range(150):
                t0 = time.perf_counter()
                model.action_with_estimate(x)
                samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        p50_ms = samples[len(samples) // 2]
        # 10 Hz control budget is 100 ms; < 5.0 ms eager 1-thread CPU provides > 20x margin without Windows dispatch flake
        assert p50_ms < 5.0, f'Expected < 5.0 ms eager CPU single-frame latency, got {p50_ms:.3f} ms'
    finally:
        torch.set_num_threads(prev_threads)


def test_him_dreamwaq_autonomous_env_integration_smoke():
    hw = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
    task = AutonomousTaskConfig(elements_per_strip=6, physics_dt_s=0.002, include_q_err=True, history_len=16)
    env = AutonomousPhaseEnv(
        n=2, hardware=hw, task=task, dr=DomainRandConfig(enabled=False),
        device='cpu', plate_solver='segment_lu', seed=19, noise=False, settle_ticks=6,
    )
    model = HIMDreamWaQActorCritic(
        frame_size=env.frame_size,
        history_len=env.history_len,
        privileged_size=env.privileged_size,
        est_size=env.est_size,
        action_size=env.action_size,
        config=HIMDreamWaQConfig(),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs, priv, est_tgt = env.obs, env.critic_obs, env.est_target
    model.obs_norm.update(obs[:, -1, :])
    model.priv_norm.update(priv)
    with torch.no_grad():
        act, lat, _, _ = model.sample(obs, priv)
    nobs, npriv, nest, rew, term, trunc, tobs, tpriv, _ = env.step(act)

    lp, ent, val, mu, est_p = model.evaluate_full(obs, priv, lat)
    aux_loss, aux_m = model.compute_aux_losses(obs, tobs, act, est_tgt)
    loss = -lp.mean() + 0.5 * val.square().mean() + aux_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    model.update_ema_target()

    assert torch.isfinite(loss) and torch.isfinite(rew).all()
    assert torch.isfinite(mu).all() and torch.isfinite(est_p).all()
    assert float(aux_m['rec_loss']) >= 0.0


# ---------------------------------------------------------------------------
# 4. Z2 Saddle-Point Symmetry-Breaking Gate & Eval Sweep Post-Failure Masking
# ---------------------------------------------------------------------------

def test_group_equivariant_saddle_point_gate_behaviors():
    """Verify saddle-point symmetry breaking on straight moving rest while preserving exact 0.0 equivariance
    off the saddle manifold, on turning commands, and on static stop=1.0 commands."""
    torch.manual_seed(123)
    base = ProprioceptiveEstimatorActorCritic(
        frame_size=65, history_len=16, privileged_size=108, est_size=7, action_size=14
    ).eval()
    eq_safe = GroupEquivariantPolicyWrapper(base, batched=False, saddle_break_tol=0.02).eval()
    eq_safe_b = GroupEquivariantPolicyWrapper(base, batched=True, saddle_break_tol=0.02).eval()
    eq_naive = GroupEquivariantPolicyWrapper(base, batched=False, saddle_break_tol=0.0).eval()

    # Exact symmetric rest base observation: q=0, dq=0, q_err=0, gyro=0, proj_g=(0,0,-1), prev_act=0
    rest = torch.zeros(4, 16, 65)
    rest[:, :, 47] = -1.0  # gz = -1 (upright)
    # Row 0: stop=1.0 (must stay strictly symmetric, yaw == 0)
    rest[0, :, -3:] = torch.tensor([0.0, 0.0, 1.0])
    # Row 1: straight moving vx=100 (saddle gate MUST fire -> non-zero yaw initiation)
    rest[1, :, -3:] = torch.tensor([0.6, 0.0, 0.0])
    # Row 2: turning left cmd_wz=+0.5 (not straight -> exact Reynolds equivariance)
    rest[2, :, -3:] = torch.tensor([0.6, 0.5, 0.0])
    # Row 3: turning right cmd_wz=-0.5 (exact mirror of row 2)
    rest[3, :, -3:] = torch.tensor([0.6, -0.5, 0.0])

    with torch.no_grad():
        raw_a = base.action(rest)
        safe_a, safe_e = eq_safe.action_with_estimate(rest)
        safe_ab, safe_eb = eq_safe_b.action_with_estimate(rest)
        naive_a, _ = eq_naive.action_with_estimate(rest)

        # (a) Static stop=1.0 stays strictly zero-yaw even with saddle_break_tol > 0
        assert float(safe_a[0, 10:14].abs().max().item()) == 0.0
        assert float(safe_ab[0, 10:14].abs().max().item()) < 1e-6

        # (b) Straight moving at exact symmetric rest breaks saddle (matches raw base action)
        assert float(naive_a[1, 10:14].abs().max().item()) == 0.0
        assert float(raw_a[1, 10:14].abs().max().item()) > 1e-3
        assert torch.allclose(safe_a[1], raw_a[1], atol=1e-6)
        assert torch.allclose(safe_ab[1], raw_a[1], atol=1e-5)
        # Estimator stays strictly equivariant (lateral vy/wz/wall_side == 0 at symmetric rest)
        assert float(safe_e[1, [1, 2, 4]].abs().max().item()) == 0.0
        assert float(safe_eb[1, [1, 2, 4]].abs().max().item()) < 1e-6

        # (c) Turning commands preserve exact left/right mirror equivariance (0.0 on matched batch lanes, <1e-6 across lanes)
        assert float((safe_a[3] - eq_safe.mirror_action(safe_a[2])).abs().max().item()) < 1e-6
        assert float((safe_e[3] - eq_safe.mirror_est(safe_e[2])).abs().max().item()) < 1e-6
        turn_l = rest[2:3]
        turn_r = eq_safe.mirror_obs(turn_l)
        al, el = eq_safe.action_with_estimate(turn_l)
        ar, er = eq_safe.action_with_estimate(turn_r)
        assert float((ar - eq_safe.mirror_action(al)).abs().max().item()) == 0.0
        assert float((er - eq_safe.mirror_est(el)).abs().max().item()) == 0.0

        # (d) Off the saddle manifold (||o - L_g o||_inf >= tol) on straight moving -> bit-exact 0.0 equivariance
        off_saddle = rest[1:2].clone()
        off_saddle[:, -1, 10] = 0.05  # yaw angle asymmetry -> |o - L_g o| = 0.10 >> 0.02
        m_off = eq_safe.mirror_obs(off_saddle)
        a_off = eq_safe.action(off_saddle)
        a_off_m = eq_safe.action(m_off)
        assert float((a_off_m - eq_safe.mirror_action(a_off)).abs().max().item()) == 0.0
        assert torch.equal(a_off, eq_naive.action(off_saddle))

    # Also verify optional saddle_break_tol on FlexGraph FiniteGroupEquivariantWrapper using real FlexGraphEnv tokens
    from flexgraph import (
        FiniteGroupEquivariantWrapper,
        FlexGraphEnv,
        TokenizedHIMDreamWaQ,
        TokenizedPolicyConfig,
        build_serpworm_graph,
        default_glob_signs_for_group,
        make_z2_bilateral_symmetry,
    )
    from flexgraph.policy import apply_signed_perm

    fg_env = FlexGraphEnv(
        build_serpworm_graph(3, elements_per_strip=4),
        batch=2,
        history_len=8,
        device='cpu',
        dtype=torch.float64,
    )
    tok, glob, mask = fg_env.reset()  # real metadata: side=+1/-1 on tendon pairs, 0 on yaw
    assert float(tok[0, 0, 0, 1].item()) == 1.0 and float(tok[0, 0, 1, 1].item()) == -1.0
    glob = glob.clone()
    glob[0, :, 6:9] = torch.tensor([0.0, 0.0, 1.0])  # stop
    glob[1, :, 6:9] = torch.tensor([1.0, 0.0, 0.0])  # straight forward

    grp = make_z2_bilateral_symmetry(3)  # N=8 (6 tendons + 2 yaw)
    gs = default_glob_signs_for_group(grp, d_glob=9)
    fg_base = TokenizedHIMDreamWaQ(TokenizedPolicyConfig(d_model=16, n_heads=4, n_attn_layers=1)).eval()
    fg_naive = FiniteGroupEquivariantWrapper(fg_base, grp, glob_signs=gs, saddle_break_tol=0.0).eval()
    fg_safe = FiniteGroupEquivariantWrapper(fg_base, grp, glob_signs=gs, saddle_break_tol=0.02).eval()

    with torch.no_grad():
        s_mask = fg_safe._saddle_break_mask(tok, glob, mask)
        assert s_mask is not None and s_mask.tolist() == [False, True]
        fg_raw_m = fg_base(tok, glob, active_mask=mask).action_mean
        fg_naive_m = fg_naive(tok, glob, active_mask=mask).action_mean
        fg_safe_m = fg_safe(tok, glob, active_mask=mask).action_mean
        assert float(fg_naive_m[:, 6:8].abs().max().item()) < 1e-6
        assert float(fg_safe_m[0, 6:8].abs().max().item()) < 1e-6  # stop stays symmetric
        assert float(fg_raw_m[1, 6:8].abs().max().item()) > 1e-4
        assert torch.allclose(fg_safe_m[1], fg_raw_m[1], atol=1e-6)  # straight forward breaks saddle

        # Off-saddle on real FlexGraphEnv tokens -> gate turns off and exact Reynolds equivariance holds
        tok_off = tok[1:2].clone()
        glob_off = glob[1:2].clone()
        mask_off = mask[1:2].clone()
        tok_off[:, -1, 6, 4] = 0.05  # yaw q asymmetry -> |dyn - L_g dyn| = 0.10 >> 0.02
        assert fg_safe._saddle_break_mask(tok_off, glob_off, mask_off) is None
        tok_off_g, glob_off_g, mask_off_g = fg_safe.transform_inputs(tok_off, glob_off, 1, mask_off)
        a_off_fg = fg_safe(tok_off, glob_off, active_mask=mask_off).action_mean
        a_off_fg_g = fg_safe(tok_off_g, glob_off_g, active_mask=mask_off_g).action_mean
        a_off_expected = apply_signed_perm(a_off_fg, fg_safe.act_perms[1], fg_safe.act_signs[1], dim=-1)
        assert float((a_off_fg_g - a_off_expected).abs().max().item()) < 1e-6
        assert torch.allclose(a_off_fg, fg_naive(tok_off, glob_off, active_mask=mask_off).action_mean, atol=1e-6)


def test_eval_tracking_sweep_post_failure_masking_and_summary():
    """Verify eval_tracking_sweep.py excludes post-failure auto-reset steps from all metrics."""
    import importlib.util
    import types
    import numpy as np

    spec = importlib.util.spec_from_file_location('eval_tracking_sweep', REPO_ROOT / 'scripts' / 'eval_tracking_sweep.py')
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tracks = mod.build_steady_tracks()
    n = len(tracks)

    class _FakeCore:
        def __init__(self, batch: int):
            self.x = torch.zeros((batch, 10, 7), dtype=torch.float32)
            self.x[:, 9, 0] = 1.0

    class _FakeEnv:
        def __init__(self, batch: int, fail_idx: int | None, fail_step: int):
            self.n = batch
            self.device = torch.device('cpu')
            self.dtype = torch.float32
            self.policy_dt = 0.1
            self.task = types.SimpleNamespace(vx_max_mm_s=175.0, wz_max_rad_s=0.25)
            self.cmd = torch.zeros((batch, 3), dtype=torch.float32)
            self.history = torch.zeros((batch, 4, 65), dtype=torch.float32)
            self.c = _FakeCore(batch)
            self.last_guard_reasons: dict[str, torch.Tensor] = {}
            self._fail_idx = fail_idx
            self._fail_step = fail_step
            self._step = 0
            self._failed_already = torch.zeros(batch, dtype=torch.bool)

        def reset(self, mask: torch.Tensor):
            self._step = 0
            self._failed_already.zero_()
            return self.history.clone(), None, None

        def step(self, act: torch.Tensor):
            term = torch.zeros(self.n, dtype=torch.bool)
            clr = torch.full((self.n,), 0.005, dtype=torch.float32)
            bvel = torch.zeros((self.n, 3), dtype=torch.float32)
            # Healthy tracks track their commanded speed/turn cleanly
            bvel[:, 0] = self.cmd[:, 0]
            bvel[:, 2] = self.cmd[:, 1]
            # Simulate auto-reset contamination AFTER failure (fake 150 mm/s cruise on stop track!)
            if self._failed_already.any():
                bvel[self._failed_already, 0] = 0.150
                bvel[self._failed_already, 2] = -0.15

            wc_mask = torch.zeros(self.n, dtype=torch.bool)
            if self._fail_idx is not None and self._step == self._fail_step:
                term[self._fail_idx] = True
                wc_mask[self._fail_idx] = True
                clr[self._fail_idx] = 0.00083
                bvel[self._fail_idx, 0] = 0.002  # pre-failure velocity
                self._failed_already[self._fail_idx] = True
            self.last_guard_reasons = {'wheel_clearance': wc_mask}
            self.c.x[:, :10, 0] += bvel[:, 0:1] * self.policy_dt
            self._step += 1
            info = {
                'body_vel': bvel,
                'filtered_vel': bvel,
                'yaw_rms_deg': torch.zeros(self.n),
                'slip': torch.zeros(self.n),
                'clearance': clr,
            }
            return self.history.clone(), None, None, torch.zeros(self.n), term, torch.zeros(self.n, dtype=torch.bool), None, None, info

    class _DummyPolicy:
        def action_with_estimate(self, obs: torch.Tensor):
            b = obs.shape[0]
            return torch.zeros((b, 14)), torch.zeros((b, 7))

    # Steady sweep: track 0 (vx_0, stop=1.0) fails at step 5 (t=0.6s < steady_start_s=1.0s)
    env_s = _FakeEnv(n, fail_idx=0, fail_step=5)
    rows, arr_s = mod.run_steady_sweep(env_s, _DummyPolicy(), tracks, seconds=3.0, steady_start_s=1.0)
    r0 = rows[0]
    assert r0['name'] == 'vx_0' and r0['failed'] is True
    assert r0['fail_step'] == 5 and r0['fail_time_s'] == pytest.approx(0.6, abs=1e-5)
    assert r0['fail_reason'] == 'wheel_clearance'
    assert r0['actual_vx_mm_s'] is None  # post-reset 150 mm/s MUST NOT contaminate stop track!
    assert r0['pre_fail_vx_mm_s'] == pytest.approx(0.0, abs=1e-5)
    assert r0['min_clearance_mm'] == pytest.approx(0.83, abs=1e-3)
    assert not bool(arr_s['alive'][5:, 0].any())
    assert bool(np.isnan(arr_s['vx'][5:, 0]).all())

    metrics = mod.summarize_metrics(rows)
    assert metrics['failed_tracks_count'] == 1
    assert metrics['failed_tracks'] == ['vx_0']
    assert metrics['stop_failed'] is True
    assert metrics['stop_fail_reason'] == 'wheel_clearance'
    assert metrics['stop_vx_mm_s'] is None
    assert metrics['speed_mae_mm_s'] == pytest.approx(0.0, abs=1e-5)
    assert metrics['speed_r2'] == pytest.approx(1.0, abs=1e-5)

    # Dynamic profile: fail at step 10 -> steps >= 10 must be excluded (NaN / alive=False)
    env_d = _FakeEnv(1, fail_idx=0, fail_step=10)
    dyn_sum, dyn_arr = mod.run_dynamic_profile(env_d, _DummyPolicy(), seconds=6.0)
    assert dyn_sum['failed'] is True
    assert dyn_sum['fail_step'] == 10
    assert dyn_sum['fail_reason'] == 'wheel_clearance'
    assert dyn_sum['valid_steps'] == 10
    assert dyn_sum['overall_vx_mae_mm_s'] == pytest.approx(0.0, abs=1e-5)
    assert bool(np.isnan(dyn_arr['act_vx'][10:]).all())
    assert dyn_sum['segments'][-1]['actual_vx_mm_s'] is None

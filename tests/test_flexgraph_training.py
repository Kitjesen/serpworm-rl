"""Unit and end-to-end smoke tests for FlexGraph RL training, running normalization,
batched environment resets, finite-group equivariant PPO, and Rust FFI backend fallback.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flexgraph import (
    BUILTIN_ARCHETYPES,
    ComposableSDFWorld,
    FiniteGroupEquivariantWrapper,
    FlexGraphEnv,
    FlexGraphPPOTrainer,
    FlexGraphTrainConfig,
    HeightFieldSDF,
    RustGraphSolverBackend,
    TokenRunningNorm,
    TokenizedHIMDreamWaQ,
    TokenizedPolicyConfig,
    benchmark_backend_parity,
    build_archetype,
    build_bristle_worm_graph,
    build_flexspine_crawler_graph,
    build_serpworm_graph,
    build_tendon_endoscope_graph,
    compute_gae,
    default_glob_signs_for_group,
    graph_to_rust_ir_dict,
    graph_to_rust_ir_json,
    load_flexgraph_ffi,
    make_z2_bilateral_symmetry,
    world_to_ffi_config_dict,
)
from flexgraph.policy import apply_signed_perm
from flexgraph.rust_backend import (
    REQUIRED_BATCH_C_ABI_SYMBOLS,
    REQUIRED_C_ABI_SYMBOLS,
    REQUIRED_WORLD_C_ABI_SYMBOLS,
    discover_ffi_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / 'scripts' / 'train_flexgraph.py'


@pytest.fixture(autouse=True)
def _single_thread_cpu():
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# 1. Rust FFI Backend Bridge, IR Serialization, and Graceful Fallback
# ---------------------------------------------------------------------------
def test_rust_ffi_loader_ir_schema_and_safe_fallback():
    cands = discover_ffi_candidates()
    assert any('flexgraph-rs' in str(p) for p in cands)

    ffi = load_flexgraph_ffi()
    assert isinstance(ffi.available, bool)
    assert ffi.mode in ('cdylib', 'pyo3', 'cdylib_incomplete', 'unavailable')
    summary = ffi.summary_dict()
    assert 'mode' in summary and 'missing_symbols' in summary
    if not ffi.available:
        assert len(ffi.reason) > 0
        assert any(sym in ffi.missing_symbols for sym in REQUIRED_C_ABI_SYMBOLS)

    # All 4 robot archetypes serialize cleanly to the Rust fg-ir serde JSON schema
    graphs = [
        build_serpworm_graph(n_segments=2, elements_per_strip=4),
        build_bristle_worm_graph(n_segments=2, elements_per_strip=4),
        build_tendon_endoscope_graph(n_sections=2, elements_per_section=4, free_cantilever_tip=True),
        build_flexspine_crawler_graph(n_segments=2, elements_per_spine=4),
    ]
    for g in graphs:
        ir = graph_to_rust_ir_dict(g, dt=0.001)
        assert ir['name'] == g.name
        assert len(ir['bodies']) == len(g.bodies)
        assert len(ir['beams']) == len(g.beams)
        assert len(ir['tendons']) == len(g.tendons)
        assert len(ir['joints']) == len(g.joints)
        assert len(ir['contact_sites']) == len(g.contact_sites)
        assert ir['solver_hints']['preferred_dt'] == pytest.approx(0.001)
        raw_json = graph_to_rust_ir_json(g, dt=0.001)
        parsed = json.loads(raw_json)
        assert parsed['name'] == g.name

    # Cantilever endpoint must map to JSON null (Option<usize>::None in fg-ir)
    endo_ir = graph_to_rust_ir_dict(graphs[2])
    assert endo_ir['beams'][-1]['body_b'] is None
    assert endo_ir['beams'][-1]['clamp_b'] is None

    # Safe fallback in 'auto' and non-strict 'rust' modes
    g_small = graphs[0]
    w = ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)])
    sim_auto = RustGraphSolverBackend(g_small, world=w, batch=2, dt=0.001, backend='auto', strict_rust=False)
    sim_auto.step(torch.zeros((2, len(g_small.actuators)), dtype=torch.float64), steps=3)
    assert not bool(sim_auto.failed.item())
    assert torch.isfinite(sim_auto.x).all()
    assert sim_auto.active_backend in ('rust', 'python')
    sim_auto.reset_envs(torch.tensor([True, False]))
    assert torch.allclose(sim_auto.x[0], sim_auto.rest[0])
    sim_auto.close()

    # Non-existent library path falls back cleanly unless strict_rust=True
    missing_lib = REPO_ROOT / 'nonexistent_fg_ffi.dll'
    sim_fb = RustGraphSolverBackend(
        g_small, world=w, batch=1, dt=0.001, backend='rust', strict_rust=False, lib_path=missing_lib
    )
    assert sim_fb.active_backend == 'python'
    assert len(sim_fb.fallback_reason) > 0
    sim_fb.close()

    with pytest.raises(RuntimeError, match='Rust fg_ffi backend unavailable'):
        RustGraphSolverBackend(
            g_small, world=w, batch=1, dt=0.001, backend='rust', strict_rust=True, lib_path=missing_lib
        )

    # Parity benchmark helper runs cleanly on missing-lib fallback
    rep = benchmark_backend_parity(g_small, world=w, steps=4, dt=0.001, batch=1, lib_path=missing_lib)
    assert rep['active_backend'] == 'python'
    assert rep['max_pos_err_m'] == pytest.approx(0.0, abs=1e-15)
    assert rep['max_vel_err'] == pytest.approx(0.0, abs=1e-15)
    assert not rep['python_failed'] and not rep['backend_failed']

    # FfiConfigJson serializer matches Rust fg-ffi FfiConfigJson schema
    cfg_flat = world_to_ffi_config_dict(w, implicit_contact=True, use_kelvin_normal=False)
    assert cfg_flat['world_sdf'] == 'flat_ground' and cfg_flat['ground_mat_id'] == 0
    cfg_hf = world_to_ffi_config_dict(
        ComposableSDFWorld(primitives=[HeightFieldSDF(slope_x=0.05, wave_amp_m=0.002, mat_id=2)])
    )
    assert cfg_hf['world_sdf'] == 'heightfield'
    assert len(cfg_hf['ground_geom']) == 5 and cfg_hf['ground_mat_id'] == 2


def test_live_rust_fg_ffi_when_compiled():
    """Exercise the live compiled Rust fg_ffi cdylib (FgBatchHandle + parity) whenever present."""
    ffi = load_flexgraph_ffi()
    if not ffi.available:
        pytest.skip(f'Compiled fg_ffi library not present: {ffi.reason}')

    assert ffi.mode in ('cdylib', 'pyo3')
    if ffi.mode == 'cdylib':
        assert ffi.supports_batch_handle
        assert ffi.supports_world_handle
        for sym in REQUIRED_BATCH_C_ABI_SYMBOLS + REQUIRED_WORLD_C_ABI_SYMBOLS:
            assert sym in ffi.exported_symbols

    # Live Rayon batch handle creation, stepping, selective mask reset, and EstNet contact summary
    g = build_serpworm_graph(n_segments=2, elements_per_strip=4)
    w = ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)])
    sim = RustGraphSolverBackend(g, world=w, batch=3, dt=0.0005, backend='rust', strict_rust=True)
    assert sim.active_backend == 'rust'
    tgt = torch.tensor(
        [
            [0.4, 0.2, 0.3, 0.5, 0.1],
            [0.6, 0.6, 0.4, 0.4, -0.1],
            [0.2, 0.5, 0.5, 0.2, 0.0],
        ],
        dtype=torch.float64,
    )
    sim.step(tgt, steps=5)
    assert not bool(sim.failed.item())
    assert torch.isfinite(sim.x).all()
    assert torch.isfinite(sim.v).all()
    assert torch.isfinite(sim.act_state).all()
    assert sim.last_contact_summary is not None and sim.last_contact_summary.shape == (3, 7)
    assert torch.isfinite(sim.last_contact_summary).all()
    assert float(sim.v[0, : sim.P].abs().max().item()) > 1e-6

    # Selective Rayon reset: reset env 1 only; envs 0 and 2 retain non-zero velocities
    sim.reset_envs(torch.tensor([False, True, False]))
    assert torch.allclose(sim.x[1], sim.rest[1], atol=1e-12)
    assert float(sim.v[1].abs().max().item()) == 0.0
    assert float(sim.v[0, : sim.P].abs().max().item()) > 1e-6
    assert float(sim.v[2, : sim.P].abs().max().item()) > 1e-6
    sim.close()

    # Live Rust vs Python FP64 parity benchmark (using kelvin parity mode for exact match)
    rep = benchmark_backend_parity(
        g,
        world=w,
        steps=8,
        dt=0.0005,
        batch=1,
        use_kelvin_normal=True,
        implicit_contact=False,
    )
    assert rep['native_rust_available'] is True
    assert rep['active_backend'] == 'rust'
    assert not rep['python_failed'] and not rep['backend_failed']
    assert rep['max_pos_err_m'] < 1e-5, f'Unexpected Rust/Python parity divergence: {rep}'


# ---------------------------------------------------------------------------
# 2. TokenRunningNorm Masking, Metadata Preservation, and Eval Freeze
# ---------------------------------------------------------------------------
def test_token_running_norm_masking_and_eval_freeze():
    norm = TokenRunningNorm(d_tok=8, d_glob=9, d_priv=16, tok_norm_start=4)
    tokens = torch.randn(4, 6, 5, 8)
    glob = torch.randn(4, 6, 9)
    priv = torch.randn(4, 16)

    # Identity pass-through prior to any updates
    assert torch.equal(norm.normalize_tokens(tokens), tokens)
    assert torch.equal(norm.normalize_glob(glob), glob)
    assert torch.equal(norm.normalize_priv(priv), priv)

    # Masked-out actuators with NaN / extreme garbage must not pollute stats
    mask = torch.ones((4, 5), dtype=torch.bool)
    mask[0, 2] = False
    mask[3, 4] = False
    dirty_tokens = tokens.clone()
    dirty_tokens[0, :, 2, 4:] = float('nan')
    dirty_tokens[3, :, 4, 4:] = 1e9
    norm.train()
    norm.update(dirty_tokens, glob, active_mask=mask, priv=priv)
    assert float(norm.tok_count.item()) == 18.0  # 4*5 - 2 active actuators
    assert torch.isfinite(norm.tok_mean).all() and torch.isfinite(norm.tok_var).all()

    out_tok = norm.normalize_tokens(tokens)
    # Static metadata channels [0:4] must remain strictly untouched
    assert torch.equal(out_tok[..., :4], tokens[..., :4])
    assert torch.isfinite(out_tok).all()

    # Eval mode freezes statistics
    prev_count = float(norm.tok_count.item())
    norm.eval()
    norm.update(torch.full_like(tokens, 50.0), torch.full_like(glob, 50.0), active_mask=mask, priv=priv + 50.0)
    assert float(norm.tok_count.item()) == prev_count


# ---------------------------------------------------------------------------
# 3. FlexGraphEnv Batched API, Partial Reset, Truncation, and NaN Safety
# ---------------------------------------------------------------------------
def test_env_batch_api_partial_reset_and_nan_safety():
    g = build_bristle_worm_graph(n_segments=2, elements_per_strip=4)
    env = FlexGraphEnv(
        g,
        world=ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)]),
        batch=4,
        dt=0.001,
        substeps_per_policy=2,
        history_len=6,
        backend='auto',
        max_episode_steps=3,
        auto_reset=False,
        return_done=True,
    )
    tok, glob, mask = env.reset()
    assert tok.shape == (4, 6, env.N_act, 8)
    assert glob.shape == (4, 6, 9)
    assert mask.shape == (4, env.N_act)

    # Inject NaN/Inf action -> sanitized safely with zero NaNs in obs/reward/priv/est
    bad_act = torch.zeros((4, env.N_act))
    bad_act[0, 0] = float('nan')
    bad_act[1, 1] = float('inf')
    (tok1, glob1, _), rew1, done1, info1 = env.step(bad_act)
    assert torch.isfinite(tok1).all()
    assert torch.isfinite(glob1).all()
    assert torch.isfinite(rew1).all()
    assert info1['priv'].shape == (4, 16) and torch.isfinite(info1['priv']).all()
    assert info1['est_target'].shape == (4, 7) and torch.isfinite(info1['est_target']).all()
    assert done1.shape == (4,) and done1.dtype == torch.bool
    assert 'time_outs' in info1

    # Partial reset only resets masked environments
    env.reset_done(torch.tensor([True, False, True, False]))
    assert env.step_count.tolist() == [0, 1, 0, 1]

    # Step batch auto-reset on truncation (max_episode_steps=3)
    env.reset()
    for step_i in range(3):
        out = env.step_batch(torch.zeros((4, env.N_act)), auto_reset=True)
        if step_i < 2:
            assert not bool(out.done.any().item())
        else:
            assert bool(out.truncated.all().item())
            assert bool(out.done.all().item())
            assert not bool(out.terminated.any().item())
            assert env.step_count.tolist() == [0, 0, 0, 0]
            assert out.terminal_tokens.shape == (4, 6, env.N_act, 8)
            assert out.terminal_priv.shape == (4, 16)


# ---------------------------------------------------------------------------
# 4. Policy Asymmetric Critic, Gradient Flow, and Equivariant Training Ops
# ---------------------------------------------------------------------------
def test_policy_asymmetric_critic_gradients_and_equivariant_obs_norm():
    torch.manual_seed(11)
    cfg = TokenizedPolicyConfig(d_tok=8, d_glob=9, d_model=32, n_heads=4, n_attn_layers=1)
    policy = TokenizedHIMDreamWaQ(cfg)
    policy.train()

    # Check gradient flow through shared transformer + actor + critic across variable N
    tok = torch.randn(4, 8, 8, 8)
    glob = torch.randn(4, 8, 9)
    mask = torch.ones((4, 8), dtype=torch.bool)
    mask[0, 3] = False
    priv = torch.randn(4, 16)
    samp = policy.sample_actions(tok, glob, active_mask=mask, priv=priv)
    assert float(samp.action[0, 3].item()) == 0.0
    ev = policy.evaluate_actions(tok, glob, samp.latent.detach(), active_mask=mask, priv=priv)
    aux = policy.compute_aux_losses(tok, glob, next_glob_sens=torch.randn(4, 6), est_target=torch.randn(4, 7), active_mask=mask)
    loss = -ev.log_prob.mean() + ev.value.pow(2).mean() + aux['total_aux']
    loss.backward()

    for name in ('tok_tcn.proj.0.weight', 'glob_tcn.proj.0.weight', 'actor_head.0.weight', 'critic.0.weight', 'log_std'):
        param = dict(policy.named_parameters())[name]
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert float(param.grad.abs().max().item()) > 0.0, f'Zero gradient on {name}'

    # Equivariance holds to machine precision even AFTER orbit-averaged TokenRunningNorm updates!
    group = make_z2_bilateral_symmetry(3)  # N_act = 8
    gs = default_glob_signs_for_group(group, d_glob=9)
    base_eq = TokenizedHIMDreamWaQ(TokenizedPolicyConfig(d_model=16, n_heads=4, n_attn_layers=1)).double()
    eq = FiniteGroupEquivariantWrapper(base_eq, group, glob_signs=gs).double()
    eq.train()
    t64 = torch.randn(3, 8, 8, 8, dtype=torch.float64)
    g64 = torch.randn(3, 8, 9, dtype=torch.float64)
    p64 = torch.randn(3, 16, dtype=torch.float64)
    eq.update_obs_norm(t64, g64, priv=p64)
    eq.eval()

    out0 = eq(t64, g64)
    v0 = eq.evaluate_value(t64, g64, priv=p64)
    t1, g1, _ = eq.transform_inputs(t64, g64, 1)
    out1 = eq(t1, g1)
    v1 = eq.evaluate_value(t1, g1, priv=p64)

    expected_a1 = apply_signed_perm(out0.action_mean, eq.act_perms[1], eq.act_signs[1].double(), dim=-1)
    assert float((out1.action_mean - expected_a1).abs().max().item()) < 1e-12
    assert float((out1.z_mu - out0.z_mu).abs().max().item()) < 1e-12
    assert float((v1 - v0).abs().max().item()) < 1e-12


def test_gae_truncation_bootstrapping_vs_termination():
    rew = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    val = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    nval = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    term = torch.tensor([[False, False], [True, False]])
    trunc = torch.tensor([[False, False], [False, True]])
    adv, ret = compute_gae(rew, val, nval, term, trunc, gamma=0.9, gae_lambda=0.95)
    # Env 0 terminated at t=1 (no bootstrap: ret = 1.0); Env 1 truncated at t=1 (bootstraps: ret = 1.0 + 0.9*2.0 = 2.8)
    assert float(ret[1, 0].item()) == pytest.approx(1.0, abs=1e-6)
    assert float(ret[1, 1].item()) == pytest.approx(2.8, abs=1e-6)
    assert torch.isfinite(adv).all()


# ---------------------------------------------------------------------------
# 5. Multi-Morphology PPO Update, Fault Curriculum, Equivariant Mode & Checkpoint Roundtrip
# ---------------------------------------------------------------------------
def test_multi_morphology_ppo_update_and_checkpoint_roundtrip(tmp_path: Path):
    cfg = FlexGraphTrainConfig(
        morphologies=('bristle_worm', 'tendon_endoscope'),  # N=8 (Z2xZ2) + N=4 (D4)!
        schedule='mixture',
        envs_per_morph=2,
        horizon=4,
        rollouts=1,
        epochs=2,
        minibatches=2,
        substeps_per_policy=2,
        history_len=8,
        elements_per_beam=4,
        symmetry_reg_coef=0.15,
        fault_prob_start=0.5,
        fault_prob_end=0.5,
        device='cpu',
        backend='auto',
        eval_every=1,
        eval_steps=2,
        save_every=1,
        seed=13,
        policy=TokenizedPolicyConfig(d_model=32, n_heads=4, n_attn_layers=1),
    )
    trainer = FlexGraphPPOTrainer(cfg, out_dir=tmp_path / 'run1')
    before = {k: v.detach().clone() for k, v in trainer.policy.named_parameters()}
    summary = trainer.train()
    after = {k: v.detach().clone() for k, v in trainer.policy.named_parameters()}

    # Verify non-zero gradient updates on core modules
    changed = [
        k for k in before if float((after[k] - before[k]).abs().max().item()) > 0.0
    ]
    assert 'tok_tcn.proj.0.weight' in changed
    assert 'actor_head.0.weight' in changed
    assert 'critic.0.weight' in changed

    # Verify finite metrics and artifacts
    latest_rec = summary['latest']
    assert latest_rec is not None
    for key in ('policy_loss', 'value_loss', 'aux_loss', 'est_loss', 'approx_kl', 'grad_norm'):
        assert torch.isfinite(torch.tensor(latest_rec[key]))
    assert set(latest_rec['morphologies']) == {'bristle_worm', 'tendon_endoscope'}
    assert (tmp_path / 'run1' / 'latest.pt').is_file()
    assert (tmp_path / 'run1' / 'best.pt').is_file()
    assert (tmp_path / 'run1' / 'progress.json').is_file()
    assert (tmp_path / 'run1' / 'eval_summary.json').is_file()

    # Checkpoint save/load roundtrip preserves exact deterministic inference & obs_norm stats
    trainer2 = FlexGraphPPOTrainer(cfg, out_dir=tmp_path / 'run2', resume=tmp_path / 'run1' / 'latest.pt')
    assert trainer2.start_rollout == 1
    assert trainer2.total_samples == trainer.total_samples
    assert torch.allclose(trainer2.policy.obs_norm.tok_mean, trainer.policy.obs_norm.tok_mean)

    trainer.policy.eval()
    trainer2.policy.eval()
    tok_t = torch.randn(1, 8, 8, 8)
    glob_t = torch.randn(1, 8, 9)
    with torch.no_grad():
        a1 = trainer.policy(tok_t, glob_t).action_mean
        a2 = trainer2.policy(tok_t, glob_t).action_mean
    assert torch.allclose(a1, a2, atol=1e-7)
    trainer.close()
    trainer2.close()


def test_equivariant_wrapper_multi_morphology_rollout(tmp_path: Path):
    cfg = FlexGraphTrainConfig(
        morphologies=('serpworm_N3', 'tendon_endoscope'),
        schedule='round_robin',
        envs_per_morph=2,
        horizon=3,
        rollouts=2,
        epochs=1,
        minibatches=2,
        substeps_per_policy=2,
        history_len=8,
        elements_per_beam=4,
        use_equivariant_wrapper=True,
        symmetry_reg_coef=0.0,
        device='cpu',
        backend='python',
        eval_every=2,
        eval_steps=2,
        seed=5,
        policy=TokenizedPolicyConfig(d_model=16, n_heads=4, n_attn_layers=1),
    )
    trainer = FlexGraphPPOTrainer(cfg, out_dir=tmp_path / 'eq_run')
    summary = trainer.train()
    assert summary['rollouts_completed'] == 2
    assert summary['eval'] is not None
    assert set(summary['eval']['per_morphology'].keys()) == {'serpworm_N3', 'tendon_endoscope'}
    trainer.close()


def test_all_builtin_archetypes_construct_cleanly():
    assert set(BUILTIN_ARCHETYPES) == {
        'serpworm_N3',
        'serpworm_N5',
        'bristle_worm',
        'tendon_endoscope',
        'flexspine_crawler',
    }
    for name in BUILTIN_ARCHETYPES:
        g, w = build_archetype(name, elements_per_beam=4)
        assert len(g.bodies) >= 2 and len(g.actuators) >= 4
        assert w is not None


def test_trainer_post_reset_command_refresh_and_eval_failure_denominator(tmp_path: Path):
    """Regression checks for:
    1. collect_rollout refreshing env.glob_hist[:, :, 6:9] & returned glob after auto-reset.
    2. evaluate() dividing rew_sum by executed steps (not 1) if an env fails mid-episode.
    """
    cfg = FlexGraphTrainConfig(
        morphologies=('tendon_endoscope',),
        envs_per_morph=2,
        horizon=4,
        rollouts=1,
        epochs=1,
        minibatches=2,
        substeps_per_policy=2,
        history_len=4,
        max_episode_steps=2,
        elements_per_beam=4,
        device='cpu',
        backend='python',
        seed=19,
        policy=TokenizedPolicyConfig(d_model=16, n_heads=4, n_attn_layers=1),
    )
    trainer = FlexGraphPPOTrainer(cfg, out_dir=tmp_path / 'reg')
    rb = trainer.collect_rollout('tendon_endoscope', rollout_idx=1)
    env = trainer.envs['tendon_endoscope']
    # After 4 steps with max_episode_steps=2, resets occurred at t=1 and t=3; cached glob must match live env.cmd!
    _, cached_glob, _, _ = trainer.obs_cache['tendon_endoscope']
    assert torch.allclose(cached_glob[:, -1, 6:9], env.cmd.float(), atol=1e-6)
    assert torch.allclose(cached_glob[:, 0, 6:9], env.cmd.float(), atol=1e-6)
    # And at t=2 (immediately following reset at t=1), recorded rollout glob_buf[2] must match uniform post-reset cmd
    assert torch.allclose(rb.glob[2, :, 0, 6:9], rb.glob[2, :, -1, 6:9], atol=1e-6)

    # Verify evaluate() divides by actual executed step count on mid-episode termination
    orig_step_batch = FlexGraphEnv.step_batch
    call_counter = {'n': 0}

    def _fail_on_step4(self_env, action_delta, *, delta_scale=0.50, auto_reset=True):
        out = orig_step_batch(self_env, action_delta, delta_scale=delta_scale, auto_reset=auto_reset)
        call_counter['n'] += 1
        fixed_rew = torch.full_like(out.reward, 0.5)
        if call_counter['n'] == 4:
            term = torch.ones_like(out.terminated)
            info = dict(out.info)
            info['failed'] = True
            return out._replace(reward=fixed_rew, terminated=term, done=term, info=info)
        return out._replace(reward=fixed_rew)

    FlexGraphEnv.step_batch = _fail_on_step4  # type: ignore[method-assign]
    try:
        ev = trainer.evaluate(eval_steps=10)
    finally:
        FlexGraphEnv.step_batch = orig_step_batch  # type: ignore[method-assign]
        trainer.close()

    # 4 steps each with reward 0.5 -> mean_reward MUST be 0.5 (not 2.0 = sum/1!)
    assert call_counter['n'] == 4
    assert ev['per_morphology']['tendon_endoscope']['mean_reward'] == pytest.approx(0.5, abs=1e-4)
    assert ev['per_morphology']['tendon_endoscope']['failed_frac'] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 6. CLI --help and End-to-End 2-Rollout CPU Smoke Run
# ---------------------------------------------------------------------------
def test_cli_help_and_2rollout_cpu_smoke(tmp_path: Path):
    env_vars = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    res_help = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), '--help'],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env_vars,
        timeout=30,
    )
    assert res_help.returncode == 0, res_help.stderr
    for flag in ('--morphologies', '--schedule', '--device', '--backend', '--equivariant', '--fault-prob-end'):
        assert flag in res_help.stdout

    out_dir = tmp_path / 'cli_smoke'
    res_smoke = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            '--morphologies',
            'serpworm_N3,flexspine_crawler',
            '--schedule',
            'round_robin',
            '--device',
            'cpu',
            '--backend',
            'auto',
            '--envs',
            '2',
            '--horizon',
            '4',
            '--rollouts',
            '2',
            '--epochs',
            '1',
            '--minibatches',
            '2',
            '--substeps',
            '2',
            '--elements-per-beam',
            '4',
            '--history-len',
            '8',
            '--d-model',
            '32',
            '--n-heads',
            '4',
            '--n-attn-layers',
            '1',
            '--eval-every',
            '2',
            '--eval-steps',
            '2',
            '--out-dir',
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env_vars,
        timeout=300,
    )
    assert res_smoke.returncode == 0, f'STDOUT:\n{res_smoke.stdout}\nSTDERR:\n{res_smoke.stderr}'
    assert (out_dir / 'latest.pt').is_file()
    assert (out_dir / 'best.pt').is_file()
    assert (out_dir / 'config.json').is_file()
    assert (out_dir / 'progress.json').is_file()
    assert (out_dir / 'eval_summary.json').is_file()

    prog = json.loads((out_dir / 'progress.json').read_text(encoding='utf-8'))
    assert len(prog['history']) == 2
    assert prog['history'][0]['morphologies'] == ['serpworm_N3']
    assert prog['history'][1]['morphologies'] == ['flexspine_crawler']

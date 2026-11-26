"""Reproducible Multi-Morphology Asymmetric PPO + HIM-DreamWaQ Trainer for FlexGraph.

Trains a single shared ``TokenizedHIMDreamWaQ`` policy across one or multiple heterogeneous
rigid-flexible robot archetypes (``serpworm_N3``, ``serpworm_N5``, ``bristle_worm``,
``tendon_endoscope``, ``flexspine_crawler``) with:
  - Round-robin OR simultaneous mixture multi-morphology training schedules
  - Running entity-token/global/privileged observation normalization (``TokenRunningNorm``)
  - GAE with exact ``time_outs`` (truncation) bootstrapping and physical termination masks
  - ``rsl_rl``-style adaptive KL learning-rate scheduling + clipped Huber/MSE value loss
  - Coupled HIM-DreamWaQ auxiliary losses (7D EstNet + beta-VAE KL + reconstruction + InfoNCE + SwAV)
  - Dual-mode symmetry: exact ``FiniteGroupEquivariantWrapper`` OR fast rollout-time group-equivariant
    consistency regularization (across Z2, Z2xZ2, and D4 groups)
  - Progressive actuator fault-masking curriculum
  - Atomic checkpoints (``latest.pt``, ``best.pt``), strict JSON ``progress.json``, and deterministic
    cross-morphology evaluation summaries.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from flexgraph.env import FlexGraphEnv
from flexgraph.graph import (
    RigidFlexGraph,
    build_bristle_worm_graph,
    build_flexspine_crawler_graph,
    build_serpworm_graph,
    build_tendon_endoscope_graph,
)
from flexgraph.policy import (
    FiniteGroupEquivariantWrapper,
    TokenizedHIMDreamWaQ,
    TokenizedPolicyConfig,
    apply_signed_perm,
    default_glob_signs_for_group,
)
from flexgraph.rust_backend import benchmark_backend_parity, load_flexgraph_ffi
from flexgraph.sdf_world import ComposableSDFWorld, HeightFieldSDF, SplineTubeSDF

__all__ = [
    'BUILTIN_ARCHETYPES',
    'FlexGraphPPOTrainer',
    'FlexGraphTrainConfig',
    'RolloutBatch',
    'build_archetype',
    'compute_gae',
    'main',
    'parse_morphologies',
]

BUILTIN_ARCHETYPES: tuple[str, ...] = (
    'serpworm_N3',
    'serpworm_N5',
    'bristle_worm',
    'tendon_endoscope',
    'flexspine_crawler',
)


def parse_morphologies(spec: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Parse CLI morphology selector (comma-separated string, list, or ``'all'``)."""
    if isinstance(spec, str):
        raw = [tok.strip() for tok in spec.split(',') if tok.strip()]
    else:
        raw = []
        for item in spec:
            raw.extend(tok.strip() for tok in str(item).split(',') if tok.strip())
    if not raw or any(r.lower() == 'all' for r in raw):
        return BUILTIN_ARCHETYPES
    resolved: list[str] = []
    for name in raw:
        if name not in BUILTIN_ARCHETYPES:
            raise ValueError(f'Unknown morphology {name!r}; expected one of {BUILTIN_ARCHETYPES} or "all"')
        if name not in resolved:
            resolved.append(name)
    return tuple(resolved)


def build_archetype(
    name: str,
    *,
    elements_per_beam: int | None = None,
) -> tuple[RigidFlexGraph, ComposableSDFWorld]:
    """Instantiate a built-in FlexGraph archetype and its matched SDF world."""
    if name == 'serpworm_N3':
        ne = int(elements_per_beam) if elements_per_beam is not None else 8
        g = build_serpworm_graph(n_segments=3, elements_per_strip=ne)
        w = ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)])
        return g, w
    if name == 'serpworm_N5':
        ne = int(elements_per_beam) if elements_per_beam is not None else 8
        g = build_serpworm_graph(n_segments=5, elements_per_strip=ne)
        w = ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)])
        return g, w
    if name == 'bristle_worm':
        ne = int(elements_per_beam) if elements_per_beam is not None else 8
        g = build_bristle_worm_graph(n_segments=3, elements_per_strip=ne)
        w = ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)])
        return g, w
    if name == 'tendon_endoscope':
        ne = int(elements_per_beam) if elements_per_beam is not None else 6
        g = build_tendon_endoscope_graph(n_sections=3, elements_per_section=ne, free_cantilever_tip=False)
        tube = SplineTubeSDF(
            control_points=[(0.0, 0.0, 0.12), (0.09, 0.0, 0.12), (0.18, 0.025, 0.125)],
            radii_m=[0.030, 0.028, 0.026],
            samples_per_span=8,
            funnel_len_m=0.0,
            funnel_flare_m=0.0,
            mat_id=1,
        )
        w = ComposableSDFWorld(primitives=[tube])
        return g, w
    if name == 'flexspine_crawler':
        ne = int(elements_per_beam) if elements_per_beam is not None else 6
        g = build_flexspine_crawler_graph(n_segments=2, elements_per_spine=ne)
        w = ComposableSDFWorld(primitives=[HeightFieldSDF(mat_id=0)])
        return g, w
    raise ValueError(f'Unknown archetype {name!r}; expected one of {BUILTIN_ARCHETYPES}')


def compute_gae(
    reward: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) with exact time-out bootstrapping.

    - Bootstraps ``next_value`` on time-limit ``truncated`` steps (``~terminated``).
    - Zeroes bootstrap on physical ``terminated`` failures.
    - Cuts the eligibility trace across both ``terminated`` and ``truncated``.
    """
    adv = torch.zeros_like(reward)
    running = torch.zeros_like(reward[0])
    not_term = (~terminated).to(dtype=reward.dtype)
    not_done = (~(terminated | truncated)).to(dtype=reward.dtype)
    for t in reversed(range(reward.shape[0])):
        delta = reward[t] + gamma * not_term[t] * next_value[t] - value[t]
        running = delta + gamma * gae_lambda * not_done[t] * running
        adv[t] = running
    return adv, adv + value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


@dataclass
class FlexGraphTrainConfig:
    morphologies: tuple[str, ...] = ('serpworm_N3', 'bristle_worm', 'flexspine_crawler')
    schedule: str = 'round_robin'  # 'round_robin' | 'mixture'
    envs_per_morph: int = 32
    horizon: int = 24
    rollouts: int = 100
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    min_lr: float = 5e-5
    max_lr: float = 1e-3
    weight_decay: float = 0.0
    target_kl: float = 0.012
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    value_clip: bool = True
    use_huber_value_loss: bool = True
    huber_delta: float = 1.0
    entropy_coef: float = 0.005
    aux_coef: float = 0.5
    max_grad_norm: float = 1.0
    # Dual-mode symmetry & fault curriculum
    use_equivariant_wrapper: bool = False
    symmetry_reg_coef: float = 0.15
    fault_prob_start: float = 0.0
    fault_prob_end: float = 0.0
    fault_max_actuators: int = 1
    fault_warmup_frac: float = 0.2
    # Physics & observation setup
    dt: float = 0.001
    substeps_per_policy: int = 10
    history_len: int = 16
    max_episode_steps: int = 200
    elements_per_beam: int | None = None
    delta_scale: float = 0.50
    implicit_contact: bool = True
    enable_self_contact: bool = False
    # Runtime & logging
    device: str = 'auto'    # 'auto' | 'cpu' | 'cuda'
    backend: str = 'auto'   # 'auto' | 'python' | 'rust'
    amp_bf16: bool = False
    seed: int = 7
    eval_every: int = 10
    eval_steps: int = 50
    save_every: int = 10
    policy: TokenizedPolicyConfig = field(default_factory=TokenizedPolicyConfig)


@dataclass
class RolloutBatch:
    morph_name: str
    tokens: torch.Tensor          # (T, N, H, N_act, d_tok)
    glob: torch.Tensor            # (T, N, H, d_glob)
    active_mask: torch.Tensor     # (T, N, N_act)
    priv: torch.Tensor            # (T, N, d_priv)
    est_target: torch.Tensor      # (T, N, 7)
    next_glob_sens: torch.Tensor  # (T, N, 6)
    latent: torch.Tensor          # (T, N, N_act)
    old_logp: torch.Tensor        # (T, N)
    old_val: torch.Tensor         # (T, N)
    old_loc: torch.Tensor         # (T, N, N_act)
    old_std: torch.Tensor         # (T, N, N_act)
    advantages: torch.Tensor      # (T, N)
    returns: torch.Tensor         # (T, N)
    mean_reward: float
    mean_v_arc: float
    term_frac: float
    trunc_frac: float
    fault_frac: float


class FlexGraphPPOTrainer:
    """Multi-morphology Asymmetric PPO + HIM-DreamWaQ Trainer over ``FlexGraphEnv``."""

    def __init__(
        self,
        cfg: FlexGraphTrainConfig,
        out_dir: str | Path,
        *,
        resume: str | Path | None = None,
    ):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        req_dev = str(cfg.device).lower().strip()
        if req_dev == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(req_dev)
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA device requested but torch.cuda.is_available() is False')

        if self.device.type == 'cpu':
            torch.set_num_threads(1)

        torch.manual_seed(int(cfg.seed))
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(int(cfg.seed))
        self._rng = torch.Generator(device='cpu')
        self._rng.manual_seed(int(cfg.seed) + 1009)

        self.morphologies = parse_morphologies(cfg.morphologies)
        self.envs: dict[str, FlexGraphEnv] = {}
        self.eq_wrappers: dict[str, FiniteGroupEquivariantWrapper | None] = {}
        self.obs_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self.policy = TokenizedHIMDreamWaQ(cfg.policy).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            eps=1e-5,
        )

        for mname in self.morphologies:
            g, w = build_archetype(mname, elements_per_beam=cfg.elements_per_beam)
            env = FlexGraphEnv(
                g,
                world=w,
                batch=cfg.envs_per_morph,
                dt=cfg.dt,
                substeps_per_policy=cfg.substeps_per_policy,
                history_len=cfg.history_len,
                device=self.device,
                dtype=torch.float64,
                implicit_contact=cfg.implicit_contact,
                enable_self_contact=cfg.enable_self_contact,
                backend=cfg.backend,
                max_episode_steps=cfg.max_episode_steps,
                auto_reset=True,
                return_done=True,
            )
            self.envs[mname] = env
            if g.symmetry is not None:
                gs = default_glob_signs_for_group(g.symmetry, cfg.policy.d_glob)
                self.eq_wrappers[mname] = FiniteGroupEquivariantWrapper(
                    self.policy, g.symmetry, glob_signs=gs
                ).to(self.device)
            else:
                self.eq_wrappers[mname] = None

            tok, glob, mask = env.reset()
            priv = env.get_privileged_obs()
            self.obs_cache[mname] = (tok, glob, mask, priv)

        self.amp_ctx = (
            (lambda: torch.amp.autocast('cuda', dtype=torch.bfloat16))
            if (cfg.amp_bf16 and self.device.type == 'cuda')
            else nullcontext
        )

        self.current_lr: float = float(cfg.lr)
        self.start_rollout: int = 0
        self.total_samples: int = 0
        self.best_score: float = -1e9
        self.progress_history: list[dict[str, Any]] = []

        if resume is not None:
            self.load_checkpoint(resume)

    def _actor_module_for(self, morph_name: str) -> nn.Module:
        if self.cfg.use_equivariant_wrapper and self.eq_wrappers.get(morph_name) is not None:
            return self.eq_wrappers[morph_name]  # type: ignore[return-value]
        return self.policy

    def _fault_prob_for_rollout(self, rollout_idx: int) -> float:
        c = self.cfg
        if c.fault_prob_end <= 0.0 and c.fault_prob_start <= 0.0:
            return 0.0
        total_r = max(1, self.start_rollout + c.rollouts)
        frac = float(rollout_idx) / float(total_r)
        if frac < c.fault_warmup_frac:
            return float(c.fault_prob_start)
        ramp = min(1.0, (frac - c.fault_warmup_frac) / max(1e-6, 1.0 - c.fault_warmup_frac))
        return float(c.fault_prob_start + ramp * (c.fault_prob_end - c.fault_prob_start))

    def _sample_commands(self, env: FlexGraphEnv) -> torch.Tensor:
        cmd = torch.zeros((env.batch, 3), device=self.device, dtype=env.dtype)
        if env._has_fixed_base:
            # Sample 3D tip deflection targets in [-1, 1] for continuum endoscopes
            cmd[:, 0] = 0.0
            cmd[:, 1] = torch.empty((env.batch,), device=self.device, dtype=env.dtype).uniform_(-1.0, 1.0)
            cmd[:, 2] = torch.empty((env.batch,), device=self.device, dtype=env.dtype).uniform_(-1.0, 1.0)
        else:
            cmd[:, 0] = 1.0
        return cmd

    def collect_rollout(self, morph_name: str, *, rollout_idx: int = 1) -> RolloutBatch:
        """Collect one vectorized rollout horizon for ``morph_name`` with GAE computation."""
        env = self.envs[morph_name]
        actor = self._actor_module_for(morph_name)
        eq_wrap = self.eq_wrappers.get(morph_name)
        c = self.cfg
        T, B, H, N_act = c.horizon, env.batch, env.H, env.N_act

        # Optional actuator fault curriculum applied at start of rollout
        f_prob = self._fault_prob_for_rollout(rollout_idx)
        tok, glob, mask, priv = self.obs_cache[morph_name]
        if f_prob > 0.0:
            mask = env.sample_actuator_faults(f_prob, max_faults=c.fault_max_actuators, generator=self._rng)
        else:
            env.active_mask.fill_(True)
            mask = env.active_mask.clone()

        tok_buf = torch.empty((T, B, H, N_act, c.policy.d_tok), device=self.device, dtype=torch.float32)
        glob_buf = torch.empty((T, B, H, c.policy.d_glob), device=self.device, dtype=torch.float32)
        mask_buf = torch.empty((T, B, N_act), device=self.device, dtype=torch.bool)
        priv_buf = torch.empty((T, B, c.policy.d_priv), device=self.device, dtype=torch.float32)
        est_buf = torch.empty((T, B, c.policy.est_dim), device=self.device, dtype=torch.float32)
        nglob_buf = torch.empty((T, B, max(1, c.policy.d_glob - 3)), device=self.device, dtype=torch.float32)
        lat_buf = torch.empty((T, B, N_act), device=self.device, dtype=torch.float32)
        logp_buf = torch.empty((T, B), device=self.device, dtype=torch.float32)
        val_buf = torch.empty((T, B), device=self.device, dtype=torch.float32)
        nval_buf = torch.empty((T, B), device=self.device, dtype=torch.float32)
        loc_buf = torch.empty((T, B, N_act), device=self.device, dtype=torch.float32)
        std_buf = torch.empty((T, B, N_act), device=self.device, dtype=torch.float32)
        rew_buf = torch.empty((T, B), device=self.device, dtype=torch.float32)
        term_buf = torch.empty((T, B), device=self.device, dtype=torch.bool)
        trunc_buf = torch.empty((T, B), device=self.device, dtype=torch.bool)
        varc_sum = 0.0

        self.policy.train()
        with torch.no_grad():
            for t in range(T):
                # Update running normalizer (orbit-symmetrized when symmetry is active)
                if (c.use_equivariant_wrapper or c.symmetry_reg_coef > 0.0) and eq_wrap is not None:
                    eq_wrap.update_obs_norm(tok, glob, active_mask=mask, priv=priv)
                else:
                    self.policy.update_obs_norm(tok, glob, active_mask=mask, priv=priv)

                with self.amp_ctx():
                    samp = actor.sample_actions(tok, glob, active_mask=mask, priv=priv, deterministic=False)
                raw_loc = samp.raw_loc.float()

                step_out = env.step_batch(samp.action, delta_scale=c.delta_scale, auto_reset=True)
                has_done = bool(step_out.done.any().item())
                if has_done:
                    # Refresh commands and optional faults on newly reset episodes (updating glob_hist too!)
                    new_cmd = self._sample_commands(env)
                    env.cmd[step_out.done] = new_cmd[step_out.done]
                    env.glob_hist[step_out.done, :, 6:9] = new_cmd[step_out.done, None, :].to(
                        dtype=env.glob_hist.dtype
                    )
                    if f_prob > 0.0:
                        env.sample_actuator_faults(
                            f_prob, max_faults=c.fault_max_actuators, env_mask=step_out.done, generator=self._rng
                        )

                with self.amp_ctx():
                    next_v = actor.evaluate_value(
                        step_out.terminal_tokens,
                        step_out.terminal_glob,
                        active_mask=mask,
                        priv=step_out.terminal_priv,
                    ).float()

                tok_buf[t] = tok
                glob_buf[t] = glob
                mask_buf[t] = mask
                priv_buf[t] = priv
                est_buf[t] = step_out.est_target
                nglob_buf[t] = step_out.next_glob_sens
                lat_buf[t] = samp.latent.float()
                logp_buf[t] = samp.log_prob.float()
                val_buf[t] = samp.value.float()
                nval_buf[t] = next_v
                loc_buf[t] = raw_loc
                std_buf[t] = samp.forward_out.action_std.float()
                rew_buf[t] = step_out.reward.float()
                term_buf[t] = step_out.terminated
                trunc_buf[t] = step_out.truncated
                varc_sum += float(step_out.info['v_arc_m_s'].mean().item())

                if has_done:
                    tok = env.tok_hist.clone()
                    glob = env.glob_hist.clone()
                    mask = env.active_mask.clone()
                    priv = env.get_privileged_obs()
                else:
                    tok, glob, mask = step_out.obs
                    priv = step_out.priv

            adv, ret = compute_gae(rew_buf, val_buf, nval_buf, term_buf, trunc_buf, c.gamma, c.gae_lambda)
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)

        self.obs_cache[morph_name] = (tok, glob, mask, priv)
        self.total_samples += T * B

        return RolloutBatch(
            morph_name=morph_name,
            tokens=tok_buf,
            glob=glob_buf,
            active_mask=mask_buf,
            priv=priv_buf,
            est_target=est_buf,
            next_glob_sens=nglob_buf,
            latent=lat_buf,
            old_logp=logp_buf,
            old_val=val_buf,
            old_loc=loc_buf,
            old_std=std_buf,
            advantages=adv,
            returns=ret,
            mean_reward=float(rew_buf.mean().item()),
            mean_v_arc=varc_sum / max(T, 1),
            term_frac=float(term_buf.float().mean().item()),
            trunc_frac=float(trunc_buf.float().mean().item()),
            fault_frac=float((~mask_buf).float().mean().item()),
        )

    def _compute_symmetry_reg(
        self,
        morph_name: str,
        tok: torch.Tensor,
        glob: torch.Tensor,
        mask: torch.Tensor,
        priv: torch.Tensor,
        raw_loc: torch.Tensor,
        value: torch.Tensor,
        est_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Fast rollout-time symmetry consistency loss over a random non-identity g in G."""
        eq = self.eq_wrappers.get(morph_name)
        if eq is None or eq.order <= 1:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        g_idx = 1 if eq.order == 2 else int(torch.randint(1, eq.order, (), generator=self._rng).item())
        tg, gg, mg = eq.transform_inputs(tok, glob, g_idx, mask)
        raw_loc_g, _, val_g, fwd_g = self.policy.raw_actor_critic(tg, gg, active_mask=mg, priv=priv)

        loc_tgt = apply_signed_perm(
            raw_loc.detach(), eq.act_perms[g_idx], eq.act_signs[g_idx].to(dtype=raw_loc.dtype), dim=-1
        )
        loss_act = (raw_loc_g.float() - loc_tgt.float()).square().mean()
        loss_val = (val_g.float() - value.detach().float()).square().mean()
        loss_est = torch.zeros((), device=self.device, dtype=torch.float32)
        if eq.group.est_perms:
            ep = torch.tensor(eq.group.est_perms[g_idx], device=self.device, dtype=torch.long)
            es = torch.tensor(eq.group.est_signs[g_idx], device=self.device, dtype=est_pred.dtype)
            est_tgt = apply_signed_perm(est_pred.detach(), ep, es, dim=-1)
            loss_est = (fwd_g.est_pred.float() - est_tgt.float()).square().mean()
        return loss_act + 0.25 * loss_val + 0.25 * loss_est

    def ppo_update(self, batches: list[RolloutBatch], *, rollout_idx: int = 1) -> dict[str, float]:
        """Run PPO + HIM-DreamWaQ + optional symmetry regularization updates over one or more morphologies."""
        c = self.cfg
        self.policy.train()

        # Flatten per-morphology rollout buffers cleanly
        flat_items: list[dict[str, Any]] = []
        for rb in batches:
            T, B = rb.old_logp.shape
            n_tot = T * B
            flat_items.append(
                {
                    'morph': rb.morph_name,
                    'n': n_tot,
                    'tok': rb.tokens.reshape(n_tot, *rb.tokens.shape[2:]),
                    'glob': rb.glob.reshape(n_tot, *rb.glob.shape[2:]),
                    'mask': rb.active_mask.reshape(n_tot, *rb.active_mask.shape[2:]),
                    'priv': rb.priv.reshape(n_tot, *rb.priv.shape[2:]),
                    'est': rb.est_target.reshape(n_tot, *rb.est_target.shape[2:]),
                    'nglob': rb.next_glob_sens.reshape(n_tot, *rb.next_glob_sens.shape[2:]),
                    'lat': rb.latent.reshape(n_tot, *rb.latent.shape[2:]),
                    'logp': rb.old_logp.reshape(n_tot),
                    'val': rb.old_val.reshape(n_tot),
                    'loc': rb.old_loc.reshape(n_tot, *rb.old_loc.shape[2:]),
                    'std': rb.old_std.reshape(n_tot, *rb.old_std.shape[2:]),
                    'adv': rb.advantages.reshape(n_tot),
                    'ret': rb.returns.reshape(n_tot),
                }
            )

        log_std_lo = math.log(max(c.policy.min_std, 1e-3))
        log_std_hi = math.log(max(c.policy.max_std, c.policy.min_std + 1e-3))
        n_mb = max(1, int(c.minibatches))
        n_morphs = max(1, len(flat_items))

        sums = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'aux_loss': 0.0,
            'est_loss': 0.0,
            'kl_vae': 0.0,
            'infonce_loss': 0.0,
            'sym_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_frac': 0.0,
            'grad_norm': 0.0,
        }
        steps = 0
        stop_early = False

        for ep in range(int(c.epochs)):
            perms = [torch.randperm(item['n'], device=self.device) for item in flat_items]
            ep_kl_sum = 0.0
            ep_kl_n = 0

            for mb_idx in range(n_mb):
                self.optimizer.zero_grad(set_to_none=True)
                mb_kl_acc = 0.0
                mb_valid = 0

                for item, perm in zip(flat_items, perms):
                    n_tot = item['n']
                    mb_size = max(1, math.ceil(n_tot / n_mb))
                    start = mb_idx * mb_size
                    if start >= n_tot:
                        continue
                    idx = perm[start : min(n_tot, start + mb_size)]
                    if idx.numel() == 0:
                        continue

                    mname = item['morph']
                    actor = self._actor_module_for(mname)
                    tok = item['tok'][idx]
                    glob = item['glob'][idx]
                    mask = item['mask'][idx]
                    priv = item['priv'][idx]
                    est_tgt = item['est'][idx]
                    nglob = item['nglob'][idx]
                    lat = item['lat'][idx]
                    old_lp = item['logp'][idx]
                    old_v = item['val'][idx]
                    old_loc = item['loc'][idx]
                    old_std = item['std'][idx]
                    adv = item['adv'][idx]
                    ret = item['ret'][idx]

                    with self.amp_ctx():
                        ev = actor.evaluate_actions(tok, glob, lat, active_mask=mask, priv=priv)
                        aux = actor.compute_aux_losses(
                            tok, glob, next_glob_sens=nglob, est_target=est_tgt, active_mask=mask
                        )

                    new_lp = ev.log_prob.float()
                    ent = ev.entropy.float()
                    new_v = ev.value.float()
                    new_loc = ev.raw_loc.float()
                    new_std = ev.std.float()

                    # Exact PPO ratio & per-active-actuator closed-form Gaussian KL
                    log_ratio = (new_lp - old_lp).clamp(-8.0, 8.0)
                    ratio = log_ratio.exp()
                    w_act = mask.to(dtype=torch.float32)
                    n_act_eff = w_act.sum(dim=-1).clamp_min(1.0)
                    kl_per_dim = (
                        torch.log((new_std / old_std.clamp_min(1e-5)).clamp_min(1e-6))
                        + (old_std.square() + (old_loc - new_loc).square())
                        / (2.0 * new_std.square().clamp_min(1e-8))
                        - 0.5
                    )
                    kl = ((kl_per_dim * w_act).sum(dim=-1) / n_act_eff).mean()
                    clip_frac = ((ratio - 1.0).abs() > c.clip_eps).float().mean()

                    surr1 = ratio * adv
                    surr2 = ratio.clamp(1.0 - c.clip_eps, 1.0 + c.clip_eps) * adv
                    pol_loss = -torch.minimum(surr1, surr2).mean()

                    # Clipped Huber (SmoothL1) or MSE value loss
                    if c.use_huber_value_loss:
                        v_err_raw = F.huber_loss(new_v, ret, reduction='none', delta=c.huber_delta)
                        if c.value_clip:
                            v_cl = old_v + (new_v - old_v).clamp(-c.clip_eps, c.clip_eps)
                            v_err_cl = F.huber_loss(v_cl, ret, reduction='none', delta=c.huber_delta)
                            v_loss = torch.maximum(v_err_raw, v_err_cl).mean()
                        else:
                            v_loss = v_err_raw.mean()
                    else:
                        if c.value_clip:
                            v_cl = old_v + (new_v - old_v).clamp(-c.clip_eps, c.clip_eps)
                            v_loss = 0.5 * torch.maximum((new_v - ret).square(), (v_cl - ret).square()).mean()
                        else:
                            v_loss = 0.5 * (new_v - ret).square().mean()

                    ent_mean = ent.mean()
                    aux_total = aux['total_aux'].float()
                    sym_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                    if (not c.use_equivariant_wrapper) and c.symmetry_reg_coef > 0.0:
                        sym_loss = self._compute_symmetry_reg(
                            mname, tok, glob, mask, priv, new_loc, new_v, ev.forward_out.est_pred
                        )

                    loss = (
                        pol_loss
                        + c.value_coef * v_loss
                        + c.aux_coef * aux_total
                        + c.symmetry_reg_coef * sym_loss
                        - c.entropy_coef * ent_mean
                    ) / float(n_morphs)

                    if not torch.isfinite(loss):
                        continue
                    loss.backward()

                    sums['policy_loss'] += float(pol_loss.item())
                    sums['value_loss'] += float(v_loss.item())
                    sums['aux_loss'] += float(aux_total.item())
                    sums['est_loss'] += float(aux['l_est'].item())
                    sums['kl_vae'] += float(aux['l_kl'].item())
                    sums['infonce_loss'] += float(aux['l_infonce'].item())
                    sums['sym_loss'] += float(sym_loss.item())
                    sums['entropy'] += float(ent_mean.item())
                    sums['approx_kl'] += float(kl.item())
                    sums['clip_frac'] += float(clip_frac.item())
                    mb_kl_acc += float(kl.item())
                    mb_valid += 1

                if mb_valid == 0:
                    continue

                gnorm = nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
                if torch.isfinite(gnorm):
                    self.optimizer.step()
                    self.policy.log_std.data.clamp_(log_std_lo, log_std_hi)
                    sums['grad_norm'] += float(gnorm.item())
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                steps += mb_valid
                mb_kl = mb_kl_acc / mb_valid
                ep_kl_sum += mb_kl
                ep_kl_n += 1
                if c.target_kl > 0.0 and ep > 0 and (ep_kl_sum / max(1, ep_kl_n)) > 3.0 * c.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        denom = float(max(1, steps))
        metrics = {k: v / denom for k, v in sums.items()}
        metrics['grad_norm'] = sums['grad_norm'] / float(max(1, steps // n_morphs)) if steps >= n_morphs else sums['grad_norm']

        # rsl_rl-style adaptive KL LR scheduling
        if c.target_kl > 0.0 and steps > 0:
            mean_kl = metrics['approx_kl']
            if mean_kl > 2.0 * c.target_kl:
                self.current_lr = max(c.min_lr, self.current_lr / 1.5)
            elif mean_kl < 0.5 * c.target_kl:
                self.current_lr = min(c.max_lr, self.current_lr * 1.2)
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.current_lr

        metrics['lr'] = self.current_lr
        metrics['action_std'] = float(torch.exp(self.policy.log_std).clamp(c.policy.min_std, c.policy.max_std).item())
        return metrics

    @torch.no_grad()
    def evaluate(self, eval_steps: int | None = None) -> dict[str, Any]:
        """Deterministic evaluation across all configured robot morphologies."""
        n_steps = int(eval_steps if eval_steps is not None else self.cfg.eval_steps)
        was_training = self.policy.training
        self.policy.eval()
        per_morph: dict[str, dict[str, float]] = {}

        for mname in self.morphologies:
            g, w = build_archetype(mname, elements_per_beam=self.cfg.elements_per_beam)
            eval_env = FlexGraphEnv(
                g,
                world=w,
                batch=1,
                dt=self.cfg.dt,
                substeps_per_policy=self.cfg.substeps_per_policy,
                history_len=self.cfg.history_len,
                device=self.device,
                dtype=torch.float64,
                implicit_contact=self.cfg.implicit_contact,
                enable_self_contact=self.cfg.enable_self_contact,
                backend=self.cfg.backend,
                max_episode_steps=max(n_steps + 10, self.cfg.max_episode_steps),
                auto_reset=False,
                return_done=True,
            )
            actor = self._actor_module_for(mname)
            tok, glob, mask = eval_env.reset()
            priv = eval_env.get_privileged_obs()
            rew_sum = 0.0
            varc_sum = 0.0
            est_err_sum = 0.0
            executed = 0
            failed_steps = 0
            arc_start = float(eval_env._prev_arc_s[0].item())
            arc_end = arc_start

            for _ in range(n_steps):
                samp = actor.sample_actions(tok, glob, active_mask=mask, priv=priv, deterministic=True)
                step_out = eval_env.step_batch(samp.action, delta_scale=self.cfg.delta_scale, auto_reset=False)
                executed += 1
                rew_sum += float(step_out.reward.mean().item())
                varc_sum += float(step_out.info['v_arc_m_s'].mean().item())
                est_err_sum += float((samp.forward_out.est_pred.float() - step_out.est_target.float()).abs().mean().item())
                arc_end = float(step_out.info['arc_s_m'][0].item())
                if step_out.info['failed'] or bool(step_out.terminated.any().item()):
                    failed_steps += 1
                    break
                tok, glob, mask = step_out.obs
                priv = step_out.priv

            denom_exec = max(1, executed)
            per_morph[mname] = {
                'mean_reward': round(rew_sum / denom_exec, 5),
                'mean_v_arc_m_s': round(varc_sum / denom_exec, 5),
                'total_arc_m': round(arc_end - arc_start, 5),
                'est_mae': round(est_err_sum / denom_exec, 5),
                'failed_frac': round(float(failed_steps > 0), 4),
            }
            if hasattr(eval_env.solver, 'close'):
                eval_env.solver.close()

        if was_training:
            self.policy.train()

        agg_rew = sum(v['mean_reward'] for v in per_morph.values()) / max(1, len(per_morph))
        agg_varc = sum(v['mean_v_arc_m_s'] for v in per_morph.values()) / max(1, len(per_morph))
        return {
            'aggregate_mean_reward': round(agg_rew, 5),
            'aggregate_mean_v_arc_m_s': round(agg_varc, 5),
            'per_morphology': per_morph,
        }

    def save_checkpoint(self, path: str | Path, rollout: int, metrics: dict[str, Any]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema': 'flexgraph_ppo_v1',
            'model': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'policy_cfg': asdict(self.cfg.policy),
            'train_cfg': asdict(self.cfg),
            'morphologies': list(self.morphologies),
            'rollout': int(rollout),
            'total_samples': int(self.total_samples),
            'current_lr': float(self.current_lr),
            'best_score': float(self.best_score),
            'metrics': metrics,
        }
        tmp = target.with_suffix(target.suffix + '.tmp')
        torch.save(payload, tmp)
        tmp.replace(target)
        return target

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_rollout = int(ckpt.get('rollout', 0))
        self.total_samples = int(ckpt.get('total_samples', 0))
        self.current_lr = float(ckpt.get('current_lr', self.cfg.lr))
        self.best_score = float(ckpt.get('best_score', -1e9))
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.current_lr
        return ckpt

    def train(self) -> dict[str, Any]:
        """Run the full multi-morphology PPO training loop and return final summary."""
        c = self.cfg
        backends = {m: env.active_backend for m, env in self.envs.items()}
        _write_json_atomic(
            self.out_dir / 'config.json',
            {
                'schema': 'flexgraph_ppo_v1',
                'morphologies': list(self.morphologies),
                'device': str(self.device),
                'gpu': torch.cuda.get_device_name() if self.device.type == 'cuda' else 'cpu',
                'active_backends': backends,
                'ffi': load_flexgraph_ffi().summary_dict(),
                'train_cfg': asdict(c),
            },
        )

        t_start = time.monotonic()
        last_eval: dict[str, Any] | None = None

        for step_i in range(1, int(c.rollouts) + 1):
            rollout_idx = self.start_rollout + step_i
            t0 = time.monotonic()

            if c.schedule == 'mixture':
                selected = list(self.morphologies)
            else:
                selected = [self.morphologies[(rollout_idx - 1) % len(self.morphologies)]]

            batches = [self.collect_rollout(m, rollout_idx=rollout_idx) for m in selected]
            t_sim = time.monotonic() - t0

            t_opt0 = time.monotonic()
            opt_metrics = self.ppo_update(batches, rollout_idx=rollout_idx)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t_opt = time.monotonic() - t_opt0

            mean_rew = sum(b.mean_reward for b in batches) / max(1, len(batches))
            mean_varc = sum(b.mean_v_arc for b in batches) / max(1, len(batches))
            step_samples = sum(b.old_logp.numel() for b in batches)
            sps = step_samples / max(t_sim + t_opt, 1e-6)

            rec: dict[str, Any] = {
                'rollout': rollout_idx,
                'samples': self.total_samples,
                'morphologies': selected,
                'mean_reward': round(mean_rew, 5),
                'mean_v_arc_m_s': round(mean_varc, 5),
                'term_frac': round(sum(b.term_frac for b in batches) / len(batches), 4),
                'fault_frac': round(sum(b.fault_frac for b in batches) / len(batches), 4),
                'policy_loss': round(opt_metrics['policy_loss'], 5),
                'value_loss': round(opt_metrics['value_loss'], 5),
                'aux_loss': round(opt_metrics['aux_loss'], 5),
                'est_loss': round(opt_metrics['est_loss'], 5),
                'kl_vae': round(opt_metrics['kl_vae'], 5),
                'sym_loss': round(opt_metrics['sym_loss'], 5),
                'approx_kl': round(opt_metrics['approx_kl'], 6),
                'clip_frac': round(opt_metrics['clip_frac'], 4),
                'entropy': round(opt_metrics['entropy'], 5),
                'grad_norm': round(opt_metrics['grad_norm'], 4),
                'action_std': round(opt_metrics['action_std'], 4),
                'lr': round(opt_metrics['lr'], 7),
                'sim_s': round(t_sim, 3),
                'opt_s': round(t_opt, 3),
                'sps': round(sps, 1),
                'elapsed_s': round(time.monotonic() - t_start, 2),
                'per_morphology': {
                    b.morph_name: {
                        'mean_reward': round(b.mean_reward, 5),
                        'mean_v_arc_m_s': round(b.mean_v_arc, 5),
                        'term_frac': round(b.term_frac, 4),
                    }
                    for b in batches
                },
            }

            if c.eval_every > 0 and (step_i % c.eval_every == 0 or step_i == c.rollouts):
                last_eval = self.evaluate()
                rec['eval'] = last_eval
                score = float(last_eval['aggregate_mean_reward'])
                _write_json_atomic(self.out_dir / 'eval_summary.json', last_eval)
            else:
                score = mean_rew

            self.progress_history.append(rec)
            _write_json_atomic(
                self.out_dir / 'progress.json',
                {
                    'schema': 'flexgraph_ppo_v1',
                    'latest': rec,
                    'best_score': round(max(self.best_score, score), 5),
                    'history': self.progress_history,
                },
            )

            if score >= self.best_score:
                self.best_score = score
                self.save_checkpoint(self.out_dir / 'best.pt', rollout_idx, rec)
            if c.save_every > 0 and (step_i % c.save_every == 0 or step_i == c.rollouts):
                self.save_checkpoint(self.out_dir / 'latest.pt', rollout_idx, rec)

            print(
                f"[flexgraph r={rollout_idx:04d} morph={','.join(selected)}] "
                f"rew={mean_rew:+.4f} v_arc={mean_varc:+.4f} "
                f"vloss={opt_metrics['value_loss']:.4f} aux={opt_metrics['aux_loss']:.4f} "
                f"kl={opt_metrics['approx_kl']:.5f} std={opt_metrics['action_std']:.3f} "
                f"lr={opt_metrics['lr']:.2e} sps={sps:.1f}",
                flush=True,
            )

        self.save_checkpoint(self.out_dir / 'latest.pt', self.start_rollout + c.rollouts, self.progress_history[-1])
        return {
            'out_dir': str(self.out_dir),
            'rollouts_completed': int(c.rollouts),
            'total_samples': int(self.total_samples),
            'best_score': float(self.best_score),
            'latest': self.progress_history[-1] if self.progress_history else None,
            'eval': last_eval,
        }

    def close(self) -> None:
        for env in self.envs.values():
            if hasattr(env.solver, 'close'):
                env.solver.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='FlexGraph Multi-Morphology Asymmetric PPO + HIM-DreamWaQ Trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--morphologies',
        type=str,
        default='serpworm_N3,bristle_worm,flexspine_crawler',
        help=f'Comma-separated archetypes from {list(BUILTIN_ARCHETYPES)} or "all".',
    )
    p.add_argument('--schedule', choices=('round_robin', 'mixture'), default='round_robin')
    p.add_argument('--out-dir', type=str, default='runs/flexgraph_multi_v1')
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint (.pt) to resume from.')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--backend', choices=('auto', 'python', 'rust'), default='auto')
    p.add_argument('--envs', type=int, default=64, help='Batch environments per morphology.')
    p.add_argument('--horizon', type=int, default=24, help='Policy steps per rollout.')
    p.add_argument('--rollouts', type=int, default=100, help='Total PPO rollout iterations.')
    p.add_argument('--epochs', type=int, default=4, help='PPO epochs per rollout.')
    p.add_argument('--minibatches', type=int, default=4, help='PPO minibatches per epoch.')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min-lr', type=float, default=5e-5)
    p.add_argument('--max-lr', type=float, default=1e-3)
    p.add_argument('--target-kl', type=float, default=0.012)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--entropy-coef', type=float, default=0.005)
    p.add_argument('--aux-coef', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-attn-layers', type=int, default=2)
    p.add_argument('--history-len', type=int, default=16)
    p.add_argument('--dt', type=float, default=0.001)
    p.add_argument('--substeps', type=int, default=10, help='Physics substeps per policy step.')
    p.add_argument('--max-episode-steps', type=int, default=200)
    p.add_argument('--elements-per-beam', type=int, default=None, help='Override beam element count for speed/fidelity.')
    p.add_argument('--equivariant', action='store_true', help='Enable exact FiniteGroupEquivariantWrapper.')
    p.add_argument('--symmetry-reg', type=float, default=0.15, help='Group symmetry regularization weight (0 disables).')
    p.add_argument('--fault-prob-start', type=float, default=0.0)
    p.add_argument('--fault-prob-end', type=float, default=0.0, help='End probability of actuator fault curriculum.')
    p.add_argument('--fault-max-actuators', type=int, default=1)
    p.add_argument('--amp-bf16', action='store_true', help='Enable CUDA BF16 autocast for policy forward/backward.')
    p.add_argument('--no-obs-norm', action='store_true', help='Disable TokenRunningNorm.')
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--eval-steps', type=int, default=50)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument(
        '--bench-backend',
        action='store_true',
        help='Run backend FFI parity/benchmark diagnostic on the first morphology and exit.',
    )
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    morphs = parse_morphologies(args.morphologies)

    if args.bench_backend:
        g, w = build_archetype(morphs[0], elements_per_beam=args.elements_per_beam or 6)
        rep = benchmark_backend_parity(g, world=w, steps=15, dt=args.dt, batch=1)
        print(json.dumps(rep, indent=2), flush=True)
        return rep

    policy_cfg = TokenizedPolicyConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_attn_layers=args.n_attn_layers,
        use_obs_norm=not args.no_obs_norm,
    )
    train_cfg = FlexGraphTrainConfig(
        morphologies=morphs,
        schedule=args.schedule,
        envs_per_morph=args.envs,
        horizon=args.horizon,
        rollouts=args.rollouts,
        epochs=args.epochs,
        minibatches=args.minibatches,
        lr=args.lr,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        target_kl=args.target_kl,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        aux_coef=args.aux_coef,
        max_grad_norm=args.max_grad_norm,
        use_equivariant_wrapper=bool(args.equivariant),
        symmetry_reg_coef=float(args.symmetry_reg),
        fault_prob_start=float(args.fault_prob_start),
        fault_prob_end=float(args.fault_prob_end),
        fault_max_actuators=int(args.fault_max_actuators),
        dt=float(args.dt),
        substeps_per_policy=int(args.substeps),
        history_len=int(args.history_len),
        max_episode_steps=int(args.max_episode_steps),
        elements_per_beam=args.elements_per_beam,
        device=args.device,
        backend=args.backend,
        amp_bf16=bool(args.amp_bf16),
        seed=int(args.seed),
        eval_every=int(args.eval_every),
        eval_steps=int(args.eval_steps),
        save_every=int(args.save_every),
        policy=policy_cfg,
    )
    trainer = FlexGraphPPOTrainer(train_cfg, out_dir=args.out_dir, resume=args.resume)
    try:
        summary = trainer.train()
    finally:
        trainer.close()
    return summary


if __name__ == '__main__':
    main()

"""v8 Paper-Inspired HIM-DreamWaQ Actor-Critic & Exact Z2 Group-Equivariant Inference Wrapper.

Synthesizes four Sim-to-Real proprioceptive locomotion frameworks cleanly onto SerpWorm-RL
without modifying existing v5/v6/v7 modules (reuses ``_CausalConv1d``, ``RunningNorm``, and
``_tanh_squashed_log_prob`` directly from ``serpworm.rl.models``):

1. **DreamWaQ CENet beta-VAE (Nahrendra et al., ICRA 2023, arXiv:2301.10602)**:
   - Explicit 7D physical state estimator ``[vx, vy, wz, wall_conf, wall_side, terrain_drag, slip]``.
   - Variational Gaussian terrain/contact context ``(z_mu, z_logvar) -> z_t`` (``latent_dim=16``)
     with residual next-step proprioceptive sensor decoder (excluding exogenous ``cmd`` channels).
   - Uncertainty-gated latent coupling ``z_gated = sigmoid(W_g [sg(z_mu), sg(z_logvar)]) * sg(z)``
     plus AdaBoot bootstrap probability helper ``adaboot_prob`` (Eq. 8).

2. **HIMLoco Hybrid Internal Model Contrastive Dynamics (Long et al., ICLR 2024, arXiv:2312.11460)**:
   - Predicts L2-normalized next-step dynamics embedding from ``(cenet_feat, z_t, a_t)`` and aligns
     it against a next-frame proprioceptive target encoder via symmetric InfoNCE (with optional
     momentum EMA target) + optional 3-step Sinkhorn-Knopp SwAV prototype clustering.

3. **RMA Latent Extrinsics Insight (Kumar et al., RSS 2021, arXiv:2107.04034)**:
   - Complements explicit 7D physical estimation with implicit 16D latent context to disambiguate
     covarying unobservable causes (rolling drag, slope, Capstan friction, ratchet backlash, payload).

4. **Exact Z2 Left-Right Group-Equivariant Inference Wrapper (Mittal et al., ICRA 2024, arXiv:2403.04359)**:
   - Wraps ANY trained policy (``ProprioceptiveEstimatorActorCritic``, ``TemporalActorCritic``,
     or ``HIMDreamWaQActorCritic``) across 51D, 65D, 54D (14D unified), and 49D (9D unified)
     observation layouts to enforce exact ``pi_eq(L_g o) == K_g pi_eq(o)`` and exact mirror-averaged
     EstNet estimates ``e_eq(L_g o) == M_g e_eq(o)`` by construction.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from serpworm.rl.models import (
    ProprioceptiveEstimatorActorCritic,
    RunningNorm,
    TemporalActorCritic,
    _CausalConv1d,
    _tanh_squashed_log_prob,
)

__all__ = [
    'ContextOutput',
    'HIMDreamWaQConfig',
    'HIMDreamWaQActorCritic',
    'GroupEquivariantPolicyWrapper',
    'adaboot_prob',
    'build_obs_mirror_spec',
    'build_action_mirror_spec',
    'build_est_mirror_spec',
    'mirror_tensor_by_spec',
    'sinkhorn_knopp',
]

_PAIR_SWAP_10 = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8)


class ContextOutput(NamedTuple):
    """Structured output of the shared CENet / beta-VAE context encoder."""

    est_pred: torch.Tensor
    z: torch.Tensor
    z_mu: torch.Tensor
    z_logvar: torch.Tensor
    z_gated: torch.Tensor
    z_gate: torch.Tensor
    cenet_feat: torch.Tensor
    ctx_enc: torch.Tensor


@dataclass(frozen=True)
class HIMDreamWaQConfig:
    """Hyperparameters for v8 HIM-DreamWaQ architecture and auxiliary losses."""

    latent_dim: int = 16
    contrast_dim: int = 32
    num_prototypes: int = 16
    beta_kl: float = 0.002
    est_coef: float = 1.0
    rec_coef: float = 0.5
    infonce_coef: float = 0.25
    swav_coef: float = 0.05
    temperature: float = 0.10
    sinkhorn_iters: int = 3
    sinkhorn_eps: float = 0.05
    ema_momentum: float = 0.99
    use_ema_target: bool = True
    logvar_min: float = -6.0
    logvar_max: float = 2.0
    mean_bound: float = 4.0


def adaboot_prob(episode_rewards: torch.Tensor | list[float], eps: float = 1e-6) -> float:
    """DreamWaQ (2301.10602 Eq. 8) adaptive bootstrapping probability: ``1 - tanh(CV(R))``.

    Returns a probability in ``[0.0, 1.0]``: low when reward variance is high (rely on true/safe
    bootstrapping) and approaching 1.0 once policy returns stabilize.
    """
    if isinstance(episode_rewards, torch.Tensor):
        r = episode_rewards.detach().float().reshape(-1)
    else:
        r = torch.as_tensor(episode_rewards, dtype=torch.float32).reshape(-1)
    if r.numel() < 2:
        return 0.0
    mean_r = float(r.mean().item())
    std_r = float(r.std(unbiased=False).item())
    if not (math.isfinite(mean_r) and math.isfinite(std_r)):
        return 0.0
    cv = std_r / max(abs(mean_r), float(eps))
    return float(max(0.0, min(1.0, 1.0 - math.tanh(cv))))


@torch.no_grad()
def sinkhorn_knopp(scores: torch.Tensor, eps: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    """Numerically stable FP32 Sinkhorn-Knopp optimal transport for SwAV prototype assignments.

    Args:
        scores: Cosine similarity matrix ``[B, K]`` between embeddings and unit prototypes.
        eps: Entropic regularization temperature.
        n_iters: Number of Sinkhorn normalization iterations (default 3 per HIMLoco / SwAV).

    Returns:
        Detached soft assignment matrix ``Q`` of shape ``[B, K]`` with rows summing to 1.
    """
    logits = scores.detach().float()
    b_size, k_size = logits.shape
    if b_size == 0 or k_size == 0:
        return torch.empty_like(logits)
    shifted = (logits - logits.max(dim=-1, keepdim=True).values) / max(float(eps), 1e-4)
    q = torch.exp(shifted.clamp(-30.0, 0.0)).t()  # [K, B]
    q = q / q.sum().clamp_min(1e-8)
    for _ in range(max(1, int(n_iters))):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
        q = q / float(k_size)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
        q = q / float(b_size)
    q = q * float(b_size)  # each column sums to 1
    return q.t().contiguous()  # [B, K]


class _RightAlignedCausalConv1d(_CausalConv1d):
    """Right-aligned causal strided 1D conv extending ``_CausalConv1d`` so the newest frame ``t=H-1``
    always lands on the final output slot (zero future leakage, zero dropped recent frames for even ``H``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride = self.stride[0]
        k = self.kernel_size[0]
        # Align stride grid to x[..., -1]: for stride=2 & even length, left_pad = k - 2 preserves
        # exact output length ceil(L / 2) while ensuring the rightmost window ends at t = L - 1.
        left_pad = self._left_pad - 1 if (stride == 2 and k >= 2 and (x.shape[-1] % 2 == 0)) else self._left_pad
        return F.conv1d(F.pad(x, (left_pad, 0)), self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class _MultiScaleCausalTCN(nn.Module):
    """Dual-scale right-aligned causal 1D-TCN capturing both fast wall/slip impacts and gait cycles.

    Both branches use ``_RightAlignedCausalConv1d`` with stride 2 twice, so for every ``H >= 1`` both
    yield exact output length ``T_out = ceil(ceil(H / 2) / 2)`` ending strictly on the newest frame ``t=H-1``:
      - Fast branch (``k=3, s=2`` x2): receptive field 7 frames (~0.7 s at 10 Hz) -> slip/wall impacts.
      - Slow branch (``k=5, s=2`` x2): receptive field 13 frames (~1.3 s at 10 Hz) -> full gait cycle.
    """

    def __init__(
        self,
        in_channels: int,
        fast_c1: int = 48,
        fast_c2: int = 32,
        slow_c1: int = 32,
        slow_c2: int = 16,
    ):
        super().__init__()
        self.out_channels = int(fast_c2) + int(slow_c2)
        self.fast_branch = nn.Sequential(
            _RightAlignedCausalConv1d(in_channels, fast_c1, kernel_size=3, stride=2),
            nn.SiLU(),
            _RightAlignedCausalConv1d(fast_c1, fast_c2, kernel_size=3, stride=2),
            nn.SiLU(),
        )
        self.slow_branch = nn.Sequential(
            _RightAlignedCausalConv1d(in_channels, slow_c1, kernel_size=5, stride=2),
            nn.SiLU(),
            _RightAlignedCausalConv1d(slow_c1, slow_c2, kernel_size=5, stride=2),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.fast_branch(x), self.slow_branch(x)), dim=1)


class HIMDreamWaQActorCritic(nn.Module):
    """v8 Hybrid Internal Model + DreamWaQ CENet beta-VAE Asymmetric Actor-Critic.

    Drop-in compatible with v6/v7 PPO rollouts and evaluators (``sample``, ``action``,
    ``action_with_estimate``, ``value``, ``evaluate_full``), while adding:
      - ``encode_context`` / ``action_with_context`` exposing explicit 7D estimates and 16D latent context;
      - ``compute_aux_losses`` combining supervised EstNet SmoothL1, DreamWaQ beta-VAE residual
        next-step proprioceptive reconstruction + KL, and HIMLoco symmetric InfoNCE + SwAV dynamics loss;
      - ``update_ema_target`` for smooth target dynamics projection updates.
    """

    def __init__(
        self,
        frame_size: int,
        history_len: int,
        privileged_size: int,
        est_size: int = 7,
        action_size: int = 14,
        initial_std: float = 0.45,
        latent_dim: int | None = None,
        contrast_dim: int | None = None,
        num_prototypes: int | None = None,
        sensor_dim: int | None = None,
        mean_bound: float | None = None,
        use_ema_target: bool | None = None,
        ema_momentum: float | None = None,
        config: HIMDreamWaQConfig | None = None,
    ):
        super().__init__()
        base_cfg = config or HIMDreamWaQConfig()
        self.cfg = HIMDreamWaQConfig(
            latent_dim=int(base_cfg.latent_dim if latent_dim is None else latent_dim),
            contrast_dim=int(base_cfg.contrast_dim if contrast_dim is None else contrast_dim),
            num_prototypes=int(base_cfg.num_prototypes if num_prototypes is None else num_prototypes),
            beta_kl=float(base_cfg.beta_kl),
            est_coef=float(base_cfg.est_coef),
            rec_coef=float(base_cfg.rec_coef),
            infonce_coef=float(base_cfg.infonce_coef),
            swav_coef=float(base_cfg.swav_coef),
            temperature=float(base_cfg.temperature),
            sinkhorn_iters=int(base_cfg.sinkhorn_iters),
            sinkhorn_eps=float(base_cfg.sinkhorn_eps),
            ema_momentum=float(base_cfg.ema_momentum if ema_momentum is None else ema_momentum),
            use_ema_target=bool(base_cfg.use_ema_target if use_ema_target is None else use_ema_target),
            logvar_min=float(base_cfg.logvar_min),
            logvar_max=float(base_cfg.logvar_max),
            mean_bound=float(base_cfg.mean_bound if mean_bound is None else mean_bound),
        )

        self.frame_size = int(frame_size)
        self.history_len = int(history_len)
        self.privileged_size = int(privileged_size)
        self.est_size = int(est_size)
        self.action_size = int(action_size)
        self.latent_dim = self.cfg.latent_dim
        self.contrast_dim = self.cfg.contrast_dim
        self.num_prototypes = self.cfg.num_prototypes
        self.mean_bound = self.cfg.mean_bound
        # Predict pure proprioceptive sensor channels (excluding trailing 3D exogenous command [vx, wz, stop])
        self.sensor_dim = int(sensor_dim) if sensor_dim is not None else (
            self.frame_size - 3 if self.frame_size > 3 else self.frame_size
        )
        if not (1 <= self.sensor_dim <= self.frame_size):
            raise ValueError(f'sensor_dim ({self.sensor_dim}) must be within [1, {self.frame_size}]')

        self.obs_norm = RunningNorm(self.frame_size)
        self.priv_norm = RunningNorm(self.privileged_size)

        t_out = math.ceil(math.ceil(self.history_len / 2) / 2)
        self.t_out = t_out

        # 1. Shared Multi-Scale Causal 1D-TCN for CENet + HIM + Actor (48 * t_out + F)
        #    Detached before Actor MLP so PPO gradients never corrupt self-supervised CENet/HIM features,
        #    matching DreamWaQ/HIMLoco single-encoder deployment efficiency (< 1.5 ms CPU 1-thread).
        self.context_tcn = _MultiScaleCausalTCN(self.frame_size, fast_c1=48, fast_c2=32, slow_c1=24, slow_c2=16)
        ctx_in = self.context_tcn.out_channels * t_out + self.frame_size

        # 2. Asymmetric Privileged Critic TCN (training only, matches v6 capacity)
        self.critic_tcn = nn.Sequential(
            _RightAlignedCausalConv1d(self.frame_size, 64, kernel_size=3, stride=2),
            nn.SiLU(),
            _RightAlignedCausalConv1d(64, 48, kernel_size=3, stride=2),
            nn.SiLU(),
        )
        critic_tcn_flat = 48 * t_out

        # 3. DreamWaQ CENet beta-VAE Encoder (deployed on real robot)
        self.cenet_trunk = nn.Sequential(
            nn.Linear(ctx_in, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.est_head = nn.Linear(128, self.est_size)
        self.z_mu_head = nn.Linear(128, self.latent_dim)
        self.z_logvar_head = nn.Linear(128, self.latent_dim)
        # Uncertainty gate trained end-to-end by PPO on detached (z_mu, z_logvar)
        self.z_gate_head = nn.Linear(2 * self.latent_dim, self.latent_dim)

        # 4. Mode-Free Actor MLP (consumes sg(multi-scale ctx_enc) + sg(est_pred) + uncertainty-gated sg(z))
        actor_in = ctx_in + self.est_size + self.latent_dim
        self.actor_mlp = nn.Sequential(
            nn.Linear(actor_in, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, self.action_size),
        )

        # 5. Asymmetric Privileged Critic MLP (training only)
        critic_in = critic_tcn_flat + self.frame_size + self.privileged_size
        self.critic_mlp = nn.Sequential(
            nn.Linear(critic_in, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # 6. Training-Only Information-Bottleneck Auxiliary Heads (zero shortcut bypass around z_t!):
        #    - DreamWaQ next-step sensor decoder: p(o_{t+1}^{sens} | est_pred, z_t, a_t) (2301.10602 Fig. 2)
        #    - HIMLoco contrastive dynamics predictor: f_src(z_t, a_t) <-> f_tgt(o_{t+1}^{sens}) (2312.11460 Sec. 3.3)
        dec_in = self.est_size + self.latent_dim + self.action_size
        self.obs_decoder = nn.Sequential(
            nn.Linear(dec_in, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, self.sensor_dim),
        )

        him_src_in = self.latent_dim + self.action_size
        self.him_src_proj = nn.Sequential(
            nn.Linear(him_src_in, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, self.contrast_dim),
        )
        self.him_tgt_proj = nn.Sequential(
            nn.Linear(self.sensor_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, self.contrast_dim),
        )
        self.him_prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.contrast_dim))
        self.logstd = nn.Parameter(torch.full((self.action_size,), math.log(max(float(initial_std), 0.05))))

        self._init_weights()

        # Momentum EMA shadow of him_tgt_proj (frozen parameters: not optimized by Adam)
        self.him_tgt_ema = copy.deepcopy(self.him_tgt_proj)
        for p in self.him_tgt_ema.parameters():
            p.requires_grad_(False)

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(mod.weight, math.sqrt(2.0))
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.orthogonal_(self.est_head.weight, 0.25)
        nn.init.orthogonal_(self.z_mu_head.weight, 0.25)
        nn.init.orthogonal_(self.z_logvar_head.weight, 0.10)
        nn.init.constant_(self.z_logvar_head.bias, -1.0)
        nn.init.orthogonal_(self.z_gate_head.weight, 0.20)
        nn.init.constant_(self.z_gate_head.bias, 0.5)
        nn.init.orthogonal_(self.actor_mlp[-1].weight, 0.05)
        nn.init.orthogonal_(self.critic_mlp[-1].weight, 1.0)
        nn.init.orthogonal_(self.obs_decoder[-1].weight, 0.20)
        nn.init.orthogonal_(self.him_src_proj[-1].weight, 0.50)
        nn.init.orthogonal_(self.him_tgt_proj[-1].weight, 0.50)
        nn.init.orthogonal_(self.him_prototypes)
        with torch.no_grad():
            self.him_prototypes.copy_(F.normalize(self.him_prototypes, dim=-1))

    @torch.no_grad()
    def update_ema_target(self, momentum: float | None = None) -> None:
        """Polyak-update ``him_tgt_ema`` toward ``him_tgt_proj`` after an optimizer step."""
        m = self.cfg.ema_momentum if momentum is None else float(momentum)
        for p_ema, p_online in zip(self.him_tgt_ema.parameters(), self.him_tgt_proj.parameters()):
            p_ema.data.mul_(m).add_(p_online.data, alpha=1.0 - m)

    def parameter_summary(self) -> dict[str, int]:
        """Return parameter counts split into deployable actor+CENet vs training-only modules."""
        deploy_mods = (
            self.context_tcn,
            self.cenet_trunk,
            self.est_head,
            self.z_mu_head,
            self.z_logvar_head,
            self.z_gate_head,
            self.actor_mlp,
        )
        deploy_params = sum(p.numel() for m in deploy_mods for p in m.parameters())
        critic_params = sum(p.numel() for m in (self.critic_tcn, self.critic_mlp) for p in m.parameters())
        aux_params = sum(
            p.numel() for m in (self.obs_decoder, self.him_src_proj, self.him_tgt_proj) for p in m.parameters()
        ) + self.him_prototypes.numel()
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'deploy_params': int(deploy_params),
            'critic_params': int(critic_params),
            'aux_params': int(aux_params),
            'total_trainable_params': int(total_trainable),
        }

    @staticmethod
    def _encode_seq(seq_norm: torch.Tensor, tcn: nn.Module) -> torch.Tensor:
        feat = tcn(seq_norm.transpose(1, 2))
        return torch.cat((feat.flatten(1), seq_norm[:, -1, :]), dim=-1)

    def _activate_est(self, raw: torch.Tensor) -> torch.Tensor:
        if self.est_size == 7:
            vel = raw[:, :3]
            w_conf = torch.sigmoid(raw[:, 3:4])
            w_side = torch.tanh(raw[:, 4:5])
            drag_slip = torch.sigmoid(raw[:, 5:7])
            return torch.cat((vel, w_conf, w_side, drag_slip), dim=-1)
        return raw

    def _encode_context_from_norm(
        self,
        seq_n: torch.Tensor,
        *,
        deterministic: bool | None = None,
    ) -> ContextOutput:
        ctx_enc = self._encode_seq(seq_n, self.context_tcn)
        cenet_feat = self.cenet_trunk(ctx_enc)
        est_pred = self._activate_est(self.est_head(cenet_feat))
        z_mu = self.z_mu_head(cenet_feat)
        z_logvar = self.z_logvar_head(cenet_feat).clamp(self.cfg.logvar_min, self.cfg.logvar_max)

        use_det = (not self.training) if deterministic is None else bool(deterministic)
        if use_det:
            z = z_mu
        else:
            # Reparameterize in FP32 to avoid bf16 variance underflow, then cast back
            std_f32 = torch.exp(0.5 * z_logvar.float())
            eps = torch.randn_like(std_f32)
            z = (z_mu.float() + std_f32 * eps).to(dtype=z_mu.dtype)

        # Uncertainty gate: PPO trains z_gate_head on detached posterior stats without perturbing CENet
        gate_in = torch.cat((z_mu.detach(), z_logvar.detach()), dim=-1)
        z_gate = torch.sigmoid(self.z_gate_head(gate_in))
        z_gated = z_gate * z.detach()
        return ContextOutput(
            est_pred=est_pred,
            z=z,
            z_mu=z_mu,
            z_logvar=z_logvar,
            z_gated=z_gated,
            z_gate=z_gate,
            cenet_feat=cenet_feat,
            ctx_enc=ctx_enc,
        )

    def encode_context(self, obs_seq: torch.Tensor, *, deterministic: bool | None = None) -> ContextOutput:
        """Pure-proprioceptive forward pass returning explicit 7D estimates & 16D variational context."""
        seq_n = self.obs_norm(obs_seq)
        return self._encode_context_from_norm(seq_n, deterministic=deterministic)

    def estimate(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Pure-proprioceptive 7D physical state estimate ``[vx, vy, wz, wall_conf, wall_side, drag, slip]``."""
        return self.encode_context(obs_seq, deterministic=True).est_pred

    def _actor_mean_and_context(
        self,
        seq_n: torch.Tensor,
        *,
        deterministic_z: bool | None = None,
    ) -> tuple[torch.Tensor, ContextOutput]:
        ctx = self._encode_context_from_norm(seq_n, deterministic=deterministic_z)
        actor_in = torch.cat((ctx.ctx_enc.detach(), ctx.est_pred.detach(), ctx.z_gated), dim=-1)
        raw_mean = self.actor_mlp(actor_in)
        mean = self.mean_bound * torch.tanh(raw_mean / self.mean_bound)
        return mean, ctx

    def _actor_dist_and_context(
        self,
        seq_n: torch.Tensor,
        *,
        deterministic_z: bool | None = None,
    ) -> tuple[torch.distributions.Normal, ContextOutput]:
        mean, ctx = self._actor_mean_and_context(seq_n, deterministic_z=deterministic_z)
        std = self.logstd.clamp(-2.8, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean.float(), std.float(), validate_args=False), ctx

    @staticmethod
    def log_probability(dist: torch.distributions.Normal, latent: torch.Tensor) -> torch.Tensor:
        return _tanh_squashed_log_prob(dist, latent)

    def value(self, obs_seq: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        seq_n = self.obs_norm(obs_seq)
        priv_n = self.priv_norm(priv)
        enc = self._encode_seq(seq_n, self.critic_tcn)
        return self.critic_mlp(torch.cat((enc, priv_n), dim=-1)).squeeze(-1)

    def sample(self, obs_seq: torch.Tensor, priv: torch.Tensor):
        seq_n = self.obs_norm(obs_seq)
        dist, _ = self._actor_dist_and_context(seq_n, deterministic_z=False if self.training else True)
        latent = dist.rsample()
        priv_n = self.priv_norm(priv)
        val = self.critic_mlp(torch.cat((self._encode_seq(seq_n, self.critic_tcn), priv_n), dim=-1)).squeeze(-1)
        return latent.tanh(), latent, self.log_probability(dist, latent), val

    def evaluate_full(self, obs_seq: torch.Tensor, priv: torch.Tensor, latent: torch.Tensor):
        """Drop-in single-pass PPO + EstNet evaluation returning ``(logp, entropy, value, det_mu, est_pred)``."""
        seq_n = self.obs_norm(obs_seq)
        dist, ctx = self._actor_dist_and_context(seq_n, deterministic_z=False if self.training else True)
        logp = self.log_probability(dist, latent)
        entropy = -self.log_probability(dist, dist.rsample())
        priv_n = self.priv_norm(priv)
        val = self.critic_mlp(torch.cat((self._encode_seq(seq_n, self.critic_tcn), priv_n), dim=-1)).squeeze(-1)
        return logp, entropy, val, dist.mean.tanh(), ctx.est_pred

    def action(self, obs_seq: torch.Tensor) -> torch.Tensor:
        seq_n = self.obs_norm(obs_seq)
        mean, _ = self._actor_mean_and_context(seq_n, deterministic_z=True)
        return mean.float().tanh()

    def action_with_estimate(self, obs_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_n = self.obs_norm(obs_seq)
        mean, ctx = self._actor_mean_and_context(seq_n, deterministic_z=True)
        return mean.float().tanh(), ctx.est_pred

    def action_with_context(
        self,
        obs_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic inference returning ``(action, est_pred, z_mu, z_gated)``."""
        seq_n = self.obs_norm(obs_seq)
        mean, ctx = self._actor_mean_and_context(seq_n, deterministic_z=True)
        return mean.float().tanh(), ctx.est_pred, ctx.z_mu, ctx.z_gated

    def decode_next_sensor(
        self,
        act: torch.Tensor,
        est_pred: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next-step normalized proprioceptive sensor channels strictly through ``(est_pred, z, act)``."""
        dec_in = torch.cat((est_pred, z, act.to(dtype=z.dtype)), dim=-1)
        return self.obs_decoder(dec_in)

    def compute_aux_losses(
        self,
        obs_seq: torch.Tensor,
        next_obs: torch.Tensor,
        act: torch.Tensor,
        est_target: torch.Tensor,
        *,
        beta_kl: float | None = None,
        est_coef: float | None = None,
        rec_coef: float | None = None,
        infonce_coef: float | None = None,
        swav_coef: float | None = None,
        temperature: float | None = None,
        use_ema_target: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute full FP32 self-supervised + supervised auxiliary objective.

        Components:
          1. ``est_loss``: SmoothL1 (``beta=0.15``) against 7D physical ``est_target``.
          2. ``rec_loss``: MSE between residual decoded ``o_{t+1}^{norm,sens}`` and detached target.
          3. ``kl_loss``: beta-VAE Gaussian KL ``D_KL(N(z_mu, diag(exp(z_logvar))) || N(0, I))``.
          4. ``infonce_loss``: HIMLoco symmetric InfoNCE contrastive dynamics loss over in-batch negatives.
          5. ``swav_loss``: Optional 3-step Sinkhorn-Knopp SwAV prototype swap cross-entropy.

        Accepts ``next_obs`` either as ``[B, H, F]`` (takes latest frame ``[:, -1, :]``) or ``[B, F]``.
        """
        b_kl = self.cfg.beta_kl if beta_kl is None else float(beta_kl)
        w_est = self.cfg.est_coef if est_coef is None else float(est_coef)
        w_rec = self.cfg.rec_coef if rec_coef is None else float(rec_coef)
        w_nce = self.cfg.infonce_coef if infonce_coef is None else float(infonce_coef)
        w_swav = self.cfg.swav_coef if swav_coef is None else float(swav_coef)
        tau = max(1e-3, self.cfg.temperature if temperature is None else float(temperature))
        ema_mode = self.cfg.use_ema_target if use_ema_target is None else bool(use_ema_target)

        seq_n = self.obs_norm(obs_seq)
        ctx = self._encode_context_from_norm(seq_n, deterministic=False if self.training else True)

        next_frame = next_obs[:, -1, :] if next_obs.ndim == 3 else next_obs
        with torch.no_grad():
            next_sens_n = self.obs_norm(next_frame)[:, :self.sensor_dim].detach()

        # 1. Explicit 7D physical state estimation loss (FP32)
        est_loss = F.smooth_l1_loss(ctx.est_pred.float(), est_target.float(), beta=0.15)

        # 2. DreamWaQ next-step proprioceptive sensor reconstruction strictly through (est_pred, z, act)
        pred_next_sens = self.decode_next_sensor(act, ctx.est_pred, ctx.z)
        rec_loss = F.mse_loss(pred_next_sens.float(), next_sens_n.float())

        # 3. DreamWaQ beta-VAE KL divergence (FP32)
        z_mu_f = ctx.z_mu.float()
        z_lv_f = ctx.z_logvar.float()
        kl_loss = (-0.5 * (1.0 + z_lv_f - z_mu_f.square() - z_lv_f.exp()).sum(dim=-1)).mean()

        # 4. HIMLoco Hybrid Internal Model contrastive dynamics embeddings strictly through (z, act)
        src_in = torch.cat((ctx.z, act.to(dtype=ctx.z.dtype)), dim=-1)
        l_src = F.normalize(self.him_src_proj(src_in).float(), dim=-1, eps=1e-6)
        l_tgt_online = F.normalize(self.him_tgt_proj(next_sens_n).float(), dim=-1, eps=1e-6)

        bsz = l_src.shape[0]
        labels = torch.arange(bsz, device=l_src.device, dtype=torch.long)
        if bsz > 1:
            if ema_mode:
                with torch.no_grad():
                    l_tgt_ema = F.normalize(self.him_tgt_ema(next_sens_n).float(), dim=-1, eps=1e-6)
                logits_fwd = (l_src @ l_tgt_ema.t()) / tau
                logits_bwd = (l_tgt_online @ l_src.detach().t()) / tau
                infonce_loss = 0.5 * (F.cross_entropy(logits_fwd, labels) + F.cross_entropy(logits_bwd, labels))
            else:
                logits = (l_src @ l_tgt_online.t()) / tau
                infonce_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        else:
            # Degenerate B=1 fallback: pure cosine alignment keeps gradients finite & non-NaN
            l_ref = F.normalize(self.him_tgt_ema(next_sens_n).float(), dim=-1, eps=1e-6) if ema_mode else l_tgt_online
            infonce_loss = 0.5 * ((1.0 - (l_src * l_ref.detach()).sum(dim=-1)).mean()
                                  + (1.0 - (l_tgt_online * l_src.detach()).sum(dim=-1)).mean())

        # 5. Optional SwAV Sinkhorn-Knopp prototype swap loss (HIMLoco Sec. 3.3 Eq. 2)
        if w_swav > 0.0:
            protos = F.normalize(self.him_prototypes.float(), dim=-1, eps=1e-6)  # [K, d_c]
            s_src = l_src @ protos.t()  # [B, K]
            s_tgt = l_tgt_online @ protos.t()  # [B, K]
            q_src = sinkhorn_knopp(s_src, eps=self.cfg.sinkhorn_eps, n_iters=self.cfg.sinkhorn_iters)
            q_tgt = sinkhorn_knopp(s_tgt, eps=self.cfg.sinkhorn_eps, n_iters=self.cfg.sinkhorn_iters)
            log_p_src = F.log_softmax(s_src / tau, dim=-1)
            log_p_tgt = F.log_softmax(s_tgt / tau, dim=-1)
            swav_loss = -0.5 * ((q_tgt * log_p_src).sum(dim=-1).mean() + (q_src * log_p_tgt).sum(dim=-1).mean())
        else:
            swav_loss = torch.zeros((), device=l_src.device, dtype=torch.float32)

        him_loss = w_nce * infonce_loss + w_swav * swav_loss
        total_aux = w_est * est_loss + w_rec * rec_loss + b_kl * kl_loss + him_loss

        metrics = {
            'est_loss': est_loss.detach(),
            'rec_loss': rec_loss.detach(),
            'kl_loss': kl_loss.detach(),
            'infonce_loss': infonce_loss.detach(),
            'swav_loss': swav_loss.detach(),
            'him_loss': him_loss.detach(),
            'total_aux_loss': total_aux.detach(),
            'est_pred': ctx.est_pred.detach(),
            'z_mu': ctx.z_mu.detach(),
            'z_logvar': ctx.z_logvar.detach(),
            'pred_next_sens': pred_next_sens.detach(),
        }
        return total_aux, metrics


# ---------------------------------------------------------------------------
# Exact Z2 Left-Right Group-Equivariant Inference Wrapper (2403.04359)
# ---------------------------------------------------------------------------

def build_obs_mirror_spec(
    frame_size: int,
    *,
    include_q_err: bool | None = None,
    layout: str = 'auto',
    action_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact y -> -y reflection involution ``(perm_idx, sign_vec)`` for any SerpWorm obs layout.

    Supported layouts (matching ``autonomous_env.py:91-149`` and ``unified_env.py:36-82`` identically):
      - ``autonomous_51`` (``frame_size=51``): ``q(14), dq(14), gyro(3), proj_g(3), prev_act(14), cmd(3)``
      - ``autonomous_65`` (``frame_size=65``): ``q(14), dq(14), q_err(14), gyro(3), proj_g(3), prev_act(14), cmd(3)``
      - ``unified_54`` (``frame_size=54``): ``q(14), dq(14), gyro(3), proj_g(3), prev_act(14), mode(3), cmd(3)``
      - ``unified_49`` (``frame_size=49``): ``q(14), dq(14), gyro(3), proj_g(3), prev_act(9), mode(3), cmd(3)``
    """
    f = int(frame_size)
    if layout == 'auto':
        if f == 65 or (include_q_err is True and f == 65):
            resolved = 'autonomous_65'
        elif f == 51:
            resolved = 'autonomous_51'
        elif f == 54:
            resolved = 'unified_54'
        elif f == 49:
            resolved = 'unified_49'
        else:
            raise ValueError(
                f'Unsupported frame_size={f} for automatic Z2 mirror spec; expected one of (49, 51, 54, 65).'
            )
    else:
        resolved = layout

    perm = list(range(f))
    sign = [1.0] * f

    def _apply_14d_block(start: int) -> None:
        for i, src_i in enumerate(_PAIR_SWAP_10):
            perm[start + i] = start + src_i
        for i in range(10, 14):
            sign[start + i] = -1.0

    # q(0:14) and dq(14:28) are identical across all layouts
    _apply_14d_block(0)
    _apply_14d_block(14)

    off = 28
    if resolved == 'autonomous_65':
        _apply_14d_block(off)
        off += 14

    # Gyro pseudo-vector (wx, wy, wz) -> (-wx, +wy, -wz)
    sign[off + 0] = -1.0
    sign[off + 2] = -1.0
    # Projected gravity polar vector (gx, gy, gz) -> (+gx, -gy, +gz)
    sign[off + 4] = -1.0
    off += 6

    if resolved == 'unified_49' or (action_size == 9 and resolved.startswith('unified')):
        # Paired 5D tendons invariant; 4D yaw negated
        for i in range(5, 9):
            sign[off + i] = -1.0
        off += 9
    else:
        _apply_14d_block(off)
        off += 14

    if resolved.startswith('unified'):
        # 3D mode one-hot invariant
        off += 3

    # Trailing 3D command [vx, wz, stop] -> [+vx, -wz, +stop]
    if off + 3 != f:
        raise ValueError(f'Layout {resolved!r} offset mismatch: expected {f}, reached {off + 3}')
    sign[off + 1] = -1.0

    return torch.tensor(perm, dtype=torch.long), torch.tensor(sign, dtype=torch.float32)


def build_action_mirror_spec(action_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact y -> -y action reflection involution ``K_g`` (``K_g^{-1} = K_g``)."""
    a = int(action_size)
    if a == 14:
        perm = list(_PAIR_SWAP_10) + [10, 11, 12, 13]
        sign = [1.0] * 10 + [-1.0] * 4
    elif a == 9:
        perm = list(range(9))
        sign = [1.0] * 5 + [-1.0] * 4
    else:
        raise ValueError(f'Unsupported action_size={a}; expected 14 (independent_10d) or 9 (paired_5d).')
    return torch.tensor(perm, dtype=torch.long), torch.tensor(sign, dtype=torch.float32)


def build_est_mirror_spec(est_size: int = 7) -> torch.Tensor:
    """Build exact y -> -y reflection sign vector ``M_g`` for 7D EstNet outputs."""
    e = int(est_size)
    if e == 7:
        # [vx, vy, wz, wall_conf, wall_side, terrain_drag, slip]
        return torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0], dtype=torch.float32)
    return torch.ones(e, dtype=torch.float32)


def mirror_tensor_by_spec(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Apply an exact involution ``(perm, sign)`` along the last dimension of ``x``."""
    return x[..., perm] * sign.to(device=x.device, dtype=x.dtype)


class GroupEquivariantPolicyWrapper(nn.Module):
    """Exact Z2 left-right group-equivariant inference wrapper (Mittal et al., ICRA 2024, 2403.04359).

    Wraps any trained ``ProprioceptiveEstimatorActorCritic``, ``TemporalActorCritic``, or
    ``HIMDreamWaQActorCritic`` to guarantee exact equivariance by construction:
      ``pi_eq(o)  = 0.5 * (pi(o)  + K_g^{-1} pi(L_g o))``
      ``est_eq(o) = 0.5 * (e(o)   + M_g^{-1} e(L_g o))``

    With ``batched=False`` (default), both ``o`` and ``L_g o`` execute with identical batch shapes,
    guaranteeing **bit-exact 0.0** IEEE-754 mirror error ``max|pi_eq(L_g o) - K_g pi_eq(o)| == 0.0``
    everywhere off the reflection-symmetric saddle manifold.
    Set ``batched=True`` to fuse ``[o, L_g o]`` into one ``2B`` kernel call for maximum GPU throughput.

    **Saddle-Point Symmetry-Breaking Gate (``saddle_break_tol``):**
    At an exact reflection-symmetric rest state ``L_g o == o`` under a straight-line moving command
    (``stop < 0.5``, ``|cmd_vx| > 1e-3``, ``|cmd_wz| < 1e-3``), naive Reynolds averaging cancels
    left-first and right-first gait phases into ``yaw == 0``, trapping undulating locomotion on an
    unstable saddle point. When ``saddle_break_tol > 0`` (default ``0.02``), ONLY batch items on
    this straight-moving symmetric manifold (``||o - L_g o||_inf < saddle_break_tol``) emit the
    un-averaged base action ``pi(o)`` to initiate gait phase; on ``stop >= 0.5``, any turning
    command, or once ``||o - L_g o||_inf >= saddle_break_tol`` (from step 1 onward), exact
    Reynolds averaging applies identically. Estimator outputs always remain strictly equivariant.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        frame_size: int | None = None,
        action_size: int | None = None,
        est_size: int | None = None,
        include_q_err: bool | None = None,
        layout: str = 'auto',
        batched: bool = False,
        saddle_break_tol: float = 0.02,
    ):
        super().__init__()
        self.policy = policy
        self.batched = bool(batched)
        self.saddle_break_tol = float(saddle_break_tol)

        f_sz = int(frame_size if frame_size is not None else getattr(policy, 'frame_size'))
        a_sz = int(action_size if action_size is not None else getattr(policy, 'action_size'))
        has_est = hasattr(policy, 'action_with_estimate') and (
            est_size is None or int(est_size) > 0
        ) and getattr(policy, 'est_size', 0) > 0
        e_sz = int(est_size) if est_size is not None else (int(getattr(policy, 'est_size', 0)) if has_est else 0)

        self.frame_size = f_sz
        self.action_size = a_sz
        self.est_size = e_sz
        self.has_estimator = e_sz > 0 and hasattr(policy, 'action_with_estimate')
        self.has_context = hasattr(policy, 'action_with_context')

        obs_perm, obs_sign = build_obs_mirror_spec(
            f_sz, include_q_err=include_q_err, layout=layout, action_size=a_sz
        )
        act_perm, act_sign = build_action_mirror_spec(a_sz)
        est_sign = build_est_mirror_spec(max(1, e_sz))

        self.register_buffer('obs_perm', obs_perm)
        self.register_buffer('obs_sign', obs_sign)
        self.register_buffer('act_perm', act_perm)
        self.register_buffer('act_sign', act_sign)
        self.register_buffer('est_sign', est_sign)

        # Ensure buffers live on same device as wrapped policy parameters
        first_param = next(policy.parameters(), None)
        if first_param is not None:
            self.to(device=first_param.device)

    def mirror_obs(self, obs_seq: torch.Tensor) -> torch.Tensor:
        return mirror_tensor_by_spec(obs_seq, self.obs_perm, self.obs_sign)

    def mirror_action(self, act: torch.Tensor) -> torch.Tensor:
        return mirror_tensor_by_spec(act, self.act_perm, self.act_sign)

    def mirror_est(self, est: torch.Tensor) -> torch.Tensor:
        return est * self.est_sign.to(device=est.device, dtype=est.dtype)

    def _saddle_break_mask(self, obs_seq: torch.Tensor, m_obs: torch.Tensor) -> torch.Tensor | None:
        """Identify straight-moving states trapped on the reflection-symmetric saddle ``||o - L_g o||_inf < tol``."""
        if self.saddle_break_tol <= 0.0:
            return None
        last = obs_seq[:, -1, :] if obs_seq.ndim == 3 else obs_seq
        if last.shape[-1] < 3:
            return None
        cmd_vx = last[:, -3]
        cmd_wz = last[:, -2]
        cmd_stop = last[:, -1]
        straight_moving = (cmd_stop < 0.5) & (cmd_vx.abs() > 1e-3) & (cmd_wz.abs() < 1e-3)
        if not bool(straight_moving.any().item()):
            return None
        reduce_dims = tuple(range(1, obs_seq.ndim))
        asym = (obs_seq - m_obs).abs().amax(dim=reduce_dims)
        mask = straight_moving & (asym < self.saddle_break_tol)
        return mask if bool(mask.any().item()) else None

    def _apply_saddle_gate(
        self,
        obs_seq: torch.Tensor,
        m_obs: torch.Tensor,
        a_orig: torch.Tensor,
        eq_act: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._saddle_break_mask(obs_seq, m_obs)
        if mask is None:
            return eq_act
        return torch.where(mask[:, None], a_orig, eq_act)

    def _run_action_pair(self, obs_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m_obs = self.mirror_obs(obs_seq)
        if self.batched:
            bsz = obs_seq.shape[0]
            both = self.policy.action(torch.cat((obs_seq, m_obs), dim=0))
            return m_obs, both[:bsz], both[bsz:]
        return m_obs, self.policy.action(obs_seq), self.policy.action(m_obs)

    def _run_action_est_pair(
        self,
        obs_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m_obs = self.mirror_obs(obs_seq)
        if self.batched:
            bsz = obs_seq.shape[0]
            act_both, est_both = self.policy.action_with_estimate(torch.cat((obs_seq, m_obs), dim=0))
            return m_obs, act_both[:bsz], act_both[bsz:], est_both[:bsz], est_both[bsz:]
        a_o, e_o = self.policy.action_with_estimate(obs_seq)
        a_m, e_m = self.policy.action_with_estimate(m_obs)
        return m_obs, a_o, a_m, e_o, e_m

    def action(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Exact Z2-equivariant deterministic action with optional straight-moving saddle-point gate."""
        m_obs, a_orig, a_mirr = self._run_action_pair(obs_seq)
        eq_act = 0.5 * (a_orig + self.mirror_action(a_mirr))
        return self._apply_saddle_gate(obs_seq, m_obs, a_orig, eq_act)

    def estimate(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Exact Z2-equivariant 7D EstNet prediction ``0.5 * (e(o) + M_g e(L_g o))``."""
        if not self.has_estimator:
            raise AttributeError('Wrapped policy does not expose an estimator head.')
        m_obs = self.mirror_obs(obs_seq)
        if self.batched:
            bsz = obs_seq.shape[0]
            e_both = self.policy.estimate(torch.cat((obs_seq, m_obs), dim=0))
            e_orig, e_mirr = e_both[:bsz], e_both[bsz:]
        else:
            e_orig = self.policy.estimate(obs_seq)
            e_mirr = self.policy.estimate(m_obs)
        return 0.5 * (e_orig + self.mirror_est(e_mirr))

    def action_with_estimate(self, obs_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-pair exact equivariant action (with optional saddle gate) + mirror-averaged 7D EstNet."""
        if not self.has_estimator:
            raise AttributeError('Wrapped policy does not implement action_with_estimate().')
        m_obs, a_orig, a_mirr, e_orig, e_mirr = self._run_action_est_pair(obs_seq)
        eq_act = 0.5 * (a_orig + self.mirror_action(a_mirr))
        eq_act = self._apply_saddle_gate(obs_seq, m_obs, a_orig, eq_act)
        eq_est = 0.5 * (e_orig + self.mirror_est(e_mirr))
        return eq_act, eq_est

    def action_with_context(
        self,
        obs_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact equivariant action & estimate plus Z2-invariant averaged latent context."""
        if not self.has_context:
            raise AttributeError('Wrapped policy does not implement action_with_context().')
        m_obs = self.mirror_obs(obs_seq)
        if self.batched:
            bsz = obs_seq.shape[0]
            a_b, e_b, z_b, zg_b = self.policy.action_with_context(torch.cat((obs_seq, m_obs), dim=0))
            a_o, a_m = a_b[:bsz], a_b[bsz:]
            e_o, e_m = e_b[:bsz], e_b[bsz:]
            z_o, z_m = z_b[:bsz], z_b[bsz:]
            zg_o, zg_m = zg_b[:bsz], zg_b[bsz:]
        else:
            a_o, e_o, z_o, zg_o = self.policy.action_with_context(obs_seq)
            a_m, e_m, z_m, zg_m = self.policy.action_with_context(m_obs)
        eq_act = 0.5 * (a_o + self.mirror_action(a_m))
        eq_act = self._apply_saddle_gate(obs_seq, m_obs, a_o, eq_act)
        eq_est = 0.5 * (e_o + self.mirror_est(e_m))
        inv_z = 0.5 * (z_o + z_m)
        inv_zg = 0.5 * (zg_o + zg_m)
        return eq_act, eq_est, inv_z, inv_zg

    def forward(self, obs_seq: torch.Tensor) -> Any:
        if self.has_estimator:
            return self.action_with_estimate(obs_seq)
        return self.action(obs_seq)

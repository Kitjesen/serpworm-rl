"""Morphology-Agnostic Entity-Tokenized HIM-DreamWaQ Policy & Universal Finite-Group Equivariance.

Decouples proprioceptive RL control from any fixed robot dimension (e.g. 5-segment 65D/14D):
  1. ``TokenRunningNorm`` -- FP64 Welford running normalizer for variable-N entity tokens
     (masking inactive actuators and preserving static morphology metadata channels), global
     IMU/command streams, and fixed-width privileged critic observations.
  2. ``TokenizedHIMDreamWaQ`` -- processes variable-length actuator/module tokens
     ``(B, H, N_act, d_tok)`` + global IMU/command history ``(B, H, d_glob)`` + optional
     ``active_mask (B, N_act)`` (enabling mixed-length training, zero-shot cross-length transfer
     e.g. N=3..6 -> N=7, and live mid-mission actuator failure resilience).
     Includes morphology-invariant conditioning, an asymmetric value head (``critic``),
     exact FP32 tanh-squashed PPO distribution methods, DreamWaQ beta-VAE uncertainty-gated
     context, and HIMLoco contrastive dynamics (InfoNCE + 3-step Sinkhorn-Knopp SwAV).
  3. ``FiniteGroupEquivariantWrapper`` -- generalizes Reynolds group averaging
     ``pi_eq(o) = (1/|G|) sum_{g in G} K_g^{-1} pi(L_g o)``
     to ANY finite symmetry group (Z2, Z2xZ2, C4, D4, ...) defined by ``SymmetryGroupSpec``,
     guaranteeing exact algebraic equivariance by construction for both inference and PPO training.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from flexgraph.graph import SymmetryGroupSpec


@dataclass(frozen=True)
class TokenizedPolicyConfig:
    d_tok: int = 8              # per-actuator token: [pos_norm, side, is_tendon, is_joint, q, dq, q_err, u_prev]
    d_glob: int = 9             # global token: [omega_b(3), grav_b(3), cmd(3)]
    d_model: int = 64           # shared token embedding dimension
    n_heads: int = 4
    n_attn_layers: int = 2
    est_dim: int = 7            # explicit proprioceptive state estimate [vx, vy, wz, wall_conf, wall_side, drag, slip]
    latent_dim: int = 16        # beta-VAE implicit terrain/contact context z_t
    contrast_dim: int = 32      # HIM contrastive unit-sphere dimension
    num_prototypes: int = 16
    beta_kl: float = 0.002
    temperature: float = 0.10
    sinkhorn_iters: int = 3
    sinkhorn_eps: float = 0.05
    # Additive multi-morphology RL & asymmetric PPO extensions (100% backward-compatible defaults)
    d_priv: int = 16            # morphology-invariant privileged critic summary dimension
    morph_dim: int = 8          # automatically extracted permutation-invariant morphology descriptor
    critic_hidden: int = 128
    use_obs_norm: bool = True
    tok_norm_start: int = 4     # preserve static metadata [0:4] (seg_norm, side, is_tendon, is_joint)
    norm_clip: float = 10.0
    initial_log_std: float = -0.5
    min_std: float = 0.05
    max_std: float = 1.5
    detach_actor_tokens: bool = False  # False allows PPO actor gradients through tok_tcn/glob_tcn/module_attn


class TokenizedForwardOut(NamedTuple):
    action_mean: torch.Tensor   # (B, N_act)
    action_std: torch.Tensor    # (B, N_act)
    est_pred: torch.Tensor      # (B, est_dim)
    z_mu: torch.Tensor          # (B, latent_dim)
    z_logvar: torch.Tensor      # (B, latent_dim)
    z_gate: torch.Tensor        # (B, latent_dim)
    z_gated: torch.Tensor       # (B, latent_dim)


class ActionSampleOut(NamedTuple):
    action: torch.Tensor        # (B, N_act) in [-1, 1], 0 on masked actuators
    latent: torch.Tensor        # (B, N_act) pre-tanh sample
    log_prob: torch.Tensor      # (B,) joint log-prob over active actuators
    value: torch.Tensor         # (B,) critic value estimate
    raw_loc: torch.Tensor       # (B, N_act) pre-tanh Gaussian mean (for exact PPO KL)
    forward_out: TokenizedForwardOut


class ActionEvalOut(NamedTuple):
    log_prob: torch.Tensor      # (B,) joint log-prob over active actuators
    entropy: torch.Tensor       # (B,) mean per-active-actuator entropy
    value: torch.Tensor         # (B,) critic value estimate
    raw_loc: torch.Tensor       # (B, N_act) pre-tanh Gaussian mean
    std: torch.Tensor           # (B, N_act) Gaussian std
    forward_out: TokenizedForwardOut


class TokenRunningNorm(nn.Module):
    """Device-resident FP64 Welford running normalizer for variable-N tokens, global obs, and privileged obs.

    - Respects ``active_mask (B, N_act)`` so jammed/padded actuators never corrupt token statistics.
    - Leaves static token metadata channels ``[0:tok_norm_start]`` untouched to preserve morphology
      geometry and exact finite-group reflection signs.
    - Acts as an exact identity pass-through before the first ``.update()`` call so freshly
      initialized policies retain bit-exact zero-mean behavior and algebraic equivariance.
    """

    def __init__(
        self,
        d_tok: int,
        d_glob: int,
        d_priv: int = 16,
        *,
        tok_norm_start: int = 4,
        clip: float = 10.0,
        min_var: float = 1e-2,
    ):
        super().__init__()
        self.d_tok = int(d_tok)
        self.d_glob = int(d_glob)
        self.d_priv = int(d_priv)
        self.tok_norm_start = max(0, min(int(tok_norm_start), self.d_tok))
        self.tok_dyn_dim = max(0, self.d_tok - self.tok_norm_start)
        self.clip = float(clip)
        self.min_var = float(min_var)

        self.register_buffer('tok_mean', torch.zeros(max(1, self.tok_dyn_dim), dtype=torch.float64))
        self.register_buffer('tok_var', torch.ones(max(1, self.tok_dyn_dim), dtype=torch.float64))
        self.register_buffer('tok_count', torch.zeros((), dtype=torch.float64))

        self.register_buffer('glob_mean', torch.zeros(max(1, self.d_glob), dtype=torch.float64))
        self.register_buffer('glob_var', torch.ones(max(1, self.d_glob), dtype=torch.float64))
        self.register_buffer('glob_count', torch.zeros((), dtype=torch.float64))

        self.register_buffer('priv_mean', torch.zeros(max(1, self.d_priv), dtype=torch.float64))
        self.register_buffer('priv_var', torch.ones(max(1, self.d_priv), dtype=torch.float64))
        self.register_buffer('priv_count', torch.zeros((), dtype=torch.float64))

    @staticmethod
    @torch.no_grad()
    def _welford_update(
        mean: torch.Tensor,
        var: torch.Tensor,
        count: torch.Tensor,
        flat: torch.Tensor,
        min_var: float,
    ) -> None:
        if flat.numel() == 0:
            return
        finite_rows = torch.isfinite(flat).all(dim=-1)
        if not bool(finite_rows.any().item()):
            return
        x = flat[finite_rows].to(device=mean.device, dtype=torch.float64)
        n = float(x.shape[0])
        bm = x.mean(dim=0)
        bv = x.var(dim=0, unbiased=False)
        c_old = count.item()
        if c_old <= 0.0:
            mean.copy_(bm)
            var.copy_(bv.clamp_min(min_var))
            count.fill_(n)
            return
        tot = c_old + n
        delta = bm - mean
        mean.add_(delta * (n / tot))
        m2 = var * c_old + bv * n + delta.square() * (c_old * n / tot)
        var.copy_((m2 / tot).clamp_min(min_var))
        count.fill_(tot)

    @torch.no_grad()
    def update(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Update running statistics from the newest frame (frozen in ``.eval()`` unless ``force=True``)."""
        if (not self.training) and (not force):
            return
        if self.tok_dyn_dim > 0 and tokens.numel() > 0:
            # Use newest frame when 4D (B, H, N, d_tok) or directly when 3D (B, N, d_tok)
            t_cur = tokens[:, -1] if tokens.ndim == 4 else tokens
            t_dyn = t_cur[..., self.tok_norm_start:]
            if active_mask is not None:
                m = active_mask.to(device=t_dyn.device, dtype=torch.bool)
                t_flat = t_dyn[m]
            else:
                t_flat = t_dyn.reshape(-1, self.tok_dyn_dim)
            self._welford_update(self.tok_mean, self.tok_var, self.tok_count, t_flat, self.min_var)

        if self.d_glob > 0 and glob.numel() > 0:
            g_cur = glob[:, -1] if glob.ndim == 3 else glob
            self._welford_update(
                self.glob_mean, self.glob_var, self.glob_count, g_cur.reshape(-1, self.d_glob), self.min_var
            )

        if priv is not None and self.d_priv > 0 and priv.numel() > 0:
            self._welford_update(
                self.priv_mean, self.priv_var, self.priv_count, priv.reshape(-1, self.d_priv), self.min_var
            )

    def normalize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.tok_dyn_dim == 0 or float(self.tok_count.item()) <= 0.0:
            return tokens
        mean = self.tok_mean.to(device=tokens.device, dtype=tokens.dtype)
        inv_std = self.tok_var.clamp_min(self.min_var).rsqrt().to(device=tokens.device, dtype=tokens.dtype)
        dyn = ((tokens[..., self.tok_norm_start:] - mean) * inv_std).clamp(-self.clip, self.clip)
        if self.tok_norm_start == 0:
            return dyn
        return torch.cat((tokens[..., :self.tok_norm_start], dyn), dim=-1)

    def normalize_glob(self, glob: torch.Tensor) -> torch.Tensor:
        if self.d_glob == 0 or float(self.glob_count.item()) <= 0.0:
            return glob
        mean = self.glob_mean.to(device=glob.device, dtype=glob.dtype)
        inv_std = self.glob_var.clamp_min(self.min_var).rsqrt().to(device=glob.device, dtype=glob.dtype)
        return ((glob - mean) * inv_std).clamp(-self.clip, self.clip)

    def normalize_priv(self, priv: torch.Tensor) -> torch.Tensor:
        if self.d_priv == 0 or float(self.priv_count.item()) <= 0.0:
            return priv
        mean = self.priv_mean.to(device=priv.device, dtype=priv.dtype)
        inv_std = self.priv_var.clamp_min(self.min_var).rsqrt().to(device=priv.device, dtype=priv.dtype)
        return ((priv - mean) * inv_std).clamp(-self.clip, self.clip)


def _tanh_squashed_log_prob(
    loc: torch.Tensor,
    std: torch.Tensor,
    latent: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact tanh-squashed Normal log-probability in FP32, summed strictly over active actuators."""
    loc_f = loc.float()
    std_f = std.float().clamp_min(1e-5)
    lat_f = latent.float().clamp(-6.0, 6.0)
    dist = torch.distributions.Normal(loc_f, std_f, validate_args=False)
    log_jac = 2.0 * (math.log(2.0) - lat_f - F.softplus(-2.0 * lat_f))
    per_dim = dist.log_prob(lat_f) - log_jac
    if active_mask is not None:
        per_dim = per_dim * active_mask.to(device=per_dim.device, dtype=torch.float32)
    return per_dim.sum(dim=-1)


def _mean_active_entropy(std: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean per-active-actuator Gaussian entropy in FP32 (scale-invariant across robot actuator count N)."""
    std_f = std.float().clamp_min(1e-5)
    per_dim = 0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(std_f)
    if active_mask is not None:
        w = active_mask.to(device=per_dim.device, dtype=torch.float32)
        return (per_dim * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1.0)
    return per_dim.mean(dim=-1)


@torch.no_grad()
def sinkhorn_knopp(scores: torch.Tensor, eps: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    logits = scores.detach().float()
    b_size, k_size = logits.shape
    if b_size == 0 or k_size == 0:
        return torch.empty_like(logits)
    shifted = (logits - logits.max(dim=-1, keepdim=True).values) / max(float(eps), 1e-4)
    q = torch.exp(shifted.clamp(-30.0, 0.0)).t()
    q = q / q.sum().clamp_min(1e-8)
    for _ in range(max(1, int(n_iters))):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8) / float(k_size)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-8) / float(b_size)
    return (q * float(b_size)).t().contiguous()


class _RightAlignedCausalConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left_pad = self.kernel_size - 1
        if self.stride == 2 and self.kernel_size >= 2 and (x.shape[-1] % 2 == 0):
            left_pad -= 1
        return self.conv(F.pad(x, (left_pad, 0)))


class _EntityMultiScaleTCN(nn.Module):
    """Shared right-aligned multi-scale causal temporal encoder applied independently per entity token."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        c1 = d_out // 2
        c2 = d_out - c1
        self.short_branch = nn.Sequential(
            _RightAlignedCausalConv1d(d_in, c1, kernel_size=3, stride=2),
            nn.SiLU(),
            _RightAlignedCausalConv1d(c1, c1, kernel_size=3, stride=2),
            nn.SiLU(),
        )
        self.long_branch = nn.Sequential(
            _RightAlignedCausalConv1d(d_in, c2, kernel_size=5, stride=2),
            nn.SiLU(),
            _RightAlignedCausalConv1d(c2, c2, kernel_size=5, stride=2),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(nn.Linear(d_out + d_in, d_out), nn.LayerNorm(d_out), nn.SiLU())

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (N_entities, H, d_in) -> (N_entities, d_out)."""
        x_t = seq.transpose(1, 2)  # (N_ent, d_in, H)
        h_s = self.short_branch(x_t)[..., -1]  # newest right-aligned slot
        h_l = self.long_branch(x_t)[..., -1]
        return self.proj(torch.cat((h_s, h_l, seq[:, -1, :]), dim=-1))


class TokenizedHIMDreamWaQ(nn.Module):
    """Morphology-agnostic proprioceptive Actor-Critic + HIM-DreamWaQ bottleneck over arbitrary N actuators."""

    def __init__(self, cfg: TokenizedPolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg or TokenizedPolicyConfig()
        c = self.cfg

        self.obs_norm = TokenRunningNorm(
            d_tok=c.d_tok,
            d_glob=c.d_glob,
            d_priv=c.d_priv,
            tok_norm_start=c.tok_norm_start,
            clip=c.norm_clip,
        )

        self.tok_tcn = _EntityMultiScaleTCN(c.d_tok, c.d_model)
        self.glob_tcn = _EntityMultiScaleTCN(c.d_glob, c.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=2 * c.d_model,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.module_attn = nn.TransformerEncoder(enc_layer, num_layers=c.n_attn_layers, enable_nested_tensor=False)

        # Shared DreamWaQ CENet trunk -> 7D explicit EstNet + 16D beta-VAE + uncertainty gate
        self.cenet_trunk = nn.Sequential(nn.Linear(2 * c.d_model + c.morph_dim, 128), nn.LayerNorm(128), nn.SiLU())
        self.est_head = nn.Linear(128, c.est_dim)
        self.vae_mu = nn.Linear(128, c.latent_dim)
        self.vae_logvar = nn.Linear(128, c.latent_dim)
        self.gate_net = nn.Linear(2 * c.latent_dim, c.latent_dim)

        # Shared per-actuator action head (independent of N_act!)
        self.actor_head = nn.Sequential(
            nn.Linear(c.d_model + c.est_dim + c.latent_dim + c.morph_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Small init on final actor projection prevents initial tanh saturation
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.actor_head[-1].bias)
        self.log_std = nn.Parameter(torch.full((1,), float(c.initial_log_std)))

        # Asymmetric morphology-agnostic Critic head: pools variable-N tokens + fixed-width privileged summary
        critic_in_dim = 3 * c.d_model + c.morph_dim + c.est_dim + c.latent_dim + c.d_priv
        self.critic = nn.Sequential(
            nn.Linear(critic_in_dim, c.critic_hidden),
            nn.LayerNorm(c.critic_hidden),
            nn.SiLU(),
            nn.Linear(c.critic_hidden, c.critic_hidden),
            nn.LayerNorm(c.critic_hidden),
            nn.SiLU(),
            nn.Linear(c.critic_hidden, 1),
        )
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

        # Bottleneck HIM contrastive dynamics & next-global-sensor decoder (training-only auxiliary)
        self.next_glob_dec = nn.Sequential(
            nn.Linear(c.est_dim + c.latent_dim + c.d_model, 64),
            nn.SiLU(),
            nn.Linear(64, max(1, c.d_glob - 3)),  # predict next IMU [omega, grav], excluding exogenous cmd
        )
        self.him_src = nn.Sequential(nn.Linear(c.latent_dim + c.d_model, 64), nn.SiLU(), nn.Linear(64, c.contrast_dim))
        self.him_tgt = nn.Sequential(nn.Linear(max(1, c.d_glob - 3), 64), nn.SiLU(), nn.Linear(64, c.contrast_dim))
        self.prototypes = nn.Parameter(F.normalize(torch.randn(c.num_prototypes, c.contrast_dim), dim=-1))

    def update_obs_norm(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
    ) -> None:
        if self.cfg.use_obs_norm:
            self.obs_norm.update(tokens, glob, active_mask=active_mask, priv=priv)

    def _compute_morph_descriptor(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Permutation- & reflection-invariant 8D morphology summary derived purely from entity tokens."""
        B, _, N, d_t = tokens.shape
        if self.cfg.morph_dim == 0:
            return torch.zeros((B, 0), device=tokens.device, dtype=tokens.dtype)
        cur = tokens[:, -1]  # (B, N, d_t)
        if active_mask is not None:
            w = active_mask.to(device=tokens.device, dtype=tokens.dtype)
        else:
            w = torch.ones((B, N), device=tokens.device, dtype=tokens.dtype)
        n_act_eff = w.sum(dim=1)  # (B,)
        denom = n_act_eff.clamp_min(1.0)

        log_n_active = torch.log1p(n_act_eff) / 3.0
        active_frac = n_act_eff / float(max(N, 1))
        log_n_total = torch.full((B,), math.log1p(float(N)) / 3.0, device=tokens.device, dtype=tokens.dtype)

        if d_t >= 4:
            seg = cur[..., 0]
            side = cur[..., 1]
            is_ten = cur[..., 2]
            is_jnt = cur[..., 3]
            ten_frac = (is_ten * w).sum(dim=1) / denom
            jnt_frac = (is_jnt * w).sum(dim=1) / denom
            # Symmetric around segment midpoint 0.5 -> invariant under Z2, Z2xZ2 (F-B reversal), C4, D4
            seg_spread = ((seg - 0.5).abs() * w).sum(dim=1) / denom
            seg_mean = (seg * w).sum(dim=1) / denom
            seg_std = torch.sqrt((((seg - seg_mean[:, None]) ** 2) * w).sum(dim=1) / denom + 1e-6)
            side_abs = (side.abs() * w).sum(dim=1) / denom
        else:
            zeros = torch.zeros((B,), device=tokens.device, dtype=tokens.dtype)
            ten_frac = jnt_frac = seg_spread = seg_std = side_abs = zeros

        desc = torch.stack(
            (log_n_active, active_frac, ten_frac, jnt_frac, seg_spread, seg_std, side_abs, log_n_total),
            dim=-1,
        )
        if desc.shape[-1] >= self.cfg.morph_dim:
            return desc[..., :self.cfg.morph_dim]
        pad = torch.zeros((B, self.cfg.morph_dim - desc.shape[-1]), device=tokens.device, dtype=tokens.dtype)
        return torch.cat((desc, pad), dim=-1)

    def _encode_context(
        self,
        tokens: torch.Tensor,          # (B, H, N_act, d_tok)
        glob: torch.Tensor,            # (B, H, d_glob)
        active_mask: torch.Tensor | None = None,  # (B, N_act) bool, True = healthy/present
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        morph_desc = self._compute_morph_descriptor(tokens, active_mask)
        if self.cfg.use_obs_norm:
            tokens_in = self.obs_norm.normalize_tokens(tokens)
            glob_in = self.obs_norm.normalize_glob(glob)
        else:
            tokens_in, glob_in = tokens, glob

        B, H, N, d_t = tokens_in.shape
        tok_emb = self.tok_tcn(tokens_in.transpose(1, 2).reshape(B * N, H, d_t)).reshape(B, N, self.cfg.d_model)
        glob_emb = self.glob_tcn(glob_in)  # (B, d_model)

        # Prepend global body token at slot 0 -> (B, 1+N, d_model)
        seq = torch.cat((glob_emb[:, None, :], tok_emb), dim=1)
        if active_mask is not None:
            act_b = active_mask.bool()
            pad_mask = torch.cat(
                (torch.zeros((B, 1), device=tokens.device, dtype=torch.bool), ~act_b), dim=1
            )
            h_all = self.module_attn(seq, src_key_padding_mask=pad_mask)
            w = act_b.to(dtype=tokens.dtype)[..., None]
            tok_pool = (h_all[:, 1:] * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
            neg_inf = torch.full_like(h_all[:, 1:], -1e4)
            has_any = act_b.any(dim=1, keepdim=True)
            tok_max_raw = torch.where(act_b[..., None], h_all[:, 1:], neg_inf).max(dim=1).values
            tok_max = torch.where(has_any, tok_max_raw, torch.zeros_like(tok_max_raw))
        else:
            h_all = self.module_attn(seq)
            tok_pool = h_all[:, 1:].mean(dim=1)
            tok_max = h_all[:, 1:].max(dim=1).values

        h_glob = h_all[:, 0]
        h_tok = h_all[:, 1:]  # (B, N, d_model)
        ctx = self.cenet_trunk(torch.cat((h_glob, tok_pool, morph_desc), dim=-1))

        raw_e = self.est_head(ctx)
        if self.cfg.est_dim == 7:
            est_pred = torch.stack(
                (
                    raw_e[:, 0],
                    raw_e[:, 1],
                    raw_e[:, 2],
                    torch.sigmoid(raw_e[:, 3]),
                    torch.tanh(raw_e[:, 4]),
                    torch.sigmoid(raw_e[:, 5]),
                    torch.sigmoid(raw_e[:, 6]),
                ),
                dim=-1,
            )
        else:
            est_pred = raw_e

        z_mu = self.vae_mu(ctx)
        z_logvar = self.vae_logvar(ctx).clamp(-6.0, 2.0)
        z_sample = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mu) if self.training else z_mu
        z_gate = torch.sigmoid(self.gate_net(torch.cat((z_mu.detach(), z_logvar.detach()), dim=-1)))
        z_gated = z_gate * z_sample.detach()
        return h_glob, h_tok, tok_pool, tok_max, morph_desc, est_pred, z_sample, z_mu, z_logvar, z_gate, z_gated

    def _critic_from_ctx(
        self,
        h_glob: torch.Tensor,
        tok_pool: torch.Tensor,
        tok_max: torch.Tensor,
        morph_desc: torch.Tensor,
        est_pred: torch.Tensor,
        z_gated: torch.Tensor,
        priv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = h_glob.shape[0]
        if self.cfg.d_priv > 0:
            if priv is None:
                priv_in = torch.zeros((B, self.cfg.d_priv), device=h_glob.device, dtype=h_glob.dtype)
            else:
                p = priv.to(device=h_glob.device, dtype=h_glob.dtype)
                if p.shape[-1] < self.cfg.d_priv:
                    pad = torch.zeros((B, self.cfg.d_priv - p.shape[-1]), device=h_glob.device, dtype=h_glob.dtype)
                    p = torch.cat((p, pad), dim=-1)
                elif p.shape[-1] > self.cfg.d_priv:
                    p = p[..., :self.cfg.d_priv]
                priv_in = self.obs_norm.normalize_priv(p) if self.cfg.use_obs_norm else p
            critic_in = torch.cat(
                (h_glob, tok_pool, tok_max, morph_desc, est_pred.detach(), z_gated.detach(), priv_in),
                dim=-1,
            )
        else:
            critic_in = torch.cat(
                (h_glob, tok_pool, tok_max, morph_desc, est_pred.detach(), z_gated.detach()),
                dim=-1,
            )
        return self.critic(critic_in).squeeze(-1)

    def raw_actor_critic(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TokenizedForwardOut]:
        """Compute unsquashed Gaussian loc, std, critic value, and ``TokenizedForwardOut``."""
        (
            h_glob,
            h_tok,
            tok_pool,
            tok_max,
            morph_desc,
            est_pred,
            _,
            z_mu,
            z_logvar,
            z_gate,
            z_gated,
        ) = self._encode_context(tokens, glob, active_mask)
        B, N, _ = h_tok.shape
        cond = torch.cat((est_pred.detach(), z_gated, morph_desc), dim=-1)[:, None, :].expand(B, N, -1)
        h_tok_in = h_tok.detach() if self.cfg.detach_actor_tokens else h_tok
        act_in = torch.cat((h_tok_in, cond), dim=-1)
        raw_loc = self.actor_head(act_in).squeeze(-1).clamp(-5.0, 5.0)
        mean = torch.tanh(raw_loc)
        if active_mask is not None:
            m = active_mask.to(dtype=mean.dtype)
            raw_loc = raw_loc * m
            mean = mean * m
        std = torch.exp(self.log_std).clamp(self.cfg.min_std, self.cfg.max_std).expand_as(mean)
        value = self._critic_from_ctx(h_glob, tok_pool, tok_max, morph_desc, est_pred, z_gated, priv=priv)
        fwd = TokenizedForwardOut(
            action_mean=mean,
            action_std=std,
            est_pred=est_pred,
            z_mu=z_mu,
            z_logvar=z_logvar,
            z_gate=z_gate,
            z_gated=z_gated,
        )
        return raw_loc, std, value, fwd

    def forward(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> TokenizedForwardOut:
        _, _, _, fwd = self.raw_actor_critic(tokens, glob, active_mask=active_mask, priv=None)
        return fwd

    def evaluate_value(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_glob, _, tok_pool, tok_max, morph_desc, est_pred, _, _, _, _, z_gated = self._encode_context(
            tokens, glob, active_mask
        )
        return self._critic_from_ctx(h_glob, tok_pool, tok_max, morph_desc, est_pred, z_gated, priv=priv)

    def sample_actions(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> ActionSampleOut:
        """Sample (or take deterministic mean) tanh-squashed actions + exact log-prob & critic value."""
        raw_loc, std, value, fwd = self.raw_actor_critic(tokens, glob, active_mask=active_mask, priv=priv)
        if deterministic:
            latent = raw_loc
        else:
            latent = (raw_loc + std * torch.randn_like(raw_loc)).clamp(-5.5, 5.5)
        action = torch.tanh(latent)
        if active_mask is not None:
            m = active_mask.to(dtype=action.dtype)
            latent = latent * m
            action = action * m
        log_prob = _tanh_squashed_log_prob(raw_loc, std, latent, active_mask)
        return ActionSampleOut(
            action=action, latent=latent, log_prob=log_prob, value=value, raw_loc=raw_loc, forward_out=fwd
        )

    def evaluate_actions(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        latent: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
    ) -> ActionEvalOut:
        """Evaluate stored pre-tanh ``latent`` for PPO update."""
        raw_loc, std, value, fwd = self.raw_actor_critic(tokens, glob, active_mask=active_mask, priv=priv)
        log_prob = _tanh_squashed_log_prob(raw_loc, std, latent, active_mask)
        entropy = _mean_active_entropy(std, active_mask)
        return ActionEvalOut(
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            raw_loc=raw_loc,
            std=std,
            forward_out=fwd,
        )

    def compute_aux_losses(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        next_glob_sens: torch.Tensor,  # (B, d_glob - 3)
        est_target: torch.Tensor,      # (B, est_dim)
        active_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute FP32 EstNet + beta-VAE reconstruction + HIM InfoNCE + SwAV auxiliary losses."""
        _, _, tok_pool, _, _, est_pred, z_s, z_mu, z_lv, _, _ = self._encode_context(tokens, glob, active_mask)
        l_est = F.smooth_l1_loss(est_pred.float(), est_target.detach().float(), beta=0.15)
        pred_next = self.next_glob_dec(torch.cat((est_pred, z_s, tok_pool.detach()), dim=-1))
        l_rec = F.mse_loss(pred_next.float(), next_glob_sens.detach().float())
        l_kl = -0.5 * torch.mean(1.0 + z_lv.float() - z_mu.float().pow(2) - z_lv.float().exp())

        # HIM symmetric InfoNCE + 3-step Sinkhorn SwAV
        src = F.normalize(self.him_src(torch.cat((z_s, tok_pool.detach()), dim=-1)).float(), dim=-1, eps=1e-6)
        tgt = F.normalize(self.him_tgt(next_glob_sens.detach().to(dtype=tokens.dtype)).float(), dim=-1, eps=1e-6)
        logits = (src @ tgt.t()) / max(self.cfg.temperature, 1e-3)
        labels = torch.arange(logits.shape[0], device=logits.device)
        l_infonce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        proto = F.normalize(self.prototypes.float(), dim=-1, eps=1e-6)
        s_src = src @ proto.t()
        s_tgt = tgt @ proto.t()
        q_src = sinkhorn_knopp(s_src, self.cfg.sinkhorn_eps, self.cfg.sinkhorn_iters)
        q_tgt = sinkhorn_knopp(s_tgt, self.cfg.sinkhorn_eps, self.cfg.sinkhorn_iters)
        p_src = F.log_softmax(s_src / max(self.cfg.temperature, 1e-3), dim=-1)
        p_tgt = F.log_softmax(s_tgt / max(self.cfg.temperature, 1e-3), dim=-1)
        l_swav = -0.5 * ((q_src * p_tgt).sum(-1).mean() + (q_tgt * p_src).sum(-1).mean())
        total = l_est + 0.5 * (l_rec + self.cfg.beta_kl * l_kl) + 0.25 * l_infonce + 0.05 * l_swav
        return {
            'total_aux': total,
            'l_est': l_est,
            'l_rec': l_rec,
            'l_kl': l_kl,
            'l_infonce': l_infonce,
            'l_swav': l_swav,
        }


# ---------------------------------------------------------------------------
# Universal Finite-Group Reynolds Equivariance Wrapper (Z2, Z2xZ2, C4, D4, ...)
# ---------------------------------------------------------------------------
def _invert_perm_sign(perm: list[int], sign: list[float]) -> tuple[list[int], list[float]]:
    """Compute inverse of signed permutation matrix K_g (so K_g^{-1} K_g = I)."""
    n = len(perm)
    inv_p = [0] * n
    inv_s = [1.0] * n
    for i, p_i in enumerate(perm):
        inv_p[p_i] = i
        inv_s[p_i] = float(sign[i])
    return inv_p, inv_s


def apply_signed_perm(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor, dim: int = -1) -> torch.Tensor:
    y = x.index_select(dim, perm)
    shape = [1] * y.ndim
    shape[dim] = sign.numel()
    return y * sign.reshape(shape)


def default_glob_signs_for_group(group: SymmetryGroupSpec, d_glob: int = 9) -> list[list[float]] | None:
    """Return physically consistent IMU/command sign flips for standard bilateral Z2 groups."""
    if d_glob != 9:
        return None
    name_l = group.name.lower()
    if group.order == 2 and 'z2' in name_l and 'z2xz2' not in name_l:
        # [wx, wy, wz, gx, gy, gz, cmd_vx, cmd_vy_or_yaw, cmd_stop]
        # Under L-R mirror (y -> -y): wx -> -wx, wz -> -wz, gy -> -gy, cmd_y/yaw -> -cmd_y/yaw
        return [
            [1.0] * 9,
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
        ]
    return None


class FiniteGroupEquivariantWrapper(nn.Module):
    """Exact Reynolds group-averaging equivariant wrapper for ANY finite symmetry group G.

    Given any base policy taking actuator tokens ``(B, H, N_act, d_tok)`` + global history
    ``(B, H, d_glob)``, enforces by construction:
        pi_eq(L_g(o)) == K_g(pi_eq(o))     for every g in G (error = 0.0 up to floating-point rounding)
    Works seamlessly across Z2, Z2xZ2, C4, D4, or any custom permutation-sign group, for both
    deployment inference AND end-to-end PPO training (equivariant actor + G-invariant critic).
    """

    def __init__(
        self,
        base_policy: nn.Module,
        group: SymmetryGroupSpec,
        *,
        glob_signs: list[list[float]] | None = None,
        tok_feat_signs: list[list[float]] | None = None,
        saddle_break_tol: float = 0.0,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.group = group
        self.order = group.order
        self.saddle_break_tol = float(saddle_break_tol)

        act_p = torch.tensor(group.act_perms, dtype=torch.long)
        act_s = torch.tensor(group.act_signs, dtype=torch.float32)
        inv_p_list, inv_s_list = [], []
        for p, s in zip(group.act_perms, group.act_signs):
            ip, isg = _invert_perm_sign(p, s)
            inv_p_list.append(ip)
            inv_s_list.append(isg)
        self.register_buffer('act_perms', act_p)
        self.register_buffer('act_signs', act_s)
        self.register_buffer('act_inv_perms', torch.tensor(inv_p_list, dtype=torch.long))
        self.register_buffer('act_inv_signs', torch.tensor(inv_s_list, dtype=torch.float32))

        if group.est_perms:
            est_ip, est_is = [], []
            for p, s in zip(group.est_perms, group.est_signs):
                ip, isg = _invert_perm_sign(p, s)
                est_ip.append(ip)
                est_is.append(isg)
            self.register_buffer('est_inv_perms', torch.tensor(est_ip, dtype=torch.long))
            self.register_buffer('est_inv_signs', torch.tensor(est_is, dtype=torch.float32))
        else:
            self.est_inv_perms = None
            self.est_inv_signs = None

        if glob_signs is not None:
            self.register_buffer('glob_signs', torch.tensor(glob_signs, dtype=torch.float32))
        else:
            self.glob_signs = None

        if tok_feat_signs is not None:
            self.register_buffer('tok_feat_signs', torch.tensor(tok_feat_signs, dtype=torch.float32))
        else:
            self.tok_feat_signs = None

    @property
    def cfg(self) -> Any:
        return getattr(self.base_policy, 'cfg', None)

    @property
    def log_std(self) -> nn.Parameter:
        return self.base_policy.log_std

    def transform_inputs(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        g_idx: int,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply group element L_g to entity tokens (permuting actuator slots & signing antisymmetric features)."""
        perm = self.act_perms[g_idx]
        act_s = self.act_signs[g_idx].to(dtype=tokens.dtype)
        # Permute actuator axis (dim=2 of (B, H, N_act, d_tok))
        t_g = tokens.index_select(2, perm)
        # Signed action/position features (q, dq, q_err, u_prev at indices 4..7) follow act_s
        if t_g.shape[-1] >= 8:
            t_g = t_g.clone()
            t_g[..., 4:8] = t_g[..., 4:8] * act_s[None, None, :, None]
        if self.tok_feat_signs is not None:
            fs = self.tok_feat_signs[g_idx].to(dtype=tokens.dtype)
            t_g = t_g * fs[None, None, None, :]
        g_g = glob * self.glob_signs[g_idx].to(dtype=glob.dtype)[None, None, :] if self.glob_signs is not None else glob
        m_g = active_mask.index_select(1, perm) if active_mask is not None else None
        return t_g, g_g, m_g

    def _orbit_batch(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        t_all, g_all, m_all = [], [], []
        for g_idx in range(self.order):
            tg, gg, mg = self.transform_inputs(tokens, glob, g_idx, active_mask)
            t_all.append(tg)
            g_all.append(gg)
            if mg is not None:
                m_all.append(mg)
        t_cat = torch.cat(t_all, dim=0)
        g_cat = torch.cat(g_all, dim=0)
        m_cat = torch.cat(m_all, dim=0) if active_mask is not None else None
        p_cat = priv.repeat(self.order, 1) if priv is not None else None
        return t_cat, g_cat, m_cat, p_cat

    def update_obs_norm(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
    ) -> None:
        """Update base normalizer over the full group orbit so running stats stay G-invariant."""
        if not hasattr(self.base_policy, 'update_obs_norm'):
            return
        t_cat, g_cat, m_cat, p_cat = self._orbit_batch(tokens, glob, active_mask, priv)
        self.base_policy.update_obs_norm(t_cat, g_cat, active_mask=m_cat, priv=p_cat)

    def _saddle_break_mask(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Optional fixed-point saddle-breaking gate for straight-moving commands (off when tol <= 0)."""
        if self.saddle_break_tol <= 0.0 or self.order <= 1 or glob.shape[-1] < 3:
            return None
        cmd = glob[:, -1, -3:]
        straight_moving = (cmd[:, 2] < 0.5) & (cmd[:, 0].abs() > 1e-3) & (cmd[:, 1].abs() < 1e-3)
        if not bool(straight_moving.any().item()):
            return None
        max_asym = torch.zeros((tokens.shape[0],), device=tokens.device, dtype=tokens.dtype)
        for g_idx in range(1, self.order):
            tg, gg, mg = self.transform_inputs(tokens, glob, g_idx, active_mask)
            if tokens.shape[-1] >= 8:
                # Compare dynamic actuator channels [4:] only; static metadata [0:4] (e.g. side=±1)
                # swaps under left/right permutation when tok_feat_signs is omitted.
                t_diff = (tokens[..., 4:] - tg[..., 4:]).abs()
            else:
                t_diff = (tokens - tg).abs()
            if active_mask is not None:
                t_diff = t_diff * active_mask[:, None, :, None].to(dtype=tokens.dtype)
            asym_g = torch.maximum(t_diff.amax(dim=(1, 2, 3)), (glob - gg).abs().amax(dim=(1, 2)))
            if active_mask is not None and mg is not None:
                asym_g = torch.maximum(asym_g, (active_mask != mg).any(dim=1).to(dtype=tokens.dtype))
            max_asym = torch.maximum(max_asym, asym_g)
        mask = straight_moving & (max_asym < self.saddle_break_tol)
        return mask if bool(mask.any().item()) else None

    def raw_actor_critic(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TokenizedForwardOut]:
        B, _, N, _ = tokens.shape
        t_cat, g_cat, m_cat, p_cat = self._orbit_batch(tokens, glob, active_mask, priv)
        raw_loc_cat, std_cat, val_cat, out = self.base_policy.raw_actor_critic(
            t_cat, g_cat, active_mask=m_cat, priv=p_cat
        )
        locs = raw_loc_cat.reshape(self.order, B, N)
        means = out.action_mean.reshape(self.order, B, N)
        stds = std_cat.reshape(self.order, B, N)
        vals = val_cat.reshape(self.order, B)
        ests = out.est_pred.reshape(self.order, B, -1)

        avg_loc = torch.zeros((B, N), device=tokens.device, dtype=raw_loc_cat.dtype)
        avg_mean = torch.zeros((B, N), device=tokens.device, dtype=out.action_mean.dtype)
        avg_std = torch.zeros((B, N), device=tokens.device, dtype=std_cat.dtype)
        avg_est = torch.zeros_like(ests[0])
        for g_idx in range(self.order):
            ip = self.act_inv_perms[g_idx]
            isg_loc = self.act_inv_signs[g_idx].to(dtype=raw_loc_cat.dtype)
            isg_m = self.act_inv_signs[g_idx].to(dtype=out.action_mean.dtype)
            avg_loc = avg_loc + apply_signed_perm(locs[g_idx], ip, isg_loc, dim=-1)
            avg_mean = avg_mean + apply_signed_perm(means[g_idx], ip, isg_m, dim=-1)
            avg_std = avg_std + stds[g_idx].index_select(-1, ip)
            if self.est_inv_perms is not None:
                avg_est = avg_est + apply_signed_perm(
                    ests[g_idx], self.est_inv_perms[g_idx], self.est_inv_signs[g_idx].to(dtype=ests.dtype), dim=-1
                )
            else:
                avg_est = avg_est + ests[g_idx]

        inv_ord = 1.0 / float(self.order)
        avg_loc = avg_loc * inv_ord
        avg_mean = avg_mean * inv_ord
        avg_std = avg_std * inv_ord
        saddle_mask = self._saddle_break_mask(tokens, glob, active_mask)
        if saddle_mask is not None:
            avg_loc = torch.where(saddle_mask[:, None], locs[0], avg_loc)
            avg_mean = torch.where(saddle_mask[:, None], means[0], avg_mean)
        avg_val = vals.mean(dim=0)
        avg_z_mu = out.z_mu.reshape(self.order, B, -1).mean(dim=0)
        avg_z_lv = out.z_logvar.reshape(self.order, B, -1).mean(dim=0)
        avg_z_gate = out.z_gate.reshape(self.order, B, -1).mean(dim=0)
        avg_z_gated = out.z_gated.reshape(self.order, B, -1).mean(dim=0)
        fwd = TokenizedForwardOut(
            action_mean=avg_mean,
            action_std=avg_std,
            est_pred=avg_est * inv_ord,
            z_mu=avg_z_mu,
            z_logvar=avg_z_lv,
            z_gate=avg_z_gate,
            z_gated=avg_z_gated,
        )
        return avg_loc, avg_std, avg_val, fwd

    def forward(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> TokenizedForwardOut:
        """Batched exact Reynolds average across all |G| group elements in a single parallel pass."""
        if hasattr(self.base_policy, 'raw_actor_critic'):
            _, _, _, fwd = self.raw_actor_critic(tokens, glob, active_mask=active_mask, priv=None)
            return fwd
        B, _, N, _ = tokens.shape
        t_cat, g_cat, m_cat, _ = self._orbit_batch(tokens, glob, active_mask, None)
        out = self.base_policy(t_cat, g_cat, m_cat)
        means = out.action_mean.reshape(self.order, B, N)
        stds = out.action_std.reshape(self.order, B, N)
        ests = out.est_pred.reshape(self.order, B, -1)

        avg_mean = torch.zeros((B, N), device=tokens.device, dtype=out.action_mean.dtype)
        avg_std = torch.zeros((B, N), device=tokens.device, dtype=out.action_std.dtype)
        avg_est = torch.zeros_like(ests[0])
        for g_idx in range(self.order):
            ip = self.act_inv_perms[g_idx]
            isg = self.act_inv_signs[g_idx].to(dtype=out.action_mean.dtype)
            avg_mean = avg_mean + apply_signed_perm(means[g_idx], ip, isg, dim=-1)
            avg_std = avg_std + stds[g_idx].index_select(-1, ip)
            if self.est_inv_perms is not None:
                avg_est = avg_est + apply_signed_perm(
                    ests[g_idx], self.est_inv_perms[g_idx], self.est_inv_signs[g_idx].to(dtype=ests.dtype), dim=-1
                )
            else:
                avg_est = avg_est + ests[g_idx]

        inv_ord = 1.0 / float(self.order)
        return TokenizedForwardOut(
            action_mean=avg_mean * inv_ord,
            action_std=avg_std * inv_ord,
            est_pred=avg_est * inv_ord,
            z_mu=out.z_mu.reshape(self.order, B, -1).mean(dim=0),
            z_logvar=out.z_logvar.reshape(self.order, B, -1).mean(dim=0),
            z_gate=out.z_gate.reshape(self.order, B, -1).mean(dim=0),
            z_gated=out.z_gated.reshape(self.order, B, -1).mean(dim=0),
        )

    def evaluate_value(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = tokens.shape[0]
        t_cat, g_cat, m_cat, p_cat = self._orbit_batch(tokens, glob, active_mask, priv)
        vals = self.base_policy.evaluate_value(t_cat, g_cat, active_mask=m_cat, priv=p_cat)
        return vals.reshape(self.order, B).mean(dim=0)

    def sample_actions(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> ActionSampleOut:
        raw_loc, std, value, fwd = self.raw_actor_critic(tokens, glob, active_mask=active_mask, priv=priv)
        if deterministic:
            latent = raw_loc
        else:
            latent = (raw_loc + std * torch.randn_like(raw_loc)).clamp(-5.5, 5.5)
        action = torch.tanh(latent)
        if active_mask is not None:
            m = active_mask.to(dtype=action.dtype)
            latent = latent * m
            action = action * m
        log_prob = _tanh_squashed_log_prob(raw_loc, std, latent, active_mask)
        return ActionSampleOut(
            action=action, latent=latent, log_prob=log_prob, value=value, raw_loc=raw_loc, forward_out=fwd
        )

    def evaluate_actions(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        latent: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        priv: torch.Tensor | None = None,
    ) -> ActionEvalOut:
        raw_loc, std, value, fwd = self.raw_actor_critic(tokens, glob, active_mask=active_mask, priv=priv)
        log_prob = _tanh_squashed_log_prob(raw_loc, std, latent, active_mask)
        entropy = _mean_active_entropy(std, active_mask)
        return ActionEvalOut(
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            raw_loc=raw_loc,
            std=std,
            forward_out=fwd,
        )

    def compute_aux_losses(
        self,
        tokens: torch.Tensor,
        glob: torch.Tensor,
        next_glob_sens: torch.Tensor,
        est_target: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.base_policy.compute_aux_losses(
            tokens, glob, next_glob_sens=next_glob_sens, est_target=est_target, active_mask=active_mask
        )

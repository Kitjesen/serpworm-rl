"""Observation normalizers and asymmetric Actor-Critic networks."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'RunningNorm',
    'RunningMeanStd',
    'ActorCritic',
    'TemporalActorCritic',
    'ProprioceptiveEstimatorActorCritic',
]


class RunningNorm(nn.Module):
    """Device-resident Welford normalizer persisted inside module state_dict."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-4, min_var: float = 1e-2):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim, dtype=torch.float64))
        self.register_buffer('var', torch.ones(dim, dtype=torch.float64))
        self.register_buffer('count', torch.tensor(eps, dtype=torch.float64))
        self.clip = float(clip)
        self.min_var = float(min_var)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, self.mean.numel()).to(dtype=torch.float64)
        n = float(flat.shape[0])
        if n <= 0:
            return
        bm = flat.mean(0)
        bv = flat.var(0, unbiased=False)
        delta = bm - self.mean
        tot = self.count + n
        self.mean.add_(delta * (n / tot))
        m2 = self.var * self.count + bv * n + delta.square() * (self.count * n / tot)
        self.var.copy_((m2 / tot).clamp_min(self.min_var))
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        inv_std = self.var.clamp_min(1e-8).rsqrt().to(device=x.device, dtype=x.dtype)
        return ((x - mean) * inv_std).clamp(-self.clip, self.clip)


class RunningMeanStd:
    """Standalone Welford normalizer with flat-checkpoint serialization."""

    def __init__(self, size: int, device: str | torch.device = 'cuda', eps: float = 1e-4, clip: float = 10.0):
        self.mean = torch.zeros(size, device=device, dtype=torch.float64)
        self.var = torch.ones(size, device=device, dtype=torch.float64)
        self.count = torch.tensor(eps, device=device, dtype=torch.float64)
        self.clip = float(clip)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        x = batch.reshape(-1, self.mean.numel()).to(dtype=torch.float64)
        if x.numel() == 0:
            return
        batch_count = float(x.shape[0])
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        m2 = self.var * self.count + batch_var * batch_count + delta.square() * (self.count * batch_count / total)
        self.mean.copy_(new_mean)
        self.var.copy_((m2 / total).clamp_min(1e-8))
        self.count.copy_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        inv_std = self.var.clamp_min(1e-8).rsqrt().to(device=x.device, dtype=x.dtype)
        return ((x - mean) * inv_std).clamp(-self.clip, self.clip)

    def state_dict(self) -> dict:
        return {'mean': self.mean.cpu(), 'var': self.var.cpu(), 'count': self.count.cpu(), 'clip': self.clip}

    def load_state_dict(self, state: dict) -> None:
        self.mean.copy_(state['mean'].to(device=self.mean.device, dtype=torch.float64))
        self.var.copy_(state['var'].to(device=self.var.device, dtype=torch.float64))
        self.count.copy_(state['count'].to(device=self.count.device, dtype=torch.float64))
        self.clip = float(state.get('clip', self.clip))


def _tanh_squashed_log_prob(dist: torch.distributions.Normal, latent: torch.Tensor) -> torch.Tensor:
    """Exact log-probability under tanh squashing, computed in FP32 to prevent bf16 KL spikes."""
    lat = latent.float()
    d_f32 = torch.distributions.Normal(dist.loc.float(), dist.scale.float(), validate_args=False)
    log_jacobian = 2.0 * (math.log(2.0) - lat - F.softplus(-2.0 * lat))
    return (d_f32.log_prob(lat) - log_jacobian).sum(-1)


class ActorCritic(nn.Module):
    """MLP asymmetric Actor-Critic with exact stored-latent tanh-squashed log-prob."""

    def __init__(
        self,
        obs_size: int,
        action_size: int = 9,
        initial_std: float = 0.65,
        critic_obs_size: int | None = None,
    ):
        super().__init__()
        critic_in = obs_size if critic_obs_size is None else int(critic_obs_size)
        self.obs_size = int(obs_size)
        self.critic_obs_size = critic_in

        def make_mlp(inp: int, output: int, width: int = 128) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(inp, width),
                nn.Tanh(),
                nn.Linear(width, width),
                nn.Tanh(),
                nn.Linear(width, output),
            )

        self.actor = make_mlp(obs_size, action_size, 128)
        self.critic = make_mlp(critic_in, 1, 256 if critic_in > obs_size else 128)
        if not 0.05 <= initial_std <= 1.5:
            raise ValueError('initial_std must be within [.05,1.5]')
        self.logstd = nn.Parameter(torch.full((action_size,), math.log(initial_std)))
        for network in (self.actor, self.critic):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, 2 ** 0.5)
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        std = self.logstd.clamp(-3.0, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std, validate_args=False)

    @staticmethod
    def log_probability(dist: torch.distributions.Normal, latent: torch.Tensor) -> torch.Tensor:
        return _tanh_squashed_log_prob(dist, latent)

    def value(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(obs if critic_obs is None else critic_obs).squeeze(-1)

    def sample(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None):
        dist = self.distribution(obs)
        latent = dist.rsample()
        return latent.tanh(), latent, self.log_probability(dist, latent), self.value(obs, critic_obs)

    def evaluate_latent(self, obs: torch.Tensor, latent: torch.Tensor, critic_obs: torch.Tensor | None = None):
        dist = self.distribution(obs)
        logp = self.log_probability(dist, latent)
        entropy_mc = -self.log_probability(dist, dist.rsample())
        return logp, entropy_mc, self.value(obs, critic_obs)

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs).tanh()


class _CausalConv1d(nn.Conv1d):
    """Left-padded causal 1D convolution preserving exact nn.Conv1d state_dict keys & output length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self._left_pad = int(kernel_size) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(F.pad(x, (self._left_pad, 0)), self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class TemporalActorCritic(nn.Module):
    """Causal 1D-TCN Actor over [B, H, D_obs] + Asymmetric Privileged Critic over [B, H, D_obs] + [B, D_priv].

    Retains exact stored-latent tanh-squashed log-probability ratios for sound PPO.
    """

    def __init__(
        self,
        frame_size: int,
        history_len: int,
        privileged_size: int,
        action_size: int = 9,
        initial_std: float = 0.60,
    ):
        super().__init__()
        self.frame_size = int(frame_size)
        self.history_len = int(history_len)
        self.privileged_size = int(privileged_size)
        self.action_size = int(action_size)

        self.obs_norm = RunningNorm(frame_size)
        self.priv_norm = RunningNorm(privileged_size)

        def make_tcn() -> nn.Sequential:
            # Causal order-preserving strided 1D-TCN: zero right-side future padding distortion
            # while maintaining identical weight shapes and T_out for all history_len >= 1.
            return nn.Sequential(
                _CausalConv1d(frame_size, 64, kernel_size=3, stride=2),
                nn.SiLU(),
                _CausalConv1d(64, 48, kernel_size=3, stride=2),
                nn.SiLU(),
            )

        self.actor_tcn = make_tcn()
        self.critic_tcn = make_tcn()

        t_out = math.ceil(math.ceil(self.history_len / 2) / 2)
        tcn_flat = 48 * t_out

        # Combine order-preserving temporal slots + latest frame skip (frame_size)
        actor_in = tcn_flat + frame_size
        self.actor_mlp = nn.Sequential(
            nn.Linear(actor_in, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, action_size),
        )

        critic_in = tcn_flat + frame_size + privileged_size
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

        self.logstd = nn.Parameter(torch.full((action_size,), math.log(initial_std)))
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(mod.weight, math.sqrt(2.0))
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.orthogonal_(self.actor_mlp[-1].weight, 0.05)
        nn.init.orthogonal_(self.critic_mlp[-1].weight, 1.0)

    def _encode_seq(self, seq_norm: torch.Tensor, tcn: nn.Module) -> torch.Tensor:
        feat = tcn(seq_norm.transpose(1, 2))  # [B, 48, T_out] (strictly ordered in time!)
        return torch.cat((feat.flatten(1), seq_norm[:, -1, :]), dim=-1)

    def distribution(self, obs_seq: torch.Tensor) -> torch.distributions.Normal:
        seq_n = self.obs_norm(obs_seq)
        mean = self.actor_mlp(self._encode_seq(seq_n, self.actor_tcn))
        std = self.logstd.clamp(-2.8, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std, validate_args=False)

    @staticmethod
    def log_probability(dist: torch.distributions.Normal, latent: torch.Tensor) -> torch.Tensor:
        return _tanh_squashed_log_prob(dist, latent)

    def value(self, obs_seq: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        seq_n = self.obs_norm(obs_seq)
        priv_n = self.priv_norm(priv)
        enc = self._encode_seq(seq_n, self.critic_tcn)
        return self.critic_mlp(torch.cat((enc, priv_n), dim=-1)).squeeze(-1)

    def sample(self, obs_seq: torch.Tensor, priv: torch.Tensor):
        dist = self.distribution(obs_seq)
        latent = dist.rsample()
        return latent.tanh(), latent, self.log_probability(dist, latent), self.value(obs_seq, priv)

    def evaluate_latent(self, obs_seq: torch.Tensor, priv: torch.Tensor, latent: torch.Tensor):
        dist = self.distribution(obs_seq)
        logp = self.log_probability(dist, latent)
        entropy = -self.log_probability(dist, dist.rsample())
        return logp, entropy, self.value(obs_seq, priv)

    def evaluate_latent_with_mean(self, obs_seq: torch.Tensor, priv: torch.Tensor, latent: torch.Tensor):
        """Single-pass evaluation returning (logp, entropy, value, deterministic_mu) for fast symmetry PPO."""
        dist = self.distribution(obs_seq)
        logp = self.log_probability(dist, latent)
        entropy = -self.log_probability(dist, dist.rsample())
        return logp, entropy, self.value(obs_seq, priv), dist.mean.tanh()

    def action(self, obs_seq: torch.Tensor) -> torch.Tensor:
        seq_n = self.obs_norm(obs_seq)
        return self.actor_mlp(self._encode_seq(seq_n, self.actor_tcn)).tanh()


class ProprioceptiveEstimatorActorCritic(nn.Module):
    """Pure-proprioceptive causal 1D-TCN + EstNet Actor with Asymmetric Privileged Critic (v6).

    EstNet predicts physical body velocity & contact/terrain states purely from onboard
    servo/IMU history (zero privileged inputs at deployment). Its detached predictions
    feed the Actor MLP alongside the Actor's own TCN encoding, while supervised MSE/L1
    loss trains EstNet against simulator ground truth during PPO updates.
    """

    def __init__(
        self,
        frame_size: int,
        history_len: int,
        privileged_size: int,
        est_size: int = 7,
        action_size: int = 14,
        initial_std: float = 0.45,
        mean_bound: float = 4.0,
    ):
        super().__init__()
        self.frame_size = int(frame_size)
        self.history_len = int(history_len)
        self.privileged_size = int(privileged_size)
        self.est_size = int(est_size)
        self.action_size = int(action_size)
        self.mean_bound = float(mean_bound)

        self.obs_norm = RunningNorm(frame_size)
        self.priv_norm = RunningNorm(privileged_size)

        def make_tcn(c1: int = 64, c2: int = 48) -> nn.Sequential:
            return nn.Sequential(
                _CausalConv1d(frame_size, c1, kernel_size=3, stride=2),
                nn.SiLU(),
                _CausalConv1d(c1, c2, kernel_size=3, stride=2),
                nn.SiLU(),
            )

        self.est_tcn = make_tcn(64, 48)
        self.actor_tcn = make_tcn(64, 48)
        self.critic_tcn = make_tcn(64, 48)

        t_out = math.ceil(math.ceil(self.history_len / 2) / 2)
        tcn_flat = 48 * t_out

        # Proprioceptive State & Contact Estimator (deployed on real robot; zero privileged inputs)
        self.est_mlp = nn.Sequential(
            nn.Linear(tcn_flat + frame_size, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, self.est_size),
        )

        # Mode-free Actor consumes pure proprioception + detached EstNet prediction
        actor_in = tcn_flat + frame_size + self.est_size
        self.actor_mlp = nn.Sequential(
            nn.Linear(actor_in, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, action_size),
        )

        # Asymmetric Privileged Critic (training only)
        critic_in = tcn_flat + frame_size + privileged_size
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

        self.logstd = nn.Parameter(torch.full((action_size,), math.log(initial_std)))
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(mod.weight, math.sqrt(2.0))
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.orthogonal_(self.est_mlp[-1].weight, 0.25)
        nn.init.orthogonal_(self.actor_mlp[-1].weight, 0.05)
        nn.init.orthogonal_(self.critic_mlp[-1].weight, 1.0)

    def _encode_seq(self, seq_norm: torch.Tensor, tcn: nn.Module) -> torch.Tensor:
        feat = tcn(seq_norm.transpose(1, 2))
        return torch.cat((feat.flatten(1), seq_norm[:, -1, :]), dim=-1)

    def estimate(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Pure-proprioceptive forward prediction of [vx, vy, wz, wall_conf, wall_side, drag, slip]."""
        seq_n = self.obs_norm(obs_seq)
        return self._estimate_from_norm(seq_n)

    def _estimate_from_norm(self, seq_n: torch.Tensor) -> torch.Tensor:
        raw = self.est_mlp(self._encode_seq(seq_n, self.est_tcn))
        # Indices 0..2: unbounded velocity estimates; 3,5,6: [0,1] intensities; 4: [-1,1] signed wall side
        if self.est_size == 7:
            vel = raw[:, :3]
            w_conf = torch.sigmoid(raw[:, 3:4])
            w_side = torch.tanh(raw[:, 4:5])
            drag_slip = torch.sigmoid(raw[:, 5:7])
            return torch.cat((vel, w_conf, w_side, drag_slip), dim=-1)
        return raw

    def _actor_dist_and_est(self, seq_n: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        est_pred = self._estimate_from_norm(seq_n)
        enc = self._encode_seq(seq_n, self.actor_tcn)
        raw_mean = self.actor_mlp(torch.cat((enc, est_pred.detach()), dim=-1))
        mean = self.mean_bound * torch.tanh(raw_mean / self.mean_bound)
        std = self.logstd.clamp(-2.8, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean.float(), std.float(), validate_args=False), est_pred

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
        dist, _ = self._actor_dist_and_est(seq_n)
        latent = dist.rsample()
        priv_n = self.priv_norm(priv)
        val = self.critic_mlp(torch.cat((self._encode_seq(seq_n, self.critic_tcn), priv_n), dim=-1)).squeeze(-1)
        return latent.tanh(), latent, self.log_probability(dist, latent), val

    def evaluate_full(self, obs_seq: torch.Tensor, priv: torch.Tensor, latent: torch.Tensor):
        """Single-pass PPO + EstNet evaluation returning (logp, entropy, value, det_mu, est_pred)."""
        seq_n = self.obs_norm(obs_seq)
        dist, est_pred = self._actor_dist_and_est(seq_n)
        logp = self.log_probability(dist, latent)
        entropy = -self.log_probability(dist, dist.rsample())
        priv_n = self.priv_norm(priv)
        val = self.critic_mlp(torch.cat((self._encode_seq(seq_n, self.critic_tcn), priv_n), dim=-1)).squeeze(-1)
        return logp, entropy, val, dist.mean.tanh(), est_pred

    def evaluate_full_dist(self, obs_seq: torch.Tensor, priv: torch.Tensor, latent: torch.Tensor):
        """Single-pass evaluation returning (logp, entropy, value, raw_loc, std, est_pred) for exact Gaussian KL & loc symmetry."""
        seq_n = self.obs_norm(obs_seq)
        dist, est_pred = self._actor_dist_and_est(seq_n)
        logp = self.log_probability(dist, latent)
        entropy = -self.log_probability(dist, dist.rsample())
        priv_n = self.priv_norm(priv)
        val = self.critic_mlp(torch.cat((self._encode_seq(seq_n, self.critic_tcn), priv_n), dim=-1)).squeeze(-1)
        return logp, entropy, val, dist.loc.float(), dist.scale.float(), est_pred

    def action(self, obs_seq: torch.Tensor) -> torch.Tensor:
        seq_n = self.obs_norm(obs_seq)
        dist, _ = self._actor_dist_and_est(seq_n)
        return dist.mean.tanh()

    def action_with_estimate(self, obs_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_n = self.obs_norm(obs_seq)
        dist, est_pred = self._actor_dist_and_est(seq_n)
        return dist.mean.tanh(), est_pred

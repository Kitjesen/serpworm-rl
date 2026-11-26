"""Lab-bench system identification & actuator calibration toolkit for SerpWorm.

Implements four ready-to-use physical identification routines:
  (a) ``fit_rebound_rayleigh_beta(t, z)``: 240 fps rebound decay envelope + variable-projection
      damped-sinusoid refinement to estimate Rayleigh stiffness damping ``rayleigh_beta`` and ``zeta``.
  (b) ``fit_slope_contact_params(...)``: tilt-board rolling angle + static/kinetic pull gauge
      identification of ``wheel_rolling``, ``wheel_mu``, and Stribeck ``wheel_static_mu_ratio``.
  (c) ``fit_capstan_from_pull_release(t_pull_n, t_release_n)``: bidirectional Euler-Eytelwein
      Capstan friction ``capstan_mu_theta = 0.5 * ln(T_pull / T_release)`` with diagnostics.
  (d) ``ActuatorNet`` & ``train_actuator_net(logs)``: Hwangbo et al. (arXiv:1901.08652) compact
      3-layer actuator dynamics MLP mapping FashionStar bus ``(q_des - q, dq)`` history -> torque.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn

__all__ = [
    'ReboundFitResult',
    'SlopeContactParams',
    'CapstanFitResult',
    'ActuatorTrainResult',
    'ActuatorNet',
    'fit_rebound_rayleigh_beta',
    'fit_slope_contact_params',
    'fit_capstan_from_pull_release',
    'build_actuator_features',
    'train_actuator_net',
]


# ---------------------------------------------------------------------------
# (a) 240 fps Free-Rebound Structural Rayleigh Damping Identification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReboundFitResult:
    """Estimated Rayleigh stiffness damping and modal parameters from a rebound trace."""

    rayleigh_beta: float
    zeta: float
    omega_d_rad_s: float
    omega_n_rad_s: float
    freq_hz: float
    decay_rate_s: float
    amplitude: float
    z_eq: float
    phase_rad: float
    r2: float
    rmse: float
    n_peaks: int
    rayleigh_alpha_assumed: float


def _varproj_solve(tau: np.ndarray, z: np.ndarray, sigma: float, omega_d: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve linear parameters ``[z_eq, c, s]`` for fixed ``(sigma, omega_d)``."""
    env = np.exp(-sigma * tau)
    phi = np.column_stack((np.ones_like(tau), env * np.cos(omega_d * tau), env * np.sin(omega_d * tau)))
    coef, _, _, _ = np.linalg.lstsq(phi, z, rcond=None)
    resid = phi @ coef - z
    return coef, resid


def fit_rebound_rayleigh_beta(
    t: Sequence[float] | np.ndarray,
    z: Sequence[float] | np.ndarray,
    *,
    rayleigh_alpha: float = 0.05,
    max_iter: int = 60,
) -> ReboundFitResult:
    """Fit damped rebound oscillation ``z(t) = z_eq + A exp(-sigma t) cos(omega_d t + phi)``.

    Uses FFT + envelope peak initialization followed by 2D Variable-Projection
    Levenberg-Marquardt refinement to estimate ``omega_d``, ``sigma = zeta * omega_n``,
    damping ratio ``zeta``, and Rayleigh stiffness damping
    ``rayleigh_beta = (2 * zeta * omega_n - rayleigh_alpha) / omega_n**2``.
    """
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    if t_arr.size != z_arr.size or t_arr.size < 10:
        raise ValueError('t and z must be 1D arrays of equal length >= 10')
    if not (np.all(np.isfinite(t_arr)) and np.all(np.isfinite(z_arr))):
        raise ValueError('t and z must contain only finite values')
    dt_diffs = np.diff(t_arr)
    if np.any(dt_diffs <= 0.0):
        raise ValueError('t must be strictly increasing')

    tau = t_arr - t_arr[0]
    duration = float(tau[-1])
    dt_med = float(np.median(dt_diffs))
    fs = 1.0 / dt_med

    # Stage 1: Equilibrium estimate & zero-padded FFT for initial damped frequency omega_d
    tail_start = max(1, int(0.75 * z_arr.size))
    z_eq0 = float(np.mean(z_arr[tail_start:]))
    y = z_arr - z_eq0
    if float(np.std(y)) < 1e-12:
        raise ValueError('Rebound trace z has near-zero oscillation variance')

    n_fft = 1 << int(math.ceil(math.log2(max(4096, 8 * y.size))))
    win = np.hanning(y.size)
    spec = np.abs(np.fft.rfft(y * win, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=dt_med)
    min_bin = max(1, int(np.searchsorted(freqs, 0.5 / max(duration, 1e-3))))
    peak_bin = min_bin + int(np.argmax(spec[min_bin:]))
    f_init = max(float(freqs[peak_bin]), 0.25 / max(duration, 1e-3))
    omega_d = 2.0 * math.pi * f_init

    # Envelope peak detection on |y| for initial decay rate sigma0 & peak count
    abs_y = np.abs(y)
    min_sep = max(1, int(round(0.35 * (0.5 * fs / max(f_init, 1e-3)))))
    raw_peaks = [i for i in range(1, abs_y.size - 1) if abs_y[i] >= abs_y[i - 1] and abs_y[i] > abs_y[i + 1]]
    peaks: list[int] = []
    for idx in raw_peaks:
        if not peaks or (idx - peaks[-1]) >= min_sep:
            peaks.append(idx)
        elif abs_y[idx] > abs_y[peaks[-1]]:
            peaks[-1] = idx

    peak_vals = abs_y[peaks] if peaks else np.array([], dtype=np.float64)
    peak_times = tau[peaks] if peaks else np.array([], dtype=np.float64)
    if peak_vals.size >= 2:
        thresh = 0.08 * float(peak_vals.max())
        keep = peak_vals > max(thresh, 1e-12)
        if int(np.sum(keep)) >= 2:
            pt = peak_times[keep]
            pv = peak_vals[keep]
            w = pv / float(pv.max())
            # Weighted linear regression on log(pv) = a - sigma * pt
            W = w * w
            t_bar = float(np.sum(W * pt) / np.sum(W))
            l_pv = np.log(pv)
            l_bar = float(np.sum(W * l_pv) / np.sum(W))
            denom = float(np.sum(W * (pt - t_bar) ** 2))
            sigma = max(1e-4, -float(np.sum(W * (pt - t_bar) * (l_pv - l_bar)) / max(denom, 1e-12)))
        else:
            sigma = max(0.05, 1.0 / max(duration, 1e-3))
    else:
        sigma = max(0.05, 1.0 / max(duration, 1e-3))
    sigma = min(sigma, 0.85 * omega_d)

    # Stage 2: Variable-Projection Levenberg-Marquardt over (sigma, omega_d)
    coef, resid = _varproj_solve(tau, z_arr, sigma, omega_d)
    cost = 0.5 * float(np.dot(resid, resid))
    lam = 1e-3

    for _ in range(int(max_iter)):
        eps_s = max(1e-6, 1e-5 * max(abs(sigma), 1.0))
        eps_w = max(1e-6, 1e-5 * max(abs(omega_d), 1.0))
        _, r_sp = _varproj_solve(tau, z_arr, sigma + eps_s, omega_d)
        _, r_sm = _varproj_solve(tau, z_arr, max(1e-8, sigma - eps_s), omega_d)
        ds_eff = (sigma + eps_s) - max(1e-8, sigma - eps_s)
        _, r_wp = _varproj_solve(tau, z_arr, sigma, omega_d + eps_w)
        _, r_wm = _varproj_solve(tau, z_arr, sigma, max(1e-6, omega_d - eps_w))
        dw_eff = (omega_d + eps_w) - max(1e-6, omega_d - eps_w)

        J = np.column_stack(((r_sp - r_sm) / ds_eff, (r_wp - r_wm) / dw_eff))
        H = J.T @ J
        g = J.T @ resid
        diag = np.maximum(np.diag(H), 1e-12)
        try:
            step = -np.linalg.solve(H + lam * np.diag(diag), g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        cand_sigma = max(1e-7, float(sigma + step[0]))
        cand_omega = max(1e-4, float(omega_d + step[1]))
        cand_coef, cand_resid = _varproj_solve(tau, z_arr, cand_sigma, cand_omega)
        cand_cost = 0.5 * float(np.dot(cand_resid, cand_resid))

        if cand_cost < cost:
            rel_step = max(abs(cand_sigma - sigma) / max(sigma, 1e-6), abs(cand_omega - omega_d) / max(omega_d, 1e-6))
            sigma, omega_d = cand_sigma, cand_omega
            coef, resid, cost = cand_coef, cand_resid, cand_cost
            lam = max(1e-9, lam * 0.3)
            if rel_step < 1e-10:
                break
        else:
            lam = min(1e9, lam * 5.0)
            if lam >= 1e9:
                break

    z_eq, c_val, s_val = float(coef[0]), float(coef[1]), float(coef[2])
    amplitude = math.hypot(c_val, s_val)
    phase_rad = math.atan2(-s_val, c_val)
    omega_n = math.hypot(omega_d, sigma)
    zeta = sigma / max(omega_n, 1e-12)
    rayleigh_beta = (2.0 * sigma - float(rayleigh_alpha)) / max(omega_n * omega_n, 1e-12)

    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((z_arr - float(np.mean(z_arr))) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-18)
    rmse = math.sqrt(ss_res / float(z_arr.size))

    return ReboundFitResult(
        rayleigh_beta=float(rayleigh_beta),
        zeta=float(zeta),
        omega_d_rad_s=float(omega_d),
        omega_n_rad_s=float(omega_n),
        freq_hz=float(omega_d / (2.0 * math.pi)),
        decay_rate_s=float(sigma),
        amplitude=float(amplitude),
        z_eq=z_eq,
        phase_rad=float(phase_rad),
        r2=float(r2),
        rmse=float(rmse),
        n_peaks=len(peaks),
        rayleigh_alpha_assumed=float(rayleigh_alpha),
    )


# ---------------------------------------------------------------------------
# (b) Tilt-Board & Pull-Gauge Ground Contact Parameter Identification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlopeContactParams:
    """Identified ARPC wheel-ground contact parameters from incline & pull-gauge trials."""

    wheel_rolling: float
    wheel_mu: float
    wheel_static_mu: float
    wheel_static_mu_ratio: float
    normal_force_n: float
    roll_angle_deg_mean: float
    wheel_rolling_std: float
    wheel_mu_std: float
    wheel_static_mu_ratio_std: float
    n_trials: int
    warnings: tuple[str, ...] = ()

    def to_hardware_overrides(self) -> dict[str, float]:
        """Return exact ``HardwareConfig`` keyword overrides."""
        return {
            'wheel_rolling': self.wheel_rolling,
            'wheel_mu': self.wheel_mu,
            'wheel_static_mu_ratio': self.wheel_static_mu_ratio,
        }


def fit_slope_contact_params(
    roll_angle_deg: float | Sequence[float] | np.ndarray,
    static_pull_n: float | Sequence[float] | np.ndarray,
    kinetic_pull_n: float | Sequence[float] | np.ndarray,
    total_mass_kg: float,
    *,
    g: float = 9.80665,
) -> SlopeContactParams:
    """Compute ``wheel_rolling``, ``wheel_mu``, and Stribeck ``wheel_static_mu_ratio``.

    Physics:
      - Incline free-rolling threshold ``theta``: ``wheel_rolling = tan(theta)``.
      - Horizontal backward / locked-ratchet pull force on mass ``m`` (normal load ``N = m*g``):
        ``wheel_mu = F_kinetic / N``, ``wheel_static_mu = F_static / N``,
        ``wheel_static_mu_ratio = F_static / F_kinetic``.
    Accepts either scalar measurements or 1D sequences of repeated bench trials.
    """
    mass = float(total_mass_kg)
    grav = float(g)
    if mass <= 0.0 or grav <= 0.0:
        raise ValueError('total_mass_kg and g must be strictly positive')

    angles = np.asarray(roll_angle_deg, dtype=np.float64).reshape(-1)
    f_stat = np.asarray(static_pull_n, dtype=np.float64).reshape(-1)
    f_kin = np.asarray(kinetic_pull_n, dtype=np.float64).reshape(-1)
    if angles.size == 0 or f_stat.size == 0 or f_kin.size == 0:
        raise ValueError('Measurements cannot be empty')
    if np.any((angles <= 0.0) | (angles >= 85.0)):
        raise ValueError('roll_angle_deg must lie strictly in (0, 85) degrees')
    if np.any(f_stat <= 0.0) or np.any(f_kin <= 0.0):
        raise ValueError('static_pull_n and kinetic_pull_n must be strictly positive')

    normal_n = mass * grav
    rolling_samples = np.tan(np.deg2rad(angles))
    mu_k_samples = f_kin / normal_n
    mu_s_samples = f_stat / normal_n

    warnings_list: list[str] = []
    if f_stat.size == f_kin.size:
        raw_ratio_samples = f_stat / f_kin
    else:
        raw_ratio_samples = np.array([float(np.mean(f_stat) / np.mean(f_kin))], dtype=np.float64)

    if np.any(raw_ratio_samples < 1.0):
        warnings_list.append('Some static_pull_n < kinetic_pull_n; clamping Stribeck ratio >= 1.0')
    ratio_samples = np.maximum(raw_ratio_samples, 1.0)

    wheel_rolling = float(np.mean(rolling_samples))
    wheel_mu = float(np.mean(mu_k_samples))
    wheel_static_mu = float(np.mean(mu_s_samples))
    wheel_static_mu_ratio = float(max(1.0, np.mean(ratio_samples)))

    return SlopeContactParams(
        wheel_rolling=wheel_rolling,
        wheel_mu=wheel_mu,
        wheel_static_mu=wheel_static_mu,
        wheel_static_mu_ratio=wheel_static_mu_ratio,
        normal_force_n=normal_n,
        roll_angle_deg_mean=float(np.mean(angles)),
        wheel_rolling_std=float(np.std(rolling_samples, ddof=1)) if rolling_samples.size > 1 else 0.0,
        wheel_mu_std=float(np.std(mu_k_samples, ddof=1)) if mu_k_samples.size > 1 else 0.0,
        wheel_static_mu_ratio_std=float(np.std(ratio_samples, ddof=1)) if ratio_samples.size > 1 else 0.0,
        n_trials=int(max(angles.size, f_stat.size, f_kin.size)),
        warnings=tuple(warnings_list),
    )


# ---------------------------------------------------------------------------
# (c) Bidirectional Pull/Release Capstan Tendon Friction Identification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapstanFitResult:
    """Estimated tendon guide-hole Capstan friction ``mu * theta`` with diagnostics."""

    capstan_mu_theta: float
    capstan_mu_theta_ols: float
    capstan_mu_theta_std: float
    capstan_mu_theta_median: float
    transmission_efficiency: float
    intrinsic_tension_mean_n: float
    r2_ratio: float
    relative_spread: float
    n_valid: int
    n_rejected: int
    quality: Literal['good', 'fair', 'poor']
    warnings: tuple[str, ...] = ()


def fit_capstan_from_pull_release(
    t_pull_n: float | Sequence[float] | np.ndarray,
    t_release_n: float | Sequence[float] | np.ndarray,
) -> CapstanFitResult:
    """Compute ``capstan_mu_theta = 0.5 * ln(T_pull / T_release)`` with quality diagnostics.

    From Euler-Eytelwein bidirectional tendon hysteresis at matched displacement:
      ``T_pull = T_0 * exp(+mu * theta)``, ``T_release = T_0 * exp(-mu * theta)``
      => ``mu * theta = 0.5 * ln(T_pull / T_release)``, ``T_0 = sqrt(T_pull * T_release)``.
    """
    tp = np.asarray(t_pull_n, dtype=np.float64).reshape(-1)
    tr = np.asarray(t_release_n, dtype=np.float64).reshape(-1)
    if tp.size != tr.size or tp.size == 0:
        raise ValueError('t_pull_n and t_release_n must be non-empty arrays of equal length')

    finite = np.isfinite(tp) & np.isfinite(tr)
    pos = finite & (tp > 0.0) & (tr > 0.0)
    physical = pos & (tp >= tr)

    warnings_list: list[str] = []
    n_rejected = int(tp.size - np.sum(physical))
    if not np.any(physical):
        raise ValueError('No valid pairs with T_pull >= T_release > 0')
    if n_rejected > 0:
        warnings_list.append(f'Rejected {n_rejected}/{tp.size} non-physical or inverted pairs (T_pull < T_release or <= 0)')

    tp_v = tp[physical]
    tr_v = tr[physical]
    mu_samples = 0.5 * np.log(tp_v / tr_v)
    t0_samples = np.sqrt(tp_v * tr_v)

    mu_mean = float(np.mean(mu_samples))
    mu_med = float(np.median(mu_samples))
    mu_std = float(np.std(mu_samples, ddof=1)) if mu_samples.size > 1 else 0.0

    # Zero-intercept OLS ratio T_pull = k * T_release -> k = exp(2 * mu * theta)
    slope = float(np.dot(tr_v, tp_v) / max(float(np.dot(tr_v, tr_v)), 1e-18))
    mu_ols = float(0.5 * math.log(max(slope, 1.0)))

    if tp_v.size >= 2 and float(np.var(tp_v)) > 1e-12:
        pred_tp = slope * tr_v
        ss_res = float(np.sum((tp_v - pred_tp) ** 2))
        ss_tot = float(np.sum((tp_v - float(np.mean(tp_v))) ** 2))
        r2_ratio = float(1.0 - ss_res / max(ss_tot, 1e-18))
    else:
        r2_ratio = 1.0

    rel_spread = mu_std / max(mu_mean, 1e-6)
    if mu_mean > 0.65:
        warnings_list.append(f'Unusually high capstan_mu_theta={mu_mean:.3f}; check eyelet snagging or wrap count')

    if n_rejected == 0 and r2_ratio >= 0.95 and rel_spread <= 0.15:
        quality: Literal['good', 'fair', 'poor'] = 'good'
    elif r2_ratio >= 0.80 and rel_spread <= 0.35:
        quality = 'fair'
    else:
        quality = 'poor'
        warnings_list.append(f'Low Capstan fit consistency (R^2={r2_ratio:.3f}, rel_spread={rel_spread:.2f})')

    return CapstanFitResult(
        capstan_mu_theta=mu_mean,
        capstan_mu_theta_ols=mu_ols,
        capstan_mu_theta_std=mu_std,
        capstan_mu_theta_median=mu_med,
        transmission_efficiency=float(math.exp(-mu_mean)),
        intrinsic_tension_mean_n=float(np.mean(t0_samples)),
        r2_ratio=r2_ratio,
        relative_spread=float(rel_spread),
        n_valid=int(tp_v.size),
        n_rejected=n_rejected,
        quality=quality,
        warnings=tuple(warnings_list),
    )


# ---------------------------------------------------------------------------
# (d) Hwangbo et al. (1901.08652) Compact 3-Layer ActuatorNet + Trainer
# ---------------------------------------------------------------------------

class ActuatorNet(nn.Module):
    """Compact 3-layer MLP (Hwangbo et al., arXiv:1901.08652) predicting servo torque from ``(q_err, dq)`` history."""

    def __init__(
        self,
        history_taps: int = 3,
        hidden_dim: int = 32,
        activation: Literal['softsign', 'silu', 'tanh'] = 'softsign',
        torque_limit_nm: float | None = None,
    ):
        super().__init__()
        self.history_taps = int(history_taps)
        self.in_dim = 2 * self.history_taps
        self.hidden_dim = int(hidden_dim)
        self.torque_limit_nm = float(torque_limit_nm) if torque_limit_nm is not None else None

        def _act() -> nn.Module:
            if activation == 'softsign':
                return nn.Softsign()
            if activation == 'silu':
                return nn.SiLU()
            if activation == 'tanh':
                return nn.Tanh()
            raise ValueError(f'Unknown activation: {activation!r}')

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            _act(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            _act(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer('in_mean', torch.zeros(self.in_dim, dtype=torch.float32))
        self.register_buffer('in_std', torch.ones(self.in_dim, dtype=torch.float32))
        self.register_buffer('out_mean', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('out_std', torch.ones(1, dtype=torch.float32))

    def set_normalizers(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        with torch.no_grad():
            self.in_mean.copy_(x_train.mean(dim=0))
            self.in_std.copy_(x_train.std(dim=0, unbiased=False).clamp_min(1e-4))
            self.out_mean.copy_(y_train.mean(dim=0))
            self.out_std.copy_(y_train.std(dim=0, unbiased=False).clamp_min(1e-4))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict torque (Nm) from ``[B, 2*history_taps]`` ordered ``[e(t), ..., dq(t), ...]``."""
        x_norm = (features - self.in_mean) / self.in_std
        pred = self.net(x_norm) * self.out_std + self.out_mean
        if self.torque_limit_nm is not None:
            pred = pred.clamp(-self.torque_limit_nm, self.torque_limit_nm)
        return pred


@dataclass(frozen=True)
class ActuatorTrainResult:
    """Training and validation summary for ``ActuatorNet`` system identification."""

    train_rmse_nm: float
    val_rmse_nm: float
    val_mae_nm: float
    val_r2: float
    n_train: int
    n_val: int
    epochs: int
    tap_strides: tuple[int, ...]
    loss_history: tuple[float, ...]


def build_actuator_features(
    logs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    tap_strides: Sequence[int] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Extract supervised ``(X, y)`` pairs from single or multi-episode FashionStar bus logs.

    Each episode mapping must contain ``dq`` and ``torque`` plus either ``q_err``
    or both ``q_des`` and ``q`` (shapes ``[T]`` or ``[T, N_joints]``).
    """
    strides = tuple(int(s) for s in tap_strides)
    if not strides or min(strides) < 0:
        raise ValueError('tap_strides must be non-empty non-negative integers')
    max_lag = max(strides)

    episodes = [logs] if isinstance(logs, Mapping) else list(logs)
    if not episodes:
        raise ValueError('logs cannot be empty')

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for ep in episodes:
        if 'q_err' in ep:
            q_err = np.asarray(ep['q_err'], dtype=np.float32)
        elif 'q_des' in ep and 'q' in ep:
            q_err = np.asarray(ep['q_des'], dtype=np.float32) - np.asarray(ep['q'], dtype=np.float32)
        else:
            raise KeyError("Each log episode must provide 'q_err' or both 'q_des' and 'q'")
        dq = np.asarray(ep['dq'], dtype=np.float32)
        torque = np.asarray(ep['torque'], dtype=np.float32)

        if q_err.ndim == 1:
            q_err = q_err[:, None]
            dq = dq[:, None]
            torque = torque[:, None]
        if q_err.shape != dq.shape or q_err.shape != torque.shape:
            raise ValueError(f'Shape mismatch in log episode: q_err={q_err.shape}, dq={dq.shape}, torque={torque.shape}')

        T, J = q_err.shape
        if T <= max_lag:
            continue
        idx = np.arange(max_lag, T)
        err_cols = [q_err[idx - lag, :] for lag in strides]  # each [T', J]
        dq_cols = [dq[idx - lag, :] for lag in strides]
        feat = np.stack(err_cols + dq_cols, axis=-1).reshape(-1, 2 * len(strides))
        tgt = torque[idx, :].reshape(-1, 1)
        xs.append(feat)
        ys.append(tgt)

    if not xs:
        raise ValueError(f'No valid samples extracted (all episodes shorter than max_lag={max_lag})')
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train_actuator_net(
    logs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    tap_strides: Sequence[int] = (0, 1, 2),
    hidden_dim: int = 32,
    activation: Literal['softsign', 'silu', 'tanh'] = 'softsign',
    torque_limit_nm: float | None = None,
    epochs: int = 120,
    batch_size: int = 256,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    seed: int = 7,
    device: str = 'cpu',
) -> tuple[ActuatorNet, ActuatorTrainResult]:
    """Fit ``ActuatorNet`` to FashionStar servo bus logs ``(q_des - q, dq)`` history -> torque."""
    X_np, y_np = build_actuator_features(logs, tap_strides=tap_strides)
    n_total = X_np.shape[0]
    if n_total < 16:
        raise ValueError(f'Need at least 16 samples to train ActuatorNet, got {n_total}')

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_total)
    n_val = max(4, min(n_total - 8, int(round(n_total * float(val_fraction)))))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    dev = torch.device(device)
    x_tr = torch.from_numpy(X_np[tr_idx]).to(dev)
    y_tr = torch.from_numpy(y_np[tr_idx]).to(dev)
    x_va = torch.from_numpy(X_np[val_idx]).to(dev)
    y_va = torch.from_numpy(y_np[val_idx]).to(dev)

    torch.manual_seed(int(seed))
    model = ActuatorNet(
        history_taps=len(tap_strides),
        hidden_dim=hidden_dim,
        activation=activation,
        torque_limit_nm=torque_limit_nm,
    ).to(dev)
    model.set_normalizers(x_tr, y_tr)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=float(lr) * 0.05)

    n_tr = x_tr.shape[0]
    bs = max(16, min(int(batch_size), n_tr))
    loss_hist: list[float] = []
    gen = torch.Generator(device=dev).manual_seed(int(seed) + 1)

    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(n_tr, generator=gen, device=dev)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_tr, bs):
            b_idx = order[start:start + bs]
            pred = model(x_tr[b_idx])
            loss = torch.mean((pred - y_tr[b_idx]).square())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        sched.step()
        loss_hist.append(ep_loss / max(1, n_batches))

    model.eval()
    with torch.no_grad():
        tr_pred = model(x_tr)
        va_pred = model(x_va)
        tr_rmse = float(torch.sqrt(torch.mean((tr_pred - y_tr).square())).item())
        va_err = va_pred - y_va
        va_rmse = float(torch.sqrt(torch.mean(va_err.square())).item())
        va_mae = float(torch.mean(va_err.abs()).item())
        ss_res = float(torch.sum(va_err.square()).item())
        ss_tot = float(torch.sum((y_va - y_va.mean()).square()).item())
        va_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return model, ActuatorTrainResult(
        train_rmse_nm=tr_rmse,
        val_rmse_nm=va_rmse,
        val_mae_nm=va_mae,
        val_r2=float(va_r2),
        n_train=int(n_tr),
        n_val=int(n_val),
        epochs=int(epochs),
        tap_strides=tuple(int(s) for s in tap_strides),
        loss_history=tuple(loss_hist),
    )

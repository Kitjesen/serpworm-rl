"""End-to-end Pure-Proprioceptive Autonomous Phase-Transition PPO + EstNet Trainer (v6).

Trains a single mode-free ProprioceptiveEstimatorActorCritic on real 40-strip FP64
UTM-calibrated steel physics across open hard floor, narrow pipes (entry + exit probing),
carpet, and soft soil. Supports exact ablation switch ``--include-q-err`` (51D vs 65D).
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F

from serpworm.config import DomainRandConfig, HardwareConfig
from serpworm.envs.autonomous_env import AutonomousPhaseEnv, AutonomousTaskConfig, mirror_autonomous_batch
from serpworm.envs.rewards import AutonomousRewardConfig
from serpworm.rl.models import ProprioceptiveEstimatorActorCritic
from serpworm.rl.ppo import advantages, write_json

__all__ = ['AutonomousTrainConfig', 'save_v6_checkpoint', 'train_autonomous', 'main']


@dataclass
class AutonomousTrainConfig:
    hardware: HardwareConfig = field(default_factory=lambda: HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard'))
    task: AutonomousTaskConfig = field(default_factory=AutonomousTaskConfig)
    dr: DomainRandConfig = field(default_factory=DomainRandConfig)
    reward: AutonomousRewardConfig = field(default_factory=AutonomousRewardConfig)

    envs: int = 1024
    horizon: int = 48
    rollouts: int = 200
    epochs: int = 5
    minibatches: int = 24
    lr: float = 5e-4
    min_lr: float = 6e-5
    max_lr: float = 1.0e-3
    target_kl: float = 0.015
    gamma: float = 0.994
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    est_coef: float = 1.0
    entropy_coef: float = 0.003
    max_grad_norm: float = 0.8
    initial_std: float = 0.45
    stage_std_floor: float = 0.22
    seed: int = 7
    plate_solver: str = 'fused_cuda'
    compile_model: bool = False
    amp_bf16: bool = True
    symmetry_aug: bool = True
    stage2_rollout: int = 30
    stage3_rollout: int = 115
    eval_every: int = 25


def save_v6_checkpoint(
    path: Path,
    model: ProprioceptiveEstimatorActorCritic,
    optimizer: torch.optim.Optimizer,
    cfg: AutonomousTrainConfig,
    rollout: int,
    samples: int,
    stage: int,
    metrics: dict,
) -> None:
    payload = {
        'schema': 'serpworm_autonomous_v8' if cfg.task.diverse_v8 else ('serpworm_autonomous_v7' if cfg.task.curved_pipes else 'serpworm_autonomous_v6'),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'frame_size': model.frame_size,
        'history_len': model.history_len,
        'privileged_size': model.privileged_size,
        'est_size': model.est_size,
        'action_size': model.action_size,
        'include_q_err': bool(cfg.task.include_q_err),
        'hardware': cfg.hardware.to_dict(),
        'task': asdict(cfg.task),
        'dr': asdict(cfg.dr),
        'reward': asdict(cfg.reward),
        'rollout': int(rollout),
        'samples': int(samples),
        'stage': int(stage),
        'metrics': metrics,
    }
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_warm_start(model: ProprioceptiveEstimatorActorCritic, ckpt_path: Path, device: str, initial_std: float) -> None:
    """Warm-start Actor, EstNet, obs_norm, and compatible Critic prefix from an earlier checkpoint (e.g. 102D -> 108D)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    src = ckpt['model']
    dst = model.state_dict()
    loaded, padded = 0, 0
    with torch.no_grad():
        for k, v in src.items():
            if k not in dst:
                continue
            if k == 'logstd':
                dst[k].fill_(math.log(max(float(initial_std), 0.05)))
                continue
            if dst[k].shape == v.shape:
                dst[k].copy_(v)
                loaded += 1
            elif k in ('priv_norm.mean', 'priv_norm.var') and v.ndim == 1 and dst[k].shape[0] > v.shape[0]:
                dst[k][:v.shape[0]].copy_(v)
                padded += 1
            elif k == 'critic_mlp.0.weight' and v.ndim == 2 and dst[k].shape[0] == v.shape[0] and dst[k].shape[1] > v.shape[1]:
                dst[k].zero_()
                dst[k][:, :v.shape[1]].copy_(v)
                padded += 1
    model.load_state_dict(dst)
    print(f"[warm-start] loaded {loaded} tensors (+{padded} expanded critic/norm tensors) from {ckpt_path}", flush=True)


def train_autonomous(
    cfg: AutonomousTrainConfig,
    out_dir: Path,
    resume: Path | None = None,
    init_from: Path | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = AutonomousPhaseEnv(
        n=cfg.envs,
        hardware=cfg.hardware,
        task=cfg.task,
        dr=cfg.dr,
        reward_config=cfg.reward,
        device=device,
        plate_solver=cfg.plate_solver,
        seed=cfg.seed,
        noise=True,
    )
    model = ProprioceptiveEstimatorActorCritic(
        frame_size=env.frame_size,
        history_len=env.history_len,
        privileged_size=env.privileged_size,
        est_size=env.est_size,
        action_size=env.action_size,
        initial_std=cfg.initial_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    start_rollout = 0
    total_samples = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_rollout = int(ckpt.get('rollout', 0))
        total_samples = int(ckpt.get('samples', 0))
        env.set_stage(int(ckpt.get('stage', cfg.task.stage)))
    elif init_from is not None:
        _load_warm_start(model, init_from, device, cfg.initial_std)

    fast_model = torch.compile(model) if (cfg.compile_model and device == 'cuda') else model
    amp_ctx = (
        (lambda: torch.amp.autocast('cuda', dtype=torch.bfloat16))
        if (cfg.amp_bf16 and device == 'cuda')
        else nullcontext
    )

    write_json(out / 'config.json', {
        'schema': 'serpworm_autonomous_v8' if cfg.task.diverse_v8 else ('serpworm_autonomous_v7' if cfg.task.curved_pipes else 'serpworm_autonomous_v6'),
        'include_q_err': bool(cfg.task.include_q_err),
        'hardware': cfg.hardware.to_dict(),
        'task': asdict(cfg.task),
        'dr': asdict(cfg.dr),
        'reward': asdict(cfg.reward),
        'train': {k: v for k, v in asdict(cfg).items() if k not in ('hardware', 'task', 'dr', 'reward')},
        'frame_size': env.frame_size,
        'privileged_size': env.privileged_size,
        'est_size': env.est_size,
        'action_size': env.action_size,
        'effective_plate_solver': env.c.plate_solver,
        'cuda_graph_steps': env.c.graph_steps,
        'device': device,
        'gpu': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
    })

    H, N = cfg.horizon, cfg.envs
    obs_buf = torch.empty((H, N, env.history_len, env.frame_size), device=device, dtype=torch.float32)
    priv_buf = torch.empty((H, N, env.privileged_size), device=device, dtype=torch.float32)
    est_buf = torch.empty((H, N, env.est_size), device=device, dtype=torch.float32)
    lat_buf = torch.empty((H, N, env.action_size), device=device, dtype=torch.float32)
    rew_buf = torch.empty((H, N), device=device, dtype=torch.float32)
    val_buf = torch.empty((H, N), device=device, dtype=torch.float32)
    nval_buf = torch.empty((H, N), device=device, dtype=torch.float32)
    term_buf = torch.empty((H, N), device=device, dtype=torch.bool)
    trunc_buf = torch.empty((H, N), device=device, dtype=torch.bool)

    guard_names = ('joint_limit', 'tilt', 'height', 'wheel_clearance', 'nonfinite')
    sym_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 10, 11, 12, 13], device=device, dtype=torch.long)
    sym_sign = torch.tensor([1.0] * 10 + [-1.0] * 4, device=device, dtype=torch.float32)
    est_sym_sign = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    logstd_ceil = max(math.log(0.48), math.log(max(cfg.initial_std, 0.05)))
    vx_scale_mm = max(50.0, cfg.task.vx_max_mm_s)

    obs, priv, est_tgt = env.obs, env.critic_obs, env.est_target
    current_lr = float(cfg.lr)
    best_stage = env.stage
    best_reward = -1e9
    history_log = []
    t_start = time.monotonic()

    for rollout in range(start_rollout + 1, start_rollout + cfg.rollouts + 1):
        prev_stage = env.stage
        if cfg.stage3_rollout > 0 and rollout >= cfg.stage3_rollout and env.stage < 3:
            ramp_n = int(getattr(cfg.dr, 'dr_ramp_rollouts', 20))
            env.set_stage(3, dr_scale=(1.0 / max(1, ramp_n)) if ramp_n > 0 else 1.0)
        elif cfg.stage2_rollout > 0 and rollout >= cfg.stage2_rollout and env.stage < 2:
            env.set_stage(2)
        if env.stage > prev_stage and cfg.stage_std_floor > 0.0:
            model.logstd.data.clamp_min_(math.log(cfg.stage_std_floor))
        if env.stage >= 3:
            ramp_n = int(getattr(cfg.dr, 'dr_ramp_rollouts', 20))
            if ramp_n > 0:
                s3_origin = cfg.stage3_rollout if cfg.stage3_rollout > 0 else (start_rollout + 1)
                env.set_dr_scale(min(1.0, max(1, rollout - s3_origin + 1) / float(ramp_n)))
            else:
                env.set_dr_scale(1.0)
        if env.stage > best_stage:
            best_stage = env.stage
            best_reward = -1e9

        progress = min(1.0, max(0.0, (rollout - start_rollout - 1) / max(1, cfg.rollouts - 1)))
        ent_floor = 0.16 * cfg.entropy_coef
        ent_coef = ent_floor + 0.5 * (cfg.entropy_coef - ent_floor) * (1.0 + math.cos(math.pi * progress))

        t0 = time.monotonic()
        # Telemetry buckets:
        #   v6: 4 buckets (open, pipe, post_pipe, soil)
        #   v7 curved_pipes: 6 buckets (open, straight, bend90, scurve, L_duct, soil)
        #   v8 diverse_v8: 12 buckets (open, str, b90, scv, L90, U180, tap, Z2L, W3, grv, grs, soil)
        if cfg.task.diverse_v8:
            b_names = ('open', 'str', 'b90', 'scv', 'L90', 'U180', 'tap', 'Z2L', 'W3', 'grv', 'grs', 'soil')
        elif cfg.task.curved_pipes:
            b_names = ('open', 'straight', 'bend90', 'scurve', 'L_duct', 'soil')
        else:
            b_names = ('open', 'pipe', 'post_pipe', 'soil')
        nb = len(b_names)
        b_cnt = torch.zeros(nb, device=device, dtype=torch.float64)
        b_vx = torch.zeros(nb, device=device, dtype=torch.float64)
        b_yaw = torch.zeros(nb, device=device, dtype=torch.float64)
        b_ten = torch.zeros(nb, device=device, dtype=torch.float64)
        b_wall = torch.zeros(nb, device=device, dtype=torch.float64)
        b_slip = torch.zeros(nb, device=device, dtype=torch.float64)
        min_strip_z_roll = 1.0
        guard_totals = torch.zeros(len(guard_names), device=device, dtype=torch.long)

        with torch.no_grad():
            for t in range(H):
                model.obs_norm.update(obs[:, -1, :])
                model.priv_norm.update(priv)
                act, latent, _lp, val = fast_model.sample(obs, priv)
                nobs, npriv, nest, rew, term, trunc, tobs, tpriv, info = env.step(act)
                nv = fast_model.value(tobs, tpriv)

                obs_buf[t] = obs
                priv_buf[t] = priv
                est_buf[t] = est_tgt
                lat_buf[t] = latent
                rew_buf[t] = rew
                val_buf[t] = val
                nval_buf[t] = nv
                term_buf[t] = term
                trunc_buf[t] = trunc

                obs, priv, est_tgt = nobs, npriv, nest

                moving = info['cmd'][:, 2] < 0.5
                in_pipe = info['in_pipe']
                on_soil = (info['terrain_id'] == 2) & (~in_pipe)
                if cfg.task.diverse_v8:
                    pt = info['pipe_type']
                    open_hard = (info['terrain_id'] == 0) & (~in_pipe)
                    on_grv = (info['terrain_id'] == 3) & (~in_pipe)
                    on_grs = (info['terrain_id'] == 4) & (~in_pipe)
                    masks = (
                        open_hard & moving,
                        in_pipe & (pt == 1) & moving,
                        in_pipe & (pt == 2) & moving,
                        in_pipe & (pt == 3) & moving,
                        in_pipe & (pt == 4) & moving,
                        in_pipe & (pt == 5) & moving,
                        in_pipe & (pt == 6) & moving,
                        in_pipe & (pt == 7) & moving,
                        in_pipe & (pt == 8) & moving,
                        on_grv & moving,
                        on_grs & moving,
                        on_soil & moving,
                    )
                    if 'min_strip_z_m' in info:
                        min_strip_z_roll = min(min_strip_z_roll, float(info['min_strip_z_m'].min().item()))
                elif cfg.task.curved_pipes:
                    pt = info['pipe_type']
                    open_hard = (info['terrain_id'] == 0) & (~in_pipe)
                    masks = (
                        open_hard & moving,
                        in_pipe & (pt == 1) & moving,
                        in_pipe & (pt == 2) & moving,
                        in_pipe & (pt == 3) & moving,
                        in_pipe & (pt == 4) & moving,
                        on_soil & moving,
                    )
                    if 'min_strip_z_m' in info:
                        min_strip_z_roll = min(min_strip_z_roll, float(info['min_strip_z_m'].min().item()))
                else:
                    post_pipe = info['post_pipe'] & (~in_pipe)
                    open_hard = (info['terrain_id'] == 0) & (~in_pipe) & (~post_pipe)
                    masks = (open_hard & moving, in_pipe & moving, post_pipe & moving, on_soil & moving)
                arc_buckets = (2, 3, 4, 5, 6, 7, 8) if cfg.task.diverse_v8 else (2, 3, 4)
                for bi, mk in enumerate(masks):
                    mf = mk.to(torch.float64)
                    b_cnt[bi].add_(mf.sum())
                    use_arc = (cfg.task.diverse_v8 or cfg.task.curved_pipes) and (bi in arc_buckets) and ('arc_vel' in info)
                    vel_metric = info['arc_vel'] if use_arc else info['body_vel'][:, 0]
                    b_vx[bi].add_((vel_metric * mf).sum())
                    b_yaw[bi].add_((info['yaw_rms_deg'] * mf).sum())
                    b_ten[bi].add_((info['ten_act'] * mf).sum())
                    b_wall[bi].add_((info['wall_mean_n'] * mf).sum())
                    b_slip[bi].add_((info['slip'] * mf).sum())
                for gi, gname in enumerate(guard_names):
                    gmask = env.last_guard_reasons.get(gname)
                    if gmask is not None:
                        guard_totals[gi].add_(gmask.sum())

            adv, ret = advantages(rew_buf, val_buf, nval_buf, term_buf, trunc_buf, cfg.gamma, cfg.gae_lambda)
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)

        if device == 'cuda':
            torch.cuda.synchronize()
        t_sim = time.monotonic() - t0
        total_samples += H * N

        base_b = H * N
        f_obs = obs_buf.reshape(base_b, env.history_len, env.frame_size)
        f_priv = priv_buf.reshape(base_b, env.privileged_size)
        f_est = est_buf.reshape(base_b, env.est_size)
        f_lat = lat_buf.reshape(base_b, env.action_size)
        f_adv = adv.reshape(base_b)
        f_ret = ret.reshape(base_b)

        if cfg.symmetry_aug:
            with torch.no_grad():
                m_obs, m_priv, m_est, m_lat = mirror_autonomous_batch(
                    f_obs, f_priv, f_est, f_lat, include_q_err=cfg.task.include_q_err
                )
                model.obs_norm.update(m_obs[:, -1, :])
                model.priv_norm.update(m_priv)
            f_obs = torch.cat((f_obs, m_obs), dim=0)
            f_priv = torch.cat((f_priv, m_priv), dim=0)
            f_est = torch.cat((f_est, m_est), dim=0)
            f_lat = torch.cat((f_lat, m_lat), dim=0)
            f_adv = torch.cat((f_adv, f_adv), dim=0)
            f_ret = torch.cat((f_ret, f_ret), dim=0)

        total_b = f_obs.shape[0]
        mb_size = max(64, total_b // cfg.minibatches)
        t_opt0 = time.monotonic()

        with torch.no_grad():
            f_logp = torch.empty(total_b, device=device, dtype=torch.float32)
            f_val = torch.empty(total_b, device=device, dtype=torch.float32)
            f_loc = torch.empty((base_b, env.action_size), device=device, dtype=torch.float32)
            f_std = torch.empty((base_b, env.action_size), device=device, dtype=torch.float32)
            for start in range(0, total_b, mb_size):
                end = min(total_b, start + mb_size)
                with amp_ctx():
                    lp_ref, _, v_ref, loc_ref, std_ref, _ = fast_model.evaluate_full_dist(
                        f_obs[start:end], f_priv[start:end], f_lat[start:end]
                    )
                f_logp[start:end] = lp_ref.float()
                f_val[start:end] = v_ref.float()
                if start < base_b:
                    o_end = min(base_b, end)
                    f_loc[start:o_end] = loc_ref[:o_end - start]
                    f_std[start:o_end] = std_ref[:o_end - start]

        opt_stats = torch.zeros(8, device=device, dtype=torch.float64)
        opt_steps = 0
        half_mb = max(1, min(base_b, mb_size // 2))
        stop_early = False
        for ep in range(cfg.epochs):
            if cfg.symmetry_aug:
                perm = torch.randperm(base_b, device=device)
                iter_ranges = range(0, base_b, half_mb)
            else:
                perm = torch.randperm(total_b, device=device)
                iter_ranges = range(0, total_b, mb_size)

            ep_kl_sum = 0.0
            ep_kl_n = 0
            for mb_idx, start in enumerate(iter_ranges):
                if cfg.symmetry_aug:
                    sub = perm[start:start + half_mb]
                    k_sub = sub.shape[0]
                    idx = torch.cat((sub, sub + base_b), dim=0)
                else:
                    sub = perm[start:start + mb_size]
                    idx = sub
                    k_sub = sub.shape[0]

                with amp_ctx():
                    new_logp_r, ent_r, new_val_r, loc_r, std_r, est_r = fast_model.evaluate_full_dist(
                        f_obs[idx], f_priv[idx], f_lat[idx]
                    )

                new_logp = new_logp_r.float()
                ent = ent_r.float()
                new_val = new_val_r.float()
                loc = loc_r.float()
                std = std_r.float()
                est_pred = est_r.float()

                # Strictly on-policy PPO ratio & exact closed-form Gaussian KL (zero off-policy mirrored distortion!)
                log_ratio = (new_logp[:k_sub] - f_logp[sub]).clamp(-8.0, 8.0)
                ratio = log_ratio.exp()
                old_loc, old_std = f_loc[sub], f_std[sub]
                new_loc, new_std = loc[:k_sub], std[:k_sub]
                kl = (
                    torch.log((new_std / old_std).clamp_min(1e-6))
                    + (old_std.square() + (old_loc - new_loc).square()) / (2.0 * new_std.square().clamp_min(1e-8))
                    - 0.5
                ).sum(-1).mean()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()

                surr1 = ratio * f_adv[sub]
                surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * f_adv[sub]
                pol_loss = -torch.minimum(surr1, surr2).mean()

                val_clip = f_val[idx] + (new_val - f_val[idx]).clamp(-cfg.clip_eps, cfg.clip_eps)
                v_loss = 0.5 * torch.maximum((new_val - f_ret[idx]).square(), (val_clip - f_ret[idx]).square()).mean()
                ent_mean = ent[:k_sub].mean()

                # Supervised proprioceptive EstNet loss on both orig + mirrored halves
                est_loss = F.smooth_l1_loss(est_pred, f_est[idx], beta=0.15)
                est_w = cfg.est_coef * (0.5 if env.stage >= 3 else 1.0)
                loss = pol_loss + cfg.value_coef * v_loss + est_w * est_loss - ent_coef * ent_mean

                if cfg.symmetry_aug:
                    # Enforce mirror equivariance directly on unsquashed Gaussian loc (prevents tanh saturation drift!)
                    loc_orig, loc_mirr = loc[:k_sub], loc[k_sub:]
                    loc_tgt = loc_orig[:, sym_perm] * sym_sign
                    est_orig, est_mirr = est_pred[:k_sub], est_pred[k_sub:]
                    sym_loss = (
                        (loc_mirr - loc_tgt).square().mean()
                        + 0.25 * (new_val[k_sub:] - new_val[:k_sub]).square().mean()
                        + 0.25 * (est_mirr - est_orig * est_sym_sign).square().mean()
                    )
                    loss = loss + 0.50 * sym_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                model.logstd.data.clamp_(-2.6, logstd_ceil)

                with torch.no_grad():
                    e_err = (est_pred - f_est[idx]).abs()
                    vx_mae = e_err[:, 0].mean() * vx_scale_mm
                    wall_mae = e_err[:, 3].mean()
                    drag_mae = e_err[:, 5].mean()
                opt_stats.add_(torch.stack((kl, clip_frac, ent_mean, v_loss, est_loss, vx_mae, wall_mae, drag_mae)).detach().to(torch.float64))
                opt_steps += 1
                ep_kl_sum += float(kl.item())
                ep_kl_n += 1
                if cfg.target_kl > 0 and ep > 0 and (ep_kl_sum / ep_kl_n) > 3.0 * cfg.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        if device == 'cuda':
            torch.cuda.synchronize()
        t_opt = time.monotonic() - t_opt0

        mean_kl, _clip, mean_ent, _vl, mean_el, est_vx_mae, est_wall_mae, est_drag_mae = (
            opt_stats / max(1, opt_steps)
        ).cpu().tolist()
        if cfg.target_kl > 0:
            lr_floor = max(cfg.min_lr, 0.25 * cfg.lr) if progress < 0.80 else cfg.min_lr
            lr_ceil = max(lr_floor, cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress)))
            if mean_kl > 2.0 * cfg.target_kl:
                current_lr = max(lr_floor, current_lr / 1.35)
            elif mean_kl < 0.6 * cfg.target_kl:
                current_lr = min(lr_ceil, current_lr * 1.15)
            current_lr = min(current_lr, lr_ceil)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

        mean_rew = float(rew_buf.mean().item())
        sps = (H * N) / max(t_sim + t_opt, 1e-6)
        sim_sps = (H * N) / max(t_sim, 1e-6)
        total_guards = int(term_buf.sum().item())
        guard_counts = {gname: int(val) for gname, val in zip(guard_names, guard_totals.cpu().tolist())}

        bc, bvx, byw, bte, bwl, bsl = (x.cpu().tolist() for x in (b_cnt, b_vx, b_yaw, b_ten, b_wall, b_slip))
        regimes = {}
        for bi, bname in enumerate(b_names):
            if bc[bi] > 0:
                regimes[bname] = {
                    'vx_mm_s': round(bvx[bi] / bc[bi] * 1000.0, 1),
                    'yaw_rms_deg': round(byw[bi] / bc[bi], 1),
                    'ten_act': round(bte[bi] / bc[bi], 2),
                    'wall_n': round(bwl[bi] / bc[bi], 2),
                    'slip_mm_s': round(bsl[bi] / bc[bi] * 1000.0, 1),
                }

        rec = {
            'rollout': rollout,
            'samples': total_samples,
            'stage': env.stage,
            'mean_reward': round(mean_rew, 4),
            'sps': round(sps, 1),
            'sim_sps': round(sim_sps, 1),
            'kl': round(mean_kl, 5),
            'lr': current_lr,
            'entropy': round(mean_ent, 3),
            'std_mean': round(float(model.logstd.exp().mean().item()), 3),
            'est_loss': round(mean_el, 4),
            'est_vx_mae_mm_s': round(est_vx_mae, 2),
            'est_wall_mae': round(est_wall_mae, 3),
            'est_drag_mae': round(est_drag_mae, 3),
            'min_strip_z_mm': round(min_strip_z_roll * 1000.0, 1) if (cfg.task.curved_pipes or cfg.task.diverse_v8) else None,
            'guards': total_guards,
            'guard_breakdown': guard_counts,
            'regimes': regimes,
        }
        history_log.append(rec)

        pipe_keys = {'pipe', 'straight', 'bend90', 'scurve', 'L_duct', 'str', 'b90', 'scv', 'L90', 'U180', 'tap', 'Z2L', 'W3'}
        if cfg.task.diverse_v8:
            reg_str = ' '.join(
                f"{k}:{v['vx_mm_s']:.0f}" + (f"({v['wall_n']:.1f}N)" if k in pipe_keys else "")
                for k, v in regimes.items()
            )
        else:
            reg_str = ' '.join(
                f"{k}:{v['vx_mm_s']:.0f}mm/s(y={v['yaw_rms_deg']:.1f}°,t={v['ten_act']:.2f}"
                + (f",w={v['wall_n']:.1f}N)" if k in pipe_keys else ')')
                for k, v in regimes.items()
            )
        sz_str = f" sz={min_strip_z_roll*1000.0:.0f}mm" if (cfg.task.curved_pipes or cfg.task.diverse_v8) else ""
        print(
            f"[r{rollout:04d}|S{env.stage}|{'65D' if cfg.task.include_q_err else '51D'}] "
            f"rew={mean_rew:+.4f} sps={sps:.0f} kl={mean_kl:.4f} std={rec['std_mean']:.2f} "
            f"est(v={est_vx_mae:.1f}mm/s,w={est_wall_mae:.2f},d={est_drag_mae:.2f}) g={total_guards}{sz_str} | {reg_str}",
            flush=True,
        )

        save_v6_checkpoint(out / 'latest.pt', model, optimizer, cfg, rollout, total_samples, env.stage, rec)
        if mean_rew > best_reward:
            best_reward = mean_rew
            save_v6_checkpoint(out / 'best.pt', model, optimizer, cfg, rollout, total_samples, env.stage, rec)
        if cfg.eval_every > 0 and rollout % cfg.eval_every == 0:
            save_v6_checkpoint(out / f'ckpt_r{rollout:04d}.pt', model, optimizer, cfg, rollout, total_samples, env.stage, rec)
        write_json(out / 'progress.json', {
            'elapsed_s': round(time.monotonic() - t_start, 1),
            'latest': rec,
            'history': history_log,
        })

    return {'out': str(out), 'elapsed_s': round(time.monotonic() - t_start, 1), 'latest': history_log[-1] if history_log else None}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description='SerpWorm-RL v6 Mode-Free Pure-Proprioceptive Autonomous Phase-Transition Trainer')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--init-from', type=Path, default=None, help='Warm-start Actor+EstNet+Critic prefix from prior checkpoint')
    p.add_argument('--start-stage', type=int, default=1, help='Initial curriculum stage (e.g. 2 when warm-starting)')
    p.add_argument('--initial-std', type=float, default=0.45)
    p.add_argument('--include-q-err', action=argparse.BooleanOptionalAction, default=False, help='Ablation: add 14D pure-software q_target-q (65D vs 51D)')
    p.add_argument('--curved-pipes', action=argparse.BooleanOptionalAction, default=False, help='v7 mode: train across straight, 90-deg arc, S-curve & sharp 90-deg L-duct + 600-node strip contact')
    p.add_argument('--diverse-v8', action=argparse.BooleanOptionalAction, default=False, help='v8 mode: 10 regimes (8 long/wide-diameter pipe types + gravel & grass natural terrains, zero steps)')
    p.add_argument('--strip-node-contact', action=argparse.BooleanOptionalAction, default=None, help='Enable 600-node steel-strip ground+wall contact (defaults True if --curved-pipes or --diverse-v8)')
    p.add_argument('--strip-thin-shell', action=argparse.BooleanOptionalAction, default=False, help='Enable Kirchhoff-Love thin-ribbon bending + Vlasov restrained-warping correction')
    p.add_argument('--strip-prestress', type=float, default=0.0, help='Assembly prestress fraction in [0, 1]')
    p.add_argument('--strip-width-mm', type=float, default=None, help='Override spring-steel strip width (mm, e.g. 16 or 20)')
    p.add_argument('--strip-thickness-mm', type=float, default=None, help='Override spring-steel strip thickness (mm)')
    p.add_argument('--wheel-rolling', type=float, default=None, help='Override hard-floor wheel rolling resistance')
    p.add_argument('--work-weight', type=float, default=None, help='Override reward electrical/mechanical power weight')
    p.add_argument('--episode-seconds', type=float, default=24.0, help='Episode duration in seconds')
    p.add_argument('--curved-probs', type=str, default=None, help='Comma-separated 6 regime probabilities for --curved-pipes')
    p.add_argument('--v8-probs', type=str, default=None, help='Comma-separated 10 regime probabilities for --diverse-v8')
    p.add_argument('--envs', type=int, default=1024)
    p.add_argument('--horizon', type=int, default=48)
    p.add_argument('--minibatches', type=int, default=24)
    p.add_argument('--rollouts', type=int, default=200)
    p.add_argument('--stage2-rollout', type=int, default=30)
    p.add_argument('--stage3-rollout', type=int, default=115)
    p.add_argument('--dr-ramp-rollouts', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--vx-min-mm-s', type=float, default=35.0)
    p.add_argument('--vx-max-mm-s', type=float, default=175.0)
    p.add_argument('--wz-max-rad-s', type=float, default=0.26)
    p.add_argument('--delta-step-scale', type=float, default=0.50)
    p.add_argument('--yaw-slew-rad-s', type=float, default=6.8)
    p.add_argument('--est-coef', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--target-kl', type=float, default=0.015)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--plate-solver', choices=('fused_cuda', 'segment_lu'), default='fused_cuda')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--symmetry-aug', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    hw = HardwareConfig.v5_realistic(
        tendon_pairing='independent_10d', terrain='hard', ratchet_backlash_deg=10.0,
        wheel_rolling=args.wheel_rolling,
    )
    hw.yaw_slew_rad_s = float(args.yaw_slew_rad_s)
    hw.strip_node_contact = bool((args.curved_pipes or args.diverse_v8) if args.strip_node_contact is None else args.strip_node_contact)
    hw.strip_thin_shell_compat = bool(args.strip_thin_shell)
    hw.strip_prestress_frac = float(args.strip_prestress)
    if args.strip_width_mm is not None:
        hw.strip_width_m = float(args.strip_width_mm) * 1e-3
    if args.strip_thickness_mm is not None:
        hw.strip_thickness_m = float(args.strip_thickness_mm) * 1e-3
    task = AutonomousTaskConfig(
        include_q_err=bool(args.include_q_err),
        curved_pipes=bool(args.curved_pipes),
        diverse_v8=bool(args.diverse_v8),
        episode_seconds=float(args.episode_seconds),
        vx_min_mm_s=float(args.vx_min_mm_s),
        vx_max_mm_s=float(args.vx_max_mm_s),
        wz_max_rad_s=float(args.wz_max_rad_s),
        delta_step_scale=float(args.delta_step_scale),
        stage=int(args.start_stage),
    )
    if args.curved_probs is not None:
        cp = tuple(float(x) for x in args.curved_probs.split(','))
        if len(cp) != 6:
            raise ValueError('--curved-probs expects 6 comma-separated floats')
        task.curved_regime_probs = cp
    if args.v8_probs is not None:
        vp = tuple(float(x) for x in args.v8_probs.split(','))
        if len(vp) != 10:
            raise ValueError('--v8-probs expects 10 comma-separated floats')
        task.diverse_v8_probs = vp
    dr = DomainRandConfig(enabled=False, dr_ramp_rollouts=int(args.dr_ramp_rollouts))
    reward = AutonomousRewardConfig()
    if args.work_weight is not None:
        reward.work_weight = float(args.work_weight)
    cfg = AutonomousTrainConfig(
        hardware=hw, task=task, dr=dr, reward=reward,
        envs=args.envs, horizon=args.horizon, minibatches=args.minibatches, rollouts=args.rollouts,
        stage2_rollout=args.stage2_rollout, stage3_rollout=args.stage3_rollout,
        eval_every=args.eval_every, est_coef=args.est_coef,
        lr=args.lr, target_kl=args.target_kl, initial_std=args.initial_std, seed=args.seed,
        plate_solver=args.plate_solver, amp_bf16=args.amp, symmetry_aug=args.symmetry_aug,
    )
    train_autonomous(cfg, args.out, resume=args.resume, init_from=args.init_from)


if __name__ == '__main__':
    main()

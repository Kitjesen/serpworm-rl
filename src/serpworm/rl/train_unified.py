"""End-to-end Unified Multi-Mode Asymmetric PPO Trainer (snake / worm / mixed).

Trains a single TemporalActorCritic policy across parallel real-steel robots
with a 3-stage automatic curriculum, optional torch.compile, bfloat16 autocast,
and 2x left-right reflection symmetry augmentation.
"""
from __future__ import annotations

from contextlib import nullcontext
import argparse
from dataclasses import asdict
import math
from pathlib import Path
import time
import torch
from torch import nn

from serpworm.config import DomainRandConfig, HardwareConfig, TaskConfig, UnifiedTrainConfig
from serpworm.envs.rewards import UnifiedRewardConfig
from serpworm.envs.unified_env import MODE_NAMES, UnifiedSerpWormEnv, mirror_rollout_batch
from serpworm.rl.models import TemporalActorCritic
from serpworm.rl.ppo import advantages, write_json

__all__ = ['save_checkpoint', 'train_unified', 'main']


def save_checkpoint(
    path: Path,
    model: TemporalActorCritic,
    optimizer: torch.optim.Optimizer,
    cfg: UnifiedTrainConfig,
    rollout: int,
    samples: int,
    stage: int,
    metrics: dict,
) -> None:
    payload = {
        'schema': 'serpworm_unified_ppo_v1',
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'frame_size': model.frame_size,
        'history_len': model.history_len,
        'privileged_size': model.privileged_size,
        'action_size': model.action_size,
        'hardware': cfg.hardware.to_dict(),
        'task': asdict(cfg.task),
        'dr': asdict(cfg.dr),
        'rollout': int(rollout),
        'samples': int(samples),
        'stage': int(stage),
        'metrics': metrics,
    }
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def train_unified(
    cfg: UnifiedTrainConfig,
    out_dir: Path,
    tb_dir: Path | None = None,
    resume: Path | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    reward_cfg = (
        UnifiedRewardConfig.v5_utm_calibrated()
        if getattr(cfg, 'reward_preset', 'v5_utm') == 'v5_utm'
        else UnifiedRewardConfig()
    )
    env = UnifiedSerpWormEnv(
        n=cfg.envs,
        hardware=cfg.hardware,
        task=cfg.task,
        dr=cfg.dr,
        reward_config=reward_cfg,
        device=device,
        plate_solver=cfg.plate_solver,
        seed=cfg.seed,
        noise=True,
    )
    model = TemporalActorCritic(
        frame_size=env.frame_size,
        history_len=env.history_len,
        privileged_size=env.privileged_size,
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

    fast_model = torch.compile(model) if (cfg.compile_model and device == 'cuda') else model
    amp_ctx = (
        (lambda: torch.amp.autocast('cuda', dtype=torch.bfloat16))
        if (cfg.amp_bf16 and device == 'cuda')
        else nullcontext
    )

    tb_writer = None
    if tb_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(tb_dir))

    write_json(out / 'config.json', {
        'hardware': cfg.hardware.to_dict(),
        'task': asdict(cfg.task),
        'dr': asdict(cfg.dr),
        'train': {k: v for k, v in asdict(cfg).items() if k not in ('hardware', 'task', 'dr')},
        'frame_size': env.frame_size,
        'privileged_size': env.privileged_size,
        'action_size': env.action_size,
        'effective_plate_solver': env.c.plate_solver,
        'cuda_graph_steps': env.c.graph_steps,
        'device': device,
        'gpu': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
    })

    H, N = cfg.horizon, cfg.envs
    obs_buf = torch.empty((H, N, env.history_len, env.frame_size), device=device, dtype=torch.float32)
    priv_buf = torch.empty((H, N, env.privileged_size), device=device, dtype=torch.float32)
    lat_buf = torch.empty((H, N, env.action_size), device=device, dtype=torch.float32)
    logp_buf = torch.empty((H, N), device=device, dtype=torch.float32)
    rew_buf = torch.empty((H, N), device=device, dtype=torch.float32)
    val_buf = torch.empty((H, N), device=device, dtype=torch.float32)
    nval_buf = torch.empty((H, N), device=device, dtype=torch.float32)
    term_buf = torch.empty((H, N), device=device, dtype=torch.bool)
    trunc_buf = torch.empty((H, N), device=device, dtype=torch.bool)

    guard_names = ('joint_limit', 'tilt', 'height', 'wheel_clearance', 'nonfinite')
    if cfg.hardware.tendon_pairing == 'paired_5d':
        sym_perm = None
        sym_sign = torch.tensor([1.0] * 5 + [-1.0] * 4, device=device, dtype=torch.float32)
    else:
        sym_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 10, 11, 12, 13], device=device, dtype=torch.long)
        sym_sign = torch.tensor([1.0] * 10 + [-1.0] * 4, device=device, dtype=torch.float32)
    logstd_ceil = max(math.log(0.46), math.log(max(cfg.initial_std, 0.05)))

    obs, priv = env.obs, env.critic_obs
    current_lr = float(cfg.lr)
    best_stage = env.stage
    best_reward = -1e9
    history_log = []
    t_start = time.monotonic()

    try:
        for rollout in range(start_rollout + 1, start_rollout + cfg.rollouts + 1):
            prev_stage = env.stage
            if cfg.stage3_rollout > 0 and rollout >= cfg.stage3_rollout and env.stage < 3:
                ramp_n = int(getattr(cfg.dr, 'dr_ramp_rollouts', 20))
                env.set_stage(3, dr_scale=(1.0 / max(1, ramp_n)) if ramp_n > 0 else 1.0)
            elif cfg.stage2_rollout > 0 and rollout >= cfg.stage2_rollout and env.stage < 2:
                env.set_stage(2)
            if env.stage > prev_stage:
                std_floor = float(getattr(cfg, 'stage_std_floor', 0.0))
                if std_floor > 0.0:
                    model.logstd.data.clamp_min_(math.log(std_floor))
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
            # Smooth cosine entropy annealing preserving exploration through Stage 2/3 transitions
            ent_floor = 0.16 * cfg.entropy_coef
            ent_coef = ent_floor + 0.5 * (cfg.entropy_coef - ent_floor) * (1.0 + math.cos(math.pi * progress))

            t0 = time.monotonic()
            mode_vx_sum = torch.zeros(3, device=device, dtype=torch.float64)
            mode_vx_err = torch.zeros(3, device=device, dtype=torch.float64)
            mode_wz_err = torch.zeros(3, device=device, dtype=torch.float64)
            mode_yaw_max = torch.zeros(3, device=device, dtype=torch.float64)
            mode_cnt = torch.zeros(3, device=device, dtype=torch.float64)
            stop_vx_sum = torch.zeros((), device=device, dtype=torch.float64)
            stop_cnt = torch.zeros((), device=device, dtype=torch.float64)
            guard_totals = torch.zeros(len(guard_names), device=device, dtype=torch.long)

            with torch.no_grad():
                for t in range(H):
                    model.obs_norm.update(obs[:, -1, :])
                    model.priv_norm.update(priv)
                    act, latent, logp, val = fast_model.sample(obs, priv)
                    nobs, npriv, rew, term, trunc, tobs, tpriv, info = env.step(act)
                    nv = fast_model.value(tobs, tpriv)

                    obs_buf[t] = obs
                    priv_buf[t] = priv
                    lat_buf[t] = latent
                    logp_buf[t] = logp
                    rew_buf[t] = rew
                    val_buf[t] = val
                    nval_buf[t] = nv
                    term_buf[t] = term
                    trunc_buf[t] = trunc

                    obs, priv = nobs, npriv

                    mid = info['mode_id']
                    cmd = info['cmd']
                    vel = info['body_vel']
                    is_stop = cmd[:, 2] > 0.5
                    stop_f = is_stop.to(torch.float64)
                    stop_vx_sum.add_((vel[:, 0].abs() * stop_f).sum())
                    stop_cnt.add_(stop_f.sum())

                    moving = ~is_stop
                    for m in range(3):
                        sel_f = (moving & (mid == m)).to(torch.float64)
                        mode_vx_sum[m].add_((vel[:, 0] * sel_f).sum())
                        mode_vx_err[m].add_(((vel[:, 0] - cmd[:, 0]).abs() * sel_f).sum())
                        mode_wz_err[m].add_(((vel[:, 2] - cmd[:, 1]).abs() * sel_f).sum())
                        mode_yaw_max[m].add_((info['max_yaw_deg'] * sel_f).sum())
                        mode_cnt[m].add_(sel_f.sum())
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
            total_b = base_b
            f_obs = obs_buf.reshape(base_b, env.history_len, env.frame_size)
            f_priv = priv_buf.reshape(base_b, env.privileged_size)
            f_lat = lat_buf.reshape(base_b, env.action_size)
            f_adv = adv.reshape(base_b)
            f_ret = ret.reshape(base_b)

            if cfg.symmetry_aug:
                with torch.no_grad():
                    m_obs, m_priv, m_lat = mirror_rollout_batch(f_obs, f_priv, f_lat, cfg.hardware.tendon_pairing)
                    model.obs_norm.update(m_obs[:, -1, :])
                    model.priv_norm.update(m_priv)
                f_obs = torch.cat((f_obs, m_obs), dim=0)
                f_priv = torch.cat((f_priv, m_priv), dim=0)
                f_lat = torch.cat((f_lat, m_lat), dim=0)
                f_adv = torch.cat((f_adv, f_adv), dim=0)
                f_ret = torch.cat((f_ret, f_ret), dim=0)
                total_b = f_obs.shape[0]

            mb_size = max(64, total_b // cfg.minibatches)
            t_opt0 = time.monotonic()

            # Recompute pre-epoch reference logp & value under settled RunningNorm and matching autocast
            # so intra-rollout normalizer updates and bf16 precision never inflate KL or crush LR.
            with torch.no_grad():
                f_logp = torch.empty(total_b, device=device, dtype=torch.float32)
                f_val = torch.empty(total_b, device=device, dtype=torch.float32)
                for start in range(0, total_b, mb_size):
                    end = min(total_b, start + mb_size)
                    with amp_ctx():
                        lp_ref, _, v_ref = fast_model.evaluate_latent(f_obs[start:end], f_priv[start:end], f_lat[start:end])
                    f_logp[start:end] = lp_ref.float()
                    f_val[start:end] = v_ref.float()

            opt_stats = torch.zeros(5, device=device, dtype=torch.float64)
            opt_steps = 0
            half_mb = max(1, min(base_b, mb_size // 2))
            for _ in range(cfg.epochs):
                if cfg.symmetry_aug:
                    perm = torch.randperm(base_b, device=device)
                    iter_ranges = range(0, base_b, half_mb)
                else:
                    perm = torch.randperm(total_b, device=device)
                    iter_ranges = range(0, total_b, mb_size)

                for start in iter_ranges:
                    if cfg.symmetry_aug:
                        sub = perm[start:start + half_mb]
                        k_sub = sub.shape[0]
                        idx = torch.cat((sub, sub + base_b), dim=0)
                        with amp_ctx():
                            new_logp_raw, ent_raw, new_val_raw, mu_raw = fast_model.evaluate_latent_with_mean(
                                f_obs[idx], f_priv[idx], f_lat[idx]
                            )
                        mu = mu_raw.float()
                        mu_orig, mu_mirr = mu[:k_sub], mu[k_sub:]
                        mu_target = (mu_orig[:, sym_perm] if sym_perm is not None else mu_orig) * sym_sign
                    else:
                        idx = perm[start:start + mb_size]
                        with amp_ctx():
                            new_logp_raw, ent_raw, new_val_raw = fast_model.evaluate_latent(f_obs[idx], f_priv[idx], f_lat[idx])

                    new_logp = new_logp_raw.float()
                    ent = ent_raw.float()
                    new_val = new_val_raw.float()

                    log_ratio = (new_logp - f_logp[idx]).clamp(-15.0, 15.0)
                    ratio = log_ratio.exp()
                    kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()

                    surr1 = ratio * f_adv[idx]
                    surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * f_adv[idx]
                    pol_loss = -torch.minimum(surr1, surr2).mean()

                    val_clip = f_val[idx] + (new_val - f_val[idx]).clamp(-cfg.clip_eps, cfg.clip_eps)
                    v_loss = 0.5 * torch.maximum((new_val - f_ret[idx]).square(), (val_clip - f_ret[idx]).square()).mean()
                    ent_mean = ent.mean()
                    loss = pol_loss + cfg.value_coef * v_loss - ent_coef * ent_mean
                    if cfg.symmetry_aug:
                        sym_loss = (
                            (mu_mirr - mu_target).square().mean()
                            + 0.25 * (new_val[k_sub:] - new_val[:k_sub]).square().mean()
                        )
                        loss = loss + 0.25 * sym_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    model.logstd.data.clamp_(-2.6, logstd_ceil)

                    opt_stats.add_(torch.stack((kl, clip_frac, ent_mean, v_loss, pol_loss)).detach().to(torch.float64))
                    opt_steps += 1

            if device == 'cuda':
                torch.cuda.synchronize()
            t_opt = time.monotonic() - t_opt0

            mean_kl, _mean_clip, mean_ent, _mean_vl, _mean_pl = (opt_stats / max(1, opt_steps)).cpu().tolist()
            if cfg.target_kl > 0:
                lr_floor = max(cfg.min_lr, 0.25 * cfg.lr) if progress < 0.80 else cfg.min_lr
                # Cosine upper ceiling on max_lr suppresses late Stage-3 DR oscillation
                lr_ceil = max(
                    lr_floor,
                    cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress)),
                )
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

            mode_cnt_l = mode_cnt.cpu().tolist()
            mode_vx_l = mode_vx_sum.cpu().tolist()
            mode_vx_err_l = mode_vx_err.cpu().tolist()
            mode_wz_err_l = mode_wz_err.cpu().tolist()
            mode_yaw_l = mode_yaw_max.cpu().tolist()
            per_mode = {}
            for m, mname in enumerate(MODE_NAMES):
                cnt = mode_cnt_l[m]
                if cnt > 0:
                    per_mode[mname] = {
                        'vx_mm_s': round(mode_vx_l[m] / cnt * 1000.0, 2),
                        'vx_mae_mm_s': round(mode_vx_err_l[m] / cnt * 1000.0, 2),
                        'wz_mae_rad_s': round(mode_wz_err_l[m] / cnt, 4),
                        'mean_max_yaw_deg': round(mode_yaw_l[m] / cnt, 2),
                    }
            stop_abs = round(float(stop_vx_sum.item()) / max(float(stop_cnt.item()), 1.0) * 1000.0, 2)

            rec = {
                'rollout': rollout,
                'samples': total_samples,
                'stage': env.stage,
                'mean_reward': round(mean_rew, 4),
                'sps': round(sps, 1),
                'sim_sps': round(sim_sps, 1),
                'sim_s': round(t_sim, 2),
                'opt_s': round(t_opt, 2),
                'kl': round(mean_kl, 5),
                'lr': current_lr,
                'entropy': round(mean_ent, 3),
                'std_mean': round(float(model.logstd.exp().mean().item()), 3),
                'guards': total_guards,
                'guard_breakdown': guard_counts,
                'stop_abs_vx_mm_s': stop_abs,
                'modes': per_mode,
            }
            history_log.append(rec)

            if tb_writer is not None:
                tb_writer.add_scalar('train/mean_reward', mean_rew, total_samples)
                tb_writer.add_scalar('train/sps', sps, total_samples)
                tb_writer.add_scalar('train/kl', mean_kl, total_samples)
                tb_writer.add_scalar('train/lr', current_lr, total_samples)
                tb_writer.add_scalar('train/entropy', rec['entropy'], total_samples)
                tb_writer.add_scalar('train/guards', total_guards, total_samples)
                tb_writer.add_scalar('train/stage', env.stage, total_samples)
                tb_writer.add_scalar('task/stop_abs_vx_mm_s', stop_abs, total_samples)
                for mname, md in per_mode.items():
                    tb_writer.add_scalar(f'mode_{mname}/vx_mm_s', md['vx_mm_s'], total_samples)
                    tb_writer.add_scalar(f'mode_{mname}/vx_mae_mm_s', md['vx_mae_mm_s'], total_samples)
                    tb_writer.add_scalar(f'mode_{mname}/wz_mae_rad_s', md['wz_mae_rad_s'], total_samples)
                    tb_writer.add_scalar(f'mode_{mname}/max_yaw_deg', md['mean_max_yaw_deg'], total_samples)
                tb_writer.flush()

            print(
                f"[r{rollout:04d}|S{env.stage}] samples={total_samples:,} rew={mean_rew:+.4f} "
                f"sps={sps:.0f} (sim={sim_sps:.0f}) kl={mean_kl:.4f} std={rec['std_mean']:.2f} "
                f"guards={total_guards} stop={stop_abs:.1f} | "
                + " ".join(f"{k}:{v['vx_mm_s']:.1f}mm/s(y={v['mean_max_yaw_deg']:.1f}°)" for k, v in per_mode.items()),
                flush=True,
            )

            save_checkpoint(out / 'latest.pt', model, optimizer, cfg, rollout, total_samples, env.stage, rec)
            if mean_rew > best_reward:
                best_reward = mean_rew
                save_checkpoint(out / 'best.pt', model, optimizer, cfg, rollout, total_samples, env.stage, rec)
            if cfg.eval_every > 0 and rollout % cfg.eval_every == 0:
                save_checkpoint(out / f'ckpt_r{rollout:04d}.pt', model, optimizer, cfg, rollout, total_samples, env.stage, rec)
            write_json(out / 'progress.json', {
                'elapsed_s': round(time.monotonic() - t_start, 1),
                'latest': rec,
                'history': history_log,
            })

        return {'out': str(out), 'elapsed_s': round(time.monotonic() - t_start, 1), 'latest': history_log[-1] if history_log else None}
    finally:
        if tb_writer is not None:
            tb_writer.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description='SerpWorm-RL Unified 3-Mode Asymmetric PPO Trainer (v5 realistic)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tb-dir', type=Path, default=None)
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--hardware-json', type=Path, default=None)
    p.add_argument('--tendon-model', choices=('legacy_wrap', 'crank_calibrated'), default='crank_calibrated')
    p.add_argument('--tendon-pairing', choices=('paired_5d', 'independent_10d'), default='paired_5d')
    p.add_argument('--contact-model', choices=('kelvin', 'arpc_hard', 'arpc_carpet', 'arpc_soft_soil'), default='arpc_hard')
    p.add_argument('--implicit-contact', action=argparse.BooleanOptionalAction, default=True, help='Enable 12x12 implicit Contact-Schur coupling (unlocks dt=1.0 ms)')
    p.add_argument('--ratchet-backlash-deg', type=float, default=10.0, help='Mechanical ratchet one-way backlash deadzone in degrees (e.g. 360/N_teeth)')
    p.add_argument('--wheel-rolling', type=float, default=None, help='Override wheel rolling resistance mu_rr (default: 0.025 on hard floor)')
    p.add_argument('--reward-preset', choices=('v5_utm', 'legacy'), default='v5_utm')
    p.add_argument('--action-mode', choices=('delta', 'absolute'), default='delta')
    p.add_argument('--delta-step-scale', type=float, default=0.38, help='Max normalized target step per 0.1s (matches 26 kg*cm 8.2 rad/s envelope)')
    p.add_argument('--yaw-slew-rad-s', type=float, default=None, help='Optional override for yaw target slew cap (rad/s)')
    p.add_argument('--envs', type=int, default=1024)
    p.add_argument('--horizon', type=int, default=48)
    p.add_argument('--minibatches', type=int, default=24)
    p.add_argument('--rollouts', type=int, default=150)
    p.add_argument('--stage', type=int, default=1)
    p.add_argument('--stage2-rollout', type=int, default=None, help='Rollout to unlock steering & mode switching (default: ~15%% of rollouts)')
    p.add_argument('--stage3-rollout', type=int, default=None, help='Rollout to unlock domain randomization (default: ~57%% of rollouts)')
    p.add_argument('--dr-ramp-rollouts', type=int, default=20, help='Rollouts to smoothly ramp Stage-3 DR strength from 0 to 1')
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--dt', type=float, default=None, help='Physics substep in s (default: 0.001 with implicit ARPC, 0.0005 with explicit kelvin)')
    p.add_argument('--elements', type=int, default=16, help='Corotational beam elements per strip (e.g. 8 or 16)')
    p.add_argument('--tangent-interval', type=int, default=1, help='Substeps per beam tangent factorization')
    p.add_argument('--vx-min-mm-s', type=float, default=25.0)
    p.add_argument('--vx-max-mm-s', type=float, default=85.0)
    p.add_argument('--wz-max-rad-s', type=float, default=0.22)
    p.add_argument('--stop-fraction', type=float, default=0.16)
    p.add_argument('--straight-fraction', type=float, default=0.38)
    p.add_argument('--command-switch-s', type=float, default=6.0)
    p.add_argument('--mode-probs', type=float, nargs=3, default=(0.46, 0.16, 0.38), metavar=('SNAKE', 'WORM', 'MIXED'))
    p.add_argument('--plate-solver', choices=('fused_cuda', 'segment_lu'), default='fused_cuda')
    p.add_argument('--compile', action='store_true', help='Compile TemporalActorCritic via torch.compile')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True, help='Enable bfloat16 autocast during PPO minibatch forward')
    p.add_argument('--symmetry-aug', action=argparse.BooleanOptionalAction, default=True, help='Enable 2x left-right reflection augmentation & equivariance loss')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gamma', type=float, default=0.994)
    p.add_argument('--target-kl', type=float, default=0.013)
    p.add_argument('--entropy-coef', type=float, default=0.0025)
    p.add_argument('--stage-std-floor', type=float, default=0.22)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args(argv)

    stage2_r = max(5, round(0.15 * args.rollouts)) if args.stage2_rollout is None else int(args.stage2_rollout)
    stage3_r = max(stage2_r + 10, round(0.57 * args.rollouts)) if args.stage3_rollout is None else int(args.stage3_rollout)
    terrain_map = {
        'kelvin': 'kelvin',
        'arpc_hard': 'hard',
        'arpc_carpet': 'carpet',
        'arpc_soft_soil': 'soft_soil',
    }
    terrain = terrain_map[args.contact_model]

    if args.hardware_json:
        hw = HardwareConfig.load(args.hardware_json)
        hw.tendon_pairing = args.tendon_pairing
    elif args.tendon_model == 'crank_calibrated':
        hw = HardwareConfig.v5_realistic(
            tendon_pairing=args.tendon_pairing,
            terrain=terrain,
            ratchet_backlash_deg=args.ratchet_backlash_deg,
            wheel_rolling=args.wheel_rolling,
        )
        hw.implicit_contact = bool(args.implicit_contact)
    else:
        hw = HardwareConfig.ds3218_default().with_terrain(terrain, implicit_contact=bool(args.implicit_contact))
        hw.tendon_pairing = args.tendon_pairing
        if args.wheel_rolling is not None:
            hw.wheel_rolling = float(args.wheel_rolling)
        if args.contact_model != 'kelvin':
            hw.ratchet_backlash_rad = math.radians(float(args.ratchet_backlash_deg))

    if args.yaw_slew_rad_s is not None:
        hw.yaw_slew_rad_s = float(args.yaw_slew_rad_s)

    eff_dt = float(args.dt) if args.dt is not None else (0.001 if hw.implicit_contact else 0.0005)
    task = TaskConfig(
        physics_dt_s=eff_dt,
        elements_per_strip=args.elements,
        tangent_interval=args.tangent_interval,
        action_mode=args.action_mode,
        delta_step_scale=args.delta_step_scale,
        vx_min_mm_s=args.vx_min_mm_s,
        vx_max_mm_s=args.vx_max_mm_s,
        wz_max_rad_s=args.wz_max_rad_s,
        stop_fraction=args.stop_fraction,
        straight_fraction=args.straight_fraction,
        mode_probs=(float(args.mode_probs[0]), float(args.mode_probs[1]), float(args.mode_probs[2])),
        command_switch_s=args.command_switch_s,
        stage=args.stage,
    )
    dr = DomainRandConfig(enabled=(args.stage >= 3), dr_ramp_rollouts=args.dr_ramp_rollouts)
    cfg = UnifiedTrainConfig(
        hardware=hw, task=task, dr=dr,
        envs=args.envs, horizon=args.horizon, minibatches=args.minibatches, rollouts=args.rollouts,
        stage2_rollout=stage2_r, stage3_rollout=stage3_r,
        eval_every=args.eval_every, plate_solver=args.plate_solver,
        compile_model=args.compile, amp_bf16=args.amp, symmetry_aug=args.symmetry_aug,
        lr=args.lr, gamma=args.gamma, target_kl=args.target_kl,
        entropy_coef=args.entropy_coef, stage_std_floor=args.stage_std_floor,
        reward_preset=args.reward_preset, seed=args.seed,
    )
    train_unified(cfg, args.out, tb_dir=args.tb_dir, resume=args.resume)


if __name__ == '__main__':
    main()

"""Deterministic GPU-resident evaluator & optional MP4 / MuJoCo 3D renderer for unified policies."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import torch

from serpworm.config import DomainRandConfig, HardwareConfig, TaskConfig
from serpworm.envs.unified_env import MODE_NAMES, UnifiedSerpWormEnv
from serpworm.physics.gpu_core import matrix
from serpworm.rl.models import TemporalActorCritic

__all__ = ['EVAL_SCENARIOS', 'evaluate_unified', 'main']

EVAL_SCENARIOS = (
    # (name, mode_id, vx_mm_s, wz_rad_s, stop, switch_schedule)
    ('snake_straight_60', 0, 60.0,  0.00, 0.0, None),
    ('snake_turn_left',   0, 45.0, +0.18, 0.0, None),
    ('snake_turn_right',  0, 45.0, -0.18, 0.0, None),
    ('snake_stop',        0,  0.0,  0.00, 1.0, None),
    ('worm_straight_70',  1, 70.0,  0.00, 0.0, None),
    ('worm_stop',         1,  0.0,  0.00, 1.0, None),
    ('mixed_straight_75', 2, 75.0,  0.00, 0.0, None),
    ('mixed_turn_right',  2, 55.0, -0.18, 0.0, None),
    ('live_mode_switch',  0, 55.0,  0.00, 0.0, [(6.0, 1, 65.0, 0.0, 0.0), (12.0, 2, 70.0, 0.0, 0.0)]),
)


def evaluate_unified(
    ckpt_path: Path,
    out_dir: Path,
    seconds: float = 18.0,
    plate_solver: str = 'fused_cuda',
    render_mp4: bool = False,
    render_mujoco: bool = False,
    save_traj: bool = False,
) -> dict:
    from dataclasses import fields as dc_fields
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hw_fields = {f.name for f in dc_fields(HardwareConfig)}
    hw = HardwareConfig(**{k: v for k, v in ckpt['hardware'].items() if k in hw_fields})
    task_fields = {f.name for f in dc_fields(TaskConfig)}
    task_dict = {k: v for k, v in ckpt['task'].items() if k in task_fields}
    task_dict['episode_seconds'] = seconds + 2.0  # avoid auto-truncation before measurement ends
    task_dict['command_switch_s'] = 1e6           # evaluator controls commands explicitly
    task = TaskConfig(**task_dict)

    n = len(EVAL_SCENARIOS)
    env = UnifiedSerpWormEnv(
        n=n, hardware=hw, task=task, dr=DomainRandConfig(enabled=False),
        device=device, plate_solver=plate_solver if device == 'cuda' else 'segment_lu',
        noise=False,
    )
    model = TemporalActorCritic(
        frame_size=ckpt['frame_size'],
        history_len=ckpt['history_len'],
        privileged_size=ckpt['privileged_size'],
        action_size=ckpt['action_size'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    for i, (_, mid, vx, wz, stp, _) in enumerate(EVAL_SCENARIOS):
        env.mode_id[i] = mid
        env.cmd[i, 0] = vx * 1e-3
        env.cmd[i, 1] = wz
        env.cmd[i, 2] = stp
    _, _, _, _, pkt = env._read_sensors(torch.ones(n, device=device, dtype=torch.bool))
    env.history.copy_(env._build_frame(pkt)[:, None, :])
    obs = env.history.clone()

    steps = round(seconds / task.policy_dt_s)
    # Preallocate all telemetry & trajectory buffers on GPU (zero D2H sync inside loop!)
    vx_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    wz_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    yaw_buf = torch.empty((steps, n, 4), device=device, dtype=torch.float64)
    servo_buf = torch.empty((steps, n, 10), device=device, dtype=torch.float64)
    clr_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    failed_buf = torch.zeros(n, device=device, dtype=torch.bool)
    p0_gpu = env.c.x[:, :10, :3].clone()

    rec_lanes = (0, 4, 6, 8)
    rec_idx = torch.tensor(rec_lanes, device=device, dtype=torch.long)
    need_full_x = render_mp4 or render_mujoco or save_traj
    x_rec_buf = torch.empty((steps, n, env.c.D, 7), device=device, dtype=torch.float64) if (render_mujoco or save_traj) else None
    mode_rec_buf = torch.empty((steps, n), device=device, dtype=torch.long) if (render_mujoco or save_traj) else None
    mp4_x_buf = torch.empty((steps, len(rec_lanes), env.c.D, 7), device=device, dtype=torch.float64) if render_mp4 else None
    mp4_R_buf = torch.empty((steps, len(rec_lanes), 10, 3, 3), device=device, dtype=torch.float64) if render_mp4 else None

    with torch.no_grad():
        for k in range(steps):
            t_s = k * task.policy_dt_s
            for i, (_, _, _, _, _, sched) in enumerate(EVAL_SCENARIOS):
                if sched is not None:
                    for t_sw, m_sw, vx_sw, wz_sw, stp_sw in sched:
                        if abs(t_s - t_sw) < 0.5 * task.policy_dt_s:
                            env.mode_id[i] = m_sw
                            env.cmd[i, 0] = vx_sw * 1e-3
                            env.cmd[i, 1] = wz_sw
                            env.cmd[i, 2] = stp_sw

            if mp4_x_buf is not None:
                mp4_x_buf[k].copy_(env.c.x[rec_idx])
                mp4_R_buf[k].copy_(matrix(env.c.x[rec_idx, :10, 3:]))
            if x_rec_buf is not None:
                x_rec_buf[k].copy_(env.c.x)
                mode_rec_buf[k].copy_(env.mode_id)

            act = model.action(obs)
            obs, _, _, term, _, _, _, info = env.step(act)
            failed_buf.logical_or_(term)
            vx_buf[k].copy_(info['body_vel'][:, 0] * 1000.0)
            wz_buf[k].copy_(info['body_vel'][:, 2])
            _, _, _, q = env._geometry()
            yaw_buf[k].copy_(q * (180.0 / math.pi))
            servo_buf[k].copy_(env.c.servo * (180.0 / math.pi))
            clr_buf[k].copy_(info['clearance'] * 1000.0)

    # Single bulk GPU -> CPU transfer after rollout completes
    p0_world = p0_gpu.cpu().numpy()
    p1_world = env.c.x[:, :10, :3].cpu().numpy()
    vx_hist = vx_buf.cpu().numpy()
    wz_hist = wz_buf.cpu().numpy()
    yaw_hist = yaw_buf.cpu().numpy()
    servo_hist = servo_buf.cpu().numpy()
    clr_hist = clr_buf.cpu().numpy()
    failed_any = failed_buf.cpu().numpy()

    half = steps // 2
    results = []
    for i, (name, mid, vx_cmd, wz_cmd, stp, _) in enumerate(EVAL_SCENARIOS):
        fwd = p0_world[i, 9, :2] - p0_world[i, 0, :2]
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        disp_mm = float((p1_world[i, :, :2].mean(0) - p0_world[i, :, :2].mean(0)) @ fwd) * 1000.0
        axis0 = p0_world[i, 9, :2] - p0_world[i, 0, :2]
        axis1 = p1_world[i, 9, :2] - p1_world[i, 0, :2]
        net_turn_deg = float(np.degrees(np.arctan2(
            axis0[0] * axis1[1] - axis0[1] * axis1[0],
            axis0[0] * axis1[0] + axis0[1] * axis1[1],
        )))
        yaw_tail = yaw_hist[half:, i]
        servo_tail = servo_hist[half:, i]
        results.append({
            'scenario': name,
            'mode': MODE_NAMES[mid],
            'cmd_vx_mm_s': vx_cmd,
            'cmd_wz_rad_s': wz_cmd,
            'stop': bool(stp > 0.5),
            'failed': bool(failed_any[i]),
            'net_fwd_mm': round(disp_mm, 1),
            'net_turn_deg': round(net_turn_deg, 1),
            'tail_mean_vx_mm_s': round(float(vx_hist[half:, i].mean()), 2),
            'tail_mean_wz_rad_s': round(float(wz_hist[half:, i].mean()), 4),
            'yaw_min_deg': round(float(yaw_tail.min()), 1),
            'yaw_max_deg': round(float(yaw_tail.max()), 1),
            'servo_pp_deg': round(float((servo_tail.max(0) - servo_tail.min(0)).mean()), 1),
            'min_clearance_mm': round(float(clr_hist[:, i].min()), 2),
        })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'checkpoint': str(ckpt_path), 'seconds': seconds, 'results': results}
    (out_dir / 'eval_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    if x_rec_buf is not None:
        traj_path = out_dir / 'eval_trajectories.npz'
        np.savez_compressed(
            traj_path,
            x=x_rec_buf.cpu().numpy(),
            mode=mode_rec_buf.cpu().numpy(),
            strip_nodes=env.c.nodes.cpu().numpy(),
            transforms=env.c.transforms.cpu().numpy(),
            mounts=env.c.mounts.cpu().numpy(),
            anchors=env.c.anchors.cpu().numpy(),
            guides=env.c.guides.cpu().numpy(),
            fps=round(1.0 / task.policy_dt_s),
        )
        if render_mujoco and env.c.E == 16:
            from serpworm.probes.render_mujoco import render_mujoco_video
            render_mujoco_video(
                traj_path,
                out_dir / 'eval_mujoco_3d.mp4',
                lanes=(3, 5, 7),
                labels=('纯蠕虫', '蛇 + 蠕虫', '三态切换'),
            )

    if render_mp4 and mp4_x_buf is not None:
        x_np = mp4_x_buf.cpu().numpy()
        R_np = mp4_R_buf.cpu().numpy()
        rec_frames = [(k * task.policy_dt_s, x_np[k], R_np[k]) for k in range(steps)]
        _render_eval_mp4(
            rec_frames, env, vx_hist[:, list(rec_lanes)],
            [EVAL_SCENARIOS[i][0] for i in rec_lanes],
            out_dir / 'eval_unified_modes.mp4',
            fps=round(1.0 / task.policy_dt_s),
        )
    return summary


def _render_eval_mp4(rec_frames, env, vx_sub, titles, mp4_path: Path, fps: int = 10) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sn = env.c.nodes.cpu().numpy()
    tr_pos = env.c.transforms[..., :3].cpu().numpy()
    mnts = env.c.mounts.cpu().numpy()
    seg_cols = ['#3b82f6', '#06b6d4', '#10b981', '#84cc16', '#f59e0b']

    W, H = 1280, 820
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor='#0b1220')
    gs = fig.add_gridspec(4, 2, width_ratios=[1.65, 1.0], hspace=0.32, wspace=0.14, left=0.04, right=0.98, top=0.93, bottom=0.05)

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}', '-pix_fmt', 'rgba', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '21', '-preset', 'fast',
        str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    x_init = [rec_frames[0][1][r, :10, 0].mean() * 1000.0 for r in range(len(titles))]
    x_end = [rec_frames[-1][1][r, :10, 0].mean() * 1000.0 for r in range(len(titles))]

    for fi, (t_s, x_all, R_all) in enumerate(rec_frames):
        fig.clf()
        fig.suptitle(f'SerpWorm-RL Unified End-to-End RL Policy (40 Real FP64 Strips)   t = {t_s:4.1f} s',
                     color='#f8fafc', fontsize=12.5, fontweight='bold', y=0.975)
        for r, title in enumerate(titles):
            xr, Rr = x_all[r], R_all[r]
            plates = xr[:10, :3] * 1000.0
            wheels = (xr[:10, None, :3] + np.einsum('pij,psj->psi', Rr, mnts)).reshape(20, 3) * 1000.0
            init_p = rec_frames[0][1][r, :10, :3] * 1000.0
            fwd = init_p[9, :2] - init_p[0, :2]
            fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
            disp = float((plates[:, :2].mean(0) - init_p[:, :2].mean(0)) @ fwd)
            vx_now = float(vx_sub[min(fi, len(vx_sub) - 1), r])

            strips = []
            for sid in range(40):
                nodes = sn[sid]
                pts = xr[nodes, :3].copy()
                pts[0] += Rr[nodes[0]] @ tr_pos[sid, 0, 0]
                pts[-1] += Rr[nodes[-1]] @ tr_pos[sid, -1, 1]
                strips.append(pts[::max(1, env.c.E // 8)] * 1000.0)
            strips = np.asarray(strips)

            xmin = min(x_init[r], x_end[r]) - 500
            xmax = max(x_init[r], x_end[r]) + 500

            ax1 = fig.add_subplot(gs[r, 0], facecolor='#0f172a')
            for sid in range(40):
                ax1.plot(strips[sid, :, 0], strips[sid, :, 1], color=seg_cols[sid // 8], lw=1.05, alpha=0.88)
            ax1.scatter(wheels[:, 0], wheels[:, 1], c='#e2e8f0', s=12, zorder=5, edgecolors='#334155', linewidths=0.4)
            ax1.axvline(init_p[9, 0], color='#ef4444', ls='--', lw=0.9, alpha=0.7)
            ax1.axvline(plates[9, 0], color='#22c55e', ls=':', lw=0.9, alpha=0.7)
            ax1.set_xlim(xmin, xmax); ax1.set_ylim(-240, 240); ax1.set_aspect('equal')
            ax1.set_title(f'{title}  |  fwd={disp:+5.1f} mm, vx={vx_now:+5.1f} mm/s', color='#e2e8f0', fontsize=9, loc='left', pad=2)
            ax1.tick_params(colors='#64748b', labelsize=7); ax1.grid(color='#1e293b', lw=0.5)

            ax2 = fig.add_subplot(gs[r, 1], facecolor='#0f172a')
            ax2.axhline(0, color='#475569', lw=1.1)
            for sid in range(40):
                ax2.plot(strips[sid, :, 0], strips[sid, :, 2], color=seg_cols[sid // 8], lw=0.95, alpha=0.85)
            ax2.scatter(wheels[:, 0], wheels[:, 2], c='#e2e8f0', s=14, zorder=5, edgecolors='#334155', linewidths=0.4)
            ax2.set_xlim(xmin, xmax); ax2.set_ylim(-15, 260); ax2.set_aspect('equal')
            ax2.set_title('Side (XZ)', color='#94a3b8', fontsize=8, loc='left', pad=2)
            ax2.tick_params(colors='#64748b', labelsize=7); ax2.grid(color='#1e293b', lw=0.5)

        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode('utf-8', errors='replace')
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f'ffmpeg failed ({rc}): {err[-400:]}')


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description='Evaluate SerpWorm-RL unified 3-mode policy')
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=18.0)
    p.add_argument('--plate-solver', choices=('fused_cuda', 'segment_lu'), default='fused_cuda')
    p.add_argument('--render-mp4', action='store_true')
    p.add_argument('--render-mujoco', action='store_true')
    p.add_argument('--save-traj', action='store_true')
    args = p.parse_args(argv)
    summary = evaluate_unified(
        args.checkpoint, args.out, seconds=args.seconds,
        plate_solver=args.plate_solver, render_mp4=args.render_mp4,
        render_mujoco=args.render_mujoco, save_traj=args.save_traj,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

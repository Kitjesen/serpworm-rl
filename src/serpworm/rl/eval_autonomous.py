"""Deterministic pure-proprioceptive evaluator for v6 mode-free autonomous policies (51D vs 65D)."""
from __future__ import annotations

import argparse
from dataclasses import fields as dc_fields
import json
import math
from pathlib import Path
import numpy as np
import torch

from serpworm.config import DomainRandConfig, HardwareConfig
from serpworm.envs.autonomous_env import AutonomousPhaseEnv, AutonomousTaskConfig, TERRAIN_PROFILES
from serpworm.physics.pipe_geometry import (
    make_bend90_pipe,
    make_off_pipe,
    make_right_angle_duct,
    make_scurve_pipe,
    make_straight_pipe,
    make_taper_pipe,
    make_triple_wave_pipe,
    make_uturn_pipe,
    make_z_duct,
    pipe_arc_progress_np,
)
from serpworm.rl.models import ProprioceptiveEstimatorActorCritic

__all__ = ['EVAL_V6_TRACKS', 'EVAL_V7_CURVED_TRACKS', 'EVAL_V8_DIVERSE_TRACKS', 'evaluate_autonomous_v6', 'main']

# 6 deterministic paper evaluation tracks (100% mode-free, zero external FSM!):
EVAL_V6_TRACKS = (
    'open_hard_140',
    'open_hard_turn_L',
    'pipe_3zone_soil',
    'pipe_3zone_open',
    'soft_soil_100',
    'carpet_100',
)

# 9 deterministic v7 tracks adding 90-deg circular arc, S-curve double-arc, and sharp 90-deg L-duct!
EVAL_V7_CURVED_TRACKS = (
    'open_hard_140',
    'open_hard_turn_L',
    'straight_3zone_soil',
    'straight_3zone_open',
    'bend90_arc_L',
    'scurve_arc_L',
    'right_angle_L_duct',
    'soft_soil_100',
    'carpet_100',
)

# 15 deterministic v8 tracks covering ALL 8 long/wide-span pipes + gravel & grass natural terrains (zero steps!)
EVAL_V8_DIVERSE_TRACKS = (
    'open_hard_140',
    'open_hard_turn_L',
    'straight_tight_218',
    'straight_wide_296',
    'bend90_long_L',
    'scurve_long_L',
    'right_angle_L_duct',
    'uturn_180_L',
    'taper_308_to_220',
    'double_L_Z_duct',
    'triple_wave_long',
    'gravel_100',
    'grass_100',
    'soft_soil_100',
    'carpet_100',
)


def evaluate_autonomous_v6(
    ckpt_path: Path,
    out_dir: Path,
    seconds: float = 28.0,
    plate_solver: str = 'fused_cuda',
    save_traj: bool = False,
    curved: bool | None = None,
) -> dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hw_fields = {f.name for f in dc_fields(HardwareConfig)}
    hw = HardwareConfig(**{k: v for k, v in ckpt['hardware'].items() if k in hw_fields})
    t_fields = {f.name for f in dc_fields(AutonomousTaskConfig)}
    td = {k: v for k, v in ckpt['task'].items() if k in t_fields}
    td['episode_seconds'] = seconds + 5.0
    td['command_switch_s'] = 1e6
    task = AutonomousTaskConfig(**td)

    use_v8 = bool(getattr(task, 'diverse_v8', False) and curved is None)
    use_curved = bool((task.curved_pipes if curved is None else curved) and not use_v8)
    if use_v8:
        hw.strip_node_contact = True
        task.diverse_v8 = True
    elif use_curved:
        hw.strip_node_contact = True
        task.curved_pipes = True
    tracks = EVAL_V8_DIVERSE_TRACKS if use_v8 else (EVAL_V7_CURVED_TRACKS if use_curved else EVAL_V6_TRACKS)
    n = len(tracks)
    env = AutonomousPhaseEnv(
        n=n, hardware=hw, task=task, dr=DomainRandConfig(enabled=False),
        device=device, plate_solver=plate_solver if device == 'cuda' else 'segment_lu',
        noise=False, forced_regime=0,
    )
    model = ProprioceptiveEstimatorActorCritic(
        frame_size=ckpt['frame_size'],
        history_len=ckpt['history_len'],
        privileged_size=ckpt['privileged_size'],
        est_size=ckpt['est_size'],
        action_size=ckpt['action_size'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Configure deterministic test tracks explicitly
    env.c.wall_cfg.zero_()
    env.c.wall_cfg[:, 1] = 1e6
    env.c.wall_cfg[:, 2] = 1e6
    env.c.wall_cfg[:, 3] = -1e6
    env.c.wall_cfg[:, 4] = 0.0
    off_p = torch.tensor(make_off_pipe(), device=device, dtype=env.dtype)
    env.c.pipe_cfg.copy_(off_p[None, :].expand(n, -1))
    env.regime_id.zero_()
    env.post_pipe_soil.zero_()

    pipe_specs_eval: dict[int, list[float]] = {}
    if use_v8:
        env.cmd[0] = torch.tensor([0.140, 0.0, 0.0], device=device, dtype=env.dtype)
        env.cmd[1] = torch.tensor([0.095, +0.20, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[2] = make_straight_pipe(x_entry=-1.00, x_exit=-2.50, r_in=0.109)  # tight 218 mm, 1.5 m
        env.regime_id[2] = 1; env.cmd[2] = torch.tensor([0.125, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[3] = make_straight_pipe(x_entry=-1.00, x_exit=-2.60, r_in=0.148)  # wide 296 mm, 1.6 m
        env.regime_id[3] = 1; env.cmd[3] = torch.tensor([0.145, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[4] = make_bend90_pipe(x_entry=-1.00, x_bend=-1.35, R_bend=0.56, r_in=0.124, bend_sgn=+1.0, L_exit=1.10)
        env.regime_id[4] = 2; env.cmd[4] = torch.tensor([0.125, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[5] = make_scurve_pipe(x_entry=-1.00, x_bend=-1.30, R_bend=0.56, bend_angle_deg=65.0, r_in=0.124, bend_sgn=+1.0, L_exit=1.05)
        env.regime_id[5] = 3; env.cmd[5] = torch.tensor([0.125, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[6] = make_right_angle_duct(x_entry=-1.00, x_corner=-1.45, width=0.256, height=0.245, bend_sgn=+1.0, L_exit=1.10, r_fillet=0.008)
        env.regime_id[6] = 4; env.cmd[6] = torch.tensor([0.115, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[7] = make_uturn_pipe(x_entry=-1.00, x_bend=-1.32, R_bend=0.54, r_in=0.126, bend_sgn=+1.0, L_exit=1.05)
        env.regime_id[7] = 5; env.cmd[7] = torch.tensor([0.125, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[8] = make_taper_pipe(x_entry=-1.00, x_taper_start=-1.32, L_taper=0.68, r_in=0.154, r_out=0.110, L_exit=0.95)
        env.regime_id[8] = 6; env.cmd[8] = torch.tensor([0.125, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[9] = make_z_duct(x_entry=-1.00, x_corner=-1.42, L_mid=0.70, width=0.264, height=0.250, bend_sgn=+1.0, L_exit=0.95, r_fillet=0.008)
        env.regime_id[9] = 7; env.cmd[9] = torch.tensor([0.115, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[10] = make_triple_wave_pipe(x_entry=-1.00, x_bend=-1.26, R_bend=0.54, bend_angle_deg=56.0, r_in=0.126, bend_sgn=+1.0, L_exit=0.90)
        env.regime_id[10] = 8; env.cmd[10] = torch.tensor([0.125, 0.0, 0.0], device=device, dtype=env.dtype)
        for tidx, pspec in pipe_specs_eval.items():
            env.c.pipe_cfg[tidx] = torch.tensor(pspec, device=device, dtype=env.dtype)
        for tidx, terr_id, gg in (
            (11, 3, [0.02, -0.02, 0.009, 24.0, 24.0]),   # gravel
            (12, 4, [-0.03, 0.02, 0.012, 10.0, 10.0]),   # grass
            (13, 2, [0.0, 0.0, 0.0, 10.0, 10.0]),        # soft_soil
            (14, 1, [0.0, 0.0, 0.0, 10.0, 10.0]),        # carpet
        ):
            m = torch.zeros(n, device=device, dtype=torch.bool); m[tidx] = True
            env.regime_id[tidx] = 9
            env._apply_terrain_ids(m, torch.full((n,), terr_id, device=device, dtype=torch.long))
            env.c.ground_geom[tidx] = torch.tensor(gg, device=device, dtype=env.dtype)
            env.c.align_pose_to_ground(m)
            env.cmd[tidx] = torch.tensor([0.105, 0.0, 0.0], device=device, dtype=env.dtype)
    elif not use_curved:
        env.cmd[0] = torch.tensor([0.140, 0.0, 0.0], device=device, dtype=env.dtype)
        env.cmd[1] = torch.tensor([0.090, +0.20, 0.0], device=device, dtype=env.dtype)
        env.regime_id[2] = 1
        env.post_pipe_soil[2] = True
        env.c.wall_cfg[2] = torch.tensor([0.0, 0.018, -1.10, -2.05, 3200.0], device=device, dtype=env.dtype)
        env.cmd[2] = torch.tensor([0.130, 0.0, 0.0], device=device, dtype=env.dtype)
        env.regime_id[3] = 1
        env.post_pipe_soil[3] = False
        env.c.wall_cfg[3] = torch.tensor([0.0, 0.018, -1.10, -2.05, 3200.0], device=device, dtype=env.dtype)
        env.cmd[3] = torch.tensor([0.155, 0.0, 0.0], device=device, dtype=env.dtype)
        env.regime_id[4] = 3
        env._apply_terrain_ids(torch.tensor([False, False, False, False, True, False], device=device), torch.full((n,), 2, device=device, dtype=torch.long))
        env.cmd[4] = torch.tensor([0.100, 0.0, 0.0], device=device, dtype=env.dtype)
        env.regime_id[5] = 3
        env._apply_terrain_ids(torch.tensor([False, False, False, False, False, True], device=device), torch.full((n,), 1, device=device, dtype=torch.long))
        env.cmd[5] = torch.tensor([0.100, 0.0, 0.0], device=device, dtype=env.dtype)
    else:
        env.cmd[0] = torch.tensor([0.140, 0.0, 0.0], device=device, dtype=env.dtype)
        env.cmd[1] = torch.tensor([0.090, +0.20, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[2] = make_straight_pipe(x_entry=-1.05, x_exit=-2.00, r_in=0.118)
        env.regime_id[2] = 1
        env.post_pipe_soil[2] = True
        env.cmd[2] = torch.tensor([0.130, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[3] = make_straight_pipe(x_entry=-1.05, x_exit=-2.00, r_in=0.118)
        env.regime_id[3] = 1
        env.post_pipe_soil[3] = False
        env.cmd[3] = torch.tensor([0.155, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[4] = make_bend90_pipe(x_entry=-1.00, x_bend=-1.38, R_bend=0.50, r_in=0.122, bend_sgn=+1.0, L_exit=0.75)
        env.regime_id[4] = 2
        env.cmd[4] = torch.tensor([0.120, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[5] = make_scurve_pipe(x_entry=-1.00, x_bend=-1.32, R_bend=0.52, bend_angle_deg=65.0, r_in=0.122, bend_sgn=+1.0, L_exit=0.65)
        env.regime_id[5] = 3
        env.cmd[5] = torch.tensor([0.120, 0.0, 0.0], device=device, dtype=env.dtype)
        pipe_specs_eval[6] = make_right_angle_duct(x_entry=-1.00, x_corner=-1.48, width=0.256, height=0.245, bend_sgn=+1.0, L_exit=0.80, r_fillet=0.008)
        env.regime_id[6] = 4
        env.cmd[6] = torch.tensor([0.110, 0.0, 0.0], device=device, dtype=env.dtype)
        for tidx, pspec in pipe_specs_eval.items():
            env.c.pipe_cfg[tidx] = torch.tensor(pspec, device=device, dtype=env.dtype)
        m_soil = torch.zeros(n, device=device, dtype=torch.bool); m_soil[7] = True
        env.regime_id[7] = 5
        env._apply_terrain_ids(m_soil, torch.full((n,), 2, device=device, dtype=torch.long))
        env.cmd[7] = torch.tensor([0.100, 0.0, 0.0], device=device, dtype=env.dtype)
        m_carp = torch.zeros(n, device=device, dtype=torch.bool); m_carp[8] = True
        env.regime_id[8] = 5
        env._apply_terrain_ids(m_carp, torch.full((n,), 1, device=device, dtype=torch.long))
        env.cmd[8] = torch.tensor([0.100, 0.0, 0.0], device=device, dtype=env.dtype)

    _, _, _, _, pkt = env._read_sensors(torch.ones(n, device=device, dtype=torch.bool))
    env.history.copy_(env._build_frame(pkt)[:, None, :])
    obs = env.history.clone()

    steps = round(seconds / task.policy_dt_s)
    vx_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    wz_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    yaw_rms_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    ten_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    wall_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    strip_wall_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    min_sz_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    slip_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    in_pipe_buf = torch.empty((steps, n), device=device, dtype=torch.bool)
    post_pipe_buf = torch.empty((steps, n), device=device, dtype=torch.bool)
    est_pred_buf = torch.empty((steps, n, 7), device=device, dtype=torch.float32)
    est_tgt_buf = torch.empty((steps, n, 7), device=device, dtype=torch.float32)
    clr_buf = torch.empty((steps, n), device=device, dtype=torch.float64)
    failed_buf = torch.zeros(n, device=device, dtype=torch.bool)
    x_rec = torch.empty((steps, n, env.c.D, 7), device=device, dtype=torch.float32) if save_traj else None
    p0_gpu = env.c.x[:, :10, :3].clone()

    with torch.no_grad():
        for k in range(steps):
            if x_rec is not None:
                x_rec[k].copy_(env.c.x.float())
            act, est_p = model.action_with_estimate(obs)
            est_pred_buf[k].copy_(est_p)
            obs, _, est_t, _, term, _, _, _, info = env.step(act)
            est_tgt_buf[k].copy_(est_t)
            failed_buf.logical_or_(term)
            vx_buf[k].copy_(info['body_vel'][:, 0] * 1000.0)
            wz_buf[k].copy_(info['body_vel'][:, 2])
            yaw_rms_buf[k].copy_(info['yaw_rms_deg'])
            ten_buf[k].copy_(info['ten_act'])
            wall_buf[k].copy_(info['wall_mean_n'])
            strip_wall_buf[k].copy_(info.get('strip_wall_mean_n', torch.zeros_like(info['wall_mean_n'])))
            min_sz_buf[k].copy_(info.get('min_strip_z_m', torch.full_like(info['clearance'], 0.05)) * 1000.0)
            slip_buf[k].copy_(info['slip'] * 1000.0)
            in_pipe_buf[k].copy_(info['in_pipe'])
            post_pipe_buf[k].copy_(info['post_pipe'])
            clr_buf[k].copy_(info['clearance'] * 1000.0)

    p0 = p0_gpu.cpu().numpy()
    p1 = env.c.x[:, :10, :3].cpu().numpy()
    vx_h = vx_buf.cpu().numpy()
    wz_h = wz_buf.cpu().numpy()
    yw_h = yaw_rms_buf.cpu().numpy()
    te_h = ten_buf.cpu().numpy()
    wl_h = wall_buf.cpu().numpy()
    sw_h = strip_wall_buf.cpu().numpy()
    sz_h = min_sz_buf.cpu().numpy()
    sl_h = slip_buf.cpu().numpy()
    ip_h = in_pipe_buf.cpu().numpy()
    pp_h = post_pipe_buf.cpu().numpy()
    ep_h = est_pred_buf.cpu().numpy()
    et_h = est_tgt_buf.cpu().numpy()
    cl_h = clr_buf.cpu().numpy()
    fl = failed_buf.cpu().numpy()

    vx_scale_mm = max(50.0, task.vx_max_mm_s)
    results = []
    for i, name in enumerate(tracks):
        fwd0 = p0[i, 9, :2] - p0[i, 0, :2]
        fwd0 /= max(float(np.linalg.norm(fwd0)), 1e-9)
        fwd1 = p1[i, 9, :2] - p1[i, 0, :2]
        fwd1 /= max(float(np.linalg.norm(fwd1)), 1e-9)
        heading_turn_deg = math.degrees(math.atan2(fwd0[0] * fwd1[1] - fwd0[1] * fwd1[0], float(fwd0 @ fwd1)))
        disp = float((p1[i, :, :2].mean(0) - p0[i, :, :2].mean(0)) @ fwd0) * 1000.0
        row = {
            'track': name,
            'failed': bool(fl[i]),
            'net_fwd_mm': round(disp, 1),
            'heading_turn_deg': round(heading_turn_deg, 1),
            'mean_vx_mm_s': round(float(vx_h[steps // 4:, i].mean()), 1),
            'mean_wz_rad_s': round(float(wz_h[steps // 4:, i].mean()), 4),
            'mean_yaw_rms_deg': round(float(yw_h[steps // 4:, i].mean()), 1),
            'mean_ten_act': round(float(te_h[steps // 4:, i].mean()), 2),
            'mean_slip_mm_s': round(float(sl_h[steps // 4:, i].mean()), 1),
            'min_clr_mm': round(float(cl_h[:, i].min()), 2),
            'min_strip_z_mm': round(float(sz_h[:, i].min()), 1),
            'est_vx_mae_mm_s': round(float(np.abs(ep_h[:, i, 0] - et_h[:, i, 0]).mean() * vx_scale_mm), 2),
            'est_wall_mae': round(float(np.abs(ep_h[:, i, 3] - et_h[:, i, 3]).mean()), 3),
            'est_drag_mae': round(float(np.abs(ep_h[:, i, 5] - et_h[:, i, 5]).mean()), 3),
        }
        if any(k in name for k in ('pipe', 'bend', 'scurve', 'duct', 'straight', 'uturn', 'taper', 'wave')):
            pre = (~ip_h[:, i]) & (~pp_h[:, i])
            inp = ip_h[:, i]
            pst = pp_h[:, i] & (~ip_h[:, i])
            row['pre_pipe_yaw_deg'] = round(float(yw_h[pre, i].mean()), 1) if pre.any() else None
            row['in_pipe_vx_mm_s'] = round(float(vx_h[inp, i].mean()), 1) if inp.any() else None
            row['in_pipe_yaw_deg'] = round(float(yw_h[inp, i].mean()), 1) if inp.any() else None
            row['in_pipe_ten_act'] = round(float(te_h[inp, i].mean()), 2) if inp.any() else None
            row['in_pipe_wall_n'] = round(float(wl_h[inp, i].mean()), 2) if inp.any() else None
            row['in_pipe_strip_wall_n'] = round(float(sw_h[inp, i].mean()), 2) if inp.any() else None
            row['in_pipe_slip_mm_s'] = round(float(sl_h[inp, i].mean()), 1) if inp.any() else None
            row['post_pipe_vx_mm_s'] = round(float(vx_h[pst, i].mean()), 1) if pst.any() else None
            row['post_pipe_yaw_deg'] = round(float(yw_h[pst, i].mean()), 1) if pst.any() else None
            row['exited_pipe'] = bool(pst.any())
            if i in pipe_specs_eval:
                s_head, s_tot = pipe_arc_progress_np(p1[i, 9, :2], pipe_specs_eval[i])
                row['head_arc_s_mm'] = round(s_head * 1000.0, 1)
                row['pipe_total_mm'] = round(s_tot * 1000.0, 1)
                row['head_arc_pct'] = round(100.0 * s_head / max(s_tot, 1e-6), 1)
        results.append(row)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'checkpoint': str(ckpt_path),
        'include_q_err': bool(ckpt.get('include_q_err', False)),
        'frame_size': int(ckpt['frame_size']),
        'rollout': int(ckpt.get('rollout', -1)),
        'seconds': seconds,
        'results': results,
    }
    (out_dir / 'eval_v6_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if x_rec is not None:
        np.savez_compressed(
            out_dir / 'eval_v6_traj.npz',
            x=x_rec.cpu().numpy(),
            vx_mm_s=vx_h, yaw_rms_deg=yw_h, ten_act=te_h, wall_n=wl_h, slip_mm_s=sl_h,
            in_pipe=ip_h, post_pipe=pp_h, est_pred=ep_h, est_tgt=et_h,
            strip_nodes=env.c.nodes.cpu().numpy(),
            transforms=env.c.transforms.cpu().numpy(),
            mounts=env.c.mounts.cpu().numpy(),
            anchors=env.c.anchors.cpu().numpy(),
            guides=env.c.guides.cpu().numpy(),
            fps=round(1.0 / task.policy_dt_s),
        )
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description='Evaluate v6 pure-proprioceptive mode-free policy')
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=28.0)
    p.add_argument('--save-traj', action='store_true')
    p.add_argument('--curved', action=argparse.BooleanOptionalAction, default=None)
    args = p.parse_args(argv)
    res = evaluate_autonomous_v6(args.checkpoint, args.out, seconds=args.seconds, save_traj=args.save_traj, curved=args.curved)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()

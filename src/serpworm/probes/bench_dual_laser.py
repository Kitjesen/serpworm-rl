"""1:1 Digital Twin of Experiments 126–128 (Dual-Laser Single-Segment Retraction & Deflection).

Reproduces the real optical-breadboard bench setup:
- Single 8-strip pre-curved spring-steel segment simulated with FP64 corotational beams (GPUCore).
- Back plate rigidly clamped to vertical aluminum extrusion column; front plate cantilevered freely.
- Calibrated eccentric-crank horn kinematics (zero fake pulley wrap-around) matching the measured
  0° -> 180° -> 0° S-curve (T = 1.40 s, 25 kg*cm servos):
    * Exp 128 (Bilateral) : center ~36.8 mm, near-zero deflection
    * Exp 127 (Left only) : Ch2 ~35 mm, Ch1 ~13 mm, center ~24 mm, leftward bend
    * Exp 126 (Right only): Ch1 ~35 mm, Ch2 ~13 mm, center ~24 mm, rightward bend
- Exact world-fixed parallel dual-laser ray-plane intersection (Ch2 left +36 mm, Ch1 right -36 mm).
- Photorealistic 3D MuJoCo studio render matching the physical test bench (M6 breadboard, dark
  backdrop, aluminum extrusion mount, CAD STL plates, 25 kg purple servos, 8 deforming steel strips,
  4 tendon cords, dual scissor-jack laser sensors, live red laser beams & hit spots).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import torch
import mujoco

from serpworm.config import HardwareConfig
from serpworm.paths import ASSETS_DIR
from serpworm.physics.gpu_core import GPUCore, cross, local_matrices_torch, matrix, mul, norm, rotate, unit
from serpworm.physics.topology import build_topology

__all__ = ['BENCH_PARAMS', 'REAL_EXP_REF', 'simulate_dual_laser_bench', 'render_bench_mujoco_video', 'plot_bench_validation']

MESH_DIR = ASSETS_DIR / 'cad' / 'meshes' / 'longworm2'

# Jointly calibrated to:
#  1) Universal Testing Machine (UTM) 5-cycle compression (0-45 mm, 0-42.7 N): bow=30.0 mm, t=0.220 mm (RMSE=0.09 N)
#  2) Exp 126-128 dual-laser bilateral & unilateral deflection (0°->180°->0°, T=1.40 s, 25 kg*cm servos)
BENCH_PARAMS = {
    'strip_thickness_m': 0.000220,    # UTM-identified effective steel thickness (RMSE=0.09 N across 0-45 mm)
    'stress_free_bow_m': 0.0300,      # UTM-identified stress-free arch bow (matches ruler photos & nonlinear softening)
    'horn_radius_m': 0.0250,          # eccentric horn tie radius (matches 36.8 mm bilateral peak at 180°)
    'guide_dist_m': 0.0360,           # back-plate exit guide radial distance
    'tendon_slack_m': 0.0114,         # initial rope slack deadband (matches ~0°-25° flat toe)
    'tendon_stiffness_n_m': 8000.0,   # braided PE tendon + bracket effective axial stiffness
    'tendon_damping_ns_m': 2.5,
    'tendon_y_m': 0.0285,             # exact CAD tendon anchor lateral offset (+/-28.5 mm)
    'front_plate_mass_kg': 0.115,     # acrylic front plate + 1 x 25 kg yaw servo (no front wheels)
    'stop_gap_m': 0.068,
    'laser_half_spacing_m': 0.036,    # Ch2 at +36 mm (left), Ch1 at -36 mm (right); 72 mm total spacing
    'laser_z_offset_m': -0.013,       # real laser beams hit ~13 mm below front-plate center (aligned with lower-mid yaw servo)
    'cycle_period_s': 1.40,           # measured Exp 126-128 cycle period (0° -> 180° -> 0°)
    'servo_speed_rad_s': 8.2,         # measured 25 kg servo peak speed (~470 deg/s)
}

# Measured real bench reference values from 实验126-128_双激光伸缩偏转.pptx
REAL_EXP_REF = {
    'exp128_bilateral': {
        'ch1_peak_mm': 37.2, 'ch2_peak_mm': 36.4, 'center_peak_mm': 36.8,
        'angles_deg': [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0],
        'center_mm':  [0.0,  0.3,  2.6, 10.5, 21.0,  30.2,  34.8,  36.4,  36.8],
    },
    'exp127_left': {
        'ch1_peak_mm': 12.8, 'ch2_peak_mm': 35.1, 'center_peak_mm': 23.9,
        'angles_deg': [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0],
        'center_mm':  [0.0,  0.2,  1.6,  6.8, 13.8,  19.8,  22.7,  23.7,  23.9],
    },
    'exp126_right': {
        'ch1_peak_mm': 35.2, 'ch2_peak_mm': 13.2, 'center_peak_mm': 24.2,
        'angles_deg': [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0],
        'center_mm':  [0.0,  0.2,  1.8,  7.2, 14.2,  20.1,  23.0,  24.0,  24.2],
    },
}


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def _crank_step(
    core: GPUCore,
    target_rad: torch.Tensor,
    p1_ref: torch.Tensor,
    pm_custom: torch.Tensor,
    params: dict,
) -> None:
    """Single FP64 substep with clamped rear plate (plate 1) and calibrated crank-arm tendons."""
    h = core.dt
    r_m = params['horn_radius_m']
    g_m = params['guide_dist_m']
    sl_m = params['tendon_slack_m']
    k_ten = params['tendon_stiffness_n_m']
    c_ten = params['tendon_damping_ns_m']

    B = core.batch
    x = core.x[:, :2]
    v = core.v[:, :2]
    q = x[..., 3:]
    R = matrix(q)
    f = torch.zeros_like(core.v)

    # Torque- & speed-limited 25 kg servo update (smooth crank lever arm r*g*sin(sv)/chord)
    sv = core.servo
    chord = torch.sqrt(g_m * g_m + r_m * r_m - 2.0 * g_m * r_m * torch.cos(sv)).clamp_min(1e-9)
    lever = (r_m * g_m * torch.sin(sv).abs() / chord).clamp_max(r_m)
    load = core.tension.reshape(B, 2, 2).sum(-1) * lever
    s_max = core.servo_speed_limit[:, None] * h
    inc = (target_rad - sv).clamp(-s_max, s_max)
    inc = inc * torch.where(inc > 0.0, (1.0 - load / core.servo_torque_limit[:, None]).clamp(0.0, 1.0), 1.0)
    core.servo.add_(inc)

    chord = torch.sqrt(g_m * g_m + r_m * r_m - 2.0 * g_m * r_m * torch.cos(core.servo))
    takeup = chord - (g_m - r_m)

    ar = rotate(q[:, 0::2, None, :].expand(-1, -1, 4, -1), core.anchors.expand(B, 1, -1, -1))
    br = rotate(q[:, 1::2, None, :].expand(-1, -1, 4, -1), core.guides.expand(B, 1, -1, -1))
    delta = x[:, 1::2, None, :3] + br - x[:, 0::2, None, :3] - ar
    length = norm(delta)
    direction = delta / length.clamp_min(1e-15)
    va = v[:, 0::2, None, :3] + cross(v[:, 0::2, None, 3:].expand_as(ar), ar)
    vb = v[:, 1::2, None, :3] + cross(v[:, 1::2, None, 3:].expand_as(br), br)

    rest = (0.101 + sl_m) - takeup.reshape(B, 1, 2).repeat_interleave(2, -1)
    ext = length[..., 0] - rest
    spd = ((vb - va) * direction).sum(-1)
    core.tension.copy_(torch.where(ext >= 0.0, (k_ten * ext + c_ten * spd).clamp_min(0.0), 0.0))

    force = core.tension[..., None] * direction
    f[:, 0:2:2, :3] += force.sum(-2)
    f[:, 1:2:2, :3] -= force.sum(-2)
    f[:, 0:2:2, 3:] += cross(ar, force).sum(-2)
    f[:, 1:2:2, 3:] -= cross(br, force).sum(-2)

    sep = x[:, 1::2, :3] - x[:, 0::2, :3]
    axis = -R[:, 0::2, :, 0]
    gap = (sep * axis).sum(-1)
    push = 50000.0 * (core.stop_gap - gap).clamp_min(0.0)[..., None] * axis
    f[:, 1:2:2, :3] += push
    f[:, 0:2:2, :3] -= push
    f[:, 0:2:2, 3:] -= cross(sep, push)

    # Mild aerodynamic / structural hysteresis damping on free cantilevered plate
    f[:, 0, :3] -= 0.8 * v[:, 0, :3]
    f[:, 0, 3:] -= 0.025 * v[:, 0, 3:]

    elastic, K, M = core.beam()
    vg = (core.v + h * core.gravity)[:, core.edges].reshape(B, core.S, core.E, 12)
    rhs = core.scatter_vector((M @ vg[..., None])[..., 0]) + h * (f + elastic)
    rhs[:, :2] += (pm_custom @ (core.v[:, :2] + h * core.gravity[:, :2])[..., None])[..., 0]
    A = (1.0 + h * 0.05) * M + (h * 0.002 + h * h) * K
    vnew = core.solve(A, rhs, (1.0 + h * 0.05) * pm_custom)

    core.v.copy_(vnew)
    core.v[:, 1] = 0.0
    core.x[..., :3].add_(h * core.v[..., :3])
    omega = 0.5 * h * core.v[..., 3:]
    theta = norm(omega)
    exp = torch.cat((omega * torch.sinc(theta / math.pi), torch.cos(theta)), -1)
    core.x[..., 3:].copy_(unit(mul(exp, core.x[..., 3:])))
    core.x[:, 1] = p1_ref


def simulate_dual_laser_bench(
    cycles: float = 2.0,
    fps: int = 30,
    dt: float = 0.0005,
    params: dict | None = None,
    device: str = 'cpu',
) -> dict:
    """Run all 3 bench experiments (Exp 128 bilateral, Exp 127 left, Exp 126 right) in parallel."""
    p = dict(BENCH_PARAMS)
    if params:
        p.update(params)

    hw = HardwareConfig.ds3218_default()
    hw.servo_speed_limit_rad_s = p['servo_speed_rad_s']
    hw.strip_thickness_m = p['strip_thickness_m']
    hw.stress_free_bow_m = p['stress_free_bow_m']
    core = GPUCore(
        count=1, batch=3, dt=dt, device=device,
        plate_solver='segment_lu', ground_contact=False, hardware=hw,
    )
    yt = p['tendon_y_m']
    core.anchors[:, 1] = torch.tensor([yt, yt, -yt, -yt], device=core.device, dtype=core.dtype)
    core.guides[:, 1] = core.anchors[:, 1]
    core.stop_gap = p['stop_gap_m']

    pm_custom = core.plate_mass()
    pm_custom[:, 0] *= (p['front_plate_mass_kg'] / 0.28)
    p1_ref = core.x[:, 1].clone()

    # Settle free front plate at 0° rest equilibrium
    zero_tgt = torch.zeros((3, 2), device=core.device, dtype=core.dtype)
    for _ in range(800):
        _crank_step(core, zero_tgt, p1_ref, pm_custom, p)

    # Record world-frame laser baseline (fixed horizontal rays along -X at y = +y_las (Ch2) and -y_las (Ch1), z = z0_c + z_off)
    y_las = p['laser_half_spacing_m']
    z_las = core.x[:, 0, 2].clone() + float(p.get('laser_z_offset_m', -0.013))
    x0_c = core.x[:, 0, 0].clone()
    R0_init = matrix(core.x[:, 0, 3:])

    def measure_lasers():
        R0 = matrix(core.x[:, 0, 3:])
        # Front plate face normal (world +X in plate frame)
        nx, ny, nz = R0[:, 0, 0], R0[:, 1, 0], R0[:, 2, 0]
        cx, cy, cz = core.x[:, 0, 0], core.x[:, 0, 1], core.x[:, 0, 2]
        # Exact intersection of fixed rays y=+y_las (Ch2, Left) and y=-y_las (Ch1, Right) at z=z_las
        xh_L = cx - (ny * (+y_las - cy) + nz * (z_las - cz)) / nx.clamp_min(1e-6)
        xh_R = cx - (ny * (-y_las - cy) + nz * (z_las - cz)) / nx.clamp_min(1e-6)
        return xh_L, xh_R, torch.rad2deg(torch.atan2(R0[:, 1, 0], R0[:, 0, 0]))

    xh_L0, xh_R0, _ = measure_lasers()

    T_cyc = p['cycle_period_s']
    total_s = cycles * T_cyc
    n_frames = int(round(total_s * fps))
    substeps_per_frame = max(1, int(round(1.0 / (fps * dt))))

    x_hist = np.empty((n_frames, 3, core.D, 7), dtype=np.float32)
    ch2_hist = np.empty((n_frames, 3), dtype=np.float64)   # Left laser (+y_las), mm
    ch1_hist = np.empty((n_frames, 3), dtype=np.float64)   # Right laser (-y_las), mm
    hit_x_hist = np.empty((n_frames, 3, 2), dtype=np.float64)  # world X of [Ch2_L, Ch1_R] hit points
    yaw_hist = np.empty((n_frames, 3), dtype=np.float64)
    cmd_deg_hist = np.empty((n_frames,), dtype=np.float64)
    servo_deg_hist = np.empty((n_frames, 3, 2), dtype=np.float64)
    t_hist = np.empty((n_frames,), dtype=np.float64)

    for fi in range(n_frames):
        t_now = fi / fps
        # Smooth triangular 0° -> 180° -> 0° linear ramp matching bench servo sweep
        phase = (t_now % T_cyc) / T_cyc
        tri = 2.0 * phase if phase <= 0.5 else 2.0 * (1.0 - phase)
        cmd_rad = tri * math.pi
        tgt = torch.tensor(
            [[cmd_rad, cmd_rad], [cmd_rad, 0.0], [0.0, cmd_rad]],
            device=core.device, dtype=core.dtype,
        )
        for _ in range(substeps_per_frame):
            _crank_step(core, tgt, p1_ref, pm_custom, p)

        xh_L, xh_R, yaw_deg = measure_lasers()
        x_hist[fi] = core.x.cpu().numpy().astype(np.float32)
        ch2_hist[fi] = ((xh_L0 - xh_L) * 1000.0).cpu().numpy()
        ch1_hist[fi] = ((xh_R0 - xh_R) * 1000.0).cpu().numpy()
        hit_x_hist[fi, :, 0] = xh_L.cpu().numpy()
        hit_x_hist[fi, :, 1] = xh_R.cpu().numpy()
        yaw_hist[fi] = yaw_deg.cpu().numpy()
        cmd_deg_hist[fi] = math.degrees(cmd_rad)
        servo_deg_hist[fi] = torch.rad2deg(core.servo).cpu().numpy()
        t_hist[fi] = t_now

    # Compute summary metrics from the steady 2nd cycle (or last cycle)
    mid = n_frames // 2 if cycles >= 1.5 else 0
    names = ('exp128_bilateral', 'exp127_left', 'exp126_right')
    summary = {}
    for i, nm in enumerate(names):
        c1 = ch1_hist[mid:, i]
        c2 = ch2_hist[mid:, i]
        ctr = 0.5 * (c1 + c2)
        yw = yaw_hist[mid:, i]
        summary[nm] = {
            'ch1_peak_mm': round(float(c1.max()), 2),
            'ch2_peak_mm': round(float(c2.max()), 2),
            'center_peak_mm': round(float(ctr.max()), 2),
            'physical_yaw_peak_deg': round(float(yw[np.argmax(np.abs(yw))]), 2),
            'ref_ch1_mm': REAL_EXP_REF[nm]['ch1_peak_mm'],
            'ref_ch2_mm': REAL_EXP_REF[nm]['ch2_peak_mm'],
            'ref_center_mm': REAL_EXP_REF[nm]['center_peak_mm'],
        }

    return {
        'x': x_hist,
        't': t_hist,
        'ch1_mm': ch1_hist,
        'ch2_mm': ch2_hist,
        'hit_x': hit_x_hist,
        'yaw_deg': yaw_hist,
        'cmd_deg': cmd_deg_hist,
        'servo_deg': servo_deg_hist,
        'laser_z': float(z_las[0].item()),
        'laser_y': y_las,
        'strip_nodes': core.nodes.cpu().numpy(),
        'transforms': core.transforms.cpu().numpy(),
        'mounts': core.mounts.cpu().numpy(),
        'anchors': core.anchors.cpu().numpy(),
        'guides': core.guides.cpu().numpy(),
        'fps': fps,
        'params': p,
        'summary': summary,
    }


def _make_breadboard_texture(out_png: Path) -> Path:
    """Generate high-res 1024x1024 satin brushed stainless-steel optical breadboard texture (25 mm M6 grid)."""
    from PIL import Image, ImageDraw
    N = 1024
    rng = np.random.default_rng(42)
    base = np.full((N, N, 3), 146, dtype=np.float32)
    # Fine horizontal brushed stainless-steel micro-streaks
    row_grain = rng.normal(0.0, 5.5, size=(N, 1, 1))
    streak = rng.normal(0.0, 3.0, size=(N, N // 8, 1)).repeat(8, axis=1)
    fine = rng.normal(0.0, 1.8, size=(N, N, 1))
    arr = np.clip(base + row_grain + streak + fine, 112, 184).astype(np.uint8)
    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)
    step = N // 8
    for r in range(8):
        for c in range(8):
            cx = c * step + step // 2
            cy = r * step + step // 2
            # Subtle directional metallic chamfer + recessed tapped M6 bore
            draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(166, 170, 174))
            draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(116, 120, 125))
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(60, 63, 68))
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(32, 34, 38))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    return out_png


def _make_studio_textures(out_dir: Path) -> dict[str, Path]:
    """Generate procedural metallic, curtain-fabric, and label textures for photorealistic MuJoCo rendering."""
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {'breadboard': _make_breadboard_texture(out_dir / '_breadboard_tex.png')}

    # 1. Soft charcoal studio backdrop cloth with organic low-frequency drape folds (no regular pinstripes)
    rng = np.random.default_rng(7)
    H, W = 512, 1024
    x = np.linspace(0.0, 1.0, W)[None, :]
    y = np.linspace(0.0, 1.0, H)[:, None]
    folds = (
        5.5 * np.sin(2.0 * np.pi * (1.35 * x + 0.22 * y) + 0.4)
        + 4.0 * np.sin(2.0 * np.pi * (2.60 * x - 0.31 * y) + 1.9)
        + 2.5 * np.cos(2.0 * np.pi * (0.85 * y + 0.55 * x) + 0.7)
        + 2.0 * np.sin(2.0 * np.pi * (4.10 * x + 0.45 * y) + 2.8)
    )
    vign = -8.0 * ((x - 0.46) ** 2 + 0.55 * (y - 0.52) ** 2)
    weave = rng.normal(0.0, 1.4, size=(H, W))
    cloth = np.clip(42.0 + folds + vign + weave, 24, 64).astype(np.uint8)
    cloth_rgb = np.stack([cloth, cloth, np.clip(cloth.astype(np.int16) + 2, 0, 255).astype(np.uint8)], axis=-1)
    p_bg = out_dir / '_backdrop_tex.png'
    Image.fromarray(cloth_rgb).save(p_bg)
    paths['backdrop'] = p_bg

    # 2. Brushed anodized silver aluminum texture (fine extrusion grain)
    rng_a = np.random.default_rng(11)
    A = 512
    col_grain = rng_a.normal(0.0, 6.0, size=(1, A, 1))
    micro = rng_a.normal(0.0, 2.0, size=(A, A, 1))
    alu = np.clip(192.0 + col_grain + micro, 162, 220).astype(np.uint8)
    alu_rgb = np.concatenate([alu, alu, np.clip(alu.astype(np.int16) + 3, 0, 255).astype(np.uint8)], axis=-1)
    p_alu = out_dir / '_alu_tex.png'
    Image.fromarray(alu_rgb).save(p_alu)
    paths['alu'] = p_alu

    # 3. Tempered brushed blue spring-steel texture (65Mn cold-rolled sheen)
    rng_s = np.random.default_rng(19)
    S = 512
    u = np.linspace(0.0, 1.0, S)[None, :]
    band = 14.0 * np.sin(2.0 * np.pi * 2.0 * u) + 7.0 * np.cos(2.0 * np.pi * 5.0 * u)
    grain_s = rng_s.normal(0.0, 4.5, size=(S, 1)) + rng_s.normal(0.0, 2.0, size=(S, S))
    r_c = np.clip(86.0 + 0.8 * band + grain_s, 54, 132).astype(np.uint8)
    g_c = np.clip(116.0 + 0.9 * band + grain_s, 78, 164).astype(np.uint8)
    b_c = np.clip(142.0 + 1.0 * band + grain_s, 98, 194).astype(np.uint8)
    p_st = out_dir / '_steel_tex.png'
    Image.fromarray(np.stack([r_c, g_c, b_c], axis=-1)).save(p_st)
    paths['steel'] = p_st

    # 4. Crisp industrial yellow laser hazard label texture
    lw, lh = 512, 256
    lbl = Image.new('RGB', (lw, lh), (242, 206, 28))
    dl = ImageDraw.Draw(lbl)
    dl.rectangle([10, 10, lw - 10, lh - 10], outline=(24, 24, 26), width=6)
    # Classic laser warning triangle on left
    tri = [(110, 42), (46, 178), (174, 178)]
    dl.polygon(tri, fill=(242, 206, 28), outline=(22, 22, 24))
    for w_off in range(-3, 4):
        dl.line([(110 + w_off, 42), (46 + w_off, 178)], fill=(22, 22, 24), width=2)
        dl.line([(110 + w_off, 42), (174 + w_off, 178)], fill=(22, 22, 24), width=2)
    dl.line([(46, 178), (174, 178)], fill=(22, 22, 24), width=7)
    dl.ellipse([96, 118, 124, 146], fill=(22, 22, 24))
    # Printed technical text bars on right
    for ry, rx1 in [(48, 465), (78, 430), (112, 472), (140, 445), (168, 410), (198, 468)]:
        dl.rectangle([208, ry, rx1, ry + 13], fill=(28, 28, 30))
    p_lbl = out_dir / '_laser_lbl_tex.png'
    lbl.save(p_lbl)
    paths['laser_lbl'] = p_lbl

    return paths


def _build_bench_mjcf(mounts: np.ndarray, tex_paths: dict[str, Path] | Path, laser_z: float, laser_y: float) -> str:
    """Build 1:1 optical-breadboard dual-laser rig MJCF matching Exp 126–128 photos."""
    mesh_dir = MESH_DIR.resolve().as_posix()
    if isinstance(tex_paths, Path):
        tex_paths = _make_studio_textures(tex_paths.parent)
    tex_bb   = tex_paths['breadboard'].resolve().as_posix()
    tex_bg   = tex_paths['backdrop'].resolve().as_posix()
    tex_alu  = tex_paths['alu'].resolve().as_posix()
    tex_st   = tex_paths['steel'].resolve().as_posix()
    tex_lbl  = tex_paths['laser_lbl'].resolve().as_posix()

    # Rear plate (p1, clamped) keeps its 2 wheels resting flush on breadboard (z=0); front plate (p0) has no wheels
    w_geoms_p1 = []
    for s in range(2):
        mx, my, mz = mounts[1, s]
        mz_flush = mz - 0.003  # seat 16 mm wheel tire flush onto z=0 breadboard
        strut_top = -0.064
        strut_mid_z = 0.5 * (strut_top + mz_flush)
        strut_half = max(0.005, 0.5 * abs(strut_top - mz_flush))
        w_geoms_p1.append(
            f'<geom type="box" pos="{mx:.5f} {my:.5f} {strut_mid_z:.5f}" '
            f'size="0.0052 0.0036 {strut_half:.5f}" material="mat_strut" contype="0" conaffinity="0"/>'
        )
        w_geoms_p1.append(
            f'<geom type="mesh" mesh="wheel_mesh" pos="{mx:.5f} {my:.5f} {mz_flush:.5f}" '
            f'material="mat_wheel" contype="0" conaffinity="0"/>'
        )
        # Realistic recessed brushed-bronze/steel 608 ball-bearing hub (flush with wheel face, no cartoon bullseye)
        w_geoms_p1.append(
            f'<geom type="cylinder" pos="{mx:.5f} {my:.5f} {mz_flush:.5f}" euler="1.5708 0 0" '
            f'size="0.0094 0.0082" material="mat_bearing_outer" contype="0" conaffinity="0"/>'
        )
        w_geoms_p1.append(
            f'<geom type="cylinder" pos="{mx:.5f} {my:.5f} {mz_flush:.5f}" euler="1.5708 0 0" '
            f'size="0.0074 0.0084" material="mat_bearing_seal" contype="0" conaffinity="0"/>'
        )
        w_geoms_p1.append(
            f'<geom type="cylinder" pos="{mx:.5f} {my:.5f} {mz_flush:.5f}" euler="1.5708 0 0" '
            f'size="0.0046 0.0086" material="mat_steel_shaft" contype="0" conaffinity="0"/>'
        )
        w_geoms_p1.append(
            f'<geom type="cylinder" pos="{mx:.5f} {my:.5f} {mz_flush:.5f}" euler="1.5708 0 0" '
            f'size="0.0024 0.0087" material="mat_black_anod" contype="0" conaffinity="0"/>'
        )

    # Exact geometrically connected scissor-lift lab jacks + laser displacement heads (matching Exp 126-128 photo)
    lx = 0.285  # scissor jack center X; sensor seated forward on stage with front emitter lens at x = 0.228 m
    jacks_xml = []
    for idx, (side_name, ly) in enumerate([('ch2_left', +laser_y), ('ch1_right', -laser_y)]):
        head_z = laser_z
        stage_top_z = head_z - 0.019
        stage_mid_z = stage_top_z - 0.0050
        base_mid_z = 0.0055
        base_top_z = 0.011

        z_bot = base_top_z + 0.006          # 0.017 m
        z_top = stage_top_z - 0.010 - 0.006 # ~0.084 m
        z_mid = 0.5 * (z_bot + z_top)
        half_x = 0.036
        dx = 2.0 * half_x
        dz = z_top - z_bot
        arm_half_len = 0.5 * math.hypot(dx, dz) + 0.005
        arm_ang = math.atan2(dz, dx)

        arms_and_bolts = []
        for sgn in (1.0, -1.0):
            y_out = sgn * 0.0295
            y_in  = sgn * 0.0242
            y_bolt = sgn * 0.0324
            y_hex  = sgn * 0.0352
            for hx in (-half_x, +half_x):
                arms_and_bolts.append(
                    f'<geom type="box" pos="{hx:.4f} {sgn * 0.027:.4f} {base_top_z + 0.006:.4f}" '
                    f'size="0.010 0.0055 0.006" material="mat_black_anod" contype="0" conaffinity="0"/>'
                )
                arms_and_bolts.append(
                    f'<geom type="box" pos="{hx:.4f} {sgn * 0.027:.4f} {stage_top_z - 0.016:.4f}" '
                    f'size="0.010 0.0055 0.006" material="mat_black_anod" contype="0" conaffinity="0"/>'
                )
            # Anodized machined scissor bars with glossy metallic edge response
            arms_and_bolts.append(
                f'<geom type="box" pos="0 {y_out:.4f} {z_mid:.4f}" euler="0 {-arm_ang:.4f} 0" '
                f'size="{arm_half_len:.4f} 0.0026 0.0070" material="mat_black_metal" contype="0" conaffinity="0"/>'
            )
            arms_and_bolts.append(
                f'<geom type="box" pos="0 {y_in:.4f} {z_mid:.4f}" euler="0 {+arm_ang:.4f} 0" '
                f'size="{arm_half_len:.4f} 0.0026 0.0070" material="mat_black_anod" contype="0" conaffinity="0"/>'
            )
            # 5 stainless M4 socket-head cap screws WITH dark hex inner socket recess
            for bx, bz in [(-half_x, z_bot), (+half_x, z_bot), (0.0, z_mid), (-half_x, z_top), (+half_x, z_top)]:
                arms_and_bolts.append(
                    f'<geom type="cylinder" pos="{bx:.4f} {y_bolt:.4f} {bz:.4f}" euler="1.5708 0 0" '
                    f'size="0.0040 0.0026" material="mat_bolt" contype="0" conaffinity="0"/>'
                )
                arms_and_bolts.append(
                    f'<geom type="cylinder" pos="{bx:.4f} {y_hex:.4f} {bz:.4f}" euler="1.5708 0 0" '
                    f'size="0.0018 0.0004" material="mat_black_anod" contype="0" conaffinity="0"/>'
                )

        screw_z = z_bot + 0.004
        jacks_xml.append(f"""
    <!-- {side_name} precision scissor-lift lab jack + laser displacement sensor -->
    <body name="laser_{side_name}" pos="{lx:.4f} {ly:.4f} 0">
      <geom type="box" pos="0 0 {base_mid_z:.4f}" size="0.054 0.0325 0.0055" material="mat_black_anod" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.006 0 {stage_mid_z:.4f}" size="0.058 0.0325 0.0050" material="mat_black_anod" contype="0" conaffinity="0"/>
      <!-- Cross shafts (brass lower-front guide rail + machined steel pivot shafts) -->
      <geom type="cylinder" pos="0 0 {z_mid:.4f}" euler="1.5708 0 0" size="0.0044 0.0315" material="mat_steel_shaft" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="{-half_x:.4f} 0 {z_bot:.4f}" euler="1.5708 0 0" size="0.0044 0.0315" material="mat_brass" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="{+half_x:.4f} 0 {z_bot:.4f}" euler="1.5708 0 0" size="0.0038 0.0315" material="mat_steel_shaft" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="{-half_x:.4f} 0 {z_top:.4f}" euler="1.5708 0 0" size="0.0038 0.0315" material="mat_steel_shaft" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="{+half_x:.4f} 0 {z_top:.4f}" euler="1.5708 0 0" size="0.0038 0.0315" material="mat_steel_shaft" contype="0" conaffinity="0"/>
      <!-- Leadscrew drive housing, threaded rod & rear knurled adjustment knob -->
      <geom type="box" pos="0.042 0 {screw_z + 0.004:.4f}" size="0.011 0.015 0.012" material="mat_black_metal" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.014 0 {screw_z + 0.004:.4f}" euler="0 1.5708 0" size="0.0040 0.046" material="mat_steel_shaft" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.068 0 {screw_z + 0.004:.4f}" euler="0 1.5708 0" size="0.0182 0.0125" material="mat_knob" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.081 0 {screw_z + 0.004:.4f}" euler="0 1.5708 0" size="0.0072 0.0018" material="mat_bolt" contype="0" conaffinity="0"/>
      {''.join(arms_and_bolts)}
      <!-- Laser displacement sensor housing seated forward on top stage (front lens at world x = 0.228 m) -->
      <geom type="box" pos="-0.016 0 {head_z:.4f}" size="0.040 0.020 0.019" material="mat_sensor_body" contype="0" conaffinity="0"/>
      <!-- Detailed industrial yellow laser warning label on top -->
      <geom type="box" pos="-0.016 0 {head_z + 0.0194:.4f}" size="0.028 0.0145 0.0005" material="mat_yellow_lbl" contype="0" conaffinity="0"/>
      <!-- Subtle dark-anodized recessed side badge (matching real sensor without flat bright sticker look) -->
      <geom type="box" pos="-0.026 -0.0202 {head_z:.4f}" size="0.018 0.0003 0.011" material="mat_sensor_plate" contype="0" conaffinity="0"/>
      <!-- Corner mounting screws on sensor housing -->
      <geom type="cylinder" pos="0.018 -0.0202 {head_z + 0.013:.4f}" euler="1.5708 0 0" size="0.0018 0.0005" material="mat_bolt" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.018 -0.0202 {head_z - 0.013:.4f}" euler="1.5708 0 0" size="0.0018 0.0005" material="mat_bolt" contype="0" conaffinity="0"/>
      <!-- Deep glossy ruby optical emitter/receiver glass window -->
      <geom type="box" pos="-0.0564 0 {head_z:.4f}" size="0.0008 0.011 0.013" material="mat_lens" contype="0" conaffinity="0"/>
    </body>""")

    return f"""<mujoco model="serpworm_exp126_128_bench">
  <compiler angle="radian" meshdir="{mesh_dir}" autolimits="true"/>
  <visual>
    <global offwidth="1280" offheight="720" fovy="35"/>
    <quality shadowsize="4096" offsamples="4"/>
    <!-- Low headlight preserves crisp directional studio shadows & realistic metallic sheen -->
    <headlight ambient="0.18 0.18 0.20" diffuse="0.22 0.22 0.24" specular="0.16 0.16 0.16"/>
  </visual>
  <asset>
    <mesh name="plate_front" file="front2_Link.STL"/>
    <mesh name="plate_back"  file="back2_Link.STL"/>
    <mesh name="wheel_mesh"  file="w2-1_Link.STL"/>
    <texture name="tex_breadboard" type="2d" file="{tex_bb}"/>
    <texture name="tex_backdrop"   type="2d" file="{tex_bg}"/>
    <texture name="tex_alu"        type="2d" file="{tex_alu}"/>
    <texture name="tex_steel"      type="2d" file="{tex_st}"/>
    <texture name="tex_laser_lbl"  type="2d" file="{tex_lbl}"/>
    <material name="mat_breadboard"    texture="tex_breadboard" texrepeat="10 10" texuniform="true" reflectance="0.07" shininess="0.58" specular="0.48"/>
    <material name="mat_backdrop"      texture="tex_backdrop"   texrepeat="1 1"   specular="0.04" shininess="0.06"/>
    <material name="mat_alu"           texture="tex_alu"        texrepeat="2 4"   specular="0.72" shininess="0.68" reflectance="0.14"/>
    <material name="mat_alu_slot"      rgba="0.22 0.24 0.27 1.0" specular="0.28" shininess="0.30"/>
    <material name="mat_spring_steel"  rgba="0.54 0.65 0.74 1.0" specular="1.0"  shininess="0.95" reflectance="0.28"/>
    <material name="mat_acrylic"       rgba="0.88 0.92 0.96 0.28" specular="0.92" shininess="0.90"/>
    <material name="mat_plate_front"   rgba="0.88 0.92 0.96 0.24" specular="0.92" shininess="0.90"/>
    <material name="mat_plate_back"    rgba="0.18 0.20 0.24 1.0" specular="0.55" shininess="0.52"/>
    <material name="mat_servo_purple"  rgba="0.40 0.20 0.60 1.0" specular="0.82" shininess="0.78" reflectance="0.12"/>
    <material name="mat_servo_cap"     rgba="0.13 0.14 0.16 1.0" specular="0.48" shininess="0.45"/>
    <material name="mat_tape_flush"    rgba="0.84 0.85 0.86 0.78" specular="0.18" shininess="0.14"/>
    <material name="mat_pad_white"     rgba="0.89 0.90 0.89 0.94" specular="0.10" shininess="0.08"/>
    <material name="mat_strut"         rgba="0.22 0.23 0.26 1.0" specular="0.45" shininess="0.40"/>
    <material name="mat_wheel"         rgba="0.14 0.14 0.16 1.0" specular="0.25" shininess="0.25"/>
    <material name="mat_bearing_outer" rgba="0.46 0.44 0.41 1.0" specular="0.82" shininess="0.80"/>
    <material name="mat_bearing_seal"  rgba="0.20 0.19 0.18 1.0" specular="0.35" shininess="0.35"/>
    <material name="mat_black_anod"    rgba="0.12 0.13 0.15 1.0" specular="0.68" shininess="0.66"/>
    <material name="mat_black_metal"   rgba="0.19 0.20 0.23 1.0" specular="0.75" shininess="0.72"/>
    <material name="mat_bolt"          rgba="0.70 0.73 0.77 1.0" specular="0.92" shininess="0.90" reflectance="0.22"/>
    <material name="mat_steel_shaft"   rgba="0.56 0.60 0.66 1.0" specular="0.85" shininess="0.82"/>
    <material name="mat_brass"         rgba="0.74 0.60 0.30 1.0" specular="0.84" shininess="0.80"/>
    <material name="mat_knob"          rgba="0.10 0.10 0.12 1.0" specular="0.32" shininess="0.32"/>
    <material name="mat_sensor_body"   rgba="0.12 0.13 0.15 1.0" specular="0.62" shininess="0.60"/>
    <material name="mat_sensor_plate"  rgba="0.26 0.29 0.33 1.0" specular="0.55" shininess="0.55"/>
    <material name="mat_yellow_lbl"    texture="tex_laser_lbl"   texrepeat="1 1" specular="0.25" shininess="0.22"/>
    <material name="mat_lens"          rgba="0.52 0.10 0.12 1.0" specular="0.95" shininess="0.95" reflectance="0.18"/>
  </asset>
  <worldbody>
    <!-- 3-point studio photography lighting + specular softbox catching curved spring-steel arches -->
    <light name="studio_key"     pos="0.14 -0.78 1.05" dir="-0.04 0.56 -0.82" diffuse="0.90 0.89 0.87" specular="0.82 0.82 0.80" castshadow="true"/>
    <light name="studio_rim"     pos="0.18  0.48 0.88" dir="-0.08 -0.58 -0.80" diffuse="0.44 0.48 0.54" specular="0.72 0.74 0.78" castshadow="false"/>
    <light name="studio_spec"    pos="-0.02 -0.52 0.46" dir="0.02 0.76 -0.64" diffuse="0.16 0.17 0.19" specular="0.88 0.90 0.94" castshadow="false"/>
    <light name="studio_fill"    pos="-0.55 -0.52 0.58" dir="0.42 0.44 -0.46" diffuse="0.26 0.28 0.32" specular="0.18 0.18 0.20" castshadow="false"/>

    <!-- Optical breadboard table (z=0) -->
    <geom name="breadboard" type="box" pos="0.06 0 -0.015" size="0.85 0.55 0.015" material="mat_breadboard"/>
    <!-- Textured charcoal studio cloth backdrop -->
    <geom name="backdrop" type="box" pos="0.06 0.25 0.38" size="1.40 0.01 0.50" material="mat_backdrop" contype="0" conaffinity="0"/>

    <!-- Vertical 40x80 brushed silver aluminum extrusion column (real grooved T-slot geometry) -->
    <body name="alu_column" pos="-0.184 0 0">
      <!-- Real 3-rib 40x80 extrusion profile with recessed T-slot groove on -Y face (casts authentic self-shadow) -->
      <geom type="box" pos="0 0.004 0.124" size="0.020 0.032 0.124" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.0125 -0.032 0.124" size="0.0075 0.0045 0.124" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos=" 0.0125 -0.032 0.124" size="0.0075 0.0045 0.124" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 -0.0295 0.124" size="0.0052 0.0010 0.124" material="mat_alu_slot" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.0202 0 0.124" size="0.0008 0.005 0.124" material="mat_alu_slot" contype="0" conaffinity="0"/>
      <!-- Sleek cast-aluminum L-corner gussets + stainless M6 socket-head bolts (with dark hex sockets) -->
      <geom type="box" pos="-0.029 -0.014 0.0040" size="0.013 0.013 0.0040" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.023 -0.014 0.015"  size="0.0035 0.013 0.013" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.033 -0.014 0.0095" size="0.0052 0.0026" material="mat_bolt" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.033 -0.014 0.0122" size="0.0024 0.0004" material="mat_black_anod" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.000 -0.046 0.0040" size="0.015 0.013 0.0040" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.000 -0.040 0.016"  size="0.015 0.0035 0.014" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.000 -0.049 0.0095" size="0.0052 0.0026" material="mat_bolt" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.000 -0.049 0.0122" size="0.0024 0.0004" material="mat_black_anod" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.000 -0.044 0.021" euler="1.5708 0 0" size="0.0044 0.0026" material="mat_bolt" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.000 -0.0468 0.021" euler="1.5708 0 0" size="0.0020 0.0004" material="mat_black_anod" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.000  0.046 0.0040" size="0.015 0.013 0.0040" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.000  0.040 0.016"  size="0.015 0.0035 0.014" material="mat_alu" contype="0" conaffinity="0"/>
      <!-- Clear acrylic mounting plate + machined stainless bolt + compact open 4-pillar stand-off frame -->
      <geom type="box" pos="0.024 0 0.130" size="0.0032 0.046 0.064" material="mat_acrylic" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.024 -0.020 0.175" euler="0 1.5708 0" size="0.0046 0.008" material="mat_bolt" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.047  0.042 0.168" size="0.020 0.0034 0.0034" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.047 -0.042 0.168" size="0.020 0.0034 0.0034" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.047  0.042 0.098" size="0.020 0.0034 0.0034" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.047 -0.042 0.098" size="0.020 0.0034 0.0034" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.029 -0.042 0.133" size="0.0034 0.0034 0.036" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.029  0.042 0.133" size="0.0034 0.0034 0.036" material="mat_alu" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.025  0.042 0.168" euler="0 1.5708 0" size="0.0038 0.0055" material="mat_bolt" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.025 -0.042 0.168" euler="0 1.5708 0" size="0.0038 0.0055" material="mat_bolt" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.025  0.042 0.098" euler="0 1.5708 0" size="0.0038 0.0055" material="mat_bolt" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.025 -0.042 0.098" euler="0 1.5708 0" size="0.0038 0.0055" material="mat_bolt" contype="0" conaffinity="0"/>
    </body>

    <!-- Rear clamped plate (p1 = back2_Link) -->
    <body name="seg_p1" mocap="true" pos="-0.1175 0 0.134">
      <geom type="mesh" mesh="plate_back" material="mat_plate_back" contype="0" conaffinity="0"/>
      <!-- 2 internal 25 kg tendon servos: realistic black top/bottom casings + metallic anodized purple mid-band -->
      <geom type="box" pos="0.020  0.0285  0.000" size="0.0195 0.0110 0.0090" material="mat_servo_purple" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.020  0.0285  0.015" size="0.0195 0.0111 0.0062" material="mat_servo_cap"    contype="0" conaffinity="0"/>
      <geom type="box" pos="0.020  0.0285 -0.013" size="0.0195 0.0111 0.0046" material="mat_servo_cap"    contype="0" conaffinity="0"/>
      <geom type="box" pos="0.020 -0.0285  0.000" size="0.0195 0.0110 0.0090" material="mat_servo_purple" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.020 -0.0285  0.015" size="0.0195 0.0111 0.0062" material="mat_servo_cap"    contype="0" conaffinity="0"/>
      <geom type="box" pos="0.020 -0.0285 -0.013" size="0.0195 0.0111 0.0046" material="mat_servo_cap"    contype="0" conaffinity="0"/>
      {''.join(w_geoms_p1)}
    </body>

    <!-- Free cantilevered front plate (p0 = front2_Link, transparent acrylic disc front face at x = -0.0130 m) -->
    <body name="seg_p0" mocap="true" pos="0 0 0.133">
      <geom type="mesh" mesh="plate_front" material="mat_plate_front" contype="0" conaffinity="0"/>
      <!-- Flush matte-white lower crescent reflector backing + subtle translucent matte reflection tape across wings (z=-13 mm) -->
      <geom type="cylinder" pos="-0.0128 0 -0.036" euler="0 1.5708 0" size="0.0235 0.00018" material="mat_pad_white" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.0126  0.034 -0.013" size="0.00018 0.021 0.0110" material="mat_tape_flush" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.0126 -0.034 -0.013" size="0.00018 0.021 0.0110" material="mat_tape_flush" contype="0" conaffinity="0"/>
      <!-- Front 25 kg metallic purple yaw servo seated flush on front plate center -->
      <geom type="box" pos="0.004 0 -0.004" size="0.0165 0.0102 0.0092" material="mat_servo_purple" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.004 0  0.010" size="0.0165 0.0104 0.0056" material="mat_servo_cap"    contype="0" conaffinity="0"/>
      <geom type="box" pos="0.004 0 -0.016" size="0.0165 0.0104 0.0042" material="mat_servo_cap"    contype="0" conaffinity="0"/>
      <geom type="box" pos="0.0212 0 -0.003" size="0.0012 0.0108 0.0155" material="mat_steel_shaft" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.010 0 0.019" size="0.0085 0.0035" material="mat_black_anod"      contype="0" conaffinity="0"/>
    </body>
    {''.join(jacks_xml)}
  </worldbody>
</mujoco>
"""


def _add_single_seg_strips_tendons_lasers(
    scene: mujoco.MjvScene,
    x_row: np.ndarray,
    strip_nodes: np.ndarray,
    transforms: np.ndarray,
    anchors: np.ndarray,
    guides: np.ndarray,
    hit_x_pair: np.ndarray,
    laser_y: float,
    laser_z: float,
    servo_rad_pair: np.ndarray | None = None,
    steel_matid: int = -1,
) -> None:
    """Add 8 gleaming FP64 spring-steel strips, 2 animated rotating servo horns, 4 live tendons, and 2 laser beams."""
    R_p = [_quat_xyzw_to_mat(x_row[p, 3:7]) for p in range(2)]
    P_p = x_row[:2, :3].astype(np.float64)

    # Smooth gleaming cold-rolled blue-steel spring strips (zero per-segment stripes)
    steel_rgba = np.array([0.52, 0.64, 0.74, 1.0], dtype=np.float32)
    tendon_rgba = np.array([0.16, 0.18, 0.22, 0.96], dtype=np.float32)
    beam_rgba = np.array([1.0, 0.10, 0.08, 0.52], dtype=np.float32)
    spot_rgba = np.array([1.0, 0.16, 0.10, 1.0], dtype=np.float32)
    horn_hub_rgba = np.array([0.16, 0.18, 0.22, 1.0], dtype=np.float32)
    horn_arm_rgba = np.array([0.88, 0.68, 0.22, 1.0], dtype=np.float32)   # anodized gold double-arm crank horn
    horn_pin_rgba = np.array([0.88, 0.20, 0.16, 1.0], dtype=np.float32)   # high-contrast eccentric tie-pin
    horn_screw_rgba = np.array([0.82, 0.85, 0.89, 1.0], dtype=np.float32)

    seg_center = 0.5 * (P_p[0] + P_p[1])
    for sid in range(8):
        nodes = strip_nodes[sid]
        pts = x_row[nodes, :3].astype(np.float64).copy()
        pts[0] = P_p[0] + R_p[0] @ transforms[sid, 0, 0, :3]
        pts[-1] = P_p[1] + R_p[1] @ transforms[sid, -1, 1, :3]
        for e in range(len(nodes) - 1):
            if scene.ngeom >= scene.maxgeom:
                return
            a, b = pts[e], pts[e + 1]
            diff = b - a
            length = float(np.linalg.norm(diff))
            if length < 1e-6:
                continue
            mid = 0.5 * (a + b)
            ex = diff / length
            rad = mid - seg_center
            ez = rad - np.dot(rad, ex) * ex
            ez /= max(float(np.linalg.norm(ez)), 1e-9)
            ey = np.cross(ez, ex)
            ey /= max(float(np.linalg.norm(ey)), 1e-9)
            ez = np.cross(ex, ey)
            rot = np.column_stack((ex, ey, ez)).reshape(9)
            # True 20 mm wide seamless cold-rolled spring steel with bright specular metallic highlights
            size = np.array([0.58 * length, 0.0100, 0.00045], dtype=np.float64)
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_BOX), size, mid, rot, steel_rgba)
            g.matid = -1
            g.specular = 1.0
            g.shininess = 0.94
            g.reflectance = 0.26
            scene.ngeom += 1

    # Animated rotating servo horns (left=0 at +Y, right=1 at -Y) on rear outer face of clamped plate p1
    sv = np.zeros(2) if servo_rad_pair is None else np.asarray(servo_rad_pair, dtype=np.float64)
    horn_r = 0.0225
    horn_x_local = -0.021  # cleanly inside the compact 40 mm open 4-pillar stand-off frame behind rear plate
    tie_world = [None] * 4
    # Rotation matrix mapping local cylinder Z onto plate X axis
    R_z_to_x = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    for side in range(2):
        if scene.ngeom + 6 >= scene.maxgeom:
            break
        y_piv = +0.0285 if side == 0 else -0.0285
        piv_loc = np.array([horn_x_local, y_piv, 0.0], dtype=np.float64)
        piv_w = P_p[1] + R_p[1] @ piv_loc

        # Horn rotation around local +X: at sv=0, arm aligns vertically along Z (+23.5 mm / -23.5 mm)
        ang = (+1.0 if side == 0 else -1.0) * float(sv[side])
        ca, sa = math.cos(ang), math.sin(ang)
        R_spin = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=np.float64)
        R_horn = R_p[1] @ R_spin
        R_cyl = R_horn @ R_z_to_x

        # 1) Shaft spline collar connecting back plate to horn hub
        shaft_w = P_p[1] + R_p[1] @ np.array([horn_x_local + 0.011, y_piv, 0.0])
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                            np.array([0.0055, 0.011, 0.0]), shaft_w, R_cyl.reshape(9), horn_hub_rgba)
        scene.ngeom += 1

        # 2) Circular 25T horn hub disc
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                            np.array([0.0105, 0.0032, 0.0]), piv_w, R_cyl.reshape(9), horn_hub_rgba)
        scene.ngeom += 1

        # 3) Center silver screw head
        screw_w = P_p[1] + R_p[1] @ np.array([horn_x_local - 0.0036, y_piv, 0.0])
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                            np.array([0.0042, 0.0012, 0.0]), screw_w, R_cyl.reshape(9), horn_screw_rgba)
        scene.ngeom += 1

        # 4) Rotating double-sided anodized gold crank arm (47 mm span along rotating local Z)
        g_arm = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g_arm, int(mujoco.mjtGeom.mjGEOM_BOX),
                            np.array([0.0022, 0.0048, horn_r + 0.0035]), piv_w, R_horn.reshape(9), horn_arm_rgba)
        g_arm.specular = 0.90
        g_arm.shininess = 0.85
        scene.ngeom += 1

        # 5) Upper (+Z) and lower (-Z) eccentric tie-pins on rotating horn
        for u_idx, z_sgn in enumerate((+1.0, -1.0)):
            pin_loc = piv_loc + R_spin @ np.array([-0.0025, 0.0, z_sgn * horn_r])
            pin_w = P_p[1] + R_p[1] @ pin_loc
            tie_world[2 * side + u_idx] = pin_w
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE),
                                np.array([0.0036, 0.0036, 0.0036]), pin_w, np.eye(3).reshape(9),
                                horn_pin_rgba if u_idx == 0 else horn_screw_rgba)
            scene.ngeom += 1

    # 4 tendon cords: front anchor -> back guide hole -> rotating horn tie-pin
    for k in range(4):
        if scene.ngeom + 2 >= scene.maxgeom:
            return
        pa = P_p[0] + R_p[0] @ anchors[k]
        pb = P_p[1] + R_p[1] @ guides[k]
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                            np.zeros(3), np.zeros(3), np.eye(3).reshape(9), tendon_rgba)
        mujoco.mjv_connector(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE), 0.00095, pa, pb)
        scene.ngeom += 1
        if tie_world[k] is not None:
            # Guide exit behind back plate -> moving eccentric horn tie-pin
            pb_rear = P_p[1] + R_p[1] @ (guides[k] + np.array([-0.006, 0.0, 0.0]))
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                np.zeros(3), np.zeros(3), np.eye(3).reshape(9), tendon_rgba)
            mujoco.mjv_connector(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE), 0.0010, pb_rear, tie_world[k])
            scene.ngeom += 1

    # 2 fine emissive red laser beams from sensor emitter lenses (x=0.228) directly onto flush reflection tape
    disc_face_x_offset = -0.0121
    for j, ly in enumerate((+laser_y, -laser_y)):
        if scene.ngeom + 2 >= scene.maxgeom:
            return
        src = np.array([0.228, ly, laser_z], dtype=np.float64)
        hit_x = float(hit_x_pair[j]) + disc_face_x_offset / max(float(R_p[0][0, 0]), 0.5)
        hit = np.array([hit_x, ly, laser_z], dtype=np.float64)
        g_beam = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g_beam, int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                            np.zeros(3), np.zeros(3), np.eye(3).reshape(9), beam_rgba)
        mujoco.mjv_connector(g_beam, int(mujoco.mjtGeom.mjGEOM_CAPSULE), 0.00045, src, hit)
        g_beam.emission = 0.55
        scene.ngeom += 1
        g_spot = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g_spot, int(mujoco.mjtGeom.mjGEOM_SPHERE),
                            np.array([0.0022, 0.0022, 0.0022]), hit, np.eye(3).reshape(9), spot_rgba)
        g_spot.emission = 0.95
        scene.ngeom += 1


def _draw_corner_badge(img: np.ndarray, title: str, accent: tuple[int, int, int]) -> np.ndarray:
    """Clean borderless top-left title (large bold text, zero box/border/dot)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, 'RGBA')
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 38)
    except Exception:
        font = ImageFont.load_default()
    x0, y0 = 32, 22
    draw.text((x0 + 2, y0 + 2), title, fill=(0, 0, 0, 175), font=font)
    draw.text((x0, y0), title, fill=(248, 250, 252, 255), font=font)
    return np.asarray(im)


def render_bench_mujoco_video(
    sim: dict,
    out_mp4: Path,
    width: int = 1280,
    height: int = 720,
    show_badge: bool = True,
    preview_png: Path | None = None,
) -> Path:
    """Render 1:1 Exp 126–128 optical-table dual-laser video sequentially (Bilateral -> Left -> Right)."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory(prefix='serpworm_bench_tex_') as tmp_tex_dir:
        tex_paths = _make_studio_textures(Path(tmp_tex_dir))
        xml = _build_bench_mjcf(sim['mounts'], tex_paths, sim['laser_z'], sim['laser_y'])
        model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    id_p0 = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'seg_p0')]
    id_p1 = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'seg_p1')]
    steel_matid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, 'mat_spring_steel')

    # Natural 35 deg studio lens (zero wide-angle keystoning, ramrod-straight aluminum column, level horizon)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 91.5
    cam.elevation = -9.5
    cam.distance = 0.64
    cam.lookat[:] = [0.056, 0.0, 0.110]

    fps = int(sim['fps'])
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    modes = [
        (0, '双侧伸缩', (96, 165, 250)),
        (1, '左侧伸缩', (52, 211, 153)),
        (2, '右侧伸缩', (251, 191, 36)),
    ]
    T = sim['x'].shape[0]
    peak_idx = int(np.argmax(sim['ch1_mm'][:, 2]))

    for m_idx, title, accent in modes:
        for fi in range(T):
            xr = sim['x'][fi, m_idx]
            mj_data.mocap_pos[id_p0] = xr[0, :3]
            mj_data.mocap_quat[id_p0] = _quat_xyzw_to_wxyz(xr[0, 3:7])
            mj_data.mocap_pos[id_p1] = xr[1, :3]
            mj_data.mocap_quat[id_p1] = _quat_xyzw_to_wxyz(xr[1, 3:7])
            mujoco.mj_forward(model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            _add_single_seg_strips_tendons_lasers(
                renderer.scene, xr,
                sim['strip_nodes'], sim['transforms'], sim['anchors'], sim['guides'],
                sim['hit_x'][fi, m_idx], sim['laser_y'], sim['laser_z'],
                servo_rad_pair=np.deg2rad(sim['servo_deg'][fi, m_idx]),
                steel_matid=steel_matid,
            )
            rgb = renderer.render()
            if show_badge:
                rgb = _draw_corner_badge(rgb, title, accent)
            if preview_png is not None and m_idx == 2 and fi == peak_idx:
                from PIL import Image
                Image.fromarray(rgb).save(preview_png)
            proc.stdin.write(rgb.tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode('utf-8', errors='replace')
    rc = proc.wait()
    renderer.close()
    if rc != 0:
        raise RuntimeError(f'ffmpeg failed ({rc}): {err[-400:]}')
    return out_mp4


def plot_bench_validation(sim: dict, out_png: Path) -> Path:
    """Plot 1:1 validation curves comparing FP64 simulation vs measured Exp 126–128 bench data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), dpi=160, facecolor='white')
    specs = [
        (0, 'exp128_bilateral', '实验 128 · 双侧伸缩'),
        (1, 'exp127_left',      '实验 127 · 左侧伸缩'),
        (2, 'exp126_right',     '实验 126 · 右侧伸缩'),
    ]
    t = sim['t']
    n = len(t)
    # Extract one full cycle (2nd cycle for steady state)
    T_cyc = sim['params']['cycle_period_s']
    cyc_mask = (t >= T_cyc) & (t <= 2.0 * T_cyc + 1e-6)
    if cyc_mask.sum() < 10:
        cyc_mask = np.ones(n, dtype=bool)
    tc = t[cyc_mask] - t[cyc_mask][0]
    ang_full = np.where(tc <= 0.5 * T_cyc, 360.0 * tc / T_cyc, 360.0 * (1.0 - tc / T_cyc))
    # X-axis 0 -> 180 -> 0 mapped linearly to [0, 360] for clean symmetric display
    x_phase = 360.0 * tc / T_cyc

    for col, (idx, key, title) in enumerate(specs):
        ref = REAL_EXP_REF[key]
        c1 = sim['ch1_mm'][:, idx]
        c2 = sim['ch2_mm'][:, idx]
        ctr = 0.5 * (c1 + c2)

        # Row 0: Dual laser waveforms Ch1 (right) & Ch2 (left)
        ax0 = axes[0, col]
        ax0.plot(t, c1, color='#1d6fa5', lw=1.8, label=f'仿真 Ch1 ({c1.max():.1f} mm)')
        ax0.plot(t, c2, color='#1b8a6b', lw=1.8, label=f'仿真 Ch2 ({c2.max():.1f} mm)')
        ax0.axhline(ref['ch1_peak_mm'], color='#1d6fa5', ls='--', lw=1.0, alpha=0.65, label=f"实测 Ch1 ({ref['ch1_peak_mm']} mm)")
        ax0.axhline(ref['ch2_peak_mm'], color='#1b8a6b', ls=':', lw=1.2, alpha=0.65, label=f"实测 Ch2 ({ref['ch2_peak_mm']} mm)")
        ax0.set_title(title, fontsize=11.5, fontweight='bold')
        ax0.set_xlabel('时间 / s', fontsize=9)
        if col == 0:
            ax0.set_ylabel('双激光位移 / mm', fontsize=9.5)
        ax0.set_ylim(-1.5, 42.0)
        ax0.grid(alpha=0.25)
        ax0.legend(fontsize=7.5, loc='upper right', framealpha=0.9)

        # Row 1: Center axial displacement vs servo angle (0° -> 180° -> 0°) + measured points
        ax1 = axes[1, col]
        ax1.plot(x_phase, ctr[cyc_mask], color='#1d6fa5', lw=2.2, label=f'仿真中心 ({ctr[cyc_mask].max():.1f} mm)')
        ref_ang = np.asarray(ref['angles_deg'])
        ref_y = np.asarray(ref['center_mm'])
        ref_x_full = np.concatenate([ref_ang, 360.0 - ref_ang[-2::-1]])
        ref_y_full = np.concatenate([ref_y, ref_y[-2::-1]])
        ax1.scatter(ref_x_full, ref_y_full, color='#e11d48', s=26, zorder=5, label=f"实测均值 ({ref['center_peak_mm']} mm)")
        ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax1.set_xticklabels(['0', '45', '90', '135', '180', '135', '90', '45', '0'], fontsize=8.5)
        ax1.set_xlabel('舵机角度 / °', fontsize=9)
        if col == 0:
            ax1.set_ylabel('中心位移 / mm', fontsize=9.5)
        ax1.set_ylim(-1.5, 41.0)
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='1:1 Exp 126–128 dual-laser single-segment bench digital twin')
    parser.add_argument('--out-dir', type=Path, default=Path('runs/bench_exp126_128'))
    parser.add_argument('--cycles', type=float, default=2.0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--no-badge', action='store_true')
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sim = simulate_dual_laser_bench(cycles=args.cycles, fps=args.fps)
    (args.out_dir / 'bench_summary.json').write_text(json.dumps(sim['summary'], indent=2), encoding='utf-8')
    print(json.dumps(sim['summary'], indent=2))
    plot_bench_validation(sim, args.out_dir / 'bench_126_128_validation.png')
    render_bench_mujoco_video(
        sim, args.out_dir / 'bench_126_128_mujoco_3d.mp4',
        show_badge=not args.no_badge,
        preview_png=args.out_dir / 'bench_126_128_preview.png',
    )
    print(f'Saved 1:1 bench outputs to {args.out_dir}')


if __name__ == '__main__':
    main()

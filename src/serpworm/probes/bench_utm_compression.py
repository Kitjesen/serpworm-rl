"""1:1 Digital Twin & Quantitative Validation of the UTM 5-Cycle Compression Experiment.

Reproduces the vertical single-segment universal testing machine (UTM) compression test:
- Compares real 5-cycle loading/unloading UTM measurements (0 -> 45 -> 0 mm, 42.7 N peak)
  against BOTH:
    1) Current default RL preset (`ds3218_default`: t = 0.15 mm, bow = 50.7 mm)
    2) UTM-calibrated FP64 preset (`bench_exp128_calibrated`: t = 0.22 mm, bow = 30.0 mm)
- Solves exact FP64 corotational beam static equilibrium F(x), tangent stiffness dF/dx,
  and radial barrel bulge ratio across 0..45 mm.
- Renders a 1:1 photorealistic MuJoCo vertical UTM scene calibrated to match the real lab
  video perspective (telephoto fovy=25 deg), platen thickness/width, recessed left accordion
  column, low-profile acrylic base mount, 8 satin-black spring-steel strips, purple/black 26KG
  servos, right-side dual-wheel arm, and aluminum push bridge.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import torch
import mujoco
from scipy.signal import find_peaks

from serpworm.config import HardwareConfig
from serpworm.paths import ASSETS_DIR
from serpworm.physics.gpu_core import GPUCore, mul, norm, unit

__all__ = [
    'load_raw_utm_cycles',
    'track_video_platen_displacement',
    'simulate_utm_compression',
    'plot_utm_comparison_curves',
    'render_utm_videos',
]

MESH_DIR = ASSETS_DIR / 'cad' / 'meshes' / 'longworm2'

# Proper rotation R_utm (det = +1) mapping GPUCore horizontal segment coords to vertical UTM rig:
#   +x_sim (top p1 -> bottom p0) -> -Z_w (DOWN)
#   +y_sim                       -> -Y_w (into scene)
#   +z_sim (opposite of wheels)  -> -X_w (LEFT) => wheels (-z_sim) extend to RIGHT (+X_w)
R_UTM = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


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


def load_raw_utm_cycles(raw_csv: Path) -> dict:
    """Parse raw 5-cycle GBK UTM CSV into monotonic 0..45 mm loading, unloading, and mean grids."""
    rows: list[tuple[float, float, float]] = []
    with open(raw_csv, 'r', encoding='gbk', errors='ignore') as f:
        for r in csv.reader(f):
            if len(r) >= 4:
                try:
                    rows.append((float(r[1]), float(r[2]), float(r[3])))
                except ValueError:
                    pass
    arr = np.asarray(rows, dtype=np.float64)
    t, d, F = arr[:, 0], arr[:, 1], arr[:, 2]

    kernel = np.ones(501, dtype=np.float64) / 501.0
    ds = np.convolve(d, kernel, mode='same')
    mins, _ = find_peaks(-ds, prominence=5.0, distance=10000)
    maxs, _ = find_peaks(ds, prominence=5.0, distance=10000)
    starts = np.concatenate(([0], maxs))
    ends = np.concatenate((maxs, [len(d) - 1]))

    grid_mm = np.linspace(0.0, 45.0, 451)
    loads, unloads, mids = [], [], []
    energy_losses = []
    for ci in range(len(mins)):
        s, m, e = int(starts[ci]), int(mins[ci]), int(ends[ci])
        d0 = float(d[s:s + 10].mean())
        f0 = float(F[s:s + 10].mean())
        xl = -(d[s:m + 1] - d0)
        fl = -(F[s:m + 1] - f0)
        xu = -(d[m:e + 1] - d0)
        fu = -(F[m:e + 1] - f0)
        il = np.argsort(xl)
        iu = np.argsort(xu)
        fl_g = np.interp(grid_mm, xl[il], fl[il])
        fu_g = np.interp(grid_mm, xu[iu], fu[iu])
        fl_g -= fl_g[0]
        fu_g -= fu_g[0]
        loads.append(fl_g)
        unloads.append(fu_g)
        mids.append(0.5 * (fl_g + fu_g))
        e_in = float(np.trapezoid(fl_g, grid_mm * 1e-3))
        e_out = float(np.trapezoid(fu_g, grid_mm * 1e-3))
        energy_losses.append((e_in - e_out) / max(e_in, 1e-9))

    loads_a = np.asarray(loads)
    unloads_a = np.asarray(unloads)
    mids_a = np.asarray(mids)
    mean_mid = mids_a.mean(axis=0)
    sd_mid = mids_a.std(axis=0, ddof=1)
    mean_load = loads_a.mean(axis=0)
    mean_unload = unloads_a.mean(axis=0)
    hyst_N = mean_load - mean_unload

    return {
        'grid_mm': grid_mm,
        'loads_N': loads_a,
        'unloads_N': unloads_a,
        'mids_N': mids_a,
        'mean_load_N': mean_load,
        'mean_unload_N': mean_unload,
        'mean_mid_N': mean_mid,
        'sd_mid_N': sd_mid,
        'hysteresis_N': hyst_N,
        'mean_hyst_N': float(np.mean(np.abs(hyst_N))),
        'max_hyst_N': float(np.max(np.abs(hyst_N))),
        'energy_loss_pct': float(100.0 * np.mean(energy_losses)),
        'steady_energy_loss_pct': float(100.0 * np.mean(energy_losses[1:])),
        'crosshead_speed_mm_s': float(45.0 / (t[mins[0]] - t[starts[0]])),
        'cycle_duration_s': float(t[ends[0]] - t[starts[0]]),
    }


def track_video_platen_displacement(video_path: Path, max_stroke_mm: float = 45.0) -> dict:
    """Track sub-pixel vertical displacement of the silver UTM platen across all video frames."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames_rgb: list[np.ndarray] = []
    y_edges: list[float] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        prof = gray[50:320, 300:980].mean(axis=1)
        grad = np.diff(prof)
        k = int(np.argmin(grad))
        if 1 <= k < len(grad) - 1:
            g0, g1, g2 = grad[k - 1], grad[k], grad[k + 1]
            denom = (g0 - 2.0 * g1 + g2)
            delta = 0.5 * (g0 - g2) / denom if abs(denom) > 1e-9 else 0.0
            delta = max(-0.5, min(0.5, delta))
        else:
            delta = 0.0
        y_edges.append(50.0 + k + delta)
    cap.release()

    y_arr = np.asarray(y_edges, dtype=np.float64)
    y_smooth = y_arr.copy()
    if len(y_arr) >= 5:
        for i in range(1, len(y_arr) - 1):
            y_smooth[i] = 0.25 * y_arr[i - 1] + 0.5 * y_arr[i] + 0.25 * y_arr[i + 1]

    y_min = float(np.percentile(y_smooth[:3], 50))
    y_end_min = float(np.min(y_smooth[-10:]))
    y_base = min(y_min, y_end_min)
    y_max = float(np.max(y_smooth))
    disp_mm = np.clip((y_smooth - y_base) / max(y_max - y_base, 1.0) * max_stroke_mm, 0.0, max_stroke_mm)
    peak_frame = int(np.argmax(disp_mm))

    return {
        'fps': fps,
        'n_frames': len(frames_rgb),
        'frames_rgb': frames_rgb,
        'y_px': y_smooth,
        'disp_mm': disp_mm,
        'peak_frame': peak_frame,
        'px_per_mm': (y_max - y_base) / max_stroke_mm,
    }


def _solve_static_strip_equilibrium(
    core: GPUCore,
    x_disp_m: float,
    max_iters: int = 35,
    tol_n: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Exact damped Newton static equilibrium of the 8 corotational strips under prescribed plate compression."""
    rest = core.rest[0]
    core.x[0, 0] = rest[0]
    core.x[0, 1] = rest[1]
    core.x[0, 1, 0] = rest[1, 0] + float(x_disp_m)
    core.v.zero_()

    idx = core.interior_ids
    eye6 = torch.eye(6, device=core.device, dtype=core.dtype)

    for _ in range(max_iters):
        elastic, K, _ = core.beam()
        res_int = elastic[:, idx]
        if float(res_int.abs().max().item()) < tol_n:
            break

        diag = K[:, :, :-1, 6:, 6:] + K[:, :, 1:, :6, :6] + 1e-4 * eye6
        lower = K[:, :, 1:-1, 6:, :6]
        upper = K[:, :, 1:-1, :6, 6:]
        r = elastic[:, core.nodes[:, 1:-1], :, None]

        ds, rs = [], []
        for i in range(core.E - 1):
            d_blk = diag[:, :, i]
            b_blk = r[:, :, i]
            if i:
                fac = lower[:, :, i - 1] @ ds[-1]
                d_blk = d_blk - fac @ upper[:, :, i - 1]
                b_blk = b_blk - fac @ rs[-1]
            inv = torch.linalg.inv(d_blk)
            ds.append(inv)
            rs.append(b_blk)
        sol = [None] * (core.E - 1)
        sol[-1] = ds[-1] @ rs[-1]
        for i in range(core.E - 3, -1, -1):
            sol[i] = ds[i] @ (rs[i] - upper[:, :, i] @ sol[i + 1])

        du = torch.stack(sol, 2)[..., 0].reshape(1, -1, 6)
        max_dp = float(du[..., :3].abs().max().item())
        max_dr = float(du[..., 3:].abs().max().item())
        alpha = min(0.85, 0.004 / max(max_dp, 1e-12), 0.12 / max(max_dr, 1e-12))
        step = alpha * du

        core.x[:, idx, :3] = core.x[:, idx, :3] + step[..., :3]
        omega = 0.5 * step[..., 3:]
        theta = norm(omega)
        exp = torch.cat((omega * torch.sinc(theta / math.pi), torch.cos(theta)), -1)
        core.x[:, idx, 3:] = unit(mul(exp, core.x[:, idx, 3:]))

    elastic, _, _ = core.beam()
    f_axial = float(-elastic[0, 1, 0].item())
    return f_axial, core.x[0].detach().cpu().numpy().copy()


def simulate_utm_compression(
    hardware: HardwareConfig,
    disp_grid_mm: np.ndarray,
    video_disp_mm: np.ndarray | None = None,
    elements_per_strip: int = 16,
    device: str = 'cpu',
) -> dict:
    """Compute exact quasi-static FP64 compression curve F(x) and optional video-synced 3D poses."""
    core = GPUCore(
        count=1,
        batch=1,
        dt=0.001,
        device=device,
        plate_solver='segment_lu',
        ground_contact=False,
        hardware=hardware,
        elements_per_strip=elements_per_strip,
    )
    forces_N = np.empty(len(disp_grid_mm), dtype=np.float64)
    bulge_diam_mm = np.empty(len(disp_grid_mm), dtype=np.float64)
    grid_states = np.empty((len(disp_grid_mm), core.D, 7), dtype=np.float64)

    for i, d_mm in enumerate(disp_grid_mm):
        f_ax, x_state = _solve_static_strip_equilibrium(core, d_mm * 1e-3, max_iters=40)
        forces_N[i] = f_ax
        grid_states[i] = x_state
        pts_int = x_state[2:, :3]
        cy = 0.5 * (x_state[0, 1] + x_state[1, 1])
        cz = 0.5 * (x_state[0, 2] + x_state[1, 2])
        rad = np.hypot(pts_int[:, 1] - cy, pts_int[:, 2] - cz)
        bulge_diam_mm[i] = 2.0 * float(rad.max()) * 1000.0

    video_states = None
    video_forces_N = None
    if video_disp_mm is not None:
        core_v = GPUCore(
            count=1,
            batch=1,
            dt=0.001,
            device=device,
            plate_solver='segment_lu',
            ground_contact=False,
            hardware=hardware,
            elements_per_strip=elements_per_strip,
        )
        video_states = np.empty((len(video_disp_mm), core_v.D, 7), dtype=np.float32)
        video_forces_N = np.empty(len(video_disp_mm), dtype=np.float64)
        for fi, d_mm in enumerate(video_disp_mm):
            f_ax, x_state = _solve_static_strip_equilibrium(core_v, float(d_mm) * 1e-3, max_iters=30)
            video_forces_N[fi] = f_ax
            video_states[fi] = x_state.astype(np.float32)

    return {
        'name': hardware.name,
        'strip_thickness_mm': float(core.parameters['strip_thickness_m']) * 1000.0,
        'stress_free_bow_mm': float(core.parameters['stress_free_bow_m']) * 1000.0,
        'disp_mm': np.asarray(disp_grid_mm, dtype=np.float64),
        'force_N': forces_N,
        'bulge_diam_mm': bulge_diam_mm,
        'grid_states': grid_states,
        'video_states': video_states,
        'video_forces_N': video_forces_N,
        'strip_nodes': core.nodes.cpu().numpy(),
        'transforms': core.transforms.cpu().numpy(),
        'mounts': core.mounts.cpu().numpy(),
        'anchors': core.anchors.cpu().numpy(),
        'guides': core.guides.cpu().numpy(),
        'rest': core.rest[0].cpu().numpy(),
    }


def _sim_to_utm_world(pos_sim: np.ndarray, p0_rest: np.ndarray, z_bot: float) -> np.ndarray:
    """Transform point(s) from GPUCore horizontal segment coords to vertical UTM world coords."""
    rel = pos_sim - p0_rest
    out = rel @ R_UTM.T
    out[..., 2] += z_bot
    return out


def _make_breadboard_texture(out_png: Path) -> Path:
    """Warm brushed stainless optical breadboard texture with 25 mm M6 tapped hole grid."""
    from PIL import Image, ImageDraw
    N = 512
    rng = np.random.default_rng(7)
    # Subtle warm brushed metallic stainless steel matching real lab optical table
    base = np.zeros((N, N, 3), dtype=np.float32)
    base[:, :, 0] = 152.0
    base[:, :, 1] = 150.0
    base[:, :, 2] = 146.0
    grain = rng.normal(0.0, 2.8, size=(N, 1, 1))
    fine = rng.normal(0.0, 1.4, size=(N, N, 1))
    arr = np.clip(base + grain + fine, 118, 182).astype(np.uint8)
    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)
    step = N // 8
    for r in range(8):
        for c in range(8):
            cx = c * step + step // 2
            cy = r * step + step // 2
            draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(96, 95, 92))
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(34, 35, 38))
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(16, 17, 20))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    return out_png


def _build_utm_mjcf(tex_path: Path, z_bot: float) -> str:
    """Build 1:1 vertical UTM compression test rig MJCF calibrated to real video geometry & FOV."""
    mesh_dir = MESH_DIR.resolve().as_posix()
    tex_posix = tex_path.resolve().as_posix()

    # Left UTM vertical column enclosure strictly along far-left frame border (clean gap from platen)
    slats = []
    for k in range(46):
        zk = -0.065 + k * 0.0084
        slats.append(
            f'<geom type="box" pos="-0.228 0.020 {zk:.4f}" euler="0 0.28 0" '
            f'size="0.015 0.022 0.0017" material="mat_bellows" contype="0" conaffinity="0"/>'
        )
    # 4 corner brass/steel hex bolts + washers clamping the clear acrylic base flange onto breadboard
    flange_bolts = []
    for bx in (-0.056, 0.056):
        for by in (-0.048, 0.048):
            flange_bolts.append(
                f'<geom type="cylinder" pos="{bx:.4f} {by:.4f} 0.0110" size="0.0066 0.0038" '
                f'material="mat_brass" contype="0" conaffinity="0"/>'
            )
            flange_bolts.append(
                f'<geom type="cylinder" pos="{bx:.4f} {by:.4f} 0.0065" size="0.0082 0.0010" '
                f'material="mat_stainless" contype="0" conaffinity="0"/>'
            )

    return f"""<mujoco model="serpworm_utm_compression_rig">
  <compiler angle="radian" meshdir="{mesh_dir}" autolimits="true"/>
  <visual>
    <!-- Telephoto fovy=25.5 deg matches real phone lens at ~0.6 m distance (eliminates wide-angle perspective tilt) -->
    <global offwidth="1280" offheight="720" fovy="25.5"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.22 0.22 0.24" diffuse="0.36 0.36 0.38" specular="0.20 0.20 0.20"/>
  </visual>
  <asset>
    <mesh name="plate_front" file="front2_Link.STL"/>
    <mesh name="plate_back"  file="back2_Link.STL"/>
    <mesh name="wheel_mesh"  file="w2-1_Link.STL"/>
    <texture name="tex_breadboard" type="2d" file="{tex_posix}"/>
    <material name="mat_breadboard"   texture="tex_breadboard" texrepeat="8 8" texuniform="true" reflectance="0.10" shininess="0.34" specular="0.22"/>
    <material name="mat_backdrop"     rgba="0.105 0.110 0.122 1.0" specular="0.03" shininess="0.04"/>
    <material name="mat_alu_platen"   rgba="0.58 0.57 0.54 1.0" specular="0.40" shininess="0.36" reflectance="0.06"/>
    <material name="mat_alu_edge"     rgba="0.44 0.44 0.43 1.0" specular="0.32" shininess="0.30"/>
    <material name="mat_stainless"    rgba="0.60 0.60 0.59 1.0" specular="0.72" shininess="0.68" reflectance="0.16"/>
    <material name="mat_col_frame"    rgba="0.56 0.56 0.55 1.0" specular="0.48" shininess="0.42"/>
    <material name="mat_bellows_bg"   rgba="0.14 0.14 0.15 1.0" specular="0.10" shininess="0.10"/>
    <material name="mat_bellows"      rgba="0.46 0.46 0.45 1.0" specular="0.32" shininess="0.28"/>
    <material name="mat_acrylic"      rgba="0.82 0.86 0.90 0.28" specular="0.75" shininess="0.72"/>
    <material name="mat_plate_dark"   rgba="0.14 0.15 0.17 1.0" specular="0.35" shininess="0.35"/>
    <material name="mat_servo_purple" rgba="0.46 0.24 0.68 1.0" specular="0.52" shininess="0.48"/>
    <material name="mat_servo_black"  rgba="0.11 0.12 0.14 1.0" specular="0.30" shininess="0.28"/>
    <material name="mat_horn_dark"    rgba="0.19 0.20 0.23 1.0" specular="0.48" shininess="0.46"/>
    <material name="mat_silkscreen"   rgba="0.88 0.90 0.92 1.0" specular="0.15" shininess="0.15"/>
    <material name="mat_wheel_tire"   rgba="0.15 0.16 0.18 1.0" specular="0.20" shininess="0.20"/>
    <material name="mat_wheel_ring"   rgba="0.66 0.62 0.56 1.0" specular="0.62" shininess="0.58"/>
    <material name="mat_brass"        rgba="0.72 0.64 0.42 1.0" specular="0.65" shininess="0.60"/>
    <material name="mat_bridge"       rgba="0.68 0.68 0.67 1.0" specular="0.40" shininess="0.36"/>
  </asset>
  <worldbody>
    <!-- Overhead studio key lights top of platen naturally while side fill catches sleek highlights on black steel ribbons -->
    <light name="key"  pos="-0.22 -0.62 1.18" dir="0.12 0.42 -0.89" diffuse="0.66 0.65 0.62" specular="0.48 0.48 0.48" castshadow="true"/>
    <light name="fill" pos=" 0.52 -0.82 0.22" dir="-0.48 0.78 -0.18" diffuse="0.32 0.34 0.37" specular="0.48 0.48 0.48" castshadow="false"/>
    <light name="rim"  pos="-0.45 -0.78 0.26" dir=" 0.42 0.78 -0.22" diffuse="0.22 0.24 0.27" specular="0.42 0.42 0.42" castshadow="false"/>

    <!-- Optical breadboard table (z = 0) -->
    <geom name="breadboard" type="box" pos="0.015 0 -0.015" size="0.165 0.170 0.015" material="mat_breadboard"/>
    <!-- Deep matte charcoal studio backdrop curtain -->
    <geom name="backdrop" type="box" pos="0 0.26 0.20" size="0.90 0.01 0.55" material="mat_backdrop" contype="0" conaffinity="0"/>

    <!-- Left vertical UTM leadscrew column enclosure (flush vertical channel with recessed accordion slats, clean gap from platen) -->
    <geom type="box" pos="-0.236 0.032 0.13" size="0.020 0.012 0.28" material="mat_bellows_bg" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.211 0.016 0.13" size="0.0032 0.028 0.28" material="mat_col_frame" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.217 0.004 0.13" size="0.0018 0.018 0.28" material="mat_stainless" contype="0" conaffinity="0"/>
    {''.join(slats)}

    <!-- Clear acrylic square base mounting flange bolted flush onto breadboard (z = 0..7 mm) -->
    <geom type="box" pos="0 0 0.0035" size="0.072 0.064 0.0035" material="mat_acrylic" contype="0" conaffinity="0"/>
    {''.join(flange_bolts)}

    <!-- Bottom fixed plate assembly: low-profile acrylic disc (z = z_bot + 0.0165 = 0.0355 m), zero coplanar overlap -->
    <body name="seg_bot" mocap="true" pos="0 0 {z_bot:.5f}">
      <!-- Clear acrylic lower disc flush with bottom strip clamps (local x_sim = -0.0185 .. -0.0145) -->
      <geom type="cylinder" pos="-0.0165 0 0" euler="0 1.5708 0" size="0.056 0.0020" material="mat_acrylic" contype="0" conaffinity="0"/>
      <!-- Bottom 26KG servo stacked cleanly with non-overlapping faces (eliminates Z-fighting) -->
      <geom type="box" pos="-0.0245 0 0" size="0.0055 0.0190 0.0150" material="mat_servo_black" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.0095 0 0" size="0.0046 0.0204 0.0144" material="mat_servo_black" contype="0" conaffinity="0"/>
      <geom type="box" pos=" 0.0035 0 0" size="0.0082 0.0200 0.0140" material="mat_servo_purple" contype="0" conaffinity="0"/>
      <!-- Central vertical stainless stop bolt rising UP into barrel (-x_sim = +Z_w) -->
      <geom type="cylinder" pos="-0.044 0 0" euler="0 1.5708 0" size="0.0024 0.014" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.031 0 0" euler="0 1.5708 0" size="0.0055 0.0025" material="mat_stainless" contype="0" conaffinity="0"/>
    </body>

    <!-- Top moving plate assembly (back plate: strip clamps at local x_sim=0, disc covers -0.0022..+0.0022) -->
    <body name="seg_top" mocap="true" pos="0 0 {z_bot + 0.1175:.5f}">
      <!-- Matte black top disc (thin 4.4 mm structural plate flush with strip clamps) -->
      <geom type="cylinder" pos="0 0 0" euler="0 1.5708 0" size="0.056 0.0022" material="mat_plate_dark" contype="0" conaffinity="0"/>
      <!-- Dual 26KG serial bus servos (front y=+0.014, rear y=-0.014; strictly non-overlapping for zero Z-fighting) -->
      <!-- Top black caps above top disc -->
      <geom type="box" pos="-0.0092  0.0135 0" size="0.0066 0.0102 0.0195" material="mat_servo_black" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.0092 -0.0135 0" size="0.0066 0.0102 0.0195" material="mat_servo_black" contype="0" conaffinity="0"/>
      <!-- Purple anodized mid-housings protruding down into barrel -->
      <geom type="box" pos=" 0.0155  0.0135 0" size="0.0130 0.0098 0.0185" material="mat_servo_purple" contype="0" conaffinity="0"/>
      <geom type="box" pos=" 0.0155 -0.0135 0" size="0.0130 0.0098 0.0185" material="mat_servo_purple" contype="0" conaffinity="0"/>
      <!-- Bottom black servo end-caps (strictly clear of purple by 0.3 mm) -->
      <geom type="box" pos=" 0.0316  0.0135 0" size="0.0028 0.0101 0.0188" material="mat_servo_black" contype="0" conaffinity="0"/>
      <geom type="box" pos=" 0.0316 -0.0135 0" size="0.0028 0.0101 0.0188" material="mat_servo_black" contype="0" conaffinity="0"/>
      <!-- Front purple face details: 3-pin serial bus socket (- + S) + silkscreen status ticks (0.6 mm proud of y=+0.0233) -->
      <geom type="box" pos="0.0230 0.0242 -0.0040" size="0.0036 0.0009 0.0052" material="mat_servo_black" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.0230 0.0252 -0.0065" size="0.0010 0.0004 0.0008" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.0230 0.0252 -0.0040" size="0.0010 0.0004 0.0008" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.0230 0.0252 -0.0015" size="0.0010 0.0004 0.0008" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.0085 0.0238 -0.0145" size="0.0009 0.0004 0.0022" material="mat_silkscreen" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.0125 0.0238 -0.0145" size="0.0009 0.0004 0.0022" material="mat_silkscreen" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.0165 0.0238 -0.0145" size="0.0009 0.0004 0.0022" material="mat_silkscreen" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.0205 0.0238 -0.0145" size="0.0009 0.0004 0.0022" material="mat_silkscreen" contype="0" conaffinity="0"/>
      <!-- Dual horizontal servo horns (舵盘) + spline shafts + end tie-screws sitting above top servo caps inside U-bridge -->
      <geom type="cylinder" pos="-0.0205  0.0135 0" euler="0 1.5708 0" size="0.0052 0.0042" material="mat_horn_dark" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.0205 -0.0135 0" euler="0 1.5708 0" size="0.0052 0.0042" material="mat_horn_dark" contype="0" conaffinity="0"/>
      <geom type="box"      pos="-0.0268  0.0135 0" size="0.0022 0.0048 0.0265" material="mat_horn_dark" contype="0" conaffinity="0"/>
      <geom type="box"      pos="-0.0268 -0.0135 0" size="0.0022 0.0048 0.0265" material="mat_horn_dark" contype="0" conaffinity="0"/>
      <!-- Horn center spline screw & left/right tendon tie-pins -->
      <geom type="cylinder" pos="-0.0298  0.0135  0.0000" euler="0 1.5708 0" size="0.0032 0.0010" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.0298 -0.0135  0.0000" euler="0 1.5708 0" size="0.0032 0.0010" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.0268  0.0135  0.0235" euler="0 1.5708 0" size="0.0026 0.0032" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.0268  0.0135 -0.0235" euler="0 1.5708 0" size="0.0026 0.0032" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.0268 -0.0135  0.0235" euler="0 1.5708 0" size="0.0026 0.0032" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.0268 -0.0135 -0.0235" euler="0 1.5708 0" size="0.0026 0.0032" material="mat_stainless" contype="0" conaffinity="0"/>
      <!-- Right-side dual-wheel black 3D-printed arm (-z_sim maps to +X_w RIGHT) -->
      <geom type="box" pos="0.002  0.032 -0.078" size="0.0055 0.0038 0.034" material="mat_plate_dark" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.002 -0.032 -0.078" size="0.0055 0.0038 0.034" material="mat_plate_dark" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.002  0.000 -0.052" size="0.0055 0.036 0.008" material="mat_plate_dark" contype="0" conaffinity="0"/>
      <!-- Two bold concentric-ring wheels extending horizontally to the right (R = 19 mm, width = 14 mm) -->
      <geom type="cylinder" pos="0.002  0.036 -0.106" euler="1.5708 0 0" size="0.019 0.0068" material="mat_wheel_tire" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.002  0.036 -0.106" euler="1.5708 0 0" size="0.0135 0.0072" material="mat_wheel_ring" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.002  0.036 -0.106" euler="1.5708 0 0" size="0.0085 0.0075" material="mat_servo_black" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.002  0.036 -0.106" euler="1.5708 0 0" size="0.0048 0.0078" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.002 -0.036 -0.106" euler="1.5708 0 0" size="0.019 0.0068" material="mat_wheel_tire" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.002 -0.036 -0.106" euler="1.5708 0 0" size="0.0135 0.0072" material="mat_wheel_ring" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.002 -0.036 -0.106" euler="1.5708 0 0" size="0.0085 0.0075" material="mat_servo_black" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.002 -0.036 -0.106" euler="1.5708 0 0" size="0.0048 0.0078" material="mat_stainless" contype="0" conaffinity="0"/>
      <!-- Sleek open-frame silver aluminum U-bridge adapter bridging top disc to UTM platen -->
      <geom type="box" pos="-0.022 0  0.036" size="0.021 0.018 0.0032" material="mat_bridge" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.022 0 -0.036" size="0.021 0.018 0.0032" material="mat_bridge" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.042 0  0.000" size="0.0028 0.019 0.039" material="mat_bridge" contype="0" conaffinity="0"/>
    </body>

    <!-- Upper UTM brushed-aluminum compression platen (296 x 150 x 34 mm) + load-cell flange + threaded ram -->
    <body name="utm_platen" mocap="true" pos="0 0 {z_bot + 0.1175 + 0.0445 + 0.017:.5f}">
      <geom type="box" pos="0 0 0" size="0.148 0.075 0.017" material="mat_alu_platen" contype="0" conaffinity="0"/>
      <!-- Chamfered darker side end-caps matching real milled aluminum extrusion platen -->
      <geom type="box" pos="-0.149 0 0" size="0.0018 0.073 0.016" material="mat_alu_edge" contype="0" conaffinity="0"/>
      <geom type="box" pos=" 0.149 0 0" size="0.0018 0.073 0.016" material="mat_alu_edge" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 0.026" size="0.034 0.0095" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 0.066" size="0.014 0.032" material="mat_stainless" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 0.102" size="0.018 0.008" material="mat_stainless" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _add_utm_strips(
    scene: mujoco.MjvScene,
    x_row: np.ndarray,
    p0_rest: np.ndarray,
    z_bot: float,
    strip_nodes: np.ndarray,
    transforms: np.ndarray,
) -> None:
    """Add all 8 deformed FP64 spring-steel strips in satin carbon-black (true 16 mm width, thin 0.35 mm blade)."""
    # Rich satin carbon-black 65Mn spring steel (per explicit visual rule: all steel strips black)
    steel_rgba = np.array([0.095, 0.102, 0.114, 1.0], dtype=np.float32)
    bolt_rgba = np.array([0.34, 0.36, 0.40, 1.0], dtype=np.float32)

    R_p = [R_UTM @ _quat_xyzw_to_mat(x_row[p, 3:7]) for p in range(2)]
    P_p = [_sim_to_utm_world(x_row[p, :3].astype(np.float64), p0_rest, z_bot) for p in range(2)]
    seg_center = 0.5 * (P_p[0] + P_p[1])

    for sid in range(8):
        nodes = strip_nodes[sid]
        pts = _sim_to_utm_world(x_row[nodes, :3].astype(np.float64), p0_rest, z_bot)
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
            rad[2] = 0.0
            ez = rad - np.dot(rad, ex) * ex
            ez /= max(float(np.linalg.norm(ez)), 1e-9)
            ey = np.cross(ez, ex)
            ey /= max(float(np.linalg.norm(ey)), 1e-9)
            ez = np.cross(ex, ey)
            rot = np.column_stack((ex, ey, ez)).reshape(9)
            # Bold 18.4 mm ribbon width (half=0.0092 m), razor-thin spring-steel blade thickness (half=0.00028 m)
            size = np.array([0.55 * length, 0.0092, 0.00028], dtype=np.float64)
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_BOX), size, mid, rot, steel_rgba)
            g.shininess = 0.88
            g.specular = 0.78
            scene.ngeom += 1
        for end_pt in (pts[0], pts[-1]):
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE),
                np.array([0.0022, 0.0022, 0.0022]), end_pt, np.eye(3).reshape(9), bolt_rgba,
            )
            scene.ngeom += 1

    # Helper to render a smooth multi-segment flexible curve as connected capsules
    def _add_polyline(pts_3d: np.ndarray, radius: float, rgba: np.ndarray) -> None:
        for i in range(len(pts_3d) - 1):
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                np.zeros(3), np.zeros(3), np.eye(3).reshape(9), rgba,
            )
            mujoco.mjv_connector(scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE), radius, pts_3d[i], pts_3d[i + 1])
            scene.ngeom += 1

    # 1) 4 braided PE tendon lines: top horn tie-pin -> top plate guide rim -> smooth slack curve -> bottom plate hole
    tendon_rgba = np.array([0.74, 0.76, 0.78, 0.92], dtype=np.float32)
    z_top_plate = float(P_p[1][2])
    z_bot_disc = float(P_p[0][2] + 0.0165)
    span_z = max(z_top_plate - z_bot_disc, 0.03)
    # Excess rope length as top plate compresses from rest (~0.101 m) down to ~0.056 m
    slack_m = max(0.0, 0.101 - span_z)
    bow_amp = 0.004 + 0.50 * math.sqrt(slack_m * 0.055) if slack_m > 1e-4 else 0.003

    us = np.linspace(0.0, 1.0, 9)
    for y_horn, y_rim in ((+0.0135, +0.025), (-0.0135, -0.025)):
        for x_sgn in (-1.0, +1.0):
            # Horn tie-pin (local x_sim=-0.0268 -> +0.0268 above top disc; z_sim=+/-0.0235 -> x_w = -/+0.0235)
            p_horn = np.array([x_sgn * 0.0235, -y_horn, z_top_plate + 0.0268])
            p_rim_top = np.array([x_sgn * 0.044, -y_rim, z_top_plate + 0.002])
            p_rim_bot = np.array([x_sgn * 0.044, -y_rim, z_bot_disc])
            # Smooth upper routing from horn pin to top plate guide notch
            p_knee = np.array([x_sgn * 0.038, -0.5 * (y_horn + y_rim), z_top_plate + 0.021])
            _add_polyline(np.vstack([p_horn, p_knee, p_rim_top]), 0.00068, tendon_rgba)
            # Smooth hanging/bowed span between top and bottom guide holes
            out_dir = np.array([x_sgn * 0.92, -np.sign(y_rim) * 0.39, 0.0])
            pts_span = np.array([
                (1.0 - u) * p_rim_top + u * p_rim_bot + out_dir * (bow_amp * math.sin(math.pi * u))
                for u in us
            ])
            _add_polyline(pts_span, 0.00068, tendon_rgba)

    # Bottom return tendon loops draped cleanly under lower acrylic disc (matching real video bottom cords)
    for y_loop in (-0.025, +0.025):
        pts_bot_loop = np.array([
            [-0.044, y_loop, z_bot_disc],
            [-0.045, y_loop, z_bot_disc - 0.017],
            [-0.022, y_loop * 0.85, z_bot_disc - 0.021],
            [+0.022, y_loop * 0.85, z_bot_disc - 0.021],
            [+0.045, y_loop, z_bot_disc - 0.017],
            [+0.044, y_loop, z_bot_disc],
        ])
        _add_polyline(pts_bot_loop, 0.00072, tendon_rgba)

    # 2) Black 3-wire servo bus cable harness hanging gently inside the barrel between upper & lower servos
    cable_rgba = np.array([0.07, 0.075, 0.085, 1.0], dtype=np.float32)
    cable_slack = 0.010 + 0.55 * slack_m
    us_c = np.linspace(0.0, 1.0, 11)
    for idx_c, (x_off, y_off, bend_x) in enumerate(((-0.003, -0.012, -1.0), (+0.002, -0.009, -0.5))):
        p_c_top = np.array([x_off, y_off, z_top_plate - 0.032])
        p_c_bot = np.array([x_off - 0.004, y_off + 0.005, z_bot_disc + 0.006])
        pts_cab = np.array([
            (1.0 - u) * p_c_top + u * p_c_bot + np.array([
                bend_x * cable_slack * math.sin(math.pi * u),
                0.006 * math.sin(math.pi * u),
                -0.004 * math.sin(math.pi * u),
            ])
            for u in us_c
        ])
        _add_polyline(pts_cab, 0.00105, cable_rgba)


def _draw_clean_label(img: np.ndarray, text: str, x: int = 24, y: int = 16, font_size: int = 30) -> np.ndarray:
    """Clean borderless/dotless/background-free bold label with subtle drop shadow."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, 'RGBA')
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 185), font=font)
    draw.text((x, y), text, fill=(248, 250, 252, 255), font=font)
    return np.asarray(im)


class _FastLiveCurveStrip:
    """Pre-renders static matplotlib curves once and overlays live moving cursors in ~1 ms/frame."""

    def __init__(self, width: int, height: int, utm: dict, sim_cal: dict, sim_def: dict):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import ImageFont
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.width = width
        self.height = height
        self.utm = utm
        self.sim_cal = sim_cal
        self.sim_def = sim_def
        try:
            self.font_title = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 15)
        except Exception:
            self.font_title = ImageFont.load_default()

        dpi = 100
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / dpi, height / dpi), dpi=dpi, facecolor='#f8fafc')
        fig.subplots_adjust(left=0.062, right=0.985, top=0.84, bottom=0.19, wspace=0.17)

        xg = utm['grid_mm']
        # Left panel: Force-Displacement
        ax1.set_facecolor('white')
        ax1.fill_between(xg, utm['mean_unload_N'], utm['mean_load_N'], color='#93c5fd', alpha=0.42, label='实测 5 循环回滞带')
        ax1.plot(xg, utm['mean_load_N'], color='#1d4ed8', lw=2.0, label=f"拉伸机实测 ({utm['mean_load_N'][-1]:.1f} N)")
        ax1.plot(sim_cal['disp_mm'], sim_cal['force_N'], color='#dc2626', lw=2.2, ls='--',
                 label=f"标定仿真 t=0.22 mm ({sim_cal['force_N'][-1]:.1f} N)")
        ax1.plot(sim_def['disp_mm'], sim_def['force_N'], color='#64748b', lw=1.8, ls=':',
                 label=f"旧默认 t=0.15 mm ({sim_def['force_N'][-1]:.1f} N)")
        ax1.set_xlabel('轴向压缩位移 / mm', fontsize=10)
        ax1.set_ylabel('轴向压缩力 / N', fontsize=10)
        ax1.set_xlim(-1.0, 46.5)
        ax1.set_ylim(-2.0, 46.5)
        ax1.grid(alpha=0.25)
        ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.92)

        # Right panel: Tangent Stiffness (plotted cleanly over 1..44 mm)
        ax2.set_facecolor('white')
        k_real = np.gradient(utm['mean_mid_N'], xg)
        win = 17
        k_real_s = np.convolve(np.pad(k_real, (win // 2, win // 2), mode='edge'), np.ones(win) / win, mode='valid')
        self.k_cal = np.gradient(sim_cal['force_N'], sim_cal['disp_mm'])
        k_def = np.gradient(sim_def['force_N'], sim_def['disp_mm'])
        mk = (xg >= 1.5) & (xg <= 43.5)
        ax2.plot(xg[mk], k_real_s[mk], color='#1d4ed8', lw=2.0, label='实测切线刚度 (1.92 → 0.77 N/mm)')
        ax2.plot(sim_cal['disp_mm'][1:-1], self.k_cal[1:-1], color='#dc2626', lw=2.2, ls='--', label='标定仿真切线刚度')
        ax2.plot(sim_def['disp_mm'][1:-1], k_def[1:-1], color='#64748b', lw=1.8, ls=':', label='旧默认切线刚度 (~0.13 N/mm)')
        ax2.set_title('预弯钢片非线性软化刚度曲线', fontsize=11.5, fontweight='bold', color='#0f172a', pad=10)
        ax2.set_xlabel('轴向压缩位移 / mm', fontsize=10)
        ax2.set_ylabel('切线刚度 / (N/mm)', fontsize=10)
        ax2.set_xlim(-1.0, 46.5)
        ax2.set_ylim(0.0, 2.25)
        ax2.grid(alpha=0.25)
        ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.92)

        fig.canvas.draw()
        self.bg = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

        def _axis_box(ax):
            p0 = ax.transData.transform((ax.get_xlim()[0], ax.get_ylim()[0]))
            p1 = ax.transData.transform((ax.get_xlim()[1], ax.get_ylim()[1]))
            return (
                ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1],
                float(p0[0]), float(p1[0]), float(height - p0[1]), float(height - p1[1]),
            )

        self.box1 = _axis_box(ax1)
        self.box2 = _axis_box(ax2)
        plt.close(fig)

    @staticmethod
    def _to_px(box, x, y):
        x0, x1, y0, y1, px0, px1, py0, py1 = box
        px = int(round(px0 + (x - x0) / (x1 - x0) * (px1 - px0)))
        py = int(round(py0 + (y - y0) / (y1 - y0) * (py1 - py0)))
        return px, py

    def render(self, cur_disp_mm: float, cur_f_cal: float, cur_f_def: float, is_loading: bool) -> np.ndarray:
        from PIL import Image, ImageDraw
        xg = self.utm['grid_mm']
        f_real_now = float(np.interp(cur_disp_mm, xg, self.utm['mean_load_N'] if is_loading else self.utm['mean_unload_N']))
        f_real_disp = 0.0 if abs(f_real_now) < 0.05 else f_real_now
        f_cal_disp = 0.0 if abs(cur_f_cal) < 0.05 else cur_f_cal

        x_k = float(np.clip(cur_disp_mm, 1.0, 44.0))
        k_now_cal = float(np.interp(x_k, self.sim_cal['disp_mm'][1:-1], self.k_cal[1:-1]))

        im = Image.fromarray(self.bg.copy())
        draw = ImageDraw.Draw(im)

        px1, _ = self._to_px(self.box1, cur_disp_mm, 0.0)
        draw.line([(px1, int(self.box1[7])), (px1, int(self.box1[6]))], fill=(15, 23, 42), width=1)
        px2, _ = self._to_px(self.box2, x_k, 0.0)
        draw.line([(px2, int(self.box2[7])), (px2, int(self.box2[6]))], fill=(15, 23, 42), width=1)

        for val, col, r in ((f_real_now, (29, 78, 216), 6), (cur_f_cal, (220, 38, 38), 5), (cur_f_def, (100, 116, 139), 5)):
            mx, my = self._to_px(self.box1, cur_disp_mm, val)
            draw.ellipse([mx - r, my - r, mx + r, my + r], fill=col, outline=(255, 255, 255), width=1)

        mx2, my2 = self._to_px(self.box2, x_k, k_now_cal)
        draw.ellipse([mx2 - 5, my2 - 5, mx2 + 5, my2 + 5], fill=(220, 38, 38), outline=(255, 255, 255), width=1)

        phase_txt = '压缩加载' if is_loading else '回弹卸载'
        title = f'实时力-位移对照（{phase_txt} {cur_disp_mm:4.1f} mm · 实测 {f_real_disp:4.1f} N / 标定仿真 {f_cal_disp:4.1f} N）'
        cx1 = int(0.5 * (self.box1[4] + self.box1[5]))
        bbox = draw.textbbox((0, 0), title, font=self.font_title)
        tw = bbox[2] - bbox[0]
        draw.text((cx1 - tw // 2, 14), title, fill=(15, 23, 42), font=self.font_title)

        return np.asarray(im)


def _setup_utm_scene(out_dir: Path, width: int, height: int):
    tex_path = out_dir / '_utm_breadboard_tex.png'
    _make_breadboard_texture(tex_path)
    z_bot = 0.019  # lower acrylic disc at z = 0.0355 m, bottom servo neatly seated on base flange (z=0.007)
    xml = _build_utm_mjcf(tex_path, z_bot)
    model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    id_bot = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'seg_bot')]
    id_top = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'seg_top')]
    id_plat = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'utm_platen')]

    # Telephoto camera calibrated 1:1 to real UTM video framing (fovy=25.5 deg)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 90.0
    cam.elevation = -8.0
    cam.distance = 0.565
    cam.lookat[:] = [-0.012, 0.0, 0.098]
    return model, mj_data, renderer, cam, z_bot, (id_bot, id_top, id_plat)


def _render_single_utm_frame(model, mj_data, renderer, cam, z_bot, ids, xr, sim_cal) -> np.ndarray:
    id_bot, id_top, id_plat = ids
    p0_rest = sim_cal['rest'][0, :3].astype(np.float64)
    p_bot_w = _sim_to_utm_world(xr[0, :3].astype(np.float64), p0_rest, z_bot)
    p_top_w = _sim_to_utm_world(xr[1, :3].astype(np.float64), p0_rest, z_bot)
    q_bot_w = _mat_to_quat_wxyz(R_UTM @ _quat_xyzw_to_mat(xr[0, 3:7]))
    q_top_w = _mat_to_quat_wxyz(R_UTM @ _quat_xyzw_to_mat(xr[1, 3:7]))

    mj_data.mocap_pos[id_bot] = p_bot_w
    mj_data.mocap_quat[id_bot] = q_bot_w
    mj_data.mocap_pos[id_top] = p_top_w
    mj_data.mocap_quat[id_top] = q_top_w
    # Platen rests directly on top of aluminum bridge top bar (0.0445 m above top plate + 0.017 m platen half-height)
    mj_data.mocap_pos[id_plat] = np.array([0.0, 0.0, p_top_w[2] + 0.0445 + 0.017], dtype=np.float64)
    mj_data.mocap_quat[id_plat] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    mujoco.mj_forward(model, mj_data)
    renderer.update_scene(mj_data, camera=cam)
    _add_utm_strips(
        renderer.scene, xr, p0_rest, z_bot,
        sim_cal['strip_nodes'], sim_cal['transforms'],
    )
    return renderer.render()


def render_utm_videos(
    sim_cal: dict,
    sim_def: dict,
    utm: dict,
    vid_info: dict,
    out_dir: Path,
    render_scale: tuple[int, int] = (640, 360),
    preview_only: bool = False,
) -> dict[str, Path]:
    """Render both standalone 1:1 MuJoCo UTM scene video and side-by-side real vs sim comparison video + GIF."""
    import cv2
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W_tile, H_tile = render_scale
    model, mj_data, renderer, cam, z_bot, ids = _setup_utm_scene(out_dir, W_tile, H_tile)

    fps = vid_info['fps']
    n_frames = vid_info['n_frames']
    peak_frame = vid_info['peak_frame']
    mid_frame = peak_frame // 2

    if preview_only:
        rows = []
        for fi, tag in ((0, '0 mm'), (peak_frame, '45 mm')):
            sim_rgb = _render_single_utm_frame(model, mj_data, renderer, cam, z_bot, ids, sim_cal['video_states'][fi], sim_cal)
            real_rgb = cv2.resize(vid_info['frames_rgb'][fi], (W_tile, H_tile), interpolation=cv2.INTER_AREA)
            r_lbl = _draw_clean_label(real_rgb, f'实测 {tag}', x=22, y=14, font_size=28)
            s_lbl = _draw_clean_label(sim_rgb, f'仿真 {tag}', x=22, y=14, font_size=28)
            rows.append(np.concatenate([r_lbl, s_lbl], axis=1))
        renderer.close()
        chk = out_dir / '_alignment_check.png'
        Image.fromarray(np.concatenate(rows, axis=0)).save(chk)
        return {'alignment_check': chk}

    H_curve = 300
    W_comp = 2 * W_tile
    H_comp = H_tile + H_curve
    curve_renderer = _FastLiveCurveStrip(W_comp, H_curve, utm, sim_cal, sim_def)

    scene_mp4 = out_dir / 'utm_mujoco_scene.mp4'
    comp_mp4 = out_dir / 'utm_real_vs_sim_comparison.mp4'
    comp_gif = out_dir / 'utm_real_vs_sim_comparison.gif'
    preview_png = out_dir / 'utm_scene_side_by_side_preview.png'

    cmd_scene = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W_tile}x{H_tile}', '-pix_fmt', 'rgb24', '-r', f'{fps:.3f}',
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        str(scene_mp4),
    ]
    cmd_comp = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W_comp}x{H_comp}', '-pix_fmt', 'rgb24', '-r', f'{fps:.3f}',
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        str(comp_mp4),
    ]
    p_scene = subprocess.Popen(cmd_scene, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    p_comp = subprocess.Popen(cmd_comp, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    milestone_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for fi in range(n_frames):
        sim_rgb = _render_single_utm_frame(model, mj_data, renderer, cam, z_bot, ids, sim_cal['video_states'][fi], sim_cal)
        p_scene.stdin.write(sim_rgb.tobytes())

        real_rgb = cv2.resize(vid_info['frames_rgb'][fi], (W_tile, H_tile), interpolation=cv2.INTER_AREA)

        if fi == 0:
            milestone_frames['0mm'] = (real_rgb.copy(), sim_rgb.copy())
        elif fi == mid_frame:
            milestone_frames['22mm'] = (real_rgb.copy(), sim_rgb.copy())
        elif fi == peak_frame:
            milestone_frames['45mm'] = (real_rgb.copy(), sim_rgb.copy())

        real_lbl = _draw_clean_label(real_rgb, '实测', x=22, y=14, font_size=30)
        sim_lbl = _draw_clean_label(sim_rgb, '仿真', x=22, y=14, font_size=30)
        top_row = np.concatenate([real_lbl, sim_lbl], axis=1)

        d_now = float(vid_info['disp_mm'][fi])
        f_cal_now = float(sim_cal['video_forces_N'][fi])
        f_def_now = float(np.interp(d_now, sim_def['disp_mm'], sim_def['force_N']))
        curve_strip = curve_renderer.render(d_now, f_cal_now, f_def_now, is_loading=(fi <= peak_frame))
        comp_frame = np.concatenate([top_row, curve_strip], axis=0)
        p_comp.stdin.write(comp_frame.tobytes())

        if fi == peak_frame:
            Image.fromarray(comp_frame).save(preview_png)

    for proc in (p_scene, p_comp):
        proc.stdin.close()
        err = proc.stderr.read().decode('utf-8', errors='replace')
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f'ffmpeg failed ({rc}): {err[-400:]}')
    renderer.close()

    subprocess.check_call(
        [
            'ffmpeg', '-y', '-v', 'error', '-i', str(comp_mp4),
            '-vf', 'fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a',
            str(comp_gif),
        ]
    )

    return {
        'scene_mp4': scene_mp4,
        'comp_mp4': comp_mp4,
        'comp_gif': comp_gif,
        'preview_png': preview_png,
        'milestone_frames': milestone_frames,
    }


def plot_utm_comparison_curves(
    utm: dict,
    sim_cal: dict,
    sim_def: dict,
    milestone_frames: dict,
    out_png: Path,
) -> dict:
    """Create 4-panel publication-grade UTM experimental vs simulation comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    xg = utm['grid_mm']
    f_cal_on_xg = np.interp(xg, sim_cal['disp_mm'], sim_cal['force_N'])
    f_def_on_xg = np.interp(xg, sim_def['disp_mm'], sim_def['force_N'])

    err_cal = f_cal_on_xg - utm['mean_mid_N']
    err_def = f_def_on_xg - utm['mean_mid_N']
    rmse_cal = float(np.sqrt(np.mean(err_cal ** 2)))
    rmse_def = float(np.sqrt(np.mean(err_def ** 2)))
    ss_tot = float(np.sum((utm['mean_mid_N'] - utm['mean_mid_N'].mean()) ** 2))
    r2_cal = 1.0 - float(np.sum(err_cal ** 2)) / ss_tot
    r2_def = 1.0 - float(np.sum(err_def ** 2)) / ss_tot

    fig = plt.figure(figsize=(14.5, 9.6), dpi=180, facecolor='white')
    gs = GridSpec(2, 2, height_ratios=[1.05, 1.0], hspace=0.30, wspace=0.22, left=0.06, right=0.97, top=0.93, bottom=0.07)

    # Panel 1: Force-Displacement (0 -> 45 mm)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(xg, utm['mean_unload_N'], utm['mean_load_N'], color='#93c5fd', alpha=0.45, label='实测 5 循环加载/卸载回滞带')
    ax1.plot(xg, utm['mean_load_N'], color='#1d4ed8', lw=1.7, alpha=0.9, label=f"实测加载均值 ({utm['mean_load_N'][-1]:.2f} N)")
    ax1.plot(xg, utm['mean_unload_N'], color='#0284c7', lw=1.5, ls='-.', alpha=0.9, label=f"实测卸载均值 ({utm['mean_unload_N'][-1]:.2f} N)")
    ax1.plot(sim_cal['disp_mm'], sim_cal['force_N'], color='#dc2626', lw=2.3, marker='o', markersize=4.5, markevery=3,
             label=f"标定 FP64 (t=0.22 mm, 拱高 30 mm)\nRMSE={rmse_cal:.2f} N, R²={r2_cal:.5f}")
    ax1.plot(sim_def['disp_mm'], sim_def['force_N'], color='#64748b', lw=2.0, ls='--', marker='s', markersize=4.0, markevery=3,
             label=f"旧默认参数 (t=0.15 mm, 拱高 50.7 mm)\n峰值仅 {sim_def['force_N'][-1]:.2f} N (偏软 {utm['mean_mid_N'][-1]/sim_def['force_N'][-1]:.1f}×)")
    ax1.set_title('1  单体节轴向压缩力-位移对照（0 → 45 → 0 mm）', fontsize=12, fontweight='bold')
    ax1.set_xlabel('轴向压缩量 x / mm', fontsize=10.5)
    ax1.set_ylabel('轴向压缩力 F / N', fontsize=10.5)
    ax1.set_xlim(-0.5, 46.0)
    ax1.set_ylim(-1.0, 46.5)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8.5, loc='upper left', framealpha=0.94)

    # Panel 2: Tangent Stiffness Nonlinear Softening
    ax2 = fig.add_subplot(gs[0, 1])
    k_real = np.gradient(utm['mean_mid_N'], xg)
    win = 17
    k_real_s = np.convolve(np.pad(k_real, (win // 2, win // 2), mode='edge'), np.ones(win) / win, mode='valid')
    k_cal = np.gradient(sim_cal['force_N'], sim_cal['disp_mm'])
    k_def = np.gradient(sim_def['force_N'], sim_def['disp_mm'])
    mk = (xg >= 1.5) & (xg <= 43.5)
    ax2.plot(xg[mk], k_real_s[mk], color='#1d4ed8', lw=2.3, label='实测切线刚度 dF/dx (1.92 → 0.77 N/mm)')
    ax2.plot(sim_cal['disp_mm'][1:-1], k_cal[1:-1], color='#dc2626', lw=2.3, marker='o', markersize=4.5, markevery=2,
             label='标定 FP64 钢片切线刚度')
    ax2.plot(sim_def['disp_mm'][1:-1], k_def[1:-1], color='#64748b', lw=2.0, ls='--',
             label=f'旧默认参数切线刚度 (均值 {np.mean(k_def[1:-1]):.2f} N/mm)')
    ax2.axhline(utm['mean_mid_N'][-1] / 45.0, color='#0f766e', ls=':', lw=1.6,
                label=f"全程等效割线刚度 k_eq = {utm['mean_mid_N'][-1]/45.0:.3f} N/mm")
    ax2.set_title('2  预弯钢片非线性软化切线刚度对比', fontsize=12, fontweight='bold')
    ax2.set_xlabel('轴向压缩量 x / mm', fontsize=10.5)
    ax2.set_ylabel('切线刚度 k(x) / (N/mm)', fontsize=10.5)
    ax2.set_xlim(-0.5, 46.0)
    ax2.set_ylim(0.0, 2.25)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8.8, loc='upper right', framealpha=0.94)

    # Panel 3: Real 5-Cycle Hysteresis + Calibrated Simulation Residual Error
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(xg, 0.0, utm['hysteresis_N'], color='#60a5fa', alpha=0.35)
    ax3.plot(xg, utm['hysteresis_N'], color='#1d4ed8', lw=2.0,
             label=f"实测加/卸载回滞差 F_load - F_unload\n(均值 {utm['mean_hyst_N']:.2f} N, 峰值 {utm['max_hyst_N']:.2f} N, 能量耗散 {utm['steady_energy_loss_pct']:.2f}%)")
    ax3.plot(xg, err_cal, color='#dc2626', lw=2.0, ls='--',
             label=f"标定 FP64 残差 F_sim - F_实测均值\n(RMSE = {rmse_cal:.2f} N, 最大偏差 {np.max(np.abs(err_cal)):.2f} N)")
    ax3.axhline(0.0, color='#334155', lw=1.0, alpha=0.6)
    ax3.set_title('3  实测结构回滞耗散与标定仿真残差分析', fontsize=12, fontweight='bold')
    ax3.set_xlabel('轴向压缩量 x / mm', fontsize=10.5)
    ax3.set_ylabel('力差值 / N', fontsize=10.5)
    ax3.set_xlim(-0.5, 46.0)
    ax3.set_ylim(-0.65, 0.85)
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8.5, loc='upper right', framealpha=0.94)

    # Panel 4: 3-stage Real vs 1:1 MuJoCo Visual Deformation Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_title('4  关键压缩位点实拍 vs 1:1 仿真形变对照（0 / 22.5 / 45 mm）', fontsize=12, fontweight='bold', pad=8)
    keys = [('0mm', '0 mm'), ('22mm', '22.5 mm'), ('45mm', '45 mm')]
    inner_gs = gs[1, 1].subgridspec(2, 3, wspace=0.04, hspace=0.06)
    for col, (k, lbl) in enumerate(keys):
        real_im, sim_im = milestone_frames[k]
        H, W = real_im.shape[:2]
        r_crop = real_im[:, int(W * 0.08):int(W * 0.92)]
        s_crop = sim_im[:, int(W * 0.08):int(W * 0.92)]
        ax_r = fig.add_subplot(inner_gs[0, col])
        ax_r.imshow(r_crop)
        ax_r.set_xticks([]); ax_r.set_yticks([])
        ax_r.set_title(lbl, fontsize=9.5, fontweight='bold', pad=2)
        if col == 0:
            ax_r.set_ylabel('实测', fontsize=9.5, fontweight='bold')
        ax_s = fig.add_subplot(inner_gs[1, col])
        ax_s.imshow(s_crop)
        ax_s.set_xticks([]); ax_s.set_yticks([])
        if col == 0:
            ax_s.set_ylabel('仿真', fontsize=9.5, fontweight='bold')

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    bulge_ratio_cal = float(sim_cal['bulge_diam_mm'][-1] / sim_cal['bulge_diam_mm'][0])
    bulge_ratio_def = float(sim_def['bulge_diam_mm'][-1] / sim_def['bulge_diam_mm'][0])
    return {
        'rmse_cal_N': round(rmse_cal, 4),
        'r2_cal': round(r2_cal, 6),
        'max_err_cal_N': round(float(np.max(np.abs(err_cal))), 4),
        'rmse_def_N': round(rmse_def, 4),
        'r2_def': round(r2_def, 4),
        'real_peak_load_N': round(float(utm['mean_load_N'][-1]), 3),
        'real_peak_unload_N': round(float(utm['mean_unload_N'][-1]), 3),
        'real_peak_mid_N': round(float(utm['mean_mid_N'][-1]), 3),
        'cal_peak_N': round(float(sim_cal['force_N'][-1]), 3),
        'def_peak_N': round(float(sim_def['force_N'][-1]), 3),
        'real_mean_hyst_N': round(utm['mean_hyst_N'], 3),
        'real_max_hyst_N': round(utm['max_hyst_N'], 3),
        'real_steady_energy_loss_pct': round(utm['steady_energy_loss_pct'], 3),
        'cal_bulge_mm_0_to_45': [round(float(sim_cal['bulge_diam_mm'][0]), 1), round(float(sim_cal['bulge_diam_mm'][-1]), 1)],
        'cal_bulge_ratio': round(bulge_ratio_cal, 3),
        'def_bulge_mm_0_to_45': [round(float(sim_def['bulge_diam_mm'][0]), 1), round(float(sim_def['bulge_diam_mm'][-1]), 1)],
        'def_bulge_ratio': round(bulge_ratio_def, 3),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='1:1 UTM compression experiment vs FP64 simulation comparison')
    parser.add_argument('--video', type=Path, required=True, help='Path to real UTM compression MP4 video')
    parser.add_argument('--raw-csv', type=Path, required=True, help='Path to raw 5-cycle UTM CSV file')
    parser.add_argument('--out-dir', type=Path, default=Path('runs/utm_validation'))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--preview-only', action='store_true', help='Only render 0 mm + 45 mm alignment check PNG')
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    utm = load_raw_utm_cycles(args.raw_csv)
    vid_info = track_video_platen_displacement(args.video, max_stroke_mm=45.0)

    disp_grid = np.linspace(0.0, 45.0, 46)
    hw_cal = HardwareConfig.bench_exp128_calibrated()
    hw_def = HardwareConfig.ds3218_default()

    if args.preview_only:
        # Solve only 2 video frames (0 and peak) for instant visual check
        d_pair = np.array([vid_info['disp_mm'][0], vid_info['disp_mm'][vid_info['peak_frame']]], dtype=np.float64)
        sim_cal = simulate_utm_compression(hw_cal, d_pair, video_disp_mm=None, device=args.device)
        vs = np.zeros((vid_info['n_frames'], sim_cal['grid_states'].shape[1], 7), dtype=np.float32)
        vs[0] = sim_cal['grid_states'][0].astype(np.float32)
        vs[vid_info['peak_frame']] = sim_cal['grid_states'][1].astype(np.float32)
        sim_cal['video_states'] = vs
        out = render_utm_videos(sim_cal, sim_cal, utm, vid_info, args.out_dir, preview_only=True)
        print('Alignment check saved:', out['alignment_check'])
        return

    sim_cal = simulate_utm_compression(hw_cal, disp_grid, video_disp_mm=vid_info['disp_mm'], device=args.device)
    sim_def = simulate_utm_compression(hw_def, disp_grid, video_disp_mm=None, device=args.device)

    vid_out = render_utm_videos(sim_cal, sim_def, utm, vid_info, args.out_dir)
    metrics = plot_utm_comparison_curves(
        utm, sim_cal, sim_def, vid_out['milestone_frames'],
        args.out_dir / 'utm_comparison_curves.png',
    )
    (args.out_dir / 'utm_validation_summary.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved UTM comparison outputs to {args.out_dir}")


if __name__ == '__main__':
    main()

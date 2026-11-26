"""Photorealistic 3D MuJoCo visualizer for real 40-strip FP64 trajectories.

Renders the full 5-segment robot using the actual CAD STL plates and wheels
plus all 40 real-time deforming pre-curved spring-steel ribbons (640 oriented
elements per robot) and 20 tendon cords on a lit MuJoCo studio floor.

Note: MuJoCo is used purely for high-fidelity 3D visualization; all state
trajectories come directly from the FP64 corotational-beam CUDA physics core.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import numpy as np
import mujoco

from serpworm.paths import ASSETS_DIR

__all__ = ['render_mujoco_video']

MESH_DIR = ASSETS_DIR / 'cad' / 'meshes' / 'longworm2'

# Gleaming cold-rolled blue-silver 65Mn spring-steel ribbons across all 5 segments (matches real robot)
_COLD_ROLLED_STEEL = np.array([0.48, 0.60, 0.70, 1.0], dtype=np.float32)
SEG_STEEL_RGBA = (_COLD_ROLLED_STEEL,) * 5
TENDON_RGBA = np.array([0.20, 0.22, 0.26, 0.92], dtype=np.float32)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, w] (GPUCore) to [w, x, y, z] (MuJoCo)."""
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


def _build_mjcf(num_robots: int, mounts: np.ndarray, start_head_x: float, lane_ys: list[float]) -> str:
    """Build studio MJCF XML referencing real CAD STL plates and wheels."""
    mesh_dir = MESH_DIR.resolve().as_posix()
    bodies = []
    for r in range(num_robots):
        for p in range(10):
            mesh_name = 'plate_front' if p % 2 == 0 else 'plate_back'
            mat_name = 'mat_plate_front' if p % 2 == 0 else 'mat_plate_back'
            # Wheel mounts relative to plate frame
            w_geoms = []
            for s in range(2):
                mx, my, mz = mounts[p, s]
                # Wheel drop strut connecting CAD bracket (~z=-0.068) to wheel axle
                strut_top = -0.064
                strut_mid_z = 0.5 * (strut_top + mz)
                strut_half = max(0.005, 0.5 * abs(strut_top - mz))
                w_geoms.append(
                    f'<geom type="box" pos="{mx:.5f} {my:.5f} {strut_mid_z:.5f}" '
                    f'size="0.005 0.0035 {strut_half:.5f}" material="mat_strut" contype="0" conaffinity="0"/>'
                )
                w_geoms.append(
                    f'<geom type="mesh" mesh="wheel_mesh" pos="{mx:.5f} {my:.5f} {mz:.5f}" '
                    f'material="mat_wheel" contype="0" conaffinity="0"/>'
                )
                w_geoms.append(
                    f'<geom type="cylinder" pos="{mx:.5f} {my:.5f} {mz:.5f}" euler="1.5708 0 0" '
                    f'size="0.0085 0.0086" material="mat_bearing" contype="0" conaffinity="0"/>'
                )
            servo_geoms = []
            if p % 2 == 1:
                # Dual internal 25 kg metallic-purple tendon servos on rear plate of each segment
                for sy in (+0.0285, -0.0285):
                    servo_geoms.append(
                        f'<geom type="box" pos="0.018 {sy:+.4f} 0.000" size="0.017 0.0102 0.0088" '
                        f'material="mat_servo_purple" contype="0" conaffinity="0"/>'
                    )
            else:
                # Front yaw servo purple mid-band
                servo_geoms.append(
                    '<geom type="box" pos="0.004 0 -0.003" size="0.015 0.0098 0.0088" '
                    'material="mat_servo_purple" contype="0" conaffinity="0"/>'
                )
            body_xml = (
                f'    <body name="r{r}_p{p}" mocap="true" pos="0 0 0.13">\n'
                f'      <geom type="mesh" mesh="{mesh_name}" material="{mat_name}" contype="0" conaffinity="0"/>\n'
                + '\n'.join(f'      {g}' for g in servo_geoms) + '\n'
                + '\n'.join(f'      {g}' for g in w_geoms) + '\n'
                f'    </body>'
            )
            bodies.append(body_xml)

    # Flush reference grid markers (red start line at initial head X, subtle 200 mm ticks ahead)
    y_min = min(lane_ys) - 0.35
    y_max = max(lane_ys) + 0.35
    y_mid = 0.5 * (y_min + y_max)
    y_half = 0.5 * (y_max - y_min)
    markers = [
        f'    <geom name="start_line" type="box" pos="{start_head_x:.4f} {y_mid:.4f} 0.0008" '
        f'size="0.0035 {y_half:.4f} 0.0008" rgba="0.88 0.24 0.24 0.85" contype="0" conaffinity="0"/>'
    ]
    for step_mm in (200, 400, 600, 800, 1000, 1200, 1400, 1600):
        tx = start_head_x - step_mm * 1e-3
        markers.append(
            f'    <geom name="tick_{step_mm}" type="box" pos="{tx:.4f} {y_mid:.4f} 0.0006" '
            f'size="0.0020 {y_half:.4f} 0.0006" rgba="0.22 0.68 0.48 0.42" contype="0" conaffinity="0"/>'
        )

    return f"""<mujoco model="serpworm_3d_studio">
  <compiler angle="radian" meshdir="{mesh_dir}" autolimits="true"/>
  <visual>
    <global offwidth="1280" offheight="720" fovy="40"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.05" zfar="20.0" shadowclip="2.2"/>
    <headlight ambient="0.19 0.20 0.22" diffuse="0.06 0.06 0.06" specular="0.0 0.0 0.0"/>
    <rgba haze="0.78 0.81 0.85 1"/>
  </visual>
  <asset>
    <mesh name="plate_front" file="front2_Link.STL"/>
    <mesh name="plate_back"  file="back2_Link.STL"/>
    <mesh name="wheel_mesh"  file="w2-1_Link.STL"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.72 0.76 0.81" rgb2="0.82 0.85 0.89" width="512" height="512"/>
    <!-- Clean calm matte studio tiles (33 cm tile pitch, zero shadow acne / zero white blotches) -->
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.68 0.71 0.75" rgb2="0.64 0.67 0.71" width="512" height="512"/>
    <material name="mat_floor"        texture="grid_tex" texrepeat="3 3" texuniform="true" reflectance="0.0" shininess="0.05" specular="0.03"/>
    <material name="mat_plate_front"  rgba="0.78 0.82 0.87 0.72" specular="0.85" shininess="0.80" reflectance="0.10"/>
    <material name="mat_plate_back"   rgba="0.34 0.38 0.44 1.0"  specular="0.65" shininess="0.60" reflectance="0.08"/>
    <material name="mat_servo_purple" rgba="0.44 0.23 0.64 1.0"  specular="0.82" shininess="0.78" reflectance="0.12"/>
    <material name="mat_strut"        rgba="0.44 0.48 0.53 1.0"  specular="0.58" shininess="0.52"/>
    <material name="mat_wheel"        rgba="0.14 0.15 0.17 1.0"  specular="0.22" shininess="0.20"/>
    <material name="mat_bearing"      rgba="0.56 0.58 0.62 1.0"  specular="0.85" shininess="0.82"/>
  </asset>
  <worldbody>
    <!-- Balanced 3-point studio lighting: total diffuse ~0.88 (zero floor washout) + high specular for gleaming steel -->
    <light name="key"  directional="true" pos="0 0 3.2" dir="0.35 0.42 -0.84" diffuse="0.52 0.51 0.50" specular="0.90 0.90 0.88" castshadow="true"/>
    <light name="rim"  directional="true" pos="0 0 2.6" dir="-0.55 -0.35 -0.76" diffuse="0.16 0.18 0.21" specular="0.82 0.84 0.88" castshadow="false"/>
    <light name="fill" directional="true" pos="0 0 2.4" dir="-0.25 0.65 -0.72" diffuse="0.14 0.16 0.18" specular="0.18 0.18 0.20" castshadow="false"/>
    <geom name="floor" type="plane" pos="-0.6 0 0" size="8 8 0.1" material="mat_floor"/>
{chr(10).join(markers)}
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""


def _add_robot_strips_and_tendons(
    scene: mujoco.MjvScene,
    x_row: np.ndarray,       # (610, 7)
    y_offset: float,
    strip_nodes: np.ndarray, # (40, 17)
    transforms: np.ndarray,  # (40, 16, 2, 7)
    anchors: np.ndarray,     # (4, 3)
    guides: np.ndarray,      # (4, 3)
) -> None:
    """Append 40 real-time deforming flat steel ribbons + 20 tendon cords to MuJoCo scene."""
    R_plates = [_quat_xyzw_to_mat(x_row[p, 3:7]) for p in range(10)]
    p_plates = x_row[:10, :3].astype(np.float64).copy()
    p_plates[:, 1] += y_offset

    # 1. 40 pre-curved spring-steel strips (16 flat ribbon boxes each)
    for sid in range(40):
        seg = sid // 8
        rgba = SEG_STEEL_RGBA[seg]
        nodes = strip_nodes[sid]
        pts = x_row[nodes, :3].astype(np.float64).copy()
        pts[:, 1] += y_offset
        # Exact plate-clamp endpoints
        p_front, p_back = int(nodes[0]), int(nodes[-1])
        pts[0] = p_plates[p_front] + R_plates[p_front] @ transforms[sid, 0, 0, :3]
        pts[-1] = p_plates[p_back] + R_plates[p_back] @ transforms[sid, -1, 1, :3]

        seg_center = 0.5 * (p_plates[p_front] + p_plates[p_back])
        for e in range(16):
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
            # Project radial outward orthogonal to strip tangent ex
            ez = rad - np.dot(rad, ex) * ex
            n_ez = float(np.linalg.norm(ez))
            if n_ez < 1e-6:
                ez = np.array([0.0, 0.0, 1.0])
            else:
                ez /= n_ez
            ey = np.cross(ez, ex)
            ey /= max(float(np.linalg.norm(ey)), 1e-9)
            ez = np.cross(ex, ey)
            rot = np.column_stack((ex, ey, ez)).reshape(9)

            # Half-extents: seamless 20 mm wide cold-rolled spring steel with gleaming metallic highlights
            size = np.array([0.58 * length, 0.0100, 0.00055], dtype=np.float64)
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                int(mujoco.mjtGeom.mjGEOM_BOX),
                size, mid, rot, rgba,
            )
            g.matid = -1
            g.shininess = 0.94
            g.specular = 1.0
            g.reflectance = 0.24
            scene.ngeom += 1

    # 2. 20 tendon cords (4 per segment)
    for seg in range(5):
        pa, pb = 2 * seg, 2 * seg + 1
        for k in range(4):
            if scene.ngeom >= scene.maxgeom:
                return
            pt_a = p_plates[pa] + R_plates[pa] @ anchors[k]
            pt_b = p_plates[pb] + R_plates[pb] @ guides[k]
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                np.zeros(3), np.zeros(3), np.eye(3).reshape(9), TENDON_RGBA,
            )
            mujoco.mjv_connector(
                scene.geoms[scene.ngeom],
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                0.0014, pt_a, pt_b,
            )
            scene.ngeom += 1


def _project_world_to_pixel(
    world_pt: np.ndarray,
    cam: mujoco.MjvCamera,
    fovy_deg: float,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    """Project a 3D world point to 2D image pixels matching MuJoCo's free camera."""
    az = np.radians(cam.azimuth)
    el = np.radians(cam.elevation)
    # MuJoCo free camera: offset from lookat to camera eye
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = cam.lookat - cam.distance * forward
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)

    rel = world_pt - eye
    z_cam = float(np.dot(rel, forward))
    if z_cam <= 0.05:
        return None
    x_cam = float(np.dot(rel, right))
    y_cam = float(np.dot(rel, up))
    f = 0.5 * height / np.tan(0.5 * np.radians(fovy_deg))
    u = int(round(0.5 * width + f * x_cam / z_cam))
    v = int(round(0.5 * height - f * y_cam / z_cam))
    return u, v


def _draw_lane_tags(
    img: np.ndarray,
    tags: list[tuple[str, tuple[int, int] | None, tuple[int, int, int]]],
) -> np.ndarray:
    """Draw pure borderless, background-free, dot-free floating typography beside each robot."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, 'RGBA')
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 26)
    except Exception:
        font = ImageFont.load_default()

    W, H = im.size
    placed: list[tuple[int, int, int, int, str]] = []
    for text, uv, _accent in tags:
        if uv is None:
            continue
        u, v = uv
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x0 = max(18, min(W - tw - 18, u - tw // 2))
        y0 = max(18, min(H - th - 18, v - th // 2))
        placed.append((x0, y0, tw, th, text))

    # Gentle anti-overlap pass so floating text never collides
    for i in range(len(placed)):
        xi, yi, bwi, bhi, ti = placed[i]
        for j in range(i):
            xj, yj, bwj, bhj, _ = placed[j]
            if xi < xj + bwj + 10 and xi + bwi + 10 > xj:
                if abs(yi - yj) < max(bhi, bhj) + 8:
                    yi = max(18, yj - bhi - 10)
        placed[i] = (xi, yi, bwi, bhi, ti)

    for x0, y0, _tw, _th, text in placed:
        # Pure borderless dark typography with subtle light drop lift (zero box/border/dot/bg)
        draw.text((x0 + 1, y0 + 2), text, fill=(255, 255, 255, 175), font=font)
        draw.text((x0, y0), text, fill=(18, 28, 44, 255), font=font)
    return np.asarray(im)


def render_mujoco_video(
    traj_npz: Path,
    out_mp4: Path,
    lanes: tuple[int, ...] = (0, 1, 2),
    labels: tuple[str, ...] = ('纯蛇形', '纯蠕虫', '蛇 + 蠕虫'),
    width: int = 1280,
    height: int = 720,
    show_tags: bool = True,
    preview_png: Path | None = None,
    dynamic_mode_lanes: tuple[int, ...] = (),
    dynamic_label_fn=None,
    lane_ys_override: tuple[float, ...] | None = None,
    cam_distance: float = 1.92,
    cam_azimuth: float = 112.0,
    cam_elevation: float = -28.0,
    preview_frame: int | None = None,
) -> Path:
    data = np.load(traj_npz)
    x_traj = data['x']              # (T, N, 610, 7)
    mode_traj = data['mode']        # (T, N)
    strip_nodes = data['strip_nodes']
    transforms = data['transforms']
    mounts = data['mounts']
    anchors = data['anchors']
    guides = data['guides']
    fps = int(data['fps'])

    n_rob = len(lanes)
    if lane_ys_override is not None:
        lane_ys = list(lane_ys_override)
    else:
        lane_ys = [0.48, 0.0, -0.48] if n_rob == 3 else [0.46 * (0.5 * (n_rob - 1) - i) for i in range(n_rob)]
    start_head_x = float(x_traj[0, lanes[0], 9, 0])

    xml = _build_mjcf(n_rob, mounts, start_head_x, lane_ys)
    model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Precompute mocap body ids
    mocap_ids = [
        [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'r{r}_p{p}')] for p in range(10)]
        for r in range(n_rob)
    ]

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = cam_azimuth
    cam.elevation = cam_elevation
    cam.distance = cam_distance
    fovy_deg = float(model.vis.global_.fovy)

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    mode_zh = ('蛇形', '蠕虫', '融合')
    accents = [(96, 165, 250), (52, 211, 153), (251, 191, 36)]

    T = x_traj.shape[0]
    p_fi = (T - 1) if preview_frame is None else min(T - 1, max(0, int(preview_frame)))
    for fi in range(T):
        cx_sum = 0.0
        cy_sum = 0.0
        head_pts = []
        tag_texts = []
        for r, (ln, y_off, base_lbl) in enumerate(zip(lanes, lane_ys, labels)):
            xr = x_traj[fi, ln]
            for p in range(10):
                mid_id = mocap_ids[r][p]
                pos = xr[p, :3].astype(np.float64).copy()
                pos[1] += y_off
                mj_data.mocap_pos[mid_id] = pos
                mj_data.mocap_quat[mid_id] = _quat_xyzw_to_wxyz(xr[p, 3:7])
            p_cur = xr[:10, :3].astype(np.float64)
            cx_sum += float(p_cur[:, 0].mean())
            cy_sum += float(p_cur[:, 1].mean()) + y_off
            # Anchor floating text cleanly above & slightly ahead of each robot's head (zero overlap with strips)
            anchor_3d = p_cur[9].copy()
            anchor_3d[0] -= 0.16
            anchor_3d[1] += y_off
            anchor_3d[2] = 0.285
            head_pts.append(anchor_3d)
            if dynamic_label_fn is not None:
                tag_texts.append(dynamic_label_fn(fi, r, ln, base_lbl, int(mode_traj[fi, ln]), fps))
            elif ln in dynamic_mode_lanes:
                tag_texts.append(f'{base_lbl} · {mode_zh[int(mode_traj[fi, ln])]}')
            else:
                tag_texts.append(base_lbl)

        mujoco.mj_forward(model, mj_data)
        cam.lookat[:] = [cx_sum / n_rob - 0.06, cy_sum / n_rob, 0.10]
        renderer.update_scene(mj_data, camera=cam)

        for r, (ln, y_off) in enumerate(zip(lanes, lane_ys)):
            _add_robot_strips_and_tendons(
                renderer.scene, x_traj[fi, ln], y_off,
                strip_nodes, transforms, anchors, guides,
            )

        rgb = renderer.render()
        if show_tags:
            tags = [
                (txt, _project_world_to_pixel(pt, cam, fovy_deg, width, height), accents[r % len(accents)])
                for r, (txt, pt) in enumerate(zip(tag_texts, head_pts))
            ]
            rgb = _draw_lane_tags(rgb, tags)
        if preview_png is not None and fi == p_fi:
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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description='Render 3D MuJoCo studio video from real FP64 RL trajectory')
    p.add_argument('traj_npz', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preview-png', type=Path, default=None)
    p.add_argument('--no-tags', action='store_true', help='Export completely raw 3D video without floating lane tags')
    args = p.parse_args(argv)
    out = render_mujoco_video(args.traj_npz, args.out, show_tags=not args.no_tags, preview_png=args.preview_png)
    print(f'Rendered MuJoCo 3D video: {out} ({out.stat().st_size / 1024:.1f} KiB)')


if __name__ == '__main__':
    main()

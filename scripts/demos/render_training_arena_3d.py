"""Render a unified 3D MuJoCo panoramic video of the multi-robot parallel RL training arena (12 live DR bays)."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

import sys
_SRC = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from serpworm.physics.pipe_geometry import build_pipe_mujoco_geoms
from serpworm.probes.render_mujoco import (
    MESH_DIR,
    SEG_STEEL_RGBA,
    TENDON_RGBA,
    _quat_xyzw_to_mat,
    _quat_xyzw_to_wxyz,
)


def _make_soil_tex(out_dir: Path) -> Path:
    rng = np.random.default_rng(17)
    H, W = 512, 512
    x = np.linspace(0, 1, W)[None, :]
    y = np.linspace(0, 1, H)[:, None]
    ripple = 7.0 * np.sin(2.0 * np.pi * (8.0 * x + 0.5 * y))
    grain = rng.normal(0.0, 9.5, size=(H, W))
    r = np.clip(182 + ripple + grain, 125, 222).astype(np.uint8)
    g = np.clip(144 + 0.85 * ripple + 0.85 * grain, 96, 184).astype(np.uint8)
    b = np.clip(98 + 0.65 * ripple + 0.70 * grain, 62, 136).astype(np.uint8)
    p = out_dir / '_arena_soil_tex.png'
    Image.fromarray(np.stack([r, g, b], axis=-1)).save(p)
    return p


def _add_robot_strips_shifted(
    scene: mujoco.MjvScene,
    x_row: np.ndarray,       # (D, 7)
    ox: float,
    oy: float,
    strip_nodes: np.ndarray, # (40, 17)
    transforms: np.ndarray,  # (40, 16, 2, 7)
    anchors: np.ndarray,     # (4, 3)
    guides: np.ndarray,      # (4, 3)
    stride: int = 2,
) -> None:
    """Append 40 gleaming flat spring-steel strips (stride-2 smooth boxes) + 20 tendons shifted by (ox, oy)."""
    R_plates = [_quat_xyzw_to_mat(x_row[p, 3:7]) for p in range(10)]
    p_plates = x_row[:10, :3].astype(np.float64).copy()
    p_plates[:, 0] += ox
    p_plates[:, 1] += oy

    for sid in range(40):
        seg = sid // 8
        rgba = SEG_STEEL_RGBA[seg]
        nodes = strip_nodes[sid]
        pts = x_row[nodes, :3].astype(np.float64).copy()
        pts[:, 0] += ox
        pts[:, 1] += oy
        p_front, p_back = int(nodes[0]), int(nodes[-1])
        pts[0] = p_plates[p_front] + R_plates[p_front] @ transforms[sid, 0, 0, :3]
        pts[-1] = p_plates[p_back] + R_plates[p_back] @ transforms[sid, -1, 1, :3]
        seg_center = 0.5 * (p_plates[p_front] + p_plates[p_back])

        for e in range(0, 16, stride):
            if scene.ngeom >= scene.maxgeom:
                return
            a, b = pts[e], pts[min(16, e + stride)]
            diff = b - a
            length = float(np.linalg.norm(diff))
            if length < 1e-6:
                continue
            mid = 0.5 * (a + b)
            ex = diff / length
            rad = mid - seg_center
            ez = rad - np.dot(rad, ex) * ex
            n_ez = float(np.linalg.norm(ez))
            ez = np.array([0.0, 0.0, 1.0]) if n_ez < 1e-6 else (ez / n_ez)
            ey = np.cross(ez, ex)
            ey /= max(float(np.linalg.norm(ey)), 1e-9)
            ez = np.cross(ex, ey)
            rot = np.column_stack((ex, ey, ez)).reshape(9)

            size = np.array([0.56 * length, 0.0100, 0.0006], dtype=np.float64)
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_BOX), size, mid, rot, rgba)
            g.matid = -1
            g.shininess = 0.94
            g.specular = 1.0
            g.reflectance = 0.24
            scene.ngeom += 1

    for seg in range(5):
        pa, pb = 2 * seg, 2 * seg + 1
        for k in range(4):
            if scene.ngeom >= scene.maxgeom:
                return
            pt_a = p_plates[pa] + R_plates[pa] @ anchors[k]
            pt_b = p_plates[pb] + R_plates[pb] @ guides[k]
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                np.zeros(3), np.zeros(3), np.eye(3).reshape(9), TENDON_RGBA,
            )
            mujoco.mjv_connector(
                scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                0.0014, pt_a, pt_b,
            )
            scene.ngeom += 1


def render_arena_video(
    npz_path: Path,
    out_mp4: Path,
    out_gif: Path,
    out_preview: Path,
    width: int = 1280,
    height: int = 720,
) -> None:
    d = np.load(npz_path)
    x_all = d['x']                         # [T, B, D, 7]
    pipe_specs = d['pipe_specs'].copy()    # [B, 12]
    terrain_id = d['terrain_id']           # [B]
    post_pipe_soil = d['post_pipe_soil']   # [B]
    sn, tr, mn, an, gu = d['strip_nodes'], d['transforms'], d['mounts'], d['anchors'], d['guides']
    fps = int(d['fps'])
    T, B = x_all.shape[0], x_all.shape[1]

    # Spacious 3-row x 4-col training factory grid (3.45 m x 2.85 m per bay -> zero overlap!)
    n_cols = 4
    n_rows = int(math.ceil(B / n_cols))
    dx_bay, dy_bay = 3.50, 2.85
    offsets = []
    for i in range(B):
        r, c = i // n_cols, i % n_cols
        ox = (c - 0.5 * (n_cols - 1)) * dx_bay + 1.05
        oy = (r - 0.5 * (n_rows - 1)) * dy_bay
        # Center left/right turning pipes nicely inside each bay
        pt = int(round(pipe_specs[i, 0]))
        sgn = 1.0 if pipe_specs[i, 7] >= 0.0 else -1.0
        if pt in (2, 3, 4):
            oy -= 0.42 * sgn
        offsets.append((ox, oy))

    mesh_dir = MESH_DIR.resolve().as_posix()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='serpworm_arena_') as tmpdir:
        soil_tex = _make_soil_tex(Path(tmpdir))
        soil_posix = soil_tex.resolve().as_posix()

        bodies = []
        pipe_xmls = []
        pad_xmls = []

        for i in range(B):
            ox, oy = offsets[i]
            # Subtle bay frame lines on the factory floor
            cx_bay = ox - 1.15
            cy_bay = oy + (0.42 * (1.0 if pipe_specs[i, 7] >= 0.0 else -1.0) if int(round(pipe_specs[i, 0])) in (2, 3, 4) else 0.0)
            for sgn_y in (-1.0, 1.0):
                pad_xmls.append(
                    f'    <geom type="box" pos="{cx_bay:.3f} {cy_bay + sgn_y*1.28:.3f} 0.0005" '
                    f'size="1.58 0.008 0.0005" rgba="0.46 0.52 0.60 0.45" contype="0" conaffinity="0"/>'
                )
            for sgn_x in (-1.0, 1.0):
                pad_xmls.append(
                    f'    <geom type="box" pos="{cx_bay + sgn_x*1.58:.3f} {cy_bay:.3f} 0.0005" '
                    f'size="0.008 1.28 0.0005" rgba="0.46 0.52 0.60 0.45" contype="0" conaffinity="0"/>'
                )

            # Soft-soil or post-pipe sand bed tiles
            pt = int(round(pipe_specs[i, 0]))
            if int(terrain_id[i]) == 2 and pt == 0:
                pad_xmls.append(
                    f'    <geom type="box" pos="{cx_bay:.3f} {cy_bay:.3f} 0.0012" '
                    f'size="1.48 1.12 0.0012" material="mat_soil" contype="0" conaffinity="0"/>'
                )
            elif bool(post_pipe_soil[i]) and pt == 1:
                x_exit_w = ox + float(pipe_specs[i, 4])
                pad_xmls.append(
                    f'    <geom type="box" pos="{x_exit_w - 0.55:.3f} {oy + float(pipe_specs[i, 1]):.3f} 0.0012" '
                    f'size="0.58 0.72 0.0012" material="mat_soil" contype="0" conaffinity="0"/>'
                )

            # Shifted transparent acrylic pipe geoms
            if pt > 0 and pipe_specs[i, 10] > 0.0:
                pc_s = pipe_specs[i].copy()
                pc_s[1] += oy
                pc_s[3] += ox
                pc_s[4] += ox
                pipe_xmls.append(build_pipe_mujoco_geoms(pc_s))

            # Robot mocap plate bodies
            for p in range(10):
                mesh_name = 'plate_front' if p % 2 == 0 else 'plate_back'
                mat_name = 'mat_plate_front' if p % 2 == 0 else 'mat_plate_back'
                w_geoms = []
                for s in range(2):
                    mx, my, mz = mn[p, s]
                    w_geoms.append(f'<geom type="mesh" mesh="wheel_mesh" pos="{mx:.5f} {my:.5f} {mz:.5f}" material="mat_wheel" contype="0" conaffinity="0"/>')
                servo_g = (
                    '<geom type="box" pos="0.018 0.0 0.0" size="0.016 0.036 0.0085" material="mat_servo_purple" contype="0" conaffinity="0"/>'
                    if p % 2 == 1 else
                    '<geom type="box" pos="0.004 0 -0.003" size="0.015 0.0098 0.0088" material="mat_servo_purple" contype="0" conaffinity="0"/>'
                )
                bodies.append(
                    f'    <body name="r{i}_p{p}" mocap="true" pos="{ox:.3f} {oy:.3f} 0.13">\n'
                    f'      <geom type="mesh" mesh="{mesh_name}" material="{mat_name}" contype="0" conaffinity="0"/>\n'
                    f'      {servo_g}\n'
                    + '\n'.join(f'      {g}' for g in w_geoms) + '\n'
                    f'    </body>'
                )

        xml = f"""<mujoco model="serpworm_training_arena_12">
  <compiler angle="radian" meshdir="{mesh_dir}" autolimits="true"/>
  <visual>
    <global offwidth="{width}" offheight="{height}" fovy="38"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.1" zfar="45.0" shadowclip="8.0"/>
    <headlight ambient="0.24 0.25 0.27" diffuse="0.08 0.08 0.08" specular="0.0 0.0 0.0"/>
    <rgba haze="0.79 0.82 0.86 1"/>
  </visual>
  <asset>
    <mesh name="plate_front" file="front2_Link.STL"/>
    <mesh name="plate_back"  file="back2_Link.STL"/>
    <mesh name="wheel_mesh"  file="w2-1_Link.STL"/>
    <texture name="skybox"   type="skybox" builtin="gradient" rgb1="0.76 0.80 0.85" rgb2="0.87 0.90 0.93" width="512" height="512"/>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.71 0.74 0.78" rgb2="0.67 0.70 0.74" width="512" height="512"/>
    <texture name="soil_tex" type="2d" file="{soil_posix}"/>
    <material name="mat_floor"        texture="grid_tex" texrepeat="2 2" texuniform="true" reflectance="0.0" shininess="0.05" specular="0.03"/>
    <material name="mat_soil"         texture="soil_tex" texrepeat="4 4" texuniform="true" reflectance="0.0" shininess="0.02" specular="0.02"/>
    <material name="mat_acrylic_tube" rgba="0.72 0.86 0.96 0.17" specular="0.96" shininess="0.95" reflectance="0.06"/>
    <material name="mat_acrylic_rim"  rgba="0.78 0.90 0.99 0.30" specular="0.98" shininess="0.96" reflectance="0.10"/>
    <material name="mat_alu_ring"     rgba="0.62 0.67 0.73 1.0"  specular="0.76" shininess="0.72" reflectance="0.14"/>
    <material name="mat_plate_front"  rgba="0.78 0.82 0.87 0.76" specular="0.85" shininess="0.80" reflectance="0.10"/>
    <material name="mat_plate_back"   rgba="0.34 0.38 0.44 1.0"  specular="0.65" shininess="0.60" reflectance="0.08"/>
    <material name="mat_servo_purple" rgba="0.44 0.23 0.64 1.0"  specular="0.82" shininess="0.78" reflectance="0.12"/>
    <material name="mat_wheel"        rgba="0.14 0.15 0.17 1.0"  specular="0.22" shininess="0.20"/>
  </asset>
  <worldbody>
    <light name="key"  directional="true" pos="0 0 8.0" dir="0.32 0.38 -0.86" diffuse="0.54 0.53 0.52" specular="0.90 0.90 0.88" castshadow="true"/>
    <light name="rim"  directional="true" pos="0 0 6.5" dir="-0.52 -0.32 -0.79" diffuse="0.18 0.20 0.23" specular="0.82 0.84 0.88" castshadow="false"/>
    <light name="fill" directional="true" pos="0 0 6.0" dir="-0.22 0.62 -0.75" diffuse="0.16 0.18 0.21" specular="0.18 0.18 0.20" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 0" size="22 22 0.1" material="mat_floor"/>
{chr(10).join(pad_xmls)}
{chr(10).join(bodies)}
{chr(10).join(pipe_xmls)}
  </worldbody>
</mujoco>"""

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        # Ensure scene has generous geom capacity for 12 robots + pipes
        if renderer.scene.maxgeom < 14000:
            renderer._scene = mujoco.MjvScene(model, maxgeom=16000)

        mocap_ids = [
            [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'r{i}_p{p}')] for p in range(10)]
            for i in range(B)
        ]

        try:
            font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 26)
        except Exception:
            font = ImageFont.load_default()

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [-0.25, 0.05, 0.14]
        cam.elevation = -42.0
        cam.distance = 13.6

        proc = subprocess.Popen([
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'fast',
            str(out_mp4),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        buf = np.empty((height, width, 3), dtype=np.uint8)
        preview_fi = int(T * 0.55)

        for fi in range(T):
            u = fi / max(1, T - 1)
            # Gentle panoramic crane orbit showing depth across all 12 training bays
            cam.azimuth = 104.0 + 32.0 * (0.5 - 0.5 * math.cos(math.pi * u))
            cam.distance = 13.6 - 1.2 * math.sin(math.pi * u)

            for i in range(B):
                ox, oy = offsets[i]
                xr = x_all[fi, i]
                mids = mocap_ids[i]
                for p in range(10):
                    pos = xr[p, :3].astype(np.float64)
                    data.mocap_pos[mids[p]] = (pos[0] + ox, pos[1] + oy, pos[2])
                    data.mocap_quat[mids[p]] = _quat_xyzw_to_wxyz(xr[p, 3:7])

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            for i in range(B):
                ox, oy = offsets[i]
                _add_robot_strips_shifted(renderer.scene, x_all[fi, i], ox, oy, sn, tr, an, gu, stride=2)

            rgb = renderer.render(out=buf)
            im = Image.fromarray(rgb)
            draw = ImageDraw.Draw(im, 'RGBA')
            title = '并行训练场景实拍'
            draw.text((25, 19), title, fill=(255, 255, 255, 210), font=font)
            draw.text((24, 18), title, fill=(18, 28, 46, 255), font=font)
            arr = np.asarray(im)
            if fi == preview_fi:
                im.save(out_preview)
            proc.stdin.write(arr.tobytes())

        proc.stdin.close()
        err = proc.stderr.read().decode('utf-8', errors='replace')
        rc = proc.wait()
        renderer.close()
        if rc != 0:
            raise RuntimeError(f'ffmpeg failed ({rc}): {err[-400:]}')

    subprocess.run([
        'ffmpeg', '-y', '-i', str(out_mp4),
        '-vf', 'setpts=0.5*PTS,fps=12,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3',
        str(out_gif),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'Rendered {out_mp4} ({out_mp4.stat().st_size/1024/1024:.2f} MiB), {out_gif} ({out_gif.stat().st_size/1024/1024:.2f} MiB)')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('npz', type=Path)
    p.add_argument('--out-prefix', type=Path, required=True)
    p.add_argument('--artifact-dir', type=Path, default=None)
    args = p.parse_args()
    mp4 = Path(f'{args.out_prefix}.mp4')
    gif = Path(f'{args.out_prefix}.gif')
    png = Path(f'{args.out_prefix}_preview.png')
    render_arena_video(args.npz, mp4, gif, png)
    if args.artifact_dir is not None:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        for src in (mp4, gif, png):
            shutil.copy2(src, args.artifact_dir / src.name)


if __name__ == '__main__':
    main()

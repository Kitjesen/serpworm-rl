"""Render a crisp 3x2 6-panel 3D MuJoCo preview of all 4 new v8 pipes + gravel & grass terrains."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

import sys
_SRC = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from serpworm.physics.pipe_geometry import (
    build_pipe_mujoco_geoms,
    make_off_pipe,
    make_taper_pipe,
    make_triple_wave_pipe,
    make_uturn_pipe,
    make_z_duct,
)
from serpworm.probes.render_mujoco import MESH_DIR


def _make_textures(out_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    H, W = 512, 512
    # Crushed gravel texture (multi-tone gray/ochre pebbles)
    base = rng.uniform(118, 152, (H, W, 1))
    peb = np.zeros((H, W, 3), dtype=np.float32) + base
    peb[..., 0] += rng.normal(3, 14, (H, W))
    peb[..., 1] += rng.normal(1, 13, (H, W))
    peb[..., 2] += rng.normal(-3, 13, (H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(900):
        cx, cy = rng.integers(0, W), rng.integers(0, H)
        rr = rng.integers(3, 9)
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr ** 2
        col = rng.uniform(90, 190)
        peb[m, 0] = col + rng.uniform(-10, 12)
        peb[m, 1] = col + rng.uniform(-10, 10)
        peb[m, 2] = col + rng.uniform(-14, 6)
    grv_p = out_dir / '_gravel_tex.png'
    Image.fromarray(np.clip(peb, 55, 225).astype(np.uint8)).save(grv_p)

    # Natural grass meadow texture
    x = np.linspace(0, 1, W)[None, :]
    y = np.linspace(0, 1, H)[:, None]
    patch = 12.0 * np.sin(2 * np.pi * (3 * x + 2 * y)) * np.cos(2 * np.pi * (2 * x - 3 * y))
    blade = rng.normal(0, 13, (H, W))
    r = np.clip(78 + 0.7 * patch + 0.7 * blade, 42, 132).astype(np.uint8)
    g = np.clip(126 + 1.1 * patch + 1.1 * blade, 78, 185).astype(np.uint8)
    b = np.clip(62 + 0.5 * patch + 0.5 * blade, 35, 108).astype(np.uint8)
    grs_p = out_dir / '_grass_tex.png'
    Image.fromarray(np.stack([r, g, b], axis=-1)).save(grs_p)
    return grv_p, grs_p


def render_v8_preview(out_png: Path, artifact_dir: Path | None = None) -> None:
    pw, ph = 520, 360
    panels = [
        ('180° U 型掉头弯管', make_uturn_pipe(x_entry=-0.95, x_bend=-1.35, R_bend=0.54, r_in=0.128, bend_sgn=+1.0, L_exit=1.05), 'floor',
         dict(lookat=[-1.15, 0.54, 0.12], azimuth=118.0, elevation=-46.0, distance=3.05)),
        ('锥形变径缩口管', make_taper_pipe(x_entry=-0.90, x_taper_start=-1.30, L_taper=0.70, r_in=0.156, r_out=0.110, L_exit=0.95), 'floor',
         dict(lookat=[-1.88, 0.00, 0.12], azimuth=114.0, elevation=-30.0, distance=2.65)),
        ('Z 型双直角方管', make_z_duct(x_entry=-0.95, x_corner=-1.40, L_mid=0.72, width=0.268, height=0.250, bend_sgn=+1.0, L_exit=0.95, r_fillet=0.008), 'floor',
         dict(lookat=[-1.45, 0.36, 0.12], azimuth=122.0, elevation=-46.0, distance=3.10)),
        ('三连弯蛇形长管', make_triple_wave_pipe(x_entry=-0.90, x_bend=-1.20, R_bend=0.54, bend_angle_deg=58.0, r_in=0.126, bend_sgn=+1.0, L_exit=0.85), 'floor',
         dict(lookat=[-2.35, 0.18, 0.12], azimuth=108.0, elevation=-48.0, distance=3.85)),
        ('石子地微起伏', make_off_pipe(), 'gravel',
         dict(lookat=[-1.20, 0.00, 0.06], azimuth=124.0, elevation=-32.0, distance=2.10)),
        ('草地波浪坡面', make_off_pipe(), 'grass',
         dict(lookat=[-1.20, 0.00, 0.06], azimuth=124.0, elevation=-32.0, distance=2.10)),
    ]

    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 21)
    except Exception:
        font = ImageFont.load_default()

    canvas = Image.new('RGB', (3 * pw, 2 * ph))
    with tempfile.TemporaryDirectory(prefix='v8_prev_') as tmpdir:
        grv_p, grs_p = _make_textures(Path(tmpdir))
        for idx, (title, pc, mode, cam_cfg) in enumerate(panels):
            extra = ''
            if mode == 'gravel':
                extra = '    <geom type="box" pos="-1.2 0 0.004" size="1.35 1.15 0.004" material="mat_gravel" contype="0" conaffinity="0"/>\n'
                # Visual scattered low smooth pebbles (zero stairs!)
                rng = np.random.default_rng(11)
                for _ in range(95):
                    px, py = rng.uniform(-2.3, -0.1), rng.uniform(-0.9, 0.9)
                    rx, ry, rz = rng.uniform(0.012, 0.028), rng.uniform(0.010, 0.024), rng.uniform(0.004, 0.010)
                    col = rng.uniform(0.45, 0.72)
                    extra += f'    <geom type="ellipsoid" pos="{px:.3f} {py:.3f} {rz*0.7:.3f}" size="{rx:.3f} {ry:.3f} {rz:.3f}" rgba="{col:.2f} {col-0.02:.2f} {col-0.05:.2f} 1" contype="0" conaffinity="0"/>\n'
            elif mode == 'grass':
                extra = '    <geom type="box" pos="-1.2 0 0.004" size="1.35 1.15 0.004" material="mat_grass" contype="0" conaffinity="0"/>\n'
                # Gentle smooth turf swells (no stairs!)
                for gx, gy in [(-0.7, -0.35), (-1.4, 0.25), (-1.9, -0.20), (-1.1, 0.45)]:
                    extra += f'    <geom type="ellipsoid" pos="{gx:.2f} {gy:.2f} -0.028" size="0.45 0.42 0.044" material="mat_grass" contype="0" conaffinity="0"/>\n'
            xml = f"""<mujoco model="v8_p{idx}">
  <compiler angle="radian" meshdir="{MESH_DIR.resolve().as_posix()}" autolimits="true"/>
  <visual>
    <global offwidth="{pw}" offheight="{ph}" fovy="38"/>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.05" zfar="25.0" shadowclip="4.0"/>
    <headlight ambient="0.22 0.23 0.25" diffuse="0.07 0.07 0.07" specular="0 0 0"/>
    <rgba haze="0.78 0.81 0.85 1"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.76 0.80 0.85" rgb2="0.86 0.89 0.92" width="512" height="512"/>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.70 0.73 0.77" rgb2="0.66 0.69 0.73" width="512" height="512"/>
    <texture name="gravel_tex" type="2d" file="{grv_p.resolve().as_posix()}"/>
    <texture name="grass_tex" type="2d" file="{grs_p.resolve().as_posix()}"/>
    <material name="mat_floor"        texture="grid_tex" texrepeat="3 3" texuniform="true" reflectance="0.0" shininess="0.05" specular="0.03"/>
    <material name="mat_gravel"       texture="gravel_tex" texrepeat="4 4" texuniform="true" reflectance="0.0" shininess="0.04" specular="0.03"/>
    <material name="mat_grass"        texture="grass_tex" texrepeat="4 4" texuniform="true" reflectance="0.0" shininess="0.03" specular="0.02"/>
    <material name="mat_acrylic_tube" rgba="0.72 0.86 0.96 0.18" specular="0.96" shininess="0.95" reflectance="0.06"/>
    <material name="mat_acrylic_rim"  rgba="0.78 0.90 0.99 0.30" specular="0.98" shininess="0.96" reflectance="0.10"/>
    <material name="mat_alu_ring"     rgba="0.64 0.68 0.74 1.0"  specular="0.76" shininess="0.72" reflectance="0.14"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0 0 4.5" dir="0.35 0.42 -0.84" diffuse="0.54 0.53 0.52" specular="0.90 0.90 0.88" castshadow="true"/>
    <light name="fill" directional="true" pos="0 0 3.0" dir="-0.35 0.55 -0.75" diffuse="0.18 0.20 0.22" specular="0.18 0.18 0.20" castshadow="false"/>
    <geom name="floor" type="plane" pos="-1.6 0.2 0" size="10 10 0.1" material="mat_floor"/>
{extra}{build_pipe_mujoco_geoms(pc)}
  </worldbody>
</mujoco>"""
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            rend = mujoco.Renderer(model, height=ph, width=pw)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = cam_cfg['lookat']; cam.azimuth = cam_cfg['azimuth']
            cam.elevation = cam_cfg['elevation']; cam.distance = cam_cfg['distance']
            rend.update_scene(data, camera=cam)
            im = Image.fromarray(rend.render())
            rend.close()
            draw = ImageDraw.Draw(im, 'RGBA')
            draw.text((17, 13), title, fill=(255, 255, 255, 210), font=font)
            draw.text((16, 12), title, fill=(18, 28, 46, 255), font=font)
            canvas.paste(im, ((idx % 3) * pw, (idx // 3) * ph))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_png, artifact_dir / out_png.name)
    print(f'Saved {out_png} ({out_png.stat().st_size/1024:.1f} KiB)')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'runs/demos/v8_new_pipes_and_terrains_preview.png')
    p.add_argument('--artifact-dir', type=Path, default=None)
    a = p.parse_args()
    render_v8_preview(a.out, a.artifact_dir)

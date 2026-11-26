#!/usr/bin/env python3
"""Nature Communications Storyteller Flagship 5-Panel Figure (v7).

Synthesizes living natural-history illustration, explicit literature-anchored
paradigm contrast, and full technical architecture into a single iconic
Nature Communications double-column spread (183.0 x 175.3 mm, 2.0 px = 1 pt):

  Panel a — Living Ecological Cutaway & Bio-Hybrid Embodiment:
    A museum-grade Nature/Scientific American cutaway illustration depicting a
    slate-teal snake gliding in S-wave undulation above ground directly over a
    terracotta earthworm peristaltically burrowing through a narrow underground
    tunnel, overlaid with vector microscope zoom loupes (directional ventral
    scales & hydrostatic metameric muscle rings) mapping 1-to-1 onto our real
    3D CAD 5-segment continuum robot and embodied q_err mechanoreception.

  Panel b — Paradigm Contrast & Core Advantages (with explicit citations):
    Side-by-side visual contrast of (1) Classical CPG oscillators (Hirose 1993;
    Matsuoka 1985) jamming against pipe walls (12.6 N) or stalling in soft soil
    (CoT 50.8), (2) Exteroceptive vision/tactile skin + discrete FSM blinded in
    dark pipes and chattering at thresholds, vs. (3) Ours: zero vision/force
    skin, continuous single-manifold phase transition with 4 bold KPI badges
    (3.3-3.8x lower CoT, 30x lower wall force, 0.0 deg yaw drift, 42/42 pass).

  Panel c — Onboard Pure-Proprioceptive Neuromechanical Sensorimotor Loop:
    Full-width airy 4-stage backbone (Proprioception 65D -> Spinal 1D-TCN with
    o_t skip -> Cerebellar Hybrid Internal Model + sg(c_t) bypass -> Bilateral
    Z2 Symmetric Motor Policy -> 14D u_t).

  Panel d — Simulation-Only Digital Twin & Predictive Learning (68% left):
    Unified flat slate-mist zone (FP64 Continuum Twin 3-tile 3D mosaic |
    HIM-DreamWaQ beta-VAE & EMA S^31 | Privileged Critic + Min-CoT PPO).

  Panel e — Emergent Cross-Media Phase Transition (30% right):
    Directly beneath 14D u_t: Open Hard Serpenoid (CoT 1.06), Ø236 Pipe
    Peristaltic (CoT 1.22), and Soft Soil Hybrid (CoT 8.29) with physical
    medium silhouettes and dual-channel traveling waves.

Outputs:
  - fig_nc_algorithm_architecture.png  (600 DPI)
  - fig_nc_algorithm_architecture.pdf  (single-page editable vector PDF)
  - fig_nc_algorithm_architecture.svg  (self-contained editable vector SVG)
"""
from __future__ import annotations

import base64
import io
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'runs' / 'paper_figures'
_ARTIFACT_ENV = os.environ.get('SERPWORM_ARTIFACT_DIR')
ARTIFACT_DIR: Path | None = Path(_ARTIFACT_ENV) if _ARTIFACT_ENV else None

HABITAT_RAW = OUT_DIR / '_bio_habitat_cutaway.jpg'
ROBOT_PNG = OUT_DIR / '_robot_5seg_cutout.png'
TWIN_PIPE_SOIL_PNG = ROOT / 'runs' / 'demos' / 'v6b_autonomous_pipe_preview.png'
TWIN_CURVED_PNG = ROOT / 'runs' / 'demos' / 'v71_r170_all4_pipes_3d_preview.png'
TWIN_SQUEEZE_PNG = ROOT / 'runs' / 'pipe_showcase' / 'pipe_squeeze_dualview_preview.png'

# ViewBox 1040 x 996 -> exact 183.00 mm x 175.26 mm (2.0 SVG units = 1.0 print pt)
W = 1040.0
H = 996.0
WIDTH_MM = 183.0
HEIGHT_MM = WIDTH_MM * (H / W)


def _white_to_rgba_b64(path: Path, lo: float = 12.0, hi: float = 38.0,
                       contrast: float = 1.08, sharp: float = 1.15,
                       max_w: int = 920) -> str:
    im = Image.open(path).convert('RGBA')
    arr = np.asarray(im, dtype=np.float32)
    rgb = arr[..., :3]
    dist_white = np.sqrt(np.sum((255.0 - rgb) ** 2, axis=-1))
    alpha = np.clip((dist_white - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    out = arr.copy()
    out[..., 3] = alpha * 255.0
    pil = Image.fromarray(out.astype(np.uint8), mode='RGBA')
    r, g, b, a = pil.split()
    rgb_img = Image.merge('RGB', (r, g, b))
    rgb_img = ImageEnhance.Contrast(rgb_img).enhance(contrast)
    rgb_img = ImageEnhance.Sharpness(rgb_img).enhance(sharp)
    r2, g2, b2 = rgb_img.split()
    pil = Image.merge('RGBA', (r2, g2, b2, a))
    mask = a.point(lambda p: 255 if p > 16 else 0)
    bbox = mask.getbbox()
    if bbox:
        pil = pil.crop(bbox)
    if pil.width > max_w:
        nh = int(round(pil.height * (max_w / pil.width)))
        pil = pil.resize((max_w, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _crop_b64(path: Path, frac_box: tuple[float, float, float, float],
              contrast: float = 1.05, color: float = 1.00, sharp: float = 1.10,
              max_w: int = 1100) -> str:
    im = Image.open(path).convert('RGB')
    w, h = im.size
    box = (int(frac_box[0] * w), int(frac_box[1] * h),
           int(frac_box[2] * w), int(frac_box[3] * h))
    crop = im.crop(box)
    crop = ImageEnhance.Contrast(crop).enhance(contrast)
    crop = ImageEnhance.Color(crop).enhance(color)
    crop = ImageEnhance.Sharpness(crop).enhance(sharp)
    if crop.width > max_w:
        nh = int(round(crop.height * (max_w / crop.width)))
        crop = crop.resize((max_w, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _wave_path(x0: float, y_mid: float, width: float, amp: float,
               fn, n: int = 80, close_area: bool = False, y_base: float = 0.0) -> str:
    pts = []
    for i in range(n):
        t = i / (n - 1)
        x = x0 + t * width
        y = y_mid - amp * fn(t)
        pts.append((x, y))
    d = f'M {pts[0][0]:.2f} {pts[0][1]:.2f} ' + ' '.join(
        f'L {x:.2f} {y:.2f}' for x, y in pts[1:]
    )
    if close_area:
        d += f' L {pts[-1][0]:.2f} {y_base:.2f} L {pts[0][0]:.2f} {y_base:.2f} Z'
    return d


def build_svg() -> str:
    # Crop pure white sky above snake cleanly while preserving full habitat cutaway
    habitat_uri = _crop_b64(HABITAT_RAW, (0.03, 0.25, 0.97, 0.94), contrast=1.05, color=1.04, sharp=1.10)
    robot_uri = _white_to_rgba_b64(ROBOT_PNG, lo=12.0, hi=38.0, contrast=1.08, sharp=1.16, max_w=920)
    twin_pipe_soil_uri = _crop_b64(TWIN_PIPE_SOIL_PNG, (0.19, 0.24, 0.92, 0.79), color=0.96)
    twin_curved_uri = _crop_b64(TWIN_CURVED_PNG, (0.625, 0.065, 0.855, 0.475), contrast=1.08, color=0.96)
    twin_squeeze_uri = _crop_b64(TWIN_SQUEEZE_PNG, (0.595, 0.025, 0.885, 0.465), contrast=1.06, color=1.04)

    # Restrained Nature Communications Editorial Palette (Flat Matte, Zero Shadows)
    INK = '#0F172A'
    SUB = '#475569'
    MUTED = '#64748B'
    RULE = '#DCE4EE'
    RULE_LIGHT = '#E8EEF5'
    MIST = '#F4F7FA'

    TEAL = '#186073'
    TEAL_MID = '#2A7F94'
    TEAL_SOFT = '#EDF5F8'
    TEAL_BD = '#BDD9E2'

    TERRA = '#B8522E'
    TERRA_MID = '#D46D45'
    TERRA_SOFT = '#FCF2ED'
    TERRA_BD = '#EBC5B5'

    INDIGO = '#50569C'
    INDIGO_MID = '#6E74BA'
    INDIGO_SOFT = '#F3F4FB'
    INDIGO_BD = '#CCCFE8'

    WARN = '#B44437'
    WARN_SOFT = '#FDF2F1'
    WARN_BD = '#EBC2BD'
    SAND = '#B47B2B'

    # --- Panel b: Col 3 continuous phase-transition curve (Ours) ---
    b_ours_yaw = _wave_path(668.0, 344.0, 326.0, 12.5,
                            lambda t: (1.0 - 0.88 * math.exp(-((t - 0.50) / 0.19) ** 2)
                                       - 0.38 * (1.0 / (1.0 + math.exp(-16.0 * (t - 0.72)))))
                                      * math.sin(9.5 * math.pi * t), n=140)
    b_ours_ten = _wave_path(668.0, 344.0, 326.0, 12.5,
                            lambda t: (0.18 + 0.82 * math.exp(-((t - 0.50) / 0.20) ** 2)
                                       + 0.45 * (1.0 / (1.0 + math.exp(-16.0 * (t - 0.70)))))
                                      * math.sin(8.5 * math.pi * t + 0.5), n=140)

    # --- Panel c: Col c1 Proprioception waveforms (x=24..212, y=476..704) ---
    wx0, ww = 34.0, 74.0
    w_q_line = _wave_path(wx0, 526.0, ww, 7.2,
                          lambda t: 0.72 * math.sin(2.4 * math.pi * t) + 0.25 * math.sin(5.8 * math.pi * t + 0.6))
    w_q_area = _wave_path(wx0, 526.0, ww, 7.2,
                          lambda t: 0.72 * math.sin(2.4 * math.pi * t) + 0.25 * math.sin(5.8 * math.pi * t + 0.6),
                          close_area=True, y_base=538.0)

    w_qe_line = _wave_path(wx0, 562.0, ww, 8.2,
                           lambda t: 0.30 * math.sin(3.0 * math.pi * t)
                                     + 0.82 * math.exp(-((t - 0.54) / 0.14) ** 2)
                                     - 0.24 * math.exp(-((t - 0.22) / 0.10) ** 2))
    w_qe_area = _wave_path(wx0, 562.0, ww, 8.2,
                           lambda t: 0.30 * math.sin(3.0 * math.pi * t)
                                     + 0.82 * math.exp(-((t - 0.54) / 0.14) ** 2)
                                     - 0.24 * math.exp(-((t - 0.22) / 0.10) ** 2),
                           close_area=True, y_base=575.0)

    w_imu_line = _wave_path(wx0, 598.0, ww, 6.8,
                            lambda t: 0.65 * math.cos(2.0 * math.pi * t + 0.4)
                                      + 0.28 * math.sin(6.6 * math.pi * t))
    w_imu_area = _wave_path(wx0, 598.0, ww, 6.8,
                            lambda t: 0.65 * math.cos(2.0 * math.pi * t + 0.4)
                                      + 0.28 * math.sin(6.6 * math.pi * t),
                            close_area=True, y_base=609.0)

    # Isometric rolling-window tensor O_t (Col c1 bottom)
    tensor_slices = []
    for i in range(5):
        sx = 38.0 + i * 7.4
        sy = 646.0 + i * 2.1
        is_last = (i == 4)
        fill = TEAL_SOFT if is_last else ('#EDF3F7' if i % 2 == 0 else '#FFFFFF')
        stroke = TEAL if is_last else '#94A3B8'
        sw = '1.3' if is_last else '0.8'
        pts = f'{sx:.1f},{sy + 3.6:.1f} {sx + 20:.1f},{sy:.1f} {sx + 20:.1f},{sy + 26:.1f} {sx:.1f},{sy + 29.6:.1f}'
        tensor_slices.append(
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>'
        )
    tensor_slices.append(
        f'<text x="77.5" y="670.0" text-anchor="middle" font-size="8.0" font-weight="700" fill="{TEAL}">o'
        f'<tspan dy="1.8" font-size="6.0">t</tspan></text>'
    )
    tensor_markup = '\n    '.join(tensor_slices)

    # --- Panel c: Col c2 Isometric TCN Slabs ---
    def _tcn_slabs(x_left: float, y_top: float, slab_h: float, count: int,
                   fills: list[str], stroke: str) -> str:
        out = []
        dx, dy, w_s = 13.5, 3.0, 20.0
        for i in range(count):
            sx = x_left + i * dx
            sy = y_top + i * dy
            sh = slab_h - i * 3.8
            pts = (f'{sx:.1f},{sy + 2.6:.1f} '
                   f'{sx + w_s:.1f},{sy:.1f} '
                   f'{sx + w_s:.1f},{sy + sh:.1f} '
                   f'{sx:.1f},{sy + sh + 2.6:.1f}')
            out.append(
                f'<polygon points="{pts}" fill="{fills[i]}" stroke="{stroke}" '
                f'stroke-width="1.15" stroke-linejoin="round"/>'
            )
        return '\n    '.join(out)

    tcn_top = _tcn_slabs(252.0, 520.0, 40.0, 3, ['#DFF0F5', '#CBE6EE', '#B2DAE6'], TEAL)
    tcn_bot = _tcn_slabs(252.0, 598.0, 44.0, 3, ['#E4EEF4', '#D1E2EE', '#BCD4E5'], '#2E6282')

    # --- Panel c: Col c3 Explicit 7D physical state bar gauge ---
    bars = [
        ('v_x',  0.82, TEAL_MID,  '#FFFFFF'),
        ('v_y',  0.34, TEAL_MID,  '#FFFFFF'),
        ('ω_z',  0.54, TEAL_MID,  '#FFFFFF'),
        ('wall', 0.76, TERRA_MID, '#FFFFFF'),
        ('side', 0.48, TERRA_MID, '#FFFFFF'),
        ('drag', 0.66, TERRA_MID, '#FFFFFF'),
        ('slip', 0.28, TERRA_MID, '#FFFFFF'),
    ]
    bar_svg = []
    bx_start = 421.0
    bw = 19.0
    bgap = 7.4
    by_top = 514.0
    bh_max = 36.0
    bar_svg.append(
        f'<line x1="418" y1="{by_top + bh_max * 0.5:.1f}" x2="603" y2="{by_top + bh_max * 0.5:.1f}" '
        f'stroke="#DCE4EE" stroke-width="0.65" stroke-dasharray="2.5 2.5"/>'
        f'<line x1="418" y1="{by_top + bh_max:.1f}" x2="603" y2="{by_top + bh_max:.1f}" '
        f'stroke="#CBD5E1" stroke-width="0.75"/>'
    )
    for idx, (lbl, frac, col, bg) in enumerate(bars):
        bx = bx_start + idx * (bw + bgap)
        fh = bh_max * frac
        fy = by_top + (bh_max - fh)
        bar_svg.append(
            f'<rect x="{bx:.1f}" y="{by_top:.1f}" width="{bw:.1f}" height="{bh_max:.1f}" '
            f'rx="3" fill="{bg}" stroke="{RULE}" stroke-width="0.6"/>'
            f'<rect x="{bx:.1f}" y="{fy:.1f}" width="{bw:.1f}" height="{fh:.1f}" '
            f'rx="3" fill="{col}" opacity="0.90"/>'
        )
        cx_b = bx + bw / 2
        if '_' in lbl:
            base, sub = lbl.split('_')
            bar_svg.append(
                f'<text x="{cx_b:.1f}" y="563.0" text-anchor="middle" font-size="8.6" '
                f'font-weight="600" fill="{SUB}">{base}<tspan dy="1.8" font-size="6.4">{sub}</tspan></text>'
            )
        else:
            bar_svg.append(
                f'<text x="{cx_b:.1f}" y="563.0" text-anchor="middle" font-size="8.2" '
                f'font-weight="600" fill="{SUB}">{lbl}</text>'
            )
    bars_markup = '\n    '.join(bar_svg)

    # --- Panel c: Col c3 Gaussian posterior curve ---
    gx0, gw = 421.0, 94.0
    g_base = 661.0
    g_amp = 34.0
    gauss_line = _wave_path(gx0, g_base, gw, g_amp,
                            lambda t: math.exp(-0.5 * ((t - 0.5) / 0.19) ** 2), n=70)
    gauss_area = _wave_path(gx0, g_base, gw, g_amp,
                            lambda t: math.exp(-0.5 * ((t - 0.5) / 0.19) ** 2), n=70,
                            close_area=True, y_base=g_base)

    # --- Panel d: Self-supervised VAE reconstruction wave (d2 upper) ---
    vx0, vw = 414.0, 96.0
    v_true = _wave_path(vx0, 811.0, vw, 13.5,
                        lambda t: 0.78 * math.sin(2.2 * math.pi * t) + 0.20 * math.cos(4.8 * math.pi * t))
    v_rec = _wave_path(vx0, 811.0, vw, 12.8,
                       lambda t: 0.75 * math.sin(2.2 * math.pi * t + 0.08) + 0.18 * math.cos(4.8 * math.pi * t))

    # --- Panel d: Unit hypersphere S^31 4-regime clusters (d2 lower) ---
    sphere_cx, sphere_cy = 468.0, 904.0
    clusters = [
        (-12.5, -9.0, TEAL,   [(-2.6, -2.2), (2.1, -3.0), (3.0, 1.4), (-1.4, 2.6), (0.5, -0.3)]),
        ( 12.5, -8.5, INDIGO, [(-2.2, -2.2), (2.6, -1.7), (1.0, 2.8), (-2.8, 1.7), (0.3, -0.2)]),
        (-10.5, 11.0, SAND,   [(-2.8, -1.7), (2.4, -2.2), (2.6, 2.0), (-1.3, 2.6), (0.2, 0.2)]),
        ( 11.0, 10.5, TERRA,  [(-2.4, -1.8), (2.2, -2.0), (2.4, 1.8), (-1.7, 2.4), (0.4, 0.0)]),
    ]
    cluster_svg = []
    for cdx, cdy, col, pts in clusters:
        cx_c = sphere_cx + cdx
        cy_c = sphere_cy + cdy
        cluster_svg.append(f'<circle cx="{cx_c:.1f}" cy="{cy_c:.1f}" r="8.2" fill="{col}" opacity="0.14"/>')
        for pdx, pdy in pts:
            cluster_svg.append(f'<circle cx="{cx_c + pdx:.1f}" cy="{cy_c + pdy:.1f}" r="1.85" fill="{col}" opacity="0.92"/>')
    clusters_markup = '\n    '.join(cluster_svg)

    # --- Panel e: Emergent Cross-Media Phase Transition Waves ---
    dx_w, dw_w = 842.0, 102.0
    e1_yaw = _wave_path(dx_w, 811.0, dw_w, 11.2, lambda t: math.sin(3.4 * math.pi * t), n=70)
    e1_ten = _wave_path(dx_w, 811.0, dw_w, 2.6, lambda t: math.cos(3.4 * math.pi * t + 0.5), n=70)

    e2_yaw = _wave_path(dx_w, 876.0, dw_w, 0.7, lambda t: 0.28 * math.sin(2.0 * math.pi * t), n=70)
    e2_ten = _wave_path(dx_w, 876.0, dw_w, 11.5, lambda t: math.sin(3.0 * math.pi * t), n=70)

    e3_yaw = _wave_path(dx_w, 941.0, dw_w, 6.8, lambda t: math.sin(2.5 * math.pi * t + 0.4), n=70)
    e3_ten = _wave_path(dx_w, 941.0, dw_w, 9.2, lambda t: math.sin(2.5 * math.pi * t - 0.6), n=70)

    # --- Flat Perspective Layered Trapezoids ---
    def _mlp_stack(x0: float, y0: float, w_tot: float, h_left: float, h_right: float,
                   fills: list[str], stroke: str) -> str:
        cy = y0 + h_left / 2.0
        parts = []
        for idx, (ox, oy) in enumerate([(-5.2, -2.4), (-2.6, -1.2), (0.0, 0.0)]):
            pts = (
                f'{x0 + ox:.1f},{cy - h_left / 2 + oy:.1f} '
                f'{x0 + w_tot + ox:.1f},{cy - h_right / 2 + oy:.1f} '
                f'{x0 + w_tot + ox:.1f},{cy + h_right / 2 + oy:.1f} '
                f'{x0 + ox:.1f},{cy + h_left / 2 + oy:.1f}'
            )
            sw = '1.25' if idx == 2 else '0.8'
            op = '1.0' if idx == 2 else '0.72'
            parts.append(
                f'<polygon points="{pts}" fill="{fills[idx]}" stroke="{stroke}" '
                f'stroke-width="{sw}" stroke-linejoin="round" opacity="{op}"/>'
            )
        return '\n    '.join(parts)

    actor_stack = _mlp_stack(660.0, 526.0, 64.0, 116.0, 82.0, ['#D5EBF1', '#E4F3F7', '#F6FBFD'], TEAL)
    critic_stack = _mlp_stack(566.0, 777.0, 68.0, 68.0, 44.0, ['#DFE1F4', '#EAEBF8', '#FFFFFF'], INDIGO)
    vaedec_stack = _mlp_stack(316.0, 785.0, 56.0, 42.0, 56.0, ['#DFE1F4', '#EAEBF8', '#FFFFFF'], INDIGO)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{WIDTH_MM:.2f}mm" height="{HEIGHT_MM:.2f}mm"
     viewBox="0 0 {W:.0f} {H:.0f}"
     style="background:#FFFFFF; font-family:'Helvetica Neue', Arial, 'Segoe UI', sans-serif;">

  <defs>
    <radialGradient id="robotShadow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#94A3B8" stop-opacity="0.30"/>
      <stop offset="68%" stop-color="#94A3B8" stop-opacity="0.08"/>
      <stop offset="100%" stop-color="#94A3B8" stop-opacity="0.0"/>
    </radialGradient>

    <linearGradient id="gradTealWave" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{TEAL_MID}" stop-opacity="0.20"/>
      <stop offset="100%" stop-color="{TEAL_MID}" stop-opacity="0.02"/>
    </linearGradient>
    <linearGradient id="gradTerraWave" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{TERRA_MID}" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="{TERRA_MID}" stop-opacity="0.02"/>
    </linearGradient>
    <linearGradient id="gradSlateWave" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#475569" stop-opacity="0.16"/>
      <stop offset="100%" stop-color="#475569" stop-opacity="0.02"/>
    </linearGradient>
    <linearGradient id="gradGauss" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{INDIGO_MID}" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="{INDIGO_MID}" stop-opacity="0.03"/>
    </linearGradient>
    <linearGradient id="gradSpine" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#EEF7FA"/>
      <stop offset="50%" stop-color="#D8ECF2"/>
      <stop offset="100%" stop-color="#EEF7FA"/>
    </linearGradient>
    <radialGradient id="gradSphere" cx="38%" cy="32%" r="65%">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="55%" stop-color="#F1F6F9"/>
      <stop offset="100%" stop-color="#DFEBF1"/>
    </radialGradient>

    <!-- Soft feathered rounded clip for living Nature cutaway -->
    <clipPath id="clipHabitat">
      <rect x="20" y="34" width="424" height="202" rx="10"/>
    </clipPath>
    <!-- Rounded clips for 3-tile Digital Twin 3D mosaic -->
    <clipPath id="clipTwinMain">
      <rect x="30" y="775" width="236" height="72" rx="5.5"/>
    </clipPath>
    <clipPath id="clipTwinCurved">
      <rect x="30" y="852" width="115" height="52" rx="5"/>
    </clipPath>
    <clipPath id="clipTwinSqueeze">
      <rect x="151" y="852" width="115" height="52" rx="5"/>
    </clipPath>

    <!-- Crisp minimalist vector arrowheads -->
    <marker id="arrDark" viewBox="0 0 10 8" refX="8.2" refY="4" markerWidth="6.0" markerHeight="4.8" orient="auto">
      <path d="M 1 1 L 8.5 4 L 1 7 L 2.6 4 Z" fill="#334155"/>
    </marker>
    <marker id="arrTeal" viewBox="0 0 10 8" refX="8.2" refY="4" markerWidth="6.0" markerHeight="4.8" orient="auto">
      <path d="M 1 1 L 8.5 4 L 1 7 L 2.6 4 Z" fill="{TEAL}"/>
    </marker>
    <marker id="arrTerra" viewBox="0 0 10 8" refX="8.2" refY="4" markerWidth="6.4" markerHeight="5.2" orient="auto">
      <path d="M 1 1 L 8.5 4 L 1 7 L 2.6 4 Z" fill="{TERRA}"/>
    </marker>
    <marker id="arrIndigo" viewBox="0 0 10 8" refX="8.2" refY="4" markerWidth="6.0" markerHeight="4.8" orient="auto">
      <path d="M 1 1 L 8.5 4 L 1 7 L 2.6 4 Z" fill="{INDIGO}"/>
    </marker>
  </defs>

  <!-- Pure crisp editorial white canvas -->
  <rect x="0" y="0" width="{W:.0f}" height="{H:.0f}" fill="#FFFFFF"/>

  <!-- ===================================================================== -->
  <!-- PANEL a : LIVING NATURE HABITAT CUTAWAY -> BIO-HYBRID ROBOT (y: 8..258)-->
  <!-- ===================================================================== -->
  <g id="panel-a-living-bio">
    <!-- Left Heading -->
    <text x="22" y="24" font-size="11.6" font-weight="700" fill="{INK}">
      Dual-habitat limbless locomotion in nature
    </text>
    <text x="276" y="24" font-size="8.4" font-weight="600" fill="{SUB}">
      ·  Surface serpenoid &amp; subterranean peristalsis
    </text>

    <!-- Museum-Grade Nature Watercolor/Ink Cutaway Illustration -->
    <image x="20" y="34" width="424" height="202" preserveAspectRatio="xMidYMid slice"
           clip-path="url(#clipHabitat)" href="{habitat_uri}"/>
    <rect x="20" y="34" width="424" height="202" rx="10" fill="none" stroke="{RULE}" stroke-width="0.9"/>

    <!-- Minimal crisp biological annotations directly on the ecological strata -->
    <rect x="27" y="40" width="124" height="17" rx="4" fill="#FFFFFF" opacity="0.90"/>
    <text x="33" y="51.5" font-size="7.8" font-weight="700" fill="{TEAL}">Snake · open ground</text>

    <rect x="27" y="212" width="148" height="17" rx="4" fill="#FFFFFF" opacity="0.90"/>
    <text x="33" y="223.5" font-size="7.8" font-weight="700" fill="{TERRA}">Earthworm · narrow burrow</text>

    <!-- Microscope Zoom Loupe 1: Snake Ventral Scale Anisotropy (cleanly LEFT of snake head!) -->
    <circle cx="176" cy="91" r="5.2" fill="none" stroke="{TEAL}" stroke-width="1.2"/>
    <line x1="178" y1="86" x2="184" y2="71" stroke="{TEAL}" stroke-width="0.9" stroke-dasharray="2.5 2"/>
    <circle cx="187" cy="54" r="16.5" fill="#FFFFFF" stroke="{TEAL}" stroke-width="1.35"/>
    <!-- Overlapping directional ventral scales inside Loupe 1 -->
    <path d="M 174 49 L 181 54 L 174 59 M 180 49 L 187 54 L 180 59 M 186 49 L 193 54 L 186 59 M 192 49 L 199 54 L 192 59"
          fill="none" stroke="{TEAL}" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round"/>
    <rect x="208" y="39" width="126" height="29" rx="4.5" fill="#FFFFFF" opacity="0.92" stroke="{TEAL_BD}" stroke-width="0.7"/>
    <text x="214" y="50.5" font-size="7.4" font-weight="700" fill="{TEAL}">Anisotropic ventral scales</text>
    <text x="214" y="62.0" font-size="6.9" font-weight="500" fill="{SUB}">μ<tspan dy="1.4" font-size="5.2">back</tspan><tspan dy="-1.4"> ≫ μ</tspan><tspan dy="1.4" font-size="5.2">fwd</tspan><tspan dy="-1.4"> · lateral S-wave</tspan></text>

    <!-- Microscope Zoom Loupe 2: Earthworm Hydrostatic Bulge & Muscle Rings -->
    <circle cx="164" cy="186" r="6.5" fill="none" stroke="{TERRA}" stroke-width="1.2"/>
    <line x1="170" y1="181" x2="236" y2="144" stroke="{TERRA}" stroke-width="0.9" stroke-dasharray="2.5 2"/>
    <circle cx="256" cy="142" r="19" fill="#FFFFFF" stroke="{TERRA}" stroke-width="1.35"/>
    <!-- Cross-section circular/longitudinal hydrostatic muscle & radial anchor arrows inside Loupe 2 -->
    <ellipse cx="256" cy="142" rx="11.5" ry="11.5" fill="{TERRA_SOFT}" stroke="{TERRA}" stroke-width="1.3"/>
    <ellipse cx="256" cy="142" rx="6.2" ry="6.2" fill="#FFFFFF" stroke="{TERRA_MID}" stroke-width="0.95" stroke-dasharray="2 1.8"/>
    <path d="M 256 125 L 256 130 M 256 154 L 256 159 M 239 142 L 244 142 M 268 142 L 273 142"
          stroke="{TERRA}" stroke-width="1.3" stroke-linecap="round"/>
    <rect x="279" y="126" width="156" height="30" rx="5" fill="#FFFFFF" opacity="0.92" stroke="{TERRA_BD}" stroke-width="0.7"/>
    <text x="285" y="138" font-size="7.7" font-weight="700" fill="{TERRA}">Hydrostatic metameric bulge</text>
    <text x="285" y="149.5" font-size="7.1" font-weight="500" fill="{SUB}">radial wall anchor · stretch sense</text>

    <!-- Graceful Bio-Hybrid Convergence Bridge (Habitat Cutaway -> 3D CAD Robot) -->
    <path d="M 444 82 C 472 82, 466 130, 496 134" fill="none" stroke="{TEAL}" stroke-width="1.75" marker-end="url(#arrTeal)"/>
    <path d="M 444 188 C 472 188, 466 142, 496 138" fill="none" stroke="{TERRA}" stroke-width="1.75" marker-end="url(#arrTerra)"/>
    <!-- Sleek Bio-Hybrid Fusion Emblem (dual intertwined wave icon) -->
    <circle cx="476" cy="136" r="13.5" fill="#FFFFFF" stroke="{RULE}" stroke-width="1.15"/>
    <path d="M 467 136 C 470 129, 474 129, 476 136 C 478 143, 482 143, 485 136"
          fill="none" stroke="{TEAL}" stroke-width="1.8" stroke-linecap="round"/>
    <ellipse cx="476" cy="136" rx="4.5" ry="7.2" fill="none" stroke="{TERRA}" stroke-width="1.35"/>
    <text x="476" y="213" text-anchor="middle" font-size="7.4" font-weight="700" fill="{SUB}">Bio-hybrid</text>
    <text x="476" y="223" text-anchor="middle" font-size="7.2" font-weight="500" fill="{MUTED}">synthesis</text>

    <!-- Right Monumental Hero: Real 5-Segment Continuum Robot + Bio-Organ Mapping -->
    <text x="512" y="24" font-size="12.0" font-weight="700" fill="{INK}">
      Embodied bio-hybrid continuum robot
    </text>
    <text x="742" y="24" font-size="8.6" font-weight="600" fill="{TERRA}">
      ·  5-segment · 14-DoF
    </text>

    <!-- Top bio-mapped engineering leaders -->
    <text x="516" y="45" font-size="8.0" font-weight="700" fill="{TERRA}">40× 65Mn elastic arches</text>
    <text x="516" y="55" font-size="6.9" font-weight="500" fill="{MUTED}">← earthworm hydrostatic ribs</text>
    <path d="M 594 58 L 594 82" fill="none" stroke="{TERRA_MID}" stroke-width="0.9"/>

    <text x="858" y="45" text-anchor="end" font-size="8.0" font-weight="700" fill="{INDIGO}">10 tendon + 4 yaw servos</text>
    <text x="858" y="55" text-anchor="end" font-size="6.9" font-weight="500" fill="{MUTED}">← longitudinal muscles &amp; spine</text>
    <path d="M 758 58 L 742 102" fill="none" stroke="{INDIGO_MID}" stroke-width="0.9"/>

    <!-- MASSIVE gleaming 3D CAD 5-segment robot floating over soft shadow -->
    <ellipse cx="688" cy="204" rx="142" ry="14" fill="url(#robotShadow)"/>
    <image x="518" y="58" width="336" height="154" preserveAspectRatio="xMidYMid meet" href="{robot_uri}"/>

    <!-- Exact CAD anchor dots -->
    <circle cx="594" cy="82" r="2.5" fill="{TERRA}" stroke="#FFFFFF" stroke-width="0.75"/>
    <circle cx="742" cy="102" r="2.5" fill="{INDIGO}" stroke="#FFFFFF" stroke-width="0.75"/>
    <circle cx="595" cy="173" r="2.5" fill="{TEAL}" stroke="#FFFFFF" stroke-width="0.75"/>
    <circle cx="776" cy="164" r="2.5" fill="#334155" stroke="#FFFFFF" stroke-width="0.75"/>

    <!-- Bottom bio-mapped engineering leaders -->
    <path d="M 595 175 L 590 212" fill="none" stroke="{TEAL_MID}" stroke-width="0.9"/>
    <text x="516" y="224" font-size="8.0" font-weight="700" fill="{TEAL}">20× one-way ratchet wheels</text>
    <text x="516" y="234" font-size="6.9" font-weight="500" fill="{MUTED}">← snake directional ventral scales</text>

    <path d="M 776 166 L 784 212" fill="none" stroke="#64748B" stroke-width="0.9"/>
    <text x="858" y="224" text-anchor="end" font-size="8.0" font-weight="700" fill="{SUB}">1× vestibular 6-axis IMU</text>
    <text x="858" y="234" text-anchor="end" font-size="6.9" font-weight="500" fill="{MUTED}">← balance &amp; orientation cue</text>

    <!-- Right Clean Column: Embodied Mechanoreception (Golgi/Spindle Analog) -->
    <line x1="874" y1="38" x2="874" y2="236" stroke="{RULE}" stroke-width="0.95"/>
    <text x="948" y="49" text-anchor="middle" font-size="8.7" font-weight="700" fill="{TERRA}">
      Embodied load sense
    </text>
    <text x="948" y="60.5" text-anchor="middle" font-size="7.1" font-weight="500" fill="{MUTED}">
      Muscle-spindle / Golgi analog
    </text>

    <rect x="890" y="70" width="116" height="22" rx="4.5" fill="#FFFFFF" stroke="{TERRA_BD}" stroke-width="0.85"/>
    <text x="948" y="84.0" text-anchor="middle" font-size="8.1" font-weight="700" fill="{TERRA}">
      Efference  q<tspan dy="1.6" font-size="6.2">tgt</tspan>
    </text>
    <path d="M 948 92 L 948 104" fill="none" stroke="{TERRA}" stroke-width="1.15" marker-end="url(#arrTerra)"/>

    <rect x="886" y="106" width="124" height="30" rx="4.5" fill="{TERRA_SOFT}" stroke="{TERRA}" stroke-width="0.9"/>
    <text x="948" y="118.5" text-anchor="middle" font-size="7.9" font-weight="700" fill="{INK}">Compliant steel soma</text>
    <text x="948" y="130.0" text-anchor="middle" font-size="7.1" font-weight="600" fill="{TERRA}">+ pipe wall / soil drag</text>
    <path d="M 948 136 L 948 148" fill="none" stroke="{TEAL}" stroke-width="1.15" marker-end="url(#arrTeal)"/>

    <rect x="890" y="150" width="116" height="22" rx="4.5" fill="#FFFFFF" stroke="{TEAL_BD}" stroke-width="0.85"/>
    <text x="948" y="164.0" text-anchor="middle" font-size="8.1" font-weight="700" fill="{TEAL}">
      Reafference  q<tspan dy="1.6" font-size="6.2">meas</tspan>
    </text>

    <text x="948" y="193" text-anchor="middle" font-size="9.0" font-weight="700" fill="{TERRA}">
      q<tspan dy="1.8" font-size="6.8">err</tspan><tspan dy="-1.8"> = q</tspan><tspan dy="1.8" font-size="6.8">tgt</tspan><tspan dy="-1.8"> − q</tspan><tspan dy="1.8" font-size="6.8">meas</tspan>
    </text>
    <text x="948" y="207" text-anchor="middle" font-size="7.3" font-weight="600" fill="{SUB}">
      Body itself is the sensor
    </text>
    <text x="948" y="219" text-anchor="middle" font-size="7.1" font-weight="500" fill="{MUTED}">
      0 vision · 0 tactile skin
    </text>
  </g>

  <!-- Panel a lower-left label & clean hairline separator -->
  <text x="18" y="254" font-size="18.0" font-weight="700" fill="{INK}">a</text>
  <line x1="16" y1="262" x2="1024" y2="262" stroke="{RULE}" stroke-width="0.95"/>

  <!-- ===================================================================== -->
  <!-- PANEL b : PARADIGM CONTRAST & CORE ADVANTAGES (y: 270..428)           -->
  <!-- Vivid visual contrast citing Hirose 1993 / Matsuoka 1985 / Vision FSM -->
  <!-- ===================================================================== -->
  <g id="panel-b-advantages">
    <!-- Column 1: Classical CPG Oscillators (Hirose 1993; Matsuoka 1985) -->
    <rect x="20" y="272" width="298" height="142" rx="8" fill="#FAFBFD" stroke="{RULE}" stroke-width="0.9"/>
    <circle cx="33" cy="287" r="6.5" fill="{WARN_SOFT}" stroke="{WARN}" stroke-width="1.0"/>
    <path d="M 30.2 284.2 L 35.8 289.8 M 35.8 284.2 L 30.2 289.8" stroke="{WARN}" stroke-width="1.25" stroke-linecap="round"/>
    <text x="45" y="290.5" font-size="9.6" font-weight="700" fill="{INK}">Classical CPGs</text>
    <text x="308" y="290.5" text-anchor="end" font-size="7.6" font-weight="600" fill="{MUTED}">Hirose 1993; Matsuoka 1985</text>

    <!-- Vivid graphic vignette: Rigid S-wave slamming narrow pipe wall + non-steerable worm -->
    <!-- Narrow pipe walls -->
    <line x1="34" y1="316" x2="188" y2="316" stroke="#64748B" stroke-width="1.6"/>
    <line x1="34" y1="354" x2="188" y2="354" stroke="#64748B" stroke-width="1.6"/>
    <rect x="34" y="317" width="154" height="36" fill="#F1F5F9" opacity="0.65"/>
    <!-- Rigid high-amplitude S-wave hitting walls -->
    <path d="M 42 335 C 56 307, 74 307, 88 335 C 102 363, 120 363, 134 335 C 148 307, 166 307, 180 335"
          fill="none" stroke="{WARN}" stroke-width="2.2" stroke-linecap="round"/>
    <!-- Crisp geometric collision impact bursts at wall contact points -->
    <path d="M 65 316 L 61 309 M 65 316 L 65 307 M 65 316 L 69 309" stroke="{WARN}" stroke-width="1.3" stroke-linecap="round"/>
    <path d="M 111 354 L 107 361 M 111 354 L 111 363 M 111 354 L 115 361" stroke="{WARN}" stroke-width="1.3" stroke-linecap="round"/>
    <path d="M 157 316 L 153 309 M 157 316 L 157 307 M 157 316 L 161 309" stroke="{WARN}" stroke-width="1.3" stroke-linecap="round"/>
    <text x="111" y="374" text-anchor="middle" font-size="7.6" font-weight="700" fill="{WARN}">Pipe wall jam (12.6 N · CoT 10.7)</text>

    <!-- Right mini vignette: straight-only worm / soft soil stall -->
    <line x1="202" y1="306" x2="202" y2="372" stroke="{RULE}" stroke-width="0.8"/>
    <path d="M 214 328 L 280 328" stroke="#94A3B8" stroke-width="2.4" stroke-linecap="round"/>
    <path d="M 282 328 Q 294 328 296 316" fill="none" stroke="{WARN}" stroke-width="1.2" stroke-dasharray="2.5 2"/>
    <text x="256" y="344" text-anchor="middle" font-size="7.2" font-weight="700" fill="{SUB}">Peristaltic: ω<tspan dy="1.4" font-size="5.4">z</tspan><tspan dy="-1.4"> ≡ 0</tspan></text>
    <text x="256" y="357" text-anchor="middle" font-size="7.0" font-weight="500" fill="{MUTED}">cannot steer</text>
    <text x="256" y="374" text-anchor="middle" font-size="7.2" font-weight="700" fill="{WARN}">Soft stall (CoT 50.8)</text>

    <line x1="30" y1="386" x2="308" y2="386" stroke="{RULE_LIGHT}" stroke-width="0.85"/>
    <text x="169" y="401" text-anchor="middle" font-size="7.9" font-weight="600" fill="{SUB}">
      Fixed oscillators jam in confinement &amp; waste 3–5× energy
    </text>

    <!-- Column 2: Vision / Tactile Skin + Discrete FSM Switching -->
    <rect x="330" y="272" width="298" height="142" rx="8" fill="#FAFBFD" stroke="{RULE}" stroke-width="0.9"/>
    <circle cx="343" cy="287" r="6.5" fill="{WARN_SOFT}" stroke="{WARN}" stroke-width="1.0"/>
    <path d="M 340.2 284.2 L 345.8 289.8 M 345.8 284.2 L 340.2 289.8" stroke="{WARN}" stroke-width="1.25" stroke-linecap="round"/>
    <text x="355" y="290.5" font-size="9.6" font-weight="700" fill="{INK}">Vision / Skin + FSM</text>
    <text x="618" y="290.5" text-anchor="end" font-size="7.6" font-weight="600" fill="{MUTED}">Exteroceptive pipelines</text>

    <!-- Left vignette: Blinded eye/camera in dark opaque/muddy pipe -->
    <rect x="344" y="310" width="122" height="50" rx="6" fill="#1E293B" opacity="0.88"/>
    <!-- Minimal eye glyph crossed out -->
    <path d="M 380 335 Q 405 319 430 335 Q 405 351 380 335 Z" fill="none" stroke="#CBD5E1" stroke-width="1.4"/>
    <circle cx="405" cy="335" r="5.5" fill="none" stroke="#CBD5E1" stroke-width="1.4"/>
    <line x1="385" y1="350" x2="425" y2="320" stroke="#F87171" stroke-width="2.0" stroke-linecap="round"/>
    <text x="405" y="374" text-anchor="middle" font-size="7.5" font-weight="700" fill="{WARN}">Occluded in dark pipe / mud</text>

    <!-- Right vignette: Chattering discrete IF-ELSE mode switch staircase -->
    <line x1="478" y1="306" x2="478" y2="372" stroke="{RULE}" stroke-width="0.8"/>
    <line x1="488" y1="335" x2="612" y2="335" stroke="{WARN}" stroke-width="1.0" stroke-dasharray="3 2.5"/>
    <text x="612" y="331" text-anchor="end" font-size="6.8" font-weight="600" fill="{WARN}">θ<tspan dy="1.4" font-size="5.2">switch</tspan></text>
    <polyline points="490,350 512,350 512,318 524,318 524,350 536,350 536,318 548,318 548,350 564,350 564,318 606,318"
              fill="none" stroke="#475569" stroke-width="1.6" stroke-linejoin="round"/>
    <text x="550" y="374" text-anchor="middle" font-size="7.5" font-weight="700" fill="{SUB}">Discrete FSM mode chatter</text>

    <line x1="340" y1="386" x2="618" y2="386" stroke="{RULE_LIGHT}" stroke-width="0.85"/>
    <text x="479" y="401" text-anchor="middle" font-size="7.9" font-weight="600" fill="{SUB}">
      Fragile external skin · vision failure · hand-tuned thresholds
    </text>

    <!-- Column 3 (HIGHLIGHTED): OURS — Pure-Proprioceptive Continuous Phase Transition -->
    <rect x="640" y="272" width="380" height="142" rx="8" fill="{TEAL_SOFT}" stroke="{TEAL}" stroke-width="1.25"/>
    <circle cx="654" cy="287" r="6.8" fill="{TEAL}"/>
    <path d="M 650.8 287.2 L 653.0 289.6 L 657.6 284.4" fill="none" stroke="#FFFFFF" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="666" y="290.5" font-size="10.0" font-weight="700" fill="{TEAL}">Ours · Embodied Continuous Manifold</text>
    <text x="1010" y="290.5" text-anchor="end" font-size="7.7" font-weight="700" fill="{TERRA}">0 vision · 0 skin · 0 mode ID</text>

    <!-- Continuous seamless regime transition graphic (Hard -> Pipe -> Soft) -->
    <rect x="654" y="302" width="352" height="58" rx="6" fill="#FFFFFF" stroke="{TEAL_BD}" stroke-width="0.8"/>
    <!-- Subtle 3-zone background washes inside timeline -->
    <rect x="655" y="303" width="114" height="56" rx="5" fill="#F2F8FA"/>
    <rect x="769" y="303" width="122" height="56" fill="#F3F4FB"/>
    <rect x="891" y="303" width="114" height="56" rx="5" fill="#FDF5F0"/>
    <!-- Regime labels -->
    <text x="712" y="315" text-anchor="middle" font-size="7.2" font-weight="700" fill="{TEAL}">Hard · Serpenoid</text>
    <text x="830" y="315" text-anchor="middle" font-size="7.2" font-weight="700" fill="{INDIGO}">Ø236 Pipe · Peristaltic</text>
    <text x="948" y="315" text-anchor="middle" font-size="7.2" font-weight="700" fill="{TERRA}">Soft Soil · Hybrid</text>
    <!-- Continuous dual-channel phase-transition wave -->
    <path d="{b_ours_ten}" fill="none" stroke="{TERRA}" stroke-width="1.55" opacity="0.90" stroke-linecap="round"/>
    <path d="{b_ours_yaw}" fill="none" stroke="{TEAL}" stroke-width="1.75" stroke-linecap="round"/>

    <!-- 4 Bold Quantitative Advantage KPI Blocks (instant knockout evidence!) -->
    <rect x="654" y="367" width="84" height="39" rx="5.5" fill="#FFFFFF" stroke="{TEAL_BD}" stroke-width="0.8"/>
    <text x="696" y="384" text-anchor="middle" font-size="11.2" font-weight="700" fill="{TEAL}">3.3–3.8×</text>
    <text x="696" y="398" text-anchor="middle" font-size="7.1" font-weight="600" fill="{SUB}">Lower CoT vs CPG</text>

    <rect x="743" y="367" width="84" height="39" rx="5.5" fill="#FFFFFF" stroke="{TERRA_BD}" stroke-width="0.8"/>
    <text x="785" y="384" text-anchor="middle" font-size="11.2" font-weight="700" fill="{TERRA}">30× less</text>
    <text x="785" y="398" text-anchor="middle" font-size="7.1" font-weight="600" fill="{SUB}">Wall force (0.4 N)</text>

    <rect x="832" y="367" width="84" height="39" rx="5.5" fill="#FFFFFF" stroke="{INDIGO_BD}" stroke-width="0.8"/>
    <text x="874" y="384" text-anchor="middle" font-size="11.2" font-weight="700" fill="{INDIGO}">0.0° drift</text>
    <text x="874" y="398" text-anchor="middle" font-size="7.1" font-weight="600" fill="{SUB}">Exact bilateral ℤ₂</text>

    <rect x="921" y="367" width="85" height="39" rx="5.5" fill="#FFFFFF" stroke="{TEAL_BD}" stroke-width="0.8"/>
    <text x="963.5" y="384" text-anchor="middle" font-size="11.2" font-weight="700" fill="{TEAL}">42 / 42</text>
    <text x="963.5" y="398" text-anchor="middle" font-size="7.1" font-weight="600" fill="{SUB}">Zero-fail · &lt;2.2 ms</text>
  </g>

  <!-- Panel b lower-left label & clean hairline separator -->
  <text x="18" y="427" font-size="18.0" font-weight="700" fill="{INK}">b</text>
  <line x1="16" y1="435" x2="1024" y2="435" stroke="{RULE}" stroke-width="0.95"/>

  <!-- ===================================================================== -->
  <!-- PANEL c : ONBOARD PURE-PROPRIOCEPTIVE NEUROMECHANICAL LOOP (y:442..724)-->
  <!-- ===================================================================== -->
  <path d="M 968 542 L 968 456 Q 968 448 960 448 L 126 448 Q 118 448 118 456 L 118 472"
        fill="none" stroke="{TERRA}" stroke-width="1.5" marker-end="url(#arrTerra)"/>
  <rect x="330" y="439.5" width="424" height="16.5" fill="#FFFFFF"/>
  <circle cx="342" cy="447.8" r="3.0" fill="{TERRA}"/>
  <text x="546" y="451.2" text-anchor="middle" font-size="9.6" font-weight="600" fill="{TERRA}" letter-spacing="0.05">
    Efferent motor loop  ·  10 Hz policy → 50 Hz UART servo bus &amp; safety watchdog
  </text>

  <!-- COL c1 : Embodied Proprioception (x: 24..212, y: 476..704) -->
  <g id="col-c1-proprio">
    <rect x="24" y="476" width="188" height="228" rx="8" fill="#FAFBFD" stroke="{RULE}" stroke-width="0.95"/>
    <text x="34" y="494" font-size="11.0" font-weight="700" fill="{INK}">Proprioception</text>
    <text x="202" y="494" text-anchor="end" font-size="8.0" font-weight="600" fill="{TEAL}">65D · RunningNorm</text>
    <line x1="34" y1="501.5" x2="202" y2="501.5" stroke="{RULE_LIGHT}" stroke-width="0.85"/>

    <!-- Track 1: q, dq (28D) -->
    <rect x="32" y="508" width="172" height="30" rx="5" fill="#F2F8FA"/>
    <path d="{w_q_area}" fill="url(#gradTealWave)"/>
    <path d="{w_q_line}" fill="none" stroke="{TEAL_MID}" stroke-width="1.45" stroke-linecap="round"/>
    <text x="197" y="520.5" text-anchor="end" font-size="9.6" font-weight="700" fill="{TEAL}">q, q&#x0307;</text>
    <text x="197" y="532.0" text-anchor="end" font-size="7.4" font-weight="500" fill="{MUTED}">28D posture</text>

    <!-- Track 2: Highlighted Efference-Copy Mismatch q_err (14D) -->
    <rect x="32" y="543" width="172" height="33" rx="5" fill="{TERRA_SOFT}" stroke="{TERRA_BD}" stroke-width="1.05"/>
    <path d="{w_qe_area}" fill="url(#gradTerraWave)"/>
    <path d="{w_qe_line}" fill="none" stroke="{TERRA}" stroke-width="1.6" stroke-linecap="round"/>
    <text x="197" y="557.0" text-anchor="end" font-size="9.8" font-weight="700" fill="{TERRA}">
      q<tspan dy="2.0" font-size="7.0">err</tspan>
    </text>
    <text x="197" y="569.5" text-anchor="end" font-size="7.3" font-weight="700" fill="{TERRA}">14D load cue</text>

    <!-- Track 3: Vestibular IMU (6D) -->
    <rect x="32" y="581" width="172" height="29" rx="5" fill="#F4F6F9"/>
    <path d="{w_imu_area}" fill="url(#gradSlateWave)"/>
    <path d="{w_imu_line}" fill="none" stroke="#334155" stroke-width="1.4" stroke-linecap="round"/>
    <text x="197" y="593.0" text-anchor="end" font-size="9.6" font-weight="700" fill="{INK}">ω, g</text>
    <text x="197" y="604.5" text-anchor="end" font-size="7.4" font-weight="500" fill="{MUTED}">6D vestibular</text>

    <!-- Track 4: Efferent u_prev + Goal cmd inline strip -->
    <text x="36" y="625" font-size="8.0" font-weight="600" fill="{SUB}">
      <tspan font-weight="700" fill="{INK}">u</tspan><tspan dy="1.5" font-size="6.0" font-weight="700" fill="{INK}">t−1</tspan><tspan dy="-1.5"> (14D)</tspan>
    </text>
    <text x="200" y="625" text-anchor="end" font-size="8.0" font-weight="600" fill="{TEAL}">
      cmd: v<tspan dy="1.4" font-size="5.8">x</tspan><tspan dy="-2.8" font-size="6.2">*</tspan><tspan dy="1.4">, ω</tspan><tspan dy="1.4" font-size="5.8">z</tspan><tspan dy="-2.8" font-size="6.2">*</tspan><tspan dy="1.4"> (3D)</tspan>
    </text>

    <line x1="34" y1="634" x2="202" y2="634" stroke="{RULE_LIGHT}" stroke-width="0.85"/>

    <!-- Isometric Rolling-Window Tensor O_t (16 x 65) -->
    {tensor_markup}
    <text x="155" y="658" text-anchor="middle" font-size="9.4" font-weight="700" fill="{INK}">
      O<tspan dy="1.8" font-size="6.8">t</tspan><tspan dy="-1.8"> ∈ ℝ</tspan><tspan dy="-2.8" font-size="6.8">16×65</tspan>
    </text>
    <text x="155" y="672" text-anchor="middle" font-size="7.8" font-weight="500" fill="{SUB}">1.6 s @ 10 Hz</text>
    <text x="155" y="686" text-anchor="middle" font-size="7.3" font-weight="600" fill="{MUTED}">no vision · no mode ID</text>
  </g>

  <!-- COL c2 : Spinal Dual-Scale Causal 1D-TCN (x: 232..386) -->
  <g id="col-c2-spinal-tcn">
    <text x="309" y="492" text-anchor="middle" font-size="11.2" font-weight="700" fill="{INK}">Spinal 1D-TCN</text>

    <path d="M 212 584 C 232 584, 228 542, 248 542" fill="none" stroke="{TEAL}" stroke-width="1.4" marker-end="url(#arrTeal)"/>
    <path d="M 212 584 C 232 584, 228 622, 248 622" fill="none" stroke="#2E6282" stroke-width="1.4" marker-end="url(#arrTeal)"/>

    <text x="276" y="508" text-anchor="middle" font-size="8.6" font-weight="700" fill="{TEAL}">
      Fast reflex <tspan font-weight="500" fill="{SUB}">k=3 (48→32)</tspan>
    </text>
    <text x="276" y="517.5" text-anchor="middle" font-size="7.2" font-weight="500" fill="{MUTED}">0.7 s · impact / slip</text>
    {tcn_top}

    {tcn_bot}
    <text x="276" y="658" text-anchor="middle" font-size="8.6" font-weight="700" fill="#2E6282">
      Slow wave <tspan font-weight="500" fill="{SUB}">k=5 (24→16)</tspan>
    </text>
    <text x="276" y="668" text-anchor="middle" font-size="7.2" font-weight="500" fill="{MUTED}">1.3 s · gait cycle</text>

    <path d="M 300 542 C 322 542, 322 570, 344 574" fill="none" stroke="{TEAL}" stroke-width="1.35" marker-end="url(#arrTeal)"/>
    <path d="M 300 622 C 322 622, 322 600, 344 596" fill="none" stroke="#2E6282" stroke-width="1.35" marker-end="url(#arrTeal)"/>

    <!-- Direct o_t proprioceptive skip (65D) -> c_t -->
    <path d="M 212 690 L 354 690 Q 364 690 364 680 L 364 642"
          fill="none" stroke="{TEAL}" stroke-width="1.2" stroke-dasharray="3.2 2.4" marker-end="url(#arrTeal)"/>
    <rect x="252" y="682" width="68" height="15" fill="#FFFFFF"/>
    <text x="286" y="692.5" text-anchor="middle" font-size="7.9" font-weight="700" fill="{TEAL}">
      o<tspan dy="1.6" font-size="6.0">t</tspan><tspan dy="-1.6"> skip (65D)</tspan>
    </text>

    <rect x="348" y="534" width="32" height="104" rx="14"
          fill="url(#gradSpine)" stroke="{TEAL}" stroke-width="1.25"/>
    <text x="364" y="578" text-anchor="middle" font-size="11.8" font-weight="700" fill="{TEAL}">
      c<tspan dy="2.2" font-size="8.0">t</tspan>
    </text>
    <text x="364" y="594" text-anchor="middle" font-size="7.6" font-weight="700" fill="{SUB}">257D</text>
    <text x="364" y="606" text-anchor="middle" font-size="6.7" font-weight="500" fill="{MUTED}">48×4+65</text>
  </g>

  <!-- COL c3 : Cerebellar Hybrid Internal Model (x: 408..612) -->
  <g id="col-c3-cerebellar">
    <path d="M 380 568 C 394 568, 390 524, 404 524" fill="none" stroke="{TEAL}" stroke-width="1.35" marker-end="url(#arrTeal)"/>
    <path d="M 380 600 C 394 600, 390 650, 404 650" fill="none" stroke="{INDIGO}" stroke-width="1.35" marker-end="url(#arrIndigo)"/>

    <path d="M 380 586 L 620 586" fill="none" stroke="#475569" stroke-width="1.3" marker-end="url(#arrDark)"/>
    <rect x="458" y="578.5" width="104" height="15" fill="#FFFFFF"/>
    <text x="510" y="589.0" text-anchor="middle" font-size="8.1" font-weight="600" fill="{SUB}">
      skip  <tspan font-weight="700" fill="{TEAL}">sg(c</tspan><tspan dy="1.6" font-size="6.0" font-weight="700" fill="{TEAL}">t</tspan><tspan dy="-1.6" font-weight="700" fill="{TEAL}">)</tspan>  257D
    </text>

    <!-- Upper Flat Zone: Explicit Body & Contact State (7D) -->
    <rect x="408" y="476" width="204" height="96" rx="8" fill="{TEAL_SOFT}" stroke="{TEAL_BD}" stroke-width="0.95"/>
    <text x="418" y="493" font-size="10.4" font-weight="700" fill="{INK}">
      Body &amp; contact <tspan fill="{TEAL}">e&#x0302;</tspan><tspan dy="2.0" font-size="7.4" fill="{TEAL}">t</tspan>
    </text>
    <text x="602" y="493" text-anchor="end" font-size="8.0" font-weight="700" fill="{TEAL}">7D</text>
    <text x="457" y="508" text-anchor="middle" font-size="7.2" font-weight="700" fill="{TEAL}">Ego-motion (linear)</text>
    <text x="552" y="508" text-anchor="middle" font-size="7.2" font-weight="700" fill="{TERRA}">Mechanoreception (σ, tanh)</text>
    {bars_markup}

    <!-- Lower Flat Zone: Precision-Weighted Terrain Manifold (16D beta-VAE) -->
    <rect x="408" y="598" width="204" height="106" rx="8" fill="{INDIGO_SOFT}" stroke="{INDIGO_BD}" stroke-width="0.95"/>
    <text x="418" y="615" font-size="10.4" font-weight="700" fill="{INK}">
      Terrain manifold <tspan fill="{INDIGO}">z&#x0303;</tspan><tspan dy="2.0" font-size="7.4" fill="{INDIGO}">t</tspan>
    </text>
    <text x="602" y="615" text-anchor="end" font-size="8.0" font-weight="700" fill="{INDIGO}">16D β-VAE</text>

    <line x1="418" y1="661" x2="518" y2="661" stroke="#CBD5E1" stroke-width="0.9"/>
    <path d="{gauss_area}" fill="url(#gradGauss)"/>
    <path d="{gauss_line}" fill="none" stroke="{INDIGO}" stroke-width="1.65" stroke-linecap="round"/>
    <line x1="468" y1="627" x2="468" y2="661" stroke="{INDIGO_MID}" stroke-width="0.95" stroke-dasharray="2.5 2.5"/>
    <text x="468" y="674.0" text-anchor="middle" font-size="8.2" font-weight="600" fill="{INDIGO}">
      𝒩(μ<tspan dy="1.5" font-size="6.2">z</tspan><tspan dy="-1.5">, σ</tspan><tspan dy="-2.6" font-size="6.0">2</tspan><tspan dy="4.2" dx="-2.6" font-size="6.0">z</tspan><tspan dy="-1.6">) → z</tspan><tspan dy="1.5" font-size="6.2">t</tspan>
    </text>

    <rect x="534" y="623" width="66" height="16" rx="4.5" fill="#FFFFFF" stroke="{INDIGO_BD}" stroke-width="0.8"/>
    <text x="567" y="633.8" text-anchor="middle" font-size="7.4" font-weight="700" fill="{INDIGO}">
      σ(W<tspan dy="1.4" font-size="5.6">g</tspan><tspan dy="-1.4">·[μ,σ²])</tspan>
    </text>
    <path d="M 567 639 L 567 647.5" fill="none" stroke="{INDIGO}" stroke-width="1.1" marker-end="url(#arrIndigo)"/>
    <path d="M 519 658 L 552 658" fill="none" stroke="{INDIGO}" stroke-width="1.2" marker-end="url(#arrIndigo)"/>
    <circle cx="567" cy="658" r="10.0" fill="#FFFFFF" stroke="{INDIGO}" stroke-width="1.2"/>
    <circle cx="567" cy="658" r="2.3" fill="{INDIGO}"/>
    <text x="567" y="675.0" text-anchor="middle" font-size="7.4" font-weight="600" fill="{SUB}">precision ⊙</text>

    <text x="510" y="695.0" text-anchor="middle" font-size="7.7" font-weight="600" fill="{INDIGO}">
      z&#x0303;<tspan dy="1.5" font-size="5.8">t</tspan><tspan dy="-1.5"> = σ(W</tspan><tspan dy="1.5" font-size="5.8">g</tspan><tspan dy="-1.5">[sg(μ</tspan><tspan dy="1.5" font-size="5.8">z</tspan><tspan dy="-1.5">, log σ</tspan><tspan dy="-2.2" font-size="5.6">2</tspan><tspan dy="2.2">)]) ⊙ sg(z</tspan><tspan dy="1.5" font-size="5.8">t</tspan><tspan dy="-1.5">)</tspan>
    </text>
  </g>

  <!-- COL c4 : Bilateral Symmetric Motor Policy (x: 622..1016) -->
  <g id="col-c4-policy">
    <path d="M 612 524 C 624 524, 619 578, 626 582" fill="none" stroke="{TEAL}" stroke-width="1.3"/>
    <path d="M 612 650 C 624 650, 619 594, 626 590" fill="none" stroke="{INDIGO}" stroke-width="1.3"/>

    <circle cx="635" cy="586" r="11.2" fill="#FFFFFF" stroke="{TERRA}" stroke-width="1.4"/>
    <text x="635" y="589.3" text-anchor="middle" font-size="9.2" font-weight="700" fill="{TERRA}">sg</text>
    <text x="635" y="568" text-anchor="middle" font-size="7.5" font-weight="700" fill="{SUB}">280D</text>

    <path d="M 647 586 L 656 586" fill="none" stroke="#334155" stroke-width="1.35" marker-end="url(#arrDark)"/>

    <text x="658" y="492" font-size="11.4" font-weight="700" fill="{INK}">
      Bilateral symmetric motor policy
    </text>

    {actor_stack}
    <text x="692" y="572" text-anchor="middle" font-size="11.0" font-weight="700" fill="{TEAL}">Motor</text>
    <text x="692" y="586" text-anchor="middle" font-size="11.0" font-weight="700" fill="{TEAL}">Policy</text>
    <text x="692" y="600.5" text-anchor="middle" font-size="7.5" font-weight="600" fill="{SUB}">256→192→14</text>
    <text x="692" y="611.5" text-anchor="middle" font-size="7.0" font-weight="500" fill="{MUTED}">LN · SiLU · tanh</text>
    <text x="692" y="658" text-anchor="middle" font-size="7.9" font-weight="600" fill="{TEAL}">229k deploy params</text>

    <path d="M 724 584 L 752 584" fill="none" stroke="#334155" stroke-width="1.45" marker-end="url(#arrDark)"/>

    <!-- Bilateral Z2 Commissural Symmetry -->
    <rect x="756" y="510" width="130" height="150" rx="8" fill="{TEAL_SOFT}" stroke="{TEAL_BD}" stroke-width="0.95"/>
    <line x1="821" y1="521" x2="821" y2="596" stroke="{TEAL}" stroke-width="1.15" stroke-dasharray="3.2 2.5"/>
    <path d="M 808 527 C 790 536, 790 548, 803 558 C 813 567, 813 579, 795 590"
          fill="none" stroke="{TEAL}" stroke-width="2.2" stroke-linecap="round"/>
    <circle cx="808" cy="527" r="2.4" fill="{TEAL}"/>
    <circle cx="794" cy="540" r="2.1" fill="{TEAL}"/>
    <circle cx="803" cy="558" r="2.1" fill="{TEAL}"/>
    <circle cx="810" cy="574" r="2.1" fill="{TEAL}"/>
    <circle cx="795" cy="590" r="2.4" fill="{TEAL}"/>
    <path d="M 834 527 C 852 536, 852 548, 839 558 C 829 567, 829 579, 847 590"
          fill="none" stroke="{TERRA}" stroke-width="2.2" stroke-linecap="round"/>
    <circle cx="834" cy="527" r="2.4" fill="{TERRA}"/>
    <circle cx="848" cy="540" r="2.1" fill="{TERRA}"/>
    <circle cx="839" cy="558" r="2.1" fill="{TERRA}"/>
    <circle cx="832" cy="574" r="2.1" fill="{TERRA}"/>
    <circle cx="847" cy="590" r="2.4" fill="{TERRA}"/>
    <path d="M 811 558 L 831 558" fill="none" stroke="{MUTED}" stroke-width="1.0"/>
    <path d="M 814 555.5 L 810.5 558 L 814 560.5 M 828 555.5 L 831.5 558 L 828 560.5"
          fill="none" stroke="{MUTED}" stroke-width="1.0" stroke-linecap="round" stroke-linejoin="round"/>

    <text x="821" y="616" text-anchor="middle" font-size="9.6" font-weight="700" fill="{TEAL}">
      Bilateral ℤ<tspan dy="1.8" font-size="7.2">2</tspan><tspan dy="-1.8"> symmetry</tspan>
    </text>
    <text x="821" y="631.5" text-anchor="middle" font-size="7.6" font-weight="600" fill="{SUB}">
      ½[π(O) + K<tspan dy="-2.4" font-size="5.6">−1</tspan><tspan dy="4.2" dx="-5.0" font-size="5.6">g</tspan><tspan dy="-1.8">π(L</tspan><tspan dy="1.8" font-size="5.6">g</tspan><tspan dy="-1.8">O)]</tspan>
    </text>
    <text x="821" y="649" text-anchor="middle" font-size="8.2" font-weight="700" fill="{TEAL}">
      0.0° yaw drift
    </text>

    <path d="M 886 584 L 916 584" fill="none" stroke="{TERRA}" stroke-width="1.55" marker-end="url(#arrTerra)"/>

    <!-- 14D Efferent Action -->
    <rect x="920" y="542" width="96" height="86" rx="8" fill="{TERRA_SOFT}" stroke="{TERRA}" stroke-width="1.25"/>
    <text x="968" y="564" text-anchor="middle" font-size="12.0" font-weight="700" fill="{TERRA}">
      14D u<tspan dy="2.0" font-size="8.2">t</tspan>
    </text>
    <line x1="934" y1="573" x2="1002" y2="573" stroke="{TERRA_BD}" stroke-width="0.9"/>
    <text x="968" y="589" text-anchor="middle" font-size="8.8" font-weight="700" fill="#7C361E">10 tendon</text>
    <text x="968" y="603" text-anchor="middle" font-size="8.8" font-weight="700" fill="#7C361E">4 yaw</text>
    <text x="968" y="618" text-anchor="middle" font-size="7.8" font-weight="600" fill="{TERRA}">10 → 50 Hz</text>
  </g>

  <!-- Panel c lower-left label & onboard note -->
  <text x="18" y="718" font-size="18.0" font-weight="700" fill="{INK}">c</text>
  <text x="944" y="717" text-anchor="end" font-size="8.8" font-weight="600" fill="{TEAL}">
    Onboard · 10 Hz / 50 Hz · &lt;2.2 ms CPU
  </text>

  <!-- Section horizontal separator above bottom asymmetric spread -->
  <line x1="16" y1="726" x2="742" y2="726" stroke="#CBD5E1" stroke-width="0.95" stroke-dasharray="4 4"/>

  <!-- ===================================================================== -->
  <!-- PANEL d : UNIFIED FLAT SLATE-MIST SIM TRAINING ZONE (x: 18..742)      -->
  <!-- ===================================================================== -->
  <g id="panel-d-sim-training">
    <rect x="18" y="738" width="724" height="248" rx="10"
          fill="{MIST}" stroke="#CBD5E1" stroke-width="1.0" stroke-dasharray="4.5 3.5"/>

    <text x="294" y="753" text-anchor="middle" font-size="8.4" font-weight="600" fill="{MUTED}" letter-spacing="0.05">
      Simulation-only training · discarded at deployment
    </text>

    <!-- SUB-COLUMN d1 : FP64 Continuum Digital Twin -->
    <text x="30" y="768" font-size="10.4" font-weight="700" fill="{INK}">FP64 Continuum Twin</text>
    <text x="266" y="768" text-anchor="end" font-size="7.8" font-weight="700" fill="{TERRA}">1 kHz · 474× RT</text>

    <image x="30" y="775" width="236" height="72" preserveAspectRatio="xMidYMid slice"
           clip-path="url(#clipTwinMain)" href="{twin_pipe_soil_uri}"/>
    <rect x="30" y="775" width="236" height="72" rx="5.5" fill="none" stroke="#CBD5E1" stroke-width="0.85"/>
    <rect x="35" y="831" width="56" height="12" rx="3.2" fill="#0F172A" opacity="0.68"/>
    <text x="63" y="839.5" text-anchor="middle" font-size="6.9" font-weight="600" fill="#FFFFFF">Pipe → Soil</text>

    <image x="30" y="852" width="115" height="52" preserveAspectRatio="xMidYMid slice"
           clip-path="url(#clipTwinCurved)" href="{twin_curved_uri}"/>
    <rect x="30" y="852" width="115" height="52" rx="5" fill="none" stroke="#CBD5E1" stroke-width="0.85"/>
    <rect x="35" y="889" width="48" height="11.5" rx="3" fill="#0F172A" opacity="0.68"/>
    <text x="59" y="897.2" text-anchor="middle" font-size="6.6" font-weight="600" fill="#FFFFFF">90° Curved</text>

    <image x="151" y="852" width="115" height="52" preserveAspectRatio="xMidYMid slice"
           clip-path="url(#clipTwinSqueeze)" href="{twin_squeeze_uri}"/>
    <rect x="151" y="852" width="115" height="52" rx="5" fill="none" stroke="#CBD5E1" stroke-width="0.85"/>
    <rect x="156" y="889" width="54" height="11.5" rx="3" fill="#0F172A" opacity="0.68"/>
    <text x="183" y="897.2" text-anchor="middle" font-size="6.6" font-weight="600" fill="#FFFFFF">Wall Squeeze</text>

    <text x="32" y="920" font-size="7.8" font-weight="700" fill="{INK}">
      Block-Thomas–Schur FP64 <tspan font-weight="500" fill="{TEAL}">· UTM R² = 0.9999</tspan>
    </text>
    <text x="32" y="934" font-size="7.7" font-weight="500" fill="{SUB}">
      Implicit ARPC (μ<tspan dy="1.4" font-size="5.8">rr</tspan><tspan dy="-1.4"> 0.025–0.23) · 2-span Capstan</tspan>
    </text>
    <text x="32" y="948" font-size="7.7" font-weight="500" fill="{SUB}">
      5-axis DR · straight / 90° / S / U / tapered pipes
    </text>

    <line x1="282" y1="760" x2="282" y2="954" stroke="{RULE}" stroke-width="0.95"/>

    <!-- Target arrow d1 -> d2 -->
    <path d="M 268 812 L 306 812" fill="none" stroke="#64748B" stroke-width="1.2"
          stroke-dasharray="3 2.5" marker-end="url(#arrDark)"/>
    <rect x="273" y="786" width="28" height="22" fill="{MIST}"/>
    <text x="287" y="796.0" text-anchor="middle" font-size="7.4" font-weight="600" fill="{SUB}">
      e<tspan dy="-2.0" font-size="5.6">*</tspan><tspan dy="3.6" dx="-2.2" font-size="5.6">t</tspan><tspan dy="-1.6">,</tspan>
    </text>
    <text x="287" y="806.0" text-anchor="middle" font-size="7.4" font-weight="600" fill="{SUB}">
      o<tspan dy="1.4" font-size="5.4">t+1</tspan>
    </text>

    <!-- SUB-COLUMN d2 : Predictive Internal Model (HIM-DreamWaQ) -->
    <text x="296" y="768" font-size="10.4" font-weight="700" fill="{INK}">Predictive Internal Model</text>
    <text x="530" y="768" text-anchor="end" font-size="7.8" font-weight="700" fill="{INDIGO}">HIM-DreamWaQ</text>

    {vaedec_stack}
    <text x="344" y="802" text-anchor="middle" font-size="8.6" font-weight="700" fill="{INDIGO}">β-VAE</text>
    <text x="344" y="813" text-anchor="middle" font-size="8.0" font-weight="600" fill="{INDIGO}">Dec</text>
    <text x="344" y="823" text-anchor="middle" font-size="6.5" font-weight="500" fill="{MUTED}">62D sens</text>

    <path d="M 374 811 L 398 811" fill="none" stroke="{INDIGO}" stroke-width="1.2" marker-end="url(#arrIndigo)"/>

    <rect x="403" y="785" width="118" height="51" rx="5.5" fill="#FFFFFF" stroke="{INDIGO_BD}" stroke-width="0.85"/>
    <path d="{v_true}" fill="none" stroke="#94A3B8" stroke-width="1.2" stroke-dasharray="3 2.5"/>
    <path d="{v_rec}" fill="none" stroke="{INDIGO}" stroke-width="1.6" stroke-linecap="round"/>

    <text x="414" y="851" text-anchor="middle" font-size="8.0" font-weight="700" fill="{INDIGO}">
      ℒ<tspan dy="1.5" font-size="6.0">est</tspan><tspan dy="-1.5"> + ℒ</tspan><tspan dy="1.5" font-size="6.0">rec</tspan><tspan dy="-1.5"> + βD</tspan><tspan dy="1.5" font-size="6.0">KL</tspan><tspan dy="-1.5" font-weight="500" fill="{SUB}">  (o</tspan><tspan dy="1.5" font-size="5.8" font-weight="500" fill="{SUB}">t+1</tspan><tspan dy="-1.5" font-weight="500" fill="{SUB}"> vs o&#x0302;</tspan><tspan dy="1.5" font-size="5.8" font-weight="500" fill="{SUB}">t+1</tspan><tspan dy="-1.5" font-weight="500" fill="{SUB}">)</tspan>
    </text>

    <line x1="298" y1="860" x2="528" y2="860" stroke="{RULE}" stroke-width="0.85"/>

    <text x="302" y="880" font-size="8.2" font-weight="700" fill="{TEAL}">
      f<tspan dy="1.5" font-size="6.0">src</tspan><tspan dy="-1.5">(z</tspan><tspan dy="1.5" font-size="6.0">t</tspan><tspan dy="-1.5">, a</tspan><tspan dy="1.5" font-size="6.0">t</tspan><tspan dy="-1.5">)</tspan>
    </text>
    <text x="302" y="896" font-size="8.2" font-weight="700" fill="{INDIGO}">
      EMA f<tspan dy="1.5" font-size="6.0">tgt</tspan><tspan dy="-1.5"> (m=0.99)</tspan>
    </text>
    <path d="M 384 883 L 430 894" fill="none" stroke="{TEAL}" stroke-width="1.15" marker-end="url(#arrTeal)"/>
    <path d="M 384 899 L 430 908" fill="none" stroke="{INDIGO}" stroke-width="1.15" marker-end="url(#arrIndigo)"/>

    <text x="302" y="921" font-size="7.9" font-weight="700" fill="{TEAL}">
      𝕊<tspan dy="-2.2" font-size="6.0">31</tspan><tspan dy="2.2"> Sym. InfoNCE (τ=0.1)</tspan>
    </text>
    <text x="302" y="934" font-size="7.5" font-weight="500" fill="{SUB}">
      + 3-step Sinkhorn SwAV (K=16)
    </text>
    <circle cx="305" cy="947" r="2.1" fill="{TEAL}"/>
    <text x="310" y="949.5" font-size="6.9" font-weight="600" fill="{SUB}">Hard</text>
    <circle cx="336" cy="947" r="2.1" fill="{INDIGO}"/>
    <text x="341" y="949.5" font-size="6.9" font-weight="600" fill="{SUB}">Pipe</text>
    <circle cx="366" cy="947" r="2.1" fill="{SAND}"/>
    <text x="371" y="949.5" font-size="6.9" font-weight="600" fill="{SUB}">Carpet</text>
    <circle cx="403" cy="947" r="2.1" fill="{TERRA}"/>
    <text x="408" y="949.5" font-size="6.9" font-weight="600" fill="{SUB}">Soft</text>

    <circle cx="{sphere_cx:.1f}" cy="{sphere_cy:.1f}" r="28.5"
            fill="url(#gradSphere)" stroke="{TEAL_BD}" stroke-width="1.1"/>
    <ellipse cx="{sphere_cx:.1f}" cy="{sphere_cy:.1f}" rx="28.5" ry="8.5"
             fill="none" stroke="{TEAL_BD}" stroke-width="0.8" stroke-dasharray="2.5 2.5"/>
    <ellipse cx="{sphere_cx:.1f}" cy="{sphere_cy:.1f}" rx="8.6" ry="28.5"
             fill="none" stroke="{TEAL_BD}" stroke-width="0.65" opacity="0.55"/>
    {clusters_markup}

    <line x1="544" y1="760" x2="544" y2="954" stroke="{RULE}" stroke-width="0.95"/>

    <!-- SUB-COLUMN d3 : Privileged Critic & Min-CoT Symmetric PPO -->
    <text x="556" y="768" font-size="10.4" font-weight="700" fill="{INK}">Critic &amp; Min-CoT PPO</text>

    {critic_stack}
    <text x="600" y="803" text-anchor="middle" font-size="9.4" font-weight="700" fill="{INDIGO}">Critic</text>
    <text x="600" y="815" text-anchor="middle" font-size="7.8" font-weight="600" fill="{SUB}">
      V(O<tspan dy="1.4" font-size="5.8">t</tspan><tspan dy="-1.4">, s</tspan><tspan dy="1.4" font-size="5.8">priv</tspan><tspan dy="-1.4">)</tspan>
    </text>
    <text x="600" y="826" text-anchor="middle" font-size="6.5" font-weight="500" fill="{MUTED}">512→256→128</text>

    <text x="646" y="796" font-size="7.8" font-weight="700" fill="{INDIGO}">108D GT state</text>
    <text x="646" y="808" font-size="7.3" font-weight="500" fill="{SUB}">375k critic</text>
    <text x="646" y="823" font-size="7.6" font-weight="700" fill="{INDIGO}">AdaBoot</text>
    <text x="646" y="834.5" font-size="7.0" font-weight="500" fill="{MUTED}">1 − tanh(CV)</text>

    <path d="M 600 846 L 600 858" fill="none" stroke="{INDIGO}" stroke-width="1.2" marker-end="url(#arrIndigo)"/>

    <rect x="556" y="861" width="176" height="90" rx="6.5" fill="{TEAL_SOFT}" stroke="{TEAL_BD}" stroke-width="0.9"/>
    <text x="644" y="876" text-anchor="middle" font-size="9.0" font-weight="700" fill="{TEAL}">
      Pure-Physics Min-CoT PPO
    </text>
    <text x="644" y="888.5" text-anchor="middle" font-size="7.6" font-weight="600" fill="{INK}">
      2× ℤ<tspan dy="1.4" font-size="5.8">2</tspan><tspan dy="-1.4"> mirror rollouts · zero templates</tspan>
    </text>
    <line x1="566" y1="895" x2="722" y2="895" stroke="{TEAL_BD}" stroke-width="0.8"/>
    <text x="566" y="910" font-size="7.8" font-weight="700" fill="{TEAL}">+ v<tspan dy="1.4" font-size="5.8">x</tspan><tspan dy="-1.4">, ω</tspan><tspan dy="1.4" font-size="5.8">z</tspan> track</text>
    <text x="650" y="910" font-size="7.8" font-weight="700" fill="{TERRA}">− I²R Joule CoT</text>
    <text x="566" y="925.5" font-size="7.8" font-weight="700" fill="{TERRA}">− Wall force F<tspan dy="1.4" font-size="5.8">n</tspan></text>
    <text x="650" y="925.5" font-size="7.8" font-weight="700" fill="{INDIGO}">− Wheel barrier</text>
    <text x="644" y="942.5" text-anchor="middle" font-size="7.4" font-weight="700" fill="{TEAL}">
      Emergent gait phase transition
    </text>

    <!-- Bottom privileged bus + unified loss line -->
    <line x1="32" y1="961" x2="730" y2="961" stroke="{RULE}" stroke-width="0.85"/>
    <text x="58" y="976" font-size="8.0" font-weight="600" fill="{INDIGO}">
      Privileged s<tspan dy="1.5" font-size="6.0">priv</tspan><tspan dy="-1.5"> (108D: v, pipe R, μ</tspan><tspan dy="1.5" font-size="6.0">rr</tspan><tspan dy="-1.5">, F</tspan><tspan dy="1.5" font-size="6.0">n</tspan><tspan dy="-1.5">, slip) &amp; r</tspan><tspan dy="1.5" font-size="6.0">t</tspan>
    </text>
    <text x="730" y="976" text-anchor="end" font-size="8.0" font-weight="700" fill="{TEAL}">
      ℒ = ℒ<tspan dy="1.5" font-size="5.8">PPO</tspan><tspan dy="-1.5"> + ℒ</tspan><tspan dy="1.5" font-size="5.8">est</tspan><tspan dy="-1.5"> + ℒ</tspan><tspan dy="1.5" font-size="5.8">rec</tspan><tspan dy="-1.5"> + βD</tspan><tspan dy="1.5" font-size="5.8">KL</tspan><tspan dy="-1.5"> + ℒ</tspan><tspan dy="1.5" font-size="5.8">HIM</tspan>
    </text>

    <text x="24" y="979" font-size="18.0" font-weight="700" fill="{INK}">d</text>
  </g>

  <!-- ===================================================================== -->
  <!-- PANEL e : EMERGENT CROSS-MEDIA PHASE TRANSITION (x: 758..1022)        -->
  <!-- ===================================================================== -->
  <g id="panel-e-emergent">
    <line x1="758" y1="738" x2="1022" y2="738" stroke="{TEAL}" stroke-width="1.6"/>
    <text x="762" y="758" font-size="11.0" font-weight="700" fill="{INK}">Emergent phase transition</text>
    <text x="762" y="770.5" font-size="7.8" font-weight="600" fill="{TEAL}">Single manifold · zero mode switch</text>

    <line x1="948" y1="767.5" x2="958" y2="767.5" stroke="{TEAL}" stroke-width="1.8" stroke-linecap="round"/>
    <text x="961" y="770.5" font-size="7.2" font-weight="600" fill="{TEAL}">yaw</text>
    <line x1="984" y1="767.5" x2="994" y2="767.5" stroke="{TERRA}" stroke-width="1.8" stroke-linecap="round"/>
    <text x="997" y="770.5" font-size="7.2" font-weight="600" fill="{TERRA}">ten.</text>

    <!-- Strip 1: Open Hard -> Serpenoid -->
    <text x="762" y="790" font-size="8.6" font-weight="700" fill="{INK}">Open Hard</text>
    <text x="814" y="790" font-size="8.0" font-weight="600" fill="{TEAL}">· Serpenoid</text>
    <line x1="762" y1="824" x2="828" y2="824" stroke="#94A3B8" stroke-width="1.0"/>
    <path d="M 766 813 C 776 801, 788 801, 796 812 C 804 823, 816 822, 824 811"
          fill="none" stroke="{TEAL}" stroke-width="2.5" stroke-linecap="round"/>
    <circle cx="766" cy="813" r="2.1" fill="{TEAL}"/>
    <circle cx="781" cy="804" r="1.9" fill="{TEAL}"/>
    <circle cx="796" cy="812" r="1.9" fill="{TEAL}"/>
    <circle cx="810" cy="820" r="1.9" fill="{TEAL}"/>
    <circle cx="824" cy="811" r="2.1" fill="{TEAL}"/>
    <path d="{e1_ten}" fill="none" stroke="{TERRA}" stroke-width="1.3" opacity="0.8"/>
    <path d="{e1_yaw}" fill="none" stroke="{TEAL}" stroke-width="1.85" stroke-linecap="round"/>
    <text x="1020" y="809" text-anchor="end" font-size="9.2" font-weight="700" fill="{TEAL}">CoT 1.06</text>
    <text x="1020" y="822" text-anchor="end" font-size="7.6" font-weight="600" fill="{SUB}">176 mm/s</text>

    <line x1="762" y1="838" x2="1020" y2="838" stroke="{RULE_LIGHT}" stroke-width="0.9"/>

    <!-- Strip 2: Ø236 Pipe -> Peristaltic -->
    <text x="762" y="855" font-size="8.6" font-weight="700" fill="{INK}">Ø236 Pipe</text>
    <text x="812" y="855" font-size="8.0" font-weight="600" fill="{INDIGO}">· Peristaltic</text>
    <rect x="762" y="864" width="66" height="24" rx="3" fill="#F1F6FA" stroke="#94A3B8" stroke-width="0.9"/>
    <line x1="766" y1="876" x2="824" y2="876" stroke="{TERRA}" stroke-width="1.9" stroke-linecap="round"/>
    <ellipse cx="781" cy="876" rx="6.2" ry="8.4" fill="{TERRA_SOFT}" stroke="{TERRA}" stroke-width="1.15"/>
    <ellipse cx="796" cy="876" rx="7.2" ry="4.2" fill="#FFFFFF" stroke="{TEAL}" stroke-width="0.95"/>
    <ellipse cx="811" cy="876" rx="6.2" ry="8.4" fill="{TERRA_SOFT}" stroke="{TERRA}" stroke-width="1.15"/>
    <path d="{e2_yaw}" fill="none" stroke="{TEAL}" stroke-width="1.2" opacity="0.6"/>
    <path d="{e2_ten}" fill="none" stroke="{TERRA}" stroke-width="1.9" stroke-linecap="round"/>
    <text x="1020" y="874" text-anchor="end" font-size="9.2" font-weight="700" fill="{INDIGO}">CoT 1.22</text>
    <text x="1020" y="887" text-anchor="end" font-size="7.4" font-weight="600" fill="{SUB}">151 mm/s · 0.4 N</text>

    <line x1="762" y1="902" x2="1020" y2="902" stroke="{RULE_LIGHT}" stroke-width="0.9"/>

    <!-- Strip 3: Soft Soil -> Hybrid -->
    <text x="762" y="919" font-size="8.6" font-weight="700" fill="{INK}">Soft Soil</text>
    <text x="804" y="919" font-size="8.0" font-weight="600" fill="{TERRA}">· Hybrid</text>
    <rect x="762" y="947" width="66" height="7.5" rx="2" fill="#F3E6D3"/>
    <path d="M 766 941 C 778 933, 790 946, 804 937 C 812 932, 818 941, 824 939"
          fill="none" stroke="{TEAL}" stroke-width="2.0" stroke-linecap="round"/>
    <ellipse cx="784" cy="939" rx="5.2" ry="6.8" fill="{TERRA_SOFT}" stroke="{TERRA}" stroke-width="1.1"/>
    <ellipse cx="808" cy="938" rx="5.2" ry="6.8" fill="{TERRA_SOFT}" stroke="{TERRA}" stroke-width="1.1"/>
    <path d="{e3_yaw}" fill="none" stroke="{TEAL}" stroke-width="1.5" stroke-linecap="round"/>
    <path d="{e3_ten}" fill="none" stroke="{TERRA}" stroke-width="1.7" stroke-linecap="round"/>
    <text x="1020" y="939" text-anchor="end" font-size="9.2" font-weight="700" fill="{TERRA}">CoT 8.29</text>
    <text x="1020" y="952" text-anchor="end" font-size="7.6" font-weight="600" fill="{SUB}">25 mm/s</text>

    <text x="762" y="979" font-size="18.0" font-weight="700" fill="{INK}">e</text>
  </g>

  <!-- ===================================================================== -->
  <!-- FOUR 100% STRAIGHT VERTICAL CROSS-PANEL FLOWS (c <-> d & c -> e)      -->
  <!-- ===================================================================== -->
  <path d="M 118 738 L 118 707" fill="none" stroke="{TERRA}" stroke-width="1.45"
        stroke-dasharray="4 3" marker-end="url(#arrTerra)"/>
  <rect x="75" y="717" width="86" height="17" fill="#FFFFFF"/>
  <text x="118" y="728.8" text-anchor="middle" font-size="8.8" font-weight="700" fill="{TERRA}">
    Sim O<tspan dy="1.6" font-size="6.6">t</tspan><tspan dy="-1.6"> + DR</tspan>
  </text>

  <path d="M 480 704 L 480 735" fill="none" stroke="{INDIGO}" stroke-width="1.45"
        stroke-dasharray="4 3" marker-end="url(#arrIndigo)"/>
  <rect x="443" y="717" width="74" height="17" fill="#FFFFFF"/>
  <text x="480" y="728.8" text-anchor="middle" font-size="8.8" font-weight="700" fill="{INDIGO}">
    e&#x0302;<tspan dy="1.6" font-size="6.6">t</tspan><tspan dy="-1.6">, z</tspan><tspan dy="1.6" font-size="6.6">t</tspan><tspan dy="-1.6">, a</tspan><tspan dy="1.6" font-size="6.6">t</tspan>
  </text>

  <path d="M 692 738 L 692 666" fill="none" stroke="{TEAL}" stroke-width="1.45"
        stroke-dasharray="4 3" marker-end="url(#arrTeal)"/>
  <rect x="660" y="717" width="64" height="17" fill="#FFFFFF"/>
  <text x="692" y="728.8" text-anchor="middle" font-size="8.8" font-weight="700" fill="{TEAL}">
    ∇<tspan dy="1.6" font-size="6.6">θ</tspan><tspan dy="-1.6">ℒ</tspan><tspan dy="1.6" font-size="6.2">PPO</tspan>
  </text>

  <path d="M 968 628 L 968 734" fill="none" stroke="{TERRA}" stroke-width="1.5" marker-end="url(#arrTerra)"/>
  <rect x="945" y="672" width="46" height="16" fill="#FFFFFF"/>
  <text x="968" y="683.0" text-anchor="middle" font-size="8.1" font-weight="700" fill="{TERRA}">gaits</text>
</svg>
'''
    return svg


def render_outputs(svg_text: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / 'fig_nc_algorithm_architecture.svg'
    pdf_path = OUT_DIR / 'fig_nc_algorithm_architecture.pdf'
    png_path = OUT_DIR / 'fig_nc_algorithm_architecture.png'
    html_tmp = OUT_DIR / '_fig_nc_arch_render.html'

    svg_path.write_text(svg_text, encoding='utf-8')

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  @page {{ size: {WIDTH_MM:.2f}mm {HEIGHT_MM:.2f}mm; margin: 0; }}
  html, body {{
    margin: 0; padding: 0;
    width: {W:.0f}px; height: {H:.0f}px;
    background: #FFFFFF;
    overflow: hidden;
  }}
  svg {{
    display: block;
    width: {W:.0f}px;
    height: {H:.0f}px;
  }}
</style>
</head>
<body>
{svg_text}
</body>
</html>'''
    html_tmp.write_text(html, encoding='utf-8')

    from playwright.sync_api import sync_playwright

    # Exact 600 DPI at 183 mm = 4322.8 px -> 4.16x crisp oversampling
    scale = 4.16
    with sync_playwright() as p:
        browser = None
        for ch in ('msedge', 'chrome', None):
            try:
                browser = p.chromium.launch(channel=ch) if ch else p.chromium.launch()
                break
            except Exception:
                continue
        if browser is None:
            raise RuntimeError('No Chromium/Edge browser available for Playwright rendering')

        ctx = browser.new_context(
            viewport={'width': int(W), 'height': int(H)},
            device_scale_factor=scale,
        )
        page = ctx.new_page()
        page.goto(html_tmp.resolve().as_uri(), wait_until='networkidle')
        page.wait_for_timeout(150)

        raw_png = OUT_DIR / '_fig_nc_arch_raw.png'
        page.screenshot(path=str(raw_png), clip={'x': 0, 'y': 0, 'width': W, 'height': H})

        page.emulate_media(media='screen')
        page.pdf(
            path=str(pdf_path),
            width=f'{WIDTH_MM:.2f}mm',
            height=f'{HEIGHT_MM:.2f}mm',
            print_background=True,
            margin={'top': '0mm', 'right': '0mm', 'bottom': '0mm', 'left': '0mm'},
            scale=float((WIDTH_MM / 25.4 * 96.0) / W),
        )
        browser.close()

    im = Image.open(raw_png).convert('RGB')
    im.save(png_path, format='PNG', dpi=(600, 600), optimize=True)
    raw_png.unlink(missing_ok=True)
    html_tmp.unlink(missing_ok=True)

    if ARTIFACT_DIR is not None and ARTIFACT_DIR.is_dir():
        shutil.copy2(png_path, ARTIFACT_DIR / 'fig_nc_algorithm_architecture.png')

    print(f'Rendered PNG: {png_path}  ({im.size[0]}x{im.size[1]} px @ 600 DPI)')
    print(f'Rendered PDF: {pdf_path}  ({pdf_path.stat().st_size:,} bytes)')
    print(f'Rendered SVG: {svg_path}  ({svg_path.stat().st_size:,} bytes)')


def main() -> None:
    svg_text = build_svg()
    render_outputs(svg_text)


if __name__ == '__main__':
    main()

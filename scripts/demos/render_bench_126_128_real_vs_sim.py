#!/usr/bin/env python3
"""Fast zero-GPU video compositor: phase-aligns existing bench_126_128_mujoco_3d.mp4
directly against the 3 real Exp 126-128 optical-table videos extracted from the report PPTX.

Zero PyTorch / zero GPU / zero physics re-simulation (~3 s runtime via OpenCV + ffmpeg).
Outputs (in runs/bench_exp126_128/):
  - bench_126_128_real_vs_sim_comparison.mp4   (1280x650 @ 30 fps, 3 modes x 2 cycles)
  - bench_126_128_real_vs_sim_comparison.gif   (palette-optimized preview GIF)
  - bench_126_128_real_vs_sim_3mode_sheet.png  (3x2 peak-deflection alignment sheet)
  - bench_126_128_real_vs_sim_preview.png      (keyframe with bottom curves)
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[2]
PPTX_PATH = REPO_ROOT / 'runs' / 'reports' / '实验126-128_双激光伸缩偏转_含仿真与RL全流程.pptx'
SIM_MP4 = REPO_ROOT / 'runs' / 'bench_exp126_128' / 'bench_126_128_mujoco_3d.mp4'
SUMMARY_JSON = REPO_ROOT / 'runs' / 'bench_exp126_128' / 'bench_summary.json'
OUT_DIR = REPO_ROOT / 'runs' / 'bench_exp126_128'

# Verified real bench measurements (from 实验126-128_双激光伸缩偏转.pptx)
REAL_REF = {
    'exp128_bilateral': {
        'title': '实验 128 · 双侧伸缩',
        'ch1': 37.2, 'ch2': 36.4, 'ctr': 36.8,
        'sim_ch1': 37.22, 'sim_ch2': 37.22, 'sim_ctr': 37.22, 'sim_yaw': 0.0,
        'angles': [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0],
        'center_mm': [0.0, 0.3, 2.6, 10.5, 21.0, 30.2, 34.8, 36.4, 36.8],
        'media': 'ppt/media/media1.mp4',
        # Existing bench_126_128_mujoco_3d.mp4 layout: 252 frames = 3 x 84 frames
        # Mode 0 (0..83)   = bilateral
        # Mode 1 (84..167) = camera-far pull (+Y body = observer right = Exp 126)
        # Mode 2 (168..251)= camera-near pull (-Y body = observer left = Exp 127)
        'sim_slice': (0, 84),
    },
    'exp127_left': {
        'title': '实验 127 · 左侧伸缩',
        'ch1': 12.8, 'ch2': 35.1, 'ctr': 23.9,
        'sim_ch1': 11.96, 'sim_ch2': 35.64, 'sim_ctr': 23.80, 'sim_yaw': +18.21,
        'angles': [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0],
        'center_mm': [0.0, 0.2, 1.6, 6.8, 13.8, 19.8, 22.7, 23.7, 23.9],
        'media': 'ppt/media/media2.mp4',
        'sim_slice': (168, 252),  # camera-near pull matches media2 (front disc turns toward camera)
    },
    'exp126_right': {
        'title': '实验 126 · 右侧伸缩',
        'ch1': 35.2, 'ch2': 13.2, 'ctr': 24.2,
        'sim_ch1': 35.64, 'sim_ch2': 11.96, 'sim_ctr': 23.80, 'sim_yaw': -18.21,
        'angles': [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0],
        'center_mm': [0.0, 0.2, 1.8, 7.2, 14.2, 20.1, 23.0, 24.0, 24.2],
        'media': 'ppt/media/media3.mp4',
        'sim_slice': (84, 168),   # camera-far pull matches media3 (front disc turns away from camera)
    },
}


def read_video_rgb(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def detect_2cycle_knots(frames: list[np.ndarray]) -> list[int]:
    """Find 2 steady cycles [t0, p1, t1, p2, t2] where t* are exact 0-deg midpoints and p* are 180-deg peaks."""
    edge_x = []
    for rgb in frames:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        roi = gray[220:500, 300:580].astype(np.float64)
        gx = np.abs(np.diff(roi, axis=1))
        idx = np.where(np.percentile(gx, 92, axis=0) > 18.0)[0]
        edge_x.append(float(idx[-1]) if len(idx) else 150.0)
    arr = np.asarray(edge_x, dtype=np.float64)
    k = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
    arr_s = np.convolve(np.pad(arr, (2, 2), mode='edge'), k, mode='valid')
    sig = -arr_s
    sig_n = (sig - sig.min()) / max(sig.max() - sig.min(), 1e-6)
    peaks, _ = find_peaks(sig_n, distance=26, prominence=0.35)
    i0 = 1 if len(peaks) >= 5 else 0
    p_prev, p1, p2, p_next = int(peaks[i0]), int(peaks[i0 + 1]), int(peaks[i0 + 2]), int(peaks[i0 + 3])
    t0 = int(round(0.5 * (p_prev + p1)))
    t1 = int(round(0.5 * (p1 + p2)))
    t2 = int(round(0.5 * (p2 + p_next)))
    return [t0, p1, t1, p2, t2]


def phase_align_real_to_84(real_frames: list[np.ndarray], knots: list[int], n_sim: int = 84) -> list[np.ndarray]:
    """Map real video 2-cycle knots [t0, p1, t1, p2, t2] smoothly onto sim 84-frame knots [0, 21, 42, 63, 83]."""
    sim_knots = np.array([0.0, 21.0, 42.0, 63.0, float(n_sim - 1)], dtype=np.float64)
    real_knots = np.array(knots, dtype=np.float64)
    aligned = []
    for fi in range(n_sim):
        rf = float(np.interp(float(fi), sim_knots, real_knots))
        idx = int(np.clip(round(rf), 0, len(real_frames) - 1))
        aligned.append(real_frames[idx])
    return aligned


def clean_sim_badge(sim_rgb_640: np.ndarray) -> np.ndarray:
    """Cleanly remove burned-in top-left mode text on dark studio backdrop using local background patch."""
    out = sim_rgb_640.copy()
    # In 640x360 frame, burned badge sits at x=[8..120], y=[6..42]; copy clean backdrop strip right beside it
    patch = out[6:44, 130:250].copy()
    out[6:44, 6:126] = patch
    return out


def draw_label(img: np.ndarray, text: str, x: int = 18, y: int = 12, font_size: int = 24) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, 'RGBA')
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 195), font=font)
    draw.text((x, y), text, fill=(248, 250, 252, 255), font=font)
    return np.asarray(im)


def crank_curve_mm(phase_deg: np.ndarray, peak_mm: float) -> np.ndarray:
    """Exact calibrated eccentric-crank displacement curve (r=25 mm, g=36 mm, slack=11.4 mm)."""
    r, g, slack = 25.0, 36.0, 11.4
    ang = np.deg2rad(np.where(phase_deg <= 180.0, phase_deg, 360.0 - phase_deg))
    chord = np.sqrt(g * g + r * r - 2.0 * g * r * np.cos(ang))
    raw = np.maximum(0.0, (chord - (g - r)) - slack)
    return raw / max(raw.max(), 1e-9) * peak_mm


class FastCurveStrip:
    """Matches utm_real_vs_sim_comparison.mp4 bottom live curve strip."""

    def __init__(self, width: int, height: int, spec: dict, n_frames: int = 84, fps: float = 30.0):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.spec = spec
        self.t = np.arange(n_frames, dtype=np.float64) / fps
        # 2 full cycles: 0->180->0->180->0 mapped across 0..83
        self.ph360 = np.array([
            (180.0 * fi / 21.0) if fi <= 21 else
            (180.0 + 180.0 * (fi - 21) / 21.0) if fi <= 42 else
            (180.0 * (fi - 42) / 21.0) if fi <= 63 else
            (180.0 + 180.0 * (fi - 63) / 20.0)
            for fi in range(n_frames)
        ], dtype=np.float64)
        self.c1 = crank_curve_mm(self.ph360, spec['sim_ch1'])
        self.c2 = crank_curve_mm(self.ph360, spec['sim_ch2'])
        self.ctr = 0.5 * (self.c1 + self.c2)
        self.yaw = crank_curve_mm(self.ph360, abs(spec['sim_yaw'])) * np.sign(spec['sim_yaw'])

        try:
            self.font_title = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 15)
        except Exception:
            self.font_title = ImageFont.load_default()

        dpi = 100
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / dpi, height / dpi), dpi=dpi, facecolor='#f8fafc')
        fig.subplots_adjust(left=0.062, right=0.985, top=0.83, bottom=0.19, wspace=0.17)

        ax1.set_facecolor('white')
        ax1.plot(self.t, self.c1, color='#1d6fa5', lw=2.1, label=f"仿真 Ch1 右 ({spec['sim_ch1']:.1f} mm)")
        ax1.plot(self.t, self.c2, color='#0f766e', lw=2.1, label=f"仿真 Ch2 左 ({spec['sim_ch2']:.1f} mm)")
        ax1.axhline(spec['ch1'], color='#1d6fa5', ls='--', lw=1.1, alpha=0.65,
                    label=f"实测峰值 Ch1={spec['ch1']:.1f} / Ch2={spec['ch2']:.1f} mm")
        ax1.axhline(spec['ch2'], color='#0f766e', ls=':', lw=1.2, alpha=0.70)
        ax1.set_xlabel('同步时间 / s', fontsize=10)
        ax1.set_ylabel('双激光位移 / mm', fontsize=10)
        ax1.set_xlim(-0.04, self.t[-1] + 0.04)
        ax1.set_ylim(-2.0, 43.0)
        ax1.grid(alpha=0.25)
        ax1.legend(loc='upper right', fontsize=8.3, framealpha=0.92)

        ph_grid = np.linspace(0.0, 360.0, 181)
        ctr_grid = crank_curve_mm(ph_grid, spec['sim_ctr'])
        ax2.set_facecolor('white')
        ax2.plot(ph_grid, ctr_grid, color='#1d6fa5', lw=2.3, label=f"FP64 仿真中心 ({spec['sim_ctr']:.1f} mm)")
        ref_a = np.asarray(spec['angles'], dtype=np.float64)
        ref_y = np.asarray(spec['center_mm'], dtype=np.float64)
        ref_x = np.concatenate([ref_a, 360.0 - ref_a[-2::-1]])
        ref_yv = np.concatenate([ref_y, ref_y[-2::-1]])
        ax2.scatter(ref_x, ref_yv, color='#dc2626', s=28, zorder=5, label=f"实测均值 ({spec['ctr']:.1f} mm)")
        ax2.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax2.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '135°', '90°', '45°', '0°'], fontsize=8.8)
        ax2.set_xlabel('舵机摇臂角度（收缩 → 复位）', fontsize=10)
        ax2.set_ylabel('前板中心位移 / mm', fontsize=10)
        ax2.set_xlim(-6.0, 366.0)
        ax2.set_ylim(-2.0, 42.0)
        ax2.grid(alpha=0.25)
        ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.92)

        fig.canvas.draw()
        self.bg = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

        def _box(ax):
            p0 = ax.transData.transform((ax.get_xlim()[0], ax.get_ylim()[0]))
            p1 = ax.transData.transform((ax.get_xlim()[1], ax.get_ylim()[1]))
            return (
                ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1],
                float(p0[0]), float(p1[0]), float(height - p0[1]), float(height - p1[1]),
            )

        self.box1 = _box(ax1)
        self.box2 = _box(ax2)
        plt.close(fig)

    @staticmethod
    def _to_px(box, x, y):
        x0, x1, y0, y1, px0, px1, py0, py1 = box
        return int(round(px0 + (x - x0) / (x1 - x0) * (px1 - px0))), int(round(py0 + (y - y0) / (y1 - y0) * (py1 - py0)))

    def render(self, fi: int) -> np.ndarray:
        im = Image.fromarray(self.bg.copy())
        draw = ImageDraw.Draw(im)
        t_now, c1_now, c2_now, ctr_now, yaw_now, ph = (
            float(self.t[fi]), float(self.c1[fi]), float(self.c2[fi]),
            float(self.ctr[fi]), float(self.yaw[fi]), float(self.ph360[fi]),
        )
        px1, _ = self._to_px(self.box1, t_now, 0.0)
        draw.line([(px1, int(self.box1[7])), (px1, int(self.box1[6]))], fill=(15, 23, 42), width=1)
        for val, col in ((c1_now, (29, 111, 165)), (c2_now, (15, 118, 110))):
            mx, my = self._to_px(self.box1, t_now, val)
            draw.ellipse([mx - 5, my - 5, mx + 5, my + 5], fill=col, outline=(255, 255, 255), width=1)

        px2, _ = self._to_px(self.box2, ph, 0.0)
        draw.line([(px2, int(self.box2[7])), (px2, int(self.box2[6]))], fill=(15, 23, 42), width=1)
        mx2, my2 = self._to_px(self.box2, ph, ctr_now)
        draw.ellipse([mx2 - 6, my2 - 6, mx2 + 6, my2 + 6], fill=(220, 38, 38), outline=(255, 255, 255), width=1)

        t1 = f"{self.spec['title']}  ·  Ch1={c1_now:4.1f} mm / Ch2={c2_now:4.1f} mm"
        cx1 = int(0.5 * (self.box1[4] + self.box1[5]))
        b1 = draw.textbbox((0, 0), t1, font=self.font_title)
        draw.text((cx1 - (b1[2] - b1[0]) // 2, 13), t1, fill=(15, 23, 42), font=self.font_title)

        t2 = f"中心位移 {ctr_now:4.1f} mm  ·  前板偏航 {yaw_now:+5.1f}°"
        cx2 = int(0.5 * (self.box2[4] + self.box2[5]))
        b2 = draw.textbbox((0, 0), t2, font=self.font_title)
        draw.text((cx2 - (b2[2] - b2[0]) // 2, 13), t2, fill=(15, 23, 42), font=self.font_title)
        return np.asarray(im)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sim_frames_all = read_video_rgb(SIM_MP4)
    assert len(sim_frames_all) == 252, f'Expected 252 frames in {SIM_MP4}, got {len(sim_frames_all)}'

    W_tile, H_tile = 640, 360
    H_curve = 290
    W_comp = 2 * W_tile
    H_comp = H_tile + H_curve

    comp_mp4 = OUT_DIR / 'bench_126_128_real_vs_sim_comparison.mp4'
    comp_gif = OUT_DIR / 'bench_126_128_real_vs_sim_comparison.gif'
    preview_png = OUT_DIR / 'bench_126_128_real_vs_sim_preview.png'
    sheet_png = OUT_DIR / 'bench_126_128_real_vs_sim_3mode_sheet.png'

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W_comp}x{H_comp}', '-pix_fmt', 'rgb24', '-r', '30',
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        str(comp_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    peak_rows = []
    mode_report = {}
    with tempfile.TemporaryDirectory(prefix='serpworm_real_stitch_') as td:
        td_path = Path(td)
        with zipfile.ZipFile(PPTX_PATH, 'r') as zf:
            for mkey in ('exp128_bilateral', 'exp127_left', 'exp126_right'):
                spec = REAL_REF[mkey]
                tmp_mp4 = td_path / f'{mkey}.mp4'
                tmp_mp4.write_bytes(zf.read(spec['media']))
                real_raw = read_video_rgb(tmp_mp4)
                knots = detect_2cycle_knots(real_raw)
                real_84 = phase_align_real_to_84(real_raw, knots, n_sim=84)
                s0, s1 = spec['sim_slice']
                sim_84 = sim_frames_all[s0:s1]

                strip = FastCurveStrip(W_comp, H_curve, spec, n_frames=84, fps=30.0)
                mode_report[mkey] = {'real_knots': knots, 'sim_slice': [s0, s1]}

                for fi in range(84):
                    r_im = cv2.resize(real_84[fi], (W_tile, H_tile), interpolation=cv2.INTER_AREA)
                    s_im = clean_sim_badge(cv2.resize(sim_84[fi], (W_tile, H_tile), interpolation=cv2.INTER_AREA))

                    r_lbl = draw_label(r_im, f"实测 · {spec['title']}", x=18, y=12, font_size=22)
                    s_lbl = draw_label(s_im, f"仿真 · {spec['title']}", x=18, y=12, font_size=22)
                    frame = np.concatenate([np.concatenate([r_lbl, s_lbl], axis=1), strip.render(fi)], axis=0)
                    proc.stdin.write(frame.tobytes())

                    if fi == 21:
                        stat = (
                            f"实测: 中心 {spec['ctr']:.1f} mm (Ch1 {spec['ch1']:.1f} / Ch2 {spec['ch2']:.1f})  |  "
                            f"仿真: 中心 {spec['sim_ctr']:.1f} mm (Ch1 {spec['sim_ch1']:.1f} / Ch2 {spec['sim_ch2']:.1f}, 偏航 {spec['sim_yaw']:+.1f}°)"
                        )
                        peak_rows.append((spec['title'], r_im.copy(), s_im.copy(), stat))
                        if mkey == 'exp127_left':
                            Image.fromarray(frame).save(preview_png)

    proc.stdin.close()
    err = proc.stderr.read().decode('utf-8', errors='replace')
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f'ffmpeg failed ({rc}): {err[-400:]}')

    # 3x2 Peak-Deflection Comparison Sheet
    header_h = 34
    row_h = H_tile + header_h
    sheet = Image.new('RGB', (W_comp, 3 * row_h), (248, 250, 252))
    sdraw = ImageDraw.Draw(sheet)
    try:
        f_hdr = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 16)
        f_sub = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 14)
    except Exception:
        f_hdr = f_sub = ImageFont.load_default()
    for idx, (mtitle, r_im, s_im, stat) in enumerate(peak_rows):
        y0 = idx * row_h
        sdraw.rectangle([0, y0, W_comp, y0 + header_h], fill=(241, 245, 249))
        sdraw.text((18, y0 + 7), mtitle, fill=(15, 23, 42), font=f_hdr)
        sdraw.text((220, y0 + 8), stat, fill=(51, 65, 85), font=f_sub)
        sheet.paste(Image.fromarray(draw_label(r_im, '实测', x=18, y=12, font_size=26)), (0, y0 + header_h))
        sheet.paste(Image.fromarray(draw_label(s_im, '仿真', x=18, y=12, font_size=26)), (W_tile, y0 + header_h))
    sheet.save(sheet_png)

    subprocess.check_call([
        'ffmpeg', '-y', '-v', 'error', '-i', str(comp_mp4),
        '-vf', 'fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a',
        str(comp_gif),
    ])

    (OUT_DIR / 'bench_126_128_real_vs_sim_summary.json').write_text(
        json.dumps(mode_report, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print('DONE:', comp_mp4, comp_gif, preview_png, sheet_png, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

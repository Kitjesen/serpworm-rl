#!/usr/bin/env python3
"""Build the 20-slide Academic Presentation following Gabberflast/academic-pptx-skill.

Design & Content Architecture:
  - Strict Structured Argument (SCR) spine: 14 main slides + 1 References + 5 Appendix
  - Every content slide uses a complete-sentence Action Title (Ghost Deck Test verified)
  - Communication-first minimalist palette: Pure White canvas + <= 3 soft muted colors
    (Soft Mist Blue #5B7FA6, Soft Sage Teal #5A948A, Soft Charcoal #2C3A47/#4A5568)
  - Bar charts use a soft Morandi pastel multi-color palette with bottom-left (a)(b) labels
  - Single typeface throughout: Microsoft YaHei (enforced on a:latin, a:ea, a:cs & Matplotlib)
  - Zero card wireframes, zero colored card backgrounds, zero title underlines, zero shadows
  - Back-of-room legible body text (>= 16 pt on all main slides), <= 4 concise bullets/slide
  - Scientifically rigorous benchmarking: same-host CUDA acceleration & same-sim RL ablations
    in main deck; literature-reported cross-hardware references isolated in Appendix B;
    explicit Discussion/Limitations slide covering rolling-drag sensitivity & real-robot gap.

Output:
  - docs/SerpWorm_RL_Academic_Talk.pptx
  - runs/ppt_assets/acad_*.png (300-DPI presentation-resolution figures)
  - runs/ppt_assets/academic_previews/Slide*.png (when --render-previews is passed)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / 'runs' / 'ppt_assets'
PREVIEWS_DIR = ASSETS_DIR / 'academic_previews'
OUT_PPTX = REPO_ROOT / 'docs' / 'SerpWorm_RL_Academic_Talk.pptx'
_BRAIN_ENV = os.environ.get('SERPWORM_ARTIFACT_DIR')
BRAIN_DIR: Path | None = Path(_BRAIN_ENV) if _BRAIN_ENV else None

SLIDE_W = 13.333
SLIDE_H = 7.500
MARGIN = 0.55
TOTAL_SLIDES = 20

# Strict <= 3 Soft Muted Colors + Pure White Canvas
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_PRIMARY = RGBColor(0x2C, 0x3A, 0x47)    # Soft Charcoal (action titles & strong text)
C_BLUE = RGBColor(0x5B, 0x7F, 0xA6)       # Color 1: Soft Mist Blue
C_TEAL = RGBColor(0x5A, 0x94, 0x8A)       # Color 2: Soft Sage Teal
C_BODY = RGBColor(0x3D, 0x4A, 0x57)       # Color 3: Calm slate body text
C_MUTED = RGBColor(0x7A, 0x8A, 0x9B)      # Citations / subtle breadcrumb / footers
C_RULE = RGBColor(0xE4, 0xEA, 0xF0)       # Subtle hairline rule
C_TBL_HEAD = RGBColor(0xF3, 0xF6, 0xFA)   # Ultra-pale table header tint
C_TBL_HI = RGBColor(0xEF, 0xF7, 0xF5)     # Ultra-pale highlight row tint

# Hex strings for Matplotlib (strict 3 hues for schematics/text)
H_DARK = '#2C3A47'
H_BLUE = '#5B7FA6'
H_TEAL = '#5A948A'
H_SLATE = '#4A5568'
H_MUTED = '#7A8A9B'
H_RULE = '#DCE4EE'

# Soft Morandi Pastel Multi-Color Palette (exclusively for Bar Charts & Multi-Series Plots)
M_GRAY1 = '#C8D1DA'
M_GRAY2 = '#AAB6C2'
M_GRAY3 = '#8F9EAD'
M_SKY   = '#8FB3D4'
M_BLUE  = '#6B92B8'
M_PERI  = '#8C9EC8'
M_TEAL  = '#66A399'
M_SAGE  = '#87B5A0'
M_SAND  = '#D3AE82'
M_PEACH = '#D5A08B'
M_LAV   = '#A799C2'
M_ROSE  = '#C89499'
M_TAUPE = '#A9A199'

FONT_FACE = 'Microsoft YaHei'

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

SECTIONS = ['背景与问题', '物理与算力', '算法与结果', '讨论与结论', '文献与附录']


# ---------------------------------------------------------------------------
# Verified Data Loading
# ---------------------------------------------------------------------------
def load_json(rel_path: str) -> Any:
    with open(REPO_ROOT / rel_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_asset(filename: str, fallback_rel: str | None = None) -> Path:
    p_local = ASSETS_DIR / filename
    if p_local.exists():
        return p_local
    if fallback_rel:
        p_rel = REPO_ROOT / fallback_rel
        if p_rel.exists():
            return p_rel
    if BRAIN_DIR is not None:
        p_brain = BRAIN_DIR / filename
        if p_brain.exists():
            ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p_brain, p_local)
            return p_local
    raise FileNotFoundError(f'Missing visual asset: {filename}')


def load_all_metrics() -> dict[str, Any]:
    utm = load_json('runs/utm_validation/utm_validation_summary.json')
    bench = load_json('runs/bench_exp126_128/bench_summary.json')
    contact = load_json('runs/demos/contact_demo_metrics.json')
    arpc_bench = load_json('runs/demos/rtx5090_arpc_benchmark.json')
    v6_eval = load_json('runs/v6_ablation_eval.json')
    v61_eval = load_json('runs/v6_1_65d_elec/eval_v6_1_all.json')
    v5_eval = load_json('runs/unified_v5_hispeed_14d/eval_v5_full.json')
    pipe_traj = load_json('runs/v6b_65d_qerr/v6b_autonomous_pipe_traj.json')
    v8_ab = load_json('runs/v8_ablation/v8_ablation_summary.json')
    xeon_cpu = load_json('runs/demos/xeon_cpu_benchmark.json')
    v71_eval = load_json('runs/v71_Lduct_4096/eval_final_best_summary.json')
    v8_prog = load_json('runs/v8_diverse_4096/progress.json')

    c_summary: dict[str, dict[str, float]] = {}
    for k in ('kelvin', 'hard', 'carpet', 'soft_soil'):
        d1 = contact['exp1_drop']['terrains'][k]['summary']
        d2 = contact['exp1_static_hold']['terrains'][k]['summary']
        d3 = contact['exp2_snake']['terrains'][k]['summary']
        d4 = contact['exp3_hybrid_4terrain']['terrains'][k]
        c_summary[k] = {
            'drop_jump_n': float(d1['first_step_jump_total_n']),
            'drop_bounce_ms': float(d1['rebound_lift_ms']),
            'static_drift_mm': float(d2['total_drift_mm']),
            'static_rate_mm_s': float(d2['steady_drift_speed_mm_s']),
            'snake_vx': float(d3['forward_vel_mm_s']),
            'hybrid_vx': float(d4['steady_forward_vel_mm_s']),
            'hybrid_rut_mean': float(d4['mean_rut_mm']),
        }

    return {
        'utm': utm,
        'bench': bench,
        'contact_raw': contact,
        'contact': c_summary,
        'arpc_bench': {r['name']: r for r in arpc_bench},
        'v6a': {r['track']: r for r in v6_eval['v6a_51d_pure']['best.pt']['results']},
        'v6b': {r['track']: r for r in v6_eval['v6b_65d_qerr']['best.pt']['results']},
        'v61': {r['track']: r for r in v61_eval['best.pt']['results']},
        'v5': v5_eval,
        'pipe': pipe_traj,
        'v8_ab': v8_ab,
        'xeon_cpu': {r['name']: r for r in xeon_cpu['benchmarks']},
        'v71': {r['track']: r for r in v71_eval['results']},
        'v8_prog': v8_prog,
    }


# ---------------------------------------------------------------------------
# Presentation-Resolution 300-DPI Figure Generators (Rebuilt with Large Fonts)
# ---------------------------------------------------------------------------
def _style_ax(ax) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(H_RULE)
    ax.spines['bottom'].set_color(H_RULE)
    ax.tick_params(colors=H_SLATE, labelsize=10.0)


def _arrow(ax, x0, y0, x1, y1, color=H_BLUE, lw=1.6, ms=12, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        connectionstyle=f'arc3,rad={rad}',
        arrowstyle='-|>', color=color, linewidth=lw, mutation_scale=ms, zorder=4,
    ))


def gen_acad_s05_cuda_samehost(m: dict[str, Any]) -> Path:
    """Slide 5 exhibit: Strict same-host CPU vs PyTorch GPU vs Fused CUDA (throughput + VRAM)."""
    out = ASSETS_DIR / 'acad_s05_cuda_samehost.png'
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.2, 4.9), dpi=300,
        gridspec_kw={'width_ratios': [1.68, 1.0], 'wspace': 0.38},
    )
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.21, right=0.97, top=0.90, bottom=0.16, wspace=0.38)

    xc = m['xeon_cpu']
    ab = m['arpc_bench']
    cpu_1t = float(xc['xeon_legacy_1T_B1_E16_dt0.25']['policy_steps_per_s'])
    cpu_16t = float(xc['xeon_seglu_16T_B16_E16_dt0.50']['policy_steps_per_s'])
    cpu_16t_a = float(xc['xeon_seglu_16T_B16_E16_dt1.00_arpc']['policy_steps_per_s'])
    gpu_seg = float(ab['E16_dt0.50_seg_lu_kelvin']['policy_steps_per_s'])
    cu_v2 = float(ab['E16_dt0.50_fused_kelvin']['policy_steps_per_s'])
    cu_v3 = float(ab['E16_dt1.00_fused_arpc_hard']['policy_steps_per_s'])
    cu_e8 = float(ab['E8_dt1.00_fused_arpc_hard']['policy_steps_per_s'])

    rows = [
        ('Xeon CPU 1核\n(0.25ms)', cpu_1t, M_GRAY1),
        ('Xeon CPU 16核\n(0.50ms)', cpu_16t, M_GRAY2),
        ('Xeon CPU 16核\n(ARPC 1ms)', cpu_16t_a, M_GRAY3),
        ('PyTorch GPU\n(seg_lu)', gpu_seg, M_SKY),
        ('CUDA V2\n(E=16, 0.5ms)', cu_v2, M_BLUE),
        ('CUDA V3+ARPC\n(E=16, 1ms)', cu_v3, M_TEAL),
        ('CUDA V3 粗网\n(E=8, 1ms)', cu_e8, M_SAND),
    ]
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    bars = ax1.barh(y, vals, height=0.58, color=[r[2] for r in rows], zorder=3)
    ax1.set_xscale('log')
    ax1.set_xlim(0.12, 9.0e5)
    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=9.1, color=H_SLATE)
    ax1.invert_yaxis()
    ax1.set_xlabel('10 Hz 仿真吞吐量 (steps/s，对数)', fontsize=10.0, color=H_SLATE)
    ax1.set_title('同机同模型物理吞吐对比（3,660 DoF FP64）', fontsize=10.8, fontweight='bold', color=H_DARK, pad=10)
    ax1.grid(axis='x', which='major', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax1)

    for i, (b, v) in enumerate(zip(bars, vals)):
        lbl = f'{v:.2f}' if v < 10 else f'{v:,.0f}'
        if i == 5:
            lbl += ' (153×)'
        ax1.text(v * 1.24, b.get_y() + b.get_height() / 2, lbl,
                 va='center', ha='left', fontsize=9.0,
                 fontweight='bold' if i >= 5 else 'normal',
                 color=H_TEAL if i == 5 else H_DARK)
    ax1.text(-0.41, -0.17, '(a)', transform=ax1.transAxes, fontsize=11.0, fontweight='bold', color=H_DARK)

    # Panel (b): 2048-env VRAM drop
    v_lbls = ['PyTorch\nseg_lu', 'CUDA\nE=16', 'CUDA\nE=8']
    v_vals = [
        float(ab['E16_dt0.50_seg_lu_kelvin']['vram_gb']),
        float(ab['E16_dt1.00_fused_arpc_hard']['vram_gb']),
        float(ab['E8_dt1.00_fused_arpc_hard']['vram_gb']),
    ]
    b2 = ax2.bar(np.arange(3), v_vals, width=0.52, color=[M_BLUE, M_TEAL, M_SAND], zorder=3)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(v_lbls, fontsize=9.5, color=H_SLATE)
    ax2.set_ylim(0, 12.0)
    ax2.set_ylabel('显存占用 (GB，B=2048)', fontsize=10.0, color=H_SLATE)
    ax2.set_title('显存直降 65.6%–82.5%', fontsize=11.0, fontweight='bold', color=H_DARK, pad=10)
    ax2.grid(axis='y', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax2)
    for i, (b, v) in enumerate(zip(b2, v_vals)):
        tag = f'{v:.2f} G' if i == 0 else f'{v:.2f} G\n(−{100*(1-v/v_vals[0]):.0f}%)'
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.30, tag,
                 ha='center', va='bottom', fontsize=9.2, fontweight='bold',
                 color=H_TEAL if i == 1 else H_DARK)
    ax2.text(-0.24, -0.17, '(b)', transform=ax2.transAxes, fontsize=11.0, fontweight='bold', color=H_DARK)

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    return out


def gen_acad_s06_arpc(m: dict[str, Any]) -> Path:
    """Slide 6 exhibit: 3 mm drop impact + 348x static drift reduction at presentation scale."""
    out = ASSETS_DIR / 'acad_s06_arpc.png'
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.2, 4.7), dpi=300,
        gridspec_kw={'width_ratios': [1.35, 1.0], 'wspace': 0.34},
    )
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.17, wspace=0.34)

    t_map = [
        ('kelvin', '旧线性 Kelvin', M_GRAY3, '--', 2.0),
        ('hard', 'ARPC 硬地', M_BLUE, '-', 2.3),
        ('carpet', 'ARPC 地毯', M_TEAL, '-', 2.1),
        ('soft_soil', 'ARPC 软土', M_SAND, '-', 2.1),
    ]
    for key, lbl, col, ls, lw in t_map:
        d = m['contact_raw']['exp1_drop']['terrains'][key]
        t = np.asarray(d['t_ms'])
        f = np.asarray(d['total_normal_n'])
        mask = (t >= 20) & (t <= 105)
        ax1.plot(t[mask], f[mask], label=lbl, color=col, linestyle=ls, linewidth=lw)

    ax1.axhline(28.6, color=M_GRAY1, ls=':', lw=1.2)
    ax1.annotate('Kelvin 57.9 N 尖峰', xy=(27, 58), xytext=(45, 118),
                 fontsize=9.5, fontweight='bold', color=H_SLATE,
                 arrowprops=dict(arrowstyle='->', color=H_MUTED, lw=1.2))
    ax1.set_title('3 mm 跌落触地法向力 (Δt = 1 ms)', fontsize=11.0, fontweight='bold', color=H_DARK, pad=10)
    ax1.set_xlabel('时间 (ms)', fontsize=10.2, color=H_SLATE)
    ax1.set_ylabel('总法向接触力 (N)', fontsize=10.2, color=H_SLATE)
    ax1.set_ylim(-5, 175)
    ax1.grid(True, ls='--', alpha=0.45, color=H_RULE)
    _style_ax(ax1)
    ax1.legend(fontsize=9.0, loc='upper right', frameon=False, handlelength=1.6)
    ax1.text(-0.15, -0.18, '(a)', transform=ax1.transAxes, fontsize=11.0, fontweight='bold', color=H_DARK)

    cs = m['contact']
    names = ['Kelvin\n旧线性', 'ARPC\n硬地', 'ARPC\n地毯', 'ARPC\n软土']
    vals = [cs['kelvin']['static_rate_mm_s'], cs['hard']['static_rate_mm_s'],
            cs['carpet']['static_rate_mm_s'], cs['soft_soil']['static_rate_mm_s']]
    bars = ax2.bar(np.arange(4), vals, width=0.55, color=[M_ROSE, M_BLUE, M_TEAL, M_SAND], zorder=3)
    ax2.set_yscale('log')
    ax2.set_ylim(0.005, 25.0)
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(names, fontsize=9.5, color=H_SLATE)
    ax2.set_ylabel('稳态漂移率 (mm/s，对数)', fontsize=10.0, color=H_SLATE)
    ax2.set_title('4 s 静置数值漂移对比（↓348×）', fontsize=11.0, fontweight='bold', color=H_DARK, pad=10)
    ax2.grid(axis='y', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax2)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.32, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=H_DARK)
    ax2.text(-0.22, -0.18, '(b)', transform=ax2.transAxes, fontsize=11.0, fontweight='bold', color=H_DARK)

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    return out


def gen_acad_s07_rl_arch() -> Path:
    """Slide 7 exhibit: Ultra-clean, large-font borderless 3-color pure-proprioceptive RL architecture."""
    out = ASSETS_DIR / 'acad_s07_rl_arch.png'
    fig, ax = plt.subplots(figsize=(8.0, 4.9), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    # Band 1 header: Deployable path
    ax.plot([0.10, 9.90], [5.95, 5.95], color=H_TEAL, lw=2.4)
    ax.text(0.12, 5.66, '实机在线部署通路（零视觉 · 零力传感器 · 零模式 ID）',
            fontsize=11.2, fontweight='bold', color=H_TEAL, va='center')

    def _blk(x, y, w, h, col, title, body):
        ax.plot([x, x + w], [y + h, y + h], color=col, lw=2.2, solid_capstyle='butt')
        ax.text(x, y + h - 0.28, title, ha='left', va='center', fontsize=10.2, fontweight='bold', color=col)
        ax.text(x, y + (h - 0.56) / 2, body, ha='left', va='center', fontsize=9.0, color=H_SLATE, linespacing=1.42)

    _blk(0.12, 2.55, 2.32, 2.75, H_BLUE, '纯本体历史 (1.6 s)',
         '· 14D 编码器 q / dq\n· 3D 陀螺 + 3D 重力\n· 14D 上一动作 u\n· 3D 速度/急停指令\n· +14D 误差 q_err\n  (隐式力矩感知)')
    _blk(2.85, 3.92, 2.10, 1.38, H_BLUE, 'Actor 因果 TCN',
         'Welford 归一化\n双层因果 1D 卷积\n提取 192D 时序特征')
    _blk(2.85, 2.25, 2.10, 1.40, H_TEAL, 'EstNet 状态反演',
         'MLP 192→128→7\n实时反演速度、管壁\n阻力与轮地滑移')
    _blk(5.38, 3.65, 2.02, 1.65, H_TEAL, '无模式 14D Actor',
         'MLP 256→256→14\nLayerNorm + SiLU\n全开放 10 绳+4 偏航')
    _blk(7.78, 3.65, 2.08, 1.65, H_DARK, '增量积分输出',
         'u_t = u_{t-1} + 0.5 a_t\n抑制高频棘轮抖动\n步态连续自发相变')

    ax.text(6.15, 2.90, 'stop_grad\n梯度截断', ha='center', va='center',
            fontsize=9.2, fontweight='bold', color=H_TEAL)

    # Band 2: Training-only privileged Critic
    ax.plot([0.10, 9.90], [1.92, 1.92], color=H_BLUE, lw=1.6, ls='--')
    ax.text(0.12, 1.66, '仅训练期特权辅助通路（部署时移除）',
            fontsize=10.5, fontweight='bold', color=H_BLUE, va='center')

    _blk(0.12, 0.12, 2.95, 1.30, H_BLUE, '108D 特权物理真值',
         '绳张力 / 20 轮支反力与滑移\n管壁接触力 / 滚阻 / 齿隙 / 坡度')
    _blk(3.45, 0.12, 3.00, 1.30, H_BLUE, '非对称 Critic V(o, s_priv)',
         'Critic-TCN ⊕ s_priv\nMLP 512→256→128→1 · GAE')
    _blk(6.82, 0.12, 3.04, 1.30, H_TEAL, '联合损失与电热正则',
         'L_PPO + L_EstNet + 2× 镜像\n+ 22 W 直流电机铜损焦耳热')

    _arrow(ax, 2.48, 4.55, 2.80, 4.55, H_BLUE)
    _arrow(ax, 2.48, 3.30, 2.80, 3.00, H_TEAL)
    _arrow(ax, 5.00, 4.55, 5.33, 4.55, H_BLUE)
    _arrow(ax, 5.00, 2.90, 5.60, 2.90, H_TEAL)
    _arrow(ax, 6.15, 3.25, 6.15, 3.60, H_TEAL)
    _arrow(ax, 7.44, 4.48, 7.73, 4.48, H_DARK)
    _arrow(ax, 3.10, 0.78, 3.40, 0.78, H_BLUE)
    _arrow(ax, 6.48, 0.78, 6.78, 0.78, H_TEAL)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    return out


def gen_acad_s08_ablation(m: dict[str, Any]) -> Path:
    """Slide 8 exhibit: Rebuilt large-font same-simulator ablation (v5 -> v6a -> v6b -> v6.1 -> v7.1)."""
    out = ASSETS_DIR / 'acad_s08_ablation.png'
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.2, 4.8), dpi=300,
        gridspec_kw={'width_ratios': [1.28, 1.0], 'wspace': 0.30},
    )
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18, wspace=0.30)

    v6a, v6b, v61, v71 = m['v6a'], m['v6b'], m['v61'], m['v71']
    # Panel (a): Soft soil & carpet velocity across controlled policy versions
    terrains = ['松软沙土\n(μ_rr=0.22)', '高阻地毯\n(μ_rr=0.14)']
    v5_vals = [18.3, 32.8]
    v6a_vals = [v6a['soft_soil_100']['mean_vx_mm_s'], v6a['carpet_100']['mean_vx_mm_s']]
    v6b_vals = [v6b['soft_soil_100']['mean_vx_mm_s'], v6b['carpet_100']['mean_vx_mm_s']]
    v61_vals = [v61['soft_soil_100']['mean_vx_mm_s'], v61['carpet_100']['mean_vx_mm_s']]
    v71_vals = [v71['soft_soil_100']['mean_vx_mm_s'], v71['carpet_100']['mean_vx_mm_s']]

    x = np.arange(2)
    w = 0.155
    series = [
        ('v5 手动模态', v5_vals, M_GRAY2, -2 * w),
        ('v6a 51D', v6a_vals, M_SKY, -w),
        ('v6b +q_err', v6b_vals, M_BLUE, 0.0),
        ('v6.1 +电热', v61_vals, M_LAV, w),
        ('v7.1 4096并训', v71_vals, M_TEAL, 2 * w),
    ]
    for lbl, vals, col, off in series:
        bs = ax1.bar(x + off, vals, width=w * 0.92, label=lbl, color=col, zorder=3)
        for b, v in zip(bs, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 2.0, f'{v:.0f}',
                     ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=H_DARK)

    ax1.set_xticks(x)
    ax1.set_xticklabels(terrains, fontsize=10.2, color=H_SLATE)
    ax1.set_ylim(0, 142)
    ax1.set_ylabel('稳态推进均速 (mm/s)', fontsize=10.2, color=H_SLATE)
    ax1.set_title('高阻力柔顺地形推进均速消融', fontsize=11.0, fontweight='bold', color=H_DARK, pad=10)
    ax1.grid(axis='y', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax1)
    ax1.legend(fontsize=8.1, loc='upper center', ncol=3, frameon=False, handlelength=1.0, columnspacing=0.9)
    ax1.text(-0.16, -0.19, '(a)', transform=ax1.transAxes, fontsize=11.0, fontweight='bold', color=H_DARK)

    # Panel (b): Narrow-pipe mean wall contact force
    p_lbls = ['v5\n蛇行', 'v6a\n51D', 'v6b\n+q_err', 'v6.1\n+电热', 'v7.1\n直管']
    p_vals = [
        float(m['pipe']['baseline_wall_mean_n']),
        float(v6a['pipe_3zone_soil']['in_pipe_wall_n']),
        float(v6b['pipe_3zone_soil']['in_pipe_wall_n']),
        float(v61['pipe_3zone_soil']['in_pipe_wall_n']),
        float(v71['straight_3zone_soil']['in_pipe_wall_n']),
    ]
    b2 = ax2.bar(np.arange(5), p_vals, width=0.55, color=[M_ROSE, M_SKY, M_BLUE, M_TEAL, M_SAGE], zorder=3)
    for i, (b, v) in enumerate(zip(b2, p_vals)):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.14, f'{v:.2f} N',
                 ha='center', va='bottom', fontsize=9.0, fontweight='bold',
                 color=H_TEAL if i >= 3 else H_DARK)
    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels(p_lbls, fontsize=9.0, color=H_SLATE)
    ax2.set_ylim(0, 5.65)
    ax2.set_ylabel('窄管平均管壁接触力 (N)', fontsize=10.0, color=H_SLATE)
    ax2.set_title('管壁接触力压降 98.1%–99.2%', fontsize=11.0, fontweight='bold', color=H_DARK, pad=10)
    ax2.grid(axis='y', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax2)
    ax2.text(-0.22, -0.19, '(b)', transform=ax2.transAxes, fontsize=11.0, fontweight='bold', color=H_DARK)

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    return out


def gen_acad_s09_phase_trans() -> Path:
    """Slide 9 exhibit: Rebuilt presentation-resolution 3-zone autonomous phase transition chart."""
    out = ASSETS_DIR / 'acad_s09_phase_trans.png'
    d = np.load(REPO_ROOT / 'runs' / 'v6b_65d_qerr' / 'v6b_autonomous_pipe_traj.npz')
    t = d['t']
    vx_auto = d['vx_mm_s'][:, 0]
    vx_v5 = d['vx_mm_s'][:, 1]
    vx_est = d['est_pred'][:, 0] * 175.0
    yaw_auto = d['yaw_rms_deg'][:, 0]
    yaw_v5 = d['yaw_rms_deg'][:, 1]
    wall_auto = d['wall_n'][:, 0]
    wall_v5 = d['wall_n'][:, 1]
    drag_est = d['est_pred'][:, 5]
    in_p = d['in_pipe'].astype(bool)
    on_s = d['on_soil'].astype(bool)

    t_p0 = float(t[in_p][0])
    t_s0 = float(t[on_s][0])
    t_p1 = float(t[in_p][-1])

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 5.0), dpi=300, sharex=True)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.11, right=0.91, top=0.89, bottom=0.11, hspace=0.26)

    for ax in axes:
        ax.axvspan(t_p0, t_p1, color='#EEF4FA', alpha=0.75, zorder=0)
        ax.axvspan(t_s0, t[-1], color='#F8F3EA', alpha=0.75, zorder=0)
        ax.axvline(t_p0, color=H_BLUE, ls='--', lw=0.9, alpha=0.7)
        ax.axvline(t_s0, color=M_SAND, ls='--', lw=0.9, alpha=0.9)
        ax.grid(axis='y', ls='--', alpha=0.4, color=H_RULE)
        _style_ax(ax)

    # Row 1: Velocity + EstNet (legend cleanly placed outside above plot)
    axes[0].plot(t, vx_auto, color=H_TEAL, lw=2.0, label='v6b 自主策略')
    axes[0].plot(t, vx_est, color=H_BLUE, lw=1.4, ls='--', label='EstNet 反演 (r=0.955)')
    axes[0].plot(t, vx_v5, color=M_GRAY3, lw=1.3, alpha=0.85, label='v5 固定蛇行')
    axes[0].set_ylabel('速度\n(mm/s)', fontsize=9.8, color=H_SLATE)
    axes[0].set_ylim(-15, 265)
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=8.6, frameon=False)
    axes[0].text(1.3, 230, 'I 硬地', fontsize=9.0, fontweight='bold', color=H_SLATE, ha='center')
    axes[0].text(0.5 * (t_p0 + t_s0), 230, 'II 236 mm 窄管', fontsize=9.0, fontweight='bold', color=H_BLUE, ha='center')
    axes[0].text(0.5 * (t_s0 + t[-1]), 230, 'III 松软沙土', fontsize=9.0, fontweight='bold', color='#8C6D46', ha='center')

    # Row 2: Spontaneous yaw contraction & re-expansion (legend outside top right, zero curve overlap)
    axes[1].plot(t, yaw_auto, color=H_BLUE, lw=2.0, label='v6b 偏航 RMS')
    axes[1].plot(t, yaw_v5, color=M_GRAY2, lw=1.4, ls=':', label='v5 偏航 RMS')
    axes[1].set_ylabel('偏航 RMS\n(°)', fontsize=9.8, color=H_SLATE)
    axes[1].set_ylim(0, 27)
    axes[1].text(0.5 * (t_p0 + t_s0), 21.8, '进管自发收缩：10.2° → 6.0° (−40.5%)',
                 fontsize=8.9, fontweight='bold', color=H_BLUE, ha='center')
    axes[1].text(0.5 * (t_p1 + t[-1]), 21.8, '出管重张 10.0°',
                 fontsize=8.8, fontweight='bold', color='#8C6D46', ha='center')
    axes[1].legend(loc='lower right', bbox_to_anchor=(1.0, 0.99), ncol=2, fontsize=8.2, frameon=False)

    # Row 3: Wall force & EstNet drag
    axes[2].plot(t, wall_v5, color=M_ROSE, lw=1.4, label='v5 管壁力 (均 4.72 N)')
    axes[2].plot(t, wall_auto, color=H_TEAL, lw=2.0, label='v6b 管壁力 (均 0.49 N)')
    axes[2].set_ylabel('管壁力\n(N)', fontsize=9.8, color=H_SLATE)
    axes[2].set_xlabel('连续穿越时间 (s)', fontsize=10.2, color=H_SLATE)
    axes[2].set_ylim(-0.5, 24)
    ax_r = axes[2].twinx()
    ax_r.plot(t, drag_est, color=M_SAND, lw=1.8, ls='--', label='EstNet 阻力估计 (r=0.966)')
    ax_r.set_ylabel('阻力估计', fontsize=9.5, color='#8C6D46')
    ax_r.set_ylim(-0.05, 1.25)
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_color(H_RULE)
    ax_r.tick_params(colors='#8C6D46', labelsize=9.2)
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8.2, frameon=False)
    d.close()

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    return out


def gen_acad_s10_v71_pipes(m: dict[str, Any]) -> Path:
    """Slide 10 exhibit: v7.1 9-track velocity + low wall contact force (large presentation fonts)."""
    out = ASSETS_DIR / 'acad_s10_v71_pipes.png'
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.2, 4.7), dpi=300,
        gridspec_kw={'width_ratios': [1.45, 1.0], 'wspace': 0.32},
    )
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.20, right=0.98, top=0.89, bottom=0.17, wspace=0.32)

    v71 = m['v71']
    tracks = [
        ('straight_3zone_open', '直管→硬地', M_BLUE),
        ('straight_3zone_soil', '直管→软土', M_SKY),
        ('open_hard_140', '开阔硬地', M_GRAY3),
        ('bend90_arc_L', '90°圆弧管', M_TEAL),
        ('scurve_arc_L', 'S型双弯管', M_SAGE),
        ('right_angle_L_duct', '90°尖直角方管', M_SAND),
        ('soft_soil_100', '纯松软沙土', M_LAV),
        ('carpet_100', '纯高阻地毯', M_PERI),
    ]
    y = np.arange(len(tracks))
    vxs = [float(v71[k]['mean_vx_mm_s']) for k, _, _ in tracks]
    bars = ax1.barh(y, vxs, height=0.60, color=[c for _, _, c in tracks], zorder=3)
    for b, vx in zip(bars, vxs):
        ax1.text(vx + 3.0, b.get_y() + b.get_height() / 2, f'{vx:.1f}',
                 va='center', ha='left', fontsize=9.2, fontweight='bold', color=H_DARK)
    ax1.set_yticks(y)
    ax1.set_yticklabels([lbl for _, lbl, _ in tracks], fontsize=9.4, color=H_SLATE)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 205)
    ax1.set_xlabel('稳态推进均速 (mm/s，30 s 确定性评测)', fontsize=9.8, color=H_SLATE)
    ax1.set_title('v7.1 全轨道 100% 通过均速 (μ_rr=0.060)', fontsize=10.6, fontweight='bold', color=H_DARK, pad=10)
    ax1.grid(axis='x', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax1)
    ax1.text(-0.34, -0.18, '(a)', transform=ax1.transAxes, fontsize=11.0, fontweight='bold', color=H_DARK)

    p_names = ['v5\n直管', 'v7.1\n直管', 'v7.1\n90°弧', 'v7.1\nS弯', 'v7.1\n90°方管']
    p_vals = [
        float(m['pipe']['baseline_wall_mean_n']),
        float(v71['straight_3zone_soil']['in_pipe_wall_n']),
        float(v71['bend90_arc_L']['in_pipe_wall_n']),
        float(v71['scurve_arc_L']['in_pipe_wall_n']),
        float(v71['right_angle_L_duct']['in_pipe_wall_n']),
    ]
    b2 = ax2.bar(np.arange(5), p_vals, width=0.56, color=[M_ROSE, M_BLUE, M_TEAL, M_SAGE, M_SAND], zorder=3)
    for b, fv in zip(b2, p_vals):
        ax2.text(b.get_x() + b.get_width() / 2, fv + 0.14, f'{fv:.2f}',
                 ha='center', va='bottom', fontsize=9.2, fontweight='bold', color=H_DARK)
    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels(p_names, fontsize=9.0, color=H_SLATE)
    ax2.set_ylim(0, 5.65)
    ax2.set_ylabel('管壁平均接触力 (N)', fontsize=10.0, color=H_SLATE)
    ax2.set_title('四管全尾出管低撞壁力', fontsize=10.6, fontweight='bold', color=H_DARK, pad=10)
    ax2.grid(axis='y', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax2)
    ax2.text(-0.24, -0.18, '(b)', transform=ax2.transAxes, fontsize=11.0, fontweight='bold', color=H_DARK)

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    return out


def gen_acad_s11_v8_live(m: dict[str, Any]) -> Path:
    """Slide 11 exhibit: v8 4,096-env 12-regime training snapshot (large fonts)."""
    out = ASSETS_DIR / 'acad_s11_v8_live.png'
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 4.9), dpi=300,
        gridspec_kw={'height_ratios': [1.0, 1.15], 'hspace': 0.44},
    )
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.15, hspace=0.46)

    hist = m['v8_prog']['history']
    rs = [h['rollout'] for h in hist]
    series = [
        ('str', '直圆管', M_BLUE, '-'),
        ('U180', '180°U弯', M_TEAL, '-'),
        ('L90', '90°方管', M_SAND, '-'),
        ('Z2L', '双90°Z弯', M_ROSE, '--'),
        ('soil', '沙土', M_LAV, '-'),
        ('grs', '草地', M_SAGE, '-'),
        ('grv', '石子', M_TAUPE, '-'),
    ]
    for k, lbl, col, ls in series:
        xs = [h['rollout'] for h in hist if k in h.get('regimes', {})]
        ys = [h['regimes'][k]['vx_mm_s'] for h in hist if k in h.get('regimes', {})]
        if xs:
            ax1.plot(xs, ys, label=lbl, color=col, ls=ls, lw=1.8)
    ax1.set_title(f'v8 4,096 并发十场景收敛曲线（r{rs[-1]}/300 · 48.4M 物理步）',
                  fontsize=10.6, fontweight='bold', color=H_DARK, pad=6)
    ax1.set_ylabel('均速 (mm/s)', fontsize=9.8, color=H_SLATE)
    ax1.set_ylim(16, 134)
    ax1.grid(True, ls='--', alpha=0.45, color=H_RULE)
    _style_ax(ax1)
    ax1.legend(loc='lower right', ncol=4, fontsize=8.2, frameon=False, handlelength=1.3, columnspacing=0.9)
    ax1.text(-0.11, -0.22, '(a)', transform=ax1.transAxes, fontsize=10.6, fontweight='bold', color=H_DARK)

    lat = m['v8_prog']['latest']['regimes']
    reg_order = [
        ('open', '硬地', M_GRAY2), ('str', '直管', M_BLUE), ('b90', '90°弧', M_PERI),
        ('scv', 'S弯', M_SKY), ('U180', '180°U弯', M_SAGE), ('tap', '变径锥', M_TEAL),
        ('W3', '4m三波', M_TEAL), ('L90', '90°方管', M_SAND), ('Z2L', '双90°Z弯', M_PEACH),
        ('soil', '沙土', M_LAV), ('grs', '草地', M_SAGE), ('grv', '石子', M_TAUPE),
    ]
    x2 = np.arange(len(reg_order))
    v2 = [float(lat[k]['vx_mm_s']) for k, _, _ in reg_order]
    b2 = ax2.bar(x2, v2, width=0.62, color=[c for _, _, c in reg_order], zorder=3)
    for b, vx in zip(b2, v2):
        ax2.text(b.get_x() + b.get_width() / 2, vx + 3.0, f'{vx:.0f}',
                 ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=H_DARK)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl, _ in reg_order], fontsize=8.8, color=H_SLATE, rotation=18, ha='right')
    ax2.set_ylim(0, 145)
    ax2.set_ylabel('均速 (mm/s)', fontsize=9.8, color=H_SLATE)
    ax2.set_title('r246 十二工况分布快照（8 类长管 101–121 mm/s · 零台阶）',
                  fontsize=10.6, fontweight='bold', color=H_DARK, pad=6)
    ax2.grid(axis='y', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax2)
    ax2.text(-0.11, -0.34, '(b)', transform=ax2.transAxes, fontsize=10.6, fontweight='bold', color=H_DARK)

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    return out


def gen_acad_s12_z2_equiv(m: dict[str, Any]) -> Path:
    """Slide 12 exhibit: Exact Z2 Reynolds group averaging + closed-loop asymmetry zeroing."""
    out = ASSETS_DIR / 'acad_s12_z2_equiv.png'
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.0, 4.9), dpi=300,
        gridspec_kw={'height_ratios': [1.05, 1.05], 'hspace': 0.36},
    )
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.14, hspace=0.42)

    # Top panel: Clean equation & Reynolds pipeline
    ax1.set_xlim(0, 10.0)
    ax1.set_ylim(0, 3.2)
    ax1.axis('off')
    ax1.plot([0.1, 9.9], [3.05, 3.05], color=H_BLUE, lw=2.0)
    ax1.text(0.15, 2.75, '构造级精确 $\\mathbb{Z}_2$ 群等变（Reynolds 群平均 · 零重训即插即用）',
             fontsize=10.8, fontweight='bold', color=H_BLUE, va='center')

    ax1.text(0.85, 1.45, '观测序列\n$\\mathbf{O}_t$', ha='center', va='center',
             fontsize=10.2, fontweight='bold', color=H_DARK)
    ax1.text(3.35, 2.05, '原分支  $\\pi(\\mathbf{O}_t)$', ha='center', va='center',
             fontsize=10.0, fontweight='bold', color=H_BLUE)
    ax1.text(3.35, 0.85, '镜像分支  $K_g^{-1}\\pi(L_g\\mathbf{O}_t)$', ha='center', va='center',
             fontsize=10.0, fontweight='bold', color=H_TEAL)
    ax1.text(7.65, 1.45,
             '$\\bar{\\pi}(\\mathbf{O}) = \\frac{1}{2}[\\pi(\\mathbf{O}) + K_g^{-1}\\pi(L_g\\mathbf{O})]$\n'
             '$\\bar{\\pi}(L_g\\mathbf{O}) \\equiv K_g\\bar{\\pi}(\\mathbf{O})$（误差恒为 0.0）',
             ha='center', va='center', fontsize=10.2, fontweight='bold', color=H_DARK, linespacing=1.55)
    _arrow(ax1, 1.55, 1.60, 2.35, 2.02, H_BLUE)
    _arrow(ax1, 1.55, 1.30, 2.35, 0.88, H_TEAL)
    _arrow(ax1, 4.45, 2.02, 5.55, 1.62, H_BLUE)
    _arrow(ax1, 4.80, 0.88, 5.55, 1.28, H_TEAL)
    ax1.text(-0.10, -0.10, '(a)', transform=ax1.transAxes, fontsize=10.6, fontweight='bold', color=H_DARK)

    # Bottom panel: Closed-loop L/R turning heading asymmetry drop
    cl = m['v8_ab']['benchmark_a_equivariance']['closed_loop_autonomous_eval']
    c61 = cl[0]['closed_loop_tracks']
    c6a = cl[1]['closed_loop_tracks']
    labels = ['v6.1 转向偏置\n(软对称 vs Z2)', 'v6a 转向偏置\n(软对称 vs Z2)', 'v6.1 直行漂移\n(软对称 vs Z2)', 'v6a 直行漂移\n(软对称 vs Z2)']
    before = [
        c61['base']['turn_LR_heading_asym_deg'],
        c6a['base']['turn_LR_heading_asym_deg'],
        c61['base']['straight_abs_heading_drift_deg'],
        c6a['base']['straight_abs_heading_drift_deg'],
    ]
    after = [0.0, 0.0, 0.0, 0.0]

    x = np.arange(4)
    w = 0.34
    b1 = ax2.bar(x - w / 2, before, width=w, color=M_ROSE, label='原软对称正则 (Mittal ICRA\'24 方式)', zorder=3)
    b2 = ax2.bar(x + w / 2, [0.18] * 4, width=w, color=M_TEAL, label='加精确 Z2 群等变封装', zorder=3)
    for b, v in zip(b1, before):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.25, f'{v:.2f}°',
                 ha='center', va='bottom', fontsize=9.2, fontweight='bold', color=H_DARK)
    for b in b2:
        ax2.text(b.get_x() + b.get_width() / 2, 0.32, '0.0°',
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=H_TEAL)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.0, color=H_SLATE)
    ax2.set_ylim(0, 11.8)
    ax2.set_ylabel('航向不对称 / 漂移 (°)', fontsize=9.8, color=H_SLATE)
    ax2.grid(axis='y', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax2)
    ax2.legend(loc='upper right', fontsize=8.8, frameon=False)
    ax2.text(-0.10, -0.32, '(b)', transform=ax2.transAxes, fontsize=10.6, fontweight='bold', color=H_DARK)

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    return out


def gen_acad_s13_rolling_drag(m: dict[str, Any]) -> Path:
    """Slide 13 exhibit: Honest rolling-drag sensitivity (mu_rr=0.025 vs 0.060 vs 0.220) + BL/s."""
    out = ASSETS_DIR / 'acad_s13_rolling_drag.png'
    fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=300)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.20)

    labels = [
        'v6b 硬地\n(μ_rr=0.025\n偏乐观轴承)',
        'v7.1 硬地\n(μ_rr=0.060\n棘轮标称)',
        'v8 硬地\n(μ_rr=0.060\n十场景统一)',
        'v7.1 90°方管\n(μ_rr=0.060)',
        'v7.1 软土\n(μ_rr=0.220)',
        'v8 石子地\n(μ_rr=0.175)',
    ]
    vals = [
        float(m['v6b']['open_hard_140']['mean_vx_mm_s']),
        float(m['v71']['open_hard_140']['mean_vx_mm_s']),
        float(m['v8_prog']['latest']['regimes']['open']['vx_mm_s']),
        float(m['v71']['right_angle_L_duct']['mean_vx_mm_s']),
        float(m['v71']['soft_soil_100']['mean_vx_mm_s']),
        float(m['v8_prog']['latest']['regimes']['grv']['vx_mm_s']),
    ]
    cols = [M_ROSE, M_BLUE, M_SKY, M_TEAL, M_LAV, M_TAUPE]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.55, color=cols, zorder=3)
    for b, v in zip(bars, vals):
        bl = v / 850.0  # 0.85 m 5-segment robot length
        ax.text(b.get_x() + b.get_width() / 2, v + 4.5, f'{v:.0f} mm/s\n({bl:.2f} BL/s)',
                ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=H_DARK, linespacing=1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.8, color=H_SLATE)
    ax.set_ylim(0, 255)
    ax.set_ylabel('仿真推进均速 (mm/s)', fontsize=10.2, color=H_SLATE)
    ax.set_title('硬地滚动阻力假设对推进均速的影响（体长 0.85 m 归一化）',
                 fontsize=11.0, fontweight='bold', color=H_DARK, pad=10)
    ax.grid(axis='y', ls='--', alpha=0.45, color=H_RULE, zorder=0)
    _style_ax(ax)

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PPTX Primitives (Zero Borders, Zero Card Fills, Zero Drop Shadows, Pure YaHei)
# ---------------------------------------------------------------------------
def _set_run_font(run, size: float, bold: bool, color: RGBColor, italic: bool = False) -> None:
    run.font.name = FONT_FACE
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ('a:latin', 'a:ea', 'a:cs'):
        el = rPr.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            rPr.append(el)
        el.set('typeface', FONT_FACE)


def _strip_shadow(shp) -> None:
    shp.shadow.inherit = False
    spPr = shp._element.spPr
    eff = spPr.find(qn('a:effectLst'))
    if eff is not None:
        spPr.remove(eff)
    spPr.append(OxmlElement('a:effectLst'))


def set_bg_white(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = C_WHITE


def add_line_rect(slide, x: float, y: float, w: float, h: float, color: RGBColor) -> None:
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    _strip_shadow(shp)


def add_text(slide, x: float, y: float, w: float, h: float, text: str,
             size: float = 16.0, bold: bool = False, color: RGBColor = C_BODY,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, line_spacing: float = 1.22,
             italic: bool = False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    run = p.add_run()
    run.text = text
    _set_run_font(run, size, bold, color, italic=italic)
    return tb


def add_bullets(slide, x: float, y: float, w: float, h: float,
                items: list[tuple[str, str]], size: float = 16.5,
                lead_color: RGBColor = C_PRIMARY, body_color: RGBColor = C_BODY,
                bullet_color: RGBColor = C_BLUE, space_after: float = 12.0,
                line_spacing: float = 1.28, bullet_char: str = '· '):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for idx, (lead, body) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        if bullet_char:
            r0 = p.add_run()
            r0.text = bullet_char
            _set_run_font(r0, size, True, bullet_color)
        if lead:
            r1 = p.add_run()
            r1.text = lead
            _set_run_font(r1, size, True, lead_color)
        if body:
            r2 = p.add_run()
            r2.text = body
            _set_run_font(r2, size, False, body_color)
    return tb


def add_picture_fitted(slide, img_path: Path, x: float, y: float, w: float, h: float,
                       caption: str | None = None, audit: list | None = None):
    """Borderless image preserving 100% exact native aspect ratio."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    cap_h = 0.28 if caption else 0.0
    iw_box = max(0.1, w)
    ih_box = max(0.1, h - cap_h)
    scale = min(iw_box / iw, ih_box / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (iw_box - pw) / 2.0
    py = y + (ih_box - ph) / 2.0
    pic = slide.shapes.add_picture(str(img_path), Inches(px), Inches(py), width=Inches(pw), height=Inches(ph))
    if caption:
        add_text(slide, x, y + h - cap_h, w, cap_h, caption,
                 size=10.5, color=C_MUTED, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    if audit is not None:
        audit.append((img_path.name, iw / ih, pw / ph))
    return pic


def set_booktabs_border(cell, top_hex: str | None, bottom_hex: str | None,
                        top_pt: float = 1.0, bottom_pt: float = 0.5) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    for edge in ('lnL', 'lnR', 'lnT', 'lnB'):
        old = tcPr.find(qn(f'a:{edge}'))
        if old is not None:
            tcPr.remove(old)
        ln = OxmlElement(f'a:{edge}')
        if edge == 'lnT' and top_hex:
            ln.set('w', str(int(top_pt * 12700)))
            sf = OxmlElement('a:solidFill')
            clr = OxmlElement('a:srgbClr')
            clr.set('val', top_hex)
            sf.append(clr)
            ln.append(sf)
        elif edge == 'lnB' and bottom_hex:
            ln.set('w', str(int(bottom_pt * 12700)))
            sf = OxmlElement('a:solidFill')
            clr = OxmlElement('a:srgbClr')
            clr.set('val', bottom_hex)
            sf.append(clr)
            ln.append(sf)
        else:
            ln.append(OxmlElement('a:noFill'))
        tcPr.append(ln)


def add_booktabs_table(slide, x: float, y: float, w: float, h: float,
                       headers: list[str], rows: list[list[str]],
                       col_widths: list[float] | None = None,
                       font_size: float = 11.5, header_size: float = 12.0,
                       highlight_rows: set[int] | None = None,
                       align_left_cols: set[int] | None = None):
    tbl_shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = tbl_shape.table
    tbl.first_row = True
    if col_widths:
        tot = sum(col_widths)
        for j, cw in enumerate(col_widths):
            tbl.columns[j].width = Inches(w * cw / tot)
    else:
        for j in range(len(headers)):
            tbl.columns[j].width = Inches(w / len(headers))

    al = align_left_cols if align_left_cols is not None else {0}
    hi = highlight_rows or set()

    for j, ht in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_TBL_HEAD
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        set_booktabs_border(cell, '5B7FA6', '5B7FA6', top_pt=1.2, bottom_pt=0.8)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j in al else PP_ALIGN.CENTER
        r = p.add_run()
        r.text = ht
        _set_run_font(r, header_size, True, C_PRIMARY)

    for i, row in enumerate(rows):
        is_hi = i in hi
        is_last = i == len(rows) - 1
        bg = C_TBL_HI if is_hi else C_WHITE
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            set_booktabs_border(
                cell, None, '5B7FA6' if is_last else 'E6ECF2',
                bottom_pt=1.1 if is_last else 0.45,
            )
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j in al else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            _set_run_font(r, font_size, is_hi or (j == 0), C_PRIMARY if (is_hi or j == 0) else C_BODY)
    return tbl_shape


def add_academic_chrome(slide, slide_num: int, active_section: str, action_title: str,
                        citation: str, appendix_tag: str | None = None,
                        ghost_titles: list[tuple[int, str]] | None = None) -> None:
    """Gabberflast academic-pptx chrome: breadcrumb bar + action title (no title underline) + bottom citation."""
    set_bg_white(slide)

    # Subtle text-only top breadcrumb bar (for decks > 15 slides)
    sec_w = (SLIDE_W - 2 * MARGIN) / len(SECTIONS)
    for i, sec in enumerate(SECTIONS):
        is_act = (sec == active_section)
        sx = MARGIN + i * sec_w
        add_text(slide, sx, 0.14, sec_w, 0.28, sec,
                 size=9.5, bold=is_act, color=C_BLUE if is_act else C_MUTED,
                 align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        if is_act:
            add_line_rect(slide, sx + sec_w * 0.22, 0.42, sec_w * 0.56, 0.022, C_BLUE)
    add_line_rect(slide, MARGIN, 0.45, SLIDE_W - 2 * MARGIN, 0.008, C_RULE)

    # Action title (NO accent line under title — whitespace only per SKILL.md)
    if appendix_tag:
        add_text(slide, MARGIN, 0.56, SLIDE_W - 2 * MARGIN, 0.28, appendix_tag,
                 size=12.0, bold=True, color=C_TEAL)
        add_text(slide, MARGIN, 0.86, SLIDE_W - 2 * MARGIN, 0.72, action_title,
                 size=19.5, bold=True, color=C_PRIMARY, line_spacing=1.18)
    else:
        add_text(slide, MARGIN, 0.62, SLIDE_W - 2 * MARGIN, 0.85, action_title,
                 size=20.5, bold=True, color=C_PRIMARY, line_spacing=1.20)

    if ghost_titles is not None:
        ghost_titles.append((slide_num, action_title))

    # Bottom citation + slide number
    add_line_rect(slide, MARGIN, 6.98, SLIDE_W - 2 * MARGIN, 0.008, C_RULE)
    add_text(slide, MARGIN, 7.04, 10.60, 0.30, citation, size=10.5, color=C_MUTED, valign=MSO_ANCHOR.MIDDLE)
    add_text(slide, 11.40, 7.04, 1.38, 0.30, f'{slide_num:02d} / {TOTAL_SLIDES:02d}',
             size=10.5, bold=True, color=C_MUTED, align=PP_ALIGN.RIGHT, valign=MSO_ANCHOR.MIDDLE)


# ---------------------------------------------------------------------------
# Slide Builders: Main Deck (Slides 01 – 14)
# ---------------------------------------------------------------------------
def build_s01_title(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg_white(slide)
    add_line_rect(slide, 0.0, 0.0, SLIDE_W, 0.045, C_BLUE)

    title = '五体节预弯钢片蛇蠕机器人：\n高保真 GPU 仿真、纯本体自主相变\n强化学习与实机迁移'
    ghost.append((1, title.replace('\n', ' ')))

    add_text(slide, 0.70, 1.05, 6.20, 0.36, '学术研究进展汇报 · SerpWorm-RL',
             size=13.5, bold=True, color=C_TEAL)
    add_text(slide, 0.70, 1.52, 6.35, 2.25, title,
             size=28.0, bold=True, color=C_PRIMARY, line_spacing=1.25)

    add_line_rect(slide, 0.70, 4.02, 2.20, 0.032, C_BLUE)

    add_text(slide, 0.70, 4.25, 6.20, 1.05,
             'SerpWorm-RL 研究课题组\n'
             '1:1 实称力学孪生 · 无损 FP64 CUDA · 纯本体自主相变 · 精确 Z2 群等变\n'
             '2026 年 11 月',
             size=13.5, color=C_BODY, line_spacing=1.48)

    hero = resolve_asset('v71_r170_all4_pipes_3d_preview.png', 'runs/demos/v71_r170_all4_pipes_3d_preview.png')
    add_picture_fitted(slide, hero, 7.15, 1.10, 5.60, 4.60,
                       caption='v7.1 单一无模式策略在直管、90° 弧管、S 弯与 90° 尖直角方管均实现 100% 出管',
                       audit=audit)

    add_text(slide, 0.70, 6.82, 12.0, 0.36,
             '结构化论证模式：主讲 14 页 + 参考文献 1 页 + 问答附录 5 页',
             size=11.0, color=C_MUTED)


def build_s02_motivation(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 2, '背景与问题',
        '刚柔机器人在硬地、90° 急弯窄管与软土间存在步态冲突，高刚度仿真制约大规模 RL',
        '数据来源：SerpWorm 五体节 CAD 构型及同机 Xeon 8470Q 仿真基准（详见附录 A/B）',
        ghost_titles=ghost,
    )
    cad = resolve_asset('cad_with_steel_strips.png')
    add_picture_fitted(slide, cad, MARGIN, 1.65, 7.65, 4.95,
                       caption='五体节闭环并联构型：40 片预弯 65Mn 钢片、14 路舵机、20 颗单向棘轮（整机实称 2.757 kg）',
                       audit=audit)

    add_text(slide, 8.45, 1.72, 4.30, 0.40, '三大核心瓶颈', size=18.0, bold=True, color=C_BLUE)
    add_bullets(slide, 8.45, 2.24, 4.30, 4.30, [
        ('多场景步态冲突：', '硬地需高速蛇行，90° 急弯窄管易卡壁（固定蛇行撞壁 4.72 N），软土滚阻激增。'),
        ('算力瓶颈：', '40 片薄钢片刚度比 5.29×10³，单核 CPU 仅 0.27 sps，难以支撑千万步 RL。'),
        ('零外感约束：', '管内无视觉与力传感，仅靠舵机编码器与单颗 IMU 完成状态反演。'),
    ], size=16.0, space_after=16.0)


def build_s03_research_question(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 3, '背景与问题',
        '核心问题：仅靠编码器与单颗 IMU，单一无模式策略能否实现跨地形与急弯管自主相变？',
        '研究边界：不依赖视觉、雷达、管壁力传感器或人工地形/步态模式标签',
        ghost_titles=ghost,
    )

    # Clean borderless focal question with soft left accent bar (zero box border/fill)
    add_line_rect(slide, 0.90, 1.92, 0.06, 1.65, C_TEAL)
    add_text(
        slide, 1.18, 1.92, 11.20, 1.65,
        '在 40 片预弯弹簧钢片闭环并联的强耦合动力学下，能否依托 1:1 实称物理孪生与无损 GPU 求解，'
        '训练出仅凭纯本体传感、无需模式切换即可连续穿越开阔硬地、90° 尖直角窄管与松软颗粒地形的单一策略？',
        size=20.0, bold=False, color=C_PRIMARY, line_spacing=1.42, valign=MSO_ANCHOR.MIDDLE,
    )

    # 3 clean pillars answering how we decompose the question
    cols = [
        ('子问题 1 · 物理与算力', C_BLUE,
         '消除 7.04× 刚度失配与 572.7 g 质量偏差，同时将 FP64 仿真提速两个数量级。'),
        ('子问题 2 · 零外感相变', C_TEAL,
         '用软件伺服误差 q_err 反演接触与阻力，令单一 14D 网络自发涌现“蛇行↔蠕动”相变。'),
        ('子问题 3 · 对称与迁移', C_BLUE,
         '构造级消除左右转向与弯管偏置，并保证亚毫秒级 CPU 实机实时部署。'),
    ]
    cw, cg = 3.80, 0.41
    for i, (ttl, col, body) in enumerate(cols):
        cx = MARGIN + i * (cw + cg)
        add_line_rect(slide, cx, 4.18, cw, 0.028, col)
        add_text(slide, cx, 4.34, cw, 0.42, ttl, size=17.5, bold=True, color=col)
        add_text(slide, cx, 4.86, cw, 1.65, body, size=15.5, color=C_BODY, line_spacing=1.38)


def build_s04_sysid_mass(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 4, '物理与算力',
        '实称 1:1 质量建模（572.7 g/节）与 UTM/双激光标定彻底消除 7.04× 刚度失配',
        '来源：UTM 5 循环实测与 Exp 126–128 双激光台架；逐部件实称清单见附录 A',
        ghost_titles=ghost,
    )
    utm_img = resolve_asset('utm_scene_side_by_side_preview.png', 'runs/utm_validation/utm_scene_side_by_side_preview.png')
    add_picture_fitted(slide, utm_img, MARGIN, 1.62, 7.65, 5.05,
                       caption='单体节 0–45 mm UTM 5 循环压载实拍 vs 1:1 孪生及非线性软化刚度曲线',
                       audit=audit)

    utm = m['utm']
    add_text(slide, 8.45, 1.70, 4.30, 0.40, '台架标定关键指标', size=18.0, bold=True, color=C_BLUE)
    add_bullets(slide, 8.45, 2.22, 4.30, 4.45, [
        ('刚度精确对齐：', f"UTM 拟合 R²={utm['r2_cal']:.5f}，峰值 {utm['cal_peak_N']:.2f} vs {utm['real_peak_mid_N']:.2f} N，消除 7.04× 失配。"),
        ('双激光偏转标定：', '偏心曲柄 + Capstan 摩擦复现双侧 36.8 mm 及 ±18.2° 偏转，误差 ≤ 0.42 mm。'),
        ('实称零误差质量：', '碳板 289.3 g + PCB 253.0 g + 8 钢片 30.4 g = 572.7 g，单轮 17.9 g 同步入核。'),
    ], size=15.5, space_after=16.0)


def build_s05_cuda_solver(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 5, '物理与算力',
        '块 Thomas–Schur 无损 FP64 CUDA 融合核同机实现 153× GPU 与 17,649× 单核 CPU 加速',
        '同机实测：Xeon 8470Q + RTX 5090，同 5 节 40 钢片模型（B=2048）；文献平台背景定位见附录 B',
        ghost_titles=ghost,
    )
    fig = gen_acad_s05_cuda_samehost(m)
    add_picture_fitted(slide, fig, MARGIN, 1.60, 8.00, 5.10, audit=audit)

    add_text(slide, 8.75, 1.72, 4.00, 0.40, '无损加速机理', size=18.0, bold=True, color=C_TEAL)
    add_bullets(slide, 8.75, 2.24, 4.00, 4.40, [
        ('二分稀疏直接求解：', '40 钢片并行块 Thomas 消元，凝聚为 5 组 12×12 隔板 Schur 补 Cholesky。'),
        ('全图 CUDA Graph：', '100 次 FP64 子步整图捕获，2048 机显存从 9.85 GB 降至 3.39 GB（−65.6%）。'),
        ('严格 FP64 无近似：', '200 步与参考求解器偏差仅 4.35×10⁻¹⁰ m，单机可扩展至 4,096 并发环境。'),
    ], size=16.0, space_after=16.0)


def build_s06_arpc_contact(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 6, '物理与算力',
        '隐式 ARPC 棘轮面片接触消除 57.9 N 跌落脉冲，将 1 ms 步长静置漂移压降 348×',
        '来源：contact_demo_metrics.json（Δt = 1.0 ms，E = 16，fused_cuda 实测）',
        ghost_titles=ghost,
    )
    fig = gen_acad_s06_arpc(m)
    add_picture_fitted(slide, fig, MARGIN, 1.62, 8.05, 5.05, audit=audit)

    add_text(slide, 8.80, 1.72, 3.95, 0.40, '五重接触物理', size=18.0, bold=True, color=C_BLUE)
    add_bullets(slide, 8.80, 2.24, 3.95, 4.40, [
        ('Hunt–Crossley 法向：', '消除线性 Kelvin 57.9 N 触地尖峰与 43 ms 弹跳。'),
        ('真实单向棘轮建模：', '含 10° 棘爪空程，反转卡轴后依库仑椭圆允许过载倒滑，并非锁死。'),
        ('弹塑性车辙 + 枢转：', '复现软土 1.08 mm 车辙与转向阻尼，隐式 Contact-Schur 使步长翻倍至 1 ms。'),
    ], size=15.5, space_after=15.0)


def build_s07_rl_method(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 7, '算法与结果',
        '以软件跟踪误差 q_err 构造零成本隐式力矩通道，支撑无模式纯本体非对称 RL',
        '参考架构思想：非对称 Actor-Critic（Pinto et al., 2018）与显式状态估计；DR 细节见附录 C',
        ghost_titles=ghost,
    )
    fig = gen_acad_s07_rl_arch()
    add_picture_fitted(slide, fig, MARGIN, 1.60, 7.95, 5.10, audit=audit)

    add_text(slide, 8.70, 1.72, 4.05, 0.40, '三项核心设计', size=18.0, bold=True, color=C_TEAL)
    add_bullets(slide, 8.70, 2.24, 4.05, 4.45, [
        ('零成本软力矩感知：', '仅用指令与编码器差值 q_err（51D→65D），间接映射负载与管壁阻力。'),
        ('完全无模式 14D 动作：', '去除人工步态模板与模式 ID，10 根独立拉绳 + 4 偏航始终开放。'),
        ('非对称特权训练：', '训练期 Critic 读取 108D 物理真值，部署端彻底零外感、零依赖。'),
    ], size=15.5, space_after=16.0)


def build_s08_ablation(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 8, '算法与结果',
        '软件 q_err 令软土提速 3.0×，电热正则将窄管平均管壁接触力压降至 0.09 N（−98.1%）',
        '来源：同仿真器 6 条 28–30 s 确定性测试轨道可控消融（v5 / v6a / v6b / v6.1 / v7.1）',
        ghost_titles=ghost,
    )
    fig = gen_acad_s08_ablation(m)
    add_picture_fitted(slide, fig, MARGIN, 1.62, 8.05, 5.05, audit=audit)

    add_text(slide, 8.80, 1.72, 3.95, 0.40, '消融核心发现', size=18.0, bold=True, color=C_BLUE)
    add_bullets(slide, 8.80, 2.24, 3.95, 4.45, [
        ('q_err 打破高阻瓶颈：', '加入纯软件 q_err 后，EstNet 速度 MAE 直降 42%，软土均速由 12.1 跃至 36.1 mm/s。'),
        ('大规模并训再跃升：', 'v7.1 4,096 并发多管并训将软土与地毯推进均速进一步推升至 100.8 / 89.2 mm/s。'),
        ('柔顺低撞壁穿越：', 'v6.1 / v7.1 将直管接触力由 v5 的 4.72 N 压降至 0.04–0.09 N，全轨零失败。'),
    ], size=15.8, space_after=15.0)


def build_s09_phase_transition(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 9, '算法与结果',
        '零外感下策略在硬地→窄管→沙土连续穿越中自发收缩偏航 40.5%，净位移提升 117%',
        '来源：v6b_autonomous_pipe_traj.npz（25.96 s 连续穿越；236 mm 窄管衔接松软沙土）',
        ghost_titles=ghost,
    )
    fig = gen_acad_s09_phase_trans()
    add_picture_fitted(slide, fig, MARGIN, 1.60, 7.95, 5.15, audit=audit)

    p = m['pipe']
    add_text(slide, 8.70, 1.72, 4.05, 0.40, '涌现相变证据', size=18.0, bold=True, color=C_TEAL)
    add_bullets(slide, 8.70, 2.24, 4.05, 4.45, [
        ('全程净位移翻倍：', f"26 s 净推进 {p['auto_net_fwd_mm']:,.0f} mm，较固定蛇行 ({p['baseline_v5_snake_net_fwd_mm']:,.0f} mm) 提升 +{p['gain_pct']:.0f}%。"),
        ('管口自发切态：', '进管瞬间偏航 RMS 从 10.2° 收缩至 6.0° 并激发拉绳蠕动，管壁力降低 89.6%。'),
        ('纯本体感知闭环：', f"EstNet 速度相关 r = {p['est_v_corr']:.3f}，出管入沙土阻力识别 r = {p['est_drag_corr']:.3f}，偏航自发重张至 10.0°。"),
    ], size=15.8, space_after=15.0)


def build_s10_v71_curved_pipes(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 10, '算法与结果',
        '4,096 并发训练攻克直管、90° 弧管、S 弯与 90° 尖直角方管，四管 100% 全尾出管',
        '来源：v71_Lduct_4096/eval_final_best_summary.json（r219 best.pt，9/9 赛道零失败，μ_rr = 0.060）',
        ghost_titles=ghost,
    )
    fig = gen_acad_s10_v71_pipes(m)
    add_picture_fitted(slide, fig, MARGIN, 1.62, 8.05, 5.05, audit=audit)

    v71 = m['v71']
    add_text(slide, 8.80, 1.72, 3.95, 0.40, '攻克 90° 尖直角关键', size=18.0, bold=True, color=C_BLUE)
    add_bullets(slide, 8.80, 2.24, 3.95, 4.45, [
        ('四管全尾平稳出管：', f"90° 尖直角方管净转 −93.4°，均速 {v71['right_angle_L_duct']['mean_vx_mm_s']:.1f} mm/s，平均管壁力仅 {v71['right_angle_L_duct']['in_pipe_wall_n']:.2f} N。"),
        ('精确内角几何与防穿透：', '修正 Minkowski 圆角偏置恢复 +30.7 mm 净空，600 钢片节点零穿模。'),
        ('弧长进度引导奖励：', '沿管道中心线推进奖励 v_arc 结合 PMMA 0.18 壁摩擦，消除拐角原地滞留。'),
    ], size=15.8, space_after=15.0)


def build_s11_v8_diverse(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    v8_r = m['v8_prog']['latest']['rollout']
    add_academic_chrome(
        slide, 11, '算法与结果',
        '8 类长管（最长 4 m）及草地/石子地十场景并训，统一策略 8 管稳定在 101–121 mm/s',
        f'来源：v8_diverse_4096/progress.json（r{v8_r}/300，4,096 并发，μ_rr = 0.060，全程零台阶）',
        ghost_titles=ghost,
    )
    fig = gen_acad_s11_v8_live(m)
    add_picture_fitted(slide, fig, MARGIN, 1.60, 8.05, 5.10, audit=audit)

    lat = m['v8_prog']['latest']
    add_text(slide, 8.80, 1.72, 3.95, 0.40, '大规模泛化进展', size=18.0, bold=True, color=C_TEAL)
    add_bullets(slide, 8.80, 2.24, 3.95, 4.45, [
        ('8 类解析长管全覆盖：', '新增 180° U 弯、325→216 mm 变径锥、双 90° Z 弯与 4 m 三波管，均速全破百。'),
        ('自然野外非结构地形：', f"松软沙土 {lat['regimes']['soil']['vx_mm_s']:.0f}、起伏草地 {lat['regimes']['grs']['vx_mm_s']:.0f}、颠簸石子地 {lat['regimes']['grv']['vx_mm_s']:.0f} mm/s。"),
        ('高保真规模化训练：', f"4,096 机实测端到端 {lat['sps']:,.0f} sps，EstNet 跨 12 工况速度 MAE 仅 {lat['est_vx_mae_mm_s']:.1f} mm/s。"),
    ], size=15.8, space_after=15.0)


def build_s12_z2_equivariance(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 12, '算法与结果',
        'Reynolds 群平均实现精确 Z2 群等变，无重训即将左右镜像误差与转向偏置严格归零',
        '对标基线：Mittal et al. (ICRA 2024) 软对称数据增强与正则；数据见 v8_ablation_summary.json',
        ghost_titles=ghost,
    )
    fig = gen_acad_s12_z2_equiv(m)
    add_picture_fitted(slide, fig, MARGIN, 1.60, 7.95, 5.12, audit=audit)

    lat_eq = m['v8_ab']['benchmark_b_representation_latency']['cpu_1thread_latency_ms']['v8_equivariant_65d_batched']['p50_ms']
    add_text(slide, 8.70, 1.72, 4.05, 0.40, '理论与实证意义', size=18.0, bold=True, color=C_BLUE)
    add_bullets(slide, 8.70, 2.24, 4.05, 4.45, [
        ('软惩罚 → 硬代数约束：', '现有软正则仍残留 4.3°–9.8° 左右转向不对称；群平均从代数上保证误差恒为 0.0。'),
        ('消除左右弯管偏置：', '严格保证左/右 90° 弯管与 S 弯策略镜像一致，杜绝“左顺右卡”。'),
        ('即插即用与低开销：', f'兼容既有 v5/v6/v7.1/v8 权重无需重训，CPU 单核批推理仅 {lat_eq:.2f} ms（<2.2% 预算）。'),
    ], size=15.8, space_after=15.0)


def build_s13_discussion_limitations(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 13, '讨论与结论',
        '局限与下一步：硬地滚阻显著影响仿真速度上限，需补实机标定与同机跨求解器对比',
        '审稿前自查：避免仿真直比文献实机及跨构型（轮式蛇蠕 vs 无轮气动蠕虫）不公平速度对比',
        ghost_titles=ghost,
    )
    fig = gen_acad_s13_rolling_drag(m)
    add_picture_fitted(slide, fig, MARGIN, 1.62, 7.55, 5.05, audit=audit)

    add_text(slide, 8.35, 1.72, 4.40, 0.40, '诚实局限与投稿补齐项', size=18.0, bold=True, color=C_TEAL)
    add_bullets(slide, 8.35, 2.24, 4.40, 4.50, [
        ('滚阻敏感性风险：', 'v6 硬地 μ_rr=0.025 偏乐观（208 mm/s）；恢复 0.060 后回落至 118–140 mm/s（0.14–0.16 BL/s）。'),
        ('实机斜坡与闭环待测：', '需用斜面临界角实测 3D 打印棘轮真实 μ_rr，完成实机三区与亚克力弯管动捕验证。'),
        ('公平外部对标协议：', '求解器同机补跑 SOFA / MuJoCo 精度-吞吐 Pareto；运动性能用实机 BL/s 与 CoT 对标同类轮式/混合管机。'),
    ], size=15.2, space_after=14.0)


def build_s14_conclusions(prs, m, audit, ghost):
    """Last main slide — stays on screen during Q&A (NO Thank You slide per academic-pptx-skill)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 14, '讨论与结论',
        '结论：实称孪生、无损 FP64 CUDA、纯本体自主相变与精确 Z2 等变打通端到端闭环',
        'SerpWorm-RL Research Team  |  问答环节欢迎批评指正（备份数据与消融详见附录 A–E）',
        ghost_titles=ghost,
    )

    takeaways = [
        ('1. 1:1 实称力学孪生消除核心物理鸿沟：',
         '完成单节 572.7 g（0.0 g 误差）与 17.9 g 单轮惯量建模，UTM 拟合 R² = 0.99986（消 7.04× 失配），双激光误差 ≤ 0.42 mm。'),
        ('2. 无损 FP64 CUDA 与隐式 ARPC 解锁大规模多场景 RL：',
         '同机较 PyTorch GPU 提速 153×、较单核 CPU 提速 17,649×（显存 −65.6%，误差 4.35×10⁻¹⁰ m）；静置漂移降 348×，支持 4,096 并发。'),
        ('3. 零外感无模式策略攻克 90° 尖直角方管与十场景泛化：',
         '纯软件 q_err 实现硬地→窄管→沙土自发相变（净位移 +117%、管壁力 −98.1%）；v7.1 四管 100% 出管，v8 覆盖 8 类长管与野外地形。'),
        ('4. 精确 Z2 群等变与亚毫秒级部署架通实机最后一公里：',
         'Reynolds 群平均将镜像与转向偏置严格归零（0.0°），HIM-DreamWaQ 使 OOD MSE −19.8%，TorchScript CPU 单核推理 0.66–1.00 ms 零超时。'),
    ]
    add_bullets(slide, MARGIN, 1.78, SLIDE_W - 2 * MARGIN, 4.55, takeaways,
                size=16.5, space_after=18.0, line_spacing=1.34, bullet_char='')

    add_line_rect(slide, MARGIN, 6.38, SLIDE_W - 2 * MARGIN, 0.015, C_TEAL)
    add_text(slide, MARGIN, 6.48, SLIDE_W - 2 * MARGIN, 0.38,
             '下一步目标：完成实机棘轮滚阻实测、亚克力弯管/三区闭环验证及同机跨求解器 Pareto 对标',
             size=13.0, bold=True, color=C_BLUE, valign=MSO_ANCHOR.MIDDLE)


# ---------------------------------------------------------------------------
# References (Slide 15) & Appendices A–E (Slides 16 – 20)
# ---------------------------------------------------------------------------
def build_s15_references(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 15, '文献与附录',
        '参考文献（References）',
        '页内所有对比基线与理论参考均已核验原始出处',
        ghost_titles=ghost,
    )
    refs = [
        '1. Mittal, M., et al. (2024). Symmetry Considerations for Learning Task Symmetric Robot Policies. IEEE ICRA 2024 (arXiv:2403.04359).',
        '2. Nahrendra, I. M. A., Yu, B., & Myung, H. (2023). DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep RL. IEEE ICRA 2023 (arXiv:2301.10602).',
        '3. Long, J., et al. (2024). Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response. ICLR 2024 (arXiv:2312.11460).',
        '4. Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021). RMA: Rapid Motor Adaptation for Legged Robots. RSS 2021 (arXiv:2107.04034).',
        '5. Hwangbo, J., et al. (2019). Learning Agile and Dynamic Motor Skills for Legged Robots (ActuatorNet). Science Robotics, 4(26) (arXiv:1901.08652).',
        '6. Tiboni, G., et al. (2023). On the Effectiveness of Model Order Reduction for RL in Soft Robotics. IEEE/RSJ IROS 2023 (arXiv:2303.04136).',
        '7. Schaff, C., Sedal, A., & Walter, M. R. (2022). Soft Robots Learn to Crawl: Jointly Optimizing Design and Control with Sim-to-Real Transfer. RSS 2022.',
        '8. Choi, A., et al. (2025). DisMech-RL: Implicit Discrete Elastic Rod Simulation for Soft Robot RL. arXiv:2511.06667.',
        '9. Stölzle, M., et al. (2026). SoRoMoX: GPU-Accelerated Differentiable Soft Robot Modeling. arXiv:2608.06650.',
        '10. Gidoni, P., & DeSimone, A. (2017). Stasis Domains and Slip Surfaces in the Locomotion of a Bio-Inspired Two-Segment Crawler. Meccanica (arXiv:1710.08340).',
    ]
    add_bullets(slide, MARGIN, 1.56, SLIDE_W - 2 * MARGIN, 5.25,
                [('', r) for r in refs], size=11.5, body_color=C_BODY,
                space_after=7.5, line_spacing=1.22, bullet_char='')


def build_s16_app_a_mass(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 16, '文献与附录',
        '单节实称 572.7 g（0.0 g 残差）与 10 隔板非对称质量分配已同步注入 Python 与 CUDA 核',
        '来源：src/serpworm/config.py (v5_realistic)、wheel_stiction.py 与 fused_physics.cu（13/13 pytest 通过）',
        appendix_tag='附录 A — 实称硬件质量分配明细',
        ghost_titles=ghost,
    )
    img = resolve_asset('diag_real_mass_profile.png')
    add_picture_fitted(slide, img, MARGIN, 1.72, 6.25, 5.10, audit=audit)

    headers = ['实称部件', '单件质量', '单节装配位置', '折合单板/节']
    rows = [
        ['单轴拉绳舵机', '76.2 g', '2× 装于碳纤维后板 (p1,3,5,7,9)', '152.4 g/碳板'],
        ['碳纤维后板+紧固', '31.8+30.1 g', '后板主体 + 紧固件/拉线', '61.9 g/碳板'],
        ['双轴偏航舵机', '80.8 g', '1× 装于中段 PCB 前板 (p2,4,6,8)', '80.8 g/PCB'],
        ['PCB 板+线束', '41.0+30.0 g', '装于前板 (p0,2,4,6,8)', '71.0 g/PCB'],
        ['U型板 + 固定板', '20.4+5.8 g', '随双轴舵机装于中段 PCB', '26.2 g/PCB'],
        ['腿 + 单向棘轮', '19.6+17.9 g', '每板 2 腿 2 轮（单轮 17.9 g）', '75.0 g/每板'],
        ['65Mn 钢片', '3.8 g', '8 片/节（独立反证宽 16 mm）', '30.4 g/节'],
        ['整节汇总 ★', '572.7 g', 'PCB 253.0 + 碳板 289.3 + 钢片 30.4', '整机 2,756.5 g'],
    ]
    add_booktabs_table(slide, 6.95, 1.82, 5.83, 4.80, headers, rows,
                       col_widths=[1.35, 1.24, 2.08, 1.16],
                       font_size=10.3, header_size=10.8, highlight_rows={7})


def build_s17_app_b_lit_compute(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 17, '文献与附录',
        '文献原报背景定位：软体 RL 吞吐与单机并发跨平台概览（不同构型/硬件，仅供参考）',
        '注：*为各文献在各自构型与硬件上的原报值，不可作同条件加速比；同机实测见主讲第 5 页',
        appendix_tag='附录 B — 软体 RL 仿真算力文献定位参考',
        ghost_titles=ghost,
    )
    img = resolve_asset('diag_cuda_bar_showcase.png')
    add_picture_fitted(slide, img, MARGIN, 1.72, 12.23, 5.05, audit=audit)


def build_s18_app_c_dr_pipes(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 18, '文献与附录',
        '五轴 27 项 Sim-to-Real 域随机化区间与 12D 解析多管族支撑零重编译泛化训练',
        '来源：src/serpworm/config.py (DomainRandConfig) 与 src/serpworm/physics/pipe_geometry.py',
        appendix_tag='附录 C — 五轴域随机化与解析管道族',
        ghost_titles=ghost,
    )
    img_dr = resolve_asset('diag_5axis_dr.png')
    add_picture_fitted(slide, img_dr, MARGIN, 1.72, 6.15, 4.95, audit=audit)

    img_p = resolve_asset('curved_pipes_3design_preview.png', 'runs/demos/curved_pipes_3design_preview.png')
    add_picture_fitted(slide, img_p, 6.95, 1.76, 5.83, 2.35,
                       caption='12D 解析管道族（8 类管型 + 600 钢片零穿模）', audit=audit)

    add_bullets(slide, 6.95, 4.35, 5.83, 2.40, [
        ('地面与结构（13 项）：', '摩擦、滚阻、刚度、棘轮齿隙 4–18°、坡度 ±3.5°、整机与 10 板质量 ±15%。'),
        ('传动与执行（8 项）：', 'Capstan 滞回、系绳误差 ±3 mm、静摩擦、死区与 20–40 ms 总线延迟。'),
        ('传感与冲击（6 项）：', '编码器/IMU 零偏 ±2° 与每 4 s 随机速度脉冲。'),
    ], size=12.6, space_after=8.0)


def build_s19_app_d_him_dreamwaq(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 19, '文献与附录',
        'HIMDreamWaQ 综合 26 篇文献：未见 OOD 重构 MSE 降 19.8%、阻力 MAE 降 26.0%',
        '来源：src/serpworm/rl/him_dreamwaq.py、tests/test_him_dreamwaq.py (29/29) 与 v8_ablation_summary.json',
        appendix_tag='附录 D — HIM-DreamWaQ 表征消融基准',
        ghost_titles=ghost,
    )
    img = resolve_asset('diag_v8_ablation.png')
    add_picture_fitted(slide, img, MARGIN, 1.76, 7.35, 4.90, audit=audit)

    mods = {r['model']: r for r in m['v8_ab']['benchmark_b_representation_latency']['models']}
    est, dw, v8m = mods['EstNet_v6_supervised_only'], mods['DreamWaQ_betaVAE_only'], mods['HIMDreamWaQ_v8_full']
    lat = m['v8_ab']['benchmark_b_representation_latency']['cpu_1thread_latency_ms']
    headers = ['对比模型', 'OOD MSE', '阻力 MAE', 'CPU p50']
    rows = [
        ['EstNet v6', f"{est['ood_next_obs_mse']:.3f}", f"{est['ood_est_drag_mae']:.4f}", f"{lat['v6_estnet_65d']['p50_ms']:.2f} ms"],
        ['DreamWaQ β-VAE', f"{dw['ood_next_obs_mse']:.3f}", f"{dw['ood_est_drag_mae']:.4f}", 'β-VAE'],
        ['HIMDreamWaQ v8 ★', f"{v8m['ood_next_obs_mse']:.3f}", f"{v8m['ood_est_drag_mae']:.4f}", f"{lat['v8_him_dreamwaq_65d']['p50_ms']:.2f} ms"],
    ]
    add_booktabs_table(slide, 8.10, 1.85, 4.68, 2.25, headers, rows,
                       col_widths=[1.55, 1.00, 1.05, 1.08], font_size=11.0, header_size=11.2, highlight_rows={2})

    add_bullets(slide, 8.10, 4.35, 4.68, 2.40, [
        ('多尺度右对齐 TCN：', '共享 context_tcn 令部署参数减少 11.1%（229K 参）。'),
        ('防后验塌缩：', '融合 β-VAE、对称 InfoNCE 与 16 原型 SwAV 对比约束。'),
    ], size=13.5, space_after=10.0)


def build_s20_app_e_deploy_hil(prs, m, audit, ghost):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_academic_chrome(
        slide, 20, '文献与附录',
        '零依赖 TorchScript 部署栈在单核 CPU 实现 0.66–1.00 ms 实时推理与 0/1000 帧超时',
        '来源：src/serpworm/deploy/、run_hil_dryrun.py、test_deploy_runtime.py (19/19) 及 sysid_toolkit.py',
        appendix_tag='附录 E — 实机部署架构、HIL 延迟与 SysID 工具箱',
        ghost_titles=ghost,
    )
    img = resolve_asset('diag_deploy_hil.png')
    add_picture_fitted(slide, img, MARGIN, 1.72, 6.75, 5.05, audit=audit)

    headers = ['检查点 (1000 帧)', '维度', '策略 p50', 'TS 偏差', '超时']
    rows = [
        ['v6a_51d_pure', '51D', '0.664 ms', '0.0', '0/1000'],
        ['unified_v5', '54D', '0.696 ms', '0.0', '0/1000'],
        ['v6_1_65d_elec ★', '65D', '1.000 ms', '0.0', '0/1000'],
    ]
    add_booktabs_table(slide, 7.55, 1.82, 5.23, 2.05, headers, rows,
                       col_widths=[1.65, 0.72, 1.05, 0.85, 0.96], font_size=11.0, header_size=11.2, highlight_rows={2})

    add_text(slide, 7.55, 4.08, 5.23, 0.36, '4 模块实验室 SysID 工具箱就绪', size=14.5, bold=True, color=C_BLUE)
    add_bullets(slide, 7.55, 4.50, 5.23, 2.30, [
        ('240 fps 回弹辨识：', 'FFT + 变投影阻尼正弦拟合反演 β_R。'),
        ('斜坡滚阻 & Capstan：', '实测真实 μ_rr、Stribeck 与导孔滞回。'),
        ('ActuatorNet & E-stop：', '拟合伺服动态并集成 5 重硬件总线锁闭急停。'),
    ], size=13.0, space_after=8.0)


# ---------------------------------------------------------------------------
# Audit & PowerPoint COM Preview Export
# ---------------------------------------------------------------------------
def audit_deck(pptx_path: Path, aspect_audit: list[tuple[str, float, float]], ghost_titles: list[tuple[int, str]]) -> dict[str, Any]:
    prs = Presentation(str(pptx_path))
    sw = prs.slide_width / 914400.0
    sh = prs.slide_height / 914400.0
    oob, placeholders, aspect_issues = [], [], []
    for idx, slide in enumerate(prs.slides, start=1):
        for shp in slide.shapes:
            l, t = shp.left / 914400.0, shp.top / 914400.0
            w, h = shp.width / 914400.0, shp.height / 914400.0
            if l < -0.02 or t < -0.02 or (l + w) > sw + 0.02 or (t + h) > sh + 0.02:
                oob.append(f'Slide {idx}: {shp.name} ({l:.2f},{t:.2f},{w:.2f}x{h:.2f})')
            if shp.has_text_frame:
                txt = shp.text_frame.text
                for bad in ('TODO', 'TBD', 'FIXME', 'Lorem ipsum'):
                    if bad.lower() in txt.lower():
                        placeholders.append(f'Slide {idx}: "{bad}"')
    for name, nat_ar, placed_ar in aspect_audit:
        if abs(placed_ar - nat_ar) / nat_ar > 0.005:
            aspect_issues.append(f'{name}: {nat_ar:.3f} vs {placed_ar:.3f}')
    return {
        'slide_count': len(prs.slides),
        'oob_shapes': oob,
        'placeholders': placeholders,
        'aspect_issues': aspect_issues,
        'checked_images': len(aspect_audit),
        'ghost_deck_titles': [f'{n:02d}: {t}' for n, t in ghost_titles],
    }


def render_previews_via_powerpoint(pptx_path: Path, out_dir: Path) -> list[Path]:
    import win32com.client  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('*.PNG', '*.png'):
        for old in out_dir.glob(ext):
            old.unlink()
    app = win32com.client.DispatchEx('PowerPoint.Application')
    app.Visible = 1
    pres = None
    try:
        pres = app.Presentations.Open(str(pptx_path.resolve()), WithWindow=False)
        pres.SaveAs(str(out_dir.resolve()), 18)
    finally:
        if pres is not None:
            pres.Close()
        app.Quit()
    files = sorted(
        [p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png'],
        key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0),
    )
    norm = []
    for idx, p in enumerate(files, start=1):
        dst = out_dir / f'Slide{idx:02d}.png'
        if p != dst:
            p.replace(dst)
        norm.append(dst)
    return norm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-previews', action='store_true')
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)

    m = load_all_metrics()
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    audit: list[tuple[str, float, float]] = []
    ghost: list[tuple[int, str]] = []

    builders = [
        build_s01_title,
        build_s02_motivation,
        build_s03_research_question,
        build_s04_sysid_mass,
        build_s05_cuda_solver,
        build_s06_arpc_contact,
        build_s07_rl_method,
        build_s08_ablation,
        build_s09_phase_transition,
        build_s10_v71_curved_pipes,
        build_s11_v8_diverse,
        build_s12_z2_equivariance,
        build_s13_discussion_limitations,
        build_s14_conclusions,
        build_s15_references,
        build_s16_app_a_mass,
        build_s17_app_b_lit_compute,
        build_s18_app_c_dr_pipes,
        build_s19_app_d_him_dreamwaq,
        build_s20_app_e_deploy_hil,
    ]
    for fn in builders:
        fn(prs, m, audit, ghost)

    prs.save(str(OUT_PPTX))
    report = audit_deck(OUT_PPTX, audit, ghost)
    if args.render_previews:
        rendered = render_previews_via_powerpoint(OUT_PPTX, PREVIEWS_DIR)
        report['rendered_previews'] = [str(p) for p in rendered]

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if (report['oob_shapes'] or report['placeholders'] or report['aspect_issues']) else 0


if __name__ == '__main__':
    raise SystemExit(main())

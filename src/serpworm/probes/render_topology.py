"""Render the current parameterized five-body topology for visual inspection.

This is a diagnostic visualization, not a replacement for the GPU dynamics or
the CAD/SOFA mesh. It uses the rest configuration in topology_5.npz and draws
schematic plates, passive wheels, tendon spans, and steel-strip centerlines.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from serpworm.paths import INPUTS_DIR, REPO_ROOT

INPUT = INPUTS_DIR / "topology_5.npz"
OUT = REPO_ROOT / "runs" / "topology_model_3d.png"


def add_plate(ax, x, y_radius=0.095, z_center=0.134, z_radius=0.092):
    theta = np.linspace(0, 2 * np.pi, 80)
    y = y_radius * np.cos(theta)
    z = z_center + z_radius * np.sin(theta)
    verts = [list(zip(np.full_like(theta, x), y, z))]
    ax.add_collection3d(
        Poly3DCollection(verts, facecolor="#9aa4ad", edgecolor="#4e5963", alpha=0.18, linewidth=0.8)
    )
    ax.plot(np.full_like(theta, x), y, z, color="#59636d", linewidth=1.0, alpha=0.7)


def add_wheel(ax, x, y, z=0.018, radius=0.018):
    theta = np.linspace(0, 2 * np.pi, 48)
    # A vertical wheel disc: plane spanned by y and z, with its center at x.
    ax.plot(
        np.full_like(theta, x),
        y + radius * np.cos(theta),
        z + radius * np.sin(theta),
        color="#111820",
        linewidth=2.2,
    )
    ax.plot([x, x], [y, y], [z - radius, z + radius], color="#d7dce1", linewidth=0.8)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    data = np.load(args.input)
    rest = data["rest"][:, :3]
    strip_nodes = data["strip_nodes"]

    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#31688e", "#26828e", "#35b779", "#8fd744", "#fde725"]

    # Each group of eight strips belongs to one of the five body modules.
    for strip_id, node_ids in enumerate(strip_nodes):
        pts = rest[node_ids]
        color = colors[min(strip_id // 8, len(colors) - 1)]
        ax.plot(
            pts[:, 0] * 1000,
            pts[:, 1] * 1000,
            pts[:, 2] * 1000,
            color=color,
            linewidth=2.2,
            alpha=0.9,
        )

    # The ten unique singleton x coordinates in rest are the rigid partition planes.
    x_counts = {}
    for x in np.round(rest[:, 0], 9):
        x_counts[float(x)] = x_counts.get(float(x), 0) + 1
    plate_x = sorted([x for x, count in x_counts.items() if count == 1], reverse=True)
    for x in plate_x:
        add_plate(ax, x * 1000)

    # Schematic four tendon lines through the four anchor/guides positions.
    anchor_yz = np.array([[0.0285, 0.039], [0.0285, -0.039], [-0.0285, 0.039], [-0.0285, -0.039]])
    for y, z in anchor_yz:
        ax.plot(
            np.asarray(plate_x) * 1000,
            np.full(len(plate_x), y * 1000),
            np.full(len(plate_x), z * 1000 + 134),
            color="#e08e0b",
            linewidth=1.2,
            alpha=0.75,
        )

    # Two passive wheels per partition, shown schematically at both lateral sides.
    for x in plate_x:
        add_wheel(ax, x * 1000, -62)
        add_wheel(ax, x * 1000, 62)

    ax.set_xlabel("longitudinal x (mm)")
    ax.set_ylabel("lateral y (mm)")
    ax.set_zlabel("height z (mm)")
    ax.set_title("Current v5 five-body parameterized robot topology\n"
                 "40 pre-curved strips · 10 plates · 20 passive wheels · 4 tendon spans · 4 yaw joints")
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((820, 220, 260))
    ax.grid(True, alpha=0.25)
    fig.text(
        0.02,
        0.02,
        "Diagnostic topology render from topology_5.npz; schematic plates/wheels are for inspection, not CAD or collision geometry.",
        fontsize=9,
        color="#4f5963",
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()

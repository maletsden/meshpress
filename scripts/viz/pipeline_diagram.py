"""STRIDE-dup encoder pipeline diagram.

Single linear path: input -> meshlet partition -> global integer
quantization -> strip walk (+ optional IRLP fit) -> per-meshlet bitstream
packer (connectivity + 3 axis-separated residual substreams) -> bitstream.

No boundary/interior split (STRIDE-dup is per-meshlet self-contained).

Renders to docs/figs/pipeline.png

Usage:
    python scripts/viz/pipeline_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


# Shared linear stages (left to right).
STAGES = [
    ("Input\nOBJ / PLY",                                          "#E0E7FF"),
    ("Meshlet\npartitioning\n(joint-learned)",                    "#C7D2FE"),
    ("Global integer\nquantization\n(bbox q12)",                  "#A5B4FC"),
    ("Strip walk +\nvertex classify\n(anchor / delta / IRLP)",    "#818CF8"),
    ("Per-meshlet packer\nGTS connectivity +\n3 axis residual streams", "#6366F1"),
]
OUTPUT = ("Bitstream", "#3730A3")
SIDE = ("optional:\nper-mesh IRLP fit\n(IRLS L1 + int search)", "#FBBF24")


def draw():
    fig, ax = plt.subplots(figsize=(14, 4.3), dpi=240)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0); ax.axis("off")

    n = len(STAGES)
    bw, bh = 0.14, 0.30
    y_center = 0.50
    # Evenly distribute stages, leave room for output box at right.
    x_left = 0.04
    x_right = 0.79
    xs = [x_left + i * (x_right - x_left) / (n - 1) for i in range(n)]

    dark = {"#6366F1", "#4F46E5", "#3730A3", "#818CF8"}

    for (text, color), cx in zip(STAGES, xs):
        box = FancyBboxPatch(
            (cx - bw/2, y_center - bh/2), bw, bh,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9, edgecolor="#1E1B4B", facecolor=color,
        )
        ax.add_patch(box)
        tc = "white" if color in dark else "#1E1B4B"
        ax.text(cx, y_center, text, ha="center", va="center",
                fontsize=10, color=tc)

    # Linear arrows between stages.
    for i in range(n - 1):
        x0 = xs[i] + bw/2 + 0.003
        x1 = xs[i+1] - bw/2 - 0.003
        ax.add_patch(FancyArrowPatch(
            (x0, y_center), (x1, y_center), arrowstyle="-|>",
            mutation_scale=14, color="#1E1B4B", linewidth=1.2,
        ))

    # Output box at the right.
    out_x = 0.94
    out_w, out_h = 0.10, 0.30
    ax.add_patch(FancyBboxPatch(
        (out_x - out_w/2, y_center - out_h/2), out_w, out_h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9, edgecolor="#1E1B4B", facecolor=OUTPUT[1],
    ))
    ax.text(out_x, y_center, OUTPUT[0], ha="center", va="center",
            fontsize=10, weight="bold", color="white")
    ax.add_patch(FancyArrowPatch(
        (xs[-1] + bw/2 + 0.003, y_center),
        (out_x - out_w/2 - 0.003, y_center),
        arrowstyle="-|>", mutation_scale=14,
        color="#1E1B4B", linewidth=1.2,
    ))

    # Optional IRLP fit branch (sidecar above the strip-walk stage).
    side_cx = xs[3]               # below strip-walk stage
    side_cy = 0.13
    side_w, side_h = 0.20, 0.16
    ax.add_patch(FancyBboxPatch(
        (side_cx - side_w/2, side_cy - side_h/2), side_w, side_h,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        linewidth=0.9, edgecolor="#92400E", facecolor=SIDE[1],
        linestyle="--",
    ))
    ax.text(side_cx, side_cy, SIDE[0], ha="center", va="center",
            fontsize=9, color="#1E1B4B", style="italic")
    # Dashed arrow up to the per-meshlet packer (predictor weights flow into encode).
    ax.add_patch(FancyArrowPatch(
        (side_cx + side_w/2 + 0.003, side_cy),
        (xs[4] - bw/2 - 0.003, y_center - bh/2 - 0.005),
        arrowstyle="-|>", mutation_scale=12,
        color="#92400E", linewidth=1.0, linestyle="--",
        connectionstyle="arc3,rad=-0.10",
    ))
    ax.text((side_cx + xs[4]) / 2 + 0.01, side_cy + side_h/2 + 0.04,
            "(n, K) header",
            ha="center", va="center", fontsize=8, color="#92400E",
            style="italic")

    out = FIGS / "pipeline.png"
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    draw()

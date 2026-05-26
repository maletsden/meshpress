"""Plot E — VRAM stacked bars per mesh.

Reads bench_stride_decode_sweep.csv. Stack order (bottom -> top):
  bitstream / vertex buffer / index buffer / scratch. Peak GPU
allocation is overlaid as a thin marker line.

Usage:
    python scripts/viz/plot_vram.py
Out: docs/figs/vram.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

import sys as _sys
_sys.path.insert(0, str(ROOT))
from utils.bench_config import csv_suffix  # noqa: E402
_SUFFIX = csv_suffix()
CSV = ROOT / f"bench_stride_decode_sweep{_SUFFIX}.csv"

MESH_LABEL = {
    "assets/fandisk.obj": "fandisk", "assets/stanford-bunny.obj": "s-bunny",
    "assets/horse.obj": "horse", "assets/Monkey.obj": "Monkey",
    "assets/happy_buddha.obj": "buddha", "assets/crab.obj": "crab",
    "assets/tank.obj": "tank", "assets/xyzrgb_dragon.obj": "dragon",
}

STACK = [
    ("bitstream_b", "Bitstream",       "#DC2626"),
    ("vbuf_b",      "Vertex buffer",   "#0EA5E9"),
    ("ibuf_b",      "Index buffer",    "#16A34A"),
    ("scratch_b",   "Scratch / meta",  "#F59E0B"),
]


def main():
    df = pd.read_csv(CSV)
    labels = [MESH_LABEL.get(m, m) for m in df["mesh"]]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=240)
    bottom = np.zeros(len(df))
    for col, label, color in STACK:
        vals = df[col].to_numpy() / 1e6
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="black",
               linewidth=0.4, label=label)
        bottom += vals

    peak_mb = df["peak_b"].to_numpy() / 1e6
    ax.scatter(x, peak_mb, marker="_", color="black", s=380, lw=2.0,
               zorder=4, label="cudaMemGetInfo peak")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GPU memory (MB)")
    ax.set_title("STRIDE decoder VRAM footprint per mesh")
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    for i, (s, p) in enumerate(zip(bottom, peak_mb)):
        ax.text(i, s + 0.04 * max(bottom.max(), 1.0),
                f"{s:.1f}", ha="center", va="bottom", fontsize=7,
                color="#374151")

    out = FIGS / f"vram{_SUFFIX}.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
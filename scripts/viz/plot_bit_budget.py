"""Plot C — bit budget stacked bars per mesh.

Reads bench_bit_budget.csv. Stack order (bottom -> top):
  residual / connectivity / boundary / header.

Usage:
    python scripts/viz/plot_bit_budget.py
Out: docs/figs/bit_budget.png
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
CSV = ROOT / f"bench_bit_budget{_SUFFIX}.csv"

MESH_LABEL = {
    "assets/fandisk.obj": "fandisk", "assets/stanford-bunny.obj": "s-bunny",
    "assets/horse.obj": "horse", "assets/Monkey.obj": "Monkey",
    "assets/happy_buddha.obj": "buddha", "assets/crab.obj": "crab",
    "assets/tank.obj": "tank", "assets/xyzrgb_dragon.obj": "dragon",
}

STACK = [
    ("bpv_residual",     "Residual stream",  "#DC2626"),
    ("bpv_connectivity", "Connectivity",     "#0EA5E9"),
    ("bpv_boundary",     "Boundary refs",    "#16A34A"),
    ("bpv_header",       "Meshlet header",   "#F59E0B"),
]


def main():
    df = pd.read_csv(CSV)
    labels = [MESH_LABEL.get(m, m) for m in df["mesh"]]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=240)
    bottom = np.zeros(len(df))
    for col, label, color in STACK:
        vals = df[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="black",
               linewidth=0.4, label=label)
        bottom += vals

    for i, total in enumerate(bottom):
        ax.text(i, total + 0.6, f"{total:.1f}",
                ha="center", va="bottom", fontsize=8, color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bits per vertex")
    ax.set_title("STRIDE bit-budget per mesh (4 sections)")
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    out = FIGS / f"bit_budget{_SUFFIX}.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
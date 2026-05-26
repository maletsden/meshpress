"""Plot B — decode milliseconds grouped bars (per mesh, per codec).

Reads bench_competitors.csv. Bars are log-y because the CPU/GPU gap
spans ~3 orders of magnitude. DGF dec_us is derived from the cited
mtps when missing.

Usage:
    python scripts/viz/plot_decode_bars.py
Out: docs/figs/decode_bars.png
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
CSV = ROOT / f"bench_competitors{_SUFFIX}.csv"

MESH_ORDER = [
    "fandisk.obj", "stanford-bunny.obj", "horse.obj",
    "Monkey.obj", "happy_buddha.obj", "crab.obj",
    "tank.obj", "xyzrgb_dragon.obj",
]
MESH_LABEL = {
    "fandisk.obj": "fandisk", "stanford-bunny.obj": "s-bunny",
    "horse.obj": "horse", "Monkey.obj": "Monkey",
    "happy_buddha.obj": "buddha", "crab.obj": "crab",
    "tank.obj": "tank", "xyzrgb_dragon.obj": "dragon",
}

CODECS = [
    ("ParaDelta v5 (ours)", "#DC2626", "STRIDE (ours)"),
    ("DGF tb12",            "#7C3AED", "AMD DGF tb12"),
    ("meshopt q12 (CPU)",   "#16A34A", "meshopt q12 (CPU)"),
    ("Corto v12",           "#F59E0B", "Corto v12 (CPU)"),
    ("Draco q12 L7",        "#0EA5E9", "Draco q12 L7 (CPU)"),
]


def main():
    df = pd.read_csv(CSV)
    n_meshes = len(MESH_ORDER)
    n_codecs = len(CODECS)
    width = 0.82 / n_codecs
    x = np.arange(n_meshes)

    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=240)
    for i, (codec, color, label) in enumerate(CODECS):
        ys = []
        for mesh in MESH_ORDER:
            sel = df[(df["name"] == codec) & (df["mesh"] == mesh)]
            if len(sel) == 0:
                ys.append(np.nan); continue
            row = sel.iloc[0]
            dec_us = row["dec_us"]
            if pd.notna(dec_us):
                ys.append(float(dec_us) / 1000.0)
            elif pd.notna(row.get("mtps")) and pd.notna(row.get("n_t")):
                ys.append(float(row["n_t"]) / float(row["mtps"]) / 1000.0)
            else:
                ys.append(np.nan)
        ax.bar(x + (i - (n_codecs - 1) / 2) * width, ys, width,
               label=label, color=color, edgecolor="black", linewidth=0.3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([MESH_LABEL[m] for m in MESH_ORDER])
    ax.set_ylabel("Decode time (ms, log scale)")
    ax.set_title("Decode time per mesh per codec")
    ax.grid(True, axis="y", which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.92)

    out = FIGS / f"decode_bars{_SUFFIX}.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()

"""Pareto scatter (BPV vs M tris/s) for paper v4 — STRIDE-dup vs competitors."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

CSV = ROOT / "bench_competitors_q12bbox_v4.csv"

CODEC_STYLE = {
    "STRIDE-dup (ours)":  ("#DC2626", "*", 240, "STRIDE (ours)"),
    "Draco q12 L7":        ("#0EA5E9", "o", 70,  "Draco q12 L7"),
    "meshopt q12 (CPU)":   ("#16A34A", "s", 70,  "meshopt q12"),
    "DGF tb12":            ("#7C3AED", "^", 100, "AMD DGF tb12"),
    "Corto v12":           ("#F59E0B", "v", 70,  "Corto v12"),
}


def pareto_front(points):
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    front = []
    best_y = float("inf")
    for x, y in pts:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def main():
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["bpv", "mtps"])
    df = df[df["mtps"] > 0]

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=240)

    all_points = []
    for codec, sub in df.groupby("name"):
        if codec not in CODEC_STYLE:
            continue
        color, marker, size, label = CODEC_STYLE[codec]
        ax.scatter(sub["bpv"], sub["mtps"], c=color, marker=marker,
                   s=size, alpha=0.85, edgecolor="black", linewidth=0.5,
                   label=label, zorder=3)
        for _, row in sub.iterrows():
            all_points.append((row["bpv"], 1.0 / row["mtps"]))

    front = pareto_front(all_points)
    if front:
        fx = [p[0] for p in front]
        fy = [1.0 / p[1] for p in front]
        ax.plot(fx, fy, ls="--", lw=1.1, color="#374151",
                label="Pareto frontier", zorder=2)

    stride = df[df["name"] == "STRIDE-dup (ours)"]
    for _, row in stride.iterrows():
        ax.annotate(row["mesh"].replace(".obj", ""),
                    (row["bpv"], row["mtps"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7, color="#7F1D1D")

    ax.set_yscale("log")
    ax.set_xlabel("Bits per vertex (lower = smaller bytes)")
    ax.set_ylabel("Decode throughput M tris/s (higher = faster)")
    ax.set_title("STRIDE vs competitors — Pareto frontier across 8 meshes (q12, RTX 3090)")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)

    out = FIGS / "pareto_v4.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()

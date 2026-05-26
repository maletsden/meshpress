"""VRAM stacked bars per mesh for paper v4 (STRIDE-dup)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIGS = ROOT / "docs" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

BLOB_DIR = ROOT / "bench_cpp" / "blobs" / "dup"
SIDE_BYTES_PER_MESHLET = 50
HEADER_BYTES = 88

MESHES = [
    ("fandisk", "fandisk"),
    ("stanford-bunny", "s-bunny"),
    ("horse", "horse"),
    ("Monkey", "Monkey"),
    ("happy_buddha", "buddha"),
    ("crab", "crab"),
    ("tank", "tank"),
    ("xyzrgb_dragon", "dragon"),
]


def main():
    labels = []
    bs, side, vbuf, ibuf = [], [], [], []
    for name, lbl in MESHES:
        blob = BLOB_DIR / f"{name}.dup.blob"
        if not blob.exists():
            continue
        with open(blob, "rb") as f:
            _m, _v, n_meshlets, n_v_total, n_t_total, buf_size = struct.unpack(
                "<IIIIII", f.read(24))
        bitstream_mb = (HEADER_BYTES + buf_size) / (1 << 20)
        side_mb      = (SIDE_BYTES_PER_MESHLET * n_meshlets) / (1 << 20)
        vbuf_mb      = (n_v_total * 3 * 4) / (1 << 20)
        ibuf_mb      = (n_t_total * 3 * 4) / (1 << 20)
        labels.append(lbl)
        bs.append(bitstream_mb)
        side.append(side_mb)
        vbuf.append(vbuf_mb)
        ibuf.append(ibuf_mb)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=240)
    bottom = np.zeros(len(labels))
    parts = [
        (np.array(bs),   "Bitstream",   "#DC2626"),
        (np.array(side), "Side tables", "#7C3AED"),
        (np.array(vbuf), "Vertex buf",  "#0EA5E9"),
        (np.array(ibuf), "Index buf",   "#16A34A"),
    ]
    for vals, label, color in parts:
        ax.bar(x, vals, bottom=bottom, label=label, color=color,
               edgecolor="black", linewidth=0.3)
        bottom += vals

    for i, total in enumerate(bottom):
        ax.text(x[i], total + max(bottom) * 0.01, f"{total:.1f} MB",
                ha="center", fontsize=8, color="#111827")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("VRAM (MB)")
    ax.set_title("STRIDE decoder VRAM footprint per mesh (q12, RTX 3090)")
    ax.grid(True, axis="y", which="major", ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    out = FIGS / "vram_v4.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()

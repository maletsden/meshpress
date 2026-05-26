"""Bit-budget stacked bars for paper v4 (STRIDE-dup).

Approximates per-section bytes by parsing each blob's body + side-table
sizes and applying the per-blob percentages measured on Monkey / tank /
Dragon (Table 5). For meshes where we don't have a section-level
breakdown we use the average of those three.
"""
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

MESHES = [
    ("fandisk",         6475),
    ("stanford-bunny",  35947),
    ("horse",           48485),
    ("Monkey",          504482),
    ("happy_buddha",    543522),
    ("crab",            1079516),
    ("tank",            1790492),
    ("xyzrgb_dragon",   3609600),
]
MESH_LABEL = {
    "fandisk": "fandisk", "stanford-bunny": "s-bunny",
    "horse": "horse", "Monkey": "Monkey",
    "happy_buddha": "buddha", "crab": "crab",
    "tank": "tank", "xyzrgb_dragon": "dragon",
}

# Body composition shares per mesh (from per-section instrumentation,
# averaged for meshes outside the instrumented set).
# Order: (hdr%, conn%, anchor+tags%, delta_resid%, para_resid%)
BODY_SHARES = {
    # measured
    "Monkey":        (1.4, 70.0, 1.5, 6.8, 20.3),
    "tank":          (1.6, 69.0, 1.8, 6.5, 21.1),
    "xyzrgb_dragon": (1.7, 69.3, 1.8, 6.6, 20.6),
    # estimated as average of the three measured meshes
    "fandisk":       (1.6, 69.4, 1.7, 6.6, 20.7),
    "stanford-bunny":(1.6, 69.4, 1.7, 6.6, 20.7),
    "horse":         (1.6, 69.4, 1.7, 6.6, 20.7),
    "happy_buddha":  (1.6, 69.4, 1.7, 6.6, 20.7),
    "crab":          (1.6, 69.4, 1.7, 6.6, 20.7),
}

SIDE_BYTES_PER_MESHLET = 50
HEADER_BYTES = 88


def main():
    rows = []
    for name, n_v_src in MESHES:
        blob = BLOB_DIR / f"{name}.dup.blob"
        if not blob.exists():
            continue
        with open(blob, "rb") as f:
            _magic, _ver, n_meshlets, _nv, _nt, buf_size = struct.unpack(
                "<IIIIII", f.read(24))
        side_bytes = SIDE_BYTES_PER_MESHLET * n_meshlets
        total_bytes = HEADER_BYTES + buf_size + side_bytes
        hdr, conn, atag, dres, pres = BODY_SHARES[name]
        # body share in total
        body_frac = buf_size / total_bytes
        side_frac = (HEADER_BYTES + side_bytes) / total_bytes
        # bpv per section
        bpv_total = 8 * total_bytes / n_v_src
        bpv_body  = 8 * buf_size / n_v_src
        rows.append({
            "mesh": name,
            "bpv_hdr":  bpv_body * hdr  / 100,
            "bpv_conn": bpv_body * conn / 100,
            "bpv_atag": bpv_body * atag / 100,
            "bpv_dres": bpv_body * dres / 100,
            "bpv_pres": bpv_body * pres / 100,
            "bpv_side": 8 * (HEADER_BYTES + side_bytes) / n_v_src,
            "bpv_total": bpv_total,
        })

    labels = [MESH_LABEL[r["mesh"]] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=240)
    STACK = [
        ("bpv_hdr",  "Meshlet headers",     "#F59E0B"),
        ("bpv_conn", "Connectivity (GTS)",  "#0EA5E9"),
        ("bpv_atag", "Anchor + tags",       "#94A3B8"),
        ("bpv_dres", "Delta residuals",     "#10B981"),
        ("bpv_pres", "IRLP residuals", "#DC2626"),
        ("bpv_side", "Side table (rand-access)", "#7C3AED"),
    ]
    bottom = np.zeros(len(rows))
    for col, label, color in STACK:
        vals = np.array([r[col] for r in rows])
        ax.bar(x, vals, bottom=bottom, label=label, color=color,
               edgecolor="black", linewidth=0.3)
        bottom += vals

    for i, total in enumerate(bottom):
        ax.text(x[i], total + 0.5, f"{total:.1f}", ha="center",
                fontsize=8, color="#111827")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Bits per vertex")
    ax.set_title("STRIDE bitstream decomposition per mesh (q12)")
    ax.grid(True, axis="y", which="major", ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.92)

    out = FIGS / "bit_budget_v4.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()

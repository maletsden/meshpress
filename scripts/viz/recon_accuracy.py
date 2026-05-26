"""Render STRIDE reconstructions at multiple precision levels.

For each mesh: encode/decode at three bbox-relative bit widths, render
the original and three reconstructions side-by-side, annotated with BPV
+ max NN error. Output: docs/figs/recon_<mesh>_<angle>.png

Usage:
    python scripts/viz/recon_accuracy.py
    python scripts/viz/recon_accuracy.py assets/bunny.obj
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reader.reader import Reader
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5, decode_paradelta_v5

FIGS_DIR = ROOT / "docs" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Bbox-relative bit widths to compare. precision_error = 1/(2^b - 1)
# gives a uniform b-bit per-axis grid matched to Draco-style q<b>.
BIT_WIDTHS = [10, 12, 14]
ANGLES = [
    ("front",   (10,  -90)),
    ("3q",      (20,  -45)),
    ("side",    (10,    0)),
    ("3q_back", (20,  135)),
    ("top",     (75,  -90)),
]
DPI = 240


def _render_panel(ax, verts, tris, title: str, elev: float, azim: float):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = verts[tris]
    coll = Poly3DCollection(polys, facecolors=(0.7, 0.75, 0.85),
                             edgecolors="k", linewidths=0.03)
    ax.add_collection3d(coll)
    mn = verts.min(0); mx = verts.max(0)
    rng = (mx - mn).max() / 2.2
    c = (mx + mn) / 2
    ax.set_xlim(c[0] - rng, c[0] + rng)
    ax.set_ylim(c[1] - rng, c[1] + rng)
    ax.set_zlim(c[2] - rng, c[2] + rng)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=2)


def _process(mesh_path: Path):
    name = mesh_path.stem
    m = Reader.read_from_file(str(mesh_path))
    src_verts = np.array([[v.x, v.y, v.z] for v in m.vertices],
                          dtype=np.float32)
    src_tris = np.array([[t.a, t.b, t.c] for t in m.triangles],
                         dtype=np.int64)
    n_v = len(src_verts); n_t = len(src_tris)
    print(f"=== {name} n_v={n_v:,} n_t={n_t:,} ===")

    panels = []   # list of (verts, tris, title)
    panels.append((src_verts, src_tris, f"Original  ({n_v:,} verts)"))
    bbox_extent = float((src_verts.max(0) - src_verts.min(0)).max())
    for b in BIT_WIDTHS:
        pe = 1.0 / float((1 << b) - 1)
        prep = load_or_prepare(str(mesh_path),
                                max_verts=256, max_tris=256,
                                precision_error=pe,
                                precision_mode="bbox_frac",
                                gen_method="joint_learned",
                                strip_method="multiseed", verbose=False)
        data = encode_from_prepared_v5(prep, verbose=False)
        v_dec, t_dec = decode_paradelta_v5(data)
        from scipy.spatial import cKDTree
        tree = cKDTree(src_verts)
        d, idx = tree.query(v_dec, k=1)
        err = float(d.max())
        bpv = len(data) * 8 / n_v
        step = bbox_extent / float((1 << b) - 1)
        title = (f"q{b}  step={step:.3g}  BPV={bpv:.1f}\n"
                 f"maxErr={err:.4g}  ({len(data):,} B)")
        panels.append((v_dec.astype(np.float32),
                       t_dec.astype(np.int64), title))
        print(f"  q{b}: step={step:.4g} BPV={bpv:.2f} err={err:.4g}")

    for ang_name, (elev, azim) in ANGLES:
        fig = plt.figure(figsize=(15, 4.0))
        for i, (V, T, title) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
            _render_panel(ax, V, T, title, elev, azim)
        out = FIGS_DIR / f"recon_{name}_{ang_name}.png"
        plt.subplots_adjust(left=0, right=1, top=1.0, bottom=0,
                             wspace=-0.10)
        plt.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"    wrote {out.name}")


def main():
    paths = sys.argv[1:] or [
        "assets/bunny.obj",
        "assets/stanford-bunny.obj",
        "assets/Monkey.obj",
    ]
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"missing: {p}"); continue
        try:
            _process(p)
        except Exception as e:
            print(f"  ERR {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
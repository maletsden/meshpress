"""Side-by-side meshlet partition comparison: greedy | tunneling | ours.

One figure per mesh; one viewpoint per mesh. Output:
docs/figs/meshlet_compare_<mesh>.png

Default mesh / angle:
  fandisk        - 3q
  stanford-bunny - top
  eyeball        - 3q

Usage:
    python scripts/viz/meshlet_partitions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reader.reader import Reader
from utils.mesh_clean import clean_mesh
from utils.paradelta_cache import load_or_prepare

# Force tuned (packed) tunneling chunker for this viz so the figure
# matches Table 1.
import utils.meshlet_generator as _mg
from utils.meshlet_tunneling import generate_meshlets_tunneling as _tunneling
_orig_gen = _mg.generate_meshlets
def _patched_gen(*a, **kw):
    if kw.get("method") == "tunneling":
        return _tunneling(a[0], a[1],
                          max_tris=kw.get("max_tris", 256),
                          max_verts=kw.get("max_verts", 256),
                          time_budget_s=300.0, verbose=False, pack=True)
    return _orig_gen(*a, **kw)
_mg.generate_meshlets = _patched_gen

FIGS_DIR = ROOT / "docs" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
DPI = 240

ALGOS = ["greedy", "tunneling", "joint_learned"]
ALGO_LABEL = {
    "greedy":         "Greedy region-grow",
    "tunneling":      "Tunneling",
    "joint_learned":  "Ours",
}

# (mesh stem, (elev, azim))
DEFAULT_CONFIG = [
    ("fandisk",        (20,  -45)),  # 3q
    ("stanford-bunny", (75,  -90)),  # 3/4 from above (Y-up bunny)
    ("eyeball",        (20,  -45)),  # 3q
]


def _meshlet_colors(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    h = np.arange(n) / max(1, n)
    rng.shuffle(h)
    s = np.full(n, 0.65); v = np.full(n, 0.95)
    return hsv_to_rgb(np.stack([h, s, v], axis=1))


def _face_colors(meshlets, n_tris: int) -> np.ndarray:
    cols = _meshlet_colors(len(meshlets))
    out = np.zeros((n_tris, 3))
    for mid, ml_tris in enumerate(meshlets):
        for ti in ml_tris:
            out[ti] = cols[mid]
    return out


def _draw(ax, verts, tris, fc_rgb, title, elev, azim, axis_swap_yz=False):
    if axis_swap_yz:
        # Bunny OBJ is Y-up; swap so Y -> Z and we can use matplotlib's
        # Z-up elev convention to get an actual top-down view.
        verts = verts[:, [0, 2, 1]].copy()
    polys = verts[tris]
    coll = Poly3DCollection(polys, facecolors=fc_rgb,
                             edgecolors="k", linewidths=0.04)
    ax.add_collection3d(coll)
    mn = verts.min(0); mx = verts.max(0)
    rng = (mx - mn).max() / 2.2
    c = (mx + mn) / 2
    ax.set_xlim(c[0] - rng, c[0] + rng)
    ax.set_ylim(c[1] - rng, c[1] + rng)
    ax.set_zlim(c[2] - rng, c[2] + rng)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=2)


def _process(mesh_path: Path, elev: float, azim: float):
    name = mesh_path.stem
    print(f"=== {name} ===")
    # Match encoder prep pipeline exactly (clean_mesh -> prepare_paradelta)
    # so meshlet counts shown here line up with §3.2 Table 1.
    m = Reader.read_from_file(str(mesh_path))
    m, _ = clean_mesh(m, verbose=False)
    verts = np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float32)
    tris = np.array([[t.a, t.b, t.c] for t in m.triangles], dtype=np.int64)
    n_t = len(tris)

    panels = []
    for algo in ALGOS:
        prep = load_or_prepare(
            str(mesh_path),
            max_verts=256, max_tris=256, precision_error=0.0005,
            gen_method=algo, strip_method="multiseed",
            clean=True, verbose=False,
            force=(algo == "tunneling"),
        )
        meshlets = [plan["ml_tris_global"] for plan in prep["plans"]]
        fc_rgb = _face_colors(meshlets, n_t)
        title = f"{ALGO_LABEL[algo]}  ({len(meshlets):,} meshlets)"
        panels.append((fc_rgb, title))
        print(f"  {algo}: {len(meshlets):,} meshlets")

    fig = plt.figure(figsize=(13.5, 4.4))
    for i, (fc_rgb, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _draw(ax, verts, tris, fc_rgb, title, elev, azim)
    out = FIGS_DIR / f"meshlet_compare_{name}.png"
    plt.subplots_adjust(left=0, right=1, top=1.0, bottom=0, wspace=-0.10)
    plt.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    if len(sys.argv) > 1:
        # User passed explicit mesh paths; inherit configured angle if mesh
        # is in DEFAULT_CONFIG, else fall back to 3q (20, -45).
        cfg_map = {stem: ang for stem, ang in DEFAULT_CONFIG}
        cfg = [(Path(p).stem, cfg_map.get(Path(p).stem, (20, -45))) for p in sys.argv[1:]]
        path_map = {Path(p).stem: Path(p) for p in sys.argv[1:]}
    else:
        cfg = DEFAULT_CONFIG
        path_map = {stem: Path("assets") / f"{stem}.obj" for stem, _ in cfg}
    for stem, (elev, azim) in cfg:
        p = path_map[stem]
        if not p.exists():
            print(f"missing: {p}"); continue
        try:
            _process(p, elev, azim)
        except Exception as e:
            print(f"  ERR {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()

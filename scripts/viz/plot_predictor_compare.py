"""Side-by-side visualisation: canonical (1,1,-1) parallelogram predictor vs
fitted integer-rational predictor.

Picks a real (a, b, c, v) sample from happy_buddha (the mesh where the fit
deviates most from canonical) and shows the prediction error in 2D after
projecting onto the plane of the (a, b, c) triangle.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5_dup import _walk_meshlet
from encoder._irlp_fit import fit_predictor

MESH = str(ROOT / "assets" / "happy_buddha.obj")
FIGS = ROOT / "docs" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "predictor_compare_cache.npz"


def gather_samples():
    """All (a, b, c, v) para triples in lattice coords for the mesh."""
    prep = load_or_prepare(MESH, max_verts=256, max_tris=256,
                           precision_error=1.0/4096.0,
                           precision_mode="bbox_frac",
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    gc = prep["global_codes"]; plans = prep["plans"]
    A, B, C, V = [], [], [], []
    for plan in plans:
        l2g = np.asarray(plan["local_to_global"], dtype=np.int64)
        tc = gc[l2g].astype(np.int64)
        for v, kind, refs in _walk_meshlet(plan):
            if kind != "para":
                continue
            a, b, c = refs
            A.append(tc[a]); B.append(tc[b]); C.append(tc[c]); V.append(tc[v])
    return (np.array(A, dtype=np.float64), np.array(B, dtype=np.float64),
            np.array(C, dtype=np.float64), np.array(V, dtype=np.float64), prep)


def pred_int(a, b, c, n0, n1, n2, K):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    s = n0*a + n1*b + n2*c
    if K == 0:
        return s
    half = (1 << K) >> 1
    return (s + half) >> K


def project_to_plane(a, b, c, v_canon, v_fit, v_true):
    """PCA projection of all 6 points to 2D — keeps the spatial relationship
    that includes the prediction targets, not just the triangle."""
    pts3 = np.stack([a, b, c, v_canon, v_fit, v_true])
    centroid = pts3.mean(axis=0)
    X = pts3 - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axes = Vt[:2]
    return X @ axes.T


def draw_panel(ax, a2, b2, c2, vpred2, vtrue2, n, K, title,
               err_lat: float, draw_parallelogram: bool):
    """Draw source triangle (a, b, c); optionally close into full parallelogram
    (canonical case). Show prediction + true vertex + residual arrow."""
    if draw_parallelogram:
        # Full parallelogram c → a → pcomp → b → c (4 sides).
        pcomp = a2 + b2 - c2
        quad = np.vstack([c2, a2, pcomp, b2, c2])
        ax.plot(quad[:, 0], quad[:, 1], color="#475569", lw=1.8, alpha=0.85, zorder=2)
        # Lightly shade the parallelogram interior.
        ax.fill(quad[:, 0], quad[:, 1], color="#E2E8F0", alpha=0.5, zorder=1)
    else:
        # Just the triangle (a, b, c). Fitted predictor is NOT a parallelogram.
        tri = np.vstack([a2, b2, c2, a2])
        ax.plot(tri[:, 0], tri[:, 1], color="#475569", lw=1.8, alpha=0.85, zorder=2)
        ax.fill(tri[:, 0], tri[:, 1], color="#E2E8F0", alpha=0.5, zorder=1)
        # Draw thin arrows from each vertex to v_pred with weights.
        for pt, weight, color in zip([a2, b2, c2],
                                      [n[0], n[1], n[2]],
                                      ["#0EA5E9", "#0EA5E9", "#F59E0B"]):
            arr = FancyArrowPatch(pt, vpred2, arrowstyle="->",
                                  mutation_scale=10, color=color,
                                  lw=0.8, alpha=0.5, zorder=2,
                                  linestyle="--")
            ax.add_patch(arr)

    # Predicted v̂
    ax.scatter(*vpred2, s=160, color="#DC2626", zorder=5, edgecolors="white",
               linewidths=1.2, label="$\\hat{v}$ predicted")
    # True v
    ax.scatter(*vtrue2, s=160, color="#16A34A", zorder=5, edgecolors="white",
               linewidths=1.2, marker="X", label="$v$ true")
    # Residual arrow
    arr = FancyArrowPatch(vpred2, vtrue2, arrowstyle="->",
                          mutation_scale=14, color="#DC2626",
                          lw=1.4, alpha=0.85, zorder=4)
    ax.add_patch(arr)

    # Vertex labels with auto-placement above/below depending on position.
    cx = (a2[0] + b2[0] + c2[0]) / 3.0
    cy = (a2[1] + b2[1] + c2[1]) / 3.0
    for pt, lbl in zip([a2, b2, c2], ["a", "b", "c"]):
        off_x = 10 if pt[0] >= cx else -14
        off_y = 8 if pt[1] >= cy else -14
        ax.annotate(lbl, pt, xytext=(off_x, off_y), textcoords="offset points",
                    fontsize=13, fontweight="bold", color="#0F172A")
    ax.annotate("$\\hat{v}$", vpred2, xytext=(10, 4),
                textcoords="offset points", fontsize=13, color="#DC2626")
    ax.annotate("$v$", vtrue2, xytext=(10, -12),
                textcoords="offset points", fontsize=13, color="#16A34A")

    # Vertex dots
    for pt in (a2, b2, c2):
        ax.scatter(*pt, s=60, color="#0F172A", zorder=4)

    # Build formula caption string; placement happens in main() so both
    # panels can share the same figure-coord y and align exactly.
    D = 1 << K if K > 0 else 1
    if K == 0:
        formula = (f"$\\hat{{v}} = a + b - c$\n"
                   f"weights $(n_0, n_1, n_2) = ({n[0]}, {n[1]}, {n[2]})$")
    else:
        formula = ("$\\hat{v} = (n_0 a + n_1 b + n_2 c + 2^{K-1}) \\gg K$\n"
                   f"weights $(n_0, n_1, n_2)/2^K = ({n[0]}, {n[1]}, {n[2]})/{D}$")
    formula += f"\nresidual $\\|v - \\hat{{v}}\\|_2$ = {err_lat:.2f} lattice"

    # Title placement happens in main() via fig.text at fixed figure-coord
    # y so that left and right panel labels align exactly.
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Auto-expand limits to leave headroom around geometry.
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    span_x = xmax - xmin; span_y = ymax - ymin
    pad_x = span_x * 0.15
    pad_y = span_y * 0.20
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y * 0.4)
    return formula


def _load_or_build_cache():
    """Cached samples + fitted predictor. Re-fitting buddha is the slow path
    in this script; the cache lets plot iteration skip the ~minute fit.
    Delete `cache/predictor_compare_cache.npz` to force a rebuild."""
    if CACHE_FILE.exists():
        print(f"Loading cached samples + fit from {CACHE_FILE} ...")
        d = np.load(CACHE_FILE)
        return (d["A"], d["B"], d["C"], d["V"],
                d["n_fit"], d["K_fit"])

    print("Cache miss; gathering buddha samples + fitting predictor.")
    print("Gathering buddha samples...")
    A, B, C, V, prep = gather_samples()
    print(f"  {len(A):,} para tuples")

    print("Fitting predictor...")
    n_fit, K_fit = fit_predictor(prep)

    np.savez(CACHE_FILE,
             A=A, B=B, C=C, V=V,
             n_fit=np.asarray(n_fit, dtype=np.int64),
             K_fit=np.asarray(K_fit, dtype=np.int64))
    print(f"Saved cache to {CACHE_FILE}")
    return A, B, C, V, np.asarray(n_fit, dtype=np.int64), np.asarray(K_fit, dtype=np.int64)


def main():
    A, B, C, V, n_fit, K_fit = _load_or_build_cache()
    print(f"  {len(A):,} para tuples")
    print(f"  fitted weights per axis (n / 2^K):")
    for d, name in enumerate("xyz"):
        K = int(K_fit[d])
        D = 1 << K if K > 0 else 1
        print(f"    {name}: K={K}  n=({n_fit[d,0]:+d},{n_fit[d,1]:+d},{n_fit[d,2]:+d}) / {D}")

    # Score every sample by how much fitted improves over canonical (in
    # integer lattice L2). Pick a sample with strong gain AND well-shaped
    # source triangle (avoid sliver triangles that flatten in 2D projection).
    def axis_pred(a, b, c, n, K):
        return np.stack([
            pred_int(a[:, d], b[:, d], c[:, d],
                     int(n[d, 0]), int(n[d, 1]), int(n[d, 2]), int(K[d]))
            for d in range(3)
        ], axis=1)

    n_can = np.array([[1, 1, -1]] * 3, dtype=np.int64)
    K_can = np.zeros(3, dtype=np.int64)
    pred_can = axis_pred(A, B, C, n_can, K_can)
    pred_fit = axis_pred(A, B, C, n_fit, K_fit)
    res_can = V - pred_can
    res_fit = V - pred_fit
    err_can = np.linalg.norm(res_can, axis=1)
    err_fit = np.linalg.norm(res_fit, axis=1)
    gain = err_can - err_fit

    # Triangle shape score: ratio of min edge to max edge (1 = equilateral, 0 = degenerate).
    ab = np.linalg.norm(B - A, axis=1)
    bc = np.linalg.norm(C - B, axis=1)
    ca = np.linalg.norm(A - C, axis=1)
    edges = np.stack([ab, bc, ca], axis=1)
    shape = edges.min(axis=1) / np.maximum(edges.max(axis=1), 1e-9)

    candidates = np.where((err_can > 5) & (err_can < 25) & (gain > 1.5) &
                          (shape > 0.55))[0]
    if len(candidates) == 0:
        idx = int(np.argmax(gain * (shape > 0.4)))
    else:
        # Largest gain among well-shaped samples.
        idx = int(candidates[np.argmax(gain[candidates])])
    print(f"Sample {idx}: err_canon={err_can[idx]:.2f} err_fit={err_fit[idx]:.2f} "
          f"shape={shape[idx]:.2f}")

    a, b, c, v = A[idx], B[idx], C[idx], V[idx]
    vcan = pred_can[idx]
    vfit = pred_fit[idx]

    # Project all to 2D plane (PCA).
    coords = project_to_plane(a, b, c, vcan, vfit, v)
    a2, b2, c2, vcan2, vfit2, vtrue2 = coords

    err_can = float(np.linalg.norm(vcan2 - vtrue2))
    err_fit = float(np.linalg.norm(vfit2 - vtrue2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=240)
    title_left = "Canonical Touma–Gotsman: $(n, K) = ((1,1,-1), 0)$"
    formula_left = draw_panel(axes[0], a2, b2, c2, vcan2, vtrue2,
               (1, 1, -1), 0,
               title_left,
               err_can, draw_parallelogram=True)
    nfit_disp = (int(n_fit[0, 0]), int(n_fit[0, 1]), int(n_fit[0, 2]))
    kfit_disp = int(K_fit[0])
    title_right = f"IRLP (happy_buddha, x-axis): $(n, K) = ({nfit_disp}, {kfit_disp})$"
    formula_right = draw_panel(axes[1], a2, b2, c2, vfit2, vtrue2,
               nfit_disp, kfit_disp,
               title_right,
               err_fit, draw_parallelogram=False)
    panel_titles = (title_left, title_right)

    # Equalize ylim across both panels so titles render at the same height.
    ymin = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    # Single shared legend below both panels.
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#DC2626", markeredgecolor="white",
               markersize=10, label="$\\hat{v}$ predicted"),
        Line2D([0], [0], marker="X", color="w",
               markerfacecolor="#16A34A", markeredgecolor="white",
               markersize=10, label="$v$ true"),
        Line2D([0], [0], color="#DC2626", lw=1.4,
               label="residual (entropy-coded)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.30))
    fig.suptitle("Canonical parallelogram vs Integer-Rational Linear Predictor (IRLP)",
                 fontsize=12, y=0.97)

    fig.tight_layout()
    # Tight stack: suptitle ~0.97 / panel titles ~0.90 / subplot 0.87-0.34
    # / formulas ~0.32 / legend ~0.07.
    fig.subplots_adjust(top=0.87, bottom=0.45, wspace=-0.10)

    # Panel titles in figure coords so left/right labels align exactly.
    for ax, title in zip(axes, panel_titles):
        bbox = ax.get_position()
        cx = (bbox.x0 + bbox.x1) / 2
        fig.text(cx, 0.85, title, ha="center", va="bottom",
                 fontsize=11, color="#0F172A")

    # Formula captions in figure coords; va="top" with shared y anchors each
    # caption's top right under its panel, and both end at the same bottom y
    # since each is three lines at identical font size.
    for ax, formula in zip(axes, (formula_left, formula_right)):
        bbox = ax.get_position()
        cx = (bbox.x0 + bbox.x1) / 2
        fig.text(cx, 0.50, formula, ha="center", va="top",
                 fontsize=9, family="monospace", color="#0F172A")

    out = FIGS / "predictor_compare.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()

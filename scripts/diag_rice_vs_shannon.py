"""Validate Rice-vs-Shannon gap claim across meshes.

For each meshlet, per axis, compute:
  - n_int = #interior residuals
  - Shannon entropy bits = -sum p_i * log2(p_i) * n_int (over zigzagged
    residual distribution)
  - Rice bits = best-k Rice cost (matches what encoder picks)
  - Fixed bits = ceil(log2(range+1)) * n_int (header excluded)
  - Distribution shape stats: max, mean, kurtosis (heavy-tail vs
    geometric)

Aggregate per mesh. If Dragon has a much larger (Rice / Shannon) ratio
than Monkey/bunny, the user's hypothesis is confirmed: Rice diverges
from optimal entropy coding on Dragon's residual distribution.

Usage:
    python scripts/diag_rice_vs_shannon.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import _interior_pass_strip
from encoder.paradelta_codec import _fit_linear3, _best_rice_k, _best_eg_k
from utils.residual_entropy import _zigzag


def _shannon_bits(u: np.ndarray) -> float:
    """Shannon entropy * n  =  total bits needed under perfect entropy
    coder on the empirical distribution of zigzagged residuals."""
    if u.size == 0:
        return 0.0
    _, counts = np.unique(u, return_counts=True)
    p = counts / counts.sum()
    H = float(-np.sum(p * np.log2(p)))   # bits/symbol
    return H * u.size


def _kurtosis(arr: np.ndarray) -> float:
    if arr.size < 4:
        return 0.0
    m = arr.mean(); s = arr.std()
    if s < 1e-12:
        return 0.0
    return float((((arr - m) / s) ** 4).mean() - 3.0)


def _analyze(prep, name: str, max_meshlets: int | None = None):
    plans = prep["plans"]
    vn = prep["vn"]
    bnd_recon = prep["bnd_recon_norm"]
    delta = 2.0 * float(prep["per_coord_err"])

    # Pass 1: TG baseline to fit lin3
    samples = []
    for plan in plans[:max_meshlets] if max_meshlets else plans:
        _, sm = _interior_pass_strip(plan, vn, bnd_recon, delta, w3=None)
        samples.extend(sm)
    lin3_w = _fit_linear3(samples)

    n_int_total = 0
    shannon_total = 0.0
    rice_total = 0.0
    eg_total = 0.0
    fixed_total = 0.0
    abs_max_global = 0
    kurt_acc = []

    plans_iter = plans[:max_meshlets] if max_meshlets else plans
    for plan in plans_iter:
        codes, _ = _interior_pass_strip(plan, vn, bnd_recon, delta,
                                          w3=lin3_w)
        if codes.shape[0] == 0:
            continue
        for d in range(3):
            arr = codes[:, d]
            n = arr.size
            n_int_total += n
            u = _zigzag(arr)
            shannon_total += _shannon_bits(u)
            _, rc = _best_rice_k(u)
            rice_total += rc
            _, ec = _best_eg_k(u)
            eg_total += ec
            rng = int(arr.max() - arr.min())
            bw = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
            fixed_total += n * bw
            abs_max_global = max(abs_max_global, int(np.abs(arr).max()))
            kurt_acc.append(_kurtosis(u.astype(np.float64)))

    print(f"\n=== {name} ===")
    print(f"  meshlets used: {len(plans_iter):,}   "
          f"n_int_total: {n_int_total:,}")
    print(f"  lin3 = ({float(lin3_w[0]):.4f}, "
          f"{float(lin3_w[1]):.4f}, {float(lin3_w[2]):.4f})")
    print(f"  max |code| across mesh: {abs_max_global:,}")
    print(f"  median per-axis-per-meshlet kurtosis(u): "
          f"{float(np.median(kurt_acc)):.2f}")
    print(f"  TOTAL bits per coding scheme:")
    print(f"    Shannon (entropy lower bound): "
          f"{shannon_total:14,.0f}  "
          f"({shannon_total/n_int_total:6.3f} bpv-axis)")
    print(f"    Rice (encoder picks):          "
          f"{rice_total:14,.0f}  "
          f"({rice_total/n_int_total:6.3f} bpv-axis)  "
          f"ratio={rice_total/shannon_total:.3f}")
    print(f"    Exp-Golomb (encoder cand):     "
          f"{eg_total:14,.0f}  "
          f"({eg_total/n_int_total:6.3f} bpv-axis)  "
          f"ratio={eg_total/shannon_total:.3f}")
    print(f"    Fixed-width:                   "
          f"{fixed_total:14,.0f}  "
          f"({fixed_total/n_int_total:6.3f} bpv-axis)  "
          f"ratio={fixed_total/shannon_total:.3f}")
    best = min(rice_total, eg_total, fixed_total)
    print(f"  best-of-three vs Shannon: {best/shannon_total:.3f}x  "
          f"(waste: {(best-shannon_total)/shannon_total*100:.1f}%)")
    return {
        "mesh": name,
        "n_int": n_int_total,
        "shannon": shannon_total,
        "rice": rice_total,
        "eg": eg_total,
        "fixed": fixed_total,
        "best": best,
        "max_code": abs_max_global,
        "kurtosis_median": float(np.median(kurt_acc)),
    }


def main():
    meshes = [
        ("bunny",          "assets/bunny.obj"),
        ("stanford-bunny", "assets/stanford-bunny.obj"),
        ("Monkey",         "assets/Monkey.obj"),
        ("xyzrgb_dragon",  "assets/xyzrgb_dragon.obj"),
    ]
    rows = []
    for name, path in meshes:
        print(f"\n[loading {name} ...]")
        prep = load_or_prepare(path, max_verts=256, max_tris=256,
                                precision_error=0.0005,
                                gen_method="joint_learned",
                                strip_method="multiseed",
                                clean=True, verbose=False)
        rows.append(_analyze(prep, name))

    print(f"\n\n==== Summary ====")
    print(f"{'mesh':<18} {'max|c|':>9} {'kurt':>6} "
          f"{'Shannon-bpva':>14} {'Best-bpva':>11} "
          f"{'Waste%':>8}")
    for r in rows:
        shannon_bpva = r["shannon"] / r["n_int"]
        best_bpva = r["best"] / r["n_int"]
        waste = (r["best"] - r["shannon"]) / r["shannon"] * 100
        print(f"{r['mesh']:<18} {r['max_code']:>9,} "
              f"{r['kurtosis_median']:>6.1f} "
              f"{shannon_bpva:>14.3f} {best_bpva:>11.3f} "
              f"{waste:>7.1f}%")


if __name__ == "__main__":
    main()
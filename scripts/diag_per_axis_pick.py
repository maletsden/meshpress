"""Measure per-axis coder picking gain.

For each meshlet/axis: compute Rice-best-k, EG-best-k, fixed-width.
- Rice-only total = sum(rice over axes)
- EG-only total = sum(eg over axes)
- Per-axis best = sum(min(rice, eg, fixed) per axis)
- Shannon lower bound = sum(H * n per axis)

Reports each total and how often each coder wins.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import _interior_pass_strip
from encoder.paradelta_codec import _fit_linear3, _best_rice_k, _best_eg_k
from utils.residual_entropy import _zigzag, _rice_bits, _exp_golomb_bits


def _shannon_bits(u):
    if u.size == 0:
        return 0.0
    _, counts = np.unique(u, return_counts=True)
    p = counts / counts.sum()
    H = float(-np.sum(p * np.log2(p)))
    return H * u.size


def _analyze(prep, name):
    plans = prep["plans"]
    vn = prep["vn"]
    bnd_recon = prep["bnd_recon_norm"]
    delta = 2.0 * float(prep["per_coord_err"])

    samples = []
    for plan in plans:
        _, sm = _interior_pass_strip(plan, vn, bnd_recon, delta, w3=None)
        samples.extend(sm)
    lin3_w = _fit_linear3(samples)

    n_int = 0
    shannon = 0.0
    rice_only = 0
    eg_only = 0
    fixed_only = 0
    per_axis_best = 0
    win = Counter()
    rice_k_picks = Counter()
    eg_k_picks = Counter()

    for plan in plans:
        codes, _ = _interior_pass_strip(plan, vn, bnd_recon, delta,
                                          w3=lin3_w)
        if codes.shape[0] == 0:
            continue
        for d in range(3):
            arr = codes[:, d]
            n = arr.size
            n_int += n
            u = _zigzag(arr)
            shannon += _shannon_bits(u)
            kr, rc = _best_rice_k(u)
            ke, ec = _best_eg_k(u)
            rng = int(arr.max() - arr.min())
            bw = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
            fc = n * bw
            rice_only += rc
            eg_only += ec
            fixed_only += fc
            best = min(rc, ec, fc)
            per_axis_best += best
            if best == rc:
                win["rice"] += 1
                rice_k_picks[kr] += 1
            elif best == ec:
                win["eg"] += 1
                eg_k_picks[ke] += 1
            else:
                win["fixed"] += 1

    total_axes = sum(win.values())
    print(f"\n=== {name} ===  axes={total_axes:,}  n_int_total={n_int:,}")
    def bpva(x): return x / n_int
    print(f"  Shannon          {bpva(shannon):7.3f} bpva  "
          f"({shannon:14,.0f} b)")
    print(f"  Rice-only        {bpva(rice_only):7.3f} bpva  "
          f"({rice_only:14,.0f} b)  +{(rice_only-shannon)/shannon*100:5.1f}% vs Sh")
    print(f"  EG-only          {bpva(eg_only):7.3f} bpva  "
          f"({eg_only:14,.0f} b)  +{(eg_only-shannon)/shannon*100:5.1f}% vs Sh")
    print(f"  Fixed-only       {bpva(fixed_only):7.3f} bpva  "
          f"({fixed_only:14,.0f} b)  +{(fixed_only-shannon)/shannon*100:5.1f}% vs Sh")
    print(f"  Per-axis-best    {bpva(per_axis_best):7.3f} bpva  "
          f"({per_axis_best:14,.0f} b)  +{(per_axis_best-shannon)/shannon*100:5.1f}% vs Sh")
    rice_gain = (rice_only - per_axis_best) / rice_only * 100
    eg_gain   = (eg_only   - per_axis_best) / eg_only   * 100
    print(f"  Mix gain vs Rice-only: {rice_gain:5.2f}%   "
          f"vs EG-only: {eg_gain:5.2f}%")
    print(f"  Wins per axis: rice={win['rice']:,} ({win['rice']/total_axes*100:.1f}%) "
          f"eg={win['eg']:,} ({win['eg']/total_axes*100:.1f}%) "
          f"fixed={win['fixed']:,} ({win['fixed']/total_axes*100:.1f}%)")
    print(f"  Rice k histogram: {dict(sorted(rice_k_picks.items()))}")
    print(f"  EG   k histogram: {dict(sorted(eg_k_picks.items()))}")


def main():
    meshes = [
        ("bunny",          "assets/bunny.obj"),
        ("stanford-bunny", "assets/stanford-bunny.obj"),
        ("Monkey",         "assets/Monkey.obj"),
        ("xyzrgb_dragon",  "assets/xyzrgb_dragon.obj"),
    ]
    for name, path in meshes:
        prep = load_or_prepare(path, max_verts=256, max_tris=256,
                                precision_error=0.0005,
                                gen_method="joint_learned",
                                strip_method="multiseed",
                                clean=True, verbose=False)
        _analyze(prep, name)


if __name__ == "__main__":
    main()

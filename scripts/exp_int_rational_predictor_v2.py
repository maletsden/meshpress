"""Per-mesh integer-rational parallelogram predictor — v2.

Improvements over v1:
  * Fit fp32 weights PER MESH (not global). Always pick min(canonical, fit)
    so we can never regress.
  * Local integer search +/- SEARCH_RADIUS around best_rational(w, K) per K.
    Exhaustive over (2R+1)^3 candidates.
  * Per-axis fit variant: each axis d gets its own (n0_d, n1_d, n2_d, K_d).
  * Coordinate-descent refinement on the integer grid (small steps along
    each numerator until no improvement).
  * Optionally per-axis Shannon entropy reported separately.

Output: best integer-rational (shared and per-axis) Shannon BPV vs canonical,
plus the chosen (n0, n1, n2, K) for the encoder side table.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from collections import Counter

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5_dup import _walk_meshlet


SEARCH_RADIUS = 2          # +/- 2 numerator steps around fp round
KS = (4, 5, 6, 7)          # denominator exponents to try
COORD_DESCENT_ROUNDS = 3   # passes of (n0, n1, n2) +/-1 refinement


# ----------------- entropy helpers -----------------

def _shannon_axis(x: np.ndarray) -> tuple[float, int]:
    """Return (total_bits, n) for one int axis using zigzag distribution."""
    u = np.where(x < 0, (-x << 1) - 1, x << 1).astype(np.int64)
    cnt = Counter(u.tolist())
    total = sum(cnt.values())
    bits = 0.0
    for c in cnt.values():
        p = c / total
        bits += -c * np.log2(p)
    return bits, total


def shannon_bpv(R: np.ndarray) -> float:
    if R.size == 0:
        return 0.0
    tot_bits, n = 0.0, 0
    for d in range(R.shape[1]):
        b, c = _shannon_axis(R[:, d].astype(np.int64))
        tot_bits += b
        n += c
    return tot_bits / max(n, 1)


def shannon_per_axis(R: np.ndarray) -> tuple[float, float, float]:
    return tuple(_shannon_axis(R[:, d].astype(np.int64))[0] /
                 max(R.shape[0], 1) for d in range(3))


# ----------------- sample gather -----------------

def gather_samples(mesh_path: str):
    prep = load_or_prepare(mesh_path, max_verts=256, max_tris=256,
                           precision_error=1.0 / 4096.0,
                           precision_mode="bbox_frac",
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    global_codes = prep["global_codes"]
    plans = prep["plans"]
    A, B, C, V = [], [], [], []
    for plan in plans:
        local_to_global = np.asarray(plan["local_to_global"], dtype=np.int64)
        true_codes = global_codes[local_to_global].astype(np.int64)
        order = _walk_meshlet(plan)
        for v_local, kind, refs in order:
            if kind != "para":
                continue
            a, b, c = refs
            A.append(true_codes[a]); B.append(true_codes[b])
            C.append(true_codes[c]); V.append(true_codes[v_local])
    return (np.array(A, dtype=np.int64), np.array(B, dtype=np.int64),
            np.array(C, dtype=np.int64), np.array(V, dtype=np.int64))


# ----------------- fits -----------------

def fit_shared(A, B, C, V) -> np.ndarray:
    X = np.vstack([np.stack([A[:, d], B[:, d], C[:, d]], axis=1) for d in range(3)])
    y = np.concatenate([V[:, d] for d in range(3)])
    w, *_ = np.linalg.lstsq(X.astype(np.float64), y.astype(np.float64), rcond=None)
    return w


def fit_per_axis(A, B, C, V) -> np.ndarray:
    """Returns 3x3 array w[d] = (w0_d, w1_d, w2_d)."""
    W = np.zeros((3, 3), dtype=np.float64)
    for d in range(3):
        X = np.stack([A[:, d], B[:, d], C[:, d]], axis=1).astype(np.float64)
        y = V[:, d].astype(np.float64)
        wd, *_ = np.linalg.lstsq(X, y, rcond=None)
        W[d] = wd
    return W


# ----------------- integer predictor -----------------

def predict_int_axis(a, b, c, n0, n1, n2, K):
    D = 1 << K
    half = D // 2
    s = n0 * a + n1 * b + n2 * c
    return np.where(s >= 0, (s + half) // D, -((-s + half) // D)).astype(np.int64)


def search_int_shared(A, B, C, V, w_fp, K):
    """Local search around round(w*D) +/- SEARCH_RADIUS over all 3 numerators.
    Returns ((n0, n1, n2), bits)."""
    D = 1 << K
    base = [int(round(w_fp[i] * D)) for i in range(3)]
    R = SEARCH_RADIUS
    best = None
    for d0 in range(-R, R + 1):
        n0 = base[0] + d0
        for d1 in range(-R, R + 1):
            n1 = base[1] + d1
            for d2 in range(-R, R + 1):
                n2 = base[2] + d2
                tot_bits = 0.0
                # axis-by-axis to avoid materialising big array
                for d in range(3):
                    pred = predict_int_axis(A[:, d], B[:, d], C[:, d], n0, n1, n2, K)
                    b, _ = _shannon_axis(V[:, d] - pred)
                    tot_bits += b
                if best is None or tot_bits < best[1]:
                    best = ((n0, n1, n2), tot_bits)
    # coord descent
    cur, cur_bits = best
    for _ in range(COORD_DESCENT_ROUNDS):
        improved = False
        for i in range(3):
            for step in (-1, 1):
                cand = list(cur)
                cand[i] += step
                tot_bits = 0.0
                for d in range(3):
                    pred = predict_int_axis(A[:, d], B[:, d], C[:, d],
                                            cand[0], cand[1], cand[2], K)
                    b, _ = _shannon_axis(V[:, d] - pred)
                    tot_bits += b
                if tot_bits < cur_bits:
                    cur, cur_bits = tuple(cand), tot_bits
                    improved = True
        if not improved:
            break
    return cur, cur_bits


def search_int_per_axis(A, B, C, V, W_fp, K):
    """Per-axis: each d has independent (n0_d, n1_d, n2_d). Returns 3x3 ints."""
    D = 1 << K
    R = SEARCH_RADIUS
    chosen = np.zeros((3, 3), dtype=np.int64)
    tot_bits = 0.0
    for d in range(3):
        base = [int(round(W_fp[d, i] * D)) for i in range(3)]
        best = None
        for d0 in range(-R, R + 1):
            n0 = base[0] + d0
            for d1 in range(-R, R + 1):
                n1 = base[1] + d1
                for d2 in range(-R, R + 1):
                    n2 = base[2] + d2
                    pred = predict_int_axis(A[:, d], B[:, d], C[:, d], n0, n1, n2, K)
                    b, _ = _shannon_axis(V[:, d] - pred)
                    if best is None or b < best[1]:
                        best = ((n0, n1, n2), b)
        # coord descent for this axis
        cur, cur_bits = best
        for _ in range(COORD_DESCENT_ROUNDS):
            improved = False
            for i in range(3):
                for step in (-1, 1):
                    cand = list(cur)
                    cand[i] += step
                    pred = predict_int_axis(A[:, d], B[:, d], C[:, d],
                                            cand[0], cand[1], cand[2], K)
                    b, _ = _shannon_axis(V[:, d] - pred)
                    if b < cur_bits:
                        cur, cur_bits = tuple(cand), b
                        improved = True
            if not improved:
                break
        chosen[d] = cur
        tot_bits += cur_bits
    return chosen, tot_bits


# ----------------- analyse -----------------

def analyze(mesh_path: str):
    A, B, C, V = gather_samples(mesh_path)
    n = A.shape[0]
    n_total = n * 3
    print(f"\n=== {Path(mesh_path).stem} ({n:,} para steps) ===")

    # Canonical
    R_can = V - (A + B - C)
    bpv_can = shannon_bpv(R_can)
    print(f"  canonical (1,1,-1):              {bpv_can:.4f} BPV")

    # fp32 shared
    w = fit_shared(A, B, C, V)
    pred = w[0]*A.astype(np.float64) + w[1]*B.astype(np.float64) + w[2]*C.astype(np.float64)
    R_fp = V - np.round(pred).astype(np.int64)
    bpv_fp = shannon_bpv(R_fp)
    print(f"  fp32 shared ({w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f}): "
          f"{bpv_fp:.4f} BPV ({bpv_fp - bpv_can:+.3f})")

    # int-rational shared, search per K
    best_sh = None
    for K in KS:
        ns, bits = search_int_shared(A, B, C, V, w, K)
        bpv = bits / n_total
        tag = "*" if bpv < bpv_can else " "
        print(f"  int shared K={K} ({ns[0]:+d},{ns[1]:+d},{ns[2]:+d})/{1<<K}:"
              f" {bpv:.4f} BPV ({bpv - bpv_can:+.4f}) {tag}")
        if best_sh is None or bpv < best_sh[2]:
            best_sh = (K, ns, bpv)
    # final shared = min(canonical, best int shared)
    final_sh = min(bpv_can, best_sh[2])
    print(f"  -> shared best K={best_sh[0]} ns={best_sh[1]} "
          f"= {best_sh[2]:.4f} BPV; pick min(canon, fit) = {final_sh:.4f} "
          f"({final_sh - bpv_can:+.4f})")

    # fp32 per-axis
    W = fit_per_axis(A, B, C, V)
    pred_pa = np.stack([
        W[d, 0]*A[:, d] + W[d, 1]*B[:, d] + W[d, 2]*C[:, d] for d in range(3)
    ], axis=1)
    R_pa = V - np.round(pred_pa).astype(np.int64)
    bpv_pa = shannon_bpv(R_pa)
    print(f"  fp32 per-axis:                   {bpv_pa:.4f} BPV ({bpv_pa - bpv_can:+.4f})")

    # int per-axis, search per K
    best_pa = None
    for K in KS:
        chosen, bits = search_int_per_axis(A, B, C, V, W, K)
        bpv = bits / n_total
        tag = "*" if bpv < bpv_can else " "
        print(f"  int per-axis K={K}:               {bpv:.4f} BPV ({bpv - bpv_can:+.4f}) {tag}")
        if best_pa is None or bpv < best_pa[2]:
            best_pa = (K, chosen, bpv)
    final_pa = min(bpv_can, best_pa[2])
    print(f"  -> per-axis best K={best_pa[0]} = {best_pa[2]:.4f} BPV; "
          f"pick min(canon, fit) = {final_pa:.4f} ({final_pa - bpv_can:+.4f})")
    print(f"     per-axis numerators:\n{best_pa[1]}")

    return {
        "mesh": Path(mesh_path).stem,
        "n_para": n,
        "bpv_canon": bpv_can,
        "bpv_fp_shared": bpv_fp,
        "bpv_int_shared_best": best_sh[2],
        "bpv_fp_per_axis": bpv_pa,
        "bpv_int_per_axis_best": best_pa[2],
        "final_shared": final_sh,
        "final_per_axis": final_pa,
    }


if __name__ == "__main__":
    meshes = sys.argv[1:] or [
        "assets/stanford-bunny.obj",
        "assets/Monkey.obj",
        "assets/xyzrgb_dragon.obj",
    ]
    rows = [analyze(m) for m in meshes]

    print("\n\n=== SUMMARY ===")
    print(f"{'mesh':16s}  {'canon':>7s}  {'sh-int':>7s}  {'pa-int':>7s}  "
          f"{'save-sh':>8s}  {'save-pa':>8s}")
    for r in rows:
        print(f"{r['mesh']:16s}  {r['bpv_canon']:7.4f}  "
              f"{r['final_shared']:7.4f}  {r['final_per_axis']:7.4f}  "
              f"{r['bpv_canon']-r['final_shared']:+8.4f}  "
              f"{r['bpv_canon']-r['final_per_axis']:+8.4f}")

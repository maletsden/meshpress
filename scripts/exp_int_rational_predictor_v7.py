"""1 weight for Rice + 1 weight for Exp-Golomb. Per-axis-per-meshlet selector.

Fit Weight_R against per-meshlet Rice-only loss.
Fit Weight_E against per-meshlet EG-only loss.

At encode time, for each (axis, meshlet) pair, compute
  cost_R = min_k Rice_bits(R_R[axis, meshlet], k)
  cost_E = min_k EG_bits  (R_E[axis, meshlet], k)
Pick min. 1-bit selector per (axis, meshlet) = 3 bits per meshlet.

Header cost: 2 × 12 B = 24 B per mesh + 3 × n_m bits.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5_dup import _walk_meshlet


CACHE_DIR = ROOT / "cache" / "predictor_samples_v5"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

KS = (5, 6, 7, 8, 9)
SEARCH_RADIUS = 1
COORD_DESCENT_ROUNDS = 2
IRLS_ITERS = 12
IRLS_EPS = 1.0
RICE_K_MAX = 11
EG_K_MAX = 7
WEIGHT_HEADER_BYTES = 12
SUBSAMPLE_MESHLETS = 4000


def gather(mesh_path: str):
    cache = CACHE_DIR / f"{Path(mesh_path).stem}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["A"], z["B"], z["C"], z["V"], z["off"]
    prep = load_or_prepare(mesh_path, max_verts=256, max_tris=256,
                           precision_error=1.0/4096.0,
                           precision_mode="bbox_frac",
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    gc = prep["global_codes"]; plans = prep["plans"]
    A, B, C, V, offs = [], [], [], [], [0]
    for plan in plans:
        l2g = np.asarray(plan["local_to_global"], dtype=np.int64)
        tc = gc[l2g].astype(np.int64)
        m_count = 0
        for v, kind, refs in _walk_meshlet(plan):
            if kind != "para":
                continue
            a, b, c = refs
            A.append(tc[a]); B.append(tc[b]); C.append(tc[c]); V.append(tc[v])
            m_count += 1
        offs.append(offs[-1] + m_count)
    A = np.array(A, dtype=np.int64); B = np.array(B, dtype=np.int64)
    C = np.array(C, dtype=np.int64); V = np.array(V, dtype=np.int64)
    off = np.array(offs, dtype=np.int64)
    np.savez_compressed(cache, A=A, B=B, C=C, V=V, off=off)
    return A, B, C, V, off


def _zigzag(r): return np.where(r < 0, (-r << 1) - 1, r << 1).astype(np.int64)


def _segsum(arr, off):
    n_seg = len(off) - 1
    if arr.size == 0:
        return np.zeros(n_seg, dtype=np.int64)
    cs = np.concatenate(([0], np.cumsum(arr.astype(np.int64))))
    return cs[off[1:]] - cs[off[:-1]]


def best_rice_axis(u, off):
    n_m = len(off) - 1
    counts = np.diff(off).astype(np.int64)
    best = np.full(n_m, np.iinfo(np.int64).max, dtype=np.int64)
    for k in range(RICE_K_MAX + 1):
        seg = _segsum(u >> k, off) + counts * (k + 1)
        best = np.minimum(best, seg)
    best[counts == 0] = 0
    return best


def best_eg_axis(u, off):
    n_m = len(off) - 1
    counts = np.diff(off).astype(np.int64)
    best = np.full(n_m, np.iinfo(np.int64).max, dtype=np.int64)
    for k in range(EG_K_MAX + 1):
        q = (u >> k) + 1
        per = 2 * np.floor(np.log2(q.astype(np.float64))).astype(np.int64) + 1 + k
        seg = _segsum(per, off)
        best = np.minimum(best, seg)
    best[counts == 0] = 0
    return best


def predict_axis(a, b, c, n0, n1, n2, K):
    s = n0 * a + n1 * b + n2 * c
    if K == 0:
        return s
    half = (1 << K) >> 1
    return (s + half) >> K


def irls_l1(X, y, n_iter=IRLS_ITERS, eps=IRLS_EPS):
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(n_iter):
        r = y - X @ w
        wts = 1.0 / np.maximum(np.abs(r), eps)
        sw = np.sqrt(wts)
        w_new, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        if np.max(np.abs(w_new - w)) < 1e-6:
            return w_new
        w = w_new
    return w


def fit_irls(A, B, C, V):
    W = np.zeros((3, 3))
    for d in range(3):
        X = np.stack([A[:, d], B[:, d], C[:, d]], axis=1).astype(np.float64)
        W[d] = irls_l1(X, V[:, d].astype(np.float64))
    return W


def search_axis(a, b, c, v, off, w_axis, scorer_fn):
    """scorer_fn(u, off) -> per-meshlet bits."""
    best = None
    for K in KS:
        D = 1 << K
        base = [int(round(w_axis[i] * D)) for i in range(3)]
        seeds = [tuple(base), (D, D, -D)]
        for s0 in (-SEARCH_RADIUS, SEARCH_RADIUS):
            for s1 in (-SEARCH_RADIUS, SEARCH_RADIUS):
                for s2 in (-SEARCH_RADIUS, SEARCH_RADIUS):
                    seeds.append((base[0]+s0, base[1]+s1, base[2]+s2))
        for seed in seeds:
            cur = list(seed)
            pred = predict_axis(a, b, c, *cur, K)
            cur_bits = int(scorer_fn(_zigzag(v - pred), off).sum())
            for _ in range(COORD_DESCENT_ROUNDS):
                improved = False
                for i in range(3):
                    for sign in (-1, 1):
                        cand = cur.copy()
                        cand[i] += sign
                        pred = predict_axis(a, b, c, *cand, K)
                        b2 = int(scorer_fn(_zigzag(v - pred), off).sum())
                        if b2 < cur_bits:
                            cur, cur_bits = cand, b2
                            improved = True
                if not improved:
                    break
            if best is None or cur_bits < best[2]:
                best = (K, tuple(cur), cur_bits)
    return best


def fit_set(A, B, C, V, off, scorer_fn):
    W = fit_irls(A, B, C, V)
    n_arr = np.zeros((3, 3), dtype=np.int64)
    K_arr = np.zeros(3, dtype=np.int64)
    tot = 0
    for d in range(3):
        K, ns, bits = search_axis(A[:, d], B[:, d], C[:, d], V[:, d], off,
                                   W[d], scorer_fn)
        K_arr[d] = K
        n_arr[d] = ns
        tot += bits
    return n_arr, K_arr, tot


def axis_bits_under(A, B, C, V, off, n_set, K_set, scorer_fn):
    """Returns (3, n_m) array: per-axis-per-meshlet bits."""
    n_m = len(off) - 1
    out = np.zeros((3, n_m), dtype=np.int64)
    for d in range(3):
        pred = predict_axis(A[:, d], B[:, d], C[:, d],
                            int(n_set[d, 0]), int(n_set[d, 1]),
                            int(n_set[d, 2]), int(K_set[d]))
        out[d] = scorer_fn(_zigzag(V[:, d] - pred), off)
    return out


def analyze(mesh_path: str):
    t0 = time.time()
    A, B, C, V, off = gather(mesh_path)
    n = A.shape[0]; n_m = len(off) - 1
    print(f"\n=== {Path(mesh_path).stem} ({n:,} para, {n_m:,} meshlets) ===")

    # Canonical: per-axis per-meshlet min(Rice, EG)
    R_can = V - (A + B - C)
    rice_can = np.stack([best_rice_axis(_zigzag(R_can[:, d]), off) for d in range(3)])
    eg_can   = np.stack([best_eg_axis  (_zigzag(R_can[:, d]), off) for d in range(3)])
    L_canon  = int(np.minimum(rice_can, eg_can).sum())
    print(f"  canonical L_min={L_canon:,} bits ({L_canon/8/1024:.1f} KB)")

    # Subsample for fit
    if n_m > SUBSAMPLE_MESHLETS:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(n_m, size=SUBSAMPLE_MESHLETS, replace=False))
        rows_idx = np.concatenate([np.arange(off[i], off[i+1]) for i in sel])
        A_s, B_s, C_s, V_s = A[rows_idx], B[rows_idx], C[rows_idx], V[rows_idx]
        counts = np.diff(off)[sel]
        off_s = np.concatenate(([0], np.cumsum(counts)))
        tag = "sub"
    else:
        A_s, B_s, C_s, V_s, off_s = A, B, C, V, off
        tag = "full"

    # Weight_R: optimize Rice cost
    t1 = time.time()
    n_R, K_R, _ = fit_set(A_s, B_s, C_s, V_s, off_s, best_rice_axis)
    print(f"  Weight_R fit ({tag}, {time.time()-t1:.1f}s):")
    for d, name in enumerate("xyz"):
        print(f"    {name}: K={K_R[d]} ns=({n_R[d,0]:+d},{n_R[d,1]:+d},{n_R[d,2]:+d})")

    # Weight_E: optimize EG cost
    t2 = time.time()
    n_E, K_E, _ = fit_set(A_s, B_s, C_s, V_s, off_s, best_eg_axis)
    print(f"  Weight_E fit ({tag}, {time.time()-t2:.1f}s):")
    for d, name in enumerate("xyz"):
        print(f"    {name}: K={K_E[d]} ns=({n_E[d,0]:+d},{n_E[d,1]:+d},{n_E[d,2]:+d})")

    # ---- Per-axis canonical fallback against ENCODER COST = min(Rice, EG) ----
    n_canon = np.array([[1, 1, -1]] * 3, dtype=np.int64)
    K_canon = np.zeros(3, dtype=np.int64)

    def axis_full_pred(n_set, K_set, d):
        return predict_axis(A[:, d], B[:, d], C[:, d],
                            int(n_set[d, 0]), int(n_set[d, 1]),
                            int(n_set[d, 2]), int(K_set[d]))

    def axis_encoder_cost(n_set, K_set, d):
        """Per-meshlet bits = min(best_rice, best_eg) under (n_set, K_set)."""
        u = _zigzag(V[:, d] - axis_full_pred(n_set, K_set, d))
        return np.minimum(best_rice_axis(u, off), best_eg_axis(u, off))

    # Single-weight R (Rice-optimal fit): per-axis fallback to canonical
    # if canonical beats fit under encoder cost on full data.
    bits_R_per_axis = np.zeros((3, n_m), dtype=np.int64)
    for d in range(3):
        b_fit = axis_encoder_cost(n_R, K_R, d)
        b_can = axis_encoder_cost(n_canon, K_canon, d)
        if b_can.sum() < b_fit.sum():
            n_R[d] = n_canon[d]; K_R[d] = K_canon[d]
            bits_R_per_axis[d] = b_can
        else:
            bits_R_per_axis[d] = b_fit
    L_single_R = int(bits_R_per_axis.sum())

    # Single-weight E (EG-optimal fit): same per-axis fallback.
    bits_E_per_axis = np.zeros((3, n_m), dtype=np.int64)
    for d in range(3):
        b_fit = axis_encoder_cost(n_E, K_E, d)
        b_can = axis_encoder_cost(n_canon, K_canon, d)
        if b_can.sum() < b_fit.sum():
            n_E[d] = n_canon[d]; K_E[d] = K_canon[d]
            bits_E_per_axis[d] = b_can
        else:
            bits_E_per_axis[d] = b_fit
    L_single_E = int(bits_E_per_axis.sum())

    # "Two-weight" = pick whichever single set is best globally per-mesh.
    # NO per-meshlet selector — encoder picks one of {canonical, R-fit, E-fit}
    # for whole mesh; ships winner + mode flag (2 bits).
    L_two_weight = min(L_single_R, L_single_E)   # whichever fitted set wins

    overhead = WEIGHT_HEADER_BYTES * 8               # 96 bits / mesh, always
    overhead_one = overhead
    overhead_two = overhead                          # encoder picks 1 set, no mode bit

    print(f"  single_R (Rice-fit weights, encoder picks Rice/EG free):"
          f" {L_single_R:,} + {overhead_one} = {L_single_R + overhead_one:,} "
          f"Δcanon {L_single_R + overhead_one - L_canon:+,}")
    print(f"  single_E (EG-fit weights, encoder picks Rice/EG free):"
          f" {L_single_E:,} + {overhead_one} = {L_single_E + overhead_one:,} "
          f"Δcanon {L_single_E + overhead_one - L_canon:+,}")
    print(f"  best-of-2 (min(R-fit, E-fit), 2-bit mesh mode):"
          f" {L_two_weight:,} + {overhead_two} = {L_two_weight + overhead_two:,} "
          f"Δcanon {L_two_weight + overhead_two - L_canon:+,}")

    candidates = [
        ("canonical", L_canon, 0),
        ("single_R",  L_single_R, overhead_one),
        ("single_E",  L_single_E, overhead_one),
        ("best_of_2", L_two_weight, overhead_two),
    ]
    candidates.sort(key=lambda c: c[1] + c[2])
    winner = candidates[0]
    save_B = (L_canon - (winner[1] + winner[2])) / 8
    print(f"  WINNER: {winner[0]}  save vs canon: {save_B:+.0f} B  "
          f"total: {time.time()-t0:.1f}s")
    return {
        "mesh": Path(mesh_path).stem, "n_para": n, "n_meshlets": n_m,
        "L_canon": L_canon, "L_single": L_single_R + overhead_one,
        "L_two":  L_two_weight + overhead_two,
        "winner": winner[0], "save_B": save_B,
    }


if __name__ == "__main__":
    meshes = sys.argv[1:] or ["assets/stanford-bunny.obj"]
    rows = [analyze(m) for m in meshes]
    print("\n=== SUMMARY ===")
    print(f"{'mesh':16s}  {'L_canon B':>11s}  {'L_single B':>11s}  "
          f"{'L_two B':>11s}  {'winner':12s}  {'save B':>8s}")
    for r in rows:
        print(f"  {r['mesh']:16s}  {r['L_canon']/8:>11,.0f}  "
              f"{r['L_single']/8:>11,.0f}  {r['L_two']/8:>11,.0f}  "
              f"{r['winner']:12s}  {r['save_B']:>+8.0f}")

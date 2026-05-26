"""1-2 weight sets per mesh, per-meshlet selector — encoder-true loss.

Two predictor sets (A, B), each with per-axis (n0,n1,n2,K). Per meshlet
1-bit selector picks A vs B. Alternating optimization:
  (a) assign each meshlet to argmin L_min(A_pred, B_pred) bits
  (b) refit A on meshlets assigned to A (per-axis IRLS L1 → int snap
      → coord descent against per-meshlet L_min)
  (c) refit B same way on its meshlets
  loop until total bits no longer drops.

Reported per mesh: L_canon, L_single, L_two_sets, breakeven vs overhead.
Pick min across {canonical, single, two} including overhead.
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
N_ALT_ITERS = 6
WEIGHT_HEADER_BYTES = 12   # 9 int8 n + 3 uint8 K per weight set
SUBSAMPLE_MESHLETS = 4000


# =========================================================
# Gather (cached) — same as v5
# =========================================================

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


# =========================================================
# Vectorized per-meshlet Rice / EG
# =========================================================

def _zigzag(r):
    return np.where(r < 0, (-r << 1) - 1, r << 1).astype(np.int64)


def _segsum(arr: np.ndarray, off: np.ndarray) -> np.ndarray:
    n_seg = len(off) - 1
    if arr.size == 0:
        return np.zeros(n_seg, dtype=np.int64)
    cs = np.concatenate(([0], np.cumsum(arr.astype(np.int64))))
    return cs[off[1:]] - cs[off[:-1]]


def per_meshlet_l_min_axis(u: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Per-meshlet min(min_k Rice, min_k EG) bits for one axis."""
    n_m = len(off) - 1
    counts = np.diff(off).astype(np.int64)
    # Rice across k
    best = np.full(n_m, np.iinfo(np.int64).max, dtype=np.int64)
    for k in range(RICE_K_MAX + 1):
        seg = _segsum(u >> k, off) + counts * (k + 1)
        best = np.minimum(best, seg)
    # EG across k
    for k in range(EG_K_MAX + 1):
        q = (u >> k) + 1
        per = 2 * np.floor(np.log2(q.astype(np.float64))).astype(np.int64) + 1 + k
        seg = _segsum(per, off)
        best = np.minimum(best, seg)
    best[counts == 0] = 0
    return best


def per_meshlet_l_min(R: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Total bits per meshlet across all 3 axes."""
    tot = np.zeros(len(off) - 1, dtype=np.int64)
    for d in range(3):
        u = _zigzag(R[:, d])
        tot += per_meshlet_l_min_axis(u, off)
    return tot


# =========================================================
# Predictor evaluation + IRLS
# =========================================================

def predict_axis(a, b, c, n0, n1, n2, K):
    s = n0 * a + n1 * b + n2 * c
    if K == 0:
        return s
    half = (1 << K) >> 1
    return (s + half) >> K


def predict_full(A, B, C, n_3x3, K_3) -> np.ndarray:
    out = np.empty_like(A)
    for d in range(3):
        out[:, d] = predict_axis(
            A[:, d], B[:, d], C[:, d],
            int(n_3x3[d, 0]), int(n_3x3[d, 1]), int(n_3x3[d, 2]), int(K_3[d]),
        )
    return out


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


def fit_irls(A, B, C, V) -> np.ndarray:
    W = np.zeros((3, 3))
    for d in range(3):
        X = np.stack([A[:, d], B[:, d], C[:, d]], axis=1).astype(np.float64)
        W[d] = irls_l1(X, V[:, d].astype(np.float64))
    return W


# =========================================================
# Per-axis integer search with per-meshlet L_min
# =========================================================

def axis_loss(a, b, c, v, off, n, K) -> int:
    pred = predict_axis(a, b, c, int(n[0]), int(n[1]), int(n[2]), K)
    return int(per_meshlet_l_min_axis(_zigzag(v - pred), off).sum())


def search_axis(a, b, c, v, off, w_axis) -> tuple[int, tuple, int]:
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
            cur_bits = axis_loss(a, b, c, v, off, cur, K)
            for _ in range(COORD_DESCENT_ROUNDS):
                improved = False
                for i in range(3):
                    for sign in (-1, 1):
                        cand = cur.copy()
                        cand[i] += sign
                        b2 = axis_loss(a, b, c, v, off, cand, K)
                        if b2 < cur_bits:
                            cur, cur_bits = cand, b2
                            improved = True
                if not improved:
                    break
            if best is None or cur_bits < best[2]:
                best = (K, tuple(cur), cur_bits)
    return best


def fit_one_set(A, B, C, V, off, W_fp=None) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit per-axis (n, K) under per-meshlet L_min loss. Returns (n_3x3, K_3, total_bits)."""
    if W_fp is None:
        W_fp = fit_irls(A, B, C, V)
    n_arr = np.zeros((3, 3), dtype=np.int64)
    K_arr = np.zeros(3, dtype=np.int64)
    tot = 0
    for d in range(3):
        K, ns, bits = search_axis(A[:, d], B[:, d], C[:, d], V[:, d], off, W_fp[d])
        K_arr[d] = K
        n_arr[d] = ns
        tot += bits
    return n_arr, K_arr, tot


# =========================================================
# Two-weight alternating optimization
# =========================================================

def evaluate_set(A, B, C, V, off, n_set, K_set):
    """Per-meshlet bits under one weight set."""
    R = V - predict_full(A, B, C, n_set, K_set)
    return per_meshlet_l_min(R, off)


def fit_two_sets(A, B, C, V, off):
    """Returns (n_a, K_a, n_b, K_b, selector, total_bits, history)."""
    n_m = len(off) - 1
    # Init: A = IRLS fit, B = canonical
    W = fit_irls(A, B, C, V)
    n_a, K_a, _ = fit_one_set(A, B, C, V, off, W_fp=W)
    n_b = np.array([[1, 1, -1]] * 3, dtype=np.int64)
    K_b = np.zeros(3, dtype=np.int64)

    selector = np.zeros(n_m, dtype=np.int64)
    prev_tot = None
    history = []
    for it in range(N_ALT_ITERS):
        bits_a = evaluate_set(A, B, C, V, off, n_a, K_a)
        bits_b = evaluate_set(A, B, C, V, off, n_b, K_b)
        selector = (bits_b < bits_a).astype(np.int64)
        tot = int(np.minimum(bits_a, bits_b).sum())
        history.append((it, tot, int(selector.sum())))
        print(f"    alt iter {it}: L_min={tot:,}  |A|={int((selector==0).sum())} "
              f"|B|={int((selector==1).sum())}")
        if prev_tot is not None and tot >= prev_tot - 8:
            break
        prev_tot = tot
        # Refit A on selector==0 meshlets, B on selector==1.
        # Gather row-indices for each cluster
        for cluster_id, n_set, K_set in [(0, n_a, K_a), (1, n_b, K_b)]:
            sel_m = np.where(selector == cluster_id)[0]
            if len(sel_m) == 0:
                continue
            rows = np.concatenate([np.arange(off[i], off[i+1]) for i in sel_m])
            if rows.size == 0:
                continue
            A_s, B_s, C_s, V_s = A[rows], B[rows], C[rows], V[rows]
            counts = np.diff(off)[sel_m]
            off_s = np.concatenate(([0], np.cumsum(counts)))
            n_new, K_new, _ = fit_one_set(A_s, B_s, C_s, V_s, off_s)
            if cluster_id == 0:
                n_a, K_a = n_new, K_new
            else:
                n_b, K_b = n_new, K_new
    # Final eval
    bits_a = evaluate_set(A, B, C, V, off, n_a, K_a)
    bits_b = evaluate_set(A, B, C, V, off, n_b, K_b)
    selector = (bits_b < bits_a).astype(np.int64)
    tot = int(np.minimum(bits_a, bits_b).sum())
    return n_a, K_a, n_b, K_b, selector, tot, history


# =========================================================
# Analyse
# =========================================================

def analyze(mesh_path: str, do_two: bool = True):
    t0 = time.time()
    A, B, C, V, off = gather(mesh_path)
    n = A.shape[0]
    n_m = len(off) - 1
    print(f"\n=== {Path(mesh_path).stem} ({n:,} para steps, {n_m:,} meshlets) ===")

    # canonical
    R_can = V - (A + B - C)
    L_canon = int(per_meshlet_l_min(R_can, off).sum())
    print(f"  canonical: L_min={L_canon:,} bits ({L_canon/8/1024:.1f} KB)")

    # single (per-mesh weights)
    t1 = time.time()
    if n_m > SUBSAMPLE_MESHLETS:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(n_m, size=SUBSAMPLE_MESHLETS, replace=False))
        rows_idx = np.concatenate([np.arange(off[i], off[i+1]) for i in sel])
        A_s, B_s, C_s, V_s = A[rows_idx], B[rows_idx], C[rows_idx], V[rows_idx]
        counts = np.diff(off)[sel]
        off_s = np.concatenate(([0], np.cumsum(counts)))
        n_arr, K_arr, _ = fit_one_set(A_s, B_s, C_s, V_s, off_s)
        # verify on full
        L_single = int(evaluate_set(A, B, C, V, off, n_arr, K_arr).sum())
    else:
        n_arr, K_arr, L_single = fit_one_set(A, B, C, V, off)
    overhead_single = WEIGHT_HEADER_BYTES * 8
    print(f"  single set (fit {time.time()-t1:.1f}s): "
          f"L_min={L_single:,} bits + {overhead_single} hdr "
          f"= {L_single + overhead_single:,}  "
          f"Δ={L_single + overhead_single - L_canon:+,}")
    for d, name in enumerate("xyz"):
        print(f"    {name}: K={K_arr[d]} ns=({n_arr[d,0]:+d},{n_arr[d,1]:+d},{n_arr[d,2]:+d})")

    # two sets
    L_two = None
    if do_two:
        t2 = time.time()
        # do two-sets fit on subsample (or full if small)
        if n_m > SUBSAMPLE_MESHLETS:
            n_a, K_a, n_b, K_b, sel_s, _, hist = fit_two_sets(A_s, B_s, C_s, V_s, off_s)
            # verify on full
            bits_a = evaluate_set(A, B, C, V, off, n_a, K_a)
            bits_b = evaluate_set(A, B, C, V, off, n_b, K_b)
            full_selector = (bits_b < bits_a).astype(np.int64)
            L_two = int(np.minimum(bits_a, bits_b).sum())
            sub_tag = "sub-fit"
        else:
            n_a, K_a, n_b, K_b, full_selector, L_two, hist = fit_two_sets(A, B, C, V, off)
            sub_tag = "full-fit"
        overhead_two = 2 * WEIGHT_HEADER_BYTES * 8 + n_m  # 2 headers + 1 bit / meshlet
        print(f"  two sets ({sub_tag}, fit {time.time()-t2:.1f}s): "
              f"L_min={L_two:,} bits + {overhead_two} hdr+sel "
              f"= {L_two + overhead_two:,}  "
              f"Δ={L_two + overhead_two - L_canon:+,}")
        for d, name in enumerate("xyz"):
            print(f"    A {name}: K={K_a[d]} ns=({n_a[d,0]:+d},{n_a[d,1]:+d},{n_a[d,2]:+d})")
        for d, name in enumerate("xyz"):
            print(f"    B {name}: K={K_b[d]} ns=({n_b[d,0]:+d},{n_b[d,1]:+d},{n_b[d,2]:+d})")
        print(f"    selector |A|={int((full_selector==0).sum())} "
              f"|B|={int((full_selector==1).sum())}")

    # final: pick best
    candidates = [("canonical", L_canon, 0)]
    candidates.append(("single", L_single, overhead_single))
    if L_two is not None:
        candidates.append(("two", L_two, overhead_two))
    candidates.sort(key=lambda c: c[1] + c[2])
    winner = candidates[0]
    print(f"  WINNER: {winner[0]}  total={winner[1] + winner[2]:,} bits  "
          f"save vs canon={L_canon - (winner[1] + winner[2]):+,} bits  "
          f"({(L_canon - (winner[1] + winner[2]))/8:+.0f} bytes)")
    print(f"  total: {time.time()-t0:.1f}s")
    return {
        "mesh": Path(mesh_path).stem, "n_para": n, "n_meshlets": n_m,
        "L_canon": L_canon, "L_single": L_single + overhead_single,
        "L_two": (L_two + overhead_two) if L_two is not None else None,
        "winner": winner[0],
        "save_bytes": (L_canon - (winner[1] + winner[2])) / 8,
    }


if __name__ == "__main__":
    meshes = sys.argv[1:] or ["assets/stanford-bunny.obj"]
    rows = [analyze(m) for m in meshes]
    print("\n=== SUMMARY ===")
    print(f"{'mesh':16s}  {'L_canon':>10s}  {'L_single':>10s}  {'L_two':>10s}  "
          f"{'winner':10s}  {'save B':>10s}")
    for r in rows:
        lt = f"{r['L_two']:,}" if r['L_two'] is not None else "—"
        print(f"  {r['mesh']:16s}  {r['L_canon']:>10,}  "
              f"{r['L_single']:>10,}  {lt:>10s}  {r['winner']:10s}  "
              f"{r['save_bytes']:>10.0f}")

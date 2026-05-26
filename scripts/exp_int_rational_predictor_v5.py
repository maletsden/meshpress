"""Per-meshlet Rice + EG loss — matches STRIDE-dup encoder cost exactly.

Key change vs v4: residuals grouped by meshlet, per-meshlet (k, code) chosen
exhaustively, summed across meshlets. This mirrors what _pick_best_k +
_emit_axis_stream actually do at encode time.

Three loss variants reported per candidate:
  L_rice  = sum_m min_k  Rice_bits(R_m, k)
  L_eg    = sum_m min_k  EG_bits  (R_m, k)
  L_min   = sum_m min(L_rice_m, L_eg_m)        ← what encoder pays

Optimization picks weights against L_min (encoder cost). Bunny + canonical
fallback retained — pred cannot regress.
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

KS = (5, 6, 7, 8, 9)       # narrower — IRLS lands near canonical
SEARCH_RADIUS = 1          # ±1 around fp snap (3^3 = 27 candidates)
COORD_DESCENT_ROUNDS = 2
IRLS_ITERS = 15
IRLS_EPS = 1.0
RICE_K_MAX = 11
EG_K_MAX = 7
M_STARTS = 2               # just fp-snap + canonical
SUBSAMPLE_MESHLETS = 4000  # random meshlets for search; verify on full


# =========================================================
# Sample gather with per-meshlet offsets
# =========================================================

def gather(mesh_path: str):
    """(A, B, C, V, offsets) — offsets[i] = first row of meshlet i in arrays."""
    cache = CACHE_DIR / f"{Path(mesh_path).stem}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["A"], z["B"], z["C"], z["V"], z["off"]
    print(f"  [gather cold] {mesh_path}...")
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
# Vectorized per-meshlet Rice / EG cost
# =========================================================

def _zigzag(r):
    return np.where(r < 0, (-r << 1) - 1, r << 1).astype(np.int64)


def _segment_sums(arr: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Sum arr within each [off[i], off[i+1]). Uses cumsum to dodge
    add.reduceat edge cases (e.g. trailing empty meshlet hits off[-1] == size)."""
    n_seg = len(off) - 1
    if arr.size == 0:
        return np.zeros(n_seg, dtype=np.int64)
    cs = np.concatenate(([0], np.cumsum(arr.astype(np.int64))))
    return cs[off[1:]] - cs[off[:-1]]


def best_rice_per_meshlet(u: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Per-meshlet min over k of Rice cost (in bits). Vectorized across k."""
    n_m = len(off) - 1
    if u.size == 0:
        return np.zeros(n_m, dtype=np.int64)
    counts = np.diff(off).astype(np.int64)
    # Compute all (u >> k) cumsums in one shot, then segment-diff.
    n = u.size
    # build (RICE_K_MAX+1, n+1) cumulative sums
    cs = np.empty((RICE_K_MAX + 1, n + 1), dtype=np.int64)
    cs[:, 0] = 0
    for k in range(RICE_K_MAX + 1):
        cs[k, 1:] = np.cumsum(u >> k)
    seg_quot = cs[:, off[1:]] - cs[:, off[:-1]]   # (K+1, n_m)
    # add suffix term n_seg*(k+1)
    k_arr = np.arange(RICE_K_MAX + 1, dtype=np.int64)
    suffix = counts[None, :] * (k_arr[:, None] + 1)
    total = seg_quot + suffix
    best = total.min(axis=0)
    best[counts == 0] = 0
    return best


def best_eg_per_meshlet(u: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Per-meshlet min over k of Exp-Golomb cost. Vectorized across k."""
    n_m = len(off) - 1
    if u.size == 0:
        return np.zeros(n_m, dtype=np.int64)
    counts = np.diff(off).astype(np.int64)
    n = u.size
    cs = np.empty((EG_K_MAX + 1, n + 1), dtype=np.int64)
    cs[:, 0] = 0
    for k in range(EG_K_MAX + 1):
        q = (u >> k) + 1
        prefix_len = np.floor(np.log2(q.astype(np.float64))).astype(np.int64)
        per_sample = 2 * prefix_len + 1 + k
        cs[k, 1:] = np.cumsum(per_sample)
    seg = cs[:, off[1:]] - cs[:, off[:-1]]
    best = seg.min(axis=0)
    best[counts == 0] = 0
    return best


def loss_min_per_mesh(R: np.ndarray, off: np.ndarray) -> tuple[int, int, int]:
    """Returns (L_rice, L_eg, L_min) in bits over all axes."""
    L_rice = 0; L_eg = 0; L_min = 0
    for d in range(3):
        u = _zigzag(R[:, d])
        rice = best_rice_per_meshlet(u, off)
        eg   = best_eg_per_meshlet(u, off)
        L_rice += int(rice.sum())
        L_eg   += int(eg.sum())
        L_min  += int(np.minimum(rice, eg).sum())
    return L_rice, L_eg, L_min


# =========================================================
# IRLS L1 per axis
# =========================================================

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


def fit_irls_per_axis(A, B, C, V) -> np.ndarray:
    W = np.zeros((3, 3), dtype=np.float64)
    for d in range(3):
        X = np.stack([A[:, d], B[:, d], C[:, d]], axis=1).astype(np.float64)
        y = V[:, d].astype(np.float64)
        W[d] = irls_l1(X, y)
    return W


# =========================================================
# Search per axis: minimize per-axis L_min  contribution
# =========================================================

def predict_int_axis(a, b, c, n0, n1, n2, K):
    D = 1 << K; half = D >> 1
    s = n0 * a + n1 * b + n2 * c
    return (s + half) >> K if K > 0 else s


def axis_l_min(a, b, c, v, off, n0, n1, n2, K) -> int:
    """L_min contribution for ONE axis with given numerators."""
    pred = predict_int_axis(a, b, c, n0, n1, n2, K)
    u = _zigzag(v - pred)
    rice = best_rice_per_meshlet(u, off)
    eg   = best_eg_per_meshlet(u, off)
    return int(np.minimum(rice, eg).sum())


def search_axis(a, b, c, v, off, w_axis):
    """Grid + coord descent for one axis. Returns (K, ns, bits)."""
    best = None
    for K in KS:
        D = 1 << K
        base = [int(round(w_axis[i] * D)) for i in range(3)]
        # grid
        seeds = [tuple(base), (D, D, -D)]  # fp snap + canonical
        for s0 in (-SEARCH_RADIUS, SEARCH_RADIUS):
            for s1 in (-SEARCH_RADIUS, SEARCH_RADIUS):
                for s2 in (-SEARCH_RADIUS, SEARCH_RADIUS):
                    seeds.append((base[0]+s0, base[1]+s1, base[2]+s2))
        # multi-start
        rng = np.random.default_rng(7919 + K)
        while len(seeds) < M_STARTS + 10:
            seeds.append((int(rng.integers(-2*D, 2*D)),
                          int(rng.integers(-2*D, 2*D)),
                          int(rng.integers(-2*D, 2*D))))
        for seed in seeds:
            cur = list(seed)
            cur_bits = axis_l_min(a, b, c, v, off, *cur, K)
            for _ in range(COORD_DESCENT_ROUNDS):
                improved = False
                for step_mag in (1, 2):
                    for i in range(3):
                        for sign in (-1, 1):
                            cand = cur.copy()
                            cand[i] += sign * step_mag
                            bits = axis_l_min(a, b, c, v, off, *cand, K)
                            if bits < cur_bits:
                                cur, cur_bits = cand, bits
                                improved = True
                if not improved:
                    break
            if best is None or cur_bits < best[2]:
                best = (K, tuple(cur), cur_bits)
    return best


# =========================================================
# Analyse
# =========================================================

def analyze(mesh_path: str):
    t0 = time.time()
    A, B, C, V, off = gather(mesh_path)
    n = A.shape[0]
    n_m = len(off) - 1
    print(f"\n=== {Path(mesh_path).stem} ({n:,} para steps, "
          f"{n_m:,} meshlets) ===")

    # Canonical baseline
    R_can = V - (A + B - C)
    Lr_c, Le_c, Lm_c = loss_min_per_mesh(R_can, off)
    bpv_can = Lm_c / (n * 3)
    print(f"  canonical (1,1,-1): L_rice={Lr_c:,}  L_eg={Le_c:,}  "
          f"L_min={Lm_c:,}  ({bpv_can:.4f} BPV/elem)")

    # IRLS init
    t1 = time.time()
    W = fit_irls_per_axis(A, B, C, V)
    print(f"  IRLS L1 ({time.time()-t1:.2f}s):")
    for d, name in enumerate("xyz"):
        print(f"    {name}: ({W[d,0]:+.4f}, {W[d,1]:+.4f}, {W[d,2]:+.4f})")

    # Per-axis search w/ per-meshlet L_min loss
    # Subsample meshlets for speed; verify on full.
    n_m = len(off) - 1
    if n_m > SUBSAMPLE_MESHLETS:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(n_m, size=SUBSAMPLE_MESHLETS, replace=False))
        # Reconstruct subsample arrays + sub-offsets
        rows_idx = np.concatenate([
            np.arange(off[i], off[i+1]) for i in sel
        ])
        A_s, B_s, C_s, V_s = A[rows_idx], B[rows_idx], C[rows_idx], V[rows_idx]
        sub_counts = np.diff(off)[sel]
        off_s = np.concatenate(([0], np.cumsum(sub_counts)))
        sub_tag = f"sub {SUBSAMPLE_MESHLETS:,} meshlets"
    else:
        A_s, B_s, C_s, V_s, off_s = A, B, C, V, off
        sub_tag = "full"
    t2 = time.time()
    chosen_K = []
    chosen_N = np.zeros((3, 3), dtype=np.int64)
    Lm_g = 0
    for d in range(3):
        K, ns, bits_sub = search_axis(A_s[:, d], B_s[:, d], C_s[:, d],
                                       V_s[:, d], off_s, W[d])
        chosen_K.append(K)
        chosen_N[d] = ns
        # verify on full
        bits_full = axis_l_min(A[:, d], B[:, d], C[:, d], V[:, d], off,
                                int(ns[0]), int(ns[1]), int(ns[2]), K)
        Lm_g += bits_full
        print(f"    axis {d} {'xyz'[d]}: K={K} ns={ns} "
              f"L_min(full)={bits_full:,} (sub {bits_sub:,})")
    bpv_g = Lm_g / (n * 3)
    print(f"  generalized (per-meshlet L_min, search {time.time()-t2:.1f}s): "
          f"{bpv_g:.4f} BPV/elem ({bpv_g - bpv_can:+.4f})")
    # also full L_rice/L_eg under chosen weights
    R_g = np.stack([
        V[:, d] - predict_int_axis(A[:, d], B[:, d], C[:, d],
                                   int(chosen_N[d,0]), int(chosen_N[d,1]),
                                   int(chosen_N[d,2]), chosen_K[d])
        for d in range(3)
    ], axis=1)
    Lr_g, Le_g, Lm_g2 = loss_min_per_mesh(R_g, off)
    assert Lm_g == Lm_g2
    print(f"    → L_rice={Lr_g:,}  L_eg={Le_g:,}  L_min={Lm_g:,}")

    final = min(Lm_c, Lm_g)
    print(f"  -> min(canon, fit) total bits saved = {Lm_c - final:+,}  "
          f"(BPV save vs canon: {(Lm_c - final)/(n*3):+.4f})")
    print(f"  total: {time.time()-t0:.1f}s")
    return {
        "mesh": Path(mesh_path).stem,
        "n_para": n,
        "L_canon": Lm_c, "L_gen": Lm_g,
        "save_bits": Lm_c - final,
        "save_bpv": (Lm_c - final) / (n * 3),
        "ns": chosen_N.tolist(), "K": chosen_K,
    }


if __name__ == "__main__":
    meshes = sys.argv[1:] or ["assets/tank.obj"]
    rows = [analyze(m) for m in meshes]
    print("\n=== SUMMARY (per-meshlet L_min) ===")
    for r in rows:
        print(f"  {r['mesh']:16s}  save {r['save_bits']:+10,} bits  "
              f"({r['save_bpv']:+.4f} BPV/elem)")

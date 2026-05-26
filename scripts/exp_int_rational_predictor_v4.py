"""Per-mesh integer-rational parallelogram predictor — v4.

v4 changes vs v3:
  * IRLS (iteratively reweighted least squares) replaces Adam L1.
    Exact L1 closed-form via Newton-style update — converges in 5-10
    iterations, no learning-rate tuning.
  * Multi-start: M_STARTS random int triples drawn around fp64 fit, plus
    extra seeds in [-2D, 2D]. Run coord descent on each; keep min.
  * Rice loss: replace Shannon entropy with Rice code length
    rice_bits(u, k) = (u >> k) + k + 1, with k* per axis chosen to
    minimise total. This is what the deployed bitstream actually pays.
  * Always min(canonical, fit) → cannot regress.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5_dup import _walk_meshlet


CACHE_DIR = ROOT / "cache" / "predictor_samples"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

KS = (4, 5, 6, 7, 8, 9, 10)
SEARCH_RADIUS = 3
COORD_DESCENT_ROUNDS = 4
IRLS_ITERS = 20
IRLS_EPS = 1.0
IRLS_TOL = 1e-6
M_STARTS = 12                 # multi-start count per (axis, K)
SUBSAMPLE_N = 200_000
EARLY_STOP_SLACK = 1.05
EARLY_STOP_PROBES = 8


# ----------------- zigzag + Rice cost -----------------

def _zigzag(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0, (-x << 1) - 1, x << 1).astype(np.int64)


def best_rice_k(u: np.ndarray) -> int:
    """Optimal Rice parameter k for zigzag-mapped residuals."""
    if u.size == 0:
        return 0
    mu = u.mean()
    if mu <= 0:
        return 0
    return max(0, int(np.floor(np.log2(mu))))


def rice_bits_axis(r: np.ndarray) -> float:
    """Total Rice code bits for one residual axis."""
    u = _zigzag(r)
    k = best_rice_k(u)
    # cost = (u >> k) + k + 1 per element
    quotient_bits = (u >> k).sum()
    suffix_bits = u.size * (k + 1)
    return float(quotient_bits + suffix_bits)


def shannon_bits_axis(r: np.ndarray) -> float:
    """Shannon entropy via bincount — kept for reference."""
    u = _zigzag(r)
    counts = np.bincount(u)
    counts = counts[counts > 0]
    n = counts.sum()
    p = counts / n
    return float(-(counts * np.log2(p)).sum())


def total_bpv(R: np.ndarray, scorer) -> float:
    if R.size == 0:
        return 0.0
    tot_bits = sum(scorer(R[:, d].astype(np.int64)) for d in range(R.shape[1]))
    return tot_bits / R.size


# ----------------- sample gather + cache -----------------

def gather_samples(mesh_path: str):
    cache = CACHE_DIR / f"{Path(mesh_path).stem}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["A"], z["B"], z["C"], z["V"]
    print(f"  [gather] {mesh_path} (cold cache, slow)...")
    prep = load_or_prepare(mesh_path, max_verts=256, max_tris=256,
                           precision_error=1.0 / 4096.0,
                           precision_mode="bbox_frac",
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    global_codes = prep["global_codes"]
    plans = prep["plans"]
    A, B, C, V = [], [], [], []
    for plan in plans:
        l2g = np.asarray(plan["local_to_global"], dtype=np.int64)
        true_codes = global_codes[l2g].astype(np.int64)
        for v_local, kind, refs in _walk_meshlet(plan):
            if kind != "para":
                continue
            a, b, c = refs
            A.append(true_codes[a]); B.append(true_codes[b])
            C.append(true_codes[c]); V.append(true_codes[v_local])
    A = np.array(A, dtype=np.int64); B = np.array(B, dtype=np.int64)
    C = np.array(C, dtype=np.int64); V = np.array(V, dtype=np.int64)
    np.savez_compressed(cache, A=A, B=B, C=C, V=V)
    return A, B, C, V


# ----------------- IRLS for L1 regression -----------------

def irls_l1(X: np.ndarray, y: np.ndarray, n_iter=IRLS_ITERS,
            eps=IRLS_EPS, tol=IRLS_TOL) -> np.ndarray:
    """Iteratively Reweighted Least Squares for L1 regression.

    minimise sum |y - X w|. Each iter solves weighted LS with
    weight_i = 1 / max(|residual_i|, eps).
    """
    # init: standard LS
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(n_iter):
        r = y - X @ w
        wts = 1.0 / np.maximum(np.abs(r), eps)
        # weighted lstsq: (X^T W X) w = X^T W y
        sqrt_w = np.sqrt(wts)
        Xw = X * sqrt_w[:, None]
        yw = y * sqrt_w
        w_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def irls_per_axis(A, B, C, V) -> np.ndarray:
    """3x3 weights via IRLS per axis."""
    W = np.zeros((3, 3), dtype=np.float64)
    for d in range(3):
        X = np.stack([A[:, d], B[:, d], C[:, d]], axis=1).astype(np.float64)
        y = V[:, d].astype(np.float64)
        W[d] = irls_l1(X, y)
    return W


# ----------------- int predictor + Rice cost -----------------

def predict_int_axis(a, b, c, n0, n1, n2, K):
    """Single 1-div with arithmetic shift. Bit-exact w/ deploy decoder."""
    D = 1 << K
    half = D >> 1
    s = n0 * a + n1 * b + n2 * c
    return (s + half) >> K


def coord_descent_axis(a, b, c, v, start_ns, K,
                       scorer, rounds=COORD_DESCENT_ROUNDS):
    """±1 coord descent on int triple, geometric steps. Returns (ns, bits)."""
    cur = list(start_ns)
    pred = predict_int_axis(a, b, c, *cur, K)
    cur_bits = scorer(v - pred)
    for _ in range(rounds):
        improved = False
        for step_mag in (1, 2, 4):
            for i in range(3):
                for sign in (-1, 1):
                    step = sign * step_mag
                    cand = cur.copy()
                    cand[i] += step
                    pred = predict_int_axis(a, b, c, *cand, K)
                    bits = scorer(v - pred)
                    if bits < cur_bits:
                        cur, cur_bits = cand, bits
                        improved = True
        if not improved:
            break
    return tuple(cur), cur_bits


def multistart_search(a, b, c, v, w_axis, K, scorer,
                      m_starts=M_STARTS, R=SEARCH_RADIUS,
                      global_best_bits: float = float("inf")):
    """Multi-start: snap-to-int(fp fit) + grid corners + random seeds.
    Run coord descent from each, return min."""
    D = 1 << K
    base_fp = [int(round(w_axis[i] * D)) for i in range(3)]

    # early-stop: probe fp-snap point
    pred = predict_int_axis(a, b, c, *base_fp, K)
    probe_bits = scorer(v - pred)
    if probe_bits > EARLY_STOP_SLACK * global_best_bits:
        return None

    seeds = [tuple(base_fp)]
    # canonical (only valid as seed if K big enough)
    seeds.append((D, D, -D))
    # grid corners around fp snap (radius R)
    for s0 in (-R, R):
        for s1 in (-R, R):
            for s2 in (-R, R):
                seeds.append((base_fp[0]+s0, base_fp[1]+s1, base_fp[2]+s2))
    # random seeds in [-2D, 2D]
    rng = np.random.default_rng(7919 + K)
    while len(seeds) < m_starts + 9:  # 9 = 1 fp + 1 canonical + 8 corners
        seeds.append((int(rng.integers(-2*D, 2*D)),
                      int(rng.integers(-2*D, 2*D)),
                      int(rng.integers(-2*D, 2*D))))

    best_ns, best_bits = None, float("inf")
    for seed in seeds:
        ns, bits = coord_descent_axis(a, b, c, v, seed, K, scorer)
        if bits < best_bits:
            best_bits = bits
            best_ns = ns
    return best_ns, best_bits


# ----------------- analyse -----------------

def analyze(mesh_path: str, scorer_name: str = "rice"):
    scorer = rice_bits_axis if scorer_name == "rice" else shannon_bits_axis
    t0 = time.time()
    A, B, C, V = gather_samples(mesh_path)
    n = A.shape[0]
    n_total = n * 3
    t_gather = time.time() - t0
    print(f"\n=== {Path(mesh_path).stem} ({n:,} para steps, gather {t_gather:.2f}s, "
          f"scorer={scorer_name}) ===")

    # Canonical baseline
    R_can = V - (A + B - C)
    bpv_can = total_bpv(R_can, scorer)
    # also report Shannon for reference
    bpv_can_shan = total_bpv(R_can, shannon_bits_axis)
    print(f"  canonical (1,1,-1): {bpv_can:.4f} BPV (Rice) | "
          f"{bpv_can_shan:.4f} BPV (Shannon)")

    # IRLS fit
    t1 = time.time()
    W = irls_per_axis(A, B, C, V)
    t_irls = time.time() - t1
    print(f"  IRLS L1 ({t_irls:.2f}s) weights per axis:")
    for d, name in enumerate("xyz"):
        print(f"    {name}: ({W[d,0]:+.4f}, {W[d,1]:+.4f}, {W[d,2]:+.4f})")
    pred_fp = np.stack([
        W[d, 0]*A[:, d] + W[d, 1]*B[:, d] + W[d, 2]*C[:, d] for d in range(3)
    ], axis=1)
    R_fp = V - np.round(pred_fp).astype(np.int64)
    bpv_fp = total_bpv(R_fp, scorer)
    print(f"  fp64 per-axis (IRLS L1):         {bpv_fp:.4f} BPV (Rice) "
          f"({bpv_fp - bpv_can:+.4f})")

    # int-rational multi-start + per-axis K + Rice loss
    t2 = time.time()
    if n > SUBSAMPLE_N:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=SUBSAMPLE_N, replace=False)
        As, Bs, Cs, Vs = A[idx], B[idx], C[idx], V[idx]
        sub_tag = f"subsample {SUBSAMPLE_N:,}"
    else:
        As, Bs, Cs, Vs = A, B, C, V
        sub_tag = "full"
    best_per_axis_K = []
    chosen = np.zeros((3, 3), dtype=np.int64)
    skipped = 0
    for d in range(3):
        best = None
        global_best_bits = float("inf")
        for K in KS:
            res = multistart_search(
                As[:, d], Bs[:, d], Cs[:, d], Vs[:, d],
                W[d], K, scorer, global_best_bits=global_best_bits,
            )
            if res is None:
                skipped += 1
                continue
            ns, bits = res
            if bits < global_best_bits:
                global_best_bits = bits
            if best is None or bits < best[2]:
                best = (K, ns, bits)
        best_per_axis_K.append(best[0])
        chosen[d] = best[1]

    # verify on full
    tot_bits_int = 0.0
    for d in range(3):
        K = best_per_axis_K[d]
        ns = chosen[d]
        pred = predict_int_axis(A[:, d], B[:, d], C[:, d],
                                int(ns[0]), int(ns[1]), int(ns[2]), K)
        tot_bits_int += scorer(V[:, d] - pred)
    bpv_int = tot_bits_int / n_total
    t_search = time.time() - t2
    print(f"  int search on {sub_tag}, skipped {skipped} K-probes "
          f"(search {t_search:.2f}s): {bpv_int:.4f} BPV (Rice) "
          f"({bpv_int - bpv_can:+.4f})")
    for d, name in enumerate("xyz"):
        K = best_per_axis_K[d]
        ns = chosen[d]
        print(f"    {name}: K={K} ({ns[0]:+d},{ns[1]:+d},{ns[2]:+d})/{1<<K}")
    final = min(bpv_can, bpv_int)
    print(f"  -> final min(canon, int) = {final:.4f} BPV ({final - bpv_can:+.4f})")
    print(f"  total: {time.time() - t0:.2f}s")

    return {
        "mesh": Path(mesh_path).stem,
        "n_para": n,
        "bpv_canon": bpv_can,
        "bpv_fp": bpv_fp,
        "bpv_int": bpv_int,
        "final": final,
        "save": bpv_can - final,
        "per_axis_K": best_per_axis_K,
        "per_axis_ns": chosen.tolist(),
    }


if __name__ == "__main__":
    meshes = sys.argv[1:] or ["assets/stanford-bunny.obj"]
    rows = [analyze(m) for m in meshes]
    print("\n=== SUMMARY (Rice) ===")
    print(f"{'mesh':16s}  {'n_para':>10s}  {'canon':>7s}  {'fp':>7s}  "
          f"{'int':>7s}  {'final':>7s}  {'save':>8s}")
    for r in rows:
        print(f"{r['mesh']:16s}  {r['n_para']:>10,d}  {r['bpv_canon']:7.4f}  "
              f"{r['bpv_fp']:7.4f}  {r['bpv_int']:7.4f}  {r['final']:7.4f}  "
              f"{r['save']:+8.4f}")

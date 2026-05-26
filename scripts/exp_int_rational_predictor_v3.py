"""Per-mesh integer-rational parallelogram predictor — v3.

v3 changes vs v2:
  * Cache (A, B, C, V) samples to .npz per mesh (sample gather is slow).
  * Vectorized Shannon via np.bincount on zigzag-mapped residuals (~50x).
  * Adam SGD on L1 loss per-axis (Laplacian-optimal entropy surrogate).
  * Extended K in {4..10} + search radius R=3 + per-axis K selection.
  * Coord descent refinement after grid search.
  * Always pick min(canonical, fit) → cannot regress.
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
ADAM_STEPS = 300
ADAM_LR = 0.005
ADAM_BATCH = 65536
SUBSAMPLE_N = 200_000        # search on subset, verify on full
EARLY_STOP_SLACK = 1.05      # skip K if first probe > slack * best_bits
EARLY_STOP_PROBES = 8        # number of corner candidates to probe per K


# ----------------- Shannon via bincount -----------------

def _zigzag(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0, (-x << 1) - 1, x << 1).astype(np.int64)


def shannon_bits_axis(r: np.ndarray) -> float:
    """Total Shannon bits for one residual axis using bincount."""
    u = _zigzag(r)
    counts = np.bincount(u)
    counts = counts[counts > 0]
    n = counts.sum()
    p = counts / n
    return float(-(counts * np.log2(p)).sum())


def shannon_bpv(R: np.ndarray) -> float:
    if R.size == 0:
        return 0.0
    tot_bits = sum(shannon_bits_axis(R[:, d].astype(np.int64)) for d in range(R.shape[1]))
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


# ----------------- Adam L1 SGD per axis -----------------

def adam_l1_per_axis(A, B, C, V, steps=ADAM_STEPS, lr=ADAM_LR, batch=ADAM_BATCH):
    """Return 3x3 weights minimising L1 (MAE) loss per axis via Adam.

    L1 matches Laplacian residual entropy bound; SSE would overweight tails.
    """
    n = A.shape[0]
    W = np.zeros((3, 3), dtype=np.float64)
    # init from canonical
    init = np.array([1.0, 1.0, -1.0])
    for d in range(3):
        w = init.copy()
        m = np.zeros(3); v = np.zeros(3)
        b1, b2, eps = 0.9, 0.999, 1e-8
        rng = np.random.default_rng(42 + d)
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, size=min(batch, n))
            a_b = A[idx, d].astype(np.float64)
            b_b = B[idx, d].astype(np.float64)
            c_b = C[idx, d].astype(np.float64)
            v_b = V[idx, d].astype(np.float64)
            pred = w[0]*a_b + w[1]*b_b + w[2]*c_b
            err = pred - v_b
            sign = np.sign(err)
            # gradient of |pred - v|: sign * X
            g = np.array([
                float(np.mean(sign * a_b)),
                float(np.mean(sign * b_b)),
                float(np.mean(sign * c_b)),
            ])
            m = b1*m + (1 - b1)*g
            v_mom = b2*v + (1 - b2)*(g*g)
            v = v_mom
            m_hat = m / (1 - b1**t)
            v_hat = v_mom / (1 - b2**t)
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        W[d] = w
    return W


# ----------------- vectorized int-pred grid search -----------------

def predict_int_axis(a, b, c, n0, n1, n2, K):
    D = 1 << K
    half = D >> 1
    s = n0 * a + n1 * b + n2 * c
    return np.where(s >= 0, (s + half) >> K, -((-s + half) >> K))


def search_int_axis(a, b, c, v, w_axis, K, R=SEARCH_RADIUS,
                    global_best_bits: float = float("inf")):
    """Grid search ±R around round(w*D), then coord descent. One axis.

    global_best_bits is used for early-stop across K values: if the first
    EARLY_STOP_PROBES candidates are all > slack * global_best_bits we skip
    the rest of the grid and return None.
    """
    D = 1 << K
    base = [int(round(w_axis[i] * D)) for i in range(3)]
    offsets = list(range(-R, R + 1))
    all_offsets = [(d0, d1, d2) for d0 in offsets for d1 in offsets for d2 in offsets]
    # Probe centre + cube corners first for early-stop trigger
    probe_set = [(0, 0, 0)] + [(s0, s1, s2) for s0 in (-R, R) for s1 in (-R, R) for s2 in (-R, R)]
    probe_set = probe_set[:EARLY_STOP_PROBES]
    probe_bits = []
    for d0, d1, d2 in probe_set:
        n0, n1, n2 = base[0]+d0, base[1]+d1, base[2]+d2
        pred = predict_int_axis(a, b, c, n0, n1, n2, K)
        probe_bits.append(shannon_bits_axis(v - pred))
    if min(probe_bits) > EARLY_STOP_SLACK * global_best_bits:
        return None

    best_ns, best_bits = None, float("inf")
    for d0, d1, d2 in all_offsets:
        n0, n1, n2 = base[0]+d0, base[1]+d1, base[2]+d2
        pred = predict_int_axis(a, b, c, n0, n1, n2, K)
        bits = shannon_bits_axis(v - pred)
        if bits < best_bits:
            best_bits = bits
            best_ns = (n0, n1, n2)
    # coord descent
    cur, cur_bits = list(best_ns), best_bits
    for _ in range(COORD_DESCENT_ROUNDS):
        improved = False
        for i in range(3):
            for step in (-1, 1):
                cand = cur.copy()
                cand[i] += step
                pred = predict_int_axis(a, b, c, *cand, K)
                bits = shannon_bits_axis(v - pred)
                if bits < cur_bits:
                    cur, cur_bits = cand, bits
                    improved = True
        if not improved:
            break
    return tuple(cur), cur_bits


# ----------------- analyse -----------------

def analyze(mesh_path: str):
    t0 = time.time()
    A, B, C, V = gather_samples(mesh_path)
    n = A.shape[0]
    n_total = n * 3
    t_gather = time.time() - t0
    print(f"\n=== {Path(mesh_path).stem} ({n:,} para steps, gather {t_gather:.2f}s) ===")

    # Canonical
    R_can = V - (A + B - C)
    bpv_can = shannon_bpv(R_can)
    print(f"  canonical (1,1,-1):              {bpv_can:.4f} BPV")

    # Adam L1 per-axis
    t1 = time.time()
    W = adam_l1_per_axis(A, B, C, V)
    t_adam = time.time() - t1
    print(f"  Adam L1 ({t_adam:.2f}s) weights per axis:")
    for d, name in enumerate("xyz"):
        print(f"    {name}: ({W[d,0]:+.4f}, {W[d,1]:+.4f}, {W[d,2]:+.4f})")

    pred_fp = np.stack([
        W[d, 0]*A[:, d] + W[d, 1]*B[:, d] + W[d, 2]*C[:, d] for d in range(3)
    ], axis=1)
    R_fp = V - np.round(pred_fp).astype(np.int64)
    bpv_fp = shannon_bpv(R_fp)
    print(f"  fp64 per-axis (Adam L1):         {bpv_fp:.4f} BPV ({bpv_fp - bpv_can:+.4f})")

    # int-rational per-axis, per-axis K
    # Search on subsample (fast), verify chosen (K, ns) on full set.
    t2 = time.time()
    if n > SUBSAMPLE_N:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=SUBSAMPLE_N, replace=False)
        As, Bs, Cs, Vs = A[idx], B[idx], C[idx], V[idx]
        sub_n = SUBSAMPLE_N
        sub_tag = f"subsample {SUBSAMPLE_N:,}"
    else:
        As, Bs, Cs, Vs = A, B, C, V
        sub_n = n
        sub_tag = "full"
    best_per_axis_K = []
    chosen = np.zeros((3, 3), dtype=np.int64)
    skipped = 0
    for d in range(3):
        best = None  # (K, ns, bits_sub)
        global_best_bits = float("inf")
        for K in KS:
            res = search_int_axis(As[:, d], Bs[:, d], Cs[:, d], Vs[:, d],
                                  W[d], K, global_best_bits=global_best_bits)
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
    # verify on full set
    tot_bits_int = 0.0
    for d in range(3):
        K = best_per_axis_K[d]
        ns = chosen[d]
        pred = predict_int_axis(A[:, d], B[:, d], C[:, d],
                                int(ns[0]), int(ns[1]), int(ns[2]), K)
        tot_bits_int += shannon_bits_axis(V[:, d] - pred)
    bpv_int = tot_bits_int / n_total
    t_search = time.time() - t2
    print(f"  int search on {sub_tag}, skipped {skipped} K-probes")
    print(f"  int per-axis (search {t_search:.2f}s): {bpv_int:.4f} BPV ({bpv_int - bpv_can:+.4f})")
    for d, name in enumerate("xyz"):
        K = best_per_axis_K[d]
        ns = chosen[d]
        print(f"    {name}: K={K} ({ns[0]:+d},{ns[1]:+d},{ns[2]:+d})/{1<<K}")

    final = min(bpv_can, bpv_int)
    print(f"  -> final min(canon, int) = {final:.4f} BPV ({final - bpv_can:+.4f})")
    print(f"  total time: {time.time() - t0:.2f}s")

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
    meshes = sys.argv[1:] or [
        "assets/stanford-bunny.obj",
        "assets/Monkey.obj",
        "assets/tank.obj",
        "assets/xyzrgb_dragon.obj",
    ]
    rows = [analyze(m) for m in meshes]

    print("\n\n=== SUMMARY ===")
    print(f"{'mesh':16s}  {'n_para':>10s}  {'canon':>7s}  {'fp':>7s}  "
          f"{'int':>7s}  {'final':>7s}  {'save':>8s}")
    for r in rows:
        print(f"{r['mesh']:16s}  {r['n_para']:>10,d}  {r['bpv_canon']:7.4f}  "
              f"{r['bpv_fp']:7.4f}  {r['bpv_int']:7.4f}  {r['final']:7.4f}  "
              f"{r['save']:+8.4f}")

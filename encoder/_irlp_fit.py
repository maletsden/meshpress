"""Per-mesh generalized parallelogram predictor fit.

Optimised version of v7. Same search space (KS=5..9, ±1 grid, multi-start,
2-round coord descent) — only inner numerics are vectorised and axes run
in parallel processes.

Public API:
    fit_predictor(prep) -> (n_3x3 int64, K_3 int64)
"""
from __future__ import annotations
import os
import numpy as np
from multiprocessing import get_context

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco
    prange = range


KS = (5, 6, 7, 8, 9)
SEARCH_RADIUS = 1
COORD_DESCENT_ROUNDS = 2
IRLS_ITERS = 12
IRLS_EPS = 1.0
RICE_K_MAX = 11
EG_K_MAX = 7
SUBSAMPLE_MESHLETS = 4000
SEED_CHUNK = 32        # process this many seed-candidates per batch
PARALLEL_AXES = False  # numba prange already parallelises across seeds

CANON_N = np.array([[1, 1, -1]] * 3, dtype=np.int64)
CANON_K = np.zeros(3, dtype=np.int64)


# ===================== sample gather =====================

def _sample_cache_dir():
    """User cache dir for gathered (A,B,C,V,off) samples."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    p = root / "cache" / "gpred_samples"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _gather_cache_key(global_codes: np.ndarray, plans: list) -> str:
    """Stable fingerprint of (global_codes, plans). Hashes a small sketch.

    Includes n_v, n_meshlets, sum of meshlet sizes, and codes[0]/codes[-1]
    to detect any of: different mesh, different partition, different quant."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(np.int64(len(global_codes)).tobytes())
    h.update(np.int64(len(plans)).tobytes())
    if len(global_codes):
        h.update(global_codes[0].astype(np.int64).tobytes())
        h.update(global_codes[-1].astype(np.int64).tobytes())
        h.update(np.int64(int(global_codes.astype(np.int64).sum())).tobytes())
    sizes = []
    for plan in plans[:32]:           # sketch first 32 plans
        sizes.append(len(plan["local_to_global"]))
    h.update(np.array(sizes, dtype=np.int64).tobytes())
    return h.hexdigest()


def _gather(global_codes, plans, walker, *, use_cache: bool = True):
    cache_path = None
    if use_cache:
        key = _gather_cache_key(global_codes, plans)
        cache_path = _sample_cache_dir() / f"{key}.npz"
        if cache_path.exists():
            z = np.load(cache_path)
            return z["A"], z["B"], z["C"], z["V"], z["off"]
    A, B, C, V, offs = [], [], [], [], [0]
    for plan in plans:
        l2g = np.asarray(plan["local_to_global"], dtype=np.int64)
        tc = global_codes[l2g].astype(np.int64)
        m_count = 0
        for v, kind, refs in walker(plan):
            if kind != "para":
                continue
            a, b, c = refs
            A.append(tc[a]); B.append(tc[b]); C.append(tc[c]); V.append(tc[v])
            m_count += 1
        offs.append(offs[-1] + m_count)
    A_arr = np.array(A, dtype=np.int64); B_arr = np.array(B, dtype=np.int64)
    C_arr = np.array(C, dtype=np.int64); V_arr = np.array(V, dtype=np.int64)
    off_arr = np.array(offs, dtype=np.int64)
    if cache_path is not None:
        np.savez_compressed(cache_path, A=A_arr, B=B_arr, C=C_arr, V=V_arr, off=off_arr)
    return A_arr, B_arr, C_arr, V_arr, off_arr


# ===================== batched scorers =====================

def _zigzag(r):
    return np.where(r < 0, (-r << 1) - 1, r << 1).astype(np.int64)


def _segsum_2d(arr_2d: np.ndarray, off: np.ndarray) -> np.ndarray:
    """arr_2d: (S, n). off: (n_m+1,). Returns (S, n_m) per-meshlet sums."""
    S, n = arr_2d.shape
    if n == 0:
        return np.zeros((S, len(off) - 1), dtype=np.int64)
    # cumsum along axis 1 with leading zero column → diff via off indices.
    cs = np.zeros((S, n + 1), dtype=np.int64)
    np.cumsum(arr_2d, axis=1, out=cs[:, 1:])
    return cs[:, off[1:]] - cs[:, off[:-1]]


_MAX_MESHLET = 256          # STRIDE-dup encoder cap


@njit(cache=True, parallel=True, fastmath=False, boundscheck=False)
def _cost_seeds_njit(a, b, c, v, off, n_arr, K):
    """Numba kernel: per-seed total encoder cost = sum_m min_k min(Rice, EG).

    Flat prange across (n_m * S) tasks. Per-task local u_buf (256 ints).
    """
    S = n_arr.shape[0]
    n_m = off.shape[0] - 1
    total_tasks = n_m * S
    HUGE = (1 << 62)
    cost_2d = np.zeros((n_m, S), dtype=np.int64)
    for t in prange(total_tasks):
        m = t // S
        s = t % S
        start = off[m]
        end = off[m + 1]
        cnt = end - start
        if cnt == 0:
            continue
        u_buf = np.empty(256, dtype=np.int64)
        n0 = n_arr[s, 0]; n1 = n_arr[s, 1]; n2 = n_arr[s, 2]
        for j in range(cnt):
            i = start + j
            s_val = n0 * a[i] + n1 * b[i] + n2 * c[i]
            if K == 0:
                pred = s_val
            else:
                half = 1 << (K - 1)
                pred = (s_val + half) >> K
            r = v[i] - pred
            if r < 0:
                u = (-r << 1) - 1
            else:
                u = r << 1
            u_buf[j] = u
        # min over Rice k = 0..11
        best = HUGE
        for k in range(12):
            cost = 0
            for j in range(cnt):
                cost += (u_buf[j] >> k) + (k + 1)
            if cost < best:
                best = cost
        # min over EG k = 0..7
        for k in range(8):
            cost = 0
            for j in range(cnt):
                q = (u_buf[j] >> k) + 1
                pl = 0
                qq = q
                while qq > 1:
                    qq >>= 1
                    pl += 1
                cost += 2 * pl + 1 + k
            if cost < best:
                best = cost
        cost_2d[m, s] = best
    out = np.zeros(S, dtype=np.int64)
    for s in range(S):
        total = 0
        for m in range(n_m):
            total += cost_2d[m, s]
        out[s] = total
    return out


def encoder_cost_batch(a, b, c, v, off, n_arr: np.ndarray, K: int) -> np.ndarray:
    """n_arr shape (S, 3). Returns (S,) total encoder bits per seed.

    Vectorised: builds all S residuals, all RICE_K shifted views, all EG
    per-k costs in stacked tensors. Memory peak ≈ S * len(a) * 8 bytes;
    caller should chunk if needed.
    """
    S = n_arr.shape[0]
    n0 = n_arr[:, 0][:, None]
    n1 = n_arr[:, 1][:, None]
    n2 = n_arr[:, 2][:, None]
    s = n0 * a[None, :] + n1 * b[None, :] + n2 * c[None, :]   # (S, n)
    if K == 0:
        pred = s
    else:
        half = (1 << K) >> 1
        pred = (s + half) >> K
    r = v[None, :] - pred                                      # (S, n)
    u = np.where(r < 0, (-r << 1) - 1, r << 1).astype(np.int64)

    n_m = len(off) - 1
    counts = np.diff(off).astype(np.int64)                     # (n_m,)

    # Rice — per-meshlet min over k.
    best_per_meshlet = np.full((S, n_m), np.iinfo(np.int64).max, dtype=np.int64)
    for k in range(RICE_K_MAX + 1):
        seg = _segsum_2d(u >> k, off)                          # (S, n_m)
        seg += counts[None, :] * (k + 1)
        best_per_meshlet = np.minimum(best_per_meshlet, seg)
    # EG — per-meshlet min over k.
    log_cache = {}   # k -> 2*floor(log2((u>>k)+1)) + 1 + k  (S, n)
    for k in range(EG_K_MAX + 1):
        q = (u >> k) + 1
        prefix_len = np.floor(np.log2(q.astype(np.float64))).astype(np.int64)
        per = 2 * prefix_len + 1 + k
        seg = _segsum_2d(per, off)
        best_per_meshlet = np.minimum(best_per_meshlet, seg)

    # Empty meshlets contribute 0.
    best_per_meshlet[:, counts == 0] = 0
    return best_per_meshlet.sum(axis=1)


def encoder_cost_one(a, b, c, v, off, n_triple, K) -> int:
    """Convenience: cost for one seed (used by IRLS init and verify)."""
    cost = encoder_cost_batch(a, b, c, v, off,
                              np.asarray(n_triple, dtype=np.int64).reshape(1, 3),
                              K)
    return int(cost[0])


# ===================== IRLS L1 =====================

def _irls_l1(X, y, n_iter=IRLS_ITERS, eps=IRLS_EPS):
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


def _fit_irls(A, B, C, V) -> np.ndarray:
    W = np.zeros((3, 3))
    for d in range(3):
        X = np.stack([A[:, d], B[:, d], C[:, d]], axis=1).astype(np.float64)
        W[d] = _irls_l1(X, V[:, d].astype(np.float64))
    return W


# ===================== per-axis search =====================

EARLY_STOP_SLACK = 1.05


def _search_axis(a, b, c, v, off, w_axis):
    """Returns (K, (n0,n1,n2), bits) under encoder cost = min(Rice, EG).

    Batches all 10 seeds' coord-descent rounds into single numba calls.
    Across-K early stop: skip coord descent if K's best seed initial > 1.05 × best so far.
    """
    best = None
    for K in KS:
        D = 1 << K
        base = [int(round(w_axis[i] * D)) for i in range(3)]
        seeds = [tuple(base), (D, D, -D)]
        for s0 in (-SEARCH_RADIUS, SEARCH_RADIUS):
            for s1 in (-SEARCH_RADIUS, SEARCH_RADIUS):
                for s2 in (-SEARCH_RADIUS, SEARCH_RADIUS):
                    seeds.append((base[0]+s0, base[1]+s1, base[2]+s2))
        S = len(seeds)
        cur = np.array(seeds, dtype=np.int64)              # (S, 3)
        cur_costs = _batch_eval(a, b, c, v, off, cur, K)   # (S,)
        # Across-K early stop.
        if best is not None and int(cur_costs.min()) > EARLY_STOP_SLACK * best[2]:
            continue
        # Batched coord descent: build (S * 6, 3) candidate batch each round.
        active = np.ones(S, dtype=bool)
        for _ in range(COORD_DESCENT_ROUNDS):
            if not active.any():
                break
            # 6 moves per active seed
            sel = np.where(active)[0]
            n_act = len(sel)
            cands = np.empty((n_act * 6, 3), dtype=np.int64)
            for slot, si in enumerate(sel):
                base_v = cur[si]
                for mi, (i, sign) in enumerate(((0,-1),(0,1),(1,-1),(1,1),(2,-1),(2,1))):
                    row = slot * 6 + mi
                    cands[row] = base_v
                    cands[row, i] += sign
            cand_costs = _batch_eval(a, b, c, v, off, cands, K)   # (n_act*6,)
            new_active = np.zeros(S, dtype=bool)
            for slot, si in enumerate(sel):
                grid = cand_costs[slot*6:(slot+1)*6]
                local_best = int(grid.argmin())
                if grid[local_best] < cur_costs[si]:
                    cur[si] = cands[slot*6 + local_best]
                    cur_costs[si] = grid[local_best]
                    new_active[si] = True
            active = new_active
        # Best across this K.
        best_si = int(cur_costs.argmin())
        best_bits = int(cur_costs[best_si])
        if best is None or best_bits < best[2]:
            best = (K, tuple(int(x) for x in cur[best_si]), best_bits)
    return best


def _batch_eval(a, b, c, v, off, n_arr, K, max_meshlet=256):
    """Use numba kernel if available, otherwise fall back to numpy."""
    if HAVE_NUMBA:
        return _cost_seeds_njit(a, b, c, v, off, n_arr, K)
    # Numpy fallback (slower but no numba dep)
    S = n_arr.shape[0]
    if S <= SEED_CHUNK:
        return encoder_cost_batch(a, b, c, v, off, n_arr, K)
    out = np.zeros(S, dtype=np.int64)
    for i in range(0, S, SEED_CHUNK):
        out[i:i+SEED_CHUNK] = encoder_cost_batch(
            a, b, c, v, off, n_arr[i:i+SEED_CHUNK], K
        )
    return out


def _fit_one_axis(args):
    """Worker entry: fit one axis. args = (d, A, B, C, V, off, W_d)."""
    d, A, B, C, V, off, w_d = args
    K, ns, bits = _search_axis(A[:, d], B[:, d], C[:, d], V[:, d], off, w_d)
    return d, K, ns, bits


# ===================== public =====================

def _do_fit(A_s, B_s, C_s, V_s, off_s) -> tuple[np.ndarray, np.ndarray]:
    W = _fit_irls(A_s, B_s, C_s, V_s)
    n_arr = np.zeros((3, 3), dtype=np.int64)
    K_arr = np.zeros(3, dtype=np.int64)
    tasks = [(d, A_s, B_s, C_s, V_s, off_s, W[d]) for d in range(3)]
    if PARALLEL_AXES:
        ctx_name = "fork" if os.name != "nt" else "spawn"
        ctx = get_context(ctx_name)
        with ctx.Pool(3) as pool:
            for d, K, ns, _ in pool.imap_unordered(_fit_one_axis, tasks):
                K_arr[d] = K
                n_arr[d] = ns
    else:
        for task in tasks:
            d, K, ns, _ = _fit_one_axis(task)
            K_arr[d] = K
            n_arr[d] = ns
    return n_arr, K_arr


def fit_predictor(prep) -> tuple[np.ndarray, np.ndarray]:
    """Returns chosen (n_3x3 int64, K_3 int64). 12-byte spec for the bitstream
    header. Per-axis canonical fallback under encoder cost = min(Rice, EG)."""
    from encoder.paradelta_v5_dup import _walk_meshlet
    A, B, C, V, off = _gather(prep["global_codes"], prep["plans"], _walk_meshlet)
    n_m = len(off) - 1
    if n_m == 0 or len(A) == 0:
        return CANON_N.copy(), CANON_K.copy()

    if n_m > SUBSAMPLE_MESHLETS:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(n_m, size=SUBSAMPLE_MESHLETS, replace=False))
        rows = np.concatenate([np.arange(off[i], off[i+1]) for i in sel])
        A_s, B_s, C_s, V_s = A[rows], B[rows], C[rows], V[rows]
        counts = np.diff(off)[sel]
        off_s = np.concatenate(([0], np.cumsum(counts)))
    else:
        A_s, B_s, C_s, V_s, off_s = A, B, C, V, off

    n_fit, K_fit = _do_fit(A_s, B_s, C_s, V_s, off_s)

    # Per-axis canonical fallback under encoder cost on FULL.
    for d in range(3):
        b_fit = encoder_cost_one(A[:, d], B[:, d], C[:, d], V[:, d], off,
                                 n_fit[d], int(K_fit[d]))
        b_can = encoder_cost_one(A[:, d], B[:, d], C[:, d], V[:, d], off,
                                 CANON_N[d], int(CANON_K[d]))
        if b_can < b_fit:
            n_fit[d] = CANON_N[d]
            K_fit[d] = CANON_K[d]
    return n_fit, K_fit

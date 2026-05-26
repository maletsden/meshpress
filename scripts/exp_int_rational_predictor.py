"""Integer-rational parallelogram predictor — measure BPV impact.

For each meshlet, gather (a, b, c, v) tuples from para steps where v is
the new vertex and (a, b, c) is the parallelogram context. Fit global
weights (w0, w1, w2) by least squares. Then quantize each weight to a
rational n/D where D = 2^K (so the predictor becomes (n0*a + n1*b + n2*c)
>> K with rounding, all integer arithmetic). Compare per-axis residual
Shannon entropy for:

  (a) canonical             (1, 1, -1)
  (b) fp32 fitted           (w0, w1, w2)
  (c) integer-rational      (n0, n1, n2) / 2^K  at K in {4, 5, 6, 7}

We measure RESIDUAL SHANNON ENTROPY rather than wiring through Rice/EG
which would mask small wins in the selection-tag quantization.
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


def shannon_bpv(arr: np.ndarray) -> float:
    """Shannon entropy of (n, 3) residual array in bits per residual
    element (across all axes). Bound from below for Rice / Exp-Golomb."""
    if arr.size == 0:
        return 0.0
    bits = 0.0
    n = 0
    for d in range(arr.shape[1]):
        x = arr[:, d].astype(np.int64)
        # zigzag
        u = np.where(x < 0, (-x << 1) - 1, x << 1).astype(np.int64)
        cnt = Counter(u.tolist())
        total = sum(cnt.values())
        for c in cnt.values():
            p = c / total
            bits += -c * np.log2(p)
        n += total
    return bits / max(n, 1)


def gather_samples(mesh_path: str, max_meshlets: int | None = None):
    prep = load_or_prepare(mesh_path, max_verts=256, max_tris=256,
                           precision_error=1.0/4096.0,
                           precision_mode="bbox_frac",
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    global_codes = prep["global_codes"]
    plans = prep["plans"]
    if max_meshlets:
        plans = plans[:max_meshlets]
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
    return (np.array(A), np.array(B), np.array(C), np.array(V))


def fit_weights(A, B, C, V) -> np.ndarray:
    """Single fp32 weight vector shared across axes (matching v3 STRIDE)."""
    # Stack samples: each row = (a_d, b_d, c_d), target = v_d, over all axes.
    X = np.vstack([np.stack([A[:, d], B[:, d], C[:, d]], axis=1) for d in range(3)])
    y = np.concatenate([V[:, d] for d in range(3)])
    w, *_ = np.linalg.lstsq(X.astype(np.float64), y.astype(np.float64), rcond=None)
    return w


def best_rational(w: float, K: int) -> int:
    """Best n such that |w - n / 2^K| minimised."""
    D = 1 << K
    return int(round(w * D))


def predict_int(A, B, C, n0: int, n1: int, n2: int, K: int) -> np.ndarray:
    """Integer-rational predictor: round((n0*a + n1*b + n2*c) >> K) with
    arithmetic-shift rounding (add half-D before shifting)."""
    D = 1 << K
    s = n0 * A + n1 * B + n2 * C
    # Round-half-up for ints: (s + sign(s)*D/2) // D
    half = D // 2
    rounded = np.where(s >= 0, (s + half) // D, -((-s + half) // D))
    return rounded.astype(np.int64)


def analyze(mesh_path: str, max_meshlets: int | None = None):
    A, B, C, V = gather_samples(mesh_path, max_meshlets)
    n = A.shape[0]
    print(f"\n=== {Path(mesh_path).stem} ({n:,} para steps) ===")

    # (a) Canonical
    pred = A + B - C
    R = V - pred
    bpv_canon = shannon_bpv(R)
    print(f"  (1, 1, -1) canonical:                Shannon {bpv_canon:.3f} BPV")

    # (b) Fp32 fitted
    w = fit_weights(A, B, C, V)
    pred_fp = w[0]*A.astype(np.float64) + w[1]*B.astype(np.float64) + w[2]*C.astype(np.float64)
    R_fp = V - np.round(pred_fp).astype(np.int64)
    bpv_fp = shannon_bpv(R_fp)
    print(f"  fp32 fitted ({w[0]:+.4f}, {w[1]:+.4f}, {w[2]:+.4f}): "
          f"Shannon {bpv_fp:.3f} BPV  (delta {bpv_fp - bpv_canon:+.3f})")

    # (c) Integer-rational at K in {4, 5, 6, 7}
    best_save = 0.0
    for K in (4, 5, 6, 7):
        n0 = best_rational(w[0], K)
        n1 = best_rational(w[1], K)
        n2 = best_rational(w[2], K)
        pred_int = predict_int(A, B, C, n0, n1, n2, K)
        R_int = V - pred_int
        bpv_int = shannon_bpv(R_int)
        save = bpv_canon - bpv_int
        if save > best_save:
            best_save = save
        print(f"  K={K} ({n0}, {n1}, {n2})/{1<<K}: "
              f"Shannon {bpv_int:.3f} BPV  (delta {bpv_int - bpv_canon:+.3f})")
    return bpv_canon, bpv_fp, best_save


if __name__ == "__main__":
    meshes = sys.argv[1:] or [
        "assets/stanford-bunny.obj",
        "assets/Monkey.obj",
        "assets/xyzrgb_dragon.obj",
    ]
    for m in meshes:
        analyze(m)

"""
Learned linear lifting predictor for float-domain wavelet.

Structure mirrors Haar / CDF 5/3 lifting, but the `predict` step is a
per-level linear kernel whose weights are fit from the model's actual
interior vertex data by least squares:

    forward:
        detail[k] = odd[k] - sum_j w_lvl[j] * even_padded[k + j]
        approx    = even                     (identity update — like Haar)
    inverse:
        even      = approx
        odd[k]    = detail[k] + sum_j w_lvl[j] * even_padded[k + j]

Perfect reconstruction holds for ANY kernel (no numerical constraint) because
the decoder applies the same linear combination to the same even samples.
Compression hinges on the kernel cancelling odd from its even neighbors —
i.e. minimising the variance of `detail`.

Training is a closed-form lstsq per level, ridge-regularised. Per-level
kernels because successive approximations have different statistics
(sub-sampling makes the signal progressively smoother).

Storage overhead: kernels are stored once per mesh in the global header
(kernel_size × 32 bits × n_levels, typically <100 B).

Error amplification during inverse lifting is bounded by the kernel's L1
norm: per-step added error ≤ |kernel|_1 · prev_err. The quantization budget
allocator (`per_level_deltas`) accepts an `amp` override; we pass the
max kernel L1 norm so δ is tight enough for a real error bound.
"""

from __future__ import annotations

import numpy as np

from utils.wavelet import _stream_bits


# ============================================================
# Reflection padding + prediction
# ============================================================

def _pad_even_reflect(even, kernel_size):
    """Reflection-padded even array so a kernel of `kernel_size` can slide
    across all N odd positions. Offsets relative to odd index k:
        K=1: [0]              → Haar-like (predict = even[k])
        K=2: [0, 1]           → CDF 5/3-like (mean of two neighbours)
        K=4: [-1, 0, 1, 2]    → wider symmetric window
    """
    K = kernel_size
    N = len(even)
    pad_left = (K - 1) // 2
    pad_right = K - 1 - pad_left
    padded = np.empty(N + pad_left + pad_right, dtype=np.float64)
    padded[pad_left:pad_left + N] = even
    for i in range(pad_left):
        src = min(pad_left - 1 - i + 1, N - 1)
        padded[i] = even[src]
    for i in range(pad_right):
        src = max(N - 2 - i, 0)
        padded[pad_left + N + i] = even[src]
    return padded


def _predict_odd(even, kernel):
    K = len(kernel)
    N = len(even)
    padded = _pad_even_reflect(even, K)
    pred = np.zeros(N, dtype=np.float64)
    for j in range(K):
        pred += kernel[j] * padded[j:j + N]
    return pred


# ============================================================
# Forward / inverse lifting with learned predictor
# ============================================================

def learned_decompose(values, kernels, target_base=32):
    """kernels: list of 1D arrays (per level). Use kernels[-1] past the list end."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= target_base:
        return values.copy(), [], n

    n_pad = 1
    while n_pad < n:
        n_pad *= 2
    sig = np.empty(n_pad, dtype=np.float64)
    sig[:n] = values
    sig[n:] = values[-1]

    levels = []
    cur = sig
    lvl = 0
    while len(cur) > target_base and len(cur) >= 2:
        even = cur[0::2]
        odd = cur[1::2]
        k = kernels[lvl] if lvl < len(kernels) else kernels[-1]
        detail = odd - _predict_odd(even, k)
        levels.append(detail)
        cur = even
        lvl += 1
    return cur, levels, n


def learned_reconstruct(base, levels, kernels, orig_n):
    cur = np.asarray(base, dtype=np.float64).copy()
    for l_idx in range(len(levels) - 1, -1, -1):
        d = levels[l_idx]
        k = kernels[l_idx] if l_idx < len(kernels) else kernels[-1]
        pred = _predict_odd(cur, k)
        odd = d + pred
        N = len(cur)
        out = np.empty(2 * N, dtype=np.float64)
        out[0::2] = cur
        out[1::2] = odd
        cur = out
    return cur[:orig_n]


# ============================================================
# Kernel fitting (lstsq per level)
# ============================================================

def fit_kernels(streams, n_levels_max, kernel_size=4,
                target_base=32, ridge=1e-6):
    """Per-level linear predictor kernel fit by ridge-regularised lstsq.

    streams: iterable of 1D arrays — one signal per (meshlet, axis).
             Shifted by per-stream min so they all start at 0.
    n_levels_max: cap on fitted levels (e.g. log2(max_interior / target_base)).
    kernel_size: K (see _pad_even_reflect for offset convention).
    ridge: small L2 regulariser for numerical stability on thin level data.

    Returns list of kernels (length n_levels_max). Unfit levels remain at
    the default flat-average [1/K, …] kernel.
    """
    kernels = [np.full(kernel_size, 1.0 / kernel_size, dtype=np.float64)
               for _ in range(n_levels_max)]

    for target_l in range(n_levels_max):
        X_rows = []
        y_rows = []
        for stream in streams:
            n = len(stream)
            if n <= target_base:
                continue
            n_pad = 1
            while n_pad < n:
                n_pad *= 2
            sig = np.empty(n_pad, dtype=np.float64)
            sig[:n] = stream
            sig[n:] = stream[-1]
            cur = sig
            # Decompose up to (not including) target_l with identity-U so we
            # just subsample down to the level we want training data for.
            for _ in range(target_l):
                if len(cur) <= target_base or len(cur) < 2:
                    break
                cur = cur[0::2]
            if len(cur) <= target_base or len(cur) < 2:
                continue
            even = cur[0::2]
            odd = cur[1::2]
            padded = _pad_even_reflect(even, kernel_size)
            for k in range(len(odd)):
                X_rows.append(padded[k:k + kernel_size])
                y_rows.append(odd[k])
        if not X_rows:
            break
        X = np.asarray(X_rows)
        y = np.asarray(y_rows)
        XtX = X.T @ X + ridge * np.eye(kernel_size)
        Xty = X.T @ y
        kernels[target_l] = np.linalg.solve(XtX, Xty)

    return kernels


# ============================================================
# Interior encoder with learned predictor
# ============================================================

def quantize_interior_learned_wavelet(positions, per_coord_err, kernels,
                                       schedule="geometric", ratio=2.0,
                                       target_base=32, amp=None):
    """Interior encoder using a learned linear lifting predictor.

    Args:
        positions:   (n, 3) float interior positions (pre-normalised).
        per_coord_err: target per-coordinate max error.
        kernels:     per-level predictor kernels (from fit_kernels).
        schedule / ratio: δ-allocation policy (see utils.float_wavelet).
        amp:         error amplification factor. If None, uses the max
                     kernel L1 norm (conservative: guarantees the
                     δ_base + amp·Σδ_k ≤ 2ε bound).

    Returns (recon, total_bits, meta). Bits include PACKED per-level meta
    (3 × (int16 min + uint8 bits) per level) matching the float-wavelet
    packed layout.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        return positions.copy(), 0, []

    if amp is None:
        amp = max(1.0, *(float(np.abs(k).sum()) for k in kernels))

    recon = np.empty_like(positions)
    total_bits = 3 * 32  # per-axis float32 offset
    per_axis_streams = []
    L_final = 0
    for d in range(3):
        offset = float(positions[:, d].min())
        shifted = positions[:, d] - offset
        base, levels, orig_n = learned_decompose(shifted, kernels, target_base)
        L = len(levels)
        L_final = max(L_final, L)

        # Per-level δ with caller-supplied amp
        if schedule == "uniform" or ratio == 1.0 or L == 0:
            delta = 2.0 * per_coord_err / (1.0 + amp * L)
            delta_base = delta
            delta_levels = [delta] * L
        else:
            geo_sum = (ratio * (ratio ** L - 1)) / (ratio - 1)
            denom = 1.0 + amp * geo_sum
            A = 2.0 * per_coord_err / denom
            delta_base = A
            delta_levels = [A * (ratio ** (L - k)) for k in range(L)]

        base_q = np.round(base / delta_base).astype(np.int64)
        levels_q = [np.round(levels[k] / delta_levels[k]).astype(np.int64)
                    for k in range(L)]
        base_r = base_q.astype(np.float64) * delta_base
        levels_r = [levels_q[k].astype(np.float64) * delta_levels[k]
                    for k in range(L)]
        recon[:, d] = learned_reconstruct(base_r, levels_r, kernels, orig_n) + offset
        per_axis_streams.append((base_q, levels_q))

    # Packed per-level metadata (same layout as float_wavelet_packed):
    #     3 × (int16 min + uint8 bits_per_code) = 9 B per level
    def _pack_level(streams_this_level):
        body_bits = 0
        for codes in streams_this_level:
            if len(codes) == 0:
                continue
            mn = int(codes.min())
            rng = int(codes.max() - mn)
            b = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
            shifted = codes - mn
            body_bits += _stream_bits(shifted, b)
        return body_bits + 3 * (16 + 8)

    base_streams = [per_axis_streams[d][0] for d in range(3)]
    total_bits += _pack_level(base_streams)
    for lvl in range(L_final):
        # All axes produced the same L (decompose is deterministic on size)
        level_streams = [per_axis_streams[d][1][lvl] for d in range(3)]
        total_bits += _pack_level(level_streams)

    return recon, total_bits, {"packed": True, "n_levels": L_final, "amp": amp}

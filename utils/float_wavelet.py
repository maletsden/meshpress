"""
Float-domain wavelets for interior meshlet vertices.

Lifting-scheme transforms on float64 signals. Two variants supported:

    'haar':  unnormalized lifting Haar.
        forward:  d = odd - even;  a = even
    'cdf53': CDF 5/3 reversible wavelet (JPEG 2000 lossless core).
        forward:  d[k] = odd[k] - (even[k] + even[k+1]) / 2
                  a[k] = even[k] + (d[k-1] + d[k]) / 4

Both support arbitrary signal lengths via trailing last-value padding to a
power of two above `target_base`. Boundary handling inside CDF 5/3 uses
symmetric (reflect) extension, the JPEG 2000 default.

Coefficient quantization uses a per-level uniform step δ_k chosen so total
reconstruction error stays within a target ε:

    Haar:     δ_base + Σ_k δ_k         ≤ 2ε
    CDF 5/3:  δ_base + (3/2) Σ_k δ_k   ≤ 2ε   (tighter; 5/3 amplifies 1.5x)

`per_level_deltas()` distributes the budget. Two schedules:

    'uniform':   every level gets the same δ (conservative baseline).
    'geometric': δ grows by a factor `ratio` from base toward fine-scale
                 details. Intuition: fine-scale details have smaller
                 magnitude so a coarser δ still keeps few distinct codes
                 while shrinking per-coefficient bit width.

For smooth mesh signals, geometric with ratio=2 gives the base the tightest
precision and doubles δ each level toward levels[0] (highest freq). Total
error budget is preserved by construction.
"""

from __future__ import annotations

import numpy as np

from utils.wavelet import _stream_bits


# ============================================================
# Haar lifting (float)
# ============================================================

def float_haar_decompose(values, target_base=32):
    """Unnormalized lifting Haar. Returns (base, levels, original_n).
    Last-value padding to a power of two above target_base."""
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
    while len(cur) > target_base and len(cur) >= 2:
        even = cur[0::2]
        odd = cur[1::2]
        levels.append(odd - even)
        cur = even
    return cur, levels, n


def float_haar_reconstruct(base, levels, orig_n):
    cur = np.asarray(base, dtype=np.float64).copy()
    for detail in reversed(levels):
        n = len(cur)
        out = np.empty(2 * n, dtype=np.float64)
        out[0::2] = cur
        out[1::2] = cur + detail
        cur = out
    return cur[:orig_n]


# ============================================================
# CDF 5/3 lifting (float)
# ============================================================

def _cdf53_forward_step(signal):
    """One level of CDF 5/3 forward lifting with symmetric boundaries.
    Input even length N. Output (approx, detail) each of length N/2."""
    e = signal[0::2].astype(np.float64, copy=True)
    o = signal[1::2].astype(np.float64, copy=True)
    N = len(e)

    # Predict: d[k] = o[k] - (e[k] + e[k+1]) / 2
    # Symmetric boundary: at k = N-1 reflect e[k+1] -> e[k-1]
    d = np.empty_like(o)
    if N >= 2:
        d[:-1] = o[:-1] - 0.5 * (e[:-1] + e[1:])
        d[-1] = o[-1] - 0.5 * (e[-1] + e[-2])
    else:
        d[:] = o - e

    # Update: a[k] = e[k] + (d[k-1] + d[k]) / 4
    # Symmetric boundary: at k = 0 reflect d[-1] -> d[1]
    a = e.copy()
    if N >= 2:
        a[1:] += 0.25 * (d[:-1] + d[1:])
        a[0] += 0.25 * (d[1] + d[0])
    else:
        a[0] += 0.5 * d[0]

    return a, d


def _cdf53_inverse_step(a, d):
    """One level of CDF 5/3 inverse lifting (symmetric boundaries)."""
    N = len(a)
    e = a.astype(np.float64, copy=True)

    # Undo update
    if N >= 2:
        e[1:] -= 0.25 * (d[:-1] + d[1:])
        e[0] -= 0.25 * (d[1] + d[0])
    else:
        e[0] -= 0.5 * d[0]

    # Undo predict
    o = np.empty_like(d)
    if N >= 2:
        o[:-1] = d[:-1] + 0.5 * (e[:-1] + e[1:])
        o[-1] = d[-1] + 0.5 * (e[-1] + e[-2])
    else:
        o[:] = d + e

    # Interleave
    out = np.empty(2 * N, dtype=np.float64)
    out[0::2] = e
    out[1::2] = o
    return out


def float_cdf53_decompose(values, target_base=32):
    """CDF 5/3 float decomposition. Returns (base, levels, original_n)."""
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
    while len(cur) > target_base and len(cur) >= 4:
        a, d = _cdf53_forward_step(cur)
        levels.append(d)
        cur = a
    return cur, levels, n


def float_cdf53_reconstruct(base, levels, orig_n):
    cur = np.asarray(base, dtype=np.float64).copy()
    for d in reversed(levels):
        cur = _cdf53_inverse_step(cur, d)
    return cur[:orig_n]


# ============================================================
# Per-level delta allocation
# ============================================================

# Error amplification per detail level at inverse lifting.
# Haar:  1.0   (cumulative error = δ_base/2 + Σ δ_k/2)
# CDF53: 1.5   (cumulative error = δ_base/2 + (3/4) Σ δ_k)
_AMPLIFICATION = {"haar": 1.0, "cdf53": 1.5}


def per_level_deltas(per_coord_err, n_levels, wavelet="haar",
                     schedule="geometric", ratio=2.0):
    """Split the per-coord error budget across base + detail levels.

    Constraint (per _AMPLIFICATION):
        δ_base + amp * Σ δ_k ≤ 2 * per_coord_err

    Returns (delta_base, delta_levels) where len(delta_levels) == n_levels.
    level[0] = finest detail (highest freq).
    """
    amp = _AMPLIFICATION[wavelet]

    if n_levels == 0:
        return 2.0 * per_coord_err, []

    if schedule == "uniform" or ratio == 1.0:
        # δ_base = δ_k = δ.  δ (1 + amp*L) = 2ε
        delta = 2.0 * per_coord_err / (1.0 + amp * n_levels)
        return delta, [delta] * n_levels

    if schedule == "geometric":
        # weights[base] = 1, weights[level_k] = ratio^(n_levels - k)
        # (level 0 = finest = largest weight -> largest δ)
        # Sum = 1 + amp * (r + r^2 + ... + r^L) = 1 + amp * r * (r^L - 1) / (r - 1)
        if ratio == 1.0:
            return per_level_deltas(per_coord_err, n_levels, wavelet, "uniform")
        geo_sum = (ratio * (ratio ** n_levels - 1)) / (ratio - 1)
        denom = 1.0 + amp * geo_sum
        A = 2.0 * per_coord_err / denom
        delta_base = A
        delta_levels = [A * (ratio ** (n_levels - k)) for k in range(n_levels)]
        return delta_base, delta_levels

    raise ValueError(f"Unknown schedule: {schedule}")


# ============================================================
# Quantized coefficient streams
# ============================================================

def _quantize_codes_bits(codes):
    """Bits for one coefficient stream: fixed-width body + 5B metadata."""
    if len(codes) == 0:
        return 0
    rng = int(codes.max() - codes.min())
    bits = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
    shifted = codes - codes.min()
    return _stream_bits(shifted, bits) + 5 * 8


def quantize_float_wavelet_axis(values, per_coord_err, wavelet="haar",
                                 schedule="geometric", ratio=2.0,
                                 target_base=32):
    """Per-axis float wavelet + per-level uniform quantization.

    Returns (bits, recon_values, meta).
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return 0, values.copy(), {"wavelet": wavelet, "n_levels": 0}

    if wavelet == "haar":
        base, levels, orig_n = float_haar_decompose(values, target_base)
        reconstruct = float_haar_reconstruct
    elif wavelet == "cdf53":
        base, levels, orig_n = float_cdf53_decompose(values, target_base)
        reconstruct = float_cdf53_reconstruct
    else:
        raise ValueError(f"Unknown wavelet: {wavelet}")

    L = len(levels)
    delta_base, delta_levels = per_level_deltas(
        per_coord_err, L, wavelet, schedule, ratio)

    base_q = np.round(base / delta_base).astype(np.int64)
    levels_q = [np.round(levels[k] / delta_levels[k]).astype(np.int64)
                for k in range(L)]

    base_r = base_q.astype(np.float64) * delta_base
    levels_r = [levels_q[k].astype(np.float64) * delta_levels[k]
                for k in range(L)]
    recon = reconstruct(base_r, levels_r, orig_n)

    bits = _quantize_codes_bits(base_q)
    for lq in levels_q:
        bits += _quantize_codes_bits(lq)
    # Schedule is derivable from (wavelet, n, target_base, per_coord_err)
    # in the global header, so no per-axis δ metadata needed here.

    return bits, recon, {
        "wavelet": wavelet,
        "n_levels": L,
        "delta_base": delta_base,
        "delta_levels": delta_levels,
        "base_size": len(base),
    }


def quantize_interior_float_wavelet_packed(positions, per_coord_err,
                                            wavelet="haar",
                                            schedule="geometric", ratio=2.0,
                                            target_base=32):
    """Same semantics as quantize_interior_float_wavelet but with metadata
    packed per-level (shared across the 3 axes) instead of per-stream.

    Per-meshlet overhead:
        3 * float32 offset                     = 12 B
    Per-level (base + each detail level), SHARED across axes:
        1 B  max_bits_per_code (uint8)
        3 * 2 B int16 mins (one per axis)      = 6 B
        Total: 7 B per level

    vs. original per-stream layout: 3 * 5 B = 15 B per level.
    Savings: 8 B per level per meshlet.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        return positions.copy(), 0, []

    recon = np.empty_like(positions)
    # Per-axis: transform, quantize, collect integer codes
    per_axis_streams = []  # list of (base_q, [level_q_0, level_q_1, ...])
    for d in range(3):
        offset = float(positions[:, d].min())
        shifted = positions[:, d] - offset
        if wavelet == "haar":
            base, levels, orig_n = float_haar_decompose(shifted, target_base)
            reconstruct = float_haar_reconstruct
        elif wavelet == "cdf53":
            base, levels, orig_n = float_cdf53_decompose(shifted, target_base)
            reconstruct = float_cdf53_reconstruct
        else:
            raise ValueError(f"Unknown wavelet: {wavelet}")

        L = len(levels)
        delta_base, delta_levels = per_level_deltas(
            per_coord_err, L, wavelet, schedule, ratio)

        base_q = np.round(base / delta_base).astype(np.int64)
        levels_q = [np.round(levels[k] / delta_levels[k]).astype(np.int64)
                    for k in range(L)]
        base_r = base_q.astype(np.float64) * delta_base
        levels_r = [levels_q[k].astype(np.float64) * delta_levels[k]
                    for k in range(L)]
        recon[:, d] = reconstruct(base_r, levels_r, orig_n) + offset
        per_axis_streams.append((base_q, levels_q))

    # Pack metadata: per-level, shared bits_per_code across 3 axes + 3 mins.
    total_bits = 3 * 32  # 3 x float32 offsets
    L = len(per_axis_streams[0][1])

    def _pack_level(streams_this_level):
        """streams_this_level: list of 3 int64 code arrays (one per axis).

        Per-level metadata (3B/stream instead of 5B/stream):
            per axis: 2B int16 min + 1B uint8 bits_per_code = 3B
            3 axes total: 9B/level
        Bits are NOT shared across axes — axis ranges often differ (e.g.,
        flat regions on one axis), so per-axis bit widths save more on the
        body than sharing saves on metadata.
        """
        body_bits = 0
        for codes in streams_this_level:
            if len(codes) == 0:
                continue
            mn = int(codes.min())
            rng = int(codes.max() - mn)
            b = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
            shifted = codes - mn
            body_bits += _stream_bits(shifted, b)  # picks min(fixed, entropy)
        meta_bits = 3 * (16 + 8)  # per axis: int16 min + uint8 bits
        return body_bits + meta_bits

    # Base stream (one per axis)
    base_streams = [per_axis_streams[d][0] for d in range(3)]
    total_bits += _pack_level(base_streams)

    # Detail levels
    for lvl in range(L):
        level_streams = [per_axis_streams[d][1][lvl] for d in range(3)]
        total_bits += _pack_level(level_streams)

    return recon, total_bits, {"wavelet": wavelet, "n_levels": L,
                                 "packed": True}


def quantize_interior_float_wavelet(positions, per_coord_err, wavelet="haar",
                                     schedule="geometric", ratio=2.0,
                                     target_base=32):
    """Float wavelet interior encoder — drop-in for the integer baseline in
    utils.boundary_split.

    Returns (recon, total_bits, meta_per_axis).
    Total bits include the per-meshlet 3x32-bit float offset header.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        return positions.copy(), 0, []

    total_bits = 3 * 32  # per-axis float32 offset
    recon = np.empty_like(positions)
    meta = []
    for d in range(3):
        offset = float(positions[:, d].min())
        shifted = positions[:, d] - offset
        bits, rec, m = quantize_float_wavelet_axis(
            shifted, per_coord_err, wavelet, schedule, ratio, target_base)
        recon[:, d] = rec + offset
        total_bits += bits
        meta.append(m)
    return recon, total_bits, meta


# ============================================================
# Backwards-compatible Haar entry point (used by earlier encoder)
# ============================================================

def quantize_interior_float_haar(positions, per_coord_err, target_base=32,
                                  schedule="geometric", ratio=2.0):
    return quantize_interior_float_wavelet(
        positions, per_coord_err,
        wavelet="haar", schedule=schedule, ratio=ratio,
        target_base=target_base,
    )
"""
Hierarchical Haar wavelet transform for mesh vertex compression.
GPU-friendly parallel reduction: O(log N) decode steps.
"""

import numpy as np


def haar_decompose(values, target_base=32):
    """Lazy Haar wavelet decomposition.
    Repeatedly: keep even-indexed, store (odd - even) as detail.

    Args:
        values: (N,) array
        target_base: stop reducing when base reaches this size

    Returns:
        base: (M,) array of base values (M <= target_base)
        levels: list of (K,) arrays, finest level first
        original_n: original length before padding
    """
    original_n = len(values)

    # Pad to next power of 2
    n = len(values)
    if n <= target_base:
        return values.copy(), [], original_n

    n_padded = 1
    while n_padded < n:
        n_padded *= 2

    signal = np.zeros(n_padded)
    signal[:n] = values
    signal[n:] = values[-1]  # constant extension

    levels = []
    current = signal

    while len(current) > target_base and len(current) >= 2:
        n_half = len(current) // 2
        even = current[0::2]  # base (keep)
        odd = current[1::2]   # predict from even
        detail = odd - even   # residual
        levels.append(detail)
        current = even

    # levels[0] = finest (most values), levels[-1] = coarsest (fewest values)
    return current, levels, original_n


def haar_reconstruct(base, levels):
    """Inverse Haar wavelet: reconstruct from base + detail levels.

    Args:
        base: (M,) array
        levels: list of detail arrays, finest first

    Returns:
        (N,) reconstructed values
    """
    current = base.copy()

    # Process levels from coarsest to finest
    for detail in reversed(levels):
        n = len(current)
        assert len(detail) == n, f"Detail length {len(detail)} != base length {n}"
        reconstructed = np.zeros(2 * n)
        reconstructed[0::2] = current        # even = base
        reconstructed[1::2] = current + detail  # odd = base + detail
        current = reconstructed

    return current


def estimate_wavelet_bits(values, max_error_per_value, target_base=32):
    """Decompose, quantize per-level, estimate total bits.

    Args:
        values: (N,) array of one coordinate channel
        max_error_per_value: allowed quantization error per value per level
        target_base: base size after reduction

    Returns:
        dict with total_bits, per_level_bits, base_bits, and diagnostic info
    """
    from collections import Counter

    base, levels, orig_n = haar_decompose(values, target_base)
    n_levels = len(levels)

    if n_levels == 0:
        # No wavelet, just quantize directly
        rng = values.max() - values.min() if len(values) > 1 else 0.001
        bits = _bits_for_error(rng, max_error_per_value)
        codes = _quantize(values, values.min(), values.max(), bits)
        total = _stream_bits(codes, bits)
        return {
            "total_bits": total,
            "base_bits": total,
            "level_bits": [],
            "n_levels": 0,
            "base_size": len(values),
            "level_sizes": [],
            "level_ranges": [],
            "level_bit_depths": [],
        }

    # Error budget: divide across (base + n_levels) stages
    n_stages = n_levels + 1
    per_stage_err = max_error_per_value / n_stages

    # Base values
    base_range = base.max() - base.min() if len(base) > 1 else 0.001
    base_bits_depth = _bits_for_error(base_range, per_stage_err)
    base_codes = _quantize(base, base.min(), base.max(), base_bits_depth)
    base_bits = _stream_bits(base_codes, base_bits_depth)
    # Base range metadata: min, max as float16 + bits as uint8 = 5 bytes
    base_bits += 5 * 8

    # Detail levels (finest first)
    level_bits_list = []
    level_ranges = []
    level_bit_depths = []
    level_sizes = []

    for lvl_idx, detail in enumerate(levels):
        # Only count real values (not padding)
        # At this level, the real count depends on original_n
        # For simplicity, use all values (padding adds minimal overhead for large meshlets)
        rng = detail.max() - detail.min() if len(detail) > 1 else 0.001
        bits = _bits_for_error(rng, per_stage_err)
        codes = _quantize(detail, detail.min(), detail.max(), bits)
        lvl_bits = _stream_bits(codes, bits)
        # Per-level metadata: min, max, bits = 5 bytes
        lvl_bits += 5 * 8

        level_bits_list.append(lvl_bits)
        level_ranges.append(rng)
        level_bit_depths.append(bits)
        level_sizes.append(len(detail))

    total = base_bits + sum(level_bits_list)

    return {
        "total_bits": total,
        "base_bits": base_bits,
        "level_bits": level_bits_list,
        "n_levels": n_levels,
        "base_size": len(base),
        "level_sizes": level_sizes,
        "level_ranges": level_ranges,
        "level_bit_depths": level_bit_depths,
    }


def wavelet_reconstruct_quantized(values, max_error_per_value, target_base=32):
    """Decompose, quantize, dequantize, reconstruct — for accuracy measurement."""
    base, levels, orig_n = haar_decompose(values, target_base)
    n_levels = len(levels)

    if n_levels == 0:
        rng = values.max() - values.min() if len(values) > 1 else 0.001
        per_stage_err = max_error_per_value
        bits = _bits_for_error(rng, per_stage_err)
        codes = _quantize(values, values.min(), values.max(), bits)
        return _dequantize(codes, values.min(), values.max(), bits)[:orig_n]

    n_stages = n_levels + 1
    per_stage_err = max_error_per_value / n_stages

    # Quantize/dequantize base
    base_range = base.max() - base.min() if len(base) > 1 else 0.001
    base_bits = _bits_for_error(base_range, per_stage_err)
    base_q = _dequantize(_quantize(base, base.min(), base.max(), base_bits),
                         base.min(), base.max(), base_bits)

    # Quantize/dequantize each level
    levels_q = []
    for detail in levels:
        rng = detail.max() - detail.min() if len(detail) > 1 else 0.001
        bits = _bits_for_error(rng, per_stage_err)
        detail_q = _dequantize(_quantize(detail, detail.min(), detail.max(), bits),
                               detail.min(), detail.max(), bits)
        levels_q.append(detail_q)

    # Reconstruct
    reconstructed = haar_reconstruct(base_q, levels_q)
    return reconstructed[:orig_n]


# ---- Internal helpers (duplicated to keep wavelet.py self-contained) ----

def _bits_for_error(val_range, max_err):
    if max_err <= 0 or val_range <= 0:
        return 1
    return max(1, int(np.ceil(np.log2(val_range / (2 * max_err) + 1))))

def _quantize(vals, lo, hi, bits):
    mx = (1 << bits) - 1
    norm = np.clip((vals - lo) / (hi - lo + 1e-15), 0, 1)
    return np.round(norm * mx).astype(np.int64)

def _dequantize(codes, lo, hi, bits):
    return codes.astype(np.float64) / ((1 << bits) - 1) * (hi - lo) + lo

def _stream_bits(codes, fixed_bits):
    from collections import Counter
    n = len(codes)
    if n == 0:
        return 0
    plain = n * fixed_bits
    counts = Counter(codes.tolist())
    total = len(codes)
    ent = -sum((c / total) * np.log2(c / total) for c in counts.values())
    arith = n * ent + 32
    return min(plain, arith)


# ============================================================
# Integer Haar wavelet (lossless on integer input — for crack-free encoding)
# ============================================================

def haar_decompose_int(values, target_base=32):
    """Integer Haar wavelet: lossless on int64 input.
    detail = odd - even (exact integer subtraction)."""
    original_n = len(values)
    values = np.asarray(values, dtype=np.int64)

    if len(values) <= target_base:
        return values.copy(), [], original_n

    n_padded = 1
    while n_padded < len(values):
        n_padded *= 2

    signal = np.zeros(n_padded, dtype=np.int64)
    signal[:len(values)] = values
    signal[len(values):] = values[-1]

    levels = []
    current = signal

    while len(current) > target_base and len(current) >= 2:
        even = current[0::2]
        odd = current[1::2]
        detail = odd - even  # exact integer
        levels.append(detail)
        current = even

    return current, levels, original_n


def haar_reconstruct_int(base, levels):
    """Inverse integer Haar: exact reconstruction."""
    current = np.asarray(base, dtype=np.int64).copy()
    for detail in reversed(levels):
        n = len(current)
        recon = np.zeros(2 * n, dtype=np.int64)
        recon[0::2] = current
        recon[1::2] = current + detail
        current = recon
    return current


def estimate_wavelet_bits_int(int_values, target_base=32, wavelet_type="haar"):
    """Estimate bits for integer wavelet (lossless, no quantization error).
    wavelet_type: 'haar', 'cdf53', 'linear_pred', or 'delta'."""
    int_values = np.asarray(int_values, dtype=np.int64)

    if wavelet_type == "cdf53":
        base, levels, orig_n = cdf53_decompose_int(int_values, target_base)
    elif wavelet_type == "linear_pred":
        base, levels, orig_n = linear_pred_decompose_int(int_values, target_base)
    elif wavelet_type == "delta":
        base, levels, orig_n = delta_decompose_int(int_values, target_base)
    elif wavelet_type == "seg_delta":
        base, levels, orig_n = segmented_delta_decompose_int(int_values, target_base)
    elif wavelet_type == "seg_haar":
        base, levels, orig_n = segment_haar_decompose_int(int_values, target_base)
    else:
        base, levels, orig_n = haar_decompose_int(int_values, target_base)

    return _estimate_levels_bits(base, levels)


def _estimate_levels_bits(base, levels):
    """Shared bit estimation for any integer wavelet decomposition."""
    n_levels = len(levels)

    if n_levels == 0:
        rng = int(base.max() - base.min()) if len(base) > 1 else 0
        bits = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
        codes = base - base.min()
        total = _stream_bits(codes, bits)
        return {"total_bits": total + 5 * 8, "n_levels": 0,
                "base_size": len(base), "base_bits": bits,
                "level_bits": [], "level_ranges": []}

    base_rng = int(base.max() - base.min()) if len(base) > 1 else 0
    base_bits = max(1, int(np.ceil(np.log2(base_rng + 2)))) if base_rng > 0 else 1
    base_codes = base - base.min()
    base_total = _stream_bits(base_codes, base_bits) + 5 * 8

    level_bits_list = []
    level_ranges = []
    total_level_bits = 0
    for detail in levels:
        rng = int(detail.max() - detail.min()) if len(detail) > 1 else 0
        bits = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
        codes = detail - detail.min()
        lvl_bits = _stream_bits(codes, bits) + 5 * 8
        level_bits_list.append(lvl_bits)
        level_ranges.append(rng)
        total_level_bits += lvl_bits

    return {
        "total_bits": base_total + total_level_bits,
        "n_levels": n_levels,
        "base_size": len(base),
        "base_bits": base_bits,
        "level_bits": level_bits_list,
        "level_ranges": level_ranges,
    }


# ============================================================
# CDF 5/3 integer lifting wavelet (JPEG2000 lossless)
# ============================================================

def cdf53_decompose_int(values, target_base=32):
    """CDF 5/3 lifting scheme: better prediction than Haar using 2 neighbors.
    Lossless on integers via floor division.

    Predict: d[i] = odd[i] - floor((even[i] + even[i+1]) / 2)
    Update:  s[i] = even[i] + floor((d[i-1] + d[i] + 2) / 4)

    The update step smooths the approximation coefficients, giving a better
    multi-resolution representation. Detail coefficients are smaller than
    Haar because the prediction uses 2 neighbors instead of 1.
    """
    original_n = len(values)
    values = np.asarray(values, dtype=np.int64)

    if len(values) <= target_base:
        return values.copy(), [], original_n

    n_padded = 1
    while n_padded < len(values):
        n_padded *= 2

    signal = np.zeros(n_padded, dtype=np.int64)
    signal[:len(values)] = values
    signal[len(values):] = values[-1]

    levels = []
    current = signal

    while len(current) > target_base and len(current) >= 2:
        n = len(current)
        n_half = n // 2
        even = current[0::2].copy()
        odd = current[1::2].copy()

        # Predict step: d[i] = odd[i] - floor((even[i] + even[min(i+1, n_half-1)]) / 2)
        detail = np.zeros(n_half, dtype=np.int64)
        for i in range(n_half):
            i_next = min(i + 1, n_half - 1)
            detail[i] = odd[i] - ((even[i] + even[i_next]) >> 1)

        # Update step: s[i] = even[i] + floor((d[max(i-1,0)] + d[i] + 2) / 4)
        smooth = np.zeros(n_half, dtype=np.int64)
        for i in range(n_half):
            i_prev = max(i - 1, 0)
            smooth[i] = even[i] + ((detail[i_prev] + detail[i] + 2) >> 2)

        levels.append(detail)
        current = smooth

    return current, levels, original_n


def cdf53_reconstruct_int(base, levels):
    """Inverse CDF 5/3 lifting: exact reconstruction."""
    current = np.asarray(base, dtype=np.int64).copy()

    for detail in reversed(levels):
        n_half = len(current)
        n = 2 * n_half

        # Inverse update: even[i] = s[i] - floor((d[max(i-1,0)] + d[i] + 2) / 4)
        even = np.zeros(n_half, dtype=np.int64)
        for i in range(n_half):
            i_prev = max(i - 1, 0)
            even[i] = current[i] - ((detail[i_prev] + detail[i] + 2) >> 2)

        # Inverse predict: odd[i] = d[i] + floor((even[i] + even[min(i+1, n_half-1)]) / 2)
        odd = np.zeros(n_half, dtype=np.int64)
        for i in range(n_half):
            i_next = min(i + 1, n_half - 1)
            odd[i] = detail[i] + ((even[i] + even[i_next]) >> 1)

        recon = np.zeros(n, dtype=np.int64)
        recon[0::2] = even
        recon[1::2] = odd
        current = recon

    return current


# ============================================================
# Delta prediction wavelet (simplest: predict from previous value)
# ============================================================

def delta_decompose_int(values, target_base=32):
    """Delta encoding as a wavelet: detail = value[i] - value[i-1].
    NOT hierarchical — single level, all deltas at once.
    Very effective when EdgeBreaker ordering gives monotonic sequences.
    """
    original_n = len(values)
    values = np.asarray(values, dtype=np.int64)

    if len(values) <= target_base:
        return values.copy(), [], original_n

    # Single-level: store first target_base values as base, rest as deltas
    base = values[:target_base].copy()
    deltas = values[target_base:] - values[target_base - 1:-1]  # delta from previous

    return base, [deltas], original_n


def delta_reconstruct_int(base, levels):
    """Inverse delta: cumulative sum."""
    if not levels:
        return base.copy()
    deltas = levels[0]
    result = np.zeros(len(base) + len(deltas), dtype=np.int64)
    result[:len(base)] = base
    for i in range(len(deltas)):
        result[len(base) + i] = result[len(base) + i - 1] + deltas[i]
    return result


# ============================================================
# Linear-prediction wavelet (predict using linear interpolation from 2 neighbors)
# ============================================================

# ============================================================
# Segmented delta (GPU-parallel: independent segments, no cross-segment dependency)
# ============================================================

def segmented_delta_decompose_int(values, n_segments=32):
    """Segmented delta encoding: store n_segments evenly-spaced anchor values,
    then delta-encode each segment independently.

    GPU decode: each warp handles 1 segment (prefix sum of ~8 values).
    Fully parallel across segments. Only 2 __syncthreads total.
    """
    original_n = len(values)
    values = np.asarray(values, dtype=np.int64)

    if len(values) <= n_segments:
        return values.copy(), [], original_n

    # Anchor indices: evenly spaced
    seg_size = len(values) // n_segments
    remainder = len(values) - seg_size * n_segments

    base = np.zeros(n_segments, dtype=np.int64)
    all_deltas = np.zeros(len(values) - n_segments, dtype=np.int64)

    delta_idx = 0
    for s in range(n_segments):
        # This segment: [start, end)
        start = s * seg_size + min(s, remainder)
        end = (s + 1) * seg_size + min(s + 1, remainder)
        seg_vals = values[start:end]

        base[s] = seg_vals[0]
        for i in range(1, len(seg_vals)):
            all_deltas[delta_idx] = seg_vals[i] - seg_vals[i - 1]
            delta_idx += 1

    return base, [all_deltas[:delta_idx]], original_n


def segmented_delta_reconstruct_int(base, levels):
    """Inverse segmented delta: prefix sum within each segment."""
    if not levels or len(levels[0]) == 0:
        return base.copy()

    n_segments = len(base)
    deltas = levels[0]
    total_n = n_segments + len(deltas)

    # Figure out segment sizes
    seg_size = total_n // n_segments
    remainder = total_n - seg_size * n_segments

    result = np.zeros(total_n, dtype=np.int64)
    delta_idx = 0
    for s in range(n_segments):
        start = s * seg_size + min(s, remainder)
        end = (s + 1) * seg_size + min(s + 1, remainder)
        result[start] = base[s]
        for i in range(start + 1, end):
            result[i] = result[i - 1] + deltas[delta_idx]
            delta_idx += 1

    return result


# ============================================================
# Hybrid: independent Haar wavelet within each segment (GPU-parallel)
# ============================================================

def segment_haar_decompose_int(values, n_segments=32):
    """Haar wavelet within each independent segment.
    Each segment of ~8 values gets its own mini-wavelet (3 levels).
    All segments decode in parallel (same as segmented delta).
    But wavelet captures multi-scale patterns better than sequential delta.
    """
    original_n = len(values)
    values = np.asarray(values, dtype=np.int64)

    if len(values) <= n_segments:
        return values.copy(), [], original_n

    seg_size = len(values) // n_segments
    remainder = len(values) - seg_size * n_segments

    # Each segment produces: 1 base value + (seg_size-1) wavelet coefficients
    all_bases = []
    all_coeffs = []

    for s in range(n_segments):
        start = s * seg_size + min(s, remainder)
        end = (s + 1) * seg_size + min(s + 1, remainder)
        seg = values[start:end].copy()

        # Pad segment to power of 2
        n = len(seg)
        n_pad = 1
        while n_pad < n:
            n_pad *= 2
        padded = np.zeros(n_pad, dtype=np.int64)
        padded[:n] = seg
        padded[n:] = seg[-1]

        # Haar wavelet within segment
        current = padded
        seg_levels = []
        while len(current) > 1:
            even = current[0::2]
            odd = current[1::2]
            detail = odd - even
            seg_levels.append(detail)
            current = even

        # current[0] is the single base value
        all_bases.append(current[0])
        # Flatten all wavelet coefficients: finest first
        for lvl in seg_levels:
            all_coeffs.extend(lvl.tolist())

    base = np.array(all_bases, dtype=np.int64)
    coeffs = np.array(all_coeffs, dtype=np.int64) if all_coeffs else np.array([], dtype=np.int64)
    return base, [coeffs] if len(coeffs) > 0 else [], original_n


def segment_haar_reconstruct_int(base, levels):
    """Inverse: reconstruct each segment's wavelet independently."""
    if not levels or len(levels[0]) == 0:
        return base.copy()

    n_segments = len(base)
    coeffs = levels[0]
    # Figure out segment structure from coefficient count
    total_coeffs = len(coeffs)
    total_values = n_segments + total_coeffs  # base + all coefficients = original length

    seg_size = total_values // n_segments
    remainder = total_values - seg_size * n_segments

    result = np.zeros(total_values, dtype=np.int64)
    coeff_idx = 0

    for s in range(n_segments):
        start = s * seg_size + min(s, remainder)
        end = (s + 1) * seg_size + min(s + 1, remainder)
        n = end - start

        # Reconstruct wavelet for this segment
        n_pad = 1
        while n_pad < n:
            n_pad *= 2

        # Collect this segment's wavelet levels
        seg_levels = []
        sz = n_pad
        while sz > 1:
            half = sz // 2
            seg_levels.append(coeffs[coeff_idx:coeff_idx + half])
            coeff_idx += half
            sz = half

        # Inverse wavelet
        current = np.array([base[s]], dtype=np.int64)
        for detail in reversed(seg_levels):
            recon = np.zeros(len(current) * 2, dtype=np.int64)
            recon[0::2] = current
            recon[1::2] = current + detail[:len(current)]
            current = recon

        result[start:end] = current[:n]

    return result


def linear_pred_decompose_int(values, target_base=32):
    """Hierarchical wavelet with linear interpolation prediction.
    Predict: d[i] = odd[i] - round((even[i] + even[i+1]) / 2)
    No update step (simpler than CDF 5/3, avoids update overhead).
    Better than Haar for smooth signals, GPU-friendly.
    """
    original_n = len(values)
    values = np.asarray(values, dtype=np.int64)

    if len(values) <= target_base:
        return values.copy(), [], original_n

    n_padded = 1
    while n_padded < len(values):
        n_padded *= 2

    signal = np.zeros(n_padded, dtype=np.int64)
    signal[:len(values)] = values
    signal[len(values):] = values[-1]

    levels = []
    current = signal

    while len(current) > target_base and len(current) >= 2:
        n_half = len(current) // 2
        even = current[0::2].copy()
        odd = current[1::2].copy()

        # Predict with linear interpolation (2 neighbors)
        detail = np.zeros(n_half, dtype=np.int64)
        for i in range(n_half):
            i_next = min(i + 1, n_half - 1)
            detail[i] = odd[i] - ((even[i] + even[i_next]) >> 1)

        levels.append(detail)
        current = even  # NO update step — just subsample

    return current, levels, original_n


def linear_pred_reconstruct_int(base, levels):
    """Inverse linear prediction wavelet."""
    current = np.asarray(base, dtype=np.int64).copy()

    for detail in reversed(levels):
        n_half = len(current)
        even = current

        # Inverse predict
        odd = np.zeros(n_half, dtype=np.int64)
        for i in range(n_half):
            i_next = min(i + 1, n_half - 1)
            odd[i] = detail[i] + ((even[i] + even[i_next]) >> 1)

        recon = np.zeros(2 * n_half, dtype=np.int64)
        recon[0::2] = even
        recon[1::2] = odd
        current = recon

    return current
"""
Block-DCT-II encoder for cycle-3 of the interior-transforms lab.

Per axis, per meshlet, on the sorted interior float stream:
  1. Pad to a multiple of B (zero-pad).
  2. Reshape into (n_blocks, B), apply orthonormal DCT-II per block.
  3. Quantize coefficients with delta_k schedule:
       'uniform'   - delta_k = delta everywhere
       'geometric' - delta_k = delta * ratio^(k/B)  (HF coarse)
  4. Inverse DCT for reconstruction; max-error bounded analytically.

Error bound: orthonormal DCT-II preserves L2 norm. Per-sample time-domain
error in block i is

    |x_recon[i] - x[i]| = |sum_k D^T[i, k] * (c_recon[k] - c[k])|
                        <= ||D^T[i, :]||_2 * ||c_recon - c||_2
                        =  1 * (1/2) * sqrt(sum_k delta_k^2)

(D is orthonormal, so each row of D^T has unit L2 norm.) For the schedule
to keep per-coord error <= eps:

    (1/2) * sqrt(sum_k delta_k^2) <= eps
    sum_k delta_k^2 <= 4 * eps^2

Uniform schedule: delta_k = delta -> delta = 2 eps / sqrt(B).
Geometric schedule: delta_k = delta_0 * ratio^(k/B), pick delta_0 to satisfy
the same constraint.

Packing layout mirrors the existing Haar packing
(`utils.float_wavelet.quantize_interior_float_wavelet_packed`):

  Per meshlet:
    3 * float32 offset                        = 12 B
    Per axis (DCT codes flattened, one stream per axis):
      2 B int16 min + 1 B uint8 bit-width     = 3 B / axis = 9 B total

So per-meshlet header overhead is identical to one wavelet level (12 + 9 =
21 B) regardless of block size.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct

from utils.wavelet import _stream_bits


def _delta_schedule(block_size: int, eps: float, schedule: str, ratio: float):
    """Return per-coefficient quantization steps delta_k of length block_size,
    saturating the L2 budget sqrt(sum delta_k^2) = 2 * eps."""
    B = block_size
    if schedule == "uniform" or ratio == 1.0:
        delta = 2.0 * eps / np.sqrt(B)
        return np.full(B, delta, dtype=np.float64)
    weights = ratio ** (np.arange(B) / B)  # increasing with frequency index
    # delta_0 * sqrt(sum w_k^2) = 2 eps
    delta_0 = 2.0 * eps / np.sqrt((weights ** 2).sum())
    return delta_0 * weights


def quantize_interior_dct2d_packed(positions, per_coord_err, block_size=4):
    """2D DCT-II per (B, 3) block - decorrelates both along the sorted-index
    axis (length B) AND across the 3 spatial axes (length 3) jointly.

    Error bound: orthonormal 2D DCT-II preserves L2 norm in both directions,
    so for uniform delta:
        ||x_recon - x||_inf <= ||c_recon - c||_F <= (delta/2) * sqrt(B * 3)
    Setting that <= eps gives delta = 2 * eps / sqrt(B * 3).

    Packs per coefficient position (3 * B bit-widths per meshlet) so HF
    positions can spend few bits when energy concentrates in low-freq cells.

    Returns (recon, total_bits, meta).
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        return positions.copy(), 0, {"block_size": block_size, "kind": "2d"}

    B = block_size
    delta = 2.0 * per_coord_err / np.sqrt(B * 3)
    n_padded = ((n + B - 1) // B) * B
    n_blocks = n_padded // B

    padded = np.zeros((n_padded, 3), dtype=np.float64)
    offsets = np.zeros(3, dtype=np.float64)
    for d in range(3):
        offsets[d] = float(positions[:, d].min())
        padded[:n, d] = positions[:, d] - offsets[d]
    blocks = padded.reshape(n_blocks, B, 3)
    coefs = dct(dct(blocks, type=2, norm="ortho", axis=1),
                type=2, norm="ortho", axis=2)
    codes = np.round(coefs / delta).astype(np.int64)        # (n_blocks, B, 3)
    coefs_recon = codes.astype(np.float64) * delta
    rec = idct(idct(coefs_recon, type=2, norm="ortho", axis=2),
               type=2, norm="ortho", axis=1)
    rec_flat = rec.reshape(n_padded, 3)[:n] + offsets[None, :]

    # Pack: 3 * float32 offsets + per-coefficient-position bit-widths.
    # Per-position stream length = n_blocks. 3 * B = 12 streams (B=4).
    total_bits = 3 * 32
    for i in range(B):
        for j in range(3):
            stream = codes[:, i, j]
            mn = int(stream.min())
            rng = int(stream.max() - mn)
            b = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
            body = _stream_bits(stream - mn, b)
            meta = 16 + 8  # int16 min + uint8 width
            total_bits += body + meta

    return rec_flat, total_bits, {"block_size": block_size, "kind": "2d",
                                  "n_padded": n_padded}


def quantize_interior_dct_packed(positions, per_coord_err, block_size=16,
                                 schedule="uniform", ratio=2.0):
    """Block-DCT-II interior encoder with packed per-axis metadata.

    Returns (recon, total_bits, meta).
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        return positions.copy(), 0, {"block_size": block_size,
                                     "schedule": schedule, "ratio": ratio}

    B = block_size
    deltas = _delta_schedule(B, per_coord_err, schedule, ratio)
    n_padded = ((n + B - 1) // B) * B
    n_blocks = n_padded // B

    recon = np.empty_like(positions)
    per_axis_codes = []  # list of (n_padded,) int64 arrays
    for d in range(3):
        offset = float(positions[:, d].min())
        shifted = positions[:, d] - offset
        padded = np.zeros(n_padded, dtype=np.float64)
        padded[:n] = shifted

        blocks = padded.reshape(n_blocks, B)
        coefs = dct(blocks, type=2, norm="ortho", axis=1)
        codes = np.round(coefs / deltas[None, :]).astype(np.int64)  # (B,B)
        coefs_recon = codes.astype(np.float64) * deltas[None, :]
        blocks_recon = idct(coefs_recon, type=2, norm="ortho", axis=1)
        recon[:, d] = blocks_recon.flatten()[:n] + offset

        per_axis_codes.append(codes.flatten())

    # Bit accounting (same packing pattern as wavelet base level):
    #   3 * float32 offset = 12 B = 96 bits
    #   per axis: int16 min + uint8 bit-width + body
    total_bits = 3 * 32
    for codes in per_axis_codes:
        mn = int(codes.min())
        rng = int(codes.max() - mn)
        b = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
        shifted_codes = codes - mn
        body_bits = _stream_bits(shifted_codes, b)
        meta_bits = 16 + 8  # int16 min + uint8 bit-width
        total_bits += body_bits + meta_bits

    return recon, total_bits, {"block_size": block_size,
                               "schedule": schedule, "ratio": ratio,
                               "n_padded": n_padded}
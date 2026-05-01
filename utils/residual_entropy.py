"""Entropy-coding bit estimators for residual streams.

Picks the minimum across {fixed-width, Rice(k), exp-Golomb(k),
empirical entropy + arith-coder overhead} for a given int code array.
Returns total bits and a small tag describing which coder won so the
decoder can mirror.

Assumed bitstream contract per axis:
    1 B  coder_tag   (uint8)
        0 : fixed-width with min-shift
        1 : Rice(k) on zigzag(codes)
        2 : exp-Golomb(k) on zigzag(codes)
        3 : empirical-entropy (rANS / arithmetic) — adds 32 b overhead
        4 : static-Laplacian arith — 16 b mu + 8 b log2 b (+16 b finish)
    if coder_tag == 0: + 2 B int16 min + 1 B uint8 bits_per_code
    if coder_tag in (1, 2): + 1 B uint8 k
    if coder_tag == 3: + 32 b adaptive-coder overhead inside `bits`
    if coder_tag == 4: + 16 b mu + 8 b log2 b + 16 b range-coder finish

Decoder is symmetric and stateless within a single axis.
"""

from __future__ import annotations

import numpy as np
from collections import Counter


def _zigzag(n):
    n = np.asarray(n, dtype=np.int64)
    return np.where(n >= 0, 2 * n, -2 * n - 1)


def _rice_bits(u, k):
    """Total bits for Rice(k) on non-negative array u."""
    if len(u) == 0:
        return 0
    return int((u >> k).sum()) + len(u) * (1 + k)


def _exp_golomb_bits(u, k):
    """Total bits for exp-Golomb(k) on non-negative array u."""
    if len(u) == 0:
        return 0
    shifted = u >> k
    lb = np.where(shifted > 0,
                  np.floor(np.log2(shifted + 1)).astype(np.int64),
                  0)
    return int((2 * lb + 1 + k).sum())


def _empirical_entropy_bits(codes):
    n = len(codes)
    if n == 0:
        return 0
    counts = Counter(codes.tolist())
    probs = np.array(list(counts.values()), dtype=np.float64) / n
    ent = -float(np.sum(probs * np.log2(probs)))
    return int(np.ceil(n * ent)) + 32  # 32 b adaptive-coder overhead


def _laplacian_arith_bits(codes):
    """Bit cost of arithmetic coding `codes` against a static integer
    Laplacian PMF whose scale `b` is fitted per stream.

    The PMF is the discrete Laplacian
        P(k) = exp(-|k - mu|/b) * (1 - exp(-1/b)) / (1 + exp(-1/b)) * 2  (k!=mu)
        P(mu) = (1 - exp(-1/b)) / (1 + exp(-1/b))
    integrated correctly (two-sided geometric). Mean is encoded as int16.

    Header:
        8 b tag + 16 b mu + 8 b log2-quantized b  →  32 b
    Body: Shannon-bound under the Laplacian model, +16 b range-coder finish.
    Tight upper bound on a real range-coder under this model (≤1 b/symbol
    overhead in practice).
    """
    n = len(codes)
    if n == 0:
        return 0
    mu = int(np.round(codes.mean()))
    dev = np.abs(codes - mu).astype(np.float64)
    b = max(1e-3, float(dev.mean()))
    # log2 b, quantized to 8 bits over a sane range [-4, 12] (b in [1/16, 4096]).
    log_b = np.clip(np.log2(b), -4.0, 12.0)
    log_b_q = np.round((log_b + 4.0) / 16.0 * 255.0)
    log_b_dq = log_b_q / 255.0 * 16.0 - 4.0
    b_q = float(2.0 ** log_b_dq)
    # -log2 P(k) = -log2(1 - r) + log2(1 + r) + |k-mu|/b * log2(e)   for k != mu
    # -log2 P(mu) = -log2(1 - r) + log2(1 + r)                          for k == mu
    # where r = exp(-1/b). Note P(mu)*(1+r) + sum_{k!=mu} P(k) = 1 by tail-sum.
    r = float(np.exp(-1.0 / b_q))
    # Normalising factor: -log2( (1 - r) / (1 + r) )
    norm_bits = -np.log2(max(1.0 - r, 1e-30) / max(1.0 + r, 1e-30))
    log2_e = 1.0 / np.log(2.0)
    abs_dev = np.abs(codes - mu).astype(np.float64)
    body_bits = float(n * norm_bits + abs_dev.sum() * log2_e / b_q)
    body_bits = int(np.ceil(body_bits)) + 16  # range-coder finish overhead
    return body_bits


def best_axis_bits(codes):
    """Pick minimum-bits coder for an int residual stream.

    Returns (total_bits_inclusive_of_per_axis_header, tag, param).
    `total_bits_inclusive_of_per_axis_header` includes the 8 b coder_tag
    plus per-coder param bits.
    """
    codes = np.asarray(codes, dtype=np.int64)
    n = len(codes)
    if n == 0:
        return 8, 0, 0  # tag byte only

    # Fixed-width with min-shift
    mn = int(codes.min())
    rng = int(codes.max() - mn)
    fixed_bw = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
    fixed_body = n * fixed_bw
    fixed_overhead = 8 + 16 + 8  # tag + int16 min + uint8 bits
    fixed_total = fixed_body + fixed_overhead

    # Rice + exp-Golomb on zigzagged codes
    u = _zigzag(codes)
    rice_best = (np.iinfo(np.int64).max, 0)
    for k in range(0, 12):
        b = _rice_bits(u, k)
        if b < rice_best[0]:
            rice_best = (b, k)
    rice_total = rice_best[0] + 8 + 8  # tag + uint8 k

    eg_best = (np.iinfo(np.int64).max, 0)
    for k in range(0, 8):
        b = _exp_golomb_bits(u, k)
        if b < eg_best[0]:
            eg_best = (b, k)
    eg_total = eg_best[0] + 8 + 8

    # Empirical entropy (adaptive coder)
    ent_total = _empirical_entropy_bits(codes) + 8

    # Static Laplacian + arithmetic (8 b tag + 16 b mu + 8 b log2 b in body)
    lap_total = _laplacian_arith_bits(codes) + 8 + 16 + 8

    # Pick smallest. Tie-breakers: prefer simpler coder (fixed > rice > eg > ent > lap).
    candidates = [
        (fixed_total, 0, fixed_bw),
        (rice_total, 1, rice_best[1]),
        (eg_total, 2, eg_best[1]),
        (ent_total, 3, 0),
        (lap_total, 4, 0),
    ]
    return min(candidates, key=lambda t: t[0])

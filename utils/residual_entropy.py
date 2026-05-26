"""Entropy-coding bit estimators for residual streams.

Picks the minimum across {fixed-width, Rice(k), exp-Golomb(k)} for a
given int code array. Returns total bits and a small tag describing
which coder won so the decoder can mirror.

Earlier revisions included two unbuilt-coder estimators (tag 3 =
empirical-entropy / rANS, tag 4 = static-Laplacian arithmetic). Plan 13
ablation (2026-05-12) measured 0/5061 axes pick Laplacian on Monkey and
≤0.01 BPV gain from the empirical-entropy estimator once a realistic
PMF header is charged. Both removed: bitstream now only carries coders
the encoder can actually emit. See memory/project_plan13_dead.md.

Bitstream contract per axis:
    1 B  coder_tag   (uint8)
        0 : fixed-width with min-shift
        1 : Rice(k) on zigzag(codes)
        2 : exp-Golomb(k) on zigzag(codes)
    if coder_tag == 0: + 2 B int16 min + 1 B uint8 bits_per_code
    if coder_tag in (1, 2): + 1 B uint8 k

Decoder is symmetric and stateless within a single axis.
"""

from __future__ import annotations

import os
import numpy as np

# Ablation hook: counts how many axes each coder wins. Reset by clearing.
TAG_COUNTS = {0: 0, 1: 0, 2: 0}


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

    # Tie-breakers: prefer simpler coder (fixed > rice > eg).
    candidates = [
        (fixed_total, 0, fixed_bw),
        (rice_total, 1, rice_best[1]),
        (eg_total, 2, eg_best[1]),
    ]
    # Ablation: drop coders listed in RESIDUAL_DISABLE_TAGS (comma-sep).
    disable_env = os.environ.get("RESIDUAL_DISABLE_TAGS", "")
    if disable_env:
        disabled = {int(t) for t in disable_env.split(",") if t.strip()}
        candidates = [c for c in candidates if c[1] not in disabled]
    best = min(candidates, key=lambda t: t[0])
    TAG_COUNTS[best[1]] = TAG_COUNTS.get(best[1], 0) + 1
    return best

"""Numba kernels for AMD-GTS strip generation.

JIT version of `generate_strips_v2_seeded` and `generate_strips_multiseed`.
Bit-exact with the Python implementations on identical inputs.

`local_adj` (dict from Python side) is converted to CSR before entering
the kernel:
  adj_off (n+1,) int32, adj_idx (sum_deg,) int32
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def _best_neighbor_nb(tri, adj_off, adj_idx, processed, cur_val):
    """Pick neighbor of `tri` with lowest (cur_val, look-ahead_sum). -1 if none."""
    best = -1
    best_key0 = 0
    best_key1 = 0
    for k in range(adj_off[tri], adj_off[tri + 1]):
        nb = adj_idx[k]
        if processed[nb]:
            continue
        la = 0
        for k2 in range(adj_off[nb], adj_off[nb + 1]):
            nb2 = adj_idx[k2]
            if nb2 != tri and not processed[nb2]:
                la += cur_val[nb2]
        if best < 0:
            best = nb
            best_key0 = cur_val[nb]
            best_key1 = la
        else:
            k0 = cur_val[nb]
            if k0 < best_key0 or (k0 == best_key0 and la < best_key1):
                best = nb
                best_key0 = k0
                best_key1 = la
    return best


@njit(cache=True)
def _process_nb(idx, adj_off, adj_idx, processed, cur_val):
    processed[idx] = True
    for k in range(adj_off[idx], adj_off[idx + 1]):
        nb = adj_idx[k]
        if not processed[nb]:
            cur_val[nb] -= 1


@njit(cache=True)
def _strips_seeded_kernel(n_f, adj_off, adj_idx, deg, seed_first,
                          strip_starts, strip_lens, strip_tris):
    """Bidirectional strip cover from a forced first seed. Writes results
    into pre-sized output buffers. Returns (n_strips, total_tris).
    """
    processed = np.zeros(n_f, dtype=np.bool_)
    cur_val = deg.copy()
    scratch = np.empty(n_f, dtype=np.int32)

    n_strips = 0
    total = 0
    first = True
    while True:
        if first:
            seed = seed_first
            first = False
        else:
            # Find min-cur-val unprocessed
            best = -1
            best_v = 0
            for i in range(n_f):
                if processed[i]:
                    continue
                if best < 0 or cur_val[i] < best_v:
                    best = i
                    best_v = cur_val[i]
            if best < 0:
                break
            seed = best
        if processed[seed]:
            continue
        # Grow strip from seed: forward then backward.
        # Forward: append to a forward list.
        n_fwd = 0
        scratch[0] = seed
        n_fwd = 1
        _process_nb(seed, adj_off, adj_idx, processed, cur_val)
        cur = seed
        while True:
            nb = _best_neighbor_nb(cur, adj_off, adj_idx, processed, cur_val)
            if nb < 0:
                break
            scratch[n_fwd] = nb
            n_fwd += 1
            _process_nb(nb, adj_off, adj_idx, processed, cur_val)
            cur = nb
        # Backward: prepend by writing into a reversed buffer.
        cur = seed
        # We'll collect backward in another temp, then reverse + prepend.
        # Use a second array region after n_fwd.
        n_bwd = 0
        while True:
            nb = _best_neighbor_nb(cur, adj_off, adj_idx, processed, cur_val)
            if nb < 0:
                break
            # Append to backward (in temp slot n_fwd + n_bwd)
            scratch[n_fwd + n_bwd] = nb
            n_bwd += 1
            _process_nb(nb, adj_off, adj_idx, processed, cur_val)
            cur = nb
        # Final strip: reverse(backward) + forward
        strip_starts[n_strips] = total
        # write reversed backward first
        for j in range(n_bwd - 1, -1, -1):
            strip_tris[total] = scratch[n_fwd + j]
            total += 1
        for j in range(n_fwd):
            strip_tris[total] = scratch[j]
            total += 1
        strip_lens[n_strips] = n_fwd + n_bwd
        n_strips += 1
    return n_strips, total


def generate_strips_v2_seeded_nb(tris_local, adj_off, adj_idx, deg, seed_first):
    """Numba-driven version of `generate_strips_v2_seeded`. Returns
    list[list[int]] matching the Python implementation.
    """
    n_f = len(tris_local)
    if n_f == 0:
        return []
    strip_starts = np.empty(n_f, dtype=np.int32)
    strip_lens = np.empty(n_f, dtype=np.int32)
    strip_tris = np.empty(n_f, dtype=np.int32)
    n_strips, _ = _strips_seeded_kernel(
        n_f, adj_off, adj_idx, deg, int(seed_first),
        strip_starts, strip_lens, strip_tris)
    out = []
    for i in range(n_strips):
        s = int(strip_starts[i])
        L = int(strip_lens[i])
        out.append([int(x) for x in strip_tris[s:s + L]])
    return out


def generate_strips_multiseed_csr_nb(tris_local, adj_off, adj_idx,
                                     n_seeds: int = 8):
    """Multiseed strip cover taking pre-built CSR adjacency. Avoids the
    dict→CSR round-trip that ``generate_strips_multiseed_nb`` does internally.
    """
    n_f = len(tris_local)
    if n_f == 0:
        return []
    deg = (adj_off[1:] - adj_off[:-1]).astype(np.int32)
    val = deg
    seeds: set[int] = set()
    seeds.add(int(np.argmin(val)))
    seeds.add(int(np.argmax(val)))
    if n_f > 4:
        rng = np.random.default_rng(0)
        for s in rng.choice(n_f, size=min(n_seeds - 2, n_f - 2), replace=False):
            seeds.add(int(s))
    best = None
    best_n = None
    for s in seeds:
        strips = generate_strips_v2_seeded_nb(tris_local, adj_off, adj_idx,
                                              deg, s)
        if best_n is None or len(strips) < best_n:
            best_n = len(strips)
            best = strips
    return best


def generate_strips_multiseed_nb(tris_local, local_adj, n_seeds: int = 8):
    """Multiseed strip cover — bit-exact with `generate_strips_multiseed`."""
    n_f = len(tris_local)
    if n_f == 0:
        return []
    # Build CSR adjacency
    deg = np.array([len(local_adj[i]) for i in range(n_f)], dtype=np.int32)
    adj_off = np.zeros(n_f + 1, dtype=np.int32)
    adj_off[1:] = np.cumsum(deg)
    adj_idx = np.empty(int(adj_off[-1]), dtype=np.int32)
    for i in range(n_f):
        adj_idx[adj_off[i]:adj_off[i + 1]] = local_adj[i]

    # Seed candidates: argmin, argmax, plus seeded random sample
    val = deg
    seeds: set[int] = set()
    seeds.add(int(np.argmin(val)))
    seeds.add(int(np.argmax(val)))
    if n_f > 4:
        rng = np.random.default_rng(0)
        for s in rng.choice(n_f, size=min(n_seeds - 2, n_f - 2), replace=False):
            seeds.add(int(s))

    best = None
    best_n = None
    for s in seeds:
        strips = generate_strips_v2_seeded_nb(tris_local, adj_off, adj_idx, deg, s)
        if best_n is None or len(strips) < best_n:
            best_n = len(strips)
            best = strips
    return best
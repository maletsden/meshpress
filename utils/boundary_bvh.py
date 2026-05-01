"""Spatial-locality compression for the global boundary table and per-
meshlet boundary refs (plan 12).

Two techniques:

A. **Morton-sorted delta table.** Sort boundary verts by 3D Morton code
   on the global int grid, then store per-axis deltas (zigzag-rice).
   Spatially close verts get adjacent indices → small deltas.

B. **Delta refs.** Within each meshlet, sort the boundary refs into the
   sorted table, then store first ref + zigzag-rice deltas. Locality of
   meshlet boundary verts → small deltas.

Both keep the table layout intact (lossless permutation + a remap so the
caller can rewrite ref values). Decoder mirrors deterministically.
"""

from __future__ import annotations

import numpy as np

from utils.boundary_split import morton3_codes


def _zigzag(n):
    n = np.asarray(n, dtype=np.int64)
    return np.where(n >= 0, 2 * n, -2 * n - 1)


def _rice_best_bits(u, k_max=12):
    if len(u) == 0:
        return 0, 0
    best = (np.iinfo(np.int64).max, 0)
    for k in range(0, k_max):
        b = int((u >> k).sum()) + len(u) * (1 + k)
        if b < best[0]:
            best = (b, k)
    return best  # (bits, k)


def morton_permute_boundary(boundary_list, global_codes):
    """Return a (new_boundary_list, old_to_new) where new_boundary_list is
    permuted so its codes are Morton-sorted. Stable on ties."""
    if not boundary_list:
        return list(boundary_list), {}
    arr = np.asarray(boundary_list, dtype=np.int64)
    codes = global_codes[arr]
    m = morton3_codes(codes)
    order = np.argsort(m, kind='stable')
    new_list = [int(arr[i]) for i in order]
    old_to_new = {gv: i for i, gv in enumerate(new_list)}
    return new_list, old_to_new


def delta_boundary_table_bits(boundary_codes_sorted):
    """Bit cost of axis-delta + rice on a Morton-sorted boundary table.

    Args:
        boundary_codes_sorted : (n, 3) int64 codes, already permuted in
            the order they will be transmitted.

    Returns total bits (header + body). Header: 3 × (16 b first code +
    8 b rice k) = 72 b. Body: rice-coded zigzag(diff) per axis. For the
    very first sample we emit only the first code (no delta).
    """
    a = np.asarray(boundary_codes_sorted, dtype=np.int64)
    n = a.shape[0]
    if n == 0:
        return 0
    head_bits = 3 * (32 + 8)  # int32 first code per axis + 8 b rice k
    if n == 1:
        return head_bits
    body = 0
    for ax in range(3):
        diffs = a[1:, ax] - a[:-1, ax]
        u = _zigzag(diffs)
        body_ax, _ = _rice_best_bits(u)
        body += body_ax
    return head_bits + body


def delta_refs_bits(refs):
    """Bit cost of {first ref, zigzag-rice deltas} for one meshlet's
    boundary refs (already int indices into the global table).

    Refs are sorted ascending before delta-coding; the meshlet stores the
    sort permutation implicitly by re-emitting in sorted order, since AMD
    triangle indices reference the GLOBAL boundary index (via the local
    boundary slot), and the remap is symmetric encoder/decoder.
    """
    if len(refs) == 0:
        return 0
    refs_sorted = np.sort(np.asarray(refs, dtype=np.int64))
    head = 16 + 8  # first ref + rice k
    if len(refs_sorted) == 1:
        return head
    diffs = refs_sorted[1:] - refs_sorted[:-1]
    u = _zigzag(diffs.astype(np.int64))
    body, _ = _rice_best_bits(u)
    return head + body

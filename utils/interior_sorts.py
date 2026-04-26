"""
Alternative interior-vertex orderings for crack-free meshlet split encoders.

The interior order is a free design dimension: it costs zero header bits as
long as the encoder emits verts in the chosen order and the connectivity
encoding is built against the same `local_to_global` layout. Different
orderings produce different per-axis 1D streams; smoother streams compress
better under any subsequent transform (Haar / DCT / neural).

Currently supported variants:

    morton      (in utils/boundary_split.py: sort_by_morton)
                Z-order on the global integer grid. Existing baseline.

    eb          The default fallthrough order returned by
                edgebreaker_vertex_order (no extra sort). Functions as the
                strip-traversal-style ordering.

    hilbert     3D Hilbert curve. Same locality bound as Morton but no
                octant jumps; expected to produce smoother streams.

    pca         Project onto the first principal axis of the meshlet's
                interior float positions, sort by projection. Adapts to
                elongated meshlets.

    greedy_nn   Nearest-neighbor walk: seed = lowest-Morton vert, then
                repeatedly hop to nearest unvisited. O(n^2) per meshlet
                (n <= 256, so cheap in absolute terms).

Each sort function takes the interior-vert global-index list plus whatever
auxiliary data it needs, and returns a permutation of the input list.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------
# Hilbert curve (3D)
# -----------------------------------------------------------------

def _hilbert_index_3d(x, y, z, order):
    """Compute 3D Hilbert-curve index for a single (x, y, z) integer point.

    `order` is the number of bits per axis. Result fits in 3*order bits.
    Standard rotation-based algorithm from Hamilton-Rao (and Wikipedia).
    """
    x = int(x); y = int(y); z = int(z)
    rx = ry = rz = 0
    d = 0
    s = 1 << (order - 1)
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        rz = 1 if (z & s) else 0
        d += s * s * s * ((4 * rx) ^ (2 * (rx ^ ry)) ^ (rx ^ ry ^ rz))
        # Rotate octant
        if rz == 0:
            if ry == 1:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, z = z, x
            else:
                x, z = z, x
        else:
            if ry == 0:
                if rx == 1:
                    x, z = s - 1 - x, s - 1 - z
                y, z = z, y
            else:
                if rx == 1:
                    y, z = s - 1 - y, s - 1 - z
                y, z = z, y
        s >>= 1
    return d


def hilbert3_codes(int_codes_xyz, max_bits=21):
    """Vectorized-ish 3D Hilbert codes for an (n, 3) int array.

    Caller passes integer coords on the global grid (same input as
    morton3_codes). Computes Hilbert order = max bits per axis from data.
    """
    arr = np.asarray(int_codes_xyz, dtype=np.int64)
    if len(arr) == 0:
        return np.zeros(0, dtype=np.int64)
    mx = int(arr.max())
    if mx <= 0:
        return np.zeros(len(arr), dtype=np.int64)
    order = min(max_bits, max(1, int(np.ceil(np.log2(mx + 1)))))
    out = np.empty(len(arr), dtype=np.int64)
    for i in range(len(arr)):
        out[i] = _hilbert_index_3d(arr[i, 0], arr[i, 1], arr[i, 2], order)
    return out


def sort_by_hilbert(global_indices, global_codes):
    """Permute `global_indices` into Hilbert order using their global codes."""
    if len(global_indices) <= 1:
        return list(global_indices)
    codes = global_codes[global_indices]
    h = hilbert3_codes(codes)
    order = np.argsort(h, kind="stable")
    return [global_indices[i] for i in order]


# -----------------------------------------------------------------
# PCA-axis projection
# -----------------------------------------------------------------

def sort_by_pca(global_indices, vert_pos_float):
    """Sort by projection onto the meshlet's first PCA axis.

    `vert_pos_float` is the full per-mesh float vertex array (n_verts, 3);
    we only look at the rows in `global_indices`.
    """
    n = len(global_indices)
    if n <= 1:
        return list(global_indices)
    pts = vert_pos_float[global_indices].astype(np.float64)
    centered = pts - pts.mean(axis=0)
    # Single-axis PCA via covariance eigvec; for n=2 the matrix is degenerate
    # but eigh still returns something usable.
    cov = centered.T @ centered
    try:
        _, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, -1]  # largest eigenvalue last
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0, 0.0])
    proj = centered @ axis
    order = np.argsort(proj, kind="stable")
    return [global_indices[i] for i in order]


# -----------------------------------------------------------------
# Greedy nearest-neighbor walk
# -----------------------------------------------------------------

def sort_by_greedy_nn(global_indices, vert_pos_float, seed_idx=None):
    """Greedy NN walk in 3D space.

    Seed defaults to the first global index in `global_indices` (callers can
    pass `seed_idx` explicitly to use e.g. the lowest-Morton vert). Then at
    each step, hop to the nearest unvisited vert. O(n^2) but n <= 256.
    """
    n = len(global_indices)
    if n <= 1:
        return list(global_indices)
    pts = vert_pos_float[global_indices].astype(np.float64)
    if seed_idx is None:
        start = 0
    else:
        try:
            start = global_indices.index(seed_idx)
        except ValueError:
            start = 0
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    order[0] = start
    visited[start] = True
    cur = start
    for k in range(1, n):
        diff = pts - pts[cur]
        d2 = (diff * diff).sum(axis=1)
        d2[visited] = np.inf
        nxt = int(np.argmin(d2))
        order[k] = nxt
        visited[nxt] = True
        cur = nxt
    return [global_indices[i] for i in order]


# -----------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------

def sort_interior(variant, global_indices, *, global_codes=None,
                  vert_pos_float=None):
    """Single entry point for sort variants on the INTERIOR vertex list.

    Required kwargs depend on variant:
        morton    : global_codes
        hilbert   : global_codes
        pca       : vert_pos_float
        greedy_nn : vert_pos_float
        eb        : (none — returns input unchanged)

    Unknown variants raise ValueError.
    """
    if len(global_indices) <= 1:
        return list(global_indices)
    if variant == "eb":
        return list(global_indices)
    if variant == "morton":
        from utils.boundary_split import sort_by_morton
        if global_codes is None:
            raise ValueError("morton sort requires global_codes")
        return sort_by_morton(global_indices, global_codes)
    if variant == "hilbert":
        if global_codes is None:
            raise ValueError("hilbert sort requires global_codes")
        return sort_by_hilbert(global_indices, global_codes)
    if variant == "pca":
        if vert_pos_float is None:
            raise ValueError("pca sort requires vert_pos_float")
        return sort_by_pca(global_indices, vert_pos_float)
    if variant == "greedy_nn":
        if vert_pos_float is None:
            raise ValueError("greedy_nn sort requires vert_pos_float")
        return sort_by_greedy_nn(global_indices, vert_pos_float)
    raise ValueError(f"unknown sort variant: {variant}")
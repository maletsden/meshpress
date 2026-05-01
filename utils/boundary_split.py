"""
Boundary/interior vertex split for crack-free meshlet compression.

Data layout contract:

    Global header:
        g_min[3]       (3 x f32, 12B)    global AABB min
        g_range[3]     (3 x f32, 12B)    global AABB range
        g_bits[3]      (3 x u8,  3B)     bits per axis for global grid
        n_meshlets     (u32, 4B)
        n_boundary     (u32, 4B)         size of global boundary table
        ref_bits       (u8,  1B)         ceil(log2(n_boundary+1))

    Global boundary table (once):
        for b in 0..n_boundary:
            code[3]     (3 x g_bits, packed)   integer code on global grid

    Per meshlet:
        n_boundary_local (u8, 1B)        # of verts pulled from global table
        n_interior       (u8, 1B)        # of interior verts
        boundary_refs    (n_boundary_local * ref_bits bits)
        interior_offset  (3 x u32, 12B)  meshlet-local grid origin (in global-code units)
        interior_bits[3] (3 x u8,  3B)   per-axis local bits for interior
        interior_codes   (n_interior * sum(interior_bits) bits)
        amd_connectivity (AMD packed, 3 * micro_bits per triangle)

Vertex ordering within meshlet: [boundary_refs... | interior...]
    Triangles' AMD local indices therefore split branch-free into
        idx < n_boundary_local  -> load from global boundary table
        idx >= n_boundary_local -> decode interior from local codes

This keeps AMD GTS/packed connectivity oblivious to the split and lets a GPU
decoder overlap a small global-memory boundary load with the connectivity
phase before falling into the interior transform.

Boundary verts stay on the global integer grid (bitwise identical across
meshlets -> crack-free by construction). Interior verts are free to use any
representation: integer codes (this file), float wavelets, neural latents, etc.
"""

from __future__ import annotations

import numpy as np


def identify_boundary_verts(meshlets, tris_np):
    """Global vertex indices shared by >= 2 meshlets."""
    vert_count = {}
    for ml_tris in meshlets:
        vs = set()
        for ti in ml_tris:
            for j in range(3):
                vs.add(int(tris_np[ti, j]))
        for v in vs:
            vert_count[v] = vert_count.get(v, 0) + 1
    return set(v for v, c in vert_count.items() if c >= 2)


def build_boundary_table(boundary_set, global_codes):
    """Sort boundary verts and return (boundary_list, gv_to_ref).
    boundary_list[i] is the global vertex idx at boundary-table position i.
    gv_to_ref[global_idx] = boundary table index."""
    boundary_list = sorted(boundary_set)
    gv_to_ref = {gv: i for i, gv in enumerate(boundary_list)}
    # Boundary codes on the global grid (int64, per-axis code)
    boundary_codes = global_codes[boundary_list] if boundary_list else np.zeros((0, 3), dtype=np.int64)
    return boundary_list, gv_to_ref, boundary_codes


def split_meshlet_verts(vert_order, boundary_set):
    """Partition a meshlet's vertex-order list into boundary-first layout.

    Args:
        vert_order: list of global vertex indices in EdgeBreaker/BFS order
        boundary_set: global indices of boundary verts

    Returns:
        local_to_global: list of global indices in [boundary | interior] order
        boundary_local:  list of global indices for the boundary portion
        interior_local:  list of global indices for the interior portion
        remap:           dict mapping original-order local index -> new local index
                         (so callers can relabel AMD triangle indices)
    """
    boundary_local = []
    interior_local = []
    for v in vert_order:
        if v in boundary_set:
            boundary_local.append(v)
        else:
            interior_local.append(v)

    local_to_global = boundary_local + interior_local

    # Original local idx = position in vert_order. New local idx = position in
    # local_to_global. Build the permutation.
    new_pos = {g: i for i, g in enumerate(local_to_global)}
    remap = {i: new_pos[g] for i, g in enumerate(vert_order)}

    return local_to_global, boundary_local, interior_local, remap


def boundary_table_bits(n_boundary, g_bits):
    """Total bits for the global boundary table (codes only, no header).

    n_boundary * sum(g_bits)  — each boundary vertex stores one code per axis
    at the global bit width.
    """
    if n_boundary == 0:
        return 0
    return int(n_boundary * int(g_bits.sum()))


def ref_bits_for(n_boundary):
    """Bit width needed to index the global boundary table."""
    if n_boundary <= 0:
        return 1
    return max(1, int(np.ceil(np.log2(n_boundary + 1))))


def quantize_interior_local(interior_global_codes):
    """Local-grid encode for interior codes (integer baseline).

    Input: (n_interior, 3) int64 global-grid codes.
    Output: (local_offset[3], local_bits[3], local_codes[n_interior, 3], total_bits)

    Interior verts have no cross-meshlet consistency requirement, so we can
    shift to a local origin and use fewer bits per axis. Encoding is
    lossless w.r.t. the global codes (integer delta only).
    """
    if len(interior_global_codes) == 0:
        return (
            np.zeros(3, dtype=np.int64),
            np.ones(3, dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            0,
        )
    local_offset = interior_global_codes.min(axis=0).astype(np.int64)
    local_range = (interior_global_codes.max(axis=0) - local_offset).astype(np.int64)
    local_bits = np.array(
        [max(1, int(np.ceil(np.log2(int(r) + 2)))) for r in local_range],
        dtype=np.int64,
    )
    local_codes = interior_global_codes - local_offset
    total_bits = int(len(interior_global_codes) * int(local_bits.sum()))
    return local_offset, local_bits, local_codes, total_bits


def _spread3_21(v):
    """Spread 21 low bits of int64 v so output has bits at positions 0,3,6,...,60.
    Works on numpy int64 arrays or Python ints."""
    v = v & 0x1FFFFF
    v = (v | (v << 32)) & 0x001F00000000FFFF
    v = (v | (v << 16)) & 0x001F0000FF0000FF
    v = (v | (v << 8))  & 0x100F00F00F00F00F
    v = (v | (v << 4))  & 0x10C30C30C30C30C3
    v = (v | (v << 2))  & 0x1249249249249249
    return v


def morton3_codes(int_codes_xyz):
    """Vectorized 3D Morton (Z-order) code for integer coords.

    Accepts (n, 3) int array on the global quantization grid (up to 21 bits
    per axis). Returns (n,) int64 Morton codes suitable for np.argsort.
    """
    arr = np.asarray(int_codes_xyz, dtype=np.int64)
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    return _spread3_21(x) | (_spread3_21(y) << 1) | (_spread3_21(z) << 2)


def sort_by_morton(global_indices, global_codes):
    """Return `global_indices` permuted so they are in Morton order
    according to their integer codes on the global grid."""
    if len(global_indices) <= 1:
        return list(global_indices)
    codes = global_codes[global_indices]
    m = morton3_codes(codes)
    order = np.argsort(m, kind="stable")
    return [global_indices[i] for i in order]


def gts_encode_meshlet(ml_tris, tris_np, tri_adj, local_to_global, variant="v1"):
    """Run AMD GTS encode with a caller-provided vertex order.

    variant:
        'v1' = original (flat 2-bit edge_code).
        'v2' = prefix-coded edge_code + dynamic-valence strip gen.
        'v3' = AMD L/R 1-bit + L/R-constrained strip gen + FIFO reuse.
    Returns (bits, stream).
    """
    from utils.amd_gts import gts_encode, gts_encode_v2, gts_encode_v3

    n_f = len(ml_tris)
    n_local = len(local_to_global)
    if n_f == 0:
        return 0, []

    g2l = {g: i for i, g in enumerate(local_to_global)}
    tris_local = np.zeros((n_f, 3), dtype=int)
    tri_map = {}
    for li, ti in enumerate(ml_tris):
        tri_map[ti] = li
        for j in range(3):
            tris_local[li, j] = g2l[int(tris_np[ti, j])]

    local_adj = [[] for _ in range(n_f)]
    for li, ti in enumerate(ml_tris):
        for nb in tri_adj[ti]:
            if nb in tri_map:
                local_adj[li].append(tri_map[nb])

    if variant == "v3":
        return gts_encode_v3(tris_local, local_adj, n_local)
    if variant == "v2":
        return gts_encode_v2(tris_local, local_adj, n_local)
    return gts_encode(tris_local, local_adj, n_local)


def verify_crack_free(meshlets, tris_np, global_codes, boundary_set):
    """Assert that every shared vertex resolves to the same global code
    regardless of which meshlet decodes it. Returns (n_cracks, n_shared)."""
    seen = {}
    n_cracks = 0
    n_shared = 0
    for ml_tris in meshlets:
        ml_verts = set()
        for ti in ml_tris:
            for j in range(3):
                ml_verts.add(int(tris_np[ti, j]))
        for gv in ml_verts:
            if gv not in boundary_set:
                continue
            code = tuple(int(x) for x in global_codes[gv])
            if gv in seen:
                n_shared += 1
                if seen[gv] != code:
                    n_cracks += 1
            else:
                seen[gv] = code
    return n_cracks, n_shared

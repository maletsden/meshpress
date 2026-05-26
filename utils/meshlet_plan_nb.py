"""Numba kernels for per-meshlet plan construction.

Replaces Python hot loops in `encoder.paradelta_codec._plan_meshlet`:
  - `_local_face_adj`        → local_face_adj_csr_nb
  - `meshlet_bfs`            → meshlet_bfs_nb
  - `edgebreaker_vertex_order` → eb_vertex_order_nb
  - `sort_by_greedy_nn`      → greedy_nn_order_nb

Inputs use flat ndarrays + CSR adjacency. No Python dicts or sets in the
hot path.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def build_global_adj_csr_nb(tris_np):
    """Build CSR triangle-triangle adjacency from a global tri array.
    Returns (adj_off int32[n+1], adj_idx int32[*])."""
    n = tris_np.shape[0]
    n_edges = 3 * n
    # Layout matches utils.meshlet_generator.build_adjacency:
    #   [0..n)   = edge (v0,v1) for each tri
    #   [n..2n)  = edge (v1,v2)
    #   [2n..3n) = edge (v0,v2)
    e_lo = np.empty(n_edges, dtype=np.int64)
    e_hi = np.empty(n_edges, dtype=np.int64)
    e_tri = np.empty(n_edges, dtype=np.int32)
    for t in range(n):
        a = np.int64(tris_np[t, 0])
        b = np.int64(tris_np[t, 1])
        c = np.int64(tris_np[t, 2])
        # edge (v0, v1) → block 0
        if a < b:
            e_lo[t] = a; e_hi[t] = b
        else:
            e_lo[t] = b; e_hi[t] = a
        e_tri[t] = t
        # edge (v1, v2) → block 1
        if b < c:
            e_lo[n + t] = b; e_hi[n + t] = c
        else:
            e_lo[n + t] = c; e_hi[n + t] = b
        e_tri[n + t] = t
        # edge (v0, v2) → block 2
        if a < c:
            e_lo[2*n + t] = a; e_hi[2*n + t] = c
        else:
            e_lo[2*n + t] = c; e_hi[2*n + t] = a
        e_tri[2*n + t] = t
    # Find max vert for combined key
    max_v = np.int64(0)
    for i in range(n_edges):
        if e_hi[i] > max_v:
            max_v = e_hi[i]
    mult = max_v + np.int64(1)
    # Composite key = key * n_edges + input_index — gives a stable sort
    # via plain argsort, matching Python's `np.argsort(kind='mergesort')`.
    keys = np.empty(n_edges, dtype=np.int64)
    comp = np.empty(n_edges, dtype=np.int64)
    ne_i64 = np.int64(n_edges)
    for i in range(n_edges):
        k = e_lo[i] * mult + e_hi[i]
        keys[i] = k
        comp[i] = k * ne_i64 + np.int64(i)
    order = np.argsort(comp)
    sk = np.empty(n_edges, dtype=np.int64)
    st = np.empty(n_edges, dtype=np.int32)
    for i in range(n_edges):
        sk[i] = keys[order[i]]
        st[i] = e_tri[order[i]]
    deg = np.zeros(n, dtype=np.int32)
    # Allocate worst-case 2n pairs (n_edges = 3n entries → up to 3n-1 same pairs).
    pair_a = np.empty(n_edges, dtype=np.int32)
    pair_b = np.empty(n_edges, dtype=np.int32)
    n_pairs = 0
    # Match `utils.meshlet_generator.build_adjacency`: emit a pair for EVERY
    # consecutive same-key position (`same[i] = sk[i]==sk[i+1]`).
    # For runs of 3+ same keys this produces (st[i],st[i+1]) AND
    # (st[i+1],st[i+2]) — replicates Python's non-manifold over-counting.
    for i in range(n_edges - 1):
        if sk[i] == sk[i+1]:
            ta = st[i]; tb = st[i+1]
            pair_a[n_pairs] = ta
            pair_b[n_pairs] = tb
            deg[ta] += 1
            deg[tb] += 1
            n_pairs += 1
    adj_off = np.zeros(n + 1, dtype=np.int32)
    for t in range(n):
        adj_off[t + 1] = adj_off[t] + deg[t]
    total = adj_off[n]
    adj_idx = np.empty(total, dtype=np.int32)
    cur = adj_off.copy()
    for p in range(n_pairs):
        a = pair_a[p]; b = pair_b[p]
        adj_idx[cur[a]] = b; cur[a] += 1
        adj_idx[cur[b]] = a; cur[b] += 1
    return adj_off, adj_idx


@njit(cache=True)
def local_face_adj_csr_nb(ml_tris_local):
    """CSR face-face dual adjacency for one meshlet (local IDs).

    Mirrors `encoder.paradelta_codec._local_face_adj`: iterates edges in
    Python dict-insertion order (tri-major × edge-major over (a,b)/(b,c)/(c,a)).
    Pair (first_tri, second_tri) emitted at the moment the second occurrence
    is seen, preserving the per-tri neighbour ordering of the dict-based
    reference implementation.
    """
    n_tris = ml_tris_local.shape[0]
    if n_tris == 0:
        return np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32)
    n_edges = 3 * n_tris

    # Two-pass over edges in dict-insertion order.
    # Pass 1: build edge list + count per edge via hash map.
    ht = 1
    while ht < n_edges * 4:
        ht *= 2
    hk = np.full(ht, np.int64(-1), dtype=np.int64)
    hv = np.empty(ht, dtype=np.int32)  # edge index in dedup list
    mask = ht - 1
    edge_first_tri = np.empty(n_edges, dtype=np.int32)
    edge_second_tri = np.full(n_edges, -1, dtype=np.int32)
    edge_count = np.zeros(n_edges, dtype=np.int32)
    edge_order = np.empty(n_edges, dtype=np.int32)  # discovery order
    n_unique = 0
    for t in range(n_tris):
        a = np.int64(ml_tris_local[t, 0])
        b = np.int64(ml_tris_local[t, 1])
        c = np.int64(ml_tris_local[t, 2])
        for ei in range(3):
            if ei == 0:
                u = a; v = b
            elif ei == 1:
                u = b; v = c
            else:
                u = c; v = a
            if u < v:
                lo = u; hi = v
            else:
                lo = v; hi = u
            key = (lo << 32) | hi
            h = (key * np.int64(2654435761)) & mask
            found_idx = -1
            while hk[h] != -1:
                if hk[h] == key:
                    found_idx = hv[h]
                    break
                h = (h + 1) & mask
            if found_idx < 0:
                hk[h] = key
                hv[h] = n_unique
                edge_first_tri[n_unique] = t
                edge_count[n_unique] = 1
                edge_order[n_unique] = n_unique
                n_unique += 1
            else:
                edge_count[found_idx] += 1
                if edge_count[found_idx] == 2:
                    edge_second_tri[found_idx] = t

    # Pass 2: emit pairs only for edges with exactly 2 incident tris,
    # in dict-insertion order. Matches Python `edge_to_tris.values()` walk.
    pair_a = np.empty(n_unique, dtype=np.int32)
    pair_b = np.empty(n_unique, dtype=np.int32)
    n_pairs = 0
    deg = np.zeros(n_tris, dtype=np.int32)
    for e in range(n_unique):
        if edge_count[e] == 2:
            ta = edge_first_tri[e]
            tb = edge_second_tri[e]
            pair_a[n_pairs] = ta
            pair_b[n_pairs] = tb
            deg[ta] += 1
            deg[tb] += 1
            n_pairs += 1

    adj_off = np.zeros(n_tris + 1, dtype=np.int32)
    for t in range(n_tris):
        adj_off[t + 1] = adj_off[t] + deg[t]
    total = adj_off[n_tris]
    adj_idx = np.empty(total, dtype=np.int32)
    cur = adj_off.copy()
    for p in range(n_pairs):
        a = pair_a[p]; b = pair_b[p]
        adj_idx[cur[a]] = b; cur[a] += 1
        adj_idx[cur[b]] = a; cur[b] += 1
    return adj_off, adj_idx


@njit(cache=True)
def eb_vertex_order_nb(meshlet_tris, tris_np, adj_off, adj_idx):
    """EdgeBreaker-style BFS vertex order over a meshlet.

    meshlet_tris : int32[n_tris_m] in the caller's seed-iteration order
                   (NOT sorted — order matters for BPV reproducibility)
    tris_np      : int32[N_global, 3]
    adj_off,
    adj_idx      : CSR global tri-tri adjacency
    Returns (vertex_order int32[*], n_root int32).
    """
    n_tris_m = meshlet_tris.shape[0]
    cap = 3 * n_tris_m + 8

    # Open-addressing hash set for tri-in-meshlet membership.
    ts = 1
    while ts < n_tris_m * 4 + 4:
        ts *= 2
    ml_keys = np.full(ts, -1, dtype=np.int32)
    ml_vals = np.empty(ts, dtype=np.int32)  # local index into meshlet_tris
    ml_mask = ts - 1
    for li in range(n_tris_m):
        gt = meshlet_tris[li]
        h = (gt * 2654435761) & ml_mask
        while ml_keys[h] != -1:
            h = (h + 1) & ml_mask
        ml_keys[h] = gt
        ml_vals[h] = li

    # Open-addressing hash set for "seen vertices".
    vs = 1
    while vs < cap * 2:
        vs *= 2
    seen_keys = np.full(vs, -1, dtype=np.int32)
    seen_mask = vs - 1

    vertex_order = np.empty(cap, dtype=np.int32)
    n_vert = 0

    visited_tri = np.zeros(n_tris_m, dtype=np.bool_)
    queue = np.empty(n_tris_m, dtype=np.int32)
    n_root = 0

    for seed_li in range(n_tris_m):
        if visited_tri[seed_li]:
            continue
        seed_g = meshlet_tris[seed_li]
        visited_tri[seed_li] = True
        q_head = 0
        q_tail = 0
        queue[q_tail] = seed_g
        q_tail += 1
        n_root += 1
        # Root tri verts
        for j in range(3):
            v = np.int32(tris_np[seed_g, j])
            h = (v * 2654435761) & seen_mask
            found = False
            while seen_keys[h] != -1:
                if seen_keys[h] == v:
                    found = True
                    break
                h = (h + 1) & seen_mask
            if not found:
                seen_keys[h] = v
                vertex_order[n_vert] = v
                n_vert += 1
        # BFS
        while q_head < q_tail:
            cur_tri = queue[q_head]; q_head += 1
            a_lo = adj_off[cur_tri]
            a_hi = adj_off[cur_tri + 1]
            for k in range(a_lo, a_hi):
                nb = adj_idx[k]
                # membership via hash
                h2 = (nb * 2654435761) & ml_mask
                nb_li = -1
                while ml_keys[h2] != -1:
                    if ml_keys[h2] == nb:
                        nb_li = ml_vals[h2]
                        break
                    h2 = (h2 + 1) & ml_mask
                if nb_li >= 0 and not visited_tri[nb_li]:
                    visited_tri[nb_li] = True
                    queue[q_tail] = nb
                    q_tail += 1
                    for j in range(3):
                        v = np.int32(tris_np[nb, j])
                        h = (v * 2654435761) & seen_mask
                        found = False
                        while seen_keys[h] != -1:
                            if seen_keys[h] == v:
                                found = True
                                break
                            h = (h + 1) & seen_mask
                        if not found:
                            seen_keys[h] = v
                            vertex_order[n_vert] = v
                            n_vert += 1

    # Catch unreached verts (Python fallback iterates `sorted(ml_verts)`).
    extra_v = np.empty(3 * n_tris_m, dtype=np.int32)
    n_extra = 0
    for t in range(n_tris_m):
        gt = meshlet_tris[t]
        for j in range(3):
            v = np.int32(tris_np[gt, j])
            h = (v * 2654435761) & seen_mask
            found = False
            while seen_keys[h] != -1:
                if seen_keys[h] == v:
                    found = True
                    break
                h = (h + 1) & seen_mask
            if not found:
                seen_keys[h] = v
                extra_v[n_extra] = v
                n_extra += 1
    if n_extra > 0:
        extra_sorted = np.sort(extra_v[:n_extra])
        for k in range(n_extra):
            vertex_order[n_vert] = extra_sorted[k]
            n_vert += 1
    return vertex_order[:n_vert], np.int32(n_root)


@njit(cache=True)
def greedy_nn_order_nb(global_indices, vert_pos_float, start_idx):
    """Greedy nearest-neighbour walk on `vert_pos_float[global_indices]`.
    Returns a reordered copy of `global_indices` (int64 array)."""
    n = global_indices.shape[0]
    if n <= 1:
        return global_indices.copy()
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        gi = global_indices[i]
        pts[i, 0] = vert_pos_float[gi, 0]
        pts[i, 1] = vert_pos_float[gi, 1]
        pts[i, 2] = vert_pos_float[gi, 2]
    visited = np.zeros(n, dtype=np.bool_)
    order = np.empty(n, dtype=np.int64)
    order[0] = start_idx
    visited[start_idx] = True
    cur = start_idx
    for k in range(1, n):
        best_d = np.float64(1e30)
        best_i = -1
        for i in range(n):
            if visited[i]:
                continue
            dx = pts[i, 0] - pts[cur, 0]
            dy = pts[i, 1] - pts[cur, 1]
            dz = pts[i, 2] - pts[cur, 2]
            d = dx*dx + dy*dy + dz*dz
            if d < best_d:
                best_d = d
                best_i = i
        order[k] = best_i
        visited[best_i] = True
        cur = best_i
    out = np.empty(n, dtype=np.int64)
    for k in range(n):
        out[k] = global_indices[order[k]]
    return out

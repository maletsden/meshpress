"""Numba kernel for parallelogram-predictor greedy traversal order.

Drop-in fast path for `parallelogram_predictor._greedy_order`. Same algorithm,
same output, same tie-breaks — caller compares as a black box.

Inputs are converted from dict/set form into CSR + bool arrays before
entering the numba kernel. All work after conversion runs at compiled speed.

Topology format (LOCAL vert/tri IDs, per meshlet):
  tri_verts          (T, 3) int32       local IDs
  edge_id_per_tri    (T, 3) int32       canonical edges in order (ta,tb)/(tb,tc)/(tc,ta)
  edge_off           (E+1,) int32       CSR offsets into edge_tris
  edge_tris          (2E_act,) int32    CSR concat'd tri ids per edge
  vert_off           (V+1,) int32       CSR offsets into vert_tris
  vert_tris          (sum,) int32       CSR concat'd tri ids per vert
  v_in_tri_pos       (T, 3) int32       which local-vert slot in tri holds vert? (not used)

Output (numba-friendly arrays; caller maps back to (v, kind, refs) tuples):
  order_v       (n_int,) int32          interior local vert in chosen order
  order_kind    (n_int,) int32          0=para 1=mid 2=one 3=none
  order_a       (n_int,) int32          first ref (-1 if not used)
  order_b       (n_int,) int32          second ref (-1 if not used)
  order_c       (n_int,) int32          third ref (para only, else -1)
  order_d_ac    (n_int,) int32          (para only)
  order_d_bc    (n_int,) int32          (para only)
"""

from __future__ import annotations

import numpy as np
from numba import njit


KIND_PARA = 0
KIND_MID = 1
KIND_ONE = 2
KIND_NONE = 3


# =====================================================================
# Topology setup (Python; runs once per meshlet, cheap)
# =====================================================================

@njit(cache=True)
def _build_topo_csr_kernel(tri_verts_np, n_verts_local):
    """Numba implementation. Builds canonical edge IDs + CSR over edges
    and verts. Edge IDs assigned by visit order (matches Python version).
    """
    T = tri_verts_np.shape[0]
    edge_id_per_tri = np.empty((T, 3), dtype=np.int32)
    # Pair encoding: key = min_v * n_verts_local + max_v.
    # We need a hash map. Use chained linear probing on size 2*max_edges.
    cap = 1
    target = 6 * T  # 3 edges per tri × 2 directions, oversized for low collisions
    while cap < target:
        cap *= 2
    keys = np.full(cap, -1, dtype=np.int64)
    vals = np.empty(cap, dtype=np.int32)
    mask = cap - 1
    next_id = 0
    for t in range(T):
        a = tri_verts_np[t, 0]
        b = tri_verts_np[t, 1]
        c = tri_verts_np[t, 2]
        for k in range(3):
            if k == 0:
                u = a; v = b
            elif k == 1:
                u = b; v = c
            else:
                u = c; v = a
            if u <= v:
                key = u * n_verts_local + v
            else:
                key = v * n_verts_local + u
            # Probe
            h = (key * np.int64(2654435761)) & mask
            while True:
                cur_key = keys[h]
                if cur_key == key:
                    eid = vals[h]
                    break
                if cur_key < 0:
                    keys[h] = key
                    vals[h] = next_id
                    eid = next_id
                    next_id += 1
                    break
                h = (h + 1) & mask
            edge_id_per_tri[t, k] = eid

    n_edges = next_id
    edge_deg = np.zeros(n_edges, dtype=np.int32)
    for t in range(T):
        for k in range(3):
            edge_deg[edge_id_per_tri[t, k]] += 1
    edge_off = np.zeros(n_edges + 1, dtype=np.int32)
    for i in range(n_edges):
        edge_off[i + 1] = edge_off[i] + edge_deg[i]
    edge_tris = np.empty(int(edge_off[n_edges]), dtype=np.int32)
    cur = edge_off[:n_edges].copy()
    for t in range(T):
        for k in range(3):
            e = edge_id_per_tri[t, k]
            edge_tris[cur[e]] = t
            cur[e] += 1

    vert_deg = np.zeros(n_verts_local, dtype=np.int32)
    for t in range(T):
        for k in range(3):
            vert_deg[tri_verts_np[t, k]] += 1
    vert_off = np.zeros(n_verts_local + 1, dtype=np.int32)
    for i in range(n_verts_local):
        vert_off[i + 1] = vert_off[i] + vert_deg[i]
    vert_tris = np.empty(int(vert_off[n_verts_local]), dtype=np.int32)
    cur2 = vert_off[:n_verts_local].copy()
    for t in range(T):
        for k in range(3):
            v = tri_verts_np[t, k]
            vert_tris[cur2[v]] = t
            cur2[v] += 1

    return edge_id_per_tri, edge_off, edge_tris, vert_off, vert_tris


def _build_topo_csr(tri_verts_np):
    """Wrapper: derives n_verts_local then runs JIT kernel."""
    T = tri_verts_np.shape[0]
    n_verts_local = int(tri_verts_np.max()) + 1 if T > 0 else 0
    (edge_id_per_tri, edge_off, edge_tris, vert_off, vert_tris) = \
        _build_topo_csr_kernel(tri_verts_np, n_verts_local)
    return (edge_id_per_tri, edge_off, edge_tris, vert_off, vert_tris,
            n_verts_local)


# =====================================================================
# Helpers (njit)
# =====================================================================

@njit(cache=True, inline='always')
def _find_second_ring_apex_nb(a, c, t_skip, decoded,
                              tri_verts, edge_id_per_tri,
                              edge_off, edge_tris):
    """Find third vertex of a triangle sharing edge (a, c) with t_skip
    whose apex is decoded. Returns -1 if missing. Lowest-id tie-break."""
    # Look up edge ID via t_skip's slot. But t_skip may not contain edge (a, c)
    # if we're called from the dependency walker. Instead we find edge ID by
    # walking edges of any tri known to contain (a, c). t_skip contains it for
    # the (b, c) and (a, c) calls in _find_predict_context. For
    # _build_context_dependency we pass t_other (always contains the edge).
    # So edge ID = canonical edge of t_skip touching both a and c.
    edge_id = -1
    for k in range(3):
        if k == 0:
            u = tri_verts[t_skip, 0]; v = tri_verts[t_skip, 1]
        elif k == 1:
            u = tri_verts[t_skip, 1]; v = tri_verts[t_skip, 2]
        else:
            u = tri_verts[t_skip, 2]; v = tri_verts[t_skip, 0]
        if (u == a and v == c) or (u == c and v == a):
            edge_id = edge_id_per_tri[t_skip, k]
            break
    if edge_id < 0:
        return -1
    best = -1
    for k in range(edge_off[edge_id], edge_off[edge_id + 1]):
        t_other = edge_tris[k]
        if t_other == t_skip:
            continue
        ta = tri_verts[t_other, 0]
        tb = tri_verts[t_other, 1]
        tc = tri_verts[t_other, 2]
        third = -1
        if ta != a and ta != c:
            third = ta
        elif tb != a and tb != c:
            third = tb
        elif tc != a and tc != c:
            third = tc
        if third >= 0 and decoded[third]:
            if best < 0 or third < best:
                best = third
    return best


@njit(cache=True)
def _find_predict_context_nb(v, decoded, tri_verts, edge_id_per_tri,
                             edge_off, edge_tris, vert_off, vert_tris,
                             out_refs):
    """Compute predict context for vertex v. Writes (a, b, c, d_ac, d_bc) into
    out_refs[0..5) and returns kind (0=para, 1=mid, 2=one, 3=none)."""
    best_para_a = -1; best_para_b = -1; best_para_c = -1
    best_para_t = -1
    best_mid_a = -1; best_mid_b = -1
    best_one = -1

    for k in range(vert_off[v], vert_off[v + 1]):
        t = vert_tris[k]
        ta = tri_verts[t, 0]; tb = tri_verts[t, 1]; tc = tri_verts[t, 2]
        # Find the two others (not v)
        if ta == v:
            o1 = tb; o2 = tc
            opp_edge_k = 1  # (tb, tc)
        elif tb == v:
            o1 = ta; o2 = tc
            opp_edge_k = 2  # (tc, ta)
        elif tc == v:
            o1 = ta; o2 = tb
            opp_edge_k = 0  # (ta, tb)
        else:
            continue
        # Sort to canonical (a, b) with a < b
        if o1 < o2:
            a = o1; b = o2
        else:
            a = o2; b = o1
        a_in = decoded[a]
        b_in = decoded[b]
        if a_in and b_in:
            # Walk edge (a, b) -> other tris -> third in decoded
            edge_id = edge_id_per_tri[t, opp_edge_k]
            for kk in range(edge_off[edge_id], edge_off[edge_id + 1]):
                t_other = edge_tris[kk]
                if t_other == t:
                    continue
                ra = tri_verts[t_other, 0]
                rb = tri_verts[t_other, 1]
                rc = tri_verts[t_other, 2]
                third = -1
                if ra != a and ra != b:
                    third = ra
                elif rb != a and rb != b:
                    third = rb
                elif rc != a and rc != b:
                    third = rc
                if third >= 0 and decoded[third]:
                    # Tie-break: lower (a, b, c) wins lexicographically.
                    if (best_para_a < 0
                            or a < best_para_a
                            or (a == best_para_a and b < best_para_b)
                            or (a == best_para_a and b == best_para_b
                                and third < best_para_c)):
                        best_para_a = a
                        best_para_b = b
                        best_para_c = third
                        best_para_t = t_other
            # cand_mid = (a, b)
            if (best_mid_a < 0
                    or a < best_mid_a
                    or (a == best_mid_a and b < best_mid_b)):
                best_mid_a = a
                best_mid_b = b
        elif a_in or b_in:
            one = a if a_in else b
            if best_one < 0 or one < best_one:
                best_one = one

    if best_para_a >= 0:
        a = best_para_a; b = best_para_b; c = best_para_c
        d_ac = _find_second_ring_apex_nb(a, c, best_para_t, decoded,
                                          tri_verts, edge_id_per_tri,
                                          edge_off, edge_tris)
        d_bc = _find_second_ring_apex_nb(b, c, best_para_t, decoded,
                                          tri_verts, edge_id_per_tri,
                                          edge_off, edge_tris)
        out_refs[0] = a; out_refs[1] = b; out_refs[2] = c
        out_refs[3] = d_ac; out_refs[4] = d_bc
        return KIND_PARA
    if best_mid_a >= 0:
        out_refs[0] = best_mid_a; out_refs[1] = best_mid_b
        out_refs[2] = -1; out_refs[3] = -1; out_refs[4] = -1
        return KIND_MID
    if best_one >= 0:
        out_refs[0] = best_one
        out_refs[1] = -1; out_refs[2] = -1; out_refs[3] = -1; out_refs[4] = -1
        return KIND_ONE
    for j in range(5):
        out_refs[j] = -1
    return KIND_NONE


@njit(cache=True)
def _build_dep_csr_nb(int_local, in_int, tri_verts, edge_id_per_tri,
                     edge_off, edge_tris, vert_off, vert_tris):
    """Build dep CSR for interior vertices. Two-pass: count then fill.

    Returns (dep_off (n_local+1,), dep_idx (sum,)).
    """
    n_local = vert_off.shape[0] - 1
    counts = np.zeros(n_local, dtype=np.int32)
    # We use a (n_local,) bool scratch to dedupe per source u.
    scratch = np.zeros(n_local, dtype=np.bool_)
    # Pass 1: count
    for wi in range(int_local.shape[0]):
        w = int_local[wi]
        # Track which dep targets we've added for this w
        added_marks = np.zeros(n_local, dtype=np.bool_)
        for k in range(vert_off[w], vert_off[w + 1]):
            t = vert_tris[k]
            ta = tri_verts[t, 0]; tb = tri_verts[t, 1]; tc = tri_verts[t, 2]
            if ta == w:
                a = tb; b = tc
            elif tb == w:
                a = ta; b = tc
            elif tc == w:
                a = ta; b = tb
            else:
                continue
            # a, b -> dep[a].add(w), dep[b].add(w) if in remaining
            if in_int[a] and not added_marks[a]:
                counts[a] += 1; added_marks[a] = True
            if in_int[b] and not added_marks[b]:
                counts[b] += 1; added_marks[b] = True
            # Find c across edge (a, b) of t
            if (ta == w):
                opp_edge_k = 1
            elif (tb == w):
                opp_edge_k = 2
            else:
                opp_edge_k = 0
            edge_id = edge_id_per_tri[t, opp_edge_k]
            for kk in range(edge_off[edge_id], edge_off[edge_id + 1]):
                t_other = edge_tris[kk]
                if t_other == t:
                    continue
                ra = tri_verts[t_other, 0]
                rb = tri_verts[t_other, 1]
                rc = tri_verts[t_other, 2]
                third = -1
                if ra != a and ra != b:
                    third = ra
                elif rb != a and rb != b:
                    third = rb
                elif rc != a and rc != b:
                    third = rc
                if third < 0:
                    continue
                c = third
                if in_int[c] and not added_marks[c]:
                    counts[c] += 1; added_marks[c] = True
                # Second-ring apexes
                for axcx_i in range(2):
                    if axcx_i == 0:
                        ax = a; cx = c
                    else:
                        ax = b; cx = c
                    # Find edge id of (ax, cx) in t_other
                    e_xc = -1
                    for kkk in range(3):
                        if kkk == 0:
                            u_ = tri_verts[t_other, 0]; v_ = tri_verts[t_other, 1]
                        elif kkk == 1:
                            u_ = tri_verts[t_other, 1]; v_ = tri_verts[t_other, 2]
                        else:
                            u_ = tri_verts[t_other, 2]; v_ = tri_verts[t_other, 0]
                        if (u_ == ax and v_ == cx) or (u_ == cx and v_ == ax):
                            e_xc = edge_id_per_tri[t_other, kkk]
                            break
                    if e_xc < 0:
                        continue
                    for kkkk in range(edge_off[e_xc], edge_off[e_xc + 1]):
                        t3 = edge_tris[kkkk]
                        if t3 == t_other:
                            continue
                        sa = tri_verts[t3, 0]
                        sb = tri_verts[t3, 1]
                        sc = tri_verts[t3, 2]
                        apex = -1
                        if sa != ax and sa != cx:
                            apex = sa
                        elif sb != ax and sb != cx:
                            apex = sb
                        elif sc != ax and sc != cx:
                            apex = sc
                        if apex >= 0 and in_int[apex] and not added_marks[apex]:
                            counts[apex] += 1; added_marks[apex] = True

    # Build offsets
    dep_off = np.zeros(n_local + 1, dtype=np.int32)
    for i in range(n_local):
        dep_off[i + 1] = dep_off[i] + counts[i]
    total = dep_off[n_local]
    dep_idx = np.empty(total, dtype=np.int32)
    cur = dep_off[:n_local].copy()

    # Pass 2: fill
    for wi in range(int_local.shape[0]):
        w = int_local[wi]
        added_marks = np.zeros(n_local, dtype=np.bool_)
        for k in range(vert_off[w], vert_off[w + 1]):
            t = vert_tris[k]
            ta = tri_verts[t, 0]; tb = tri_verts[t, 1]; tc = tri_verts[t, 2]
            if ta == w:
                a = tb; b = tc
                opp_edge_k = 1
            elif tb == w:
                a = ta; b = tc
                opp_edge_k = 2
            elif tc == w:
                a = ta; b = tb
                opp_edge_k = 0
            else:
                continue
            if in_int[a] and not added_marks[a]:
                dep_idx[cur[a]] = w; cur[a] += 1; added_marks[a] = True
            if in_int[b] and not added_marks[b]:
                dep_idx[cur[b]] = w; cur[b] += 1; added_marks[b] = True
            edge_id = edge_id_per_tri[t, opp_edge_k]
            for kk in range(edge_off[edge_id], edge_off[edge_id + 1]):
                t_other = edge_tris[kk]
                if t_other == t:
                    continue
                ra = tri_verts[t_other, 0]
                rb = tri_verts[t_other, 1]
                rc = tri_verts[t_other, 2]
                third = -1
                if ra != a and ra != b:
                    third = ra
                elif rb != a and rb != b:
                    third = rb
                elif rc != a and rc != b:
                    third = rc
                if third < 0:
                    continue
                c = third
                if in_int[c] and not added_marks[c]:
                    dep_idx[cur[c]] = w; cur[c] += 1; added_marks[c] = True
                for axcx_i in range(2):
                    if axcx_i == 0:
                        ax = a; cx = c
                    else:
                        ax = b; cx = c
                    e_xc = -1
                    for kkk in range(3):
                        if kkk == 0:
                            u_ = tri_verts[t_other, 0]; v_ = tri_verts[t_other, 1]
                        elif kkk == 1:
                            u_ = tri_verts[t_other, 1]; v_ = tri_verts[t_other, 2]
                        else:
                            u_ = tri_verts[t_other, 2]; v_ = tri_verts[t_other, 0]
                        if (u_ == ax and v_ == cx) or (u_ == cx and v_ == ax):
                            e_xc = edge_id_per_tri[t_other, kkk]
                            break
                    if e_xc < 0:
                        continue
                    for kkkk in range(edge_off[e_xc], edge_off[e_xc + 1]):
                        t3 = edge_tris[kkkk]
                        if t3 == t_other:
                            continue
                        sa = tri_verts[t3, 0]
                        sb = tri_verts[t3, 1]
                        sc = tri_verts[t3, 2]
                        apex = -1
                        if sa != ax and sa != cx:
                            apex = sa
                        elif sb != ax and sb != cx:
                            apex = sb
                        elif sc != ax and sc != cx:
                            apex = sc
                        if apex >= 0 and in_int[apex] and not added_marks[apex]:
                            dep_idx[cur[apex]] = w
                            cur[apex] += 1
                            added_marks[apex] = True
    return dep_off, dep_idx


@njit(cache=True)
def _greedy_order_kernel(
        int_local_sorted, decoded_init_arr,
        tri_verts, edge_id_per_tri,
        edge_off, edge_tris, vert_off, vert_tris,
        n_verts_local,
        out_order_v, out_order_kind, out_order_refs):
    """Numba kernel for greedy traversal. int_local_sorted: ascending order
    of interior local IDs. decoded_init_arr: array of boundary local IDs.

    Writes results into out_order_v / kind / refs (n_int, 5).
    Returns n_int (always == len(int_local_sorted)).
    """
    n_int = int_local_sorted.shape[0]
    if n_int == 0:
        return 0
    decoded = np.zeros(n_verts_local, dtype=np.bool_)
    for i in range(decoded_init_arr.shape[0]):
        decoded[decoded_init_arr[i]] = True
    in_int = np.zeros(n_verts_local, dtype=np.bool_)
    for i in range(n_int):
        in_int[int_local_sorted[i]] = True

    # Initial dep CSR
    dep_off, dep_idx = _build_dep_csr_nb(
        int_local_sorted, in_int, tri_verts, edge_id_per_tri,
        edge_off, edge_tris, vert_off, vert_tris)

    # ctx arrays: rank (0..3), refs (5)
    ctx_rank = np.full(n_verts_local, 3, dtype=np.int32)
    ctx_refs = np.full((n_verts_local, 5), -1, dtype=np.int32)
    tmp_refs = np.empty(5, dtype=np.int32)
    for i in range(n_int):
        v = int_local_sorted[i]
        k = _find_predict_context_nb(
            v, decoded, tri_verts, edge_id_per_tri,
            edge_off, edge_tris, vert_off, vert_tris, tmp_refs)
        ctx_rank[v] = k
        for j in range(5):
            ctx_refs[v, j] = tmp_refs[j]

    remaining = np.ones(n_int, dtype=np.bool_)
    n_left = n_int
    n_written = 0
    while n_left > 0:
        # Find best: scan sorted ascending; minimize (rank, local_id).
        # Already ascending by id, so first occurrence of min rank wins.
        best_idx = -1
        best_rank = 4
        for i in range(n_int):
            if not remaining[i]:
                continue
            v = int_local_sorted[i]
            r = ctx_rank[v]
            if r < best_rank:
                best_rank = r
                best_idx = i
                if r == 0:
                    break
        if best_idx < 0:
            break
        v = int_local_sorted[best_idx]
        # Recompute fresh refs for the chosen v (cached refs may be stale
        # when rank cached at 0 but apex IDs improved during dep skips).
        k = _find_predict_context_nb(
            v, decoded, tri_verts, edge_id_per_tri,
            edge_off, edge_tris, vert_off, vert_tris, tmp_refs)
        out_order_v[n_written] = v
        out_order_kind[n_written] = k
        for j in range(5):
            out_order_refs[n_written, j] = tmp_refs[j]
        n_written += 1
        remaining[best_idx] = False
        decoded[v] = True
        # Propagate dep update: any w in dep[v] (in remaining) whose ctx
        # might improve. Skip w already at rank 0 (best); their refs are
        # recomputed at the pick step above.
        for k_off in range(dep_off[v], dep_off[v + 1]):
            w = dep_idx[k_off]
            if not in_int[w]:
                continue
            # w must still be in remaining
            # Find w's index in int_local_sorted via binary search.
            # Since int_local_sorted is ascending, do bisect.
            lo = 0; hi = n_int - 1; w_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                wm = int_local_sorted[mid]
                if wm == w:
                    w_idx = mid
                    break
                if wm < w:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if w_idx < 0 or not remaining[w_idx]:
                continue
            if ctx_rank[w] == 0:
                continue
            kw = _find_predict_context_nb(
                w, decoded, tri_verts, edge_id_per_tri,
                edge_off, edge_tris, vert_off, vert_tris, tmp_refs)
            ctx_rank[w] = kw
            for j in range(5):
                ctx_refs[w, j] = tmp_refs[j]
        n_left -= 1

    return n_written


# =====================================================================
# Wrapper exposed to the codec — mirrors original `_greedy_order` API
# =====================================================================

def greedy_order_nb(int_local, decoded_init, tri_verts_np,
                    edge_to_tris_unused=None, vert_to_tris_unused=None):
    """Same return type as the Python `_greedy_order`: list of
    (v_local, kind_str, refs_tuple). The trailing dict args are
    kept for signature compat but unused (CSR built fresh from tri_verts).
    """
    if len(int_local) == 0:
        return []
    tv = np.asarray(tri_verts_np, dtype=np.int32)
    (edge_id_per_tri, edge_off, edge_tris,
     vert_off, vert_tris, n_verts_local) = _build_topo_csr(tv)
    int_local_sorted = np.array(sorted(int(v) for v in int_local),
                                dtype=np.int32)
    decoded_init_arr = np.array([int(v) for v in decoded_init],
                                dtype=np.int32)
    n_int = int_local_sorted.shape[0]
    out_v = np.empty(n_int, dtype=np.int32)
    out_kind = np.empty(n_int, dtype=np.int32)
    out_refs = np.empty((n_int, 5), dtype=np.int32)
    _greedy_order_kernel(
        int_local_sorted, decoded_init_arr,
        tv, edge_id_per_tri, edge_off, edge_tris, vert_off, vert_tris,
        n_verts_local,
        out_v, out_kind, out_refs)
    kind_names = ('para', 'mid', 'one', 'none')
    out = []
    for i in range(n_int):
        k = int(out_kind[i])
        if k == KIND_PARA:
            refs = (int(out_refs[i, 0]), int(out_refs[i, 1]),
                    int(out_refs[i, 2]), int(out_refs[i, 3]),
                    int(out_refs[i, 4]))
        elif k == KIND_MID:
            refs = (int(out_refs[i, 0]), int(out_refs[i, 1]))
        elif k == KIND_ONE:
            refs = (int(out_refs[i, 0]),)
        else:
            refs = ()
        out.append((int(out_v[i]), kind_names[k], refs))
    return out
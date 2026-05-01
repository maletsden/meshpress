"""
Parallelogram predictor for interior meshlet vertices (idea #11).

Touma-Gotsman parallelogram: for triangle T=(v, a, b) where a, b already
decoded, and adjacent triangle T'=(a, b, c) with c already decoded,
predict v ≈ a + b - c.  Exact when (v, a, b, c) is planar.  Real meshes
deviate by curvature; residual = true - predicted is encoded.

Boundary verts stay on the global int grid (crack-free, untouched).  This
module handles only the interior stream of a single meshlet.

Encode/decode causality: each interior vertex's prediction uses already-
reconstructed positions of boundary + previously-decoded interior verts.
The traversal order is reproduced deterministically by both encoder and
decoder (greedy: pick interior vertex with the most decoded neighbours,
ties broken by lower global index), so no order needs to be transmitted.

Quantization: per-axis uniform δ = 2 * per_coord_err, giving worst-case
axis error δ/2 = per_coord_err (matches the Haar boundary).

Bitstream layout per meshlet (interior):
    9 B header  : 3 × int16 axis-min + 3 × uint8 bits-per-code
    body        : n_int × Σ bits_per_axis  (one packed code per interior
                  vertex per axis, axis-min subtracted)
"""

from __future__ import annotations

import numpy as np


def _build_meshlet_local_data(ml_tris_global_idx, tris_np):
    """Returns:
        tri_verts        : (T, 3) int — global vertex IDs per meshlet tri
        edge_to_tris     : dict (a,b)→list of meshlet-local tri indices
                           (a,b) sorted by min/max
        vert_to_tris     : dict global vert → list of meshlet-local tri idx
    """
    tri_verts = tris_np[ml_tris_global_idx]  # (T, 3) int64
    edge_to_tris = {}
    vert_to_tris = {}
    for t_local, (a, b, c) in enumerate(tri_verts):
        for v in (int(a), int(b), int(c)):
            vert_to_tris.setdefault(v, []).append(t_local)
        for u, w in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (min(u, w), max(u, w))
            edge_to_tris.setdefault(key, []).append(t_local)
    return tri_verts, edge_to_tris, vert_to_tris


def _find_second_ring_apex(a, c, t_skip, decoded, tri_verts, edge_to_tris):
    """Third vertex of a triangle sharing edge (a,c) with `t_skip` whose
    apex is already decoded. None if missing. Deterministic tie-break:
    lowest decoded apex global ID."""
    edge = (min(a, c), max(a, c))
    best = None
    for t_other in edge_to_tris.get(edge, ()):
        if t_other == t_skip:
            continue
        tri_other = [int(x) for x in tri_verts[t_other]]
        third = [x for x in tri_other if x != a and x != c]
        if len(third) == 1 and third[0] in decoded:
            cand = int(third[0])
            if best is None or cand < best:
                best = cand
    return best


def _find_predict_context(v, decoded, tri_verts, edge_to_tris, vert_to_tris):
    """Return (kind, refs):
        kind = 'para' , refs = (a, b, c, d_ac, d_bc)
                           d_ac, d_bc may be -1 if neighbour missing.
        kind = 'mid'  , refs = (a, b)
        kind = 'one'  , refs = (a,)
        kind = 'none' , refs = ()
    Tie-break: lowest a, b, c global IDs for determinism. Second-ring
    apexes resolved deterministically from the chosen (a,b,c) so encoder
    and decoder agree without a flag bit.
    """
    best_para = None        # (a, b, c)
    best_para_t_local = None  # the triangle T'(a,b,c)
    best_mid = None
    best_one = None
    for t_local in vert_to_tris.get(v, ()):
        a_, b_, c_ = (int(x) for x in tri_verts[t_local])
        others = [x for x in (a_, b_, c_) if x != v]
        if len(others) != 2:
            continue
        a, b = sorted(others)
        a_in = a in decoded
        b_in = b in decoded
        if a_in and b_in:
            edge = (a, b)
            for t_other in edge_to_tris.get(edge, ()):
                if t_other == t_local:
                    continue
                tri_other = [int(x) for x in tri_verts[t_other]]
                third = [x for x in tri_other if x != a and x != b]
                if len(third) == 1 and third[0] in decoded:
                    cand = (a, b, int(third[0]))
                    if best_para is None or cand < best_para:
                        best_para = cand
                        best_para_t_local = t_other
            cand_mid = (a, b)
            if best_mid is None or cand_mid < best_mid:
                best_mid = cand_mid
        elif a_in or b_in:
            one = a if a_in else b
            cand_one = (one,)
            if best_one is None or cand_one < best_one:
                best_one = cand_one
    if best_para is not None:
        a, b, c = best_para
        d_ac = _find_second_ring_apex(
            a, c, best_para_t_local, decoded, tri_verts, edge_to_tris)
        d_bc = _find_second_ring_apex(
            b, c, best_para_t_local, decoded, tri_verts, edge_to_tris)
        return 'para', (a, b, c,
                        -1 if d_ac is None else d_ac,
                        -1 if d_bc is None else d_bc)
    if best_mid is not None:
        return 'mid', best_mid
    if best_one is not None:
        return 'one', best_one
    return 'none', ()


_KIND_RANK = {'para': 0, 'mid': 1, 'one': 2, 'none': 3}


def _build_context_dependency(remaining, tri_verts, edge_to_tris, vert_to_tris):
    """For each vertex u in `remaining`, return the set of remaining
    vertices `w` whose predict-context can change when `u` becomes
    decoded.

    Vertex w's context (kind, refs) depends on:
      - a, b   : triangle-mates of w (1-hop) — sets 'mid'/'one'/'para' refs.
      - c      : third vertex of triangle adjacent to T(w,a,b) across edge
                 (a,b) — promotes from 'mid' to 'para'.
      - d_ac   : third vertex of triangle adjacent to T'(a,b,c) across
                 edge (a,c) — populates the second-ring apex on 'para'.
      - d_bc   : analogous on edge (b,c).
    Any of these becoming decoded changes w's representative state, so
    the dep map must list w under each.
    """
    dep = {v: set() for v in remaining}
    for w in remaining:
        for t in vert_to_tris.get(w, ()):
            tri = tri_verts[t]
            a_, b_, c_ = int(tri[0]), int(tri[1]), int(tri[2])
            others = [x for x in (a_, b_, c_) if x != w]
            if len(others) != 2:
                continue
            a, b = others
            if a in dep:
                dep[a].add(w)
            if b in dep:
                dep[b].add(w)
            edge_ab = (min(a, b), max(a, b))
            for t_other in edge_to_tris.get(edge_ab, ()):
                if t_other == t:
                    continue
                tri2 = tri_verts[t_other]
                ta, tb, tc = int(tri2[0]), int(tri2[1]), int(tri2[2])
                third = [x for x in (ta, tb, tc) if x != a and x != b]
                if len(third) != 1:
                    continue
                c = third[0]
                if c in dep:
                    dep[c].add(w)
                # Second ring: apexes across edges (a,c) and (b,c) of T'.
                for ax, cx in ((a, c), (b, c)):
                    edge_xc = (min(ax, cx), max(ax, cx))
                    for t3 in edge_to_tris.get(edge_xc, ()):
                        if t3 == t_other:
                            continue
                        tri3 = tri_verts[t3]
                        ra, rb, rc = int(tri3[0]), int(tri3[1]), int(tri3[2])
                        apex = [x for x in (ra, rb, rc) if x != ax and x != cx]
                        if len(apex) == 1 and apex[0] in dep:
                            dep[apex[0]].add(w)
    return dep


def _greedy_order(int_local, decoded_init, tri_verts, edge_to_tris, vert_to_tris):
    """Greedy traversal order over interior verts.

    At each step, pick the interior vertex with the strongest available
    context (para > mid > one > none). Ties broken by lowest global ID.
    Returns a list of (v, kind, refs) in encode/decode order.

    Implementation: cache predict-context per vertex. On each iteration
    scan the (statically) sorted remaining list and read cached states —
    no per-iteration sorting or recomputation. After picking the best,
    invalidate only the dependents of the newly-decoded vertex (computed
    once via triangle topology, including 2nd-ring apexes used by the
    13-feature MLP).

    Same output as the original O(n²·log n) version, but `_find_predict_context`
    is invoked only once per vertex per real context change instead of per
    outer-loop iteration.
    """
    decoded = set(decoded_init)
    sorted_int = sorted(int(v) for v in int_local)
    remaining = set(sorted_int)
    if not remaining:
        return []

    dep = _build_context_dependency(
        remaining, tri_verts, edge_to_tris, vert_to_tris)

    ctx = {}  # v -> (rank, kind, refs)
    for v in sorted_int:
        kind, refs = _find_predict_context(
            v, decoded, tri_verts, edge_to_tris, vert_to_tris)
        ctx[v] = (_KIND_RANK[kind], kind, refs)

    order = []
    while remaining:
        best = None  # (rank, v, kind, refs)
        for v in sorted_int:
            if v not in remaining:
                continue
            rank, kind, refs = ctx[v]
            if best is None or (rank, v) < (best[0], best[1]):
                best = (rank, v, kind, refs)
                if rank == 0:
                    break
        _, v, _kind_old, _refs_old = best
        # Recompute fresh refs for the picked vertex — its `c` or 2nd-ring
        # apex may have just become available via prior iterations whose
        # dep updates we skipped (rank stayed at 0 in the cache, so we did
        # not refresh refs there).
        kind, refs = _find_predict_context(
            v, decoded, tri_verts, edge_to_tris, vert_to_tris)
        order.append((v, kind, refs))
        remaining.discard(v)
        decoded.add(v)
        for w in dep.get(v, ()):
            if w not in remaining:
                continue
            if ctx[w][0] == 0:
                # Already at best rank. Refs may change (e.g., a lower-id
                # apex becomes decoded) but the pick order does not. The
                # final refs are recomputed when w is actually picked.
                continue
            new_kind, new_refs = _find_predict_context(
                w, decoded, tri_verts, edge_to_tris, vert_to_tris)
            ctx[w] = (_KIND_RANK[new_kind], new_kind, new_refs)
    return order


def _predict(kind, refs, recon_pos, fallback_centroid):
    if kind == 'para':
        a, b, c = refs[0], refs[1], refs[2]
        return recon_pos[a] + recon_pos[b] - recon_pos[c]
    if kind == 'mid':
        a, b = refs
        return 0.5 * (recon_pos[a] + recon_pos[b])
    if kind == 'one':
        (a,) = refs
        return recon_pos[a].copy()
    return fallback_centroid.copy()


def quantize_interior_parallelogram(int_local, bnd_local, ml_tris_global_idx,
                                    tris_np, vn, bnd_recon_float,
                                    per_coord_err, mlp=None,
                                    collect_training=False):
    """Encode interior vertices via parallelogram prediction + per-axis
    uniform quantization.

    Args:
        int_local : list of global vertex IDs (interior of meshlet)
        bnd_local : list of global vertex IDs (boundary of meshlet)
        ml_tris_global_idx : array of global triangle IDs in the meshlet
        tris_np   : (n_tri, 3) global triangle vertex IDs
        vn        : (n_vert, 3) float positions (normalized space)
        bnd_recon_float : (n_vert, 3) reconstructed boundary positions
                          (only entries indexed by bnd_local need to be valid)
        per_coord_err : per-axis error budget

    Returns:
        recon  : (len(int_local), 3) float reconstructions in input order
                 of int_local (NOT traversal order)
        bits   : total bits including 9 B header
        stats  : dict with hits per kind + traversal order
    """
    n_int = len(int_local)
    if n_int == 0:
        return np.zeros((0, 3), dtype=np.float64), 0, {
            'hits': {'para': 0, 'mid': 0, 'one': 0, 'none': 0},
            'order': [],
        }

    tri_verts, edge_to_tris, vert_to_tris = _build_meshlet_local_data(
        ml_tris_global_idx, tris_np)

    decoded_init = [int(v) for v in bnd_local]
    order = _greedy_order(
        int_local, decoded_init, tri_verts, edge_to_tris, vert_to_tris)

    # Fallback centroid = mean of boundary recons (or zero if no bnd).
    if len(bnd_local) > 0:
        fallback_centroid = np.mean(
            [bnd_recon_float[v] for v in bnd_local], axis=0)
    else:
        fallback_centroid = np.zeros(3, dtype=np.float64)

    recon_pos = {int(v): bnd_recon_float[int(v)].astype(np.float64)
                 for v in bnd_local}

    delta = 2.0 * per_coord_err  # per-axis uniform step

    codes = np.zeros((n_int, 3), dtype=np.int64)  # in traversal order
    traversal_recon = np.zeros((n_int, 3), dtype=np.float64)
    traversal_v = np.zeros(n_int, dtype=np.int64)
    hits = {'para': 0, 'mid': 0, 'one': 0, 'none': 0}

    train_feats = []   # (8,) arrays per para sample
    train_targets = []  # (3,) bias-in-local-frame arrays

    for i, (v, kind, refs) in enumerate(order):
        hits[kind] += 1
        pred = _predict(kind, refs, recon_pos, fallback_centroid)
        true = vn[v].astype(np.float64)
        # NN bias: only on full parallelogram contexts. Use 13-feat (with
        # second-ring apexes) when both d_ac and d_bc are decoded; else
        # skip NN (deterministic — encoder and decoder agree on presence).
        if kind == 'para' and (mlp is not None or collect_training):
            from utils.parallelogram_nn import build_frame_and_features
            a_id, b_id, c_id, d_ac_id, d_bc_id = refs
            if d_ac_id >= 0 and d_bc_id >= 0:
                a_pos = recon_pos[a_id]
                b_pos = recon_pos[b_id]
                c_pos = recon_pos[c_id]
                d_ac_pos = recon_pos[d_ac_id]
                d_bc_pos = recon_pos[d_bc_id]
                origin, T_frame, feats = build_frame_and_features(
                    a_pos[None, :], b_pos[None, :], c_pos[None, :],
                    d_ac_pos[None, :], d_bc_pos[None, :])
                feats = feats[0]
                T_frame = T_frame[0]
                if collect_training:
                    bias_world = true - pred
                    bias_local = T_frame.T @ bias_world
                    train_feats.append(feats)
                    train_targets.append(bias_local)
                if mlp is not None:
                    bias_local = mlp.forward(feats[None, :])[0]
                    bias_world = T_frame @ bias_local
                    pred = pred + bias_world
        residual = true - pred
        code = np.round(residual / delta).astype(np.int64)
        rec_residual = code.astype(np.float64) * delta
        rec = pred + rec_residual
        codes[i] = code
        traversal_recon[i] = rec
        traversal_v[i] = v
        recon_pos[int(v)] = rec

    # Map traversal-order recon back to int_local order
    v_to_recon = {int(traversal_v[i]): traversal_recon[i] for i in range(n_int)}
    recon = np.array([v_to_recon[int(v)] for v in int_local], dtype=np.float64)

    # Bit cost: per-axis best-of-{fixed, rice, exp-golomb, entropy}.
    from utils.residual_entropy import best_axis_bits
    bits = 0
    coder_tags = []
    for d in range(3):
        ax_bits, tag, param = best_axis_bits(codes[:, d])
        bits += ax_bits
        coder_tags.append((tag, param))

    return recon, bits, {
        'hits': hits,
        'order': [v for v, _, _ in order],
        'coders': coder_tags,
        'train_feats': train_feats,
        'train_targets': train_targets,
    }

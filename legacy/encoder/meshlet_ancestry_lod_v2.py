"""
Approach A v2: Ancestor-Redirection LOD with compact encoding + local quant.

Improvements over v1:
  v2a: Compact ancestor encoding: per-vertex (collapse_step, direct_parent)
       instead of N×K ancestor table. ~30% reduction on ancestors.
  v2b: Per-meshlet local quantization for interior vertices.
       Boundary vertices stay global-quant for cross-meshlet consistency.
       ~25% reduction on position data.
  v2c: Boundary subgraph has its own LOD (separate collapse chain).
       Allows LOD 0 below n_boundary_verts floor.

The decoder (mesh shader) logic remains ancestor redirection:
  for each triangle (v0, v1, v2):
      a0 = ancestor_at_lod(v0, k)
      a1 = ancestor_at_lod(v1, k)
      a2 = ancestor_at_lod(v2, k)
      if a0 != a1 && a1 != a2 && a0 != a2:
          emit triangle with positions[a0/1/2]

`ancestor_at_lod` now walks a short chain (compact encoding).
"""

import numpy as np
from utils.types import Model, CompressedModel
from ..encoder import Encoder
from utils.qem import (
    progressive_simplify_with_records,
    encode_ancestry_compact,
    estimate_compact_ancestor_bits,
    estimate_compact_ancestor_bits_entropy,
    ancestors_at_lod_compact_batch,
    _empirical_entropy_bits, _exp_golomb_bits,
)
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts,
)
from .meshlet_wavelet import (
    _to_numpy, _global_quantize, _dequantize_global, _local_quantize_meshlet,
    _identify_meshlet_boundary_verts, _bits_for_error,
    _amd_gts_bits, _amd_fifo_adjacency_bits,
)


# ---------------------------------------------------------------------
# Boundary subgraph simplification (v2c)
# ---------------------------------------------------------------------

def simplify_boundary_subgraph(verts_np, tris_np, boundary_verts,
                                target_frac=0.25):
    """Run QEM restricted to boundary-to-boundary collapses.

    Extracts the subgraph of boundary vertices + their connecting edges,
    then runs QEM on this subgraph using quadrics derived from ALL adjacent
    faces (full mesh quadrics for correct 3D error measure).

    Returns records analogous to `progressive_simplify_with_records`, but
    only containing boundary-to-boundary collapses.
    """
    import heapq
    from utils.qem import (
        compute_face_planes, compute_vertex_quadrics, _edge_cost,
    )

    n_v = len(verts_np)
    bnd_set = set(int(v) for v in boundary_verts)
    if not bnd_set:
        return {"records": []}

    verts = verts_np.copy()
    # Quadrics from full mesh (for 3D fidelity)
    planes = compute_face_planes(verts, tris_np)
    Q = compute_vertex_quadrics(verts, tris_np, planes)

    # Build boundary-only adjacency: two boundary verts are neighbors if they
    # share an edge in the full mesh
    bnd_nbrs = {v: set() for v in bnd_set}
    for ti in range(len(tris_np)):
        a, b, c = int(tris_np[ti, 0]), int(tris_np[ti, 1]), int(tris_np[ti, 2])
        for u, w in [(a, b), (b, c), (a, c)]:
            if u in bnd_set and w in bnd_set:
                bnd_nbrs[u].add(w)
                bnd_nbrs[w].add(u)

    vert_alive = {v: True for v in bnd_set}
    n_alive = len(bnd_set)
    target = max(3, int(len(bnd_set) * target_frac))

    heap = []
    counter = 0
    seen_edges = set()
    for vi in bnd_set:
        for vj in bnd_nbrs[vi]:
            edge = (min(vi, vj), max(vi, vj))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            Q_sum = Q[vi] + Q[vj]
            cost, opt = _edge_cost(Q_sum, verts[vi], verts[vj])
            heapq.heappush(heap, (cost, counter, vi, vj, opt))
            counter += 1

    records = []
    while heap and n_alive > target:
        cost, _, vi, vj, opt_pos = heapq.heappop(heap)
        if not vert_alive.get(vi, False) or not vert_alive.get(vj, False):
            continue
        if vj not in bnd_nbrs.get(vi, set()):
            continue

        v_rm = vi
        v_kp = vj

        v_rm_pos_orig = verts[v_rm].copy()
        v_kp_pos_before = verts[v_kp].copy()
        v_kp_pos_after = opt_pos.copy()

        # No triangle update tracking here; interior ancestor chain
        # handles those references at decode time.
        records.append({
            "v_rm": v_rm,
            "v_kp": v_kp,
            "v_rm_pos_orig": v_rm_pos_orig,
            "v_kp_pos_before": v_kp_pos_before,
            "v_kp_pos_after": v_kp_pos_after,
            "modified_tris": [],
            "removed_tris": [],
        })

        # Update adjacency
        for nb in list(bnd_nbrs[v_rm]):
            bnd_nbrs[nb].discard(v_rm)
            if nb != v_kp:
                bnd_nbrs[nb].add(v_kp)
                bnd_nbrs[v_kp].add(nb)
        bnd_nbrs[v_rm].clear()

        verts[v_kp] = opt_pos
        vert_alive[v_rm] = False
        n_alive -= 1
        Q[v_kp] = Q[v_kp] + Q[v_rm]

        # Re-insert v_kp's edges
        for nb in bnd_nbrs[v_kp]:
            if not vert_alive.get(nb, False):
                continue
            Q_sum = Q[v_kp] + Q[nb]
            c, opt = _edge_cost(Q_sum, verts[v_kp], verts[nb])
            heapq.heappush(heap, (c, counter, v_kp, nb, opt))
            counter += 1

    return {"records": records}


# ---------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------

class MeshletAncestryLODv2(Encoder):
    """Approach A v2: compact ancestors + local interior quant + boundary LOD.

    Args:
        variant: 'v2a' (compact only), 'v2b' (+ local quant),
                 'v2c' (+ boundary LOD, all three).
    """

    def __init__(self, max_verts=256, precision_error=0.0005,
                 n_lod_levels=5, variant='v2c', connectivity='gts_est',
                 verbose=False):
        """
        connectivity:
            'gts_est' - AMD GTS bit estimate (default, fast)
            'fifo'    - Real FIFO-adjacency encode (actual bitstream)
            'amd_gts' - Real AMD GTS strip encode (slower, best BPT)
        """
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.n_lod_levels = n_lod_levels
        self.variant = variant
        self.connectivity = connectivity
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)

        # Normalize
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        # Meshlets + boundary
        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)
        boundary_verts = _identify_meshlet_boundary_verts(meshlets, tris_np)

        if self.verbose:
            print(f"  Meshlets: {len(meshlets)}, boundary verts: "
                  f"{len(boundary_verts)} ({len(boundary_verts)/n_v*100:.1f}%)")

        # ------------------------------------------------------------
        # QEM pass 1: interior (boundary protected)
        # ------------------------------------------------------------
        if self.verbose:
            print("  QEM interior (boundary protected)...")
        interior_qem = progressive_simplify_with_records(
            verts_np, tris_np,
            protected_vertices=boundary_verts,
            target_frac=1.0 / 16)

        interior_compact = encode_ancestry_compact(n_v, interior_qem["records"])
        n_int_rec = len(interior_qem["records"])

        # ------------------------------------------------------------
        # QEM pass 2: boundary subgraph (v2c and v2d)
        # ------------------------------------------------------------
        if self.variant in ('v2c', 'v2d'):
            if self.verbose:
                print("  QEM boundary subgraph...")
            boundary_qem = simplify_boundary_subgraph(
                verts_np, tris_np, boundary_verts, target_frac=0.25)
            boundary_compact = encode_ancestry_compact(
                n_v, boundary_qem["records"])
            n_bnd_rec = len(boundary_qem["records"])
        else:
            boundary_compact = None
            n_bnd_rec = 0

        # ------------------------------------------------------------
        # LOD thresholds
        # ------------------------------------------------------------
        lod_int_thresh = []
        lod_bnd_thresh = []
        for k in range(self.n_lod_levels):
            # `alive = collapse_step >= threshold`.
            # threshold=0 means every collapsed vert qualifies as alive = full mesh
            # threshold=n_records means none of the collapsed verts are alive = max simplified
            # We want LOD 0 = coarsest (most simplified), LOD (n-1) = full mesh.
            # So LOD 0 → threshold = n_records, LOD (n-1) → threshold = 0.
            # Decreasing threshold with increasing k.
            ratio = k / max(1, self.n_lod_levels - 1)
            lod_int_thresh.append(int(n_int_rec * (1.0 - ratio)))
            lod_bnd_thresh.append(int(n_bnd_rec * (1.0 - ratio)))

        # ------------------------------------------------------------
        # Position encoding
        # ------------------------------------------------------------
        use_chain_delta = self.variant == 'v2d'
        direct_parent_arr = interior_compact["direct_parent"]

        if self.variant == 'v2a':
            # Global quant for all verts (same as v1)
            global_codes, g_min, g_range, g_bits = _global_quantize(
                vn, per_coord_err)
            pos_bits = n_v * int(g_bits.sum())
            boundary_pos_bytes = 0
            interior_pos_bytes = 0
        else:
            # v2b/v2c/v2d: global for boundary, local for interior
            bnd_mask = np.zeros(n_v, dtype=bool)
            for v in boundary_verts:
                bnd_mask[v] = True
            bnd_ids = np.where(bnd_mask)[0]
            bnd_pos = vn[bnd_ids]
            if len(bnd_pos) > 0:
                _, g_min_b, g_range_b, g_bits_b = _global_quantize(
                    bnd_pos, per_coord_err)
                bnd_pos_bits = len(bnd_pos) * int(g_bits_b.sum())
            else:
                g_bits_b = np.array([0, 0, 0])
                bnd_pos_bits = 0

            # Interior: per-meshlet local quant, with optional chain-delta (v2d)
            interior_bits = 0
            ml_quant_header = len(meshlets) * (12 + 3) * 8
            for ml_tris in meshlets:
                ml_verts = set()
                for ti in ml_tris:
                    for j in range(3):
                        ml_verts.add(int(tris_np[ti, j]))
                int_ids = [v for v in ml_verts if v not in boundary_verts]
                if not int_ids:
                    continue
                int_pos = vn[int_ids]
                int_codes, l_min, l_range, l_bits = _local_quantize_meshlet(
                    int_pos, per_coord_err)

                if not use_chain_delta:
                    interior_bits += len(int_ids) * int(l_bits.sum())
                else:
                    # v2d: split into roots and children based on direct_parent
                    int_set = set(int_ids)
                    vid_to_idx = {v: i for i, v in enumerate(int_ids)}
                    # 1-bit root-flag per vertex
                    interior_bits += len(int_ids)
                    root_codes = []
                    deltas = []
                    for v in int_ids:
                        parent = int(direct_parent_arr[v])
                        if parent != v and parent in int_set:
                            # child: delta from parent's local code
                            deltas.append(int_codes[vid_to_idx[v]] -
                                           int_codes[vid_to_idx[parent]])
                        else:
                            # root: absolute code
                            root_codes.append(int_codes[vid_to_idx[v]])
                    # Roots: absolute bits
                    if root_codes:
                        interior_bits += len(root_codes) * int(l_bits.sum())
                    # Deltas: per-axis entropy/Golomb bits
                    if deltas:
                        deltas_np = np.asarray(deltas, dtype=np.int64)
                        for d in range(3):
                            col = deltas_np[:, d]
                            ent = _empirical_entropy_bits(col)
                            gol = _exp_golomb_bits(col, k=1)
                            interior_bits += min(ent, gol)

            pos_bits = bnd_pos_bits + interior_bits + ml_quant_header
            boundary_pos_bytes = bnd_pos_bits / 8
            interior_pos_bytes = (interior_bits + ml_quant_header) / 8

        # ------------------------------------------------------------
        # Ancestor bits (entropy-coded for v2d, naive for others)
        # ------------------------------------------------------------
        anc_bit_fn = (estimate_compact_ancestor_bits_entropy
                       if use_chain_delta
                       else estimate_compact_ancestor_bits)
        int_anc_bits = anc_bit_fn(
            interior_compact["collapse_step"],
            interior_compact["direct_parent"],
            n_v)
        if boundary_compact is not None:
            bnd_anc_bits = anc_bit_fn(
                boundary_compact["collapse_step"],
                boundary_compact["direct_parent"],
                n_v)
        else:
            bnd_anc_bits = 0

        # ------------------------------------------------------------
        # Connectivity (per meshlet)
        # ------------------------------------------------------------
        conn_bits = 0
        if self.connectivity == 'fifo':
            for ml_tris in meshlets:
                conn_bits += _amd_fifo_adjacency_bits(ml_tris, tris_np, tri_adj)
        elif self.connectivity == 'amd_gts':
            from .meshlet_wavelet import _amd_gts_real_bits
            for ml_tris in meshlets:
                conn_bits += _amd_gts_real_bits(ml_tris, tris_np, tri_adj)
        else:  # 'gts_est'
            for ml_tris in meshlets:
                conn_bits += _amd_gts_bits(ml_tris, tris_np, tri_adj)

        # ------------------------------------------------------------
        # Total
        # ------------------------------------------------------------
        header_bits = (3 * 4 + 3 * 4 + 3 + 4 + 4) * 8
        total_bits = (header_bits + pos_bits +
                      int_anc_bits + bnd_anc_bits + conn_bits)

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        # ------------------------------------------------------------
        # Per-LOD stats (count verts/tris after redirection)
        # ------------------------------------------------------------
        lod_stats = []
        for k in range(self.n_lod_levels):
            # Apply interior ancestors at this LOD
            int_anc = ancestors_at_lod_compact_batch(
                n_v, lod_int_thresh[k],
                interior_compact["collapse_step"],
                interior_compact["direct_parent"])
            # Apply boundary ancestors (v2c only)
            if boundary_compact is not None:
                bnd_anc = ancestors_at_lod_compact_batch(
                    n_v, lod_bnd_thresh[k],
                    boundary_compact["collapse_step"],
                    boundary_compact["direct_parent"])
                # Compose: interior ancestor first, then boundary ancestor
                combined = bnd_anc[int_anc]
            else:
                combined = int_anc

            # Count alive verts
            alive_ids = np.where(combined == np.arange(n_v))[0]
            # Count non-degenerate tris
            aa = combined[tris_np[:, 0]]
            bb = combined[tris_np[:, 1]]
            cc = combined[tris_np[:, 2]]
            non_degen = int(((aa != bb) & (bb != cc) & (aa != cc)).sum())

            lod_stats.append({
                "n_verts": len(alive_ids),
                "n_tris": non_degen,
                "interior_thresh": lod_int_thresh[k],
                "boundary_thresh": lod_bnd_thresh[k],
            })

        if self.verbose:
            print(f"Approach A v2 ({self.variant}) mv={self.max_verts}:")
            print(f"  Interior records: {n_int_rec}, boundary records: {n_bnd_rec}")
            print(f"  Positions:   {pos_bits/8:>10,.0f} B "
                  f"({pos_bits/total_bits*100:.1f}%)")
            if self.variant != 'v2a':
                print(f"    boundary:  {boundary_pos_bytes:>10,.0f} B")
                print(f"    interior:  {interior_pos_bytes:>10,.0f} B")
            print(f"  Int ancestors: {int_anc_bits/8:>10,.0f} B "
                  f"({int_anc_bits/total_bits*100:.1f}%)")
            if bnd_anc_bits:
                print(f"  Bnd ancestors: {bnd_anc_bits/8:>10,.0f} B "
                      f"({bnd_anc_bits/total_bits*100:.1f}%)")
            print(f"  Connectivity:  {conn_bits/8:>10,.0f} B "
                  f"({conn_bits/total_bits*100:.1f}%)")
            print(f"  Total: {total_bits/8:,.0f} B  BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  LOD breakdown:")
            for k, s in enumerate(lod_stats):
                print(f"    LOD {k}: {s['n_verts']:>6,} v, {s['n_tris']:>6,} t "
                      f"(int_thresh={s['interior_thresh']}, "
                      f"bnd_thresh={s['boundary_thresh']})")

        data = bytes(int(np.ceil(total_bits / 8)))
        result = CompressedModel(data, bpv, bpt)
        result._lod_stats = lod_stats
        result._interior_compact = interior_compact
        result._boundary_compact = boundary_compact
        result._lod_int_thresh = lod_int_thresh
        result._lod_bnd_thresh = lod_bnd_thresh
        return result

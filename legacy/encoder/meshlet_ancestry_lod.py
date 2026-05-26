"""
Approach A: Vertex-Collapse Ancestor Redirection LOD.

Store positions + per-vertex ancestor table (multiple LOD levels).
At runtime (mesh shader), for each triangle:
  - Redirect each vertex via ancestor[V][k] to get its coarser equivalent
  - Skip triangles that became degenerate (two ancestors coincide)
  - Emit remaining triangles with ancestor positions

Crack-free by construction (shared boundary ancestors) AND no holes
(triangles are redirected, not dropped).
"""

import numpy as np
from utils.types import Model, CompressedModel
from ..encoder import Encoder
from utils.qem import progressive_simplify_with_records, compute_ancestors
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts,
)
from .meshlet_wavelet import (
    _to_numpy, _global_quantize, _dequantize_global,
    _identify_meshlet_boundary_verts, _bits_for_error,
    _amd_gts_bits, _amd_packed_bits,
)


def _estimate_position_bits(verts_np, per_coord_err):
    """Size of global quantized positions (crack-free grid)."""
    mn = verts_np.min(axis=0)
    mx = verts_np.max(axis=0)
    rng = mx - mn
    bits = sum(_bits_for_error(r, per_coord_err) for r in rng)
    return len(verts_np) * bits, bits  # total bits, bits per vertex


def _estimate_ancestor_bits(ancestors, n_total):
    """Bits to store ancestor table.

    ancestors: (n_verts, n_lod)
    Strategy: for each LOD level, delta-encode ancestor IDs vs vertex index.
    Most vertices are their own ancestor at higher LODs → delta is 0.
    At LOD 0, many verts share the same ancestor → few unique values.
    """
    n_v, n_lod = ancestors.shape
    total_bits = 0
    for k in range(n_lod):
        col = ancestors[:, k]
        # Bits needed to store log2(max possible ancestor value)
        # Plus 1 bit for "is own ancestor" flag + delta if not
        n_unique = len(np.unique(col))
        # Rough entropy coding: bits per element = log2(n_unique) if uniform
        bits_per_ref = max(1, int(np.ceil(np.log2(n_unique + 1))))
        # Bits = n_v * (1 + bits_per_ref * (fraction not self))
        n_same = int((col == np.arange(n_v)).sum())
        n_diff = n_v - n_same
        # Flag bit + ref for different
        col_bits = n_v + n_diff * bits_per_ref
        total_bits += col_bits
    return total_bits


def decode_at_lod(original_verts, original_tris, ancestors, lod_level,
                   positions_per_lod=None):
    """Simulate GPU decode at target LOD. Returns (verts, tris).

    Args:
        original_verts: (n_v, 3)
        original_tris: (n_t, 3)
        ancestors: (n_v, n_lod) ancestor table
        lod_level: which LOD to decode (0 = coarsest)
        positions_per_lod: optional (n_lod,) array of positions per LOD,
            used when LODs update position of collapsed-to vert.
            If None, uses original_verts.

    Returns:
        out_verts: (n_active, 3) positions of active vertices
        out_tris: (n_emit, 3) triangles in terms of compact indices
    """
    anc = ancestors[:, lod_level]

    # Which original verts are "alive" (self-ancestors)
    alive_mask = anc == np.arange(len(anc))
    alive_ids = np.where(alive_mask)[0]
    # Map global id → compact index
    g2c = {int(v): i for i, v in enumerate(alive_ids)}

    # Position per alive vertex
    if positions_per_lod is not None:
        out_verts = positions_per_lod[lod_level][alive_ids]
    else:
        out_verts = original_verts[alive_ids]

    # Redirect triangles
    out_tris = []
    for ti in range(len(original_tris)):
        a, b, c = int(original_tris[ti, 0]), int(original_tris[ti, 1]), int(original_tris[ti, 2])
        aa, bb, cc = int(anc[a]), int(anc[b]), int(anc[c])
        if aa != bb and bb != cc and aa != cc:
            out_tris.append([g2c[aa], g2c[bb], g2c[cc]])

    return out_verts, np.array(out_tris, dtype=np.int64) if out_tris else np.zeros((0, 3), dtype=np.int64)


class MeshletAncestryLOD(Encoder):
    """Approach A: Ancestor-redirection LOD.

    Fixed 5 LOD levels (1/16, 1/8, 1/4, 1/2, full).
    """

    def __init__(self, max_verts=256, precision_error=0.0005,
                 n_lod_levels=5, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.n_lod_levels = n_lod_levels
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)

        # Global quantize (crack-free)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)
        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        # Generate meshlets on original mesh
        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        # Identify boundary verts (protect in QEM)
        boundary_verts = _identify_meshlet_boundary_verts(meshlets, tris_np)

        if self.verbose:
            print(f"  Meshlets: {len(meshlets)}, boundary verts: "
                  f"{len(boundary_verts)} ({len(boundary_verts)/n_v*100:.1f}%)")
            print("  QEM simplification with records...")

        # Run QEM to 1/16 target
        qem = progressive_simplify_with_records(
            verts_np, tris_np,
            protected_vertices=boundary_verts,
            target_frac=1.0 / 16)

        n_rec = len(qem["records"])
        # LOD levels: LOD 0 = most collapsed, LOD 4 = full
        # lod_n_records[k] = how many records to apply for level k
        lod_n_records = []
        for k in range(self.n_lod_levels):
            # LOD 0: apply all n_rec (most simplified)
            # LOD n_lod-1: apply 0 (full mesh)
            # Geometric progression
            ratio = k / max(1, self.n_lod_levels - 1)
            apply = int(n_rec * (1.0 - ratio))
            lod_n_records.append(apply)

        # Compute ancestor table
        ancestors = compute_ancestors(n_v, qem["records"], lod_n_records)

        # Bit accounting
        # 1. Global header
        total_bits = (3 * 4 + 3 * 4 + 3 + 4 + 4) * 8

        # 2. Vertex positions (global quantization)
        pos_bits = n_v * int(g_bits.sum())

        # 3. Ancestor table
        anc_bits = _estimate_ancestor_bits(ancestors, n_v)

        # 4. Triangles (per meshlet, GTS connectivity)
        conn_bits = 0
        for ml_tris in meshlets:
            conn_bits += _amd_gts_bits(ml_tris, tris_np, tri_adj)

        total_bits += pos_bits + anc_bits + conn_bits

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        # Compute per-LOD vertex and triangle counts (for visualization)
        lod_stats = []
        for k in range(self.n_lod_levels):
            col = ancestors[:, k]
            n_alive = int((col == np.arange(n_v)).sum())
            # Count non-degenerate triangles
            aa = col[tris_np[:, 0]]
            bb = col[tris_np[:, 1]]
            cc = col[tris_np[:, 2]]
            non_degen = ((aa != bb) & (bb != cc) & (aa != cc)).sum()
            lod_stats.append({
                "n_verts": n_alive,
                "n_tris": int(non_degen),
                "n_records_applied": lod_n_records[k],
            })

        if self.verbose:
            print(f"Approach A (Ancestor LOD) mv={self.max_verts}:")
            print(f"  {len(meshlets)} meshlets, {n_rec} QEM records")
            print(f"  Positions:    {pos_bits/8:>10,.0f} B "
                  f"({pos_bits/total_bits*100:.1f}%)")
            print(f"  Ancestors:    {anc_bits/8:>10,.0f} B "
                  f"({anc_bits/total_bits*100:.1f}%)")
            print(f"  Connectivity: {conn_bits/8:>10,.0f} B "
                  f"({conn_bits/total_bits*100:.1f}%)")
            print(f"  Total: {total_bits/8:,.0f} B  BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  LOD breakdown:")
            for k, s in enumerate(lod_stats):
                print(f"    LOD {k}: {s['n_verts']:>6,} verts, "
                      f"{s['n_tris']:>6,} tris "
                      f"(collapses: {s['n_records_applied']})")

        data = bytes(int(np.ceil(total_bits / 8)))
        result = CompressedModel(data, bpv, bpt)
        # Attach debug info for comparison script
        result._lod_stats = lod_stats
        result._ancestors = ancestors
        result._qem = qem
        result._lod_n_records = lod_n_records
        return result

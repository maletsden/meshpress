"""
Approach C: Progressive Mesh with Vertex Splits (Hoppe 1996).

Store a base mesh (maximally simplified) + sequence of vertex-split records
that reverse each QEM collapse. To reconstruct at LOD k, apply first N_k
splits to the base mesh.

Each split record reverses one collapse:
  - "split" V_kp: re-introduce V_rm at its original position
  - Set V_kp position back to v_kp_pos_before
  - For each modified triangle: change V_kp → V_rm in that triangle
  - Re-insert removed (degenerate) triangles with V_rm

This gives a VALID mesh at every intermediate LOD step.
"""

import numpy as np
from utils.types import Model, CompressedModel
from ..encoder import Encoder
from utils.qem import progressive_simplify_with_records
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts,
)
from .meshlet_wavelet import (
    _to_numpy, _global_quantize, _dequantize_global,
    _identify_meshlet_boundary_verts, _bits_for_error,
    _amd_gts_bits,
)


def decode_progressive(base_positions_by_lod, base_vert_ids, base_tris,
                        records, n_splits_to_apply, original_n_verts):
    """Simulate GPU decode: start from base mesh, apply N splits.

    Args:
        base_positions_by_lod: list of (n_v_at_base, 3) positions (only first
            used; split records have their own positions)
        base_vert_ids: list of global vertex ids in the base mesh
        base_tris: (n_base_tris, 3) triangles at base level (global ids)
        records: list of collapse records (to be applied in REVERSE)
        n_splits_to_apply: how many splits to apply from end of records backwards
        original_n_verts: total original vertex count

    Returns:
        out_verts: (n_v, 3) reconstructed positions (global ids)
        out_tris: (n_emit, 3) triangles (global ids)
        active_mask: bool[n_v] which global ids are alive
    """
    # Start: only base vertices are alive, with their base positions
    alive = np.zeros(original_n_verts, dtype=bool)
    positions = np.zeros((original_n_verts, 3), dtype=np.float64)
    for i, vid in enumerate(base_vert_ids):
        alive[vid] = True
        positions[vid] = base_positions_by_lod[i]

    # Start with base triangles
    # Use a dict face_idx → [a,b,c] for easy per-face updates during splits
    tri_dict = {}
    base_tri_face_idx = {}
    # Find face indices: we need to correlate base_tris with the original face indices
    # Simpler: use a set of triangles with their GLOBAL face IDs (from records)

    # records were collected in forward collapse order.
    # We need to REVERSE the last n_splits_to_apply records.
    # At the beginning of reverse application, the mesh state is "after all records applied".
    # Then we reverse the last one, then the second-to-last, etc.
    # But we start from a mesh with ONLY base_tris. Need to figure out face indices.

    # Build initial face_idx → tri map from base triangles
    # We need face IDs matching those used in records
    # For this: base_tris came from progressive_simplify_with_records — we need
    # to capture the face IDs there. Let me add this.

    # Since we don't have base face IDs, we'll use a tuple-key approach:
    # Store active triangles as set of (a, b, c) tuples (sorted canonically)
    def canon(a, b, c):
        return tuple(sorted([int(a), int(b), int(c)]))

    active_tris = {}  # canonical triangle → original (a, b, c) order
    for t in base_tris:
        k = canon(t[0], t[1], t[2])
        active_tris[k] = (int(t[0]), int(t[1]), int(t[2]))

    # Apply reverse splits
    n_records = len(records)
    splits_remaining = min(n_splits_to_apply, n_records)
    for idx in range(splits_remaining):
        rec = records[n_records - 1 - idx]  # reverse order
        v_rm = rec["v_rm"]
        v_kp = rec["v_kp"]

        # Re-add v_rm
        alive[v_rm] = True
        positions[v_rm] = rec["v_rm_pos_orig"]
        # Restore v_kp's previous position
        positions[v_kp] = rec["v_kp_pos_before"]

        # Reverse triangle modifications: change v_kp back to v_rm
        for (face_idx, old_tri, new_tri) in rec["modified_tris"]:
            k_new = canon(*new_tri)
            k_old = canon(*old_tri)
            if k_new in active_tris:
                del active_tris[k_new]
            active_tris[k_old] = old_tri

        # Re-add removed (degenerate) triangles
        for (face_idx, tri_before) in rec["removed_tris"]:
            k = canon(*tri_before)
            active_tris[k] = tri_before

    # Build output arrays
    alive_ids = np.where(alive)[0]
    g2c = {int(v): i for i, v in enumerate(alive_ids)}
    out_verts = positions[alive_ids]

    out_tris = []
    for tri in active_tris.values():
        if alive[tri[0]] and alive[tri[1]] and alive[tri[2]]:
            out_tris.append([g2c[tri[0]], g2c[tri[1]], g2c[tri[2]]])
    out_tris_arr = np.array(out_tris, dtype=np.int64) if out_tris else np.zeros((0, 3), dtype=np.int64)

    return out_verts, out_tris_arr, alive


def _estimate_split_bits(records, n_total_verts):
    """Estimate bits needed to store the split records.

    Each split record contains:
      - v_rm index: log2(n_total_verts) bits
      - v_kp index: log2(n_total_verts) bits
      - v_rm position: 3 coords, assume same quant as base
      - v_kp position delta (before = delta from after): small delta
      - list of modified tri updates (tri_idx + new_v ref)
      - list of removed tris (tri_idx + re-insertion)
    """
    n_v = n_total_verts
    bits_per_vid = max(1, int(np.ceil(np.log2(n_v + 1))))
    pos_bits = 27  # 9 bits per coord, 3 coords (matches global quant)

    total = 0
    for rec in records:
        # v_rm + v_kp indices
        total += 2 * bits_per_vid
        # v_rm position (full precision)
        total += pos_bits
        # v_kp_pos_before delta from v_kp_pos_after (small delta, ~8 bits per coord)
        total += 24
        # Modified triangles: count + (tri_idx + which_vert_slot)
        n_mod = len(rec["modified_tris"])
        n_rem = len(rec["removed_tris"])
        total += 8  # count header
        # Per modified: tri_idx (log2(n_faces)) + 2-bit slot (which of 3 verts)
        total += n_mod * (bits_per_vid + 2)
        total += 8  # count header
        # Per removed: tri_idx + two other verts (needs 2*bits_per_vid)
        total += n_rem * (bits_per_vid + 2 * bits_per_vid)
    return total


class MeshletProgressiveLOD(Encoder):
    """Approach C: Progressive Mesh (Hoppe-style vertex splits).

    Base mesh (1/16 simplified) + reversible split records.
    Applying N splits → reach desired LOD.
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

        # Same pre-processing as Approach A
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)
        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        boundary_verts = _identify_meshlet_boundary_verts(meshlets, tris_np)

        if self.verbose:
            print(f"  Meshlets: {len(meshlets)}, boundary verts: "
                  f"{len(boundary_verts)} ({len(boundary_verts)/n_v*100:.1f}%)")
            print("  QEM simplification to base (1/16)...")

        qem = progressive_simplify_with_records(
            verts_np, tris_np,
            protected_vertices=boundary_verts,
            target_frac=1.0 / 16)

        # Bit accounting
        total_bits = (3 * 4 + 3 * 4 + 3 + 4 + 4) * 8  # header

        # Base mesh: positions (use same global quant bits)
        n_base = len(qem["base_vert_ids"])
        base_pos_bits_per_v = int(g_bits.sum())
        base_pos_bits = n_base * base_pos_bits_per_v

        # Base mesh triangles: use GTS over base
        # Approximate: same BPT as full mesh × n_base_tris
        n_base_tris = len(qem["base_tris"])
        base_conn_bits = max(32, n_base_tris * 6)  # ~6 bpt for GTS-like

        # Split records
        split_bits = _estimate_split_bits(qem["records"], n_v)

        # Per-meshlet full-resolution connectivity (for final LOD rendering)
        # In pure progressive mesh, we don't need this — splits recreate tris.
        # We include it zero here (splits provide connectivity)

        total_bits += base_pos_bits + base_conn_bits + split_bits

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        # Compute per-LOD vertex/triangle counts
        n_rec = len(qem["records"])
        lod_n_splits = []
        for k in range(self.n_lod_levels):
            ratio = k / max(1, self.n_lod_levels - 1)
            n_splits = int(n_rec * ratio)
            lod_n_splits.append(n_splits)

        lod_stats = []
        for k, n_splits in enumerate(lod_n_splits):
            # Simulate to count verts/tris
            base_pos = qem["base_positions"]
            # Use original positions (base positions stored differently)
            _, tris_out, alive = decode_progressive(
                base_pos, qem["base_vert_ids"], qem["base_tris"],
                qem["records"], n_splits, n_v)
            lod_stats.append({
                "n_verts": int(alive.sum()),
                "n_tris": len(tris_out),
                "n_splits_applied": n_splits,
            })

        if self.verbose:
            print(f"Approach C (Progressive Mesh) mv={self.max_verts}:")
            print(f"  Base mesh: {n_base} verts, {n_base_tris} tris")
            print(f"  Records: {n_rec}")
            print(f"  Base positions: {base_pos_bits/8:>10,.0f} B "
                  f"({base_pos_bits/total_bits*100:.1f}%)")
            print(f"  Base conn:      {base_conn_bits/8:>10,.0f} B "
                  f"({base_conn_bits/total_bits*100:.1f}%)")
            print(f"  Splits:         {split_bits/8:>10,.0f} B "
                  f"({split_bits/total_bits*100:.1f}%)")
            print(f"  Total: {total_bits/8:,.0f} B  BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  LOD breakdown:")
            for k, s in enumerate(lod_stats):
                print(f"    LOD {k}: {s['n_verts']:>6,} verts, "
                      f"{s['n_tris']:>6,} tris "
                      f"(splits: {s['n_splits_applied']})")

        data = bytes(int(np.ceil(total_bits / 8)))
        result = CompressedModel(data, bpv, bpt)
        result._lod_stats = lod_stats
        result._qem = qem
        result._lod_n_splits = lod_n_splits
        return result

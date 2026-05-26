"""Pre-encode mesh sanitation: merge coincident verts, drop degenerate
and duplicate triangles.

Stanford Lucy and similar large scans contain stitched coincident
vertices and zero-area triangles that break strip-based encoders
(ParaDelta v5 strip_traversal in particular: a triangle whose vertex
set is a subset of the previous triangle has no "new" vertex and
crashes the StopIteration walker).
"""
from __future__ import annotations

import numpy as np

from utils.types import Model, Vertex, Triangle


def clean_mesh(model: Model, tol_rel: float = 1e-7,
               verbose: bool = False) -> tuple[Model, dict]:
    """Return (cleaned Model, stats dict).

    Steps:
      1. Voxel-snap vertices at tol_rel * bbox_extent and merge equal
         buckets (first-occurrence position kept as representative).
      2. Remap triangle indices through the merge.
      3. Drop triangles with any repeated index (topologically
         degenerate: a==b, b==c, or a==c).
      4. Drop duplicate triangles (same unordered index triple).
    """
    if not model.vertices or not model.triangles:
        return model, dict(merged_verts=0, dropped_degen=0, dropped_dup=0)

    verts = np.asarray([(v.x, v.y, v.z) for v in model.vertices],
                       dtype=np.float64)
    tris = np.asarray([(t.a, t.b, t.c) for t in model.triangles],
                      dtype=np.int64)
    n_v0, n_t0 = len(verts), len(tris)

    ext = float((verts.max(0) - verts.min(0)).max())
    tol = max(ext * tol_rel, 1e-12)
    keys = np.round(verts / tol).astype(np.int64)

    _, first_idx, inv = np.unique(
        keys, axis=0, return_index=True, return_inverse=True)
    new_verts = verts[first_idx]
    tris2 = inv[tris]

    mask = ((tris2[:, 0] != tris2[:, 1]) &
            (tris2[:, 1] != tris2[:, 2]) &
            (tris2[:, 0] != tris2[:, 2]))
    n_degen = int((~mask).sum())
    tris2 = tris2[mask]

    sorted_tris = np.sort(tris2, axis=1)
    _, u_idx = np.unique(sorted_tris, axis=0, return_index=True)
    u_idx = np.sort(u_idx)
    n_dup = len(tris2) - len(u_idx)
    tris2 = tris2[u_idx]

    out = Model()
    out.vertices = [Vertex(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    for p in new_verts]
    out.triangles = [Triangle(a=int(t[0]), b=int(t[1]), c=int(t[2]))
                     for t in tris2]

    stats = dict(
        merged_verts=n_v0 - len(new_verts),
        dropped_degen=n_degen,
        dropped_dup=n_dup,
        n_v_before=n_v0, n_t_before=n_t0,
        n_v_after=len(new_verts), n_t_after=len(tris2),
    )
    if verbose:
        print(f"  clean_mesh: verts {n_v0:,} -> {stats['n_v_after']:,} "
              f"(merged {stats['merged_verts']:,}), "
              f"tris {n_t0:,} -> {stats['n_t_after']:,} "
              f"(degen {n_degen:,}, dup {n_dup:,})")
    return out, stats
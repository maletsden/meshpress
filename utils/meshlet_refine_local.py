"""Stage-A: local-search refinement of an initial meshlet partition.

Iterates over boundary triangles (tris adjacent to a different meshlet)
and moves each one to its best-cost neighbor meshlet if that strictly
reduces total cost. Preserves:
    - max_tris / max_verts caps
    - per-meshlet dual-graph connectivity (no orphan splits)

Cost is the joint-cost feature sum from `meshlet_gen_joint`, evaluated
incrementally on the affected pair of meshlets only.

Init from any partition (joint, spiral, greedy). No NN, no training.
"""

import numpy as np

from utils.meshlet_gen_joint import (
    DEFAULT_FEATURE_NORMS, DEFAULT_WEIGHTS, _znorm,
)


# ============================================================
# Per-meshlet stats for cost evaluation
# ============================================================

def _meshlet_stats(tri_list, tris_np, tri_adj, face_normals, verts_np):
    """Compute features used by joint cost for a whole meshlet."""
    if not tri_list:
        return None
    tri_set = set(tri_list)

    # Vertex set
    verts = set()
    for ti in tri_list:
        for j in range(3):
            verts.add(int(tris_np[ti, j]))

    # Boundary edges
    edge_count = {}
    for ti in tri_list:
        a, b, c = int(tris_np[ti, 0]), int(tris_np[ti, 1]), int(tris_np[ti, 2])
        for (u, v) in ((a, b), (b, c), (c, a)):
            e = (min(u, v), max(u, v))
            edge_count[e] = edge_count.get(e, 0) + 1
    n_bnd_edges = sum(1 for c in edge_count.values() if c == 1)

    # Avg normal
    if face_normals is not None:
        avg_n = face_normals[tri_list].astype(np.float64).mean(axis=0)
    else:
        avg_n = np.zeros(3)

    # Plane residual: PCA 3rd eigval sqrt
    if len(verts) >= 4:
        P = verts_np[list(verts)]
        P = P - P.mean(axis=0, keepdims=True)
        cov = P.T @ P / max(1, len(P) - 1)
        eigs = np.linalg.eigvalsh(cov)
        plane_resid = float(np.sqrt(max(0.0, eigs[0])))
    else:
        plane_resid = 0.0

    return {
        "tri_set": tri_set,
        "verts": verts,
        "n_bnd_edges": n_bnd_edges,
        "avg_normal": avg_n,
        "plane_resid": plane_resid,
        "tri_count": len(tri_list),
    }


def _meshlet_cost(stats, weights, feature_norms):
    """BPV-aligned cost for a whole meshlet (lower is better).

    Captures only the two effects that move BPV in MeshletParaDelta:
        1. plane_resid: residuals (interior bits) scale with surface
           "thickness" perpendicular to best-fit plane.
        2. boundary perimeter / sqrt(area): boundary verts cost more
           per-vert than interior; perimeter/sqrt(area) is the classic
           isoperimetric ratio of a 2D patch.
    Other joint-cost features (normal_sim, shared_edges, bfs_depth) are
    per-step heuristics and don't aggregate cleanly to BPV.
    """
    if stats is None or stats["tri_count"] == 0:
        return 0.0
    n_tris = stats["tri_count"]
    n_bnd = stats["n_bnd_edges"]
    plane = stats["plane_resid"]

    # Interior-bit term: tris × plane_resid (rough proxy for total residual)
    interior = n_tris * plane * 1e3   # scale plane_resid back to comparable units

    # Boundary-bit term: perimeter / sqrt(tri_count) — isoperimetric
    perim_iso = n_bnd / max(1.0, np.sqrt(n_tris))

    return float(interior + perim_iso)


# ============================================================
# Connectivity check: removing tri from meshlet keeps it connected
# ============================================================

def _stays_connected_after_remove(tri_set, tri_adj, removed):
    """True if (tri_set - {removed}) is still BFS-connected (or empty)."""
    remaining = tri_set - {removed}
    if not remaining:
        return True
    seed = next(iter(remaining))
    seen = {seed}
    stack = [seed]
    while stack:
        cur = stack.pop()
        for nb in tri_adj[cur]:
            if nb in remaining and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(remaining)


# ============================================================
# Public: refinement loop
# ============================================================

def refine_local(meshlets, tris_np, tri_adj, face_normals, verts_np,
                 max_tris=256, max_verts=256,
                 weights=None, feature_norms=None,
                 max_iters=20, min_gain=1e-6, verbose=False,
                 bpv_callback=None, bpv_check_every=200):
    """Local-search refinement.

    bpv_callback(meshlets) -> bpv: optional; if provided, the move loop
        re-evaluates true BPV every `bpv_check_every` accepted moves and
        rolls back if BPV regressed. This catches proxy/BPV divergence.
    """
    """Local-search refinement of a meshlet partition.

    Returns a new meshlets list (same number of meshlets, possibly empty
    ones removed at the end).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if feature_norms is None:
        feature_norms = DEFAULT_FEATURE_NORMS

    # tri -> meshlet_id mapping
    K = len(meshlets)
    tri_to_ml = np.full(len(tris_np), -1, dtype=np.int64)
    for mi, ml in enumerate(meshlets):
        for ti in ml:
            tri_to_ml[ti] = mi

    # Per-meshlet stats cache
    stats = [
        _meshlet_stats(meshlets[mi], tris_np, tri_adj, face_normals, verts_np)
        for mi in range(K)
    ]
    costs = [_meshlet_cost(s, weights, feature_norms) for s in stats]

    def total_cost():
        return float(sum(costs))

    initial = total_cost()

    def current_meshlets():
        out = [[] for _ in range(K)]
        for ti, m in enumerate(tri_to_ml):
            if m >= 0:
                out[int(m)].append(ti)
        return [ml for ml in out if ml]

    bpv_init = bpv_callback(current_meshlets()) if bpv_callback else None
    if verbose:
        s = f"  [refine] init total_cost={initial:.4f}"
        if bpv_init is not None:
            s += f"  bpv={bpv_init:.3f}"
        print(s)
    bpv_best = bpv_init
    snapshot_tri_to_ml = tri_to_ml.copy()
    snapshot_stats = [dict(s) if s is not None else None for s in stats]
    snapshot_costs = list(costs)
    moves_since_check = 0

    for it in range(max_iters):
        moved = 0
        # Find boundary tris: those with at least one neighbor in different meshlet
        for ti in range(len(tris_np)):
            src = int(tri_to_ml[ti])
            if src < 0:
                continue
            # Candidate target meshlets = unique meshlet ids of neighbors (excl src)
            tgt_set = set()
            for nb in tri_adj[ti]:
                m = int(tri_to_ml[nb])
                if m >= 0 and m != src:
                    tgt_set.add(m)
            if not tgt_set:
                continue

            # Connectivity: removing ti from src must not disconnect src
            if not _stays_connected_after_remove(stats[src]["tri_set"], tri_adj, ti):
                continue

            best_tgt = -1
            best_delta = -min_gain   # accept only if strict drop > min_gain
            for tgt in tgt_set:
                # Cap check on target
                tgt_tris = len(stats[tgt]["tri_set"])
                if tgt_tris + 1 > max_tris:
                    continue
                tgt_verts = stats[tgt]["verts"]
                new_v = set(int(v) for v in tris_np[ti]) - tgt_verts
                if len(tgt_verts) + len(new_v) > max_verts:
                    continue

                # Build hypothetical meshlet lists
                src_new = list(stats[src]["tri_set"] - {ti})
                tgt_new = list(stats[tgt]["tri_set"] | {ti})
                s_src = _meshlet_stats(src_new, tris_np, tri_adj,
                                       face_normals, verts_np)
                s_tgt = _meshlet_stats(tgt_new, tris_np, tri_adj,
                                       face_normals, verts_np)
                c_src_new = _meshlet_cost(s_src, weights, feature_norms)
                c_tgt_new = _meshlet_cost(s_tgt, weights, feature_norms)
                delta = (c_src_new + c_tgt_new) - (costs[src] + costs[tgt])
                if delta < best_delta:
                    best_delta = delta
                    best_tgt = tgt
                    cached = (s_src, s_tgt, c_src_new, c_tgt_new)

            if best_tgt < 0:
                continue

            # Commit move
            s_src, s_tgt, c_src_new, c_tgt_new = cached
            stats[src] = s_src
            stats[best_tgt] = s_tgt
            costs[src] = c_src_new
            costs[best_tgt] = c_tgt_new
            tri_to_ml[ti] = best_tgt
            moved += 1
            moves_since_check += 1

            if bpv_callback is not None and moves_since_check >= bpv_check_every:
                bpv_now = bpv_callback(current_meshlets())
                if bpv_now < bpv_best:
                    bpv_best = bpv_now
                    snapshot_tri_to_ml = tri_to_ml.copy()
                    snapshot_stats = [dict(s) if s is not None else None for s in stats]
                    snapshot_costs = list(costs)
                    if verbose:
                        print(f"  [refine] checkpoint accept: bpv={bpv_now:.3f}")
                else:
                    # Rollback
                    tri_to_ml = snapshot_tri_to_ml.copy()
                    stats = [dict(s) if s is not None else None for s in snapshot_stats]
                    costs = list(snapshot_costs)
                    if verbose:
                        print(f"  [refine] checkpoint REJECT: "
                              f"bpv {bpv_now:.3f} >= best {bpv_best:.3f}; rollback")
                    moves_since_check = 0
                    break    # exit current iteration after rollback
                moves_since_check = 0

        if verbose:
            s = f"  [refine] iter={it} moved={moved} total_cost={total_cost():.4f}"
            if bpv_callback is not None and bpv_best is not None:
                s += f"  bpv_best={bpv_best:.3f}"
            print(s)
        if moved == 0:
            break

    # If using BPV callback, snap to best snapshot
    if bpv_callback is not None:
        tri_to_ml = snapshot_tri_to_ml
    final = total_cost()
    if verbose:
        s = f"  [refine] done init={initial:.4f} final={final:.4f} delta={final - initial:+.4f}"
        if bpv_callback is not None:
            s += f"  bpv_init={bpv_init:.3f}  bpv_best={bpv_best:.3f}"
        print(s)

    # Rebuild meshlets list, dropping empties
    out = [[] for _ in range(K)]
    for ti, m in enumerate(tri_to_ml):
        if m >= 0:
            out[int(m)].append(ti)
    out = [ml for ml in out if ml]
    return out

"""
Quadric Error Metrics (QEM) edge collapse for progressive mesh simplification.

Garland & Heckbert 1997: Surface Simplification Using Quadric Error Metrics.

Produces an importance ordering of vertices (coarse-first) that drives
LOD-aware wavelet compression:
  - Vertices surviving to the coarsest level → wavelet base (LOD 0)
  - Later-collapsed vertices → wavelet detail levels (higher LODs)

Usage:
    from utils.qem import progressive_simplify
    result = progressive_simplify(verts_np, tris_np,
                                   target_ratios=[0.125, 0.25, 0.5])
    importance_order = result["importance_order"]
"""

import numpy as np
import heapq


# ------------------------------------------------------------------
# Quadric computation
# ------------------------------------------------------------------

def compute_face_planes(verts, tris):
    """Compute plane equation (a, b, c, d) per face.  ax+by+cz+d=0, |n|=1."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    lengths[lengths < 1e-15] = 1e-15
    n = n / lengths
    d = -np.sum(n * v0, axis=1)
    return np.column_stack([n, d])            # (n_tris, 4)


def compute_vertex_quadrics(verts, tris, planes):
    """Sum per-face quadrics Kp = pp^T to each vertex.  Returns (n_verts, 4, 4)."""
    n_v = len(verts)
    Q = np.zeros((n_v, 4, 4), dtype=np.float64)
    # Kp = outer(p, p) for each face
    Kf = planes[:, :, None] * planes[:, None, :]   # (n_tris, 4, 4)
    for j in range(3):
        np.add.at(Q, tris[:, j], Kf)
    return Q


# ------------------------------------------------------------------
# Edge cost
# ------------------------------------------------------------------

def _edge_cost(Q_sum, v1, v2):
    """Compute collapse cost and optimal position for an edge.
    Returns (cost, optimal_position_3d)."""
    # Try to solve Q_bar * v_bar = [0,0,0,1]^T
    Q4 = Q_sum.copy()
    Q4[3, :] = [0, 0, 0, 1]
    try:
        v_opt = np.linalg.solve(Q4, np.array([0.0, 0.0, 0.0, 1.0]))
        cost = float(v_opt @ Q_sum @ v_opt)
    except np.linalg.LinAlgError:
        # Singular — fall back to best of v1, v2, midpoint
        mid = (v1 + v2) * 0.5
        best_cost = float('inf')
        best_pos = mid
        for candidate in [v1, v2, mid]:
            h = np.append(candidate, 1.0)
            c = float(h @ Q_sum @ h)
            if c < best_cost:
                best_cost = c
                best_pos = candidate
        return max(best_cost, 0.0), best_pos
    return max(cost, 0.0), v_opt[:3]


# ------------------------------------------------------------------
# Adjacency helpers
# ------------------------------------------------------------------

def _build_vert_faces(n_verts, tris):
    """vertex → set of face indices."""
    vf = [set() for _ in range(n_verts)]
    for fi in range(len(tris)):
        for j in range(3):
            vf[tris[fi, j]].add(fi)
    return vf


def _build_vert_neighbors(n_verts, tris):
    """vertex → set of neighboring vertex indices."""
    vn = [set() for _ in range(n_verts)]
    for fi in range(len(tris)):
        a, b, c = int(tris[fi, 0]), int(tris[fi, 1]), int(tris[fi, 2])
        vn[a].update([b, c])
        vn[b].update([a, c])
        vn[c].update([a, b])
    return vn


def _build_edge_face_count(n_verts, tris, face_alive):
    """Build dict: (min_v, max_v) → count of alive faces containing that edge."""
    efc = {}
    for fi in range(len(tris)):
        if not face_alive[fi]:
            continue
        a, b, c = int(tris[fi, 0]), int(tris[fi, 1]), int(tris[fi, 2])
        for e in [(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(a,c), max(a,c))]:
            efc[e] = efc.get(e, 0) + 1
    return efc


def _build_boundary_set(n_verts, vert_nbrs, edge_face_count):
    """Return set of vertices that touch at least one boundary edge (count == 1)."""
    bnd = set()
    for (va, vb), cnt in edge_face_count.items():
        if cnt == 1:
            bnd.add(va)
            bnd.add(vb)
    return bnd


# ------------------------------------------------------------------
# Progressive simplification
# ------------------------------------------------------------------

def progressive_simplify(verts_np, tris_np, target_ratios=None,
                          protected_vertices=None):
    """Simplify mesh progressively using QEM edge collapse.

    Args:
        verts_np: (n_v, 3) float64 vertex positions
        tris_np: (n_t, 3) int64 triangle indices
        target_ratios: list of floats, e.g. [0.125, 0.25, 0.5].
        protected_vertices: set[int] of vertex ids that MUST NOT be collapsed.
            An edge is skipped if either endpoint is protected. Used to
            preserve meshlet boundaries for crack-free LOD.

    Returns dict with:
        importance_order: list[int]  — global vertex indices, coarse-first
            (vertices surviving simplification come first, collapsed ones last)
        collapse_history: list of (removed_vid, kept_vid, new_position)
        lod_snapshots: dict[float_ratio] → (verts_copy, tris_copy, active_verts, active_tris)
    """
    if target_ratios is None:
        target_ratios = [0.125, 0.25, 0.5]
    if protected_vertices is None:
        protected_vertices = set()
    protected_vertices = set(int(v) for v in protected_vertices)

    n_v = len(verts_np)
    n_t = len(tris_np)

    # Mutable copies
    verts = verts_np.copy()
    tris = tris_np.astype(np.int64).copy()

    # Active masks
    vert_alive = np.ones(n_v, dtype=bool)
    face_alive = np.ones(n_t, dtype=bool)
    n_alive_verts = n_v

    # Quadrics
    planes = compute_face_planes(verts, tris)
    Q = compute_vertex_quadrics(verts, tris, planes)

    # Adjacency
    vert_faces = _build_vert_faces(n_v, tris)
    vert_nbrs = _build_vert_neighbors(n_v, tris)

    # Boundary penalty
    BOUNDARY_PENALTY = 1e6

    # Precompute boundary vertices
    edge_fc = _build_edge_face_count(n_v, tris, face_alive)
    boundary_verts = _build_boundary_set(n_v, vert_nbrs, edge_fc)

    # Build edge heap: (cost, counter, v_remove, v_keep, opt_pos)
    # v_remove is always the first of the tuple; skip if either is protected.
    counter = 0
    heap = []
    edge_seen = set()
    for vi in range(n_v):
        for vj in vert_nbrs[vi]:
            edge = (min(vi, vj), max(vi, vj))
            if edge in edge_seen:
                continue
            edge_seen.add(edge)
            # Skip entirely if both endpoints protected (can't collapse either)
            if vi in protected_vertices and vj in protected_vertices:
                continue
            # Order endpoints so the protected one (if any) is v_keep
            if vi in protected_vertices:
                vi, vj = vj, vi
            Q_sum = Q[vi] + Q[vj]
            cost, opt = _edge_cost(Q_sum, verts[vi], verts[vj])
            # If keeping a protected vertex, force opt to its position
            if vj in protected_vertices:
                opt = verts[vj].copy()
                h = np.append(opt, 1.0)
                cost = float(h @ Q_sum @ h)
            elif vi in boundary_verts or vj in boundary_verts:
                cost += BOUNDARY_PENALTY
            heapq.heappush(heap, (cost, counter, vi, vj, opt))
            counter += 1

    # Collapse history
    collapse_order = []   # order in which vertices are removed
    collapse_history = []

    # Snapshot targets (sorted ascending)
    snap_targets = sorted(set(target_ratios))
    snap_counts = {r: max(3, int(n_v * r)) for r in snap_targets}
    snapshots = {}

    def _take_snapshot(ratio):
        active_v = set(np.where(vert_alive)[0])
        active_f = set(np.where(face_alive)[0])
        # Remap vertices to contiguous indices
        old2new = {}
        new_verts = []
        for ov in sorted(active_v):
            old2new[ov] = len(new_verts)
            new_verts.append(verts[ov].copy())
        new_tris = []
        for fi in sorted(active_f):
            a, b, c = int(tris[fi, 0]), int(tris[fi, 1]), int(tris[fi, 2])
            if a in old2new and b in old2new and c in old2new:
                new_tris.append([old2new[a], old2new[b], old2new[c]])
        snapshots[ratio] = {
            "verts": np.array(new_verts, dtype=np.float64),
            "tris": np.array(new_tris, dtype=np.int64) if new_tris else np.zeros((0, 3), dtype=np.int64),
            "n_verts": len(new_verts),
            "n_tris": len(new_tris),
            "active_vert_ids": sorted(active_v),
        }

    # Greedy collapse loop
    while heap and n_alive_verts > max(3, min(snap_counts.values(), default=3)):
        cost, _, vi, vj, opt_pos = heapq.heappop(heap)

        # Skip stale entries
        if not vert_alive[vi] or not vert_alive[vj]:
            continue
        # Check they're still neighbors
        if vj not in vert_nbrs[vi]:
            continue
        # Hard protection: never remove a protected vertex.
        # vi is the one that will be removed (by our convention in the collapse
        # block below). If vi is protected, swap so vj is removed instead —
        # but if both are protected, skip entirely.
        if vi in protected_vertices and vj in protected_vertices:
            continue
        if vi in protected_vertices:
            vi, vj = vj, vi
            # Recompute opt_pos to be on v_keep (now vj, which is protected)
            opt_pos = verts[vj].copy()

        # Check for topology problems: would collapse create non-manifold?
        # Simple check: shared neighbors of vi and vj (excluding each other)
        # must be exactly 2 for an interior edge, 1 for boundary
        shared_nbrs = vert_nbrs[vi] & vert_nbrs[vj] - {vi, vj}
        # Count faces sharing this edge
        edge_faces = vert_faces[vi] & vert_faces[vj]
        n_edge_faces = sum(1 for fi in edge_faces if face_alive[fi])
        if n_edge_faces == 0:
            continue
        # For manifold: shared neighbors == n_edge_faces
        # Allow a small tolerance for non-perfect meshes
        if len(shared_nbrs) > n_edge_faces + 1:
            continue

        # Collapse: remove vi, keep vj
        v_remove = vi
        v_keep = vj

        # Record
        collapse_order.append(v_remove)
        collapse_history.append((v_remove, v_keep, opt_pos.copy()))

        # Update position of kept vertex
        verts[v_keep] = opt_pos

        # Kill removed vertex
        vert_alive[v_remove] = False
        n_alive_verts -= 1

        # Update faces: replace v_remove → v_keep, kill degenerate
        faces_to_update = list(vert_faces[v_remove])
        for fi in faces_to_update:
            if not face_alive[fi]:
                continue
            tri = tris[fi]
            # Replace v_remove with v_keep
            for j in range(3):
                if tri[j] == v_remove:
                    tri[j] = v_keep
            # Check degenerate (two or more identical vertices)
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            if a == b or b == c or a == c:
                face_alive[fi] = False
                # Remove face from vertex adjacency
                for vv in [a, b, c]:
                    vert_faces[vv].discard(fi)
            else:
                # Transfer face from v_remove to v_keep
                vert_faces[v_keep].add(fi)
                vert_faces[v_remove].discard(fi)

        # Update neighbor lists
        for nb in list(vert_nbrs[v_remove]):
            vert_nbrs[nb].discard(v_remove)
            if nb != v_keep:
                vert_nbrs[nb].add(v_keep)
                vert_nbrs[v_keep].add(nb)
        vert_nbrs[v_remove].clear()

        # Update quadric of kept vertex
        Q[v_keep] = Q[v_keep] + Q[v_remove]

        # Update boundary set around v_keep (lazy: just check its edges)
        boundary_verts.discard(v_remove)
        for nb in vert_nbrs[v_keep]:
            if not vert_alive[nb]:
                continue
            e = (min(v_keep, nb), max(v_keep, nb))
            # Count alive faces sharing this edge
            cnt = 0
            for fi in vert_faces[v_keep] & vert_faces[nb]:
                if face_alive[fi]:
                    cnt += 1
            if cnt == 1:
                boundary_verts.add(v_keep)
                boundary_verts.add(nb)
            elif cnt >= 2:
                # Only remove from boundary if ALL their edges are interior
                pass  # keep existing status; full recompute is too expensive

        # Re-insert edges of v_keep with updated costs
        for nb in vert_nbrs[v_keep]:
            if not vert_alive[nb]:
                continue
            # Both protected → can't collapse either: skip
            if v_keep in protected_vertices and nb in protected_vertices:
                continue
            # Ensure v_remove (first) is not protected
            if v_keep in protected_vertices:
                a, b = nb, v_keep
            else:
                a, b = v_keep, nb
            Q_sum = Q[a] + Q[b]
            c, opt = _edge_cost(Q_sum, verts[a], verts[b])
            if b in protected_vertices:
                opt = verts[b].copy()
                h = np.append(opt, 1.0)
                c = float(h @ Q_sum @ h)
            elif a in boundary_verts or b in boundary_verts:
                c += BOUNDARY_PENALTY
            heapq.heappush(heap, (c, counter, a, b, opt))
            counter += 1

        # Take snapshots at target counts
        for ratio in list(snap_targets):
            if ratio not in snapshots and n_alive_verts <= snap_counts[ratio]:
                _take_snapshot(ratio)

    # Take remaining snapshots if we ran out of edges
    for ratio in snap_targets:
        if ratio not in snapshots:
            _take_snapshot(ratio)

    # Build importance ordering:
    # Vertices that survive are most important (come first in ordering).
    # Collapsed vertices are appended in reverse collapse order (last collapsed = second most important).
    surviving = sorted(np.where(vert_alive)[0])
    importance_order = surviving + list(reversed(collapse_order))

    return {
        "importance_order": importance_order,
        "collapse_history": collapse_history,
        "lod_snapshots": snapshots,
        "n_collapsed": len(collapse_order),
    }


# ==============================================================
# QEM variant that records detailed per-collapse modifications
# (for progressive mesh decode and ancestor computation)
# ==============================================================

def progressive_simplify_with_records(verts_np, tris_np,
                                       protected_vertices=None,
                                       target_frac=0.0625):
    """Run QEM down to `target_frac * n_v` vertices, recording
    per-collapse modifications needed for both:
      - Approach A: ancestor tables (who V ends up as at each LOD)
      - Approach C: reverse split records

    Each collapse record contains:
      v_rm: vertex removed
      v_kp: vertex kept (V_rm's neighbors now merge to it)
      v_rm_pos_orig: original position of V_rm (for re-adding in reverse)
      v_kp_pos_before: position of V_kp just before this collapse (for reverse)
      v_kp_pos_after:  new position of V_kp after this collapse
      modified_tris: list of (face_idx, old_tri, new_tri)
                     where tri = (a,b,c) int64 triples
                     these had V_rm, now have V_kp
      removed_tris:  list of (face_idx, tri_before_collapse)
                     these became degenerate (contained both V_rm and V_kp)

    Returns dict with:
      records:         list of collapse records as above
      base_vert_ids:   list of vertex ids surviving at base level (n_target verts)
      base_positions:  np.ndarray (n_target, 3) positions at base level
      base_tris:       np.ndarray (n_base_tris, 3) triangle indices (global V ids)
      original_verts:  np.ndarray (n_v, 3) original positions (for reconstruction verification)
    """
    if protected_vertices is None:
        protected_vertices = set()
    protected_vertices = set(int(v) for v in protected_vertices)

    n_v = len(verts_np)
    n_t = len(tris_np)
    verts = verts_np.copy()
    tris = tris_np.astype(np.int64).copy()

    vert_alive = np.ones(n_v, dtype=bool)
    face_alive = np.ones(n_t, dtype=bool)
    n_alive = n_v
    target = max(3, int(n_v * target_frac))

    planes = compute_face_planes(verts, tris)
    Q = compute_vertex_quadrics(verts, tris, planes)
    vert_faces = _build_vert_faces(n_v, tris)
    vert_nbrs = _build_vert_neighbors(n_v, tris)

    BOUNDARY_PENALTY = 1e6
    edge_fc = _build_edge_face_count(n_v, tris, face_alive)
    bnd_set = _build_boundary_set(n_v, vert_nbrs, edge_fc)

    counter = 0
    heap = []
    seen_edges = set()
    for vi in range(n_v):
        for vj in vert_nbrs[vi]:
            edge = (min(vi, vj), max(vi, vj))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            if vi in protected_vertices and vj in protected_vertices:
                continue
            if vi in protected_vertices:
                vi, vj = vj, vi
            Q_sum = Q[vi] + Q[vj]
            cost, opt = _edge_cost(Q_sum, verts[vi], verts[vj])
            if vj in protected_vertices:
                opt = verts[vj].copy()
                h = np.append(opt, 1.0)
                cost = float(h @ Q_sum @ h)
            elif vi in bnd_set or vj in bnd_set:
                cost += BOUNDARY_PENALTY
            heapq.heappush(heap, (cost, counter, vi, vj, opt))
            counter += 1

    records = []

    while heap and n_alive > target:
        cost, _, vi, vj, opt_pos = heapq.heappop(heap)
        if not vert_alive[vi] or not vert_alive[vj]:
            continue
        if vj not in vert_nbrs[vi]:
            continue
        if vi in protected_vertices and vj in protected_vertices:
            continue
        if vi in protected_vertices:
            vi, vj = vj, vi
            opt_pos = verts[vj].copy()

        shared_nbrs = vert_nbrs[vi] & vert_nbrs[vj] - {vi, vj}
        edge_faces = vert_faces[vi] & vert_faces[vj]
        n_edge_faces = sum(1 for fi in edge_faces if face_alive[fi])
        if n_edge_faces == 0:
            continue
        if len(shared_nbrs) > n_edge_faces + 1:
            continue

        v_rm = vi
        v_kp = vj

        # Record position changes
        v_rm_pos_orig = verts[v_rm].copy()
        v_kp_pos_before = verts[v_kp].copy()
        v_kp_pos_after = opt_pos.copy()

        modified_tris = []
        removed_tris = []

        # Apply changes to triangles
        faces_to_update = list(vert_faces[v_rm])
        for fi in faces_to_update:
            if not face_alive[fi]:
                continue
            old_tri = tuple(int(x) for x in tris[fi])
            new_tri_arr = tris[fi].copy()
            for j in range(3):
                if new_tri_arr[j] == v_rm:
                    new_tri_arr[j] = v_kp
            a, b, c = int(new_tri_arr[0]), int(new_tri_arr[1]), int(new_tri_arr[2])
            if a == b or b == c or a == c:
                # Removed (degenerate)
                face_alive[fi] = False
                for vv in old_tri:
                    vert_faces[vv].discard(fi)
                removed_tris.append((fi, old_tri))
            else:
                # Modified
                tris[fi] = new_tri_arr
                vert_faces[v_kp].add(fi)
                vert_faces[v_rm].discard(fi)
                modified_tris.append((fi, old_tri, (a, b, c)))

        # Update neighbor lists
        for nb in list(vert_nbrs[v_rm]):
            vert_nbrs[nb].discard(v_rm)
            if nb != v_kp:
                vert_nbrs[nb].add(v_kp)
                vert_nbrs[v_kp].add(nb)
        vert_nbrs[v_rm].clear()

        verts[v_kp] = opt_pos
        vert_alive[v_rm] = False
        n_alive -= 1

        Q[v_kp] = Q[v_kp] + Q[v_rm]

        # Boundary update
        bnd_set.discard(v_rm)
        for nb in vert_nbrs[v_kp]:
            if not vert_alive[nb]:
                continue
            cnt = 0
            for fi in vert_faces[v_kp] & vert_faces[nb]:
                if face_alive[fi]:
                    cnt += 1
            if cnt == 1:
                bnd_set.add(v_kp)
                bnd_set.add(nb)

        # Re-insert edges of v_kp
        for nb in vert_nbrs[v_kp]:
            if not vert_alive[nb]:
                continue
            if v_kp in protected_vertices and nb in protected_vertices:
                continue
            if v_kp in protected_vertices:
                a, b = nb, v_kp
            else:
                a, b = v_kp, nb
            Q_sum = Q[a] + Q[b]
            c, opt = _edge_cost(Q_sum, verts[a], verts[b])
            if b in protected_vertices:
                opt = verts[b].copy()
                h = np.append(opt, 1.0)
                c = float(h @ Q_sum @ h)
            elif a in bnd_set or b in bnd_set:
                c += BOUNDARY_PENALTY
            heapq.heappush(heap, (c, counter, a, b, opt))
            counter += 1

        records.append({
            "v_rm": v_rm,
            "v_kp": v_kp,
            "v_rm_pos_orig": v_rm_pos_orig,
            "v_kp_pos_before": v_kp_pos_before,
            "v_kp_pos_after": v_kp_pos_after,
            "modified_tris": modified_tris,
            "removed_tris": removed_tris,
        })

    # Base mesh
    base_vert_ids = sorted(np.where(vert_alive)[0])
    base_positions = verts[base_vert_ids].copy()
    base_tris_list = []
    for fi in range(n_t):
        if face_alive[fi]:
            base_tris_list.append([int(tris[fi, 0]), int(tris[fi, 1]), int(tris[fi, 2])])
    base_tris_arr = np.array(base_tris_list, dtype=np.int64) if base_tris_list else np.zeros((0, 3), dtype=np.int64)

    return {
        "records": records,
        "base_vert_ids": base_vert_ids,
        "base_positions": base_positions,
        "base_tris": base_tris_arr,
        "original_verts": verts_np.copy(),
        "n_total_verts": n_v,
    }


# ==============================================================
# Compact ancestor encoding (for Approach A v2)
# ==============================================================

def encode_ancestry_compact(n_total_verts, records):
    """Build compact ancestor representation.

    Instead of an N×K table, we store per-vertex:
      collapse_step[V] = step at which V was collapsed (or -1 if never)
      direct_parent[V] = vertex V was collapsed INTO directly
                         (one-hop, not full ancestor)

    At decode time, ancestor[V][k] is computed by walking the chain
    direct_parent[V] → direct_parent[direct_parent[V]] → ...
    until we find a vertex V' where collapse_step[V'] >= threshold[k].

    Args:
        n_total_verts: total original vertex count
        records: list of collapse records (from progressive_simplify_with_records)

    Returns:
        dict with:
          collapse_step: (n_v,) int64 — -1 if vertex never collapsed, else the step
          direct_parent: (n_v,) int64 — direct collapse target (self if never collapsed)
    """
    collapse_step = np.full(n_total_verts, -1, dtype=np.int64)
    direct_parent = np.arange(n_total_verts, dtype=np.int64)

    for step, rec in enumerate(records):
        v_rm = rec["v_rm"]
        v_kp = rec["v_kp"]
        collapse_step[v_rm] = step
        direct_parent[v_rm] = v_kp

    return {
        "collapse_step": collapse_step,
        "direct_parent": direct_parent,
    }


def ancestor_at_lod_compact(v, threshold, collapse_step, direct_parent,
                              max_hops=None):
    """Walk collapse chain from v until reaching a vertex alive at this LOD.

    A vertex V' is alive at `threshold` if collapse_step[V'] == -1 (never
    collapsed) OR collapse_step[V'] >= threshold.
    """
    cur = int(v)
    if max_hops is None:
        max_hops = 1 << 20
    hops = 0
    while hops < max_hops:
        step = collapse_step[cur]
        if step < 0 or step >= threshold:
            return cur
        cur = int(direct_parent[cur])
        hops += 1
    return cur


def ancestors_at_lod_compact_batch(n_verts, threshold, collapse_step, direct_parent):
    """Compute ancestor[v][threshold] for all v at once (vectorized).

    Uses iterative path compression for efficiency.
    Returns (n_verts,) array of ancestor vertex IDs.
    """
    # Flag: is each vertex alive at this LOD?
    alive = (collapse_step < 0) | (collapse_step >= threshold)
    # Start each vertex as itself
    current = np.arange(n_verts, dtype=np.int64)
    # Where current is not alive, jump to direct_parent. Iterate until fixed.
    for _ in range(30):  # chain depth rarely exceeds log2(n_verts) ~ 20
        need_jump = ~alive[current]
        if not need_jump.any():
            break
        current[need_jump] = direct_parent[current[need_jump]]
    return current


def estimate_compact_ancestor_bits(collapse_step, direct_parent, n_total_verts):
    """Estimate bit cost of compact encoding.

    Per-vertex flag bit (1 = collapsed, 0 = not).
    For collapsed vertices:
      - collapse_step: log2(n_records) bits
      - direct_parent: log2(n_total_verts) bits (can be smaller with delta)
    """
    n_v = n_total_verts
    n_collapsed = int((collapse_step >= 0).sum())
    n_records = int(collapse_step.max() + 1) if n_collapsed > 0 else 0

    flag_bits = n_v  # 1 per vertex

    if n_collapsed > 0:
        step_bits = max(1, int(np.ceil(np.log2(n_records + 1))))
        # Parent: use delta-from-self, measured entropy
        parents = direct_parent[collapse_step >= 0]
        sources = np.where(collapse_step >= 0)[0]
        deltas = np.abs(parents - sources)
        max_delta = int(deltas.max()) if len(deltas) > 0 else 0
        parent_bits = max(1, int(np.ceil(np.log2(max_delta + 2)))) + 1  # +1 for sign
        # Cap at worst case: direct int
        parent_bits = min(parent_bits,
                           max(1, int(np.ceil(np.log2(n_v + 1)))))

        payload_bits = n_collapsed * (step_bits + parent_bits)
    else:
        payload_bits = 0

    return flag_bits + payload_bits


def _empirical_entropy_bits(values, overhead=64):
    """Total bit count for a stream via Shannon entropy + small fixed overhead.

    Assumes an adaptive arithmetic/range coder that learns the distribution
    online (no explicit frequency table needed).
    Returns an idealized lower bound + small model overhead.
    """
    if values is None or len(values) == 0:
        return 0
    values = np.asarray(values)
    _, counts = np.unique(values, return_counts=True)
    n = counts.sum()
    if len(counts) == 1:
        return overhead + 16  # constant stream
    probs = counts.astype(np.float64) / n
    ent = -np.sum(probs * np.log2(probs))
    return int(np.ceil(n * ent)) + overhead


def _exp_golomb_bits(signed_values, k=0):
    """Bit count for signed values encoded with exp-Golomb of order k.

    For signed value x, map to unsigned u = 2*|x| - (1 if x > 0 else 0).
    Exp-Golomb (order k) of u uses:
      2 * floor(log2((u >> k) + 1)) + 1 + k bits
    """
    if len(signed_values) == 0:
        return 0
    arr = np.abs(np.asarray(signed_values, dtype=np.int64))
    u = 2 * arr  # non-negatives take same bits as (2*|x|)-1
    shifted = u >> k
    lb = np.where(shifted > 0, np.floor(np.log2(shifted + 1)).astype(np.int64), 0)
    bits = 2 * lb + 1 + k
    return int(bits.sum())


def estimate_compact_ancestor_bits_entropy(collapse_step, direct_parent,
                                             n_total_verts):
    """Entropy-coded bit estimate (idealized).

      - Flag bit per vertex (1 bit each)
      - Collapse step: nearly uniform → fixed log2(n_records) bits
      - Parent delta: heavy-tailed → exp-Golomb or entropy-coded
    """
    n_v = n_total_verts
    collapsed_mask = collapse_step >= 0
    n_collapsed = int(collapsed_mask.sum())

    flag_bits = n_v

    if n_collapsed == 0:
        return flag_bits

    n_records = int(collapse_step.max() + 1)
    step_bits_per = max(1, int(np.ceil(np.log2(n_records + 1))))
    step_bits = n_collapsed * step_bits_per

    sources = np.where(collapsed_mask)[0]
    parents = direct_parent[collapsed_mask]
    deltas = (parents.astype(np.int64) - sources.astype(np.int64))
    # Use min(entropy, exp-Golomb) — entropy with adaptive coding,
    # or Golomb with static model. Whichever is smaller.
    ent_bits = _empirical_entropy_bits(deltas)
    golomb_bits = _exp_golomb_bits(deltas, k=2)  # k tuned for typical distribution
    parent_bits = min(ent_bits, golomb_bits)

    return flag_bits + step_bits + parent_bits


# ==============================================================
# Ancestor computation (for Approach A)
# ==============================================================

def compute_ancestors(n_total_verts, records, lod_n_records):
    """Compute ancestor[V][k] for each LOD level k.

    Args:
        n_total_verts: total number of original vertices
        records: list of collapse records (from progressive_simplify_with_records)
        lod_n_records: list of len(LOD_levels), each entry = number of collapse
            records to "apply" for that LOD. LOD 0 uses most records (most collapsed),
            LOD K uses 0 (full mesh). Order: [LOD 0, LOD 1, ..., LOD K].

    Returns:
        ancestors: np.ndarray (n_total_verts, n_lod) — ancestors[v][k] = final vertex
                    that v collapses to if we apply first lod_n_records[k] collapses.
    """
    n_lod = len(lod_n_records)
    ancestors = np.arange(n_total_verts, dtype=np.int64)[:, None].repeat(n_lod, axis=1)

    for k in range(n_lod):
        # Start each vertex as itself, walk collapse chain up to record k
        n_apply = lod_n_records[k]
        # Build a "collapse to" table considering first n_apply records
        collapse_to = np.arange(n_total_verts, dtype=np.int64)
        for step in range(n_apply):
            rec = records[step]
            v_rm = rec["v_rm"]
            v_kp = rec["v_kp"]
            # v_rm is now aliased to v_kp
            collapse_to[v_rm] = v_kp

        # Compute transitive closure per vertex
        for v in range(n_total_verts):
            cur = v
            # Walk chain until fixed point
            visited = 0
            while collapse_to[cur] != cur and visited < n_total_verts:
                cur = collapse_to[cur]
                visited += 1
            ancestors[v, k] = cur

    return ancestors

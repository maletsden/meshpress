"""
Meshlet generation for mesh compression.
Option 3: Greedy DFS strips → cut into meshlets
Option 4: Greedy region growing with geometric cost
"""

import numpy as np
from collections import deque
import heapq


def build_adjacency(tris_np):
    """Build triangle adjacency using numpy. Returns tri_adj[i] = [neighbor indices]."""
    n = len(tris_np)
    t = tris_np.astype(np.int64)

    # Create all 3 edges per triangle: (min_v, max_v)
    pairs = [(0, 1), (1, 2), (0, 2)]
    all_v0 = np.concatenate([np.minimum(t[:, a], t[:, b]) for a, b in pairs])
    all_v1 = np.concatenate([np.maximum(t[:, a], t[:, b]) for a, b in pairs])
    all_tri = np.concatenate([np.arange(n, dtype=np.int64)] * 3)

    # Edge key (int64 to avoid overflow)
    max_v = int(t.max()) + 1
    keys = all_v0 * np.int64(max_v) + all_v1

    # Sort by edge key
    order = np.argsort(keys, kind='mergesort')
    sk = keys[order]
    st = all_tri[order]

    # Consecutive entries with same key = triangles sharing an edge
    same = sk[1:] == sk[:-1]
    # Also check that the NEXT entry is different (to handle edges with 3+ tris correctly)
    # For edges with exactly 2 tris: same[i] = True, same[i+1] = False (or end)
    t0 = st[:-1][same]
    t1 = st[1:][same]

    # Build adjacency lists
    tri_adj = [[] for _ in range(n)]
    for i in range(len(t0)):
        a, b = int(t0[i]), int(t1[i])
        tri_adj[a].append(b)
        tri_adj[b].append(a)

    return tri_adj


def compute_face_normals(verts, tris):
    """Returns (N_tris, 3) unit normals."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    return n / (lens + 1e-12)


def compute_face_centroids(verts, tris):
    """Returns (N_tris, 3) centroids."""
    return (verts[tris[:, 0]] + verts[tris[:, 1]] + verts[tris[:, 2]]) / 3


# ============================================================
# Option 4: Greedy region growing
# ============================================================

def generate_meshlets_greedy(tris_np, tri_adj, face_normals, face_centroids,
                             max_tris=128, max_verts=256,
                             w_normal=0.6, w_edges=0.3, w_dist=0.1):
    """Greedy region growing with geometric cost function."""
    n = len(tris_np)
    visited = np.zeros(n, dtype=bool)
    meshlets = []

    for seed in range(n):
        if visited[seed]:
            continue

        visited[seed] = True
        ml_tris = [seed]
        ml_verts = set(int(v) for v in tris_np[seed])
        ml_set = {seed}
        avg_normal = face_normals[seed].copy()
        avg_centroid = face_centroids[seed].copy()

        pq = []  # (-score, tri_idx)
        counter = 0  # tiebreaker for heapq

        def push_neighbors(tri):
            nonlocal counter
            norm_len = np.linalg.norm(avg_normal)
            unit_normal = avg_normal / (norm_len + 1e-12)
            for nb in tri_adj[tri]:
                if not visited[nb]:
                    n_sim = (np.dot(face_normals[nb], unit_normal) + 1) / 2
                    shared = sum(1 for nn in tri_adj[nb] if nn in ml_set) / 3.0
                    dist = np.linalg.norm(face_centroids[nb] - avg_centroid)
                    d_score = 1.0 / (1.0 + dist * 10)
                    score = w_normal * n_sim + w_edges * shared + w_dist * d_score
                    heapq.heappush(pq, (-score, counter, nb))
                    counter += 1

        push_neighbors(seed)

        while ml_tris and len(ml_tris) < max_tris and pq:
            _, _, cand = heapq.heappop(pq)
            if visited[cand]:
                continue

            # Check max_verts constraint
            new_verts = set(int(v) for v in tris_np[cand]) - ml_verts
            if len(ml_verts) + len(new_verts) > max_verts:
                continue

            visited[cand] = True
            ml_tris.append(cand)
            ml_set.add(cand)
            ml_verts.update(new_verts)

            cnt = len(ml_tris)
            avg_normal = (avg_normal * (cnt - 1) + face_normals[cand]) / cnt
            avg_centroid = (avg_centroid * (cnt - 1) + face_centroids[cand]) / cnt

            push_neighbors(cand)

        meshlets.append(ml_tris)

    return meshlets


# ============================================================
# Option 3: Greedy DFS strips → cut into meshlets
# ============================================================

def generate_strips_greedy(tris_np, tri_adj):
    """Generate long triangle strips via greedy DFS with min-degree heuristic."""
    n = len(tris_np)
    visited = np.zeros(n, dtype=bool)
    strips = []

    def extend(start):
        """Extend strip from start using min-degree neighbor selection."""
        chain = [start]
        cur = start
        while True:
            best = None
            best_deg = 999
            for nb in tri_adj[cur]:
                if not visited[nb]:
                    deg = sum(1 for nn in tri_adj[nb] if not visited[nn])
                    if deg < best_deg:
                        best_deg = deg
                        best = nb
            if best is None:
                break
            visited[best] = True
            chain.append(best)
            cur = best
        return chain

    for seed in range(n):
        if visited[seed]:
            continue
        visited[seed] = True

        # Extend forward
        forward = extend(seed)
        # Extend backward (temporarily unmark seed's visited)
        backward = extend(seed)  # seed is already visited, so this explores other direction

        # Actually: backward won't work since seed is visited.
        # We need to extend from seed in both directions before marking forward.
        # Simpler: just do forward, then try backward from seed.
        # Since seed is visited, backward looks at seed's unvisited neighbors.
        # But we already explored them in forward. Let me fix this.
        strip = forward  # forward already includes seed

        # Try to extend backward: look at seed's unvisited neighbors not in forward
        cur = seed
        back = []
        while True:
            best = None
            best_deg = 999
            for nb in tri_adj[cur]:
                if not visited[nb]:
                    deg = sum(1 for nn in tri_adj[nb] if not visited[nn])
                    if deg < best_deg:
                        best_deg = deg
                        best = nb
            if best is None:
                break
            visited[best] = True
            back.append(best)
            cur = best

        strip = list(reversed(back)) + strip
        strips.append(strip)

    return strips


def meshlets_from_strips(strips, max_tris=128):
    """Cut strips into fixed-size meshlets."""
    meshlets = []
    for strip in strips:
        for i in range(0, len(strip), max_tris):
            chunk = strip[i:i + max_tris]
            if chunk:
                meshlets.append(chunk)
    return meshlets


# ============================================================
# BFS traversal within meshlet
# ============================================================

def meshlet_bfs(meshlet_tris, tri_adj):
    """
    BFS within meshlet's local dual graph.
    Returns [(tri_idx, parent_tri_idx_or_None), ...].
    """
    ml_set = set(meshlet_tris)
    visited = set()
    result = []
    queue = deque()

    for seed in meshlet_tris:
        if seed in visited:
            continue
        visited.add(seed)
        queue.append(seed)
        result.append((seed, None))

        while queue:
            cur = queue.popleft()
            for nb in tri_adj[cur]:
                if nb in ml_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    result.append((nb, cur))

    return result


# ============================================================
# Max-verts-primary meshlet generation
# ============================================================

def generate_meshlets_by_verts(tris_np, tri_adj, face_normals, face_centroids,
                               max_verts=256, w_normal=0.6, w_edges=0.3, w_dist=0.1):
    """Greedy region growing with max_verts as PRIMARY constraint."""
    n = len(tris_np)
    visited = np.zeros(n, dtype=bool)
    meshlets = []

    for seed in range(n):
        if visited[seed]:
            continue

        visited[seed] = True
        ml_tris = [seed]
        ml_verts = set(int(v) for v in tris_np[seed])
        ml_set = {seed}
        avg_normal = face_normals[seed].copy()
        avg_centroid = face_centroids[seed].copy()

        pq = []
        counter = 0

        def push_neighbors(tri):
            nonlocal counter
            norm_len = np.linalg.norm(avg_normal)
            unit_normal = avg_normal / (norm_len + 1e-12)
            for nb in tri_adj[tri]:
                if not visited[nb]:
                    n_sim = (np.dot(face_normals[nb], unit_normal) + 1) / 2
                    shared = sum(1 for nn in tri_adj[nb] if nn in ml_set) / 3.0
                    dist = np.linalg.norm(face_centroids[nb] - avg_centroid)
                    d_score = 1.0 / (1.0 + dist * 10)
                    score = w_normal * n_sim + w_edges * shared + w_dist * d_score
                    heapq.heappush(pq, (-score, counter, nb))
                    counter += 1

        push_neighbors(seed)

        while pq and len(ml_verts) < max_verts:
            _, _, cand = heapq.heappop(pq)
            if visited[cand]:
                continue

            new_verts = set(int(v) for v in tris_np[cand]) - ml_verts
            if len(ml_verts) + len(new_verts) > max_verts:
                continue

            visited[cand] = True
            ml_tris.append(cand)
            ml_set.add(cand)
            ml_verts.update(new_verts)

            cnt = len(ml_tris)
            avg_normal = (avg_normal * (cnt - 1) + face_normals[cand]) / cnt
            avg_centroid = (avg_centroid * (cnt - 1) + face_centroids[cand]) / cnt

            push_neighbors(cand)

        meshlets.append(ml_tris)

    return meshlets


# ============================================================
# EdgeBreaker vertex ordering
# ============================================================

def edgebreaker_vertex_order(meshlet_tris, tris_np, tri_adj):
    """Extract vertex ordering from BFS traversal (EdgeBreaker-like).
    Returns:
        vertex_order: list of global vertex indices in appearance order
        opcodes: list of ('C','L','R','E') per non-root triangle
        n_root_tris: number of root triangles (disconnected components)
    """
    traversal = meshlet_bfs(meshlet_tris, tri_adj)

    vertex_order = []
    seen = set()
    opcodes = []
    n_root = 0

    for tri_idx, parent_idx in traversal:
        tri = [int(tris_np[tri_idx, j]) for j in range(3)]

        if parent_idx is None:
            n_root += 1
            for v in tri:
                if v not in seen:
                    vertex_order.append(v)
                    seen.add(v)
        else:
            parent = [int(tris_np[parent_idx, j]) for j in range(3)]
            shared = set(tri) & set(parent)
            new = [v for v in tri if v not in shared and v not in seen]

            if len(new) == 1:
                opcodes.append('C')
                vertex_order.append(new[0])
                seen.add(new[0])
            elif len(new) == 0:
                # All vertices already seen: L, R, or E
                opcodes.append('L')
            else:
                # Multiple new (shouldn't happen for manifold, but handle)
                for v in new:
                    opcodes.append('C')
                    vertex_order.append(v)
                    seen.add(v)

    # Handle any vertices in meshlet triangles not reached by BFS
    ml_verts = set()
    for ti in meshlet_tris:
        for j in range(3):
            ml_verts.add(int(tris_np[ti, j]))
    for v in sorted(ml_verts):
        if v not in seen:
            vertex_order.append(v)
            seen.add(v)

    return vertex_order, opcodes, n_root


# ============================================================
# LOD-aware meshlet generation
# ============================================================

def generate_meshlets_lod(tris_np, tri_adj, face_normals, face_centroids,
                          importance_rank, max_verts=256,
                          w_normal=0.5, w_edges=0.25, w_dist=0.1, w_importance=0.15):
    """Greedy region growing with importance-coherence bias.

    Args:
        importance_rank: dict or array mapping global vertex id → rank (0 = most important).
            Meshlets will prefer to group vertices with similar importance.
    """
    n = len(tris_np)
    visited = np.zeros(n, dtype=bool)
    meshlets = []

    # Pre-compute per-face average importance rank
    face_imp = np.zeros(n, dtype=np.float64)
    max_rank = max(importance_rank.values()) if isinstance(importance_rank, dict) else len(importance_rank) - 1
    max_rank = max(max_rank, 1)
    for fi in range(n):
        s = 0.0
        for j in range(3):
            v = int(tris_np[fi, j])
            r = importance_rank[v] if isinstance(importance_rank, dict) else importance_rank[v]
            s += r
        face_imp[fi] = s / 3.0 / max_rank   # normalized [0, 1]

    for seed in range(n):
        if visited[seed]:
            continue

        visited[seed] = True
        ml_tris = [seed]
        ml_verts = set(int(v) for v in tris_np[seed])
        ml_set = {seed}
        avg_normal = face_normals[seed].copy()
        avg_centroid = face_centroids[seed].copy()
        avg_imp = face_imp[seed]

        pq = []
        counter = 0

        def push_neighbors(tri):
            nonlocal counter
            norm_len = np.linalg.norm(avg_normal)
            unit_normal = avg_normal / (norm_len + 1e-12)
            for nb in tri_adj[tri]:
                if not visited[nb]:
                    n_sim = (np.dot(face_normals[nb], unit_normal) + 1) / 2
                    shared = sum(1 for nn in tri_adj[nb] if nn in ml_set) / 3.0
                    dist = np.linalg.norm(face_centroids[nb] - avg_centroid)
                    d_score = 1.0 / (1.0 + dist * 10)
                    imp_coherence = 1.0 - abs(face_imp[nb] - avg_imp)
                    score = (w_normal * n_sim + w_edges * shared +
                             w_dist * d_score + w_importance * imp_coherence)
                    heapq.heappush(pq, (-score, counter, nb))
                    counter += 1

        push_neighbors(seed)

        while pq and len(ml_verts) < max_verts:
            _, _, cand = heapq.heappop(pq)
            if visited[cand]:
                continue

            new_verts = set(int(v) for v in tris_np[cand]) - ml_verts
            if len(ml_verts) + len(new_verts) > max_verts:
                continue

            visited[cand] = True
            ml_tris.append(cand)
            ml_set.add(cand)
            ml_verts.update(new_verts)

            cnt = len(ml_tris)
            avg_normal = (avg_normal * (cnt - 1) + face_normals[cand]) / cnt
            avg_centroid = (avg_centroid * (cnt - 1) + face_centroids[cand]) / cnt
            avg_imp = (avg_imp * (cnt - 1) + face_imp[cand]) / cnt

            push_neighbors(cand)

        meshlets.append(ml_tris)

    return meshlets


def reorder_meshlet_vertices_by_importance(meshlet_tris, tris_np, importance_rank):
    """Sort a meshlet's vertices by importance (most important first).

    This ensures that after Haar wavelet decomposition with target_base=32:
      - First 32 vertices (base) = most important = LOD 0
      - Next 32 (level 3 detail) → LOD 1 (64 verts)
      - Next 64 (level 2 detail) → LOD 2 (128 verts)
      - Next 128 (level 1 detail) → LOD 3 (256 verts)

    Args:
        importance_rank: dict or array mapping global vid → rank (0 = most important)

    Returns:
        vertex_order: list of global vertex ids sorted by importance
    """
    ml_verts = set()
    for ti in meshlet_tris:
        for j in range(3):
            ml_verts.add(int(tris_np[ti, j]))

    def rank_of(v):
        return importance_rank[v] if isinstance(importance_rank, dict) else importance_rank[v]

    return sorted(ml_verts, key=rank_of)
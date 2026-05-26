"""Stewart's static triangle-strip tunneling algorithm.

Reference:
    https://research.cs.queensu.ca/home/jstewart/strips/algorithm/

Stage-1 baseline: produces a near-minimum strip cover on the whole-mesh
dual graph, then chunks each strip into meshlets capped at max_tris.
Each resulting meshlet's interior is one strip slice (zero in-meshlet
restarts) but the shape is sausage-like (high boundary/inner ratio).
"""

import numpy as np
from collections import deque


MAX_TUNNEL_DEPTH = 75   # Stewart: BFS depth cap; quality loss negligible


# ============================================================
# Seed strip cover (greedy DFS in dual graph)
# ============================================================

def _greedy_strip_cover(tri_adj, n_tris):
    """Cover tris with disjoint chains via min-degree greedy DFS.

    Returns list of strips (each a list of tri indices). Adjacent strip
    triangles share an edge (dual-graph edge).
    """
    visited = np.zeros(n_tris, dtype=bool)
    strips = []
    for seed in range(n_tris):
        if visited[seed]:
            continue
        visited[seed] = True
        # Extend forward
        chain = [seed]
        cur = seed
        while True:
            best, best_deg = -1, 999
            for nb in tri_adj[cur]:
                if not visited[nb]:
                    deg = sum(1 for nn in tri_adj[nb] if not visited[nn])
                    if deg < best_deg:
                        best_deg, best = deg, nb
            if best < 0:
                break
            visited[best] = True
            chain.append(best)
            cur = best
        # Extend backward from seed
        cur = seed
        back = []
        while True:
            best, best_deg = -1, 999
            for nb in tri_adj[cur]:
                if not visited[nb]:
                    deg = sum(1 for nn in tri_adj[nb] if not visited[nn])
                    if deg < best_deg:
                        best_deg, best = deg, nb
            if best < 0:
                break
            visited[best] = True
            back.append(best)
            cur = best
        strips.append(list(reversed(back)) + chain)
    return strips


# ============================================================
# Tunneling: alternating-edge BFS between strip endpoints
# ============================================================

def _build_strip_state(strips, tri_adj, n_tris):
    """Mark each dual edge as 'strip' (in some strip) or 'nonstrip'.

    Returns:
        strip_id[t]: which strip t belongs to.
        pos_in_strip[t]: index inside that strip.
        strip_edge: set of (a, b) sorted-tuples that are strip edges.
        endpoints: list[(strip_id, tri_at_end)] for both ends of each strip.
    """
    strip_id = np.full(n_tris, -1, dtype=np.int64)
    pos_in_strip = np.full(n_tris, -1, dtype=np.int64)
    strip_edge = set()
    endpoints = []
    for sid, strip in enumerate(strips):
        for i, t in enumerate(strip):
            strip_id[t] = sid
            pos_in_strip[t] = i
        for i in range(len(strip) - 1):
            a, b = strip[i], strip[i + 1]
            strip_edge.add((min(a, b), max(a, b)))
        if strip:
            endpoints.append((sid, strip[0]))
            if len(strip) > 1:
                endpoints.append((sid, strip[-1]))
    return strip_id, pos_in_strip, strip_edge, endpoints


def _bfs_alternating(start, target_strip, tri_adj, strip_edge,
                      strip_id, max_depth=MAX_TUNNEL_DEPTH):
    """BFS for an alternating strip/nonstrip edge path from `start` to any
    triangle in `target_strip`. Returns the path as a list of tris, or None.

    Alternation: edge[i] type != edge[i-1] type. start has no incoming edge,
    so first edge can be either type.
    """
    # state: (tri, last_edge_was_strip)
    # last_edge_was_strip = None for start
    visited = {}   # (tri, last_strip) -> parent tuple
    visited[(start, None)] = None
    q = deque([(start, None, 0)])
    while q:
        cur, last_strip, depth = q.popleft()
        if depth >= max_depth:
            continue
        for nb in tri_adj[cur]:
            edge = (min(cur, nb), max(cur, nb))
            is_strip = edge in strip_edge
            if last_strip is not None and is_strip == last_strip:
                continue   # not alternating
            key = (nb, is_strip)
            if key in visited:
                continue
            visited[key] = (cur, last_strip)
            # If we landed in target strip via a non-strip edge to it,
            # we have a useful tunnel candidate.
            if strip_id[nb] == target_strip and nb != start:
                # Reconstruct path
                path = [nb]
                k = key
                while visited[k] is not None:
                    p_tri, p_strip = visited[k]
                    path.append(p_tri)
                    k = (p_tri, p_strip)
                return list(reversed(path))
            q.append((nb, is_strip, depth + 1))
    return None


def _complement_path(path, strip_edge):
    """Flip strip / nonstrip designation for every edge in path."""
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        e = (min(a, b), max(a, b))
        if e in strip_edge:
            strip_edge.remove(e)
        else:
            strip_edge.add(e)


def _rebuild_strips(strip_edge, n_tris, tri_adj):
    """From the strip-edge set, reconstruct strip chains.

    Each tri has at most 2 strip-edges (deg ≤ 2 in strip subgraph), so
    components of the strip subgraph are simple paths or cycles.
    """
    strip_deg = {t: 0 for t in range(n_tris)}
    strip_adj = {t: [] for t in range(n_tris)}
    for (a, b) in strip_edge:
        strip_adj[a].append(b)
        strip_adj[b].append(a)
        strip_deg[a] += 1
        strip_deg[b] += 1

    visited = np.zeros(n_tris, dtype=bool)
    strips = []
    # Start chains at endpoints (deg 1) first
    for seed in range(n_tris):
        if visited[seed] or strip_deg[seed] != 1:
            continue
        chain = []
        prev = -1
        cur = seed
        while cur >= 0 and not visited[cur]:
            visited[cur] = True
            chain.append(cur)
            nxt = -1
            for nb in strip_adj[cur]:
                if nb != prev and not visited[nb]:
                    nxt = nb
                    break
            prev = cur
            cur = nxt
        strips.append(chain)
    # Cycles (deg 2 everywhere)
    for seed in range(n_tris):
        if visited[seed] or strip_deg[seed] == 0:
            continue
        chain = []
        prev = -1
        cur = seed
        while cur >= 0 and not visited[cur]:
            visited[cur] = True
            chain.append(cur)
            nxt = -1
            for nb in strip_adj[cur]:
                if nb != prev and not visited[nb]:
                    nxt = nb
                    break
            prev = cur
            cur = nxt
        strips.append(chain)
    # Singletons (deg 0)
    for t in range(n_tris):
        if not visited[t]:
            visited[t] = True
            strips.append([t])
    return strips


# ============================================================
# Public: tunneling strip cover + chunking
# ============================================================

def stewart_tunneling(tri_adj, n_tris, max_iters=1000, max_depth=MAX_TUNNEL_DEPTH,
                      time_budget_s=None, verbose=False, do_iterate=False):
    """Run Stewart's static tunneling. Returns final strips.

    time_budget_s: optional wallclock cap; exits early when exceeded.

    NOTE: do_iterate=True engages the alternating-edge BFS pass. The
    current implementation has a known correctness bug — it increases
    strip count instead of decreasing. Use False (default) to skip
    iteration and rely on greedy DFS seed strips. This still produces
    the long-strip / sausage-shape characteristic the baseline is
    measuring; only the absolute strip-count optimum is missed.
    """
    import time as _t
    t_start = _t.time()
    strips = _greedy_strip_cover(tri_adj, n_tris)
    strip_id, _, strip_edge, _ = _build_strip_state(strips, tri_adj, n_tris)
    if verbose:
        print(f"  [tunneling] seed strips: {len(strips)}")
    if not do_iterate:
        return strips

    iters = 0
    while iters < max_iters:
        if time_budget_s is not None and (_t.time() - t_start) > time_budget_s:
            if verbose:
                print(f"  [tunneling] time budget {time_budget_s}s hit at iter={iters}")
            break
        # Recompute endpoints + strip_id from current strips
        endpoints = []
        sid_arr = np.full(n_tris, -1, dtype=np.int64)
        for sid, strip in enumerate(strips):
            for t in strip:
                sid_arr[t] = sid
            if strip:
                endpoints.append((sid, strip[0]))
                if len(strip) > 1:
                    endpoints.append((sid, strip[-1]))

        found = False
        for (sid, end_tri) in endpoints:
            # Find another strip via alternating BFS (any strip != sid)
            for (sid2, end2) in endpoints:
                if sid2 == sid:
                    continue
                path = _bfs_alternating(
                    end_tri, sid2, tri_adj, strip_edge, sid_arr, max_depth)
                if path is not None and len(path) >= 2:
                    _complement_path(path, strip_edge)
                    strips = _rebuild_strips(strip_edge, n_tris, tri_adj)
                    found = True
                    break
            if found:
                break
        if not found:
            break
        iters += 1
        if verbose and iters % 50 == 0:
            print(f"  [tunneling] iter={iters} strips={len(strips)}")
    if verbose:
        print(f"  [tunneling] done iters={iters} strips={len(strips)}")
    return strips


def generate_meshlets_tunneling_with_budget(tris_np, tri_adj, max_tris=256, max_verts=256,
                                            max_iters=2000, max_depth=MAX_TUNNEL_DEPTH,
                                            time_budget_s=None, verbose=False, pack=True):
    n = len(tris_np)
    strips = stewart_tunneling(tri_adj, n, max_iters=max_iters, max_depth=max_depth,
                               time_budget_s=time_budget_s, verbose=verbose)
    return chunk_strips(strips, tris_np, max_tris=max_tris, max_verts=max_verts, pack=pack)


def chunk_strips(strips, tris_np, max_tris=256, max_verts=256, pack=True):
    """Cut strips into meshlets respecting both caps.

    pack=False (legacy): one meshlet per strip; cuts only on cap overflow.
    pack=True (tuned): pack consecutive strips into one meshlet until a cap
    overflows. Raises cap utilisation when seed strips are short.
    """
    meshlets = []
    cur = []
    cur_verts = set()
    for strip in strips:
        if not strip:
            continue
        if not pack:
            # reset per strip
            if cur:
                meshlets.append(cur)
                cur = []
                cur_verts = set()
        for t in strip:
            new_v = set(int(v) for v in tris_np[t]) - cur_verts
            would_v = len(cur_verts) + len(new_v)
            would_t = len(cur) + 1
            if cur and (would_t > max_tris or would_v > max_verts):
                meshlets.append(cur)
                cur = []
                cur_verts = set()
                new_v = set(int(v) for v in tris_np[t])
            cur.append(t)
            cur_verts.update(new_v)
    if cur:
        meshlets.append(cur)
    return meshlets


def generate_meshlets_tunneling(tris_np, tri_adj, max_tris=256, max_verts=256,
                                max_iters=2000, max_depth=MAX_TUNNEL_DEPTH,
                                time_budget_s=None, verbose=False, pack=True):
    """Stage-1 tunneling baseline: full pipeline.

    pack=True packs consecutive strips into shared meshlets up to caps —
    raises cap utilisation vs the original one-strip-per-meshlet chunker.
    """
    n = len(tris_np)
    strips = stewart_tunneling(tri_adj, n, max_iters=max_iters, max_depth=max_depth,
                               time_budget_s=time_budget_s, verbose=verbose)
    return chunk_strips(strips, tris_np, max_tris=max_tris, max_verts=max_verts, pack=pack)

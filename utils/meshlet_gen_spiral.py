"""Spiral / concentric-ring meshlet generator.

Construction-stage spiral (not post-processing). Each meshlet:
    - Picks a seed triangle.
    - Grows by BFS rings; each ring is ordered as an edge-adjacent chain
      starting near the last triangle of the previous ring.
    - Result: triangle list spirals from center outward, with each
      consecutive pair sharing an edge whenever possible — exploits AMD
      GTS L/R edge reuse so the strip turns consistently.
    - Stops when max_tris or max_verts cap is reached.

Meshlet shape ≈ disk (low boundary/interior ratio) and traversal order
≈ single edge-continuous strip (low restart count). Combines goals 2 + 3
of the meshlet-gen design.
"""

import numpy as np


def _order_ring_as_chain(ring_set, tri_adj, start_tri=None):
    """Order ring tris as an edge-adjacent chain.

    If start_tri is given and adjacent to some ring member, start there
    (so this ring connects edge-to-edge with the previous one). Falls
    back to a chain endpoint (min ring-internal degree).
    """
    if not ring_set:
        return []
    in_ring_adj = {
        t: [nb for nb in tri_adj[t] if nb in ring_set]
        for t in ring_set
    }

    start = -1
    if start_tri is not None:
        candidates = [t for t in ring_set if start_tri in tri_adj[t]]
        if candidates:
            # Prefer chain endpoints (deg 0 or 1 inside ring) for cleaner walk
            start = min(candidates, key=lambda x: len(in_ring_adj[x]))
    if start < 0:
        start = min(ring_set, key=lambda x: len(in_ring_adj[x]))

    chain = [start]
    seen = {start}
    cur = start
    # Walk forward
    while True:
        nxt = -1
        for nb in in_ring_adj[cur]:
            if nb not in seen:
                nxt = nb
                break
        if nxt < 0:
            break
        chain.append(nxt)
        seen.add(nxt)
        cur = nxt

    # Pick up any remaining ring tris (disconnected within ring) by
    # appending nearest-to-tail components.
    leftover = ring_set - seen
    while leftover:
        # Find a tri in leftover that is closest (any adj relation) to chain tail
        tail = chain[-1]
        bridge = None
        for cand in leftover:
            if any(nb in seen for nb in tri_adj[cand]):
                bridge = cand
                break
        if bridge is None:
            bridge = next(iter(leftover))
        chain.append(bridge)
        seen.add(bridge)
        leftover.discard(bridge)
        cur = bridge
        while True:
            nxt = -1
            for nb in in_ring_adj[cur]:
                if nb in leftover:
                    nxt = nb
                    break
            if nxt < 0:
                break
            chain.append(nxt)
            seen.add(nxt)
            leftover.discard(nxt)
            cur = nxt

    return chain


def generate_meshlets_spiral(tris_np, tri_adj, max_tris=256, max_verts=256,
                             face_centroids=None):
    """Spiral / ring-based meshlet generation.

    Args:
        face_centroids: optional, used to pick seed tris from low-density
            regions first (currently unused — first-unvisited seed order).
    Returns:
        list of meshlets (list of tri indices each).
    """
    n = len(tris_np)
    visited = np.zeros(n, dtype=bool)
    meshlets = []

    seed_order = list(range(n))

    for seed in seed_order:
        if visited[seed]:
            continue

        meshlet = [seed]
        visited[seed] = True
        m_verts = set(int(v) for v in tris_np[seed])
        last_tri = seed
        prev_ring = [seed]

        while True:
            if len(meshlet) >= max_tris:
                break
            if len(m_verts) >= max_verts:
                break
            # Build next ring: unvisited adj of any tri in prev_ring
            next_set = set()
            for t in prev_ring:
                for nb in tri_adj[t]:
                    if not visited[nb]:
                        next_set.add(nb)
            if not next_set:
                break

            next_chain = _order_ring_as_chain(
                next_set, tri_adj, start_tri=last_tri)

            stopped = False
            for t in next_chain:
                if len(meshlet) >= max_tris:
                    stopped = True
                    break
                new_v = set(int(v) for v in tris_np[t]) - m_verts
                if len(m_verts) + len(new_v) > max_verts:
                    stopped = True
                    break
                meshlet.append(t)
                visited[t] = True
                m_verts.update(new_v)
                last_tri = t

            if stopped:
                break
            prev_ring = next_chain

        meshlets.append(meshlet)

    return meshlets

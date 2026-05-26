"""Spiral / concentric-ring strip generator inside a single meshlet.

Adapts the meshlet-generation spiral logic to work on a local dual graph
(per-meshlet). Outputs one or more strips covering all triangles, edge-
adjacent within each strip. Designed to plug into AMD GTS v3 encoder as
an alternative to `generate_strips_v2`.

Why a spiral inside a meshlet:
    Disk-shaped meshlets (joint / spiral generators) have a clear center
    + concentric BFS rings. v2 greedy strip cover walks one direction
    until forced to restart, often producing 2-3 strips per meshlet.
    Spiral order traverses rings outward, hitting every triangle in one
    long chain whenever the meshlet is topologically a disk.
"""

import numpy as np


def _order_ring_edge_chain(ring_set, local_adj, start_tri=None):
    """Order ring tris as edge-adjacent chain. Falls back to greedy.

    start_tri: prefer to start at a ring member adjacent to start_tri.
    """
    if not ring_set:
        return []
    in_ring_adj = {
        t: [nb for nb in local_adj[t] if nb in ring_set]
        for t in ring_set
    }
    start = -1
    if start_tri is not None:
        candidates = [t for t in ring_set if start_tri in local_adj[t]]
        if candidates:
            start = min(candidates, key=lambda x: len(in_ring_adj[x]))
    if start < 0:
        start = min(ring_set, key=lambda x: len(in_ring_adj[x]))

    chain = [start]
    seen = {start}
    cur = start
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

    # Pick up disconnected ring components by attaching at any chain tail
    leftover = ring_set - seen
    while leftover:
        bridge = None
        for cand in leftover:
            for nb in local_adj[cand]:
                if nb in seen:
                    bridge = cand
                    break
            if bridge is not None:
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


def _split_chain_into_strips(chain, local_adj):
    """Split a tri ordering into edge-adjacent strips at any non-adjacent break."""
    if not chain:
        return []
    strips = []
    cur = [chain[0]]
    for i in range(1, len(chain)):
        if chain[i] in local_adj[chain[i - 1]]:
            cur.append(chain[i])
        else:
            strips.append(cur)
            cur = [chain[i]]
    if cur:
        strips.append(cur)
    return strips


def generate_strips_spiral(tris_local, local_adj):
    """Spiral-ring strip generator over a local dual graph.

    Picks a seed (lowest valence — usually on meshlet's outer ring) and
    grows BFS rings outward. Each ring is ordered as an edge-chain; rings
    are concatenated serpentine. Final order is split into edge-adjacent
    strips so the AMD GTS encoder doesn't have to insert mid-strip
    restarts.
    """
    n_f = len(tris_local)
    if n_f == 0:
        return []
    visited = np.zeros(n_f, dtype=bool)

    valence = np.array([len(local_adj[i]) for i in range(n_f)], dtype=np.int32)

    final_strips = []
    while True:
        unproc = np.where(~visited)[0]
        if len(unproc) == 0:
            break
        seed = int(unproc[np.argmin(valence[unproc])])
        visited[seed] = True

        chain = [seed]
        last_tri = seed
        prev_ring = [seed]
        while True:
            next_set = set()
            for t in prev_ring:
                for nb in local_adj[t]:
                    if not visited[nb]:
                        next_set.add(nb)
            if not next_set:
                break
            ring_chain = _order_ring_edge_chain(
                next_set, local_adj, start_tri=last_tri)
            for t in ring_chain:
                chain.append(t)
                visited[t] = True
                last_tri = t
            prev_ring = ring_chain

        # Split chain into edge-adjacent strips
        sub = _split_chain_into_strips(chain, local_adj)
        final_strips.extend(sub)

    return final_strips

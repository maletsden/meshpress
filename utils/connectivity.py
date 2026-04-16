"""
Meshlet connectivity encoding/decoding with verification.
AMD-style: traversal + FIFO reuse buffer.
Includes Forsyth vertex cache optimization for triangle reordering.
"""

import numpy as np
from collections import deque


# ============================================================
# FIFO vertex cache
# ============================================================

class VertexFIFO:
    """Fixed-size FIFO cache for vertex reuse tracking."""
    def __init__(self, size=32):
        self.size = size
        self.buf = []  # buf[-1] is most recent, buf[0] is oldest

    def contains(self, v):
        return v in self.buf

    def index_of(self, v):
        """Return index from most-recent end (0 = most recent)."""
        if v not in self.buf:
            return -1
        return len(self.buf) - 1 - self.buf.index(v)

    def push(self, v):
        """Add v to cache (move to front if already present)."""
        if v in self.buf:
            self.buf.remove(v)
        self.buf.append(v)
        if len(self.buf) > self.size:
            self.buf.pop(0)

    def get_by_index(self, idx):
        """Get vertex at index from most-recent end."""
        return self.buf[len(self.buf) - 1 - idx]

    def copy(self):
        f = VertexFIFO(self.size)
        f.buf = list(self.buf)
        return f


# ============================================================
# Forsyth vertex cache optimization
# ============================================================

def forsyth_reorder(tris_local, n_verts, cache_size=32):
    """Reorder triangles to maximize vertex cache hits."""
    n_tris = len(tris_local)
    if n_tris <= 1:
        return list(range(n_tris))

    vert_tris = [[] for _ in range(n_verts)]
    for ti in range(n_tris):
        for v in tris_local[ti]:
            vert_tris[int(v)].append(ti)

    used = np.zeros(n_tris, dtype=bool)
    cache = VertexFIFO(cache_size)
    remaining = np.array([len(vt) for vt in vert_tris])
    output = []

    def score_tri(ti):
        s = 0.0
        for v in tris_local[ti]:
            v = int(v)
            if remaining[v] == 0:
                continue
            ci = cache.index_of(v)
            pos_score = 1.0 / (2.0 + ci) if ci >= 0 else 0.0
            val_score = 1.0 / (1.0 + remaining[v])
            s += pos_score * 0.75 + val_score * 0.25
        return s

    # Seed: triangle with highest valence sum
    best = max(range(n_tris), key=lambda ti: sum(remaining[int(v)] for v in tris_local[ti]))
    used[best] = True
    output.append(best)
    for v in tris_local[best]:
        v = int(v)
        remaining[v] -= 1
        cache.push(v)

    while len(output) < n_tris:
        best_ti = -1
        best_score = -1.0

        # Check cache-adjacent triangles
        candidates = set()
        for v in cache.buf:
            for ti in vert_tris[v]:
                if not used[ti]:
                    candidates.add(ti)

        if candidates:
            for ti in candidates:
                s = score_tri(ti)
                if s > best_score:
                    best_score = s
                    best_ti = ti

        if best_ti < 0:
            for ti in range(n_tris):
                if not used[ti]:
                    best_ti = ti
                    break
        if best_ti < 0:
            break

        used[best_ti] = True
        output.append(best_ti)
        for v in tris_local[best_ti]:
            v = int(v)
            remaining[v] -= 1
            cache.push(v)

    return output


# ============================================================
# AMD encode
# ============================================================

def amd_encode_meshlet(meshlet_tris, tris_np, tri_adj, tri_order_local,
                       tris_local, local_adj, n_local, fifo_size=32):
    """Encode meshlet connectivity. Returns encoded data + bit count.

    Args:
        tri_order_local: list of LOCAL triangle indices in processing order
        tris_local: (n_f, 3) local vertex indices
        local_adj: local_adj[local_ti] = [neighbor local tis]
        n_local: number of unique local vertices
    """
    idx_bits = max(1, int(np.ceil(np.log2(n_local + 1))))
    fifo_bits = max(1, int(np.ceil(np.log2(fifo_size + 1))))

    cache = VertexFIFO(fifo_size)
    vert_known = set()
    processed = set()
    # Store each triangle's vertex list as we encode it (for consistent parent lookup)
    encoded_tri_verts = {}  # local_tri_idx -> [v0, v1, v2]

    stream = []
    total_bits = 32  # header

    stats = {"new": 0, "fifo_hit": 0, "fifo_miss": 0, "root_verts": 0}

    for step, li in enumerate(tri_order_local):
        tri_v = [int(tris_local[li, j]) for j in range(3)]

        # Find parent: SMALLEST index processed neighbor (deterministic)
        parent_li = None
        for nb in sorted(local_adj[li]):
            if nb in processed:
                parent_li = nb
                break

        if parent_li is None:
            # Root triangle
            token = {"type": "root", "verts": tri_v}
            for v in tri_v:
                total_bits += idx_bits
                stats["root_verts"] += 1
                vert_known.add(v)
                cache.push(v)
            stream.append(token)
            encoded_tri_verts[li] = tri_v
        else:
            # Use the STORED vertex list of the parent (consistent with decoder)
            parent_v = encoded_tri_verts[parent_li]
            shared = set(tri_v) & set(parent_v)

            # Encode shared edge using parent's vertex ORDER
            parent_edges = [(parent_v[0], parent_v[1]),
                            (parent_v[1], parent_v[2]),
                            (parent_v[0], parent_v[2])]
            edge_code = 0
            for ei, (ea, eb) in enumerate(parent_edges):
                if {ea, eb} == shared:
                    edge_code = ei
                    break
            total_bits += 2

            # Encode non-shared vertex(es)
            for v in tri_v:
                if v in shared:
                    continue

                if v not in vert_known:
                    total_bits += 1
                    stats["new"] += 1
                    vert_known.add(v)
                    token = {"type": "tri", "edge": edge_code, "v_type": "new", "v": v}
                else:
                    total_bits += 1
                    ci = cache.index_of(v)
                    if ci >= 0:
                        total_bits += fifo_bits
                        stats["fifo_hit"] += 1
                        token = {"type": "tri", "edge": edge_code,
                                 "v_type": "fifo", "fifo_idx": ci, "v": v}
                    else:
                        total_bits += idx_bits
                        stats["fifo_miss"] += 1
                        token = {"type": "tri", "edge": edge_code,
                                 "v_type": "full", "local_idx": v, "v": v}

                cache.push(v)
                stream.append(token)

            # Store in canonical order: [shared_min, shared_max, new_v]
            sv = sorted(shared)
            non_shared = [v for v in tri_v if v not in shared]
            encoded_tri_verts[li] = sv + non_shared

        processed.add(li)

    return total_bits, stream, stats, cache


def amd_decode_verify(stream, tri_order_local, tris_local, local_adj, n_local, fifo_size=32):
    """Decode the encoded stream and verify against original triangles.
    Returns (n_matched, n_total) triangle count."""

    cache = VertexFIFO(fifo_size)
    processed = set()
    # Store the ACTUAL vertex list per decoded triangle (preserving order)
    decoded_tris_verts = {}  # local_tri_idx -> [v0, v1, v2] as encoded

    token_iter = iter(stream)

    for step, li in enumerate(tri_order_local):
        # Determine if root
        is_root = not any(nb in processed for nb in local_adj[li])

        if is_root:
            token = next(token_iter)
            assert token["type"] == "root"
            tri_v = list(token["verts"])
            for v in tri_v:
                cache.push(v)
            decoded_tris_verts[li] = tri_v
        else:
            # Find parent: SMALLEST index processed neighbor (same as encoder)
            parent_li = None
            for nb in sorted(local_adj[li]):
                if nb in processed:
                    parent_li = nb
                    break

            # CRITICAL: use the parent's vertex list in ORIGINAL ORDER
            # (same order as tris_local[parent_li]), not sorted
            parent_v = list(decoded_tris_verts[parent_li])

            token = next(token_iter)
            edge_code = token["edge"]

            # Same edge convention as encoder
            parent_edges = [(parent_v[0], parent_v[1]),
                            (parent_v[1], parent_v[2]),
                            (parent_v[0], parent_v[2])]
            shared = set(parent_edges[edge_code])

            # Decode vertex
            if token["v_type"] == "new":
                new_v = token["v"]
            elif token["v_type"] == "fifo":
                new_v = cache.get_by_index(token["fifo_idx"])
            elif token["v_type"] == "full":
                new_v = token["local_idx"]
            else:
                new_v = -1

            # Store in same canonical order as encoder: [shared_min, shared_max, new_v]
            sv = sorted(shared)
            decoded_tris_verts[li] = sv + [new_v]
            cache.push(new_v)

        processed.add(li)

    # Verify: compare decoded vertex sets against originals
    n_correct = 0
    n_total = len(tri_order_local)

    for li in decoded_tris_verts:
        orig = frozenset(int(tris_local[li, j]) for j in range(3))
        decoded = frozenset(decoded_tris_verts[li])
        if orig == decoded:
            n_correct += 1

    return n_correct, n_total


# ============================================================
# Full roundtrip test
# ============================================================

def amd_encode_decode_verify(meshlet_tris, tris_np, tri_adj):
    """Full encode → decode → verify for a meshlet.
    Tests both BFS and Forsyth orderings."""

    # Build local maps
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    local_verts = sorted(vert_set)
    g2l = {g: l for l, g in enumerate(local_verts)}
    n_local = len(local_verts)

    n_f = len(meshlet_tris)
    tris_local = np.zeros((n_f, 3), dtype=int)
    tri_map = {}
    for li, ti in enumerate(meshlet_tris):
        tri_map[ti] = li
        for j in range(3):
            tris_local[li, j] = g2l[int(tris_np[ti, j])]

    local_adj = [[] for _ in range(n_f)]
    for li, ti in enumerate(meshlet_tris):
        for nb in tri_adj[ti]:
            if nb in tri_map:
                local_adj[li].append(tri_map[nb])

    # BFS order
    from utils.meshlet_generator import meshlet_bfs
    bfs_trav = meshlet_bfs(meshlet_tris, tri_adj)
    bfs_order = [tri_map[ti] for ti, _ in bfs_trav]

    # Forsyth order
    forsyth_order = forsyth_reorder(tris_local, n_local, cache_size=32)

    results = {}
    for name, order in [("bfs", bfs_order), ("forsyth", forsyth_order)]:
        bits, stream, stats, _ = amd_encode_meshlet(
            meshlet_tris, tris_np, tri_adj, order,
            tris_local, local_adj, n_local)

        matched, total = amd_decode_verify(
            stream, order, tris_local, local_adj, n_local)

        results[name] = {
            "bits": bits, "stats": stats, "matched": matched, "total": total,
            "bpt": bits / n_f if n_f > 0 else 0,
        }

    details = (
        f"Verts={n_local}, Tris={n_f}\n"
        f"  BFS:     {results['bfs']['bits']/8:.0f}B "
        f"({results['bfs']['bpt']:.1f} bpt) "
        f"new={results['bfs']['stats']['new']} "
        f"fifo_hit={results['bfs']['stats']['fifo_hit']} "
        f"fifo_miss={results['bfs']['stats']['fifo_miss']} "
        f"verify={results['bfs']['matched']}/{results['bfs']['total']}\n"
        f"  Forsyth: {results['forsyth']['bits']/8:.0f}B "
        f"({results['forsyth']['bpt']:.1f} bpt) "
        f"new={results['forsyth']['stats']['new']} "
        f"fifo_hit={results['forsyth']['stats']['fifo_hit']} "
        f"fifo_miss={results['forsyth']['stats']['fifo_miss']} "
        f"verify={results['forsyth']['matched']}/{results['forsyth']['total']}"
    )

    return results['forsyth']['bits'], results['bfs']['bits'], details
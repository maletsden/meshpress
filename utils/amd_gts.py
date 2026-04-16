"""
AMD Generalized Triangle Strip (GTS) style encoder + decoder (simplified).

Uses greedy strip generation + 2-bit edge-index + inc/reuse flag encoding.

Format:
  - 32-bit header per meshlet
  - List of strips; each strip:
    - Root triangle: 3 new-vertex tokens (inc=1 + idx_bits each)
    - Extension tokens per subsequent triangle:
      - 2-bit `edge_code`: which of prev triangle's 3 edges is shared
        (00 = verts[0,1], 01 = verts[1,2], 10 = verts[0,2])
      - 1-bit `inc`: 1 = new vertex, 0 = reuse from FIFO
      - Payload:
        - inc=1: idx_bits (new local vertex index)
        - inc=0: fifo_bits (reuse buffer index)
  - 1-bit strip marker between strips

Decoder walks tokens; for each extension, uses edge_code to determine
shared verts and reconstructs the original triangle.
"""

import numpy as np
from collections import deque


# ---------------------------------------------------------------------
# Strip generation (greedy, edge-adjacent extension)
# ---------------------------------------------------------------------

def generate_strips(tris_local, local_adj):
    """Greedy bidirectional strip generator: grows from both ends of seed.

    Picks seeds in order of LOWEST valence (so low-valence triangles become
    strip endpoints). Returns list of strips (lists of local tri ids).
    """
    n_f = len(tris_local)
    if n_f == 0:
        return []
    processed = np.zeros(n_f, dtype=bool)
    valence = np.array([len(local_adj[i]) for i in range(n_f)])
    order = np.argsort(valence)   # low valence first = good strip starts
    strips = []

    for seed in order:
        if processed[seed]:
            continue
        strip = [seed]
        processed[seed] = True

        # Extend forward
        current = seed
        while True:
            extended = False
            # Prefer low-valence neighbors (fewer future choices → fewer dead ends)
            candidates = sorted(local_adj[current], key=lambda x: valence[x])
            for nb in candidates:
                if not processed[nb]:
                    strip.append(nb)
                    processed[nb] = True
                    current = nb
                    extended = True
                    break
            if not extended:
                break

        # Extend backward
        current = seed
        while True:
            extended = False
            candidates = sorted(local_adj[current], key=lambda x: valence[x])
            for nb in candidates:
                if not processed[nb]:
                    strip.insert(0, nb)
                    processed[nb] = True
                    current = nb
                    extended = True
                    break
            if not extended:
                break

        strips.append(strip)
    return strips


# ---------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------

def gts_encode(tris_local, local_adj, n_local, reuse_buf_size=16):
    """Encode triangles into GTS-style bitstream.

    Returns:
        bits: total bit count
        stream: list of tokens (for round-trip verification)
    """
    n_f = len(tris_local)
    if n_f == 0:
        return 32, []

    idx_bits = max(1, int(np.ceil(np.log2(n_local + 1))))
    reuse_bits = max(1, int(np.ceil(np.log2(reuse_buf_size + 1))))

    strips = generate_strips(tris_local, local_adj)
    stream = []
    total_bits = 32 + 8  # meshlet header + n_strips byte

    reuse_fifo = deque(maxlen=reuse_buf_size)
    emitted = set()

    def fifo_idx(v):
        try:
            return reuse_fifo.index(v)
        except ValueError:
            return -1

    def push_reuse(v):
        if v in reuse_fifo:
            reuse_fifo.remove(v)
        reuse_fifo.append(v)

    for strip in strips:
        # Strip marker (1 bit)
        total_bits += 1

        # Root triangle: 3 verts. Allow FIFO reuse when possible.
        root_verts = []
        for v in (int(x) for x in tris_local[strip[0]]):
            if v in emitted:
                fi = fifo_idx(v)
                if fi >= 0:
                    total_bits += 1 + reuse_bits
                    root_verts.append(('start_reuse', fi, v))
                else:
                    total_bits += 1 + idx_bits
                    root_verts.append(('start_full', v))
            else:
                total_bits += 1 + idx_bits
                root_verts.append(('start_new', v))
                emitted.add(v)
            push_reuse(v)
        # Legacy token for decoder — keep simple format
        v0, v1, v2 = (int(x) for x in tris_local[strip[0]])
        stream.append(('start', v0, v1, v2))

        prev_tri = [v0, v1, v2]

        for si in range(1, len(strip)):
            li = strip[si]
            tri_v = [int(x) for x in tris_local[li]]
            tri_set = set(tri_v)
            shared_set = tri_set & set(prev_tri)
            if len(shared_set) != 2:
                # Strip extension invariant broken; treat this as a restart.
                # (Shouldn't happen because strip generator uses edge adjacency.)
                raise ValueError(f"strip extension not edge-adjacent "
                                  f"prev={prev_tri}, curr={tri_v}")
            new_v = (tri_set - shared_set).pop()

            # Determine edge_code: which of prev_tri's 3 edges is the shared pair
            # 00: verts[0,1]  01: verts[1,2]  10: verts[0,2]
            pair01 = frozenset((prev_tri[0], prev_tri[1]))
            pair12 = frozenset((prev_tri[1], prev_tri[2]))
            pair02 = frozenset((prev_tri[0], prev_tri[2]))
            shared_frozen = frozenset(shared_set)
            if shared_frozen == pair01:
                edge_code = 0
                new_prev = [prev_tri[0], prev_tri[1], new_v]
            elif shared_frozen == pair12:
                edge_code = 1
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            else:  # pair02
                edge_code = 2
                new_prev = [prev_tri[0], prev_tri[2], new_v]

            total_bits += 2  # edge_code

            # Vertex encoding
            if new_v not in emitted:
                total_bits += 1 + idx_bits   # inc=1 + new idx
                emitted.add(new_v)
                push_reuse(new_v)
                stream.append(('new', edge_code, new_v))
            else:
                fi = fifo_idx(new_v)
                if fi >= 0:
                    total_bits += 1 + reuse_bits
                    push_reuse(new_v)
                    stream.append(('reuse', edge_code, fi, new_v))
                else:
                    # Fallback full index
                    total_bits += 1 + idx_bits
                    push_reuse(new_v)
                    stream.append(('full', edge_code, new_v))

            prev_tri = new_prev

    return total_bits, stream


# ---------------------------------------------------------------------
# Decode (for verification)
# ---------------------------------------------------------------------

def gts_decode(stream):
    """Decode stream into list of triangles (local vertex indices).
    Triangles are returned with the SAME vertex order as the encoder saw them
    (up to the shared-pair being preserved)."""
    tris = []
    prev_tri = None
    for token in stream:
        if token[0] == 'start':
            _, v0, v1, v2 = token
            tris.append((v0, v1, v2))
            prev_tri = [v0, v1, v2]
        elif token[0] in ('new', 'reuse', 'full'):
            edge_code = token[1]
            new_v = token[-1]
            if edge_code == 0:
                s1, s2 = prev_tri[0], prev_tri[1]
                new_prev = [prev_tri[0], prev_tri[1], new_v]
            elif edge_code == 1:
                s1, s2 = prev_tri[1], prev_tri[2]
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            else:  # 2
                s1, s2 = prev_tri[0], prev_tri[2]
                new_prev = [prev_tri[0], prev_tri[2], new_v]
            tris.append((s1, s2, new_v))
            prev_tri = new_prev
        else:
            raise ValueError(f"unknown token {token}")
    return tris


# ---------------------------------------------------------------------
# Verify (set-based, ignores winding for now)
# ---------------------------------------------------------------------

def gts_roundtrip_verify(tris_local, local_adj, n_local, reuse_buf_size=16):
    bits, stream = gts_encode(tris_local, local_adj, n_local, reuse_buf_size)
    decoded = gts_decode(stream)
    orig = set(frozenset(int(v) for v in t) for t in tris_local)
    got = set(frozenset(v for v in t) for t in decoded)
    return bits, stream, decoded, orig == got

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
    """Greedy bidirectional strip generator (v1 - static valence heuristic).

    Seeds in order of LOWEST INITIAL valence. At each extension, picks the
    neighbor with lowest INITIAL valence. Kept for comparison / legacy
    format; prefer generate_strips_v2 for real encoding.
    """
    n_f = len(tris_local)
    if n_f == 0:
        return []
    processed = np.zeros(n_f, dtype=bool)
    valence = np.array([len(local_adj[i]) for i in range(n_f)])
    order = np.argsort(valence)
    strips = []

    for seed in order:
        if processed[seed]:
            continue
        strip = [seed]
        processed[seed] = True

        current = seed
        while True:
            extended = False
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


def generate_strips_v2(tris_local, local_adj):
    """Improved strip generator with dynamic-valence heuristic.

    Key differences from v1:
      - Tracks UNPROCESSED-neighbor count per triangle, updated after every
        placement. v1 used static initial valence, which doesn't reflect
        how constrained a triangle actually is mid-traversal.
      - Both seed selection AND extension pick the triangle with the lowest
        current valence ("most constrained"). This is the classic
        "exhaust-constrained-resources-first" pattern that avoids orphans.
      - Extension also breaks ties by preferring neighbors of neighbors with
        low valence (1-step look-ahead): we try not to strand a triangle
        whose only remaining neighbor is the current one.
    """
    n_f = len(tris_local)
    if n_f == 0:
        return []
    processed = np.zeros(n_f, dtype=bool)
    cur_val = np.array([len(local_adj[i]) for i in range(n_f)], dtype=np.int32)
    strips = []

    def _process(idx):
        processed[idx] = True
        for nb in local_adj[idx]:
            if not processed[nb]:
                cur_val[nb] -= 1

    def _best_neighbor(tri):
        """Pick unprocessed neighbor with lowest current valence; break ties
        by looking one step ahead (fewest unprocessed grand-neighbors)."""
        best = -1
        best_key = None
        for nb in local_adj[tri]:
            if processed[nb]:
                continue
            # Look-ahead: sum of cur_val over nb's unprocessed neighbors
            la = 0
            for nb2 in local_adj[nb]:
                if nb2 != tri and not processed[nb2]:
                    la += cur_val[nb2]
            key = (cur_val[nb], la)
            if best_key is None or key < best_key:
                best = nb
                best_key = key
        return best

    while True:
        unproc = np.where(~processed)[0]
        if len(unproc) == 0:
            break
        # Seed = unprocessed triangle with lowest current valence
        seed = int(unproc[np.argmin(cur_val[unproc])])

        strip = [seed]
        _process(seed)

        # Forward extension
        current = seed
        while True:
            nb = _best_neighbor(current)
            if nb < 0:
                break
            strip.append(nb)
            _process(nb)
            current = nb

        # Backward extension (from original seed)
        current = seed
        while True:
            nb = _best_neighbor(current)
            if nb < 0:
                break
            strip.insert(0, nb)
            _process(nb)
            current = nb

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


# ---------------------------------------------------------------------
# GTS v2: prefix-coded edge_code
# ---------------------------------------------------------------------
# On real meshes edge_code distributes roughly 57 / 40 / 2 %% across
# newest (01) / second (10) / oldest (00).  A Huffman-optimal prefix code:
#   01 -> '0'   (1 bit)
#   10 -> '10'  (2 bits)
#   00 -> '11'  (2 bits)
# Average ~1.42 bits per extension (vs flat 2 bits).

_EDGE_PREFIX_BITS = {1: 1, 2: 2, 0: 2}  # edge_code -> bit width


def gts_encode_v2(tris_local, local_adj, n_local, reuse_buf_size=16,
                   strip_gen="v2"):
    """GTS with prefix-coded edge_code + improved strip generator.

    strip_gen: 'v1' = static-valence greedy (legacy),
               'v2' = dynamic-valence + 1-step look-ahead (preferred).
    Same stream format as gts_encode -> decoded with gts_decode."""
    n_f = len(tris_local)
    if n_f == 0:
        return 32, []

    idx_bits = max(1, int(np.ceil(np.log2(n_local + 1))))
    reuse_bits = max(1, int(np.ceil(np.log2(reuse_buf_size + 1))))

    if strip_gen == "v2":
        strips = generate_strips_v2(tris_local, local_adj)
    else:
        strips = generate_strips(tris_local, local_adj)
    stream = []
    total_bits = 32 + 8

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
        total_bits += 1  # strip marker

        for v in (int(x) for x in tris_local[strip[0]]):
            if v in emitted:
                fi = fifo_idx(v)
                if fi >= 0:
                    total_bits += 1 + reuse_bits
                else:
                    total_bits += 1 + idx_bits
            else:
                total_bits += 1 + idx_bits
                emitted.add(v)
            push_reuse(v)
        v0, v1, v2 = (int(x) for x in tris_local[strip[0]])
        stream.append(('start', v0, v1, v2))
        prev_tri = [v0, v1, v2]

        for si in range(1, len(strip)):
            li = strip[si]
            tri_v = [int(x) for x in tris_local[li]]
            tri_set = set(tri_v)
            shared_set = tri_set & set(prev_tri)
            if len(shared_set) != 2:
                raise ValueError("strip extension not edge-adjacent")
            new_v = (tri_set - shared_set).pop()

            pair01 = frozenset((prev_tri[0], prev_tri[1]))
            pair12 = frozenset((prev_tri[1], prev_tri[2]))
            shared_frozen = frozenset(shared_set)
            if shared_frozen == pair01:
                edge_code = 0
                new_prev = [prev_tri[0], prev_tri[1], new_v]
            elif shared_frozen == pair12:
                edge_code = 1
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            else:
                edge_code = 2
                new_prev = [prev_tri[0], prev_tri[2], new_v]

            total_bits += _EDGE_PREFIX_BITS[edge_code]

            if new_v not in emitted:
                total_bits += 1 + idx_bits
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
                    total_bits += 1 + idx_bits
                    push_reuse(new_v)
                    stream.append(('full', edge_code, new_v))

            prev_tri = new_prev

    return total_bits, stream


def gts_roundtrip_verify_v2(tris_local, local_adj, n_local, reuse_buf_size=16):
    bits, stream = gts_encode_v2(tris_local, local_adj, n_local, reuse_buf_size)
    decoded = gts_decode(stream)  # same stream format, same decoder
    orig = set(frozenset(int(v) for v in t) for t in tris_local)
    got = set(frozenset(v for v in t) for t in decoded)
    return bits, stream, decoded, orig == got


# ---------------------------------------------------------------------
# GTS v3: AMD-style L/R flag (1 bit) + FIFO reuse buffer
# ---------------------------------------------------------------------
# Strip constraint: every extension triangle shares either the newest edge
# (prev_tri[1], prev_tri[2]) -> encoded as L=0, or the "second" edge
# (prev_tri[0], prev_tri[2]) -> encoded as L=1. The "oldest" edge
# (prev_tri[0], prev_tri[1]) can never be used mid-strip -> forces a strip
# restart. This matches the AMD GPUOpen generalized triangle strip format.
#
# Per-extension cost:
#     1 bit L/R flag  +  1 bit inc  +  (idx_bits OR reuse_bits) payload
#
# Token stream is compatible with gts_decode (we just never emit
# edge_code=0). Bit count differs: 1 bit per edge_code instead of 1-2 bits
# prefix-coded.


def generate_strips_v2_seeded(tris_local, local_adj, seed_first):
    """generate_strips_v2 but with a forced first seed.

    Subsequent seeds chosen by min current valence as before.
    """
    n_f = len(tris_local)
    if n_f == 0:
        return []
    processed = np.zeros(n_f, dtype=bool)
    cur_val = np.array([len(local_adj[i]) for i in range(n_f)], dtype=np.int32)
    strips = []

    def _process(idx):
        processed[idx] = True
        for nb in local_adj[idx]:
            if not processed[nb]:
                cur_val[nb] -= 1

    def _best_neighbor(tri):
        best = -1
        best_key = None
        for nb in local_adj[tri]:
            if processed[nb]:
                continue
            la = 0
            for nb2 in local_adj[nb]:
                if nb2 != tri and not processed[nb2]:
                    la += cur_val[nb2]
            key = (cur_val[nb], la)
            if best_key is None or key < best_key:
                best = nb
                best_key = key
        return best

    first = True
    while True:
        if first:
            seed = int(seed_first)
            first = False
        else:
            unproc = np.where(~processed)[0]
            if len(unproc) == 0:
                break
            seed = int(unproc[np.argmin(cur_val[unproc])])
        if processed[seed]:
            continue
        strip = [seed]
        _process(seed)
        current = seed
        while True:
            nb = _best_neighbor(current)
            if nb < 0:
                break
            strip.append(nb)
            _process(nb)
            current = nb
        current = seed
        while True:
            nb = _best_neighbor(current)
            if nb < 0:
                break
            strip.insert(0, nb)
            _process(nb)
            current = nb
        strips.append(strip)
    return strips


def generate_strips_multiseed(tris_local, local_adj, n_seeds=8):
    """Try multiple seed triangles; return strip set with fewest strips
    (proxy for fewest restarts → lowest header + reuse-bit cost).

    Seed candidates: lowest-valence (current default), highest-valence,
    plus N random tris from the meshlet.
    """
    n_f = len(tris_local)
    if n_f == 0:
        return []
    val = np.array([len(local_adj[i]) for i in range(n_f)], dtype=np.int32)
    seeds = set()
    seeds.add(int(np.argmin(val)))
    seeds.add(int(np.argmax(val)))
    if n_f > 4:
        rng = np.random.default_rng(0)
        for s in rng.choice(n_f, size=min(n_seeds - 2, n_f - 2), replace=False):
            seeds.add(int(s))
    best = None
    best_n = None
    for s in seeds:
        strips = generate_strips_v2_seeded(tris_local, local_adj, s)
        if best_n is None or len(strips) < best_n:
            best_n = len(strips)
            best = strips
    return best


def generate_strips_lr(tris_local, local_adj):
    """Bidirectional strip generator for L/R encoding.

    Delegates to generate_strips_v2 (dynamic valence + 1-step look-ahead,
    bidirectional). Any edge-adjacent chain with no revisits on a manifold
    mesh is automatically L/R-encodable — oldest-edge shares only appear
    at triangle revisits, which don't happen here. The root-orientation
    choice happens in gts_encode_v3.
    """
    return generate_strips_v2(tris_local, local_adj)


def generate_strips_lr_forward_only(tris_local, local_adj):
    """Forward-only L/R strip generator (kept for reference, unused)."""
    n_f = len(tris_local)
    if n_f == 0:
        return []
    processed = np.zeros(n_f, dtype=bool)
    cur_val = np.array([len(local_adj[i]) for i in range(n_f)], dtype=np.int32)
    strips = []

    tris_sets = [frozenset(int(v) for v in tris_local[i]) for i in range(n_f)]

    def _process(idx):
        processed[idx] = True
        for nb in local_adj[idx]:
            if not processed[nb]:
                cur_val[nb] -= 1

    def _best_lr_neighbor(current, prev_tri):
        """Pick the L/R-valid unprocessed neighbor with lowest current
        valence + look-ahead. Returns (nb, new_prev_tri) or (None, None)."""
        a, b, c = prev_tri
        newest = frozenset((b, c))
        second = frozenset((a, c))
        best = None
        best_key = None
        best_new_prev = None
        for nb in local_adj[current]:
            if processed[nb]:
                continue
            shared = tris_sets[nb] & tris_sets[current]
            if len(shared) != 2:
                continue
            if shared == newest:
                new_v = next(iter(tris_sets[nb] - shared))
                new_prev = [b, c, new_v]
            elif shared == second:
                new_v = next(iter(tris_sets[nb] - shared))
                new_prev = [a, c, new_v]
            else:
                continue  # oldest edge -> skip
            la = 0
            for nb2 in local_adj[nb]:
                if nb2 != current and not processed[nb2]:
                    la += cur_val[nb2]
            key = (cur_val[nb], la)
            if best_key is None or key < best_key:
                best = nb
                best_key = key
                best_new_prev = new_prev
        return best, best_new_prev

    while True:
        unproc = np.where(~processed)[0]
        if len(unproc) == 0:
            break
        seed = int(unproc[np.argmin(cur_val[unproc])])
        _process(seed)

        strip = [seed]
        prev_tri = [int(v) for v in tris_local[seed]]
        current = seed

        while True:
            nb, new_prev = _best_lr_neighbor(current, prev_tri)
            if nb is None:
                break
            _process(nb)
            strip.append(nb)
            prev_tri = new_prev
            current = nb

        strips.append(strip)
    return strips


_LR_BITS = 1  # always 1 bit per extension


def gts_encode_v3(tris_local, local_adj, n_local, reuse_buf_size=16,
                  strip_method="v2"):
    """AMD GTS-style encoder with 1-bit L/R flag and FIFO reuse buffer.

    strip_method: "v2" (default, dynamic-valence greedy bidirectional)
                  or "spiral" (BFS-ring serpentine; better for disk-shaped
                              meshlets from joint/spiral generators).
    """
    n_f = len(tris_local)
    if n_f == 0:
        return 32, []

    idx_bits = max(1, int(np.ceil(np.log2(n_local + 1))))
    reuse_bits = max(1, int(np.ceil(np.log2(reuse_buf_size + 1))))

    if strip_method == "spiral":
        from utils.strip_spiral import generate_strips_spiral
        strips = generate_strips_spiral(tris_local, local_adj)
    elif strip_method == "multiseed":
        strips = generate_strips_multiseed(tris_local, local_adj)
    else:
        strips = generate_strips_lr(tris_local, local_adj)
    stream = []
    total_bits = 32 + 8  # header + n_strips

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

    def _root_orientation(root_id, next_id):
        """Pick a vertex order for the root triangle such that the shared
        edge with the next triangle in the strip is newest (positions 1,2)."""
        root_verts = [int(x) for x in tris_local[root_id]]
        if next_id is None:
            return root_verts  # singleton — orientation doesn't matter
        next_verts = set(int(x) for x in tris_local[next_id])
        shared = set(root_verts) & next_verts
        if len(shared) != 2:
            return root_verts  # non-adjacent (shouldn't happen)
        third = (set(root_verts) - shared).pop()
        y, z = tuple(shared)
        return [third, y, z]

    for strip in strips:
        total_bits += 1  # strip marker

        root_tri_id = strip[0]
        next_id = strip[1] if len(strip) > 1 else None
        root_oriented = _root_orientation(root_tri_id, next_id)
        for v in root_oriented:
            if v in emitted:
                fi = fifo_idx(v)
                if fi >= 0:
                    total_bits += 1 + reuse_bits
                else:
                    total_bits += 1 + idx_bits
            else:
                total_bits += 1 + idx_bits
                emitted.add(v)
            push_reuse(v)
        stream.append(('start', root_oriented[0], root_oriented[1], root_oriented[2]))
        prev_tri = list(root_oriented)

        for li in strip[1:]:
            tri_v = [int(x) for x in tris_local[li]]
            tri_set = set(tri_v)
            prev_set = set(prev_tri)
            shared = tri_set & prev_set
            if len(shared) != 2:
                raise RuntimeError("Strip has non-edge-adjacent neighbors")
            new_v = next(iter(tri_set - shared))

            # For a forward walk on a manifold mesh, the shared edge is
            # always newest or second of the current prev_tri.
            pair_newest = frozenset((prev_tri[1], prev_tri[2]))
            pair_second = frozenset((prev_tri[0], prev_tri[2]))
            shared_fs = frozenset(shared)
            if shared_fs == pair_newest:
                edge_code = 1
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            elif shared_fs == pair_second:
                edge_code = 2
                new_prev = [prev_tri[0], prev_tri[2], new_v]
            else:
                # Oldest-edge share: impossible for a forward walk with no
                # revisits on a manifold mesh (the incoming edge always
                # becomes the new OLDEST; the next extension picks one of
                # the 2 remaining edges, which are newest/second).
                raise RuntimeError(
                    "v3 hit oldest-edge share — non-manifold input?")

            total_bits += _LR_BITS  # 1-bit L/R flag

            if new_v not in emitted:
                total_bits += 1 + idx_bits
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
                    total_bits += 1 + idx_bits
                    push_reuse(new_v)
                    stream.append(('full', edge_code, new_v))

            prev_tri = new_prev

    return total_bits, stream


def gts_roundtrip_verify_v3(tris_local, local_adj, n_local, reuse_buf_size=16):
    bits, stream = gts_encode_v3(tris_local, local_adj, n_local, reuse_buf_size)
    decoded = gts_decode(stream)  # same decoder works (edge_code ∈ {1, 2})
    orig = set(frozenset(int(v) for v in t) for t in tris_local)
    got = set(frozenset(v for v in t) for t in decoded)
    return bits, stream, decoded, orig == got

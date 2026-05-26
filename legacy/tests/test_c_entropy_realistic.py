"""Cycle-C v2: fair entropy comparison with REALISTIC per-stream overhead.

Models compared per (level, axis) group across all meshlets:

  current  - existing _stream_bits estimate (n*Shannon + 32 per stream)
  fixed    - plain fixed-width (n*bits_per_code + 24 header per stream)
  A. per-stream Huffman   - n*Shannon + canonical Huffman header per stream
                            header: A * 4 bits  (A = alphabet size)
  B. per-stream Laplacian - n*Laplacian(b_i) + 16 bits per stream (1 float b)
  C. shared Laplacian     - n*Laplacian(b_shared) per stream
                            single 16-bit b per group, paid once
  D. shared Huffman tree  - n*shared_entropy per stream
                            single A_shared * 4 bit header per group
  E. shared range coder   - n*shared_entropy per stream
                            A_shared * 12 bit probability table per group

Larger meshes amortise the shared-model header across more streams; cycle-C
showed shared-Laplacian losing on small meshes, but bigger models with
hundreds of meshlets per group might flip the result.
"""

import numpy as np
from collections import Counter

from reader import Reader
from encoder.implementation.meshlet_wavelet import _to_numpy, _global_quantize
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts, edgebreaker_vertex_order,
)
from utils.boundary_split import (
    identify_boundary_verts, split_meshlet_verts,
)
from utils.interior_sorts import sort_interior
from utils.float_wavelet import (
    float_haar_decompose, per_level_deltas,
)


TARGET_BASE = 32
RATIO = 4.0


def shannon(codes):
    if len(codes) == 0:
        return 0.0
    counts = Counter(codes.tolist())
    total = len(codes)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


def cross_entropy_under(codes, P):
    if len(codes) == 0:
        return 0.0
    total = 0.0
    for c in codes:
        p = P.get(int(c), 0.0)
        if p <= 0.0:
            return float("inf")
        total += -np.log2(p)
    return total / len(codes)


def laplacian_rate(codes, b):
    if len(codes) == 0 or b <= 0:
        return 0.0
    return float(np.abs(codes).mean()) / (b * np.log(2)) + np.log2(2.0 * b + 1e-12)


def collect_streams(path, sort_variant="greedy_nn"):
    model = Reader.read_from_file(path)
    verts_np, tris_np = _to_numpy(model)
    center = verts_np.mean(axis=0); vc = verts_np - center
    scale = float(np.max(np.linalg.norm(vc, axis=1))); vn = vc / scale
    per_coord_err = 0.0005 / scale / np.sqrt(3)

    global_codes, _, _, _ = _global_quantize(vn, per_coord_err)
    tri_adj = build_adjacency(tris_np)
    fn = compute_face_normals(vn, tris_np); fc = compute_face_centroids(vn, tris_np)
    meshlets = generate_meshlets_by_verts(tris_np, tri_adj, fn, fc, max_verts=256)
    boundary_set = identify_boundary_verts(meshlets, tris_np)

    by_group = {}
    for ml_tris in meshlets:
        vert_order, _, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
        if len(vert_order) < 1:
            continue
        _, _, int_local, _ = split_meshlet_verts(vert_order, boundary_set)
        if not int_local:
            continue
        int_local = sort_interior(sort_variant, int_local,
                                  global_codes=global_codes, vert_pos_float=vn)
        pts = vn[int_local]
        for d in range(3):
            offset = float(pts[:, d].min())
            shifted = pts[:, d] - offset
            base, levels, _ = float_haar_decompose(shifted, target_base=TARGET_BASE)
            L = len(levels)
            d_base, d_levels = per_level_deltas(per_coord_err, L, "haar", "geometric", RATIO)
            base_q = np.round(base / d_base).astype(np.int64)
            by_group.setdefault(("base", d), []).append(base_q)
            for k, dl in enumerate(levels):
                q = np.round(dl / d_levels[k]).astype(np.int64)
                by_group.setdefault((k, d), []).append(q)
    return by_group, len(meshlets)


def compare(path):
    print(f"\n=== {path} ===")
    groups, n_meshlets = collect_streams(path)
    n_streams = sum(len(v) for v in groups.values())
    print(f"  meshlets : {n_meshlets}    groups: {len(groups)}    "
          f"streams: {n_streams}")

    bits = {k: 0.0 for k in
        ["current", "fixed", "A_huff_per", "B_lap_per",
         "C_lap_shared", "D_huff_shared", "E_range_shared"]}

    for (lvl, axis), streams in groups.items():
        all_codes = np.concatenate(streams) if streams else np.zeros(0, dtype=np.int64)
        shared_b = max(1.0, float(np.abs(all_codes).mean()))
        shared_counts = Counter(all_codes.tolist())
        shared_total = len(all_codes)
        shared_P = {c: cnt / shared_total for c, cnt in shared_counts.items()} \
                   if shared_total > 0 else {}
        shared_A = len(shared_counts)

        bits["C_lap_shared"] += 16
        bits["D_huff_shared"] += shared_A * 4
        bits["E_range_shared"] += shared_A * 12

        for codes in streams:
            n = len(codes)
            if n == 0:
                continue
            cs = set(codes.tolist())
            A = len(cs)
            ent_local = shannon(codes)

            mn = int(codes.min()); rng = int(codes.max() - mn)
            bp = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
            bits["fixed"] += n * bp + 24
            bits["current"] += min(n * bp, n * ent_local + 32)
            bits["A_huff_per"] += n * ent_local + A * 4

            b_i = max(1.0, float(np.abs(codes).mean()))
            bits["B_lap_per"] += n * laplacian_rate(codes, b_i) + 16

            bits["C_lap_shared"] += n * laplacian_rate(codes, shared_b)
            xe = cross_entropy_under(codes, shared_P)
            bits["D_huff_shared"] += n * xe
            bits["E_range_shared"] += n * xe

    ref = bits["current"]
    rows = [
        ("current `_stream_bits` (Sh+32)",   bits["current"]),
        ("fixed-width per stream",            bits["fixed"]),
        ("A. per-stream Huffman (n*Sh + A*4)", bits["A_huff_per"]),
        ("B. per-stream Laplacian fit (+16)",  bits["B_lap_per"]),
        ("C. shared Laplacian per group",      bits["C_lap_shared"]),
        ("D. shared Huffman tree per group",   bits["D_huff_shared"]),
        ("E. shared range coder (12-bit)",     bits["E_range_shared"]),
    ]
    for name, b in rows:
        delta = (b - ref) / ref * 100 if ref > 0 else 0
        print(f"  {name:<38s}{b/8:>12,.0f} B  ({delta:+6.2f}% vs current)")


if __name__ == "__main__":
    # Smaller models first (so we see the overhead-dominated regime); then
    # progressively bigger so the shared-model header amortises.
    paths = [
        "assets/bunny.obj",          #   2.5K verts
        "assets/torus.obj",          #   3.8K verts
        "assets/eyeball.obj",        #  ~14K verts
        "assets/stanford-bunny.obj", #  35.9K verts
        "assets/Monkey.obj",         # 504K verts
        "assets/tank.obj",           # ~6M verts (slow load)
    ]
    import os
    for p in paths:
        if os.path.exists(p):
            try:
                compare(p)
            except Exception as e:
                print(f"  ERROR on {p}: {e}")

"""
Verify AMD connectivity encoding correctness on random meshlets.
Compare BFS vs Forsyth ordering.
"""

import numpy as np
import random
from reader import Reader
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts,
)
from utils.connectivity import amd_encode_decode_verify


def test_model(obj_path, max_verts=256, n_test=5):
    print(f"{'='*70}")
    print(f"Connectivity Verification — {obj_path}")
    print(f"{'='*70}")

    mesh = Reader.read_from_file(obj_path)
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)

    verts_np = np.empty((n_v, 3), dtype=np.float64)
    for i, v in enumerate(mesh.vertices):
        verts_np[i] = (v.x, v.y, v.z)
    tris_np = np.empty((n_t, 3), dtype=np.int64)
    for i, t in enumerate(mesh.triangles):
        tris_np[i] = (t.a, t.b, t.c)

    center = verts_np.mean(axis=0)
    vc = verts_np - center
    scale = np.max(np.linalg.norm(vc, axis=1))
    vn = vc / scale

    tri_adj = build_adjacency(tris_np)
    fn = compute_face_normals(vn, tris_np)
    fc = compute_face_centroids(vn, tris_np)

    meshlets = generate_meshlets_by_verts(
        tris_np, tri_adj, fn, fc, max_verts=max_verts)

    print(f"  {len(meshlets)} meshlets, max_verts={max_verts}")
    print()

    # Pick random meshlets (or largest if fewer than n_test)
    if len(meshlets) <= n_test:
        test_indices = list(range(len(meshlets)))
    else:
        # Pick diverse: smallest, largest, and random
        sizes = [(i, len(ml)) for i, ml in enumerate(meshlets)]
        sizes.sort(key=lambda x: x[1])
        test_indices = [
            sizes[0][0],              # smallest
            sizes[-1][0],             # largest
            sizes[len(sizes)//2][0],  # median
        ]
        remaining = [i for i, _ in sizes if i not in test_indices]
        random.seed(42)
        test_indices += random.sample(remaining, min(n_test - 3, len(remaining)))

    total_bfs_bits = 0
    total_forsyth_bits = 0
    total_tris = 0

    for idx in test_indices:
        ml_tris = meshlets[idx]
        print(f"  Meshlet {idx}:")
        forsyth_bits, bfs_bits, details = amd_encode_decode_verify(
            ml_tris, tris_np, tri_adj)
        print(f"    {details}")

        n_f = len(ml_tris)
        improvement = (1 - forsyth_bits / bfs_bits) * 100 if bfs_bits > 0 else 0
        print(f"    Forsyth vs BFS: {improvement:+.1f}% "
              f"({'better' if improvement > 0 else 'worse'})")
        print()

        total_bfs_bits += bfs_bits
        total_forsyth_bits += forsyth_bits
        total_tris += n_f

    print(f"  Totals across {len(test_indices)} meshlets ({total_tris} tris):")
    print(f"    BFS:     {total_bfs_bits/8:.0f} B ({total_bfs_bits/total_tris:.2f} bpt)")
    print(f"    Forsyth: {total_forsyth_bits/8:.0f} B ({total_forsyth_bits/total_tris:.2f} bpt)")
    imp = (1 - total_forsyth_bits / total_bfs_bits) * 100
    print(f"    Improvement: {imp:+.1f}%")


if __name__ == "__main__":
    test_model("assets/stanford-bunny.obj", max_verts=256, n_test=5)
    print("\n")
    test_model("assets/Monkey.obj", max_verts=256, n_test=5)
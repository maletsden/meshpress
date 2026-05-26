"""Test GNN refinement methods on a mesh.

Compares:
    joint            — baseline
    joint+gnn        — soft-loss GNN (utils/meshlet_gnn_fit.py)
    joint+rl         — straight-through REINFORCE GNN (real BPV reward)
"""

import argparse
import time
import numpy as np

from reader import Reader
from encoder import MeshletParaDelta
from utils.meshlet_generator import (
    generate_meshlets, build_adjacency,
    compute_face_normals, compute_face_centroids,
)


def _to_numpy(model):
    verts_np = np.array([[v.x, v.y, v.z] for v in model.vertices], dtype=np.float64)
    tris_np = np.array([[t.a, t.b, t.c] for t in model.triangles], dtype=np.int64)
    return verts_np, tris_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="assets/bunny.obj")
    ap.add_argument("--max-tris", type=int, default=256)
    ap.add_argument("--max-verts", type=int, default=256)
    ap.add_argument("--rl-steps", type=int, default=30)
    ap.add_argument("--rl-samples", type=int, default=2)
    ap.add_argument("--use-cuda", action="store_true",
                    help="encoder CUDA (not GNN device)")
    args = ap.parse_args()

    print(f"loading {args.mesh}")
    model = Reader.read_from_file(args.mesh)
    verts_np, tris_np = _to_numpy(model)
    tri_adj = build_adjacency(tris_np)
    fn = compute_face_normals(verts_np, tris_np)
    fc = compute_face_centroids(verts_np, tris_np)

    enc_kw = dict(
        max_tris=args.max_tris, max_verts=args.max_verts,
        use_nn=False, use_cuda=args.use_cuda, verbose=False,
    )

    def bpv_for(meshlets):
        enc = MeshletParaDelta(preset_meshlets=meshlets, **enc_kw)
        c = enc.encode(model)
        return float(c.bits_per_vertex)

    # --- joint init ---
    t0 = time.time()
    meshlets_init = generate_meshlets(
        tris_np, tri_adj, fn, fc, method="joint",
        max_tris=args.max_tris, max_verts=args.max_verts, verts_np=verts_np)
    t_init = time.time() - t0
    bpv_init = bpv_for(meshlets_init)
    print(f"\njoint        : n={len(meshlets_init)}  BPV={bpv_init:.3f}  t={t_init:.1f}s")

    # --- joint + soft GNN ---
    from utils.meshlet_gnn_fit import fit_partition_gnn
    t0 = time.time()
    meshlets_gnn = fit_partition_gnn(
        meshlets_init, tris_np, tri_adj, fn, verts_np,
        face_centroids=fc, max_tris=args.max_tris, max_verts=args.max_verts,
        verbose=False)
    t_gnn = time.time() - t0
    bpv_gnn = bpv_for(meshlets_gnn)
    print(f"joint+gnn    : n={len(meshlets_gnn)}  BPV={bpv_gnn:.3f}  "
          f"d={bpv_gnn - bpv_init:+.3f}  t={t_gnn:.1f}s")

    # --- joint + RL ---
    from utils.meshlet_gnn_reinforce import fit_partition_reinforce
    t0 = time.time()
    meshlets_rl = fit_partition_reinforce(
        meshlets_init, tris_np, tri_adj, fn, verts_np, fc,
        encode_fn=bpv_for,
        max_tris=args.max_tris, max_verts=args.max_verts,
        n_steps=args.rl_steps, n_samples_per_step=args.rl_samples,
        verbose=True)
    t_rl = time.time() - t0
    bpv_rl = bpv_for(meshlets_rl)
    print(f"\njoint+rl     : n={len(meshlets_rl)}  BPV={bpv_rl:.3f}  "
          f"d={bpv_rl - bpv_init:+.3f}  t={t_rl:.1f}s")


if __name__ == "__main__":
    main()

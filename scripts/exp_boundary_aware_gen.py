"""Test boundary-aware meshlet generation weights on Dragon.

The current LEARNED_WEIGHTS was tuned on stanford-bunny (small + smooth)
via Bayesian opt — it dropped w2_boundary_perim to 0.0008 (effectively
zero). On dense high-detail meshes like Dragon this leaves 43.1 % of
verts on meshlet boundaries, where the parallelogram predictor doesn't
apply.

Sweep w2 ∈ {0, 0.05, 0.10, 0.20, 0.40} keeping other learned weights
fixed and renormalising. Report meshlet count + boundary fraction +
estimated BPV delta from boundary-cost change alone.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.reader import Reader
from utils.mesh_clean import clean_mesh
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets,
)
from utils.meshlet_gen_joint import LEARNED_WEIGHTS


def _meshlet_boundary_stats(meshlets, tris_np, verts_np):
    """Return (n_meshlets, mean n_local, mean boundary fraction)."""
    # For each meshlet, count unique verts. Boundary vert = appears in
    # this meshlet AND in some other meshlet's tris.
    # First, build vert -> meshlet_set.
    vert_to_ms = {}
    meshlet_verts = []
    for mi, ml_tris in enumerate(meshlets):
        v = set()
        for t in ml_tris:
            a, b, c = tris_np[t]
            v.update((int(a), int(b), int(c)))
        meshlet_verts.append(v)
        for vi in v:
            if vi not in vert_to_ms:
                vert_to_ms[vi] = set()
            vert_to_ms[vi].add(mi)

    boundary_fracs = []
    n_locals = []
    for mi, v_set in enumerate(meshlet_verts):
        n_local = len(v_set)
        n_bnd = sum(1 for vi in v_set if len(vert_to_ms[vi]) > 1)
        boundary_fracs.append(n_bnd / max(1, n_local))
        n_locals.append(n_local)
    return (len(meshlets),
            float(np.mean(n_locals)),
            float(np.mean(boundary_fracs)),
            np.array(boundary_fracs))


def main(path: str):
    print(f"Loading {path} ...")
    m = Reader.read_from_file(path)
    m_clean, _ = clean_mesh(m, verbose=False)
    tris = np.array([[t.a, t.b, t.c] for t in m_clean.triangles],
                     dtype=np.int64)
    verts = np.array([[v.x, v.y, v.z] for v in m_clean.vertices],
                      dtype=np.float32)
    print(f"  n_v={len(verts):,}  n_t={len(tris):,}")
    tri_adj = build_adjacency(tris)
    fn = compute_face_normals(verts, tris)
    fc = compute_face_centroids(verts, tris)

    # Sweep max_verts (and matched max_tris) keeping LEARNED_WEIGHTS
    sweep = [
        ("max_v=256 (baseline)", 256),
        ("max_v=384",            384),
        ("max_v=512",            512),
        ("max_v=768",            768),
        ("max_v=1024",          1024),
    ]
    results = []
    for label, mv in sweep:
        t0 = time.time()
        meshlets = generate_meshlets(
            tris, tri_adj, fn, fc,
            method="joint_learned", max_tris=mv, max_verts=mv,
            verts_np=verts, weights=LEARNED_WEIGHTS)
        w = {"w2_boundary_perim": LEARNED_WEIGHTS["w2_boundary_perim"]}
        n_m, mean_n_local, mean_bf, bfs = _meshlet_boundary_stats(
            meshlets, tris, verts)
        elapsed = time.time() - t0
        print(f"\n  {label:25s}  w2={w['w2_boundary_perim']:.4f}  "
              f"[{elapsed:.1f}s]")
        print(f"    meshlets:   {n_m:,}")
        print(f"    mean verts: {mean_n_local:.1f}")
        print(f"    bnd frac:   mean={mean_bf*100:.1f}%  "
              f"p50={float(np.median(bfs))*100:.1f}%  "
              f"p99={float(np.percentile(bfs, 99))*100:.1f}%")
        results.append((label, n_m, mean_n_local, mean_bf))

    print(f"\n  Summary ({Path(path).stem}):")
    print(f"  {'preset':<22} {'meshlets':>10} {'mean_v':>8} "
          f"{'bnd_frac':>10}")
    for label, n_m, ml, bf in results:
        print(f"  {label:<22} {n_m:>10,} {ml:>8.1f} {bf*100:>9.1f}%")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "assets/Monkey.obj"
    main(p)
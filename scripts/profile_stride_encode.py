"""cProfile the STRIDE encode pipeline on Monkey.

Dumps:
  - top 30 cumulative-time entries (entire pipeline)
  - top 30 internal-time entries (hot inner loops)
  - per-stage wall-clock summary

Usage:
    python scripts/profile_stride_encode.py
    python scripts/profile_stride_encode.py assets/stanford-bunny.obj
"""
from __future__ import annotations

import cProfile
import io
import math
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import Reader
from utils.mesh_clean import clean_mesh
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets,
)
from utils.boundary_split import (
    identify_boundary_verts, build_boundary_table, verify_crack_free,
)
from utils.boundary_bvh import morton_permute_boundary
from encoder.paradelta_codec import (
    _quantize_global, _dequant_global, _plan_meshlet,
)
from encoder.paradelta_v5 import encode_from_prepared_v5


def run_encode(path: str, *, precision_error: float = 0.0005,
               max_verts: int = 256, max_tris: int = 256):
    model = Reader.read_from_file(path)
    model, _ = clean_mesh(model, verbose=False)
    verts = np.asarray([(v.x, v.y, v.z) for v in model.vertices],
                       dtype=np.float64)
    tris_np = np.asarray([(t.a, t.b, t.c) for t in model.triangles],
                         dtype=np.int64)
    n_v, n_t = len(verts), len(tris_np)
    center = verts.mean(axis=0)
    vc = verts - center
    scale = float(np.max(np.linalg.norm(vc, axis=1)))
    vn = vc / scale
    per_coord_err = precision_error / scale / math.sqrt(3)
    global_codes, g_min, g_range, g_bits = _quantize_global(vn, per_coord_err)
    bnd_recon_norm = _dequant_global(global_codes, g_min, g_range, g_bits)

    tri_adj = build_adjacency(tris_np)
    fn = compute_face_normals(vn, tris_np)
    fc = compute_face_centroids(vn, tris_np)

    stages = {}

    t0 = time.perf_counter()
    meshlets = generate_meshlets(
        tris_np, tri_adj, fn, fc,
        method="joint_learned", max_tris=max_tris, max_verts=max_verts,
        verts_np=vn,
    )
    stages["partition"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    boundary_set = identify_boundary_verts(meshlets, tris_np)
    boundary_list, _, _ = build_boundary_table(boundary_set, global_codes)
    boundary_list, _ = morton_permute_boundary(boundary_list, global_codes)
    gv_to_ref = {gv: i for i, gv in enumerate(boundary_list)}
    n_boundary = len(boundary_list)
    n_cracks, _ = verify_crack_free(
        meshlets, tris_np, global_codes, boundary_set)
    if n_cracks > 0:
        raise RuntimeError(f"crack-free fail: {n_cracks}")
    stages["boundary"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    plans = [
        _plan_meshlet(ml, tris_np, tri_adj, vn,
                      boundary_set, global_codes, gv_to_ref, "multiseed")
        for ml in meshlets
    ]
    stages["plan"] = time.perf_counter() - t0

    prep = {
        "center": center, "scale": scale, "per_coord_err": per_coord_err,
        "g_min": g_min, "g_range": g_range, "g_bits": g_bits,
        "n_v": n_v, "n_t": n_t, "n_boundary": n_boundary,
        "n_meshlets": len(meshlets), "boundary_list": boundary_list,
        "global_codes": global_codes, "bnd_recon_norm": bnd_recon_norm,
        "vn": vn, "plans": plans, "strip_method": "multiseed",
        "max_verts": max_verts, "max_tris": max_tris,
        "precision_error": precision_error, "gen_method": "joint_learned",
    }

    t0 = time.perf_counter()
    out = encode_from_prepared_v5(prep, verbose=False)
    stages["encode_v5"] = time.perf_counter() - t0

    return stages, len(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/Monkey.obj"
    full = str(ROOT / path) if not Path(path).is_absolute() else path

    print(f"Profiling encode on {Path(full).name}\n")
    pr = cProfile.Profile()
    pr.enable()
    stages, nbytes = run_encode(full)
    pr.disable()

    total = sum(stages.values())
    print(f"\n=== Stage wall-clock ({total:.2f} s total, {nbytes:,} B) ===")
    for k, v in stages.items():
        print(f"  {k:<12} {v:>7.2f} s  ({v/total*100:>5.1f} %)")

    out = ROOT / "profile_stride_encode.prof"
    pr.dump_stats(str(out))
    print(f"\nwrote raw profile to {out}")

    print("\n=== Top 30 by cumulative time ===")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print(s.getvalue())

    print("\n=== Top 30 by internal time ===")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    main()
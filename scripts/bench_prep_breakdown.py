"""Decompose prep stage into sub-stages to find hottest piece.

Runs prepare_paradelta_arrays inline with timing per call.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.fast_obj import load_mesh_npy, clean_mesh_npy
from encoder.paradelta_codec import (
    _quantize_global, _dequant_global, _plan_meshlet,
)
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets,
)
from utils.boundary_split import (
    identify_boundary_verts, build_boundary_table, verify_crack_free,
)
from utils.boundary_bvh import morton_permute_boundary
from utils.meshlet_plan_nb import build_global_adj_csr_nb
from utils.meshlet_gen_joint import LEARNED_WEIGHTS, DEFAULT_FEATURE_NORMS, _resolve_norm
from utils.meshlet_gen_joint_nb import generate_meshlets_joint_from_csr


def t():
    return time.perf_counter()


def prep_timed(verts, tris_np, *, max_verts=256, max_tris=256,
               precision_error=0.0005, gen_method="joint_learned",
               strip_method="multiseed", verbose=True):
    timings = {}
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    tris_np = np.ascontiguousarray(tris_np, dtype=np.int64)
    n_v, n_t = len(verts), len(tris_np)

    t0 = t()
    center = verts.mean(axis=0)
    vc = verts - center
    scale = float(np.max(np.linalg.norm(vc, axis=1)))
    vn = vc / scale
    per_coord_err = precision_error / scale / math.sqrt(3)
    global_codes, g_min, g_range, g_bits = _quantize_global(vn, per_coord_err)
    bnd_recon_norm = _dequant_global(global_codes, g_min, g_range, g_bits)
    timings["normalize+quant"] = t() - t0

    t0 = t()
    tris_np_i32 = tris_np.astype(np.int32)
    tri_adj_off, tri_adj_idx = build_global_adj_csr_nb(tris_np_i32)
    timings["adj_csr_nb (early)"] = t() - t0

    t0 = t()
    fn = compute_face_normals(vn, tris_np)
    timings["face_normals"] = t() - t0

    t0 = t()
    fc = compute_face_centroids(vn, tris_np)
    timings["face_centroids"] = t() - t0

    t0 = t()
    w = LEARNED_WEIGHTS; nrm = DEFAULT_FEATURE_NORMS
    pr_nm, pr_is = _resolve_norm("plane_resid", nrm)
    bp_nm, bp_is = _resolve_norm("boundary_perim", nrm)
    sc_nm, sc_is = _resolve_norm("strip_cont", nrm)
    ns_nm, ns_is = _resolve_norm("normal_sim", nrm)
    se_nm, se_is = _resolve_norm("shared_edges", nrm)
    bd_nm, bd_is = _resolve_norm("bfs_depth", nrm)
    K_const = (w["w1_plane_resid"]*pr_is, w["w2_boundary_perim"]*bp_is,
               w["w3_strip_cont"]*sc_is, w["w4_normal_sim"]*ns_is,
               w["w5_shared_edges"]*se_is, w["w6_bfs_depth"]*bd_is)
    NM_const = (pr_nm, bp_nm, sc_nm, ns_nm, se_nm, bd_nm)
    meshlets = generate_meshlets_joint_from_csr(
        tris_np.astype(np.int64), tri_adj_off, tri_adj_idx,
        vn, fn, max_tris=max_tris, max_verts=max_verts,
        K_const=K_const, NM_const=NM_const)
    timings["generate_meshlets_csr"] = t() - t0

    t0 = t()
    boundary_set = identify_boundary_verts(meshlets, tris_np)
    timings["identify_boundary_verts"] = t() - t0

    t0 = t()
    boundary_list, _, _ = build_boundary_table(boundary_set, global_codes)
    timings["build_boundary_table"] = t() - t0

    t0 = t()
    boundary_list, _ = morton_permute_boundary(boundary_list, global_codes)
    timings["morton_permute_boundary"] = t() - t0

    t0 = t()
    gv_to_ref = {gv: i for i, gv in enumerate(boundary_list)}
    timings["gv_to_ref dict"] = t() - t0

    t0 = t()
    n_cracks, _ = verify_crack_free(
        meshlets, tris_np, global_codes, boundary_set)
    timings["verify_crack_free"] = t() - t0
    if n_cracks > 0:
        raise RuntimeError(f"cracks: {n_cracks}")

    t0 = t()
    n_v_total = len(vn)
    is_boundary = np.zeros(n_v_total, dtype=np.bool_)
    for v in boundary_set:
        is_boundary[v] = True
    gv_to_ref_arr = np.full(n_v_total, -1, dtype=np.int64)
    for gv, ri in gv_to_ref.items():
        gv_to_ref_arr[gv] = ri
    timings["lookup arrays"] = t() - t0

    t0 = t()
    plans = []
    for ml in meshlets:
        p = _plan_meshlet(ml, tris_np_i32, tri_adj_off, tri_adj_idx, vn,
                          is_boundary, global_codes, gv_to_ref_arr,
                          strip_method)
        plans.append(p)
    timings["plan_meshlet_loop"] = t() - t0

    total = sum(timings.values())
    if verbose:
        print(f"\n  meshlets: {len(meshlets)}  total prep: {total:.2f}s")
        for k, v in sorted(timings.items(), key=lambda x: -x[1]):
            print(f"    {k:<32} {v:7.2f}s  ({100*v/total:5.1f}%)")
    return timings, len(meshlets)


def main():
    targets = sys.argv[1:] or ["assets/Monkey.obj", "assets/xyzrgb_dragon.obj"]

    # Warm JIT on bunny
    print("warming Numba JIT ...", flush=True)
    v, ti = load_mesh_npy(str(ROOT / "assets/bunny.obj"))
    v, ti = clean_mesh_npy(v, ti)
    prep_timed(v, ti, verbose=False)
    print("warm done", flush=True)

    for p in targets:
        full = str(ROOT / p)
        print(f"\n=== {Path(p).stem} ===", flush=True)
        v, ti = load_mesh_npy(full)
        v, ti = clean_mesh_npy(v, ti)
        prep_timed(v, ti)


if __name__ == "__main__":
    main()
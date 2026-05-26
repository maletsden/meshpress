"""Verify generate_meshlets_joint_from_csr matches generate_meshlets_joint_nb."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.fast_obj import load_mesh_npy, clean_mesh_npy
from utils.meshlet_generator import build_adjacency, compute_face_normals
from utils.meshlet_gen_joint import LEARNED_WEIGHTS, DEFAULT_FEATURE_NORMS, _resolve_norm
from utils.meshlet_gen_joint_nb import (
    generate_meshlets_joint_nb, generate_meshlets_joint_from_csr,
)
from utils.meshlet_plan_nb import build_global_adj_csr_nb


def consts():
    w, nrm = LEARNED_WEIGHTS, DEFAULT_FEATURE_NORMS
    pr_nm, pr_is = _resolve_norm("plane_resid", nrm)
    bp_nm, bp_is = _resolve_norm("boundary_perim", nrm)
    sc_nm, sc_is = _resolve_norm("strip_cont", nrm)
    ns_nm, ns_is = _resolve_norm("normal_sim", nrm)
    se_nm, se_is = _resolve_norm("shared_edges", nrm)
    bd_nm, bd_is = _resolve_norm("bfs_depth", nrm)
    K = (w["w1_plane_resid"]*pr_is, w["w2_boundary_perim"]*bp_is,
         w["w3_strip_cont"]*sc_is, w["w4_normal_sim"]*ns_is,
         w["w5_shared_edges"]*se_is, w["w6_bfs_depth"]*bd_is)
    NM = (pr_nm, bp_nm, sc_nm, ns_nm, se_nm, bd_nm)
    return K, NM


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/tank.obj"
    full = str(ROOT / path)
    verts, tris = load_mesh_npy(full)
    verts, tris = clean_mesh_npy(verts, tris)

    center = verts.mean(0)
    vc = verts - center
    scale = float(np.max(np.linalg.norm(vc, axis=1)))
    vn = vc / scale
    fn = compute_face_normals(vn, tris)

    K, NM = consts()

    # Compare degree arrays
    tri_adj_py = build_adjacency(tris)
    tris_i32_ = tris.astype(np.int32)
    adj_off_, adj_idx_ = build_global_adj_csr_nb(tris_i32_)
    deg_py = np.array([len(tri_adj_py[i]) for i in range(len(tris))], dtype=np.int32)
    deg_nb = (adj_off_[1:] - adj_off_[:-1]).astype(np.int32)
    diff_deg = np.where(deg_py != deg_nb)[0]
    print(f"deg mismatch tris: {len(diff_deg)}")
    n_order_diff = 0
    first_diff = None
    for i in range(len(tris)):
        py_l = list(tri_adj_py[i])
        nb_l = adj_idx_[adj_off_[i]:adj_off_[i+1]].tolist()
        if py_l != nb_l:
            n_order_diff += 1
            if first_diff is None:
                first_diff = (i, py_l, nb_l)
    print(f"neighbor-order diffs: {n_order_diff}")
    if first_diff:
        i, py_l, nb_l = first_diff
        print(f"  first tri {i}: py={py_l}  nb={nb_l}")

    print("Old path (Python dict edge IDs)...")
    tri_adj = build_adjacency(tris)
    ml_old = generate_meshlets_joint_nb(tris, tri_adj, fn, vn,
                                        max_tris=256, max_verts=256,
                                        K_const=K, NM_const=NM)
    print(f"  n_meshlets = {len(ml_old)}")

    print("New path (CSR + np.unique edge IDs)...")
    tris_i32 = tris.astype(np.int32)
    adj_off, adj_idx = build_global_adj_csr_nb(tris_i32)
    ml_new = generate_meshlets_joint_from_csr(
        tris.astype(np.int64), adj_off, adj_idx, vn, fn,
        max_tris=256, max_verts=256, K_const=K, NM_const=NM)
    print(f"  n_meshlets = {len(ml_new)}")

    if len(ml_old) != len(ml_new):
        print(f"MISMATCH: {len(ml_old)} vs {len(ml_new)}")
        return

    n_diff = 0
    for i, (a, b) in enumerate(zip(ml_old, ml_new)):
        if a != b:
            n_diff += 1
            if n_diff <= 5:
                print(f"  meshlet[{i}] differs: old={a[:10]}... new={b[:10]}...")
    print(f"meshlets differ: {n_diff} / {len(ml_old)}")


if __name__ == "__main__":
    main()

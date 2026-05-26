"""Profile MeshletParaDelta encode on Monkey to find the bottleneck.
Times each major stage individually."""
import time
import numpy as np
from reader import Reader
from encoder.implementation.meshlet_wavelet import (
    _to_numpy, _global_quantize, _dequantize_global, _amd_packed_bits,
)
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts,
)
from utils.boundary_split import (
    identify_boundary_verts, build_boundary_table, split_meshlet_verts,
    sort_by_morton, gts_encode_meshlet,
)
from utils.boundary_bvh import (
    morton_permute_boundary, delta_boundary_table_bits, delta_refs_bits,
)
from utils.parallelogram_predictor import quantize_interior_parallelogram
from utils.interior_sorts import sort_interior


from utils.meshlet_generator import edgebreaker_vertex_order


def section(name):
    print(f"\n--- {name} ---", flush=True)


eps = 0.0005
mv = 512
print(f"Loading Monkey...", flush=True)
t0 = time.time()
model = Reader.read_from_file("assets/Monkey.obj")
print(f"  reader: {time.time()-t0:.1f}s", flush=True)

verts_np, tris_np = _to_numpy(model)
n_v, n_t = len(verts_np), len(tris_np)
print(f"verts={n_v:,} tris={n_t:,}", flush=True)

t0 = time.time()
center = verts_np.mean(axis=0)
vc = verts_np - center
scale = np.max(np.linalg.norm(vc, axis=1))
vn = vc / scale
per_coord_err = eps / scale / np.sqrt(3)
global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)
print(f"global quantize:        {time.time()-t0:6.1f}s", flush=True)

t0 = time.time()
tri_adj = build_adjacency(tris_np)
print(f"build_adjacency:        {time.time()-t0:6.1f}s", flush=True)

t0 = time.time()
fn = compute_face_normals(vn, tris_np)
fc = compute_face_centroids(vn, tris_np)
print(f"face normals/centroids: {time.time()-t0:6.1f}s", flush=True)

t0 = time.time()
meshlets = generate_meshlets_by_verts(tris_np, tri_adj, fn, fc, max_verts=mv)
print(f"meshlet generation:     {time.time()-t0:6.1f}s  ({len(meshlets)} meshlets)", flush=True)

t0 = time.time()
boundary_set = identify_boundary_verts(meshlets, tris_np)
boundary_list, gv_to_ref, _ = build_boundary_table(boundary_set, global_codes)
boundary_list, _ = morton_permute_boundary(boundary_list, global_codes)
gv_to_ref = {gv: i for i, gv in enumerate(boundary_list)}
n_boundary = len(boundary_list)
print(f"boundary setup:         {time.time()-t0:6.1f}s  ({n_boundary} bnd verts)", flush=True)

t0 = time.time()
bnd_recon_global = _dequantize_global(global_codes, g_min, g_range, g_bits)
print(f"dequant global recon:   {time.time()-t0:6.1f}s", flush=True)

# Per-meshlet loop
t_eb = 0
t_split = 0
t_sort = 0
t_para = 0
t_conn = 0
t_bnd_refs = 0
for ml_tris in meshlets:
    t0 = time.time()
    vert_order, _, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
    t_eb += time.time() - t0

    t0 = time.time()
    local_to_global, bnd_local, int_local, _ = split_meshlet_verts(vert_order, boundary_set)
    t_split += time.time() - t0

    t0 = time.time()
    bnd_local = sort_by_morton(bnd_local, global_codes)
    int_local = sort_interior("greedy_nn", int_local, global_codes=global_codes, vert_pos_float=vn)
    local_to_global = bnd_local + int_local
    t_sort += time.time() - t0

    t0 = time.time()
    refs = [gv_to_ref[gv] for gv in bnd_local]
    _ = delta_refs_bits(refs)
    t_bnd_refs += time.time() - t0

    t0 = time.time()
    if int_local:
        quantize_interior_parallelogram(
            int_local=int_local, bnd_local=bnd_local,
            ml_tris_global_idx=ml_tris, tris_np=tris_np,
            vn=vn, bnd_recon_float=bnd_recon_global,
            per_coord_err=per_coord_err, mlp=None,
            collect_training=False)
    t_para += time.time() - t0

    t0 = time.time()
    gts_encode_meshlet(ml_tris, tris_np, tri_adj, local_to_global, variant="v3")
    t_conn += time.time() - t0

print(f"--- per-meshlet aggregate ---", flush=True)
print(f"edgebreaker order:      {t_eb:6.1f}s", flush=True)
print(f"split verts:            {t_split:6.1f}s", flush=True)
print(f"interior+morton sort:   {t_sort:6.1f}s", flush=True)
print(f"delta_refs_bits:        {t_bnd_refs:6.1f}s", flush=True)
print(f"para predictor:         {t_para:6.1f}s", flush=True)
print(f"GTS connectivity:       {t_conn:6.1f}s", flush=True)

"""Profile CUDA-path on Monkey to find remaining Python bottlenecks."""
import time
import numpy as np
import cupy as cp
from reader import Reader
from encoder.implementation.meshlet_wavelet import (
    _to_numpy, _global_quantize, _dequantize_global, _amd_packed_bits,
)
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts, edgebreaker_vertex_order,
)
from utils.boundary_split import (
    identify_boundary_verts, build_boundary_table, split_meshlet_verts,
    sort_by_morton, gts_encode_meshlet,
)
from utils.boundary_bvh import (
    morton_permute_boundary, delta_boundary_table_bits, delta_refs_bits,
)
from utils.interior_sorts import sort_interior
from utils.parallelogram_predictor_cuda import (
    pack_meshlets, encode_meshlets_cuda,
)
from utils.residual_entropy import best_axis_bits


def stage(name, dt):
    print(f"  {name:30s}  {dt:6.2f}s", flush=True)


eps = 0.0005
mv = 512
print("Loading...", flush=True)
model = Reader.read_from_file("assets/Monkey.obj")
verts_np, tris_np = _to_numpy(model)
center = verts_np.mean(axis=0)
vc = verts_np - center
scale = np.max(np.linalg.norm(vc, axis=1))
vn = vc / scale
per_coord_err = eps / scale / np.sqrt(3)
global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

t0 = time.time()
tri_adj = build_adjacency(tris_np)
fn = compute_face_normals(vn, tris_np)
fc = compute_face_centroids(vn, tris_np)
meshlets = generate_meshlets_by_verts(tris_np, tri_adj, fn, fc, max_verts=mv)
stage("setup + meshlet gen", time.time() - t0)

t0 = time.time()
boundary_set = identify_boundary_verts(meshlets, tris_np)
boundary_list, gv_to_ref, _ = build_boundary_table(boundary_set, global_codes)
boundary_list, _ = morton_permute_boundary(boundary_list, global_codes)
gv_to_ref = {gv: i for i, gv in enumerate(boundary_list)}
bnd_recon_global = _dequantize_global(global_codes, g_min, g_range, g_bits)
stage("boundary setup", time.time() - t0)

# Phase A
t0 = time.time()
ml_local_to_global = []
ml_bnd_local = []
ml_int_local = []
ml_ref_totals = []
ml_conn_bits = []
for ml_tris in meshlets:
    vert_order, _, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
    _, bnd_local, int_local, _ = split_meshlet_verts(vert_order, boundary_set)
    bnd_local = sort_by_morton(bnd_local, global_codes)
    int_local = sort_interior("greedy_nn", int_local,
                              global_codes=global_codes, vert_pos_float=vn)
    local_to_global = bnd_local + int_local
    ml_local_to_global.append(local_to_global)
    ml_bnd_local.append(bnd_local)
    ml_int_local.append(int_local)
    if len(bnd_local) > 0:
        refs = [gv_to_ref[gv] for gv in bnd_local]
        ml_ref_totals.append(delta_refs_bits(refs))
    else:
        ml_ref_totals.append(0)
    conn_bits, _ = gts_encode_meshlet(ml_tris, tris_np, tri_adj,
                                      local_to_global, variant="v3")
    ml_conn_bits.append(conn_bits)
stage("phase A (CPU pre)", time.time() - t0)

t0 = time.time()
gpu_in, meta = pack_meshlets(
    meshlets=meshlets, tris_np=tris_np, vn=vn,
    bnd_recon_global=bnd_recon_global,
    boundary_set=boundary_set,
    per_coord_err=per_coord_err,
    vert_orders=ml_local_to_global,
    sort_fn=lambda b, i: (b, i),
)
stage("pack_meshlets", time.time() - t0)

t0 = time.time()
codes_all, order_all, pos_recon_all = encode_meshlets_cuda(gpu_in, meta)
cp.cuda.runtime.deviceSynchronize()
stage("CUDA kernel + transfer", time.time() - t0)

t0 = time.time()
total_int = 0
int_off = gpu_in['int_off']
for m, _ in enumerate(meshlets):
    s = int(int_off[m])
    e = int(int_off[m + 1])
    if e > s:
        codes_m = codes_all[s:e]
        for d in range(3):
            ax_bits, _, _ = best_axis_bits(codes_m[:, d])
            total_int += ax_bits
stage("phase C (bit count)", time.time() - t0)

t0 = time.time()
n_int_arr = gpu_in['n_int']
n_bnd_arr = gpu_in['n_bnd']
v_off_arr = gpu_in['v_off']
all_err = []
for m in range(len(meshlets)):
    base = int(v_off_arr[m])
    n_bnd = int(n_bnd_arr[m])
    n_int = int(n_int_arr[m])
    if n_int > 0:
        recon = pos_recon_all[base + n_bnd:base + n_bnd + n_int]
        err = np.linalg.norm(recon - vn[ml_int_local[m]], axis=1) * scale
        all_err.extend(err.tolist())
stage("error compute", time.time() - t0)

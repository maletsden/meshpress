"""CUDA-accelerated parallelogram predictor for big meshes.

Each CUDA thread processes one meshlet. Within a meshlet, encoding is
inherently causal (vertex i depends on previous reconstructions), so a
single thread runs the greedy_order + per-vertex quantize loop. Across
meshlets there is no dependency, so we launch (M + 31)/32 blocks of 32
threads — full warp utilization, ~1500 simultaneous meshlets resident on
an RTX 3090.

Assumes `use_nn=False`. NN bias path is left to the CPU encoder; it is
already amortized by the encoder header cost on big meshes (~0.06 BPV).

Workflow:
  1. Host packs every meshlet's topology + boundary recon into flat int32 /
     float64 arrays with prefix-sum offsets.
  2. Host transfers buffers, launches kernel.
  3. Kernel emits per-meshlet `(v, kind, refs)` order + integer residual
     codes per axis.
  4. Host runs the existing per-axis Rice/EG/arith bit-count over the
     codes (best_axis_bits).

Output is bit-identical to the CPU `_greedy_order` + `quantize_interior_parallelogram`
path on the same input (verified against 20 meshlets per mesh in tests).
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


# ---------------------------------------------------------------
# Host packer
# ---------------------------------------------------------------

def pack_meshlets(meshlets, tris_np, vn, bnd_recon_global, boundary_set,
                  per_coord_err, vert_orders, sort_fn=None):
    """Build flat host buffers describing every meshlet's topology and
    initial state. Returns a dict of numpy arrays ready for upload, plus
    bookkeeping lists for unpacking results.

    Args:
        meshlets : list[np.ndarray] of global triangle indices per meshlet.
        tris_np  : (T, 3) global triangle vertex IDs.
        vn       : (V, 3) normalized float positions.
        bnd_recon_global : (V, 3) boundary-reconstructed float positions.
        boundary_set : set of global vertex IDs that are boundary.
        per_coord_err : float (delta = 2 * per_coord_err).
        vert_orders : list[list[int]] — vert order per meshlet (incl. bnd).
        sort_fn : callable(bnd_local, int_local) -> (bnd_local', int_local').

    Returns:
        gpu_in : dict of arrays to upload.
        meta   : dict of host-side metadata (per-meshlet local_to_global,
                 n_int, n_bnd, etc.) for unpacking.
    """
    M = len(meshlets)
    delta = 2.0 * per_coord_err

    # Pass 1: per-meshlet sizes and local->global maps.
    n_local = np.zeros(M, dtype=np.int32)
    n_tri = np.zeros(M, dtype=np.int32)
    n_int = np.zeros(M, dtype=np.int32)
    n_bnd = np.zeros(M, dtype=np.int32)
    local_to_global_per = []
    int_local_local_per = []
    bnd_local_local_per = []
    for m, ml_tris in enumerate(meshlets):
        vert_order = vert_orders[m]
        bnd_global = [v for v in vert_order if v in boundary_set]
        int_global = [v for v in vert_order if v not in boundary_set]
        if sort_fn is not None:
            bnd_global, int_global = sort_fn(bnd_global, int_global)
        local_to_global = bnd_global + int_global
        n_local[m] = len(local_to_global)
        n_bnd[m] = len(bnd_global)
        n_int[m] = len(int_global)
        n_tri[m] = len(ml_tris)
        local_to_global_per.append(np.asarray(local_to_global, dtype=np.int64))
        bnd_local_local_per.append(np.arange(len(bnd_global), dtype=np.int32))
        int_local_local_per.append(np.arange(
            len(bnd_global), len(local_to_global), dtype=np.int32))

    # Prefix offsets.
    v_off = np.zeros(M + 1, dtype=np.int32)
    v_off[1:] = np.cumsum(n_local)
    t_off = np.zeros(M + 1, dtype=np.int32)
    t_off[1:] = np.cumsum(n_tri)
    int_off = np.zeros(M + 1, dtype=np.int32)
    int_off[1:] = np.cumsum(n_int)

    total_v = int(v_off[-1])
    total_t = int(t_off[-1])
    total_int = int(int_off[-1])

    # Flat global ids of every local vertex (concat of local_to_global_per).
    vert_global = np.concatenate(local_to_global_per).astype(np.int64)
    # Positions and boundary mask, vectorised.
    pos_true = vn[vert_global].astype(np.float64)
    pos_recon = np.zeros((total_v, 3), dtype=np.float64)
    is_bnd = np.zeros(total_v, dtype=np.uint8)
    # is_bnd: per-meshlet first n_bnd locals are boundary.
    # Build per-vert meshlet index then per-vert local index.
    vert_meshlet = np.repeat(np.arange(M), n_local)
    vert_local_in_meshlet = np.arange(total_v) - v_off[vert_meshlet]
    bnd_mask = vert_local_in_meshlet < n_bnd[vert_meshlet]
    is_bnd[bnd_mask] = 1
    pos_recon[bnd_mask] = bnd_recon_global[vert_global[bnd_mask]]

    # Build tris_local_flat: for each tri, map its 3 global vert IDs to the
    # local index within its meshlet. Use a sorted-search lookup per meshlet
    # (vectorised over its 3*n_tri entries).
    tris_global_flat = np.zeros((total_t, 3), dtype=np.int64)
    for m, ml_tris in enumerate(meshlets):
        tris_global_flat[t_off[m]:t_off[m + 1]] = tris_np[ml_tris]
    # Per meshlet: argsort local_to_global, searchsorted.
    tris_local_flat = np.zeros((total_t, 3), dtype=np.int32)
    for m in range(M):
        ltg = local_to_global_per[m]
        order = np.argsort(ltg, kind='stable')
        sorted_ltg = ltg[order]
        ts = t_off[m]
        te = t_off[m + 1]
        if te == ts:
            continue
        flat = tris_global_flat[ts:te].reshape(-1)
        idx = np.searchsorted(sorted_ltg, flat)
        local = order[idx].astype(np.int32)
        tris_local_flat[ts:te] = local.reshape(-1, 3)

    # Per-vert vert-to-tris CSR. Each tri contributes 3 (vert_global_idx,
    # tri_local_within_meshlet) pairs. Stable sort by vert_global_idx
    # produces the CSR data array; bincount the keys to get the offset.
    tri_meshlet = np.repeat(np.arange(M), n_tri)
    vert_base_per_tri = v_off[tri_meshlet]                      # (total_t,)
    tri_local_idx = (np.arange(total_t) - t_off[tri_meshlet]).astype(np.int32)
    keys = (tris_local_flat + vert_base_per_tri[:, None]).reshape(-1)
    vals = np.repeat(tri_local_idx, 3)
    cnt = np.bincount(keys, minlength=total_v).astype(np.int32)
    v2t_off = np.zeros(total_v + 1, dtype=np.int32)
    v2t_off[1:] = np.cumsum(cnt)
    order = np.argsort(keys, kind='stable')
    v2t_data = vals[order].astype(np.int32)

    gpu_in = {
        'M': np.int32(M),
        'delta': np.float64(delta),
        'n_local': n_local,
        'n_tri': n_tri,
        'n_int': n_int,
        'n_bnd': n_bnd,
        'v_off': v_off,
        't_off': t_off,
        'int_off': int_off,
        'tris_local': tris_local_flat,
        'v2t_off': v2t_off,
        'v2t_data': v2t_data,
        'pos_true': pos_true,
        'pos_recon': pos_recon,
        'is_bnd': is_bnd,
        'vert_global': vert_global.astype(np.int32),
    }
    meta = {
        'M': M,
        'total_int': total_int,
        'local_to_global_per': local_to_global_per,
        'int_local_local_per': int_local_local_per,
        'bnd_local_local_per': bnd_local_local_per,
    }
    return gpu_in, meta


# ---------------------------------------------------------------
# CUDA kernel
# ---------------------------------------------------------------

_KERNEL_SRC = r'''
extern "C" __global__ void para_encode(
    const int M,
    const double delta,
    const int* __restrict__ n_local,
    const int* __restrict__ n_tri,
    const int* __restrict__ n_int,
    const int* __restrict__ n_bnd,
    const int* __restrict__ v_off,
    const int* __restrict__ t_off,
    const int* __restrict__ int_off,
    const int* __restrict__ tris_local,    // (total_t, 3)
    const int* __restrict__ v2t_off,
    const int* __restrict__ v2t_data,
    const double* __restrict__ pos_true,
    const int* __restrict__ vert_global,
    double* pos_recon,                     // updated in place
    long long* codes_out,                  // (total_int * 3) int64
    int* order_out                         // (total_int) local vert id
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const int nl = n_local[m];
    const int nt = n_tri[m];
    const int ni = n_int[m];
    const int nb = n_bnd[m];
    const int vbase = v_off[m];
    const int tbase = t_off[m];
    const int v2t_meshlet_base = v2t_off[vbase];
    const int ibase = int_off[m];

    // Decoded flag per local vertex. n_local can reach ~768 in pathological
    // cases; 1024 bits is the safe upper bound. 32 u32s = 1024 bits.
    unsigned int decoded[32];
    for (int i = 0; i < 32; i++) decoded[i] = 0;
    for (int v = 0; v < nb; v++) decoded[v >> 5] |= (1u << (v & 31));

    // Greedy order: at each step pick remaining vertex with lowest (rank, v).
    // remaining[i] = is interior i still unprocessed. (i is local vert id
    // within meshlet, in [nb, nl).)
    int remaining_count = ni;
    // Track encoded-axis-min/max already implicit via pos_recon write.
    int order_pos = 0;

    while (remaining_count > 0) {
        // Find best (rank, v_global). rank: 0=para, 1=mid, 2=one, 3=none.
        // Tie-break by lowest GLOBAL vertex id (matches CPU encoder).
        int best_rank = 4;
        int best_v = -1;
        int best_v_global = 0x7fffffff;
        int best_a = -1, best_b = -1, best_c = -1;
        int best_a_g = 0x7fffffff, best_b_g = 0x7fffffff, best_c_g = 0x7fffffff;
        for (int v_local = nb; v_local < nl; v_local++) {
            if (decoded[v_local >> 5] & (1u << (v_local & 31))) continue;
            int v_g = vert_global[vbase + v_local];
            int v2t_lo = v2t_off[vbase + v_local] - v2t_meshlet_base;
            int v2t_hi = v2t_off[vbase + v_local + 1] - v2t_meshlet_base;
            int v2t_pos_base = v2t_off[vbase + v_local];
            int rank_v = 3;
            int a_v = -1, b_v = -1, c_v = -1;
            int a_g = 0x7fffffff, b_g = 0x7fffffff, c_g = 0x7fffffff;
            for (int kk = v2t_pos_base; kk < v2t_pos_base + (v2t_hi - v2t_lo); kk++) {
                int t_local = v2t_data[kk];
                int t_global = tbase + t_local;
                int va = tris_local[t_global * 3 + 0];
                int vb = tris_local[t_global * 3 + 1];
                int vc = tris_local[t_global * 3 + 2];
                int o0, o1;
                if (va == v_local) { o0 = vb; o1 = vc; }
                else if (vb == v_local) { o0 = va; o1 = vc; }
                else                    { o0 = va; o1 = vb; }
                int o0_g = vert_global[vbase + o0];
                int o1_g = vert_global[vbase + o1];
                int a_loc, b_loc, a_loc_g, b_loc_g;
                if (o0_g < o1_g) {
                    a_loc = o0; b_loc = o1; a_loc_g = o0_g; b_loc_g = o1_g;
                } else {
                    a_loc = o1; b_loc = o0; a_loc_g = o1_g; b_loc_g = o0_g;
                }
                bool a_in = decoded[a_loc >> 5] & (1u << (a_loc & 31));
                bool b_in = decoded[b_loc >> 5] & (1u << (b_loc & 31));
                if (a_in && b_in) {
                    int al_lo = v2t_off[vbase + a_loc] - v2t_meshlet_base;
                    int al_hi = v2t_off[vbase + a_loc + 1] - v2t_meshlet_base;
                    int al_base = v2t_off[vbase + a_loc];
                    for (int jj = al_base; jj < al_base + (al_hi - al_lo); jj++) {
                        int t2 = v2t_data[jj];
                        if (t2 == t_local) continue;
                        int t2g = tbase + t2;
                        int p0 = tris_local[t2g*3+0];
                        int p1 = tris_local[t2g*3+1];
                        int p2 = tris_local[t2g*3+2];
                        bool has_a = (p0==a_loc)||(p1==a_loc)||(p2==a_loc);
                        bool has_b = (p0==b_loc)||(p1==b_loc)||(p2==b_loc);
                        if (!(has_a && has_b)) continue;
                        int third = (p0!=a_loc && p0!=b_loc) ? p0
                                  : (p1!=a_loc && p1!=b_loc) ? p1 : p2;
                        if (!(decoded[third>>5] & (1u<<(third&31)))) continue;
                        int third_g = vert_global[vbase + third];
                        bool take_para = false;
                        if (rank_v > 0) take_para = true;
                        else {
                            if (a_loc_g < a_g) take_para = true;
                            else if (a_loc_g == a_g && b_loc_g < b_g) take_para = true;
                            else if (a_loc_g == a_g && b_loc_g == b_g && third_g < c_g) take_para = true;
                        }
                        if (take_para) {
                            rank_v = 0;
                            a_v = a_loc; b_v = b_loc; c_v = third;
                            a_g = a_loc_g; b_g = b_loc_g; c_g = third_g;
                        }
                    }
                    // Mid: lex-min on global (a,b).
                    if (rank_v > 1) {
                        rank_v = 1;
                        a_v = a_loc; b_v = b_loc;
                        a_g = a_loc_g; b_g = b_loc_g;
                    } else if (rank_v == 1) {
                        if (a_loc_g < a_g || (a_loc_g == a_g && b_loc_g < b_g)) {
                            a_v = a_loc; b_v = b_loc;
                            a_g = a_loc_g; b_g = b_loc_g;
                        }
                    }
                } else if (a_in || b_in) {
                    int one = a_in ? a_loc : b_loc;
                    int one_g = a_in ? a_loc_g : b_loc_g;
                    if (rank_v > 2) {
                        rank_v = 2; a_v = one; a_g = one_g;
                    } else if (rank_v == 2 && one_g < a_g) {
                        a_v = one; a_g = one_g;
                    }
                }
            }
            if (rank_v < best_rank ||
                (rank_v == best_rank && v_g < best_v_global)) {
                best_rank = rank_v;
                best_v = v_local;
                best_v_global = v_g;
                best_a = a_v; best_b = b_v; best_c = c_v;
            }
        }

        // Predict.
        int v_local = best_v;
        int v_idx = vbase + v_local;
        double pred_x, pred_y, pred_z;
        if (best_rank == 0) {
            int a_idx = vbase + best_a;
            int b_idx = vbase + best_b;
            int c_idx = vbase + best_c;
            pred_x = pos_recon[a_idx*3+0] + pos_recon[b_idx*3+0] - pos_recon[c_idx*3+0];
            pred_y = pos_recon[a_idx*3+1] + pos_recon[b_idx*3+1] - pos_recon[c_idx*3+1];
            pred_z = pos_recon[a_idx*3+2] + pos_recon[b_idx*3+2] - pos_recon[c_idx*3+2];
        } else if (best_rank == 1) {
            int a_idx = vbase + best_a;
            int b_idx = vbase + best_b;
            pred_x = 0.5 * (pos_recon[a_idx*3+0] + pos_recon[b_idx*3+0]);
            pred_y = 0.5 * (pos_recon[a_idx*3+1] + pos_recon[b_idx*3+1]);
            pred_z = 0.5 * (pos_recon[a_idx*3+2] + pos_recon[b_idx*3+2]);
        } else if (best_rank == 2) {
            int a_idx = vbase + best_a;
            pred_x = pos_recon[a_idx*3+0];
            pred_y = pos_recon[a_idx*3+1];
            pred_z = pos_recon[a_idx*3+2];
        } else {
            // Fallback: mean of boundary recons.
            pred_x = 0.0; pred_y = 0.0; pred_z = 0.0;
            for (int v = 0; v < nb; v++) {
                pred_x += pos_recon[(vbase+v)*3+0];
                pred_y += pos_recon[(vbase+v)*3+1];
                pred_z += pos_recon[(vbase+v)*3+2];
            }
            if (nb > 0) {
                pred_x /= (double)nb;
                pred_y /= (double)nb;
                pred_z /= (double)nb;
            }
        }

        double tx = pos_true[v_idx*3+0];
        double ty = pos_true[v_idx*3+1];
        double tz = pos_true[v_idx*3+2];
        double rx = tx - pred_x;
        double ry = ty - pred_y;
        double rz = tz - pred_z;
        long long cx = (long long)llrint(rx / delta);
        long long cy = (long long)llrint(ry / delta);
        long long cz = (long long)llrint(rz / delta);
        double recon_x = pred_x + (double)cx * delta;
        double recon_y = pred_y + (double)cy * delta;
        double recon_z = pred_z + (double)cz * delta;
        pos_recon[v_idx*3+0] = recon_x;
        pos_recon[v_idx*3+1] = recon_y;
        pos_recon[v_idx*3+2] = recon_z;

        // Write output: codes in traversal order.
        long long out_pos = (long long)(ibase + order_pos);
        codes_out[out_pos * 3 + 0] = cx;
        codes_out[out_pos * 3 + 1] = cy;
        codes_out[out_pos * 3 + 2] = cz;
        order_out[ibase + order_pos] = v_local;

        decoded[v_local >> 5] |= (1u << (v_local & 31));
        order_pos++;
        remaining_count--;
    }
}
'''


_kernel = None


def _get_kernel():
    global _kernel
    if not _HAS_CUPY:
        raise RuntimeError("cupy not available")
    if _kernel is None:
        _kernel = cp.RawKernel(_KERNEL_SRC, "para_encode")
    return _kernel


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------

def encode_meshlets_cuda(gpu_in, meta):
    """Run the kernel. Returns (codes, order, pos_recon) on host.

    codes : (total_int, 3) int64
    order : (total_int,)  int32 — per-meshlet local vert id in pick order
                                  (concatenated by `int_off`)
    pos_recon : (total_v, 3) float64
    """
    if not _HAS_CUPY:
        raise RuntimeError("cupy not available")

    # Upload.
    n_local_d = cp.asarray(gpu_in['n_local'])
    n_tri_d = cp.asarray(gpu_in['n_tri'])
    n_int_d = cp.asarray(gpu_in['n_int'])
    n_bnd_d = cp.asarray(gpu_in['n_bnd'])
    v_off_d = cp.asarray(gpu_in['v_off'])
    t_off_d = cp.asarray(gpu_in['t_off'])
    int_off_d = cp.asarray(gpu_in['int_off'])
    tris_local_d = cp.asarray(gpu_in['tris_local'])
    v2t_off_d = cp.asarray(gpu_in['v2t_off'])
    v2t_data_d = cp.asarray(gpu_in['v2t_data'])
    pos_true_d = cp.asarray(gpu_in['pos_true'])
    pos_recon_d = cp.asarray(gpu_in['pos_recon'])
    vert_global_d = cp.asarray(gpu_in['vert_global'])

    M = int(gpu_in['M'])
    total_int = int(meta['total_int'])
    codes_d = cp.zeros((total_int, 3), dtype=cp.int64)
    order_d = cp.zeros(total_int, dtype=cp.int32)

    threads_per_block = 32
    blocks = (M + threads_per_block - 1) // threads_per_block

    kernel = _get_kernel()
    kernel((blocks,), (threads_per_block,),
           (cp.int32(M), cp.float64(gpu_in['delta']),
            n_local_d, n_tri_d, n_int_d, n_bnd_d,
            v_off_d, t_off_d, int_off_d,
            tris_local_d, v2t_off_d, v2t_data_d,
            pos_true_d, vert_global_d, pos_recon_d,
            codes_d, order_d))
    cp.cuda.runtime.deviceSynchronize()

    return (cp.asnumpy(codes_d), cp.asnumpy(order_d),
            cp.asnumpy(pos_recon_d))

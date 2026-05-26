"""Byte-level codec for MeshletParaDelta (Python, no CUDA).

Round-trippable serialization of:
  - global normalization + quantization parameters
  - Morton-permuted boundary table (axis-delta Rice)
  - per-meshlet: header, boundary refs (delta Rice), GTS v3 connectivity
    tokens, parallelogram interior residuals (best-of fixed/Rice/EG)

Encoder + decoder use LOCAL vertex IDs only inside each meshlet. Boundary
verts get local IDs [0..n_bnd); interior get [n_bnd..n_local). Both run
the same `_greedy_order` over locals so traversal matches without
transmitting the order.

Stream layout (MSB-first bit packing):

  GLOBAL HEADER
    magic        u32   'PDLT' (0x504C4454)
    version      u8    (1)
    center       3×f32
    scale        f32
    per_coord_err f32  (in normalized space)
    g_min        3×f32
    g_range      3×f32
    g_bits       3×u8
    n_v          u32
    n_t          u32
    n_boundary   u32
    n_meshlets   u32
    nn_flag      u8    (0 = no NN, 1 = MLP bias on para predictor)
    if nn_flag:
      W1 (N_HIDDEN × N_FEATURES) f32, b1 (N_HIDDEN) f32,
      W2 (N_OUT × N_HIDDEN)      f32, b2 (N_OUT)    f32

  BOUNDARY TABLE
    for axis 0..2:
      first_code  g_bits[axis]
      if n_boundary > 1: rice_k u8; then Rice(k) on zigzag(diffs) × (n-1)

  PER MESHLET
    n_bnd        u16
    n_int        u16
    n_tris       u16
    n_strips     u16

    BOUNDARY REFS
      if n_bnd > 0: first_ref u32
      if n_bnd > 1: rice_k u8; then Rice(k) on (delta − 1) × (n_bnd - 1)

    CONNECTIVITY (GTS v3)
      idx_bits = ceil(log2(n_local + 1)); reuse_bits = ceil(log2(16+1)) = 5
      for s = 0..n_strips - 1:
        strip_len u16
        ROOT × 3 verts:
          flag 1b (0=fifo, 1=idx); idx or fifo-idx
        per subsequent tri: 1b edge_code (0=newest,1=second); then vert

    INTERIOR (per-axis)
      for d 0..2:
        tag u8 (0=fixed, 1=Rice, 2=EG)
        if 0: min i16; bw u8; (code - min) × bw bits × n_int
        else (Rice/EG): k u8; coder body × n_int
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque

from utils.bit_codec import BitWriter, BitReader
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets, edgebreaker_vertex_order,
)
from utils.boundary_split import (
    identify_boundary_verts, build_boundary_table, split_meshlet_verts,
    sort_by_morton, verify_crack_free,
)
from utils.interior_sorts import sort_interior
from utils.boundary_bvh import morton_permute_boundary
from utils.residual_entropy import _zigzag, _rice_bits, _exp_golomb_bits
from utils.amd_gts import (
    generate_strips_multiseed, generate_strips_v2_seeded, generate_strips_lr,
)
try:
    from utils.amd_gts_nb import (
        generate_strips_multiseed_nb,
        generate_strips_v2_seeded_nb,
    )
    _STRIP_NB_OK = True
except ImportError:
    _STRIP_NB_OK = False
from utils.parallelogram_predictor import _greedy_order, _predict
from utils.parallelogram_nn import (
    N_FEATURES, N_HIDDEN, N_OUT, TinyMLP,
    build_frame_and_features, train_bias_mlp,
    quantize_weights, dequantize_weights,
)


MAGIC = 0x504C4454
VERSION = 4  # v4: + n_meshlets×u32 absolute bit-offset table after boundary
             # (byte-aligned; enables random-access GPU per-meshlet decode)
REUSE_BUF_SIZE = 16

PREDICTOR_PLAIN = 0
PREDICTOR_MLP = 1
PREDICTOR_LIN3 = 2
PREDICTOR_LIN9 = 3      # 9 fp32 = per-axis (w_a, w_b, w_c)
PREDICTOR_LIN3_PM = 4   # 3 int8 per meshlet, quant range [-2, 2]
PREDICTOR_LINK = 5      # K global weights on K topological neighbors
PREDICTOR_LIN5 = 6      # 5 weights over (a, b, c, d_ac, d_bc); lin3 fallback
PREDICTOR_LIN_GROW = 7  # triangular: pos p ≥ 3 → p weights on interior preds

LINK_K = 4
LIN3_PM_SCALE = 2.0     # int8 quant range ±SCALE
LIN_GROW_KMAX = 16      # cap position; p ≥ K falls back to linear3


# =====================================================================
# Shared helpers
# =====================================================================

def _idx_bits_for(n_local: int) -> int:
    return max(1, int(math.ceil(math.log2(n_local + 1))))


_REUSE_BITS = max(1, int(math.ceil(math.log2(REUSE_BUF_SIZE + 1))))


def _best_rice_k(u: np.ndarray, k_max: int = 11) -> tuple[int, int]:
    best = (0, np.iinfo(np.int64).max)
    for k in range(0, k_max + 1):
        b = _rice_bits(u, k)
        if b < best[1]:
            best = (k, b)
    return best


def _best_eg_k(u: np.ndarray, k_max: int = 7) -> tuple[int, int]:
    best = (0, np.iinfo(np.int64).max)
    for k in range(0, k_max + 1):
        b = _exp_golomb_bits(u, k)
        if b < best[1]:
            best = (k, b)
    return best


def _quantize_global(vn: np.ndarray, per_coord_err: float):
    g_min = vn.min(axis=0).astype(np.float64)
    g_range = (vn.max(axis=0) - g_min).astype(np.float64)
    step = 2.0 * per_coord_err
    g_bits = np.zeros(3, dtype=np.int32)
    codes = np.zeros((len(vn), 3), dtype=np.int64)
    for d in range(3):
        rng = float(g_range[d])
        if rng <= 0:
            g_bits[d] = 1
            continue
        nb = max(1, int(np.ceil(np.log2(rng / step + 1))))
        g_bits[d] = nb
        mx = (1 << nb) - 1
        x = (vn[:, d] - g_min[d]) / rng * mx
        codes[:, d] = np.round(x).clip(0, mx).astype(np.int64)
    return codes, g_min.astype(np.float32), g_range.astype(np.float32), g_bits


def _dequant_global(codes: np.ndarray, g_min, g_range, g_bits) -> np.ndarray:
    out = np.zeros_like(codes, dtype=np.float64)
    for d in range(3):
        mx = (1 << int(g_bits[d])) - 1
        if mx == 0:
            out[:, d] = float(g_min[d])
        else:
            out[:, d] = float(g_min[d]) + codes[:, d].astype(np.float64) / mx * float(g_range[d])
    return out


def _build_meshlet_local_topo(ml_tris_local: np.ndarray):
    """edge_to_tris + vert_to_tris over local-ID tris."""
    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    vert_to_tris: dict[int, list[int]] = {}
    for t, tri in enumerate(ml_tris_local):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for vv in (a, b, c):
            vert_to_tris.setdefault(vv, []).append(t)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edge_to_tris.setdefault(key, []).append(t)
    return edge_to_tris, vert_to_tris


def _local_face_adj(ml_tris_local: np.ndarray):
    """Face-face dual adjacency for one meshlet, keyed by local tri index."""
    edge_to_tris, _ = _build_meshlet_local_topo(ml_tris_local)
    n = len(ml_tris_local)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for tlist in edge_to_tris.values():
        if len(tlist) == 2:
            a, b = tlist
            adj[a].append(b)
            adj[b].append(a)
    return adj


def _root_orient(ml_tris_local, root_id, next_id):
    root = [int(x) for x in ml_tris_local[root_id]]
    if next_id is None:
        return root
    nxt = set(int(x) for x in ml_tris_local[next_id])
    shared = set(root) & nxt
    if len(shared) != 2:
        return root
    third = (set(root) - shared).pop()
    y, z = sorted(shared)
    return [third, y, z]


def _emit_vert(w: BitWriter, v: int, reuse_fifo: deque, idx_bits: int):
    if v in reuse_fifo:
        fi = list(reuse_fifo).index(v)
        w.write_bits(0, 1)
        w.write_bits(fi, _REUSE_BITS)
    else:
        w.write_bits(1, 1)
        w.write_bits(v, idx_bits)
    if v in reuse_fifo:
        reuse_fifo.remove(v)
    reuse_fifo.append(v)


def _grow_predict(kind, refs, recon, fallback, weights,
                  pos: int, interior_preds: list):
    """LIN_GROW pred. weights = {'lin3': (3,), 'grow': {p: (p,)}, 'k_max': int}.

    pos < 3                 → plain TG.
    3 <= pos < k_max        → use weights['grow'][pos] over interior_preds[0..pos-1].
    pos >= k_max + para     → lin3 fallback on (a, b, c).
    pos >= k_max + non-para → plain pred.
    """
    if weights is None or pos < 3:
        return _predict(kind, refs, recon, fallback)
    k_max = int(weights.get('k_max', LIN_GROW_KMAX))
    if pos < k_max:
        w_p = weights['grow'][pos]
        out = np.zeros(3, dtype=np.float64)
        for k in range(pos):
            out = out + float(w_p[k]) * recon[interior_preds[k]]
        return out
    if kind == 'para':
        a_id, b_id, c_id = refs[0], refs[1], refs[2]
        w3 = weights['lin3']
        return (float(w3[0]) * recon[a_id]
                + float(w3[1]) * recon[b_id]
                + float(w3[2]) * recon[c_id])
    return _predict(kind, refs, recon, fallback)


def _para_predict(kind, refs, recon, fallback, predictor_mode, weights,
                  meshlet_idx: int = 0):
    """Predictor dispatch for the interior parallelogram loop.

    predictor_mode:
      0 (PLAIN)     : Touma-Gotsman P = a + b − c.
      1 (MLP)       : plain + MLP curvature bias when both 2nd-ring apexes.
      2 (LIN3)      : weights = (3,). Same w across xyz.
      3 (LIN9)      : weights = (3, 3) rows = (w_a, w_b, w_c) per axis x/y/z.
      4 (LIN3_PM)   : weights = (n_meshlets, 3). Use weights[meshlet_idx].
      5 (LINK)      : K-tap on topological neighbors (handled in caller, not here).

    Returns (pred, T_frame_or_None, feats_or_None) — T_frame/feats only
    surface for MLP training collection.
    """
    base = _predict(kind, refs, recon, fallback)
    if kind != 'para' or predictor_mode == PREDICTOR_PLAIN:
        return base, None, None
    a_id, b_id, c_id, d_ac_id, d_bc_id = refs
    a_pos = recon[a_id]; b_pos = recon[b_id]; c_pos = recon[c_id]
    if predictor_mode == PREDICTOR_LIN3:
        if weights is None:
            return base, None, None
        w = weights
        return (float(w[0]) * a_pos
                + float(w[1]) * b_pos
                + float(w[2]) * c_pos), None, None
    if predictor_mode == PREDICTOR_LIN9:
        if weights is None:
            return base, None, None
        W = weights  # (3, 3): rows per axis, cols (a, b, c)
        return (W[:, 0] * a_pos + W[:, 1] * b_pos
                + W[:, 2] * c_pos), None, None
    if predictor_mode == PREDICTOR_LIN3_PM:
        if weights is None:
            return base, None, None
        w = weights[meshlet_idx]
        return (float(w[0]) * a_pos
                + float(w[1]) * b_pos
                + float(w[2]) * c_pos), None, None
    if predictor_mode == PREDICTOR_LIN5:
        # weights = dict {'lin3': (3,), 'lin5': (5,)}.
        if weights is None:
            return base, None, None
        w3 = weights['lin3']
        if d_ac_id >= 0 and d_bc_id >= 0:
            w5 = weights['lin5']
            d_ac_pos = recon[d_ac_id]; d_bc_pos = recon[d_bc_id]
            return (float(w5[0]) * a_pos
                    + float(w5[1]) * b_pos
                    + float(w5[2]) * c_pos
                    + float(w5[3]) * d_ac_pos
                    + float(w5[4]) * d_bc_pos), None, None
        return (float(w3[0]) * a_pos
                + float(w3[1]) * b_pos
                + float(w3[2]) * c_pos), None, None
    # MLP
    if d_ac_id < 0 or d_bc_id < 0:
        return base, None, None
    d_ac_pos = recon[d_ac_id]; d_bc_pos = recon[d_bc_id]
    _, T_frame, feats = build_frame_and_features(
        a_pos[None, :], b_pos[None, :], c_pos[None, :],
        d_ac_pos[None, :], d_bc_pos[None, :])
    T_frame = T_frame[0]
    feats = feats[0]
    if weights is not None:
        bias_local = weights.forward(feats[None, :])[0]
        return base + T_frame @ bias_local, T_frame, feats
    return base, T_frame, feats


def _read_vert(r: BitReader, reuse_fifo: deque, idx_bits: int) -> int:
    flag = r.read_bits(1)
    if flag == 0:
        fi = r.read_bits(_REUSE_BITS)
        v = list(reuse_fifo)[fi]
    else:
        v = r.read_bits(idx_bits)
    if v in reuse_fifo:
        reuse_fifo.remove(v)
    reuse_fifo.append(v)
    return v


# =====================================================================
# Encoder
# =====================================================================

def _plan_meshlet(ml_tris_global, tris_np_i32, tri_adj_off, tri_adj_idx, vn,
                  is_boundary, global_codes, gv_to_ref_arr,
                  strip_method: str):
    """Build per-meshlet plan via Numba kernels."""
    from utils.meshlet_plan_nb import (
        eb_vertex_order_nb, local_face_adj_csr_nb, greedy_nn_order_nb,
    )

    ml_tris_arr = np.asarray(ml_tris_global, dtype=np.int32)
    vert_order_arr, _n_root = eb_vertex_order_nb(
        ml_tris_arr, tris_np_i32, tri_adj_off, tri_adj_idx)
    vert_order = vert_order_arr.astype(np.int64)

    # Boundary / interior split via boolean mask
    bnd_mask = is_boundary[vert_order]
    bnd_local_arr = vert_order[bnd_mask]
    int_local_arr = vert_order[~bnd_mask]

    # Morton-sort boundary
    bnd_local = sort_by_morton(list(bnd_local_arr.tolist()), global_codes)

    # Greedy-NN sort interior via Numba
    if len(int_local_arr) > 1:
        int_local_arr = greedy_nn_order_nb(int_local_arr, vn, 0)
    int_local = int_local_arr.tolist()

    local_to_global = list(bnd_local) + int_local
    n_bnd = len(bnd_local)
    n_int = len(int_local)
    n_tris_m = len(ml_tris_global)
    n_local = n_bnd + n_int

    # Build ml_tris_local via searchsorted: global_v -> local_id
    l2g_arr = np.asarray(local_to_global, dtype=np.int64)
    l2g_argsort = np.argsort(l2g_arr)
    l2g_sorted = l2g_arr[l2g_argsort]
    ml_tris_global_arr = np.asarray(ml_tris_global, dtype=np.int64)
    flat_g = tris_np_i32[ml_tris_global_arr].astype(np.int64).ravel()
    pos = np.searchsorted(l2g_sorted, flat_g)
    ml_tris_local = l2g_argsort[pos].reshape(n_tris_m, 3).astype(np.int64)

    # Local face adj (CSR) via Numba; strip generator consumes CSR directly.
    local_adj_off, local_adj_idx = local_face_adj_csr_nb(ml_tris_local)

    def _build_strips_csr(off, idx):
        from utils.amd_gts_nb import generate_strips_multiseed_csr_nb
        if strip_method == "multiseed":
            return generate_strips_multiseed_csr_nb(ml_tris_local, off, idx)
        if strip_method == "v2":
            from utils.amd_gts_nb import generate_strips_v2_seeded_nb
            deg = (off[1:] - off[:-1]).astype(np.int32)
            seed = int(np.argmin(deg))
            return generate_strips_v2_seeded_nb(ml_tris_local, off, idx,
                                                deg, seed)
        # Fallback: dict-based legacy generator
        local_adj = {}
        for t in range(n_tris_m):
            local_adj[t] = [int(x) for x in idx[off[t]:off[t+1]]]
        return generate_strips_lr(ml_tris_local, local_adj)

    strips = _build_strips_csr(local_adj_off, local_adj_idx)
    n_strips = len(strips)

    refs = gv_to_ref_arr[np.asarray(bnd_local, dtype=np.int64)]
    if n_bnd > 1 and not np.all(refs[:-1] < refs[1:]):
        order = np.argsort(refs)
        refs = refs[order]
        bnd_local = [bnd_local[i] for i in order]
        local_to_global = list(bnd_local) + int_local
        l2g_arr = np.asarray(local_to_global, dtype=np.int64)
        l2g_argsort = np.argsort(l2g_arr)
        l2g_sorted = l2g_arr[l2g_argsort]
        pos = np.searchsorted(l2g_sorted, flat_g)
        ml_tris_local = l2g_argsort[pos].reshape(n_tris_m, 3).astype(np.int64)
        local_adj_off, local_adj_idx = local_face_adj_csr_nb(ml_tris_local)
        new_strips = _build_strips_csr(local_adj_off, local_adj_idx)
        if len(new_strips) != n_strips:
            raise RuntimeError("strip count changed after bnd reorder")
        strips = new_strips

    return {
        "ml_tris_global": ml_tris_global,
        "ml_tris_local": ml_tris_local,
        "local_to_global": local_to_global,
        "bnd_local": bnd_local,
        "int_local": int_local,
        "n_bnd": n_bnd,
        "n_int": n_int,
        "n_tris_m": n_tris_m,
        "n_strips": n_strips,
        "strips": strips,
        "refs": refs.astype(np.int64),
    }


def _interior_pass(plan, vn, bnd_recon_norm, delta,
                   predictor_mode, weights, collect: bool,
                   meshlet_idx: int = 0):
    """Run interior parallelogram pass.

    Returns (codes_traversal, collected) where collected is a dict:
      'lin3'        : list of (A_pos, B_pos, C_pos, true_pos) — used by
                      LIN3 / LIN9 / LIN3_PM training pass
      'mlp_feats'   : list of (13,) feat vectors (MLP collect path)
      'mlp_targets' : list of (3,) bias_local targets
    """
    n_bnd = plan["n_bnd"]
    n_int = plan["n_int"]
    n_local = n_bnd + n_int
    collected = {"lin3": [], "lin5": [],
                 "mlp_feats": [], "mlp_targets": [],
                 "grow": []}  # list of (p, preds_array, true_3d)
    if n_int == 0:
        return np.zeros((0, 3), dtype=np.int64), collected
    ml_tris_local = plan["ml_tris_local"]
    local_to_global = plan["local_to_global"]
    int_ids = list(range(n_bnd, n_local))
    bnd_ids = list(range(n_bnd))
    recon: dict[int, np.ndarray] = {}
    for lid in bnd_ids:
        recon[lid] = bnd_recon_norm[int(local_to_global[lid])].copy()
    # Numba greedy_order ignores edge_to_tris / vert_to_tris dicts (builds
    # its own CSR). Pass None to skip the costly Python dict construction.
    order = _greedy_order(int_ids, bnd_ids, ml_tris_local, None, None)
    if n_bnd > 0:
        fallback = np.mean([recon[i] for i in bnd_ids], axis=0)
    else:
        fallback = np.zeros(3, dtype=np.float64)
    codes_traversal = np.zeros((n_int, 3), dtype=np.int64)
    LIN_FAMILY = (PREDICTOR_LIN3, PREDICTOR_LIN9, PREDICTOR_LIN3_PM)
    interior_preds: list[int] = []  # interior local IDs in decode order
    for i, (v_local, kind, refs) in enumerate(order):
        if predictor_mode == PREDICTOR_LIN_GROW:
            pos = len(interior_preds)
            pred = _grow_predict(kind, refs, recon, fallback,
                                 weights, pos, interior_preds)
            T_frame, feats = None, None
        else:
            pred, T_frame, feats = _para_predict(
                kind, refs, recon, fallback, predictor_mode, weights,
                meshlet_idx=meshlet_idx)
        true = vn[int(local_to_global[v_local])].astype(np.float64)
        if collect and kind == 'para':
            a_id, b_id, c_id, d_ac_id, d_bc_id = refs
            if predictor_mode in LIN_FAMILY:
                collected["lin3"].append(
                    (recon[a_id].copy(), recon[b_id].copy(),
                     recon[c_id].copy(), true.copy()))
            elif predictor_mode == PREDICTOR_LIN5:
                if d_ac_id >= 0 and d_bc_id >= 0:
                    collected["lin5"].append(
                        (recon[a_id].copy(), recon[b_id].copy(),
                         recon[c_id].copy(),
                         recon[d_ac_id].copy(), recon[d_bc_id].copy(),
                         true.copy()))
                else:
                    collected["lin3"].append(
                        (recon[a_id].copy(), recon[b_id].copy(),
                         recon[c_id].copy(), true.copy()))
            elif predictor_mode == PREDICTOR_MLP and T_frame is not None:
                plain_pred = recon[a_id] + recon[b_id] - recon[c_id]
                bias_world = true - plain_pred
                bias_local = T_frame.T @ bias_world
                collected["mlp_feats"].append(feats)
                collected["mlp_targets"].append(bias_local)
        if collect and predictor_mode == PREDICTOR_LIN_GROW:
            pos = len(interior_preds)
            if pos >= 3:
                if pos < LIN_GROW_KMAX:
                    preds_pos = np.stack(
                        [recon[interior_preds[k]] for k in range(pos)])
                    collected["grow"].append(
                        (pos, preds_pos, true.copy()))
                elif kind == 'para':
                    a_id, b_id, c_id = refs[0], refs[1], refs[2]
                    collected["lin3"].append(
                        (recon[a_id].copy(), recon[b_id].copy(),
                         recon[c_id].copy(), true.copy()))
        residual = true - pred
        code = np.round(residual / delta).astype(np.int64)
        rec = pred + code.astype(np.float64) * delta
        recon[v_local] = rec
        codes_traversal[i] = code
        interior_preds.append(v_local)
    return codes_traversal, collected


def _write_meshlet(w, plan, codes_traversal):
    """Write one meshlet's bytes given pre-computed plan + interior codes."""
    n_bnd = plan["n_bnd"]
    n_int = plan["n_int"]
    n_tris_m = plan["n_tris_m"]
    n_strips = plan["n_strips"]
    ml_tris_local = plan["ml_tris_local"]
    refs = plan["refs"]
    strips = plan["strips"]
    n_local = n_bnd + n_int

    w.write_fixed(n_bnd, 16)
    w.write_fixed(n_int, 16)
    w.write_fixed(n_tris_m, 16)
    w.write_fixed(n_strips, 16)

    if n_bnd > 0:
        w.write_fixed(int(refs[0]), 32)
        if n_bnd > 1:
            diffs = refs[1:] - refs[:-1] - 1
            u = diffs.astype(np.int64)
            k, _ = _best_rice_k(u)
            w.write_fixed(k, 8)
            for x in u:
                w.write_rice(int(x), k)

    idx_bits = _idx_bits_for(n_local)
    reuse_fifo: deque[int] = deque(maxlen=REUSE_BUF_SIZE)
    for strip in strips:
        strip_len = len(strip)
        w.write_fixed(strip_len, 16)
        root_id = strip[0]
        next_id = strip[1] if len(strip) > 1 else None
        root = _root_orient(ml_tris_local, root_id, next_id)
        for v in root:
            _emit_vert(w, v, reuse_fifo, idx_bits)
        prev_tri = list(root)
        for li in strip[1:]:
            tri_v = [int(x) for x in ml_tris_local[li]]
            tri_set = set(tri_v)
            prev_set = set(prev_tri)
            shared = tri_set & prev_set
            new_v = next(iter(tri_set - shared))
            pair_newest = frozenset((prev_tri[1], prev_tri[2]))
            pair_second = frozenset((prev_tri[0], prev_tri[2]))
            shared_fs = frozenset(shared)
            if shared_fs == pair_newest:
                edge_code = 0
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            elif shared_fs == pair_second:
                edge_code = 1
                new_prev = [prev_tri[0], prev_tri[2], new_v]
            else:
                raise RuntimeError("oldest-edge share")
            w.write_bits(edge_code, 1)
            _emit_vert(w, new_v, reuse_fifo, idx_bits)
            prev_tri = new_prev

    if n_int == 0:
        return
    for d in range(3):
        arr = codes_traversal[:, d]
        u_arr = _zigzag(arr)
        mn = int(arr.min())
        rng = int(arr.max() - mn)
        fixed_bw = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
        fixed_total = 8 + 16 + 8 + n_int * fixed_bw
        if mn < -32768 or mn > 32767:
            fixed_total = float("inf")  # i16 header field can't carry mn
        rice_k, rice_body = _best_rice_k(u_arr)
        rice_total = 8 + 8 + rice_body
        eg_k, eg_body = _best_eg_k(u_arr)
        eg_total = 8 + 8 + eg_body
        cands = [(fixed_total, 0), (rice_total, 1), (eg_total, 2)]
        _, tag = min(cands, key=lambda t: t[0])
        if tag == 0:
            w.write_fixed(0, 8)
            if mn < -32768 or mn > 32767:
                raise RuntimeError(f"mn out of i16 range: {mn}")
            w.write_fixed(mn & 0xFFFF, 16)
            w.write_fixed(fixed_bw, 8)
            w.write_fixed_array((arr - mn).astype(np.int64), fixed_bw)
        elif tag == 1:
            w.write_fixed(1, 8)
            w.write_fixed(rice_k, 8)
            w.write_rice_array(u_arr.astype(np.int64), rice_k)
        else:
            w.write_fixed(2, 8)
            w.write_fixed(eg_k, 8)
            w.write_exp_golomb_array(u_arr.astype(np.int64), eg_k)


def _fit_linear3(samples):
    """Solve min sum_i ||true_i - (w_a·a_i + w_b·b_i + w_c·c_i)||^2 over
    flattened axes. Returns (w_a, w_b, w_c) as fp32 (header storage).
    """
    if len(samples) == 0:
        return np.array([1.0, 1.0, -1.0], dtype=np.float32)
    A = np.stack([s[0] for s in samples])
    B = np.stack([s[1] for s in samples])
    C = np.stack([s[2] for s in samples])
    T = np.stack([s[3] for s in samples])
    M = np.stack([A.ravel(), B.ravel(), C.ravel()], axis=1)
    t = T.ravel()
    w, *_ = np.linalg.lstsq(M, t, rcond=None)
    return w.astype(np.float32)


def _fit_linear9(samples):
    """Per-axis weights. Returns (3, 3) fp32: row d = (w_a, w_b, w_c) for
    axis d. 3 independent lstsq problems."""
    if len(samples) == 0:
        W = np.zeros((3, 3), dtype=np.float32)
        W[:, 0] = 1.0; W[:, 1] = 1.0; W[:, 2] = -1.0
        return W
    A = np.stack([s[0] for s in samples])  # (N, 3)
    B = np.stack([s[1] for s in samples])
    C = np.stack([s[2] for s in samples])
    T = np.stack([s[3] for s in samples])
    W = np.zeros((3, 3), dtype=np.float32)
    for d in range(3):
        M = np.stack([A[:, d], B[:, d], C[:, d]], axis=1)
        t = T[:, d]
        w, *_ = np.linalg.lstsq(M, t, rcond=None)
        W[d] = w.astype(np.float32)
    return W


def _fit_linear3_per_meshlet(samples, fallback_global):
    """Fit one (w_a, w_b, w_c) for ONE meshlet's samples. Falls back to
    `fallback_global` (3,) if meshlet has too few samples (< 3)."""
    if len(samples) < 3:
        return np.asarray(fallback_global, dtype=np.float32)
    A = np.stack([s[0] for s in samples])
    B = np.stack([s[1] for s in samples])
    C = np.stack([s[2] for s in samples])
    T = np.stack([s[3] for s in samples])
    M = np.stack([A.ravel(), B.ravel(), C.ravel()], axis=1)
    t = T.ravel()
    w, *_ = np.linalg.lstsq(M, t, rcond=None)
    return w.astype(np.float32)


def _fit_lin_grow(samples, k_max: int = LIN_GROW_KMAX):
    """samples: list of (pos, preds_array (pos, 3), true (3,)).
    Returns list[pos] -> (pos,) fp32 weights, for pos in [3, k_max)."""
    out: dict[int, np.ndarray] = {}
    by_pos: dict[int, list] = {}
    for (pos, preds, true) in samples:
        by_pos.setdefault(int(pos), []).append((preds, true))
    for p in range(3, k_max):
        bucket = by_pos.get(p, [])
        if len(bucket) < p:  # need at least p samples to solve p unknowns
            out[p] = np.zeros(p, dtype=np.float32)
            continue
        # Build (3 * N, p) and (3 * N,)
        N = len(bucket)
        M = np.zeros((3 * N, p), dtype=np.float64)
        t = np.zeros(3 * N, dtype=np.float64)
        for i, (preds, true) in enumerate(bucket):
            # preds: (p, 3). For each axis, row = preds[:, axis].
            for d in range(3):
                M[3 * i + d] = preds[:, d]
                t[3 * i + d] = true[d]
        w, *_ = np.linalg.lstsq(M, t, rcond=None)
        out[p] = w.astype(np.float32)
    return out


def _fit_linear5(samples):
    """samples: list of (A, B, C, D_ac, D_bc, true) tuples of (3,) arrays.
    Returns 5 fp32 weights minimizing axis-flattened L2."""
    if len(samples) == 0:
        return np.array([1.0, 1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    A = np.stack([s[0] for s in samples])
    B = np.stack([s[1] for s in samples])
    C = np.stack([s[2] for s in samples])
    D1 = np.stack([s[3] for s in samples])
    D2 = np.stack([s[4] for s in samples])
    T = np.stack([s[5] for s in samples])
    M = np.stack([A.ravel(), B.ravel(), C.ravel(),
                  D1.ravel(), D2.ravel()], axis=1)
    t = T.ravel()
    w, *_ = np.linalg.lstsq(M, t, rcond=None)
    return w.astype(np.float32)


def _quantize_lin3_pm(W_per_ml: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize (n_meshlets, 3) fp32 weights to int8 in range
    [-LIN3_PM_SCALE, +LIN3_PM_SCALE]. Returns (codes_int8, dequant_fp64)."""
    step = LIN3_PM_SCALE / 127.0
    clipped = np.clip(W_per_ml, -LIN3_PM_SCALE, LIN3_PM_SCALE)
    codes = np.round(clipped / step).astype(np.int8)
    dequant = codes.astype(np.float64) * step
    return codes, dequant


def prepare_paradelta(model, *, max_verts: int = 256, max_tris: int = 256,
                      precision_error: float = 0.0005,
                      precision_mode: str = "world",
                      gen_method: str = "joint_learned",
                      strip_method: str = "multiseed") -> dict:
    """Run the expensive deterministic setup (normalize, quantize, generate
    meshlets, build per-meshlet plans). Result is pickle-safe and reusable
    across predictor experiments via `encode_from_prepared`.

    precision_mode:
      "world"     — precision_error is the per-vertex Euclidean error in
                    world units (default; matches the historical API).
      "bbox_frac" — precision_error is the per-axis quantization step as a
                    *fraction of the longest bbox axis*. E.g.
                    precision_error=1/4096 matches Draco q12 / DGF tb12
                    (12-bit per-axis uniform grid).
    """
    verts = np.asarray([(v.x, v.y, v.z) for v in model.vertices],
                       dtype=np.float64)
    tris_np = np.asarray([(t.a, t.b, t.c) for t in model.triangles],
                         dtype=np.int64)
    return prepare_paradelta_arrays(
        verts, tris_np, max_verts=max_verts, max_tris=max_tris,
        precision_error=precision_error, precision_mode=precision_mode,
        gen_method=gen_method, strip_method=strip_method)


def prepare_paradelta_arrays(verts: np.ndarray, tris_np: np.ndarray,
                             *, max_verts: int = 256, max_tris: int = 256,
                             precision_error: float = 0.0005,
                             precision_mode: str = "world",
                             gen_method: str = "joint_learned",
                             strip_method: str = "multiseed") -> dict:
    """Fast path: skips Vertex/Triangle object construction."""
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    tris_np = np.ascontiguousarray(tris_np, dtype=np.int64)
    n_v, n_t = len(verts), len(tris_np)

    center = verts.mean(axis=0)
    vc = verts - center
    scale = float(np.max(np.linalg.norm(vc, axis=1)))
    vn = vc / scale
    if precision_mode == "world":
        per_coord_err = precision_error / scale / math.sqrt(3)
    elif precision_mode == "bbox_frac":
        # precision_error = per-axis grid step as a fraction of the longest
        # bbox axis. Encoder grid step in normalised coords is the matching
        # absolute fraction (bbox_extent / scale carries the source-frame
        # ratio); divide by 2 because the encoder applies delta = 2*pce.
        bbox_extent = float((verts.max(0) - verts.min(0)).max())
        per_coord_err = (precision_error * bbox_extent) / (2.0 * scale)
    else:
        raise ValueError(f"unknown precision_mode={precision_mode!r}; "
                         "expected 'world' or 'bbox_frac'")

    global_codes, g_min, g_range, g_bits = _quantize_global(vn, per_coord_err)
    bnd_recon_norm = _dequant_global(global_codes, g_min, g_range, g_bits)

    from utils.meshlet_plan_nb import build_global_adj_csr_nb
    tris_np_i32 = tris_np.astype(np.int32)
    tri_adj_off, tri_adj_idx = build_global_adj_csr_nb(tris_np_i32)
    fn = compute_face_normals(vn, tris_np)
    fc = compute_face_centroids(vn, tris_np)
    import time as _t
    _gen_t0 = _t.perf_counter()
    if gen_method == "meshopt":
        from utils.meshopt_clusterizer import build_meshlets as _meshopt_build
        meshlets = _meshopt_build(
            vn, tris_np, max_verts=max_verts, max_tris=max_tris,
            cone_weight=0.0)
    elif gen_method == "joint_learned":
        from utils.meshlet_gen_joint import LEARNED_WEIGHTS, DEFAULT_FEATURE_NORMS, _resolve_norm
        from utils.meshlet_gen_joint_nb import generate_meshlets_joint_from_csr
        w = LEARNED_WEIGHTS
        nrm = DEFAULT_FEATURE_NORMS
        pr_nm, pr_is = _resolve_norm("plane_resid", nrm)
        bp_nm, bp_is = _resolve_norm("boundary_perim", nrm)
        sc_nm, sc_is = _resolve_norm("strip_cont", nrm)
        ns_nm, ns_is = _resolve_norm("normal_sim", nrm)
        se_nm, se_is = _resolve_norm("shared_edges", nrm)
        bd_nm, bd_is = _resolve_norm("bfs_depth", nrm)
        K_const = (
            w["w1_plane_resid"] * pr_is,
            w["w2_boundary_perim"] * bp_is,
            w["w3_strip_cont"] * sc_is,
            w["w4_normal_sim"] * ns_is,
            w["w5_shared_edges"] * se_is,
            w["w6_bfs_depth"] * bd_is,
        )
        NM_const = (pr_nm, bp_nm, sc_nm, ns_nm, se_nm, bd_nm)
        meshlets = generate_meshlets_joint_from_csr(
            tris_np.astype(np.int64), tri_adj_off, tri_adj_idx,
            vn, fn, max_tris=max_tris, max_verts=max_verts,
            K_const=K_const, NM_const=NM_const)
    else:
        tri_adj = build_adjacency(tris_np)
        meshlets = generate_meshlets(
            tris_np, tri_adj, fn, fc,
            method=gen_method, max_tris=max_tris, max_verts=max_verts,
            verts_np=vn,
        )
    _gen_time_ms = (_t.perf_counter() - _gen_t0) * 1000.0

    boundary_set = identify_boundary_verts(meshlets, tris_np)
    boundary_list, _, _ = build_boundary_table(boundary_set, global_codes)
    boundary_list, _ = morton_permute_boundary(boundary_list, global_codes)
    gv_to_ref = {gv: i for i, gv in enumerate(boundary_list)}
    n_boundary = len(boundary_list)

    n_cracks, _ = verify_crack_free(
        meshlets, tris_np, global_codes, boundary_set)
    if n_cracks > 0:
        raise RuntimeError(f"crack-free check failed: {n_cracks}")

    # Dense lookup tables (built once, indexed by global vert id)
    n_v_total = len(vn)
    is_boundary = np.zeros(n_v_total, dtype=np.bool_)
    for v in boundary_set:
        is_boundary[v] = True
    gv_to_ref_arr = np.full(n_v_total, -1, dtype=np.int64)
    for gv, ri in gv_to_ref.items():
        gv_to_ref_arr[gv] = ri

    plans = [
        _plan_meshlet(ml, tris_np_i32, tri_adj_off, tri_adj_idx, vn,
                      is_boundary, global_codes, gv_to_ref_arr, strip_method)
        for ml in meshlets
    ]

    return {
        "center": center,
        "scale": scale,
        "per_coord_err": per_coord_err,
        "g_min": g_min,
        "g_range": g_range,
        "g_bits": g_bits,
        "n_v": n_v,
        "n_t": n_t,
        "n_boundary": n_boundary,
        "n_meshlets": len(meshlets),
        "boundary_list": boundary_list,
        "global_codes": global_codes,
        "bnd_recon_norm": bnd_recon_norm,
        "vn": vn,
        "plans": plans,
        "strip_method": strip_method,
        "max_verts": max_verts,
        "max_tris": max_tris,
        "precision_error": precision_error,
        "gen_method": gen_method,
        "gen_time_ms": _gen_time_ms,
    }


def encode_from_prepared(prep: dict, *, predictor: str = "linear3",
                         use_nn: bool = False,
                         nn_steps: int = 300, nn_lr: float = 1e-2,
                         verbose: bool = False) -> bytes:
    """Write ParaDelta bytes from a prepared setup. Cheap; only runs the
    interior pass(es) and writes the bitstream."""
    if use_nn:
        predictor = "mlp"
    mode_map = {"plain": PREDICTOR_PLAIN, "mlp": PREDICTOR_MLP,
                "linear3": PREDICTOR_LIN3, "linear9": PREDICTOR_LIN9,
                "lin3_pm": PREDICTOR_LIN3_PM,
                "linear5": PREDICTOR_LIN5,
                "lin_grow": PREDICTOR_LIN_GROW}
    if predictor not in mode_map:
        raise ValueError(f"unknown predictor {predictor!r}")
    predictor_mode = mode_map[predictor]

    center = prep["center"]; scale = prep["scale"]
    per_coord_err = prep["per_coord_err"]
    g_min = prep["g_min"]; g_range = prep["g_range"]; g_bits = prep["g_bits"]
    n_v = prep["n_v"]; n_t = prep["n_t"]; n_boundary = prep["n_boundary"]
    n_meshlets = prep["n_meshlets"]
    boundary_list = prep["boundary_list"]
    global_codes = prep["global_codes"]
    bnd_recon_norm = prep["bnd_recon_norm"]
    vn = prep["vn"]; plans = prep["plans"]
    delta = 2.0 * per_coord_err

    qweights = None; mlp_q = None
    lin3_w = None; lin9_W = None
    lin3_pm_codes = None; lin3_pm_dequant = None
    lin5_w = None; lin5_fallback_w = None
    grow_w = None; grow_fallback_w = None
    if predictor_mode == PREDICTOR_MLP:
        feats_all: list[np.ndarray] = []
        targets_all: list[np.ndarray] = []
        for plan in plans:
            _, c = _interior_pass(
                plan, vn, bnd_recon_norm, delta,
                PREDICTOR_MLP, weights=None, collect=True)
            feats_all.extend(c["mlp_feats"])
            targets_all.extend(c["mlp_targets"])
        feats_arr = (np.array(feats_all, dtype=np.float64)
                     if feats_all else np.zeros((0, N_FEATURES)))
        targets_arr = (np.array(targets_all, dtype=np.float64)
                       if targets_all else np.zeros((0, N_OUT)))
        if verbose:
            print(f"  NN train samples: {len(feats_arr)}")
        mlp = train_bias_mlp(
            feats_arr, targets_arr,
            n_steps=nn_steps, lr=nn_lr, verbose=verbose)
        qweights, _ = quantize_weights(mlp)
        mlp_q = dequantize_weights(qweights)
    elif predictor_mode == PREDICTOR_LIN3:
        samples = []
        for plan in plans:
            _, c = _interior_pass(
                plan, vn, bnd_recon_norm, delta,
                PREDICTOR_LIN3, weights=None, collect=True)
            samples.extend(c["lin3"])
        lin3_w = _fit_linear3(samples)
        if verbose:
            print(f"  lin3 samples: {len(samples)}  "
                  f"w=({float(lin3_w[0]):.4f}, "
                  f"{float(lin3_w[1]):.4f}, "
                  f"{float(lin3_w[2]):.4f})")
    elif predictor_mode == PREDICTOR_LIN9:
        samples = []
        for plan in plans:
            _, c = _interior_pass(
                plan, vn, bnd_recon_norm, delta,
                PREDICTOR_LIN9, weights=None, collect=True)
            samples.extend(c["lin3"])
        lin9_W = _fit_linear9(samples)
        if verbose:
            print(f"  lin9 samples: {len(samples)}  W=\n{lin9_W}")
    elif predictor_mode == PREDICTOR_LIN5:
        lin5_samples = []
        lin3_samples = []
        for plan in plans:
            _, c = _interior_pass(
                plan, vn, bnd_recon_norm, delta,
                PREDICTOR_LIN5, weights=None, collect=True)
            lin5_samples.extend(c["lin5"])
            lin3_samples.extend(c["lin3"])
        lin5_w = _fit_linear5(lin5_samples)
        lin5_fallback_w = _fit_linear3(lin3_samples)
        if verbose:
            print(f"  lin5 samples: full={len(lin5_samples)} "
                  f"partial={len(lin3_samples)}")
            print(f"    w5={lin5_w}")
            print(f"    w3_fallback={lin5_fallback_w}")
    elif predictor_mode == PREDICTOR_LIN_GROW:
        grow_samples = []
        lin3_overflow = []
        for plan in plans:
            _, c = _interior_pass(
                plan, vn, bnd_recon_norm, delta,
                PREDICTOR_LIN_GROW, weights=None, collect=True)
            grow_samples.extend(c["grow"])
            lin3_overflow.extend(c["lin3"])
        grow_w = _fit_lin_grow(grow_samples, k_max=LIN_GROW_KMAX)
        grow_fallback_w = _fit_linear3(lin3_overflow)
        if verbose:
            by_p = {}
            for (p, _, _) in grow_samples:
                by_p[p] = by_p.get(p, 0) + 1
            print(f"  lin_grow K={LIN_GROW_KMAX}  overflow_lin3="
                  f"{len(lin3_overflow)} samples")
            print(f"    samples by pos: {dict(sorted(by_p.items()))}")
            print(f"    fallback_w={grow_fallback_w}")
    elif predictor_mode == PREDICTOR_LIN3_PM:
        # Collect per-meshlet, fit per-meshlet, fall back to global on
        # under-sampled meshlets.
        per_ml_samples = []
        all_samples = []
        for plan in plans:
            _, c = _interior_pass(
                plan, vn, bnd_recon_norm, delta,
                PREDICTOR_LIN3_PM, weights=None, collect=True)
            per_ml_samples.append(c["lin3"])
            all_samples.extend(c["lin3"])
        fallback_w = _fit_linear3(all_samples)
        W_per_ml = np.zeros((len(plans), 3), dtype=np.float32)
        n_skipped = 0
        for i, smp in enumerate(per_ml_samples):
            w_i = _fit_linear3_per_meshlet(smp, fallback_w)
            W_per_ml[i] = w_i
            if len(smp) < 3:
                n_skipped += 1
        lin3_pm_codes, lin3_pm_dequant = _quantize_lin3_pm(W_per_ml)
        if verbose:
            mn = lin3_pm_dequant.min(axis=0)
            mx = lin3_pm_dequant.max(axis=0)
            print(f"  lin3_pm: {len(plans)} meshlets ({n_skipped} fallback), "
                  f"w_a∈[{mn[0]:.3f},{mx[0]:.3f}] "
                  f"w_b∈[{mn[1]:.3f},{mx[1]:.3f}] "
                  f"w_c∈[{mn[2]:.3f},{mx[2]:.3f}]")

    w = BitWriter()
    w.write_fixed(MAGIC, 32)
    w.write_fixed(VERSION, 8)
    for c in center:
        w.write_f32(float(c))
    w.write_f32(scale)
    w.write_f32(per_coord_err)
    for v in g_min:
        w.write_f32(float(v))
    for v in g_range:
        w.write_f32(float(v))
    for b in g_bits:
        w.write_fixed(int(b), 8)
    w.write_fixed(n_v, 32)
    w.write_fixed(n_t, 32)
    w.write_fixed(n_boundary, 32)
    w.write_fixed(n_meshlets, 32)
    w.write_fixed(predictor_mode, 8)
    if predictor_mode == PREDICTOR_MLP:
        for name in ('W1', 'b1', 'W2', 'b2'):
            arr = qweights[name].astype(np.float32).ravel()
            for v in arr:
                w.write_f32(float(v))
    elif predictor_mode == PREDICTOR_LIN3:
        for v in lin3_w:
            w.write_f32(float(v))
    elif predictor_mode == PREDICTOR_LIN9:
        for v in lin9_W.ravel():
            w.write_f32(float(v))
    elif predictor_mode == PREDICTOR_LIN3_PM:
        # Stream n_meshlets × 3 int8 (24 bits each)
        for row in lin3_pm_codes:
            for v in row:
                w.write_fixed(int(v) & 0xFF, 8)
    elif predictor_mode == PREDICTOR_LIN5:
        # 3 fallback fp32 + 5 main fp32 = 32 B
        for v in lin5_fallback_w:
            w.write_f32(float(v))
        for v in lin5_w:
            w.write_f32(float(v))
    elif predictor_mode == PREDICTOR_LIN_GROW:
        # K_MAX u16 + lin3 fallback (3 fp32) + sum_{p=3}^{K-1} p fp32 weights
        w.write_fixed(LIN_GROW_KMAX, 16)
        for v in grow_fallback_w:
            w.write_f32(float(v))
        for p in range(3, LIN_GROW_KMAX):
            for v in grow_w[p]:
                w.write_f32(float(v))

    if n_boundary > 0:
        bnd_codes = global_codes[boundary_list]
        for d in range(3):
            arr = bnd_codes[:, d].astype(np.int64)
            w.write_fixed(int(arr[0]), int(g_bits[d]))
            if n_boundary > 1:
                diffs = arr[1:] - arr[:-1]
                u = _zigzag(diffs)
                k, _ = _best_rice_k(u)
                w.write_fixed(k, 8)
                for x in u:
                    w.write_rice(int(x), k)

    if predictor_mode == PREDICTOR_MLP:
        active_weights = mlp_q
    elif predictor_mode == PREDICTOR_LIN3:
        active_weights = (lin3_w.astype(np.float32).astype(np.float64)
                          if lin3_w is not None else None)
    elif predictor_mode == PREDICTOR_LIN9:
        active_weights = (lin9_W.astype(np.float32).astype(np.float64)
                          if lin9_W is not None else None)
    elif predictor_mode == PREDICTOR_LIN3_PM:
        active_weights = lin3_pm_dequant
    elif predictor_mode == PREDICTOR_LIN5:
        active_weights = {
            'lin3': lin5_fallback_w.astype(np.float32).astype(np.float64),
            'lin5': lin5_w.astype(np.float32).astype(np.float64),
        }
    elif predictor_mode == PREDICTOR_LIN_GROW:
        # Quantize-to-fp32 to match decoder.
        grow_q = {p: grow_w[p].astype(np.float32).astype(np.float64)
                  for p in grow_w}
        active_weights = {
            'lin3': grow_fallback_w.astype(np.float32).astype(np.float64),
            'grow': grow_q,
        }
    else:
        active_weights = None

    # Write meshlet bodies into a separate BitWriter so we can capture
    # per-meshlet bit offsets before stamping the offset table into the
    # main stream. Random-access decode requires this.
    w_ml = BitWriter()
    meshlet_offsets_rel: list[int] = []
    for i_ml, plan in enumerate(plans):
        codes, _ = _interior_pass(
            plan, vn, bnd_recon_norm, delta,
            predictor_mode, active_weights, collect=False,
            meshlet_idx=i_ml)
        meshlet_offsets_rel.append(w_ml.bit_pos())
        _write_meshlet(w_ml, plan, codes)
    meshlet_bytes = w_ml.finalize()

    # Pad main stream to byte boundary so the offset table is byte-aligned.
    pad = (-w.bit_pos()) & 7
    if pad:
        w.write_bits(0, pad)

    # offset table starts at current bit_pos in main stream
    offset_table_start = w.bit_pos()
    offset_table_bits = n_meshlets * 32
    meshlet_region_start = offset_table_start + offset_table_bits
    assert meshlet_region_start % 8 == 0

    for rel in meshlet_offsets_rel:
        w.write_fixed(meshlet_region_start + rel, 32)

    return w.finalize() + meshlet_bytes


def encode_paradelta(model, *, max_verts: int = 256, max_tris: int = 256,
                     precision_error: float = 0.0005,
                     gen_method: str = "joint_learned",
                     strip_method: str = "multiseed",
                     predictor: str = "linear3",
                     use_nn: bool = False, nn_steps: int = 300,
                     nn_lr: float = 1e-2,
                     verbose: bool = False) -> bytes:
    """One-shot: prepare + encode_from_prepared. For experiments, call
    `prepare_paradelta` once and reuse the result via `encode_from_prepared`.
    """
    prep = prepare_paradelta(
        model, max_verts=max_verts, max_tris=max_tris,
        precision_error=precision_error,
        gen_method=gen_method, strip_method=strip_method)
    return encode_from_prepared(
        prep, predictor=predictor, use_nn=use_nn,
        nn_steps=nn_steps, nn_lr=nn_lr, verbose=verbose)


# =====================================================================
# Decoder
# =====================================================================

def decode_paradelta(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Returns (verts (V,3) float32, tris (T,3) int64) in world space."""
    r = BitReader(data)

    magic = r.read_fixed(32)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X}")
    version = r.read_fixed(8)
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    center = np.array([r.read_f32() for _ in range(3)], dtype=np.float64)
    scale = float(r.read_f32())
    per_coord_err = float(r.read_f32())
    g_min = np.array([r.read_f32() for _ in range(3)], dtype=np.float64)
    g_range = np.array([r.read_f32() for _ in range(3)], dtype=np.float64)
    g_bits = np.array([r.read_fixed(8) for _ in range(3)], dtype=np.int32)
    n_v = r.read_fixed(32)
    n_t = r.read_fixed(32)
    n_boundary = r.read_fixed(32)
    n_meshlets = r.read_fixed(32)
    predictor_mode = r.read_fixed(8)
    weights = None
    if predictor_mode == PREDICTOR_MLP:
        W1 = np.array([r.read_f32() for _ in range(N_HIDDEN * N_FEATURES)],
                      dtype=np.float32).reshape(N_HIDDEN, N_FEATURES)
        b1 = np.array([r.read_f32() for _ in range(N_HIDDEN)],
                      dtype=np.float32)
        W2 = np.array([r.read_f32() for _ in range(N_OUT * N_HIDDEN)],
                      dtype=np.float32).reshape(N_OUT, N_HIDDEN)
        b2 = np.array([r.read_f32() for _ in range(N_OUT)], dtype=np.float32)
        weights = TinyMLP(W1, b1, W2, b2)
    elif predictor_mode == PREDICTOR_LIN3:
        weights = np.array(
            [r.read_f32(), r.read_f32(), r.read_f32()], dtype=np.float64)
    elif predictor_mode == PREDICTOR_LIN9:
        weights = np.array([r.read_f32() for _ in range(9)],
                           dtype=np.float64).reshape(3, 3)
    elif predictor_mode == PREDICTOR_LIN3_PM:
        step = LIN3_PM_SCALE / 127.0
        codes = np.zeros((n_meshlets, 3), dtype=np.int8)
        for i in range(n_meshlets):
            for j in range(3):
                u = r.read_fixed(8)
                if u & 0x80:
                    u -= 0x100
                codes[i, j] = u
        weights = codes.astype(np.float64) * step
    elif predictor_mode == PREDICTOR_LIN5:
        lin3_fb = np.array([r.read_f32() for _ in range(3)], dtype=np.float64)
        lin5_main = np.array([r.read_f32() for _ in range(5)], dtype=np.float64)
        weights = {'lin3': lin3_fb, 'lin5': lin5_main}
    elif predictor_mode == PREDICTOR_LIN_GROW:
        k_max = r.read_fixed(16)
        lin3_fb = np.array([r.read_f32() for _ in range(3)], dtype=np.float64)
        grow_d = {}
        for p in range(3, k_max):
            grow_d[p] = np.array([r.read_f32() for _ in range(p)],
                                 dtype=np.float64)
        weights = {'lin3': lin3_fb, 'grow': grow_d, 'k_max': k_max}
    elif predictor_mode != PREDICTOR_PLAIN:
        raise ValueError(f"unknown predictor_mode {predictor_mode}")

    delta = 2.0 * per_coord_err

    # Boundary codes
    bnd_codes = np.zeros((n_boundary, 3), dtype=np.int64)
    for d in range(3):
        if n_boundary == 0:
            continue
        first = r.read_fixed(int(g_bits[d]))
        bnd_codes[0, d] = first
        if n_boundary > 1:
            k = r.read_fixed(8)
            for i in range(n_boundary - 1):
                u = r.read_rice(k)
                d_val = (u >> 1) ^ -(u & 1)
                bnd_codes[i + 1, d] = bnd_codes[i, d] + d_val

    # Boundary positions in normalized space
    bnd_pos_norm = np.zeros((n_boundary, 3), dtype=np.float64)
    for d in range(3):
        mx = (1 << int(g_bits[d])) - 1
        if mx == 0:
            bnd_pos_norm[:, d] = float(g_min[d])
        else:
            bnd_pos_norm[:, d] = float(g_min[d]) + bnd_codes[:, d].astype(np.float64) / mx * float(g_range[d])

    # v4: byte-align then read offset table (used by GPU decoder; CPU
    # decoder just consumes it sequentially).
    pad = (-r.bit_pos()) & 7
    if pad:
        r.read_bits(pad)
    meshlet_offsets = [r.read_fixed(32) for _ in range(n_meshlets)]
    _ = meshlet_offsets  # CPU decoder doesn't use them; GPU path will

    # Output buffers
    all_verts_norm: list[np.ndarray] = list(bnd_pos_norm)
    all_tris: list[tuple[int, int, int]] = []

    for _ml_idx in range(n_meshlets):
        n_bnd = r.read_fixed(16)
        n_int = r.read_fixed(16)
        n_tris_m = r.read_fixed(16)
        n_strips = r.read_fixed(16)
        n_local = n_bnd + n_int

        # Boundary refs
        ref_indices: list[int] = []
        if n_bnd > 0:
            first = r.read_fixed(32)
            ref_indices.append(first)
            if n_bnd > 1:
                k = r.read_fixed(8)
                prev = first
                for _ in range(n_bnd - 1):
                    u = r.read_rice(k)
                    nxt = prev + u + 1
                    ref_indices.append(nxt)
                    prev = nxt
        # Decoder-side: local IDs [0..n_bnd) map to ref_indices (which index
        # the boundary table = bnd_pos_norm) in ascending order. Local IDs
        # [n_bnd..n_local) are interior, assigned globally as we emit them.

        # We need a way to convert local meshlet vert ID into "global decoder
        # vert ID" for the output buffer.
        local_to_global_dec: list[int] = [
            ref_indices[i] for i in range(n_bnd)
        ]
        # Reserve n_int slots for interior verts (filled below). For now we
        # assign a sequential ID starting from current len(all_verts_norm).
        first_interior_global = len(all_verts_norm)
        for k in range(n_int):
            local_to_global_dec.append(first_interior_global + k)
        # Pre-extend the global vert buffer for interior (filled later)
        for _ in range(n_int):
            all_verts_norm.append(np.zeros(3, dtype=np.float64))

        # Connectivity
        idx_bits = _idx_bits_for(n_local)
        reuse_fifo: deque[int] = deque(maxlen=REUSE_BUF_SIZE)
        local_tris: list[tuple[int, int, int]] = []
        for _s in range(n_strips):
            strip_len = r.read_fixed(16)
            v0 = _read_vert(r, reuse_fifo, idx_bits)
            v1 = _read_vert(r, reuse_fifo, idx_bits)
            v2 = _read_vert(r, reuse_fifo, idx_bits)
            local_tris.append((v0, v1, v2))
            prev_tri = [v0, v1, v2]
            for _ in range(strip_len - 1):
                edge_code = r.read_bits(1)
                new_v = _read_vert(r, reuse_fifo, idx_bits)
                if edge_code == 0:
                    s1, s2 = prev_tri[1], prev_tri[2]
                    new_prev = [prev_tri[1], prev_tri[2], new_v]
                else:
                    s1, s2 = prev_tri[0], prev_tri[2]
                    new_prev = [prev_tri[0], prev_tri[2], new_v]
                local_tris.append((s1, s2, new_v))
                prev_tri = new_prev

        ml_tris_local = np.array(local_tris, dtype=np.int64)

        # Interior residuals
        if n_int > 0:
            int_ids = list(range(n_bnd, n_local))
            bnd_ids = list(range(n_bnd))

            edge_to_tris_local, vert_to_tris_local = _build_meshlet_local_topo(
                ml_tris_local)

            recon: dict[int, np.ndarray] = {}
            for lid in bnd_ids:
                gid = local_to_global_dec[lid]
                recon[lid] = all_verts_norm[gid].copy()

            order = _greedy_order(
                int_ids, bnd_ids, ml_tris_local,
                edge_to_tris_local, vert_to_tris_local)

            if n_bnd > 0:
                fallback = np.mean([recon[i] for i in bnd_ids], axis=0)
            else:
                fallback = np.zeros(3, dtype=np.float64)

            # Read per-axis codes in traversal order
            codes_traversal = np.zeros((n_int, 3), dtype=np.int64)
            for d in range(3):
                tag = r.read_fixed(8)
                if tag == 0:
                    mn_u = r.read_fixed(16)
                    # Sign-extend 16-bit
                    if mn_u & 0x8000:
                        mn = mn_u - 0x10000
                    else:
                        mn = mn_u
                    bw = r.read_fixed(8)
                    for i in range(n_int):
                        codes_traversal[i, d] = mn + r.read_fixed(bw)
                elif tag == 1:
                    k = r.read_fixed(8)
                    for i in range(n_int):
                        u = r.read_rice(k)
                        codes_traversal[i, d] = (u >> 1) ^ -(u & 1)
                elif tag == 2:
                    k = r.read_fixed(8)
                    for i in range(n_int):
                        u = r.read_exp_golomb(k)
                        codes_traversal[i, d] = (u >> 1) ^ -(u & 1)
                else:
                    raise ValueError(f"bad tag {tag}")

            interior_preds: list[int] = []
            for i, (v_local, kind, refs) in enumerate(order):
                if predictor_mode == PREDICTOR_LIN_GROW:
                    pred = _grow_predict(
                        kind, refs, recon, fallback,
                        weights, len(interior_preds), interior_preds)
                else:
                    pred, _, _ = _para_predict(
                        kind, refs, recon, fallback,
                        predictor_mode, weights, meshlet_idx=_ml_idx)
                code = codes_traversal[i].astype(np.float64)
                rec = pred + code * delta
                recon[v_local] = rec
                gid = local_to_global_dec[v_local]
                all_verts_norm[gid] = rec
                interior_preds.append(v_local)

        # Push tris with decoder global IDs
        for (a, b, c) in local_tris:
            all_tris.append((
                local_to_global_dec[a],
                local_to_global_dec[b],
                local_to_global_dec[c],
            ))

    # De-normalize verts: world = norm * scale + center
    V = np.array(all_verts_norm, dtype=np.float64) * scale + center
    T = np.array(all_tris, dtype=np.int64)
    return V.astype(np.float32), T


# =====================================================================
# Structured-arrays decoder (Phase 1 GPU prep)
# =====================================================================

# kind codes for traversal_order
KIND_PARA = 0
KIND_MID = 1
KIND_ONE = 2
KIND_NONE = 3

_KIND_MAP = {"para": KIND_PARA, "mid": KIND_MID, "one": KIND_ONE,
             "none": KIND_NONE}


def decode_paradelta_to_struct(data: bytes) -> dict:
    """Parse bitstream into flat numpy arrays for GPU consumption.

    Same byte-level parsing as `decode_paradelta` but stops short of the
    per-meshlet recon math. Output arrays:

      Globals (small):
        center (3,) f32, scale f32, delta f32
        g_min/g_range (3,) f32, g_bits (3,) i32
        predictor_mode i32, lin5_w3 (3,) f32, lin5_w5 (5,) f32 (LIN5 only)
        n_v, n_t, n_boundary, n_meshlets

      Boundary positions: bnd_pos_norm (n_boundary, 3) f32

      Per-meshlet (flat + offsets):
        ml_n_bnd, ml_n_int, ml_n_tris, ml_n_strips (n_meshlets,) i32
        ml_l2g_off, ml_l2g (sum n_local,) i32     -- local→global decoder ID
        ml_tris_off, ml_tris (sum n_tris, 3) i32  -- local tri indices
        ml_codes_off, ml_codes (sum n_int, 3) i32 -- interior residual codes
        ml_order_off, ml_order (sum n_int, 7) i32 -- traversal: v_local,kind,
                                                      a,b,c,d_ac,d_bc
                                                      (refs = -1 if unused)

    The traversal_order is computed on CPU via _greedy_order over the
    decoded local topology.
    """
    r = BitReader(data)
    magic = r.read_fixed(32)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X}")
    version = r.read_fixed(8)
    if version != VERSION:
        raise ValueError(f"unsupported version {version}; need {VERSION}")
    center = np.array([r.read_f32() for _ in range(3)], dtype=np.float32)
    scale = float(r.read_f32())
    per_coord_err = float(r.read_f32())
    g_min = np.array([r.read_f32() for _ in range(3)], dtype=np.float32)
    g_range = np.array([r.read_f32() for _ in range(3)], dtype=np.float32)
    g_bits = np.array([r.read_fixed(8) for _ in range(3)], dtype=np.int32)
    n_v = r.read_fixed(32)
    n_t = r.read_fixed(32)
    n_boundary = r.read_fixed(32)
    n_meshlets = r.read_fixed(32)
    predictor_mode = r.read_fixed(8)
    lin5_w3 = np.zeros(3, dtype=np.float32)
    lin5_w5 = np.zeros(5, dtype=np.float32)
    if predictor_mode == PREDICTOR_LIN5:
        lin5_w3 = np.array([r.read_f32() for _ in range(3)], dtype=np.float32)
        lin5_w5 = np.array([r.read_f32() for _ in range(5)], dtype=np.float32)
    elif predictor_mode != PREDICTOR_PLAIN:
        raise ValueError(
            f"decode_paradelta_to_struct: only PLAIN/LIN5 supported in Phase 1 "
            f"(got predictor_mode={predictor_mode})")

    delta = 2.0 * per_coord_err

    # Boundary table
    bnd_codes = np.zeros((n_boundary, 3), dtype=np.int64)
    for d in range(3):
        if n_boundary == 0:
            continue
        first = r.read_fixed(int(g_bits[d]))
        bnd_codes[0, d] = first
        if n_boundary > 1:
            k = r.read_fixed(8)
            for i in range(n_boundary - 1):
                u = r.read_rice(k)
                d_val = (u >> 1) ^ -(u & 1)
                bnd_codes[i + 1, d] = bnd_codes[i, d] + d_val

    bnd_pos_norm = np.zeros((n_boundary, 3), dtype=np.float32)
    for d in range(3):
        mx = (1 << int(g_bits[d])) - 1
        if mx == 0:
            bnd_pos_norm[:, d] = g_min[d]
        else:
            bnd_pos_norm[:, d] = (
                float(g_min[d])
                + bnd_codes[:, d].astype(np.float64) / mx * float(g_range[d])
            )

    # Skip meshlet offset table (v4): CPU parser walks sequentially.
    pad = (-r.bit_pos()) & 7
    if pad:
        r.read_bits(pad)
    for _ in range(n_meshlets):
        r.read_fixed(32)

    # Per-meshlet accumulators
    ml_n_bnd = np.zeros(n_meshlets, dtype=np.int32)
    ml_n_int = np.zeros(n_meshlets, dtype=np.int32)
    ml_n_tris = np.zeros(n_meshlets, dtype=np.int32)
    ml_n_strips = np.zeros(n_meshlets, dtype=np.int32)
    l2g_chunks: list[np.ndarray] = []
    tri_chunks: list[np.ndarray] = []
    code_chunks: list[np.ndarray] = []
    order_chunks: list[np.ndarray] = []

    interior_global_cursor = n_boundary

    for ml in range(n_meshlets):
        n_bnd = r.read_fixed(16)
        n_int = r.read_fixed(16)
        n_tris_m = r.read_fixed(16)
        n_strips = r.read_fixed(16)
        n_local = n_bnd + n_int
        ml_n_bnd[ml] = n_bnd
        ml_n_int[ml] = n_int
        ml_n_tris[ml] = n_tris_m
        ml_n_strips[ml] = n_strips

        # Boundary refs (delta-Rice prefix sum)
        l2g = np.zeros(n_local, dtype=np.int32)
        if n_bnd > 0:
            first = r.read_fixed(32)
            l2g[0] = first
            if n_bnd > 1:
                k = r.read_fixed(8)
                prev = first
                for i in range(1, n_bnd):
                    u = r.read_rice(k)
                    prev = prev + u + 1
                    l2g[i] = prev
        # Reserve interior global IDs
        for k in range(n_int):
            l2g[n_bnd + k] = interior_global_cursor + k
        interior_global_cursor += n_int
        l2g_chunks.append(l2g)

        # Connectivity (strip + reuse FIFO)
        idx_bits = _idx_bits_for(n_local)
        reuse_fifo: deque[int] = deque(maxlen=REUSE_BUF_SIZE)
        local_tris: list[tuple[int, int, int]] = []
        for _s in range(n_strips):
            strip_len = r.read_fixed(16)
            v0 = _read_vert(r, reuse_fifo, idx_bits)
            v1 = _read_vert(r, reuse_fifo, idx_bits)
            v2 = _read_vert(r, reuse_fifo, idx_bits)
            local_tris.append((v0, v1, v2))
            prev_tri = [v0, v1, v2]
            for _ in range(strip_len - 1):
                edge_code = r.read_bits(1)
                new_v = _read_vert(r, reuse_fifo, idx_bits)
                if edge_code == 0:
                    s1, s2 = prev_tri[1], prev_tri[2]
                    new_prev = [prev_tri[1], prev_tri[2], new_v]
                else:
                    s1, s2 = prev_tri[0], prev_tri[2]
                    new_prev = [prev_tri[0], prev_tri[2], new_v]
                local_tris.append((s1, s2, new_v))
                prev_tri = new_prev
        tri_arr = np.array(local_tris, dtype=np.int32) if local_tris \
            else np.zeros((0, 3), dtype=np.int32)
        tri_chunks.append(tri_arr)

        # Interior residual codes (3 per-axis streams, traversal order)
        codes = np.zeros((n_int, 3), dtype=np.int32)
        if n_int > 0:
            for d in range(3):
                tag = r.read_fixed(8)
                if tag == 0:
                    mn_u = r.read_fixed(16)
                    if mn_u & 0x8000:
                        mn = mn_u - 0x10000
                    else:
                        mn = mn_u
                    bw = r.read_fixed(8)
                    for i in range(n_int):
                        codes[i, d] = mn + r.read_fixed(bw)
                elif tag == 1:
                    k = r.read_fixed(8)
                    for i in range(n_int):
                        u = r.read_rice(k)
                        codes[i, d] = (u >> 1) ^ -(u & 1)
                elif tag == 2:
                    k = r.read_fixed(8)
                    for i in range(n_int):
                        u = r.read_exp_golomb(k)
                        codes[i, d] = (u >> 1) ^ -(u & 1)
                else:
                    raise ValueError(f"bad tag {tag}")
        code_chunks.append(codes)

        # Traversal order via greedy_nn over the decoded local topology
        if n_int > 0:
            edge_to_tris_local, vert_to_tris_local = \
                _build_meshlet_local_topo(tri_arr.astype(np.int64))
            order = _greedy_order(
                list(range(n_bnd, n_local)), list(range(n_bnd)),
                tri_arr.astype(np.int64),
                edge_to_tris_local, vert_to_tris_local)
            order_arr = np.full((n_int, 7), -1, dtype=np.int32)
            for i, (v_local, kind, refs) in enumerate(order):
                order_arr[i, 0] = v_local
                order_arr[i, 1] = _KIND_MAP[kind]
                for j, ref in enumerate(refs):
                    if j >= 5:
                        break
                    order_arr[i, 2 + j] = int(ref)
        else:
            order_arr = np.zeros((0, 7), dtype=np.int32)
        order_chunks.append(order_arr)

    def _flatten_with_offsets(chunks: list[np.ndarray], extra_dim: int):
        sizes = np.array([c.shape[0] for c in chunks], dtype=np.int32)
        offsets = np.concatenate(
            [[0], np.cumsum(sizes)]).astype(np.int32)  # len = n_meshlets + 1
        if extra_dim == 1:
            flat = np.concatenate(chunks).astype(np.int32) if chunks else \
                np.zeros(0, dtype=np.int32)
        else:
            flat = np.concatenate(chunks, axis=0).astype(np.int32) if chunks \
                else np.zeros((0, extra_dim), dtype=np.int32)
        return flat, offsets

    ml_l2g, ml_l2g_off = _flatten_with_offsets(l2g_chunks, 1)
    ml_tris, ml_tris_off = _flatten_with_offsets(tri_chunks, 3)
    ml_codes, ml_codes_off = _flatten_with_offsets(code_chunks, 3)
    ml_order, ml_order_off = _flatten_with_offsets(order_chunks, 7)

    return {
        # Globals
        "center": center, "scale": np.float32(scale),
        "delta": np.float32(delta),
        "g_min": g_min, "g_range": g_range, "g_bits": g_bits,
        "predictor_mode": np.int32(predictor_mode),
        "lin5_w3": lin5_w3, "lin5_w5": lin5_w5,
        "n_v": int(n_v), "n_t": int(n_t),
        "n_boundary": int(n_boundary), "n_meshlets": int(n_meshlets),
        # Boundary positions in normalized space
        "bnd_pos_norm": bnd_pos_norm,
        # Per-meshlet
        "ml_n_bnd": ml_n_bnd, "ml_n_int": ml_n_int,
        "ml_n_tris": ml_n_tris, "ml_n_strips": ml_n_strips,
        "ml_l2g": ml_l2g, "ml_l2g_off": ml_l2g_off,
        "ml_tris": ml_tris, "ml_tris_off": ml_tris_off,
        "ml_codes": ml_codes, "ml_codes_off": ml_codes_off,
        "ml_order": ml_order, "ml_order_off": ml_order_off,
    }
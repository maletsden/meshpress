"""
Meshlet-based encoders with Haar wavelet vertex compression.
Two connectivity strategies:
  1. MeshletWaveletEB  — EdgeBreaker (~1.5 bpt, sequential GPU decode)
  2. MeshletWaveletAMD — AMD pre-packed micro-indices (~5.7 bpt, parallel GPU decode)

Both support optional Bezier surface fitting (variant V3: wavelet on u,v + flat d).

GPU decode architecture:
  - 1 CUDA block per meshlet
  - Phase 1: Connectivity (EdgeBreaker: 1 thread sequential; AMD: all threads parallel)
  - Phase 2: Inverse Haar wavelet (all threads, log(N) steps)
  - Phase 3: Bezier evaluation if enabled (all threads parallel)
"""

import numpy as np
from collections import Counter

from utils.types import Model, CompressedModel
from ..encoder import Encoder
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts, edgebreaker_vertex_order, meshlet_bfs,
)
from utils.wavelet import estimate_wavelet_bits, wavelet_reconstruct_quantized
from utils.bezier import (
    fit_bezier, parameterize_pca, compute_displacements, bezier_derivatives,
    reconstruct_from_bezier, n_control_points,
)


# ============================================================
# Shared utilities
# ============================================================

def _to_numpy(model):
    n_v = len(model.vertices)
    n_t = len(model.triangles)
    verts = np.empty((n_v, 3), dtype=np.float64)
    for i, v in enumerate(model.vertices):
        verts[i] = (v.x, v.y, v.z)
    tris = np.empty((n_t, 3), dtype=np.int64)
    for i, t in enumerate(model.triangles):
        tris[i] = (t.a, t.b, t.c)
    return verts, tris


def _quantize(vals, lo, hi, bits):
    mx = (1 << bits) - 1
    norm = np.clip((vals - lo) / (hi - lo + 1e-15), 0, 1)
    return np.round(norm * mx).astype(np.int64)


def _bits_for_error(val_range, max_err):
    if max_err <= 0 or val_range <= 0:
        return 1
    return max(1, int(np.ceil(np.log2(val_range / (2 * max_err) + 1))))


def _entropy(codes):
    if len(codes) == 0:
        return 0.0
    counts = Counter(codes.tolist() if hasattr(codes, 'tolist') else list(codes))
    total = len(codes)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


def _stream_bits(codes, fixed_bits):
    n = len(codes)
    if n == 0:
        return 0
    plain = n * fixed_bits
    ent = _entropy(codes)
    arith = n * ent + 32
    return min(plain, arith)


def _flat_quantize_bits(values, max_err):
    if len(values) == 0:
        return 0
    rng = values.max() - values.min() if len(values) > 1 else 0.001
    bits = _bits_for_error(rng, max_err)
    codes = _quantize(values, values.min(), values.max(), bits)
    return _stream_bits(codes, bits) + 5 * 8  # +5B metadata


# ============================================================
# EdgeBreaker connectivity estimation
# ============================================================

def _edgebreaker_bits(opcodes, n_roots, n_local_verts):
    idx_bits = max(1, int(np.ceil(np.log2(n_local_verts + 1))))
    root_bits = n_roots * 3 * idx_bits
    if opcodes:
        counts = Counter(opcodes)
        total = len(opcodes)
        ent = -sum((c / total) * np.log2(c / total)
                    for c in counts.values() if c > 0)
        opcode_bits = total * ent + 16
    else:
        opcode_bits = 0
    return int(32 + root_bits + opcode_bits)


# ============================================================
# AMD packed micro-index connectivity estimation (raw, no strip)
# ============================================================

def _amd_packed_bits(meshlet_tris, tris_np, tri_adj):
    """AMD pre-packed: 3 local uint8 indices per triangle. Simplest GPU decode."""
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    n_local = len(vert_set)
    n_f = len(meshlet_tris)
    if n_f == 0:
        return 0
    micro_bits = min(max(1, int(np.ceil(np.log2(n_local + 1)))), 8)
    return 32 + n_f * 3 * micro_bits


def _amd_fifo_adjacency_encode(meshlet_tris, tris_np, tri_adj):
    """Real FIFO-adjacency encode using utils.connectivity.amd_encode_meshlet.

    Produces the actual token stream and bit count; also verifies decode.
    Returns (total_bits, stream, local_to_global, tris_local).
    """
    from utils.meshlet_generator import meshlet_bfs
    from utils.connectivity import amd_encode_meshlet, amd_decode_verify

    # Build local maps
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    local_verts = sorted(vert_set)
    g2l = {g: l for l, g in enumerate(local_verts)}
    n_local = len(local_verts)
    n_f = len(meshlet_tris)
    if n_f == 0:
        return 0, [], local_verts, np.zeros((0, 3), dtype=int)

    tris_local = np.zeros((n_f, 3), dtype=int)
    tri_map = {}
    for li, ti in enumerate(meshlet_tris):
        tri_map[ti] = li
        for j in range(3):
            tris_local[li, j] = g2l[int(tris_np[ti, j])]

    local_adj = [[] for _ in range(n_f)]
    for li, ti in enumerate(meshlet_tris):
        for nb in tri_adj[ti]:
            if nb in tri_map:
                local_adj[li].append(tri_map[nb])

    # BFS triangle order
    bfs_trav = meshlet_bfs(meshlet_tris, tri_adj)
    bfs_order = [tri_map[ti] for ti, _ in bfs_trav]

    # Encode
    total_bits, stream, _, _ = amd_encode_meshlet(
        meshlet_tris, tris_np, tri_adj, bfs_order,
        tris_local, local_adj, n_local)
    return total_bits, stream, local_verts, tris_local


def _amd_fifo_adjacency_bits(meshlet_tris, tris_np, tri_adj):
    """Just the bit count from the real encode."""
    bits, _, _, _ = _amd_fifo_adjacency_encode(meshlet_tris, tris_np, tri_adj)
    return bits


def _amd_gts_real_bits(meshlet_tris, tris_np, tri_adj):
    """Real (simplified) AMD GTS bit count using the encoder in utils/amd_gts.py."""
    from utils.amd_gts import gts_encode
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    local_verts = sorted(vert_set)
    g2l = {g: l for l, g in enumerate(local_verts)}
    n_local = len(local_verts)
    n_f = len(meshlet_tris)
    if n_f == 0:
        return 0

    tris_local = np.zeros((n_f, 3), dtype=int)
    tri_map = {}
    for li, ti in enumerate(meshlet_tris):
        tri_map[ti] = li
        for j in range(3):
            tris_local[li, j] = g2l[int(tris_np[ti, j])]
    local_adj = [[] for _ in range(n_f)]
    for li, ti in enumerate(meshlet_tris):
        for nb in tri_adj[ti]:
            if nb in tri_map:
                local_adj[li].append(tri_map[nb])

    bits, _ = gts_encode(tris_local, local_adj, n_local)
    return bits


# ============================================================
# AMD GTS connectivity estimation (from AMD GPUOpen article)
# Generalized Triangle Strip + inc/reuse packing
# GPU decode via countbits + firstbithigh (parallel)
# ============================================================

def _amd_gts_bits(meshlet_tris, tris_np, tri_adj):
    """AMD GTS: L/R flags + inc flags + reuse buffer.
    Estimates ~6-7 bpt. GPU-parallel via bit intrinsics."""
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    n_local = len(vert_set)
    n_f = len(meshlet_tris)
    if n_f == 0:
        return 0

    # Estimate strip quality: ~5% restart rate for good strip generation
    n_restarts = max(1, int(n_f * 0.05))
    strip_len = n_f + 2 + 2 * n_restarts  # strip entries incl. degenerates
    n_reuse = max(0, strip_len - n_local)

    lr_bits = n_f                     # 1 bit per triangle (L/R flag)
    inc_bits = strip_len              # 1 bit per strip entry (new vs reuse)
    reuse_bits = n_reuse * 8          # uint8 per reuse (local vertex index)
    header_bits = 4 * 8              # meshlet strip metadata

    return lr_bits + inc_bits + reuse_bits + header_bits


# ============================================================
# Per-meshlet vertex estimation (wavelet, optionally with Bezier)
# ============================================================

def _estimate_meshlet_vertex_bits(meshlet_tris, verts, tris_np, tri_adj,
                                   max_error, use_bezier=True, bezier_deg=2,
                                   wavelet_base=32):
    """Estimate vertex data bits for one meshlet."""
    vert_order, opcodes, n_roots = edgebreaker_vertex_order(
        meshlet_tris, tris_np, tri_adj)
    n_v = len(vert_order)

    if n_v < 3:
        return n_v * 96, opcodes, n_roots, n_v, len(meshlet_tris), 0, np.array([0.0])

    pts = verts[vert_order]
    per_coord_err = max_error / np.sqrt(3)

    if use_bezier:
        # Bezier + wavelet on u,v + flat d (Variant 3)
        u, v, _ = parameterize_pca(pts)
        cp = fit_bezier(u, v, pts, bezier_deg)
        disps, _, _ = compute_displacements(u, v, pts, cp, bezier_deg)

        Su, Sv = bezier_derivatives(u, v, cp, bezier_deg)
        max_Su = max(np.max(np.linalg.norm(Su, axis=1)), 1e-6)
        max_Sv = max(np.max(np.linalg.norm(Sv, axis=1)), 1e-6)

        u_err = per_coord_err / max_Su
        v_err = per_coord_err / max_Sv

        wu = estimate_wavelet_bits(u, u_err, wavelet_base)
        wv = estimate_wavelet_bits(v, v_err, wavelet_base)
        d_bits = _flat_quantize_bits(disps, per_coord_err)

        vertex_bits = wu["total_bits"] + wv["total_bits"] + d_bits

        # Header: Bezier CPs + PCA frame
        n_cp = n_control_points(bezier_deg)
        header_bits = n_cp * 3 * 16 + (3 * 4 + 4 * 2) * 8 + 64
        vertex_bits += header_bits

        # Accuracy
        u_r = wavelet_reconstruct_quantized(u, u_err, wavelet_base)
        v_r = wavelet_reconstruct_quantized(v, v_err, wavelet_base)
        from utils.wavelet import _bits_for_error as bfe, _quantize as qq, _dequantize as dq
        d_rng = disps.max() - disps.min() if len(disps) > 1 else 0.001
        d_b = bfe(d_rng, per_coord_err)
        d_r = dq(qq(disps, disps.min(), disps.max(), d_b), disps.min(), disps.max(), d_b)
        recon = reconstruct_from_bezier(u_r, v_r, d_r, cp, bezier_deg)
        errors = np.linalg.norm(recon - pts, axis=1)
    else:
        # Direct wavelet on x,y,z (Variant 2)
        wx = estimate_wavelet_bits(pts[:, 0], per_coord_err, wavelet_base)
        wy = estimate_wavelet_bits(pts[:, 1], per_coord_err, wavelet_base)
        wz = estimate_wavelet_bits(pts[:, 2], per_coord_err, wavelet_base)
        vertex_bits = wx["total_bits"] + wy["total_bits"] + wz["total_bits"] + 64

        x_r = wavelet_reconstruct_quantized(pts[:, 0], per_coord_err, wavelet_base)
        y_r = wavelet_reconstruct_quantized(pts[:, 1], per_coord_err, wavelet_base)
        z_r = wavelet_reconstruct_quantized(pts[:, 2], per_coord_err, wavelet_base)
        recon = np.stack([x_r, y_r, z_r], axis=1)
        errors = np.linalg.norm(recon - pts, axis=1)

    return vertex_bits, opcodes, n_roots, n_v, len(meshlet_tris), \
           wu.get("n_levels", 0) if use_bezier else wx.get("n_levels", 0), errors


# ============================================================
# Encoder: Meshlet Wavelet + EdgeBreaker
# ============================================================

class MeshletWaveletEB(Encoder):
    """Meshlet compression with Haar wavelet vertices + EdgeBreaker connectivity.
    Best compression ratio. Sequential GPU decode for connectivity."""

    def __init__(self, max_verts=256, precision_error=0.0005,
                 use_bezier=True, bezier_deg=2, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.use_bezier = use_bezier
        self.bezier_deg = bezier_deg
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)

        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        norm_err = self.precision_error / scale

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        # Global header
        total_bits = (3 * 4 + 4 + 4 + 1) * 8
        total_vtx = 0
        total_conn = 0
        total_hdr = total_bits
        all_errors = []

        for ml_tris in meshlets:
            vtx_bits, opcodes, n_roots, nv, nf, _, errors = \
                _estimate_meshlet_vertex_bits(
                    ml_tris, vn, tris_np, tri_adj, norm_err,
                    self.use_bezier, self.bezier_deg)
            conn_bits = _edgebreaker_bits(opcodes, n_roots, nv)

            total_bits += vtx_bits + conn_bits
            total_vtx += vtx_bits
            total_conn += conn_bits
            all_errors.extend((errors * scale).tolist())

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            err_arr = np.array(all_errors)
            pct = (err_arr <= self.precision_error).sum() / len(err_arr) * 100
            mode = "Bezier+Wav(uv)" if self.use_bezier else "Direct(xyz)"
            print(f"MeshletWaveletEB [{mode}] mv={self.max_verts}:")
            print(f"  {len(meshlets)} meshlets")
            print(f"  Vertex: {total_vtx/8:,.0f}B  Connectivity: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Accuracy: max={err_arr.max():.6f}  %OK={pct:.1f}%")

        data = bytes(int(np.ceil(total_bits / 8)))
        return CompressedModel(data, bpv, bpt)


# ============================================================
# Encoder: Meshlet Wavelet + AMD Packed
# ============================================================

class MeshletWaveletAMD(Encoder):
    """Meshlet compression with Haar wavelet vertices + AMD pre-packed micro-indices.
    GPU-optimized: fully parallel decode. ~3x more connectivity bits than EdgeBreaker."""

    def __init__(self, max_verts=256, precision_error=0.0005,
                 use_bezier=True, bezier_deg=2, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.use_bezier = use_bezier
        self.bezier_deg = bezier_deg
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)

        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        norm_err = self.precision_error / scale

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        total_bits = (3 * 4 + 4 + 4 + 1) * 8
        total_vtx = 0
        total_conn = 0
        all_errors = []

        for ml_tris in meshlets:
            vtx_bits, _, _, nv, nf, _, errors = \
                _estimate_meshlet_vertex_bits(
                    ml_tris, vn, tris_np, tri_adj, norm_err,
                    self.use_bezier, self.bezier_deg)
            conn_bits = _amd_packed_bits(ml_tris, tris_np, tri_adj)

            total_bits += vtx_bits + conn_bits
            total_vtx += vtx_bits
            total_conn += conn_bits
            all_errors.extend((errors * scale).tolist())

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            err_arr = np.array(all_errors)
            pct = (err_arr <= self.precision_error).sum() / len(err_arr) * 100
            mode = "Bezier+Wav(uv)" if self.use_bezier else "Direct(xyz)"
            print(f"MeshletWaveletAMD [{mode}] mv={self.max_verts}:")
            print(f"  {len(meshlets)} meshlets")
            print(f"  Vertex: {total_vtx/8:,.0f}B  Connectivity: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Accuracy: max={err_arr.max():.6f}  %OK={pct:.1f}%")
            print(f"  GPU decode: parallel (zero connectivity overhead)")

        data = bytes(int(np.ceil(total_bits / 8)))
        return CompressedModel(data, bpv, bpt)


# ============================================================
# AMD Global-Grid Quantization (crack-free, from AMD GPUOpen paper)
# Global quantization grid ensures shared vertices dequantize identically.
# Per-meshlet: quantized offset (1 uint32 per axis) + local uint16 values.
# ============================================================

class MeshletPlainAMD(Encoder):
    """AMD approach with global quantization grid (crack-free).
    Per meshlet: quantized offset per axis + local quantized vertices + packed indices.
    GPU decode: value = (offset + local) * dequant_factor + global_min"""

    def __init__(self, max_verts=256, precision_error=0.0005, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)

        # Normalize
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        norm_err = self.precision_error / scale
        per_coord_err = norm_err / np.sqrt(3)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        # ---- AMD Global Quantization Grid ----

        # Step 1: Global AABB
        global_min = np.array([vn[:, d].min() for d in range(3)])
        global_max = np.array([vn[:, d].max() for d in range(3)])
        global_delta = global_max - global_min
        global_delta[global_delta < 1e-12] = 1e-12

        # Step 2: Find largest meshlet extent per axis
        largest_meshlet_delta = np.zeros(3)
        meshlet_verts_list = []
        for ml_tris in meshlets:
            vs = set()
            for ti in ml_tris:
                for j in range(3):
                    vs.add(int(tris_np[ti, j]))
            lv = sorted(vs)
            pts = vn[lv]
            meshlet_verts_list.append((lv, pts))
            if len(pts) > 1:
                for d in range(3):
                    delta = pts[:, d].max() - pts[:, d].min()
                    largest_meshlet_delta[d] = max(largest_meshlet_delta[d], delta)

        # Step 3: Compute target bits from error constraint
        # quant_step = largest_meshlet_delta / (2^target_bits - 1)
        # Max error = quant_step / 2 ≤ per_coord_err
        # => 2^target_bits - 1 ≥ largest_meshlet_delta / (2 * per_coord_err)
        target_bits = np.zeros(3, dtype=int)
        for d in range(3):
            target_bits[d] = _bits_for_error(largest_meshlet_delta[d], per_coord_err)

        # Step 4: Compute global grid
        meshlet_quant_step = np.zeros(3)
        global_quant_states = np.zeros(3, dtype=np.int64)
        effective_bits = np.zeros(3)
        quant_factor = np.zeros(3)
        dequant_factor = np.zeros(3)

        for d in range(3):
            max_local = (1 << target_bits[d]) - 1
            meshlet_quant_step[d] = largest_meshlet_delta[d] / max_local if max_local > 0 else 1e-12
            global_quant_states[d] = max(1, int(global_delta[d] / meshlet_quant_step[d]))
            effective_bits[d] = np.log2(global_quant_states[d]) if global_quant_states[d] > 1 else 1
            quant_factor[d] = (global_quant_states[d] - 1) / global_delta[d]
            dequant_factor[d] = global_delta[d] / (global_quant_states[d] - 1)

        # ---- Encode meshlets ----

        # Global header: global_min(3f), dequant_factor(3f), n_meshlets, target_bits(3B)
        global_header_bits = (3 * 4 + 3 * 4 + 4 + 3) * 8

        total_bits = global_header_bits
        total_vtx = 0
        total_conn = 0
        all_errors = []
        n_shared_cracks = 0

        # Track all vertex reconstructions to detect cracks
        vertex_recon = {}  # global_vert_idx -> reconstructed position

        for ml_idx, ml_tris in enumerate(meshlets):
            lv, pts = meshlet_verts_list[ml_idx]
            n_mv = len(lv)
            n_mf = len(ml_tris)

            # Per-meshlet header: quantized offset per axis (3 × uint32) + counts
            meshlet_header_bits = (3 * 4 + 4) * 8  # 3 offsets + counts

            # Compute quantized meshlet offset per axis
            quant_offset = np.zeros(3, dtype=np.int64)
            for d in range(3):
                ml_min = pts[:, d].min() if n_mv > 0 else global_min[d]
                quant_offset[d] = int((ml_min - global_min[d]) * quant_factor[d] + 0.5)

            # Quantize vertices using global grid, then shift to local
            vertex_bits = 0
            recon = np.zeros_like(pts)
            for d in range(3):
                local_codes = np.zeros(n_mv, dtype=np.int64)
                for i in range(n_mv):
                    # Global quantize
                    global_q = int((pts[i, d] - global_min[d]) * quant_factor[d] + 0.5)
                    global_q = max(0, min(int(global_quant_states[d] - 1), global_q))
                    # Local = global - offset
                    local_q = global_q - quant_offset[d]
                    local_codes[i] = local_q
                    # Dequantize: (offset + local) * dequant_factor + global_min
                    recon[i, d] = (quant_offset[d] + local_q) * dequant_factor[d] + global_min[d]

                vertex_bits += n_mv * target_bits[d]

            errors = np.linalg.norm(recon - pts, axis=1)
            all_errors.extend((errors * scale).tolist())

            # Check crack-free: shared vertices should get identical reconstruction
            for i, gv in enumerate(lv):
                recon_pos = tuple(recon[i])
                if gv in vertex_recon:
                    if vertex_recon[gv] != recon_pos:
                        n_shared_cracks += 1
                else:
                    vertex_recon[gv] = recon_pos

            # Connectivity
            conn_bits = _amd_packed_bits(ml_tris, tris_np, tri_adj)

            meshlet_total = meshlet_header_bits + vertex_bits + conn_bits
            total_bits += meshlet_total
            total_vtx += meshlet_header_bits + vertex_bits
            total_conn += conn_bits

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            err_arr = np.array(all_errors)
            pct = (err_arr <= self.precision_error).sum() / len(err_arr) * 100
            print(f"MeshletPlainAMD (global grid, crack-free) mv={self.max_verts}:")
            print(f"  {len(meshlets)} meshlets")
            print(f"  Target bits per axis: [{target_bits[0]}, {target_bits[1]}, {target_bits[2]}]")
            print(f"  Effective bits: [{effective_bits[0]:.1f}, {effective_bits[1]:.1f}, {effective_bits[2]:.1f}]")
            print(f"  Shared vertex cracks: {n_shared_cracks}")
            print(f"  Vertex: {total_vtx/8:,.0f}B  Connectivity: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Accuracy: max={err_arr.max():.6f}  %OK={pct:.1f}%")

        data = bytes(int(np.ceil(total_bits / 8)))
        return CompressedModel(data, bpv, bpt)


# ============================================================
# Solution 1: Global quantize → integer wavelet (crack-free)
# ============================================================

def _global_quantize(vn, per_coord_err):
    """Quantize all vertices to a global integer grid."""
    global_min = vn.min(axis=0)
    global_max = vn.max(axis=0)
    global_range = global_max - global_min
    global_range[global_range < 1e-12] = 1e-12
    global_bits = np.array([_bits_for_error(global_range[d], per_coord_err)
                            for d in range(3)])
    codes = np.zeros((len(vn), 3), dtype=np.int64)
    for d in range(3):
        mx = (1 << global_bits[d]) - 1
        codes[:, d] = np.round((vn[:, d] - global_min[d]) / global_range[d] * mx
                                ).clip(0, mx).astype(np.int64)
    return codes, global_min, global_range, global_bits


def _local_quantize_meshlet(positions, per_coord_err):
    """Quantize a subset of vertex positions (one meshlet's interior) to
    a LOCAL integer grid with per-coord bit counts fitted to the subset's bbox.

    Args:
        positions: (n, 3) float positions of verts in this meshlet's interior
        per_coord_err: target quantization error per coordinate (same scale as positions)

    Returns:
        codes: (n, 3) int64 quantization codes
        local_min: (3,) float origin of local grid
        local_range: (3,) float range per axis
        local_bits: (3,) int bit count per axis
    """
    if len(positions) == 0:
        return (np.zeros((0, 3), dtype=np.int64),
                np.zeros(3), np.ones(3), np.array([1, 1, 1]))
    local_min = positions.min(axis=0)
    local_max = positions.max(axis=0)
    local_range = local_max - local_min
    local_range[local_range < 1e-12] = 1e-12
    local_bits = np.array([_bits_for_error(local_range[d], per_coord_err)
                            for d in range(3)])
    codes = np.zeros((len(positions), 3), dtype=np.int64)
    for d in range(3):
        mx = (1 << local_bits[d]) - 1
        codes[:, d] = np.round(
            (positions[:, d] - local_min[d]) / local_range[d] * mx
        ).clip(0, mx).astype(np.int64)
    return codes, local_min, local_range, local_bits


def _dequantize_global(codes, global_min, global_range, global_bits):
    result = np.zeros((len(codes), 3), dtype=np.float64)
    for d in range(3):
        mx = (1 << global_bits[d]) - 1
        result[:, d] = codes[:, d].astype(np.float64) / mx * global_range[d] + global_min[d]
    return result


class MeshletWaveletGlobalEB(Encoder):
    """Solution 1: Global quantize → per-meshlet integer Haar wavelet + EdgeBreaker.
    Crack-free: integer wavelet is lossless on global-quantized input."""

    def __init__(self, max_verts=256, precision_error=0.0005, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        from utils.wavelet import estimate_wavelet_bits_int
        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        total_bits = (3*4 + 3*4 + 3 + 4) * 8
        total_vtx = 0; total_conn = 0
        vertex_recon = {}; n_cracks = 0

        for ml_tris in meshlets:
            vert_order, opcodes, n_roots = edgebreaker_vertex_order(
                ml_tris, tris_np, tri_adj)
            n_mv = len(vert_order)
            if n_mv < 1: continue
            int_pts = global_codes[vert_order]
            ml_vtx = 12 * 8  # per-meshlet header
            for d in range(3):
                w = estimate_wavelet_bits_int(int_pts[:, d], target_base=32)
                ml_vtx += w["total_bits"]
            total_vtx += ml_vtx
            conn = _edgebreaker_bits(opcodes, n_roots, n_mv)
            total_conn += conn
            total_bits += ml_vtx + conn
            # Crack check
            dq = _dequantize_global(int_pts, g_min, g_range, g_bits)
            for i, gv in enumerate(vert_order):
                pos = tuple(dq[i])
                if gv in vertex_recon:
                    if vertex_recon[gv] != pos: n_cracks += 1
                else: vertex_recon[gv] = pos

        all_recon = _dequantize_global(global_codes, g_min, g_range, g_bits)
        errors = np.linalg.norm(all_recon - vn, axis=1) * scale
        bpv = total_bits / n_v; bpt = total_bits / n_t

        if self.verbose:
            pct = (errors <= self.precision_error).sum() / n_v * 100
            print(f"MeshletWaveletGlobalEB (crack-free) mv={self.max_verts}:")
            print(f"  {len(meshlets)} meshlets, global bits=[{g_bits[0]},{g_bits[1]},{g_bits[2]}]")
            print(f"  Cracks: {n_cracks}")
            print(f"  Vertex: {total_vtx/8:,.0f}B  Conn: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Accuracy: max={errors.max():.6f}  %OK={pct:.1f}%")
        return CompressedModel(bytes(int(np.ceil(total_bits / 8))), bpv, bpt)


class MeshletWaveletGlobalAMD(Encoder):
    """Solution 1 with AMD packed. Crack-free + parallel GPU decode."""

    def __init__(self, max_verts=256, precision_error=0.0005, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        from utils.wavelet import estimate_wavelet_bits_int
        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)
        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)
        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        total_bits = (3*4 + 3*4 + 3 + 4) * 8
        total_vtx = 0; total_conn = 0
        for ml_tris in meshlets:
            vert_order, _, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
            n_mv = len(vert_order)
            if n_mv < 1: continue
            int_pts = global_codes[vert_order]
            ml_vtx = 12 * 8
            for d in range(3):
                ml_vtx += estimate_wavelet_bits_int(int_pts[:, d], 32)["total_bits"]
            total_vtx += ml_vtx
            conn = _amd_packed_bits(ml_tris, tris_np, tri_adj)
            total_conn += conn
            total_bits += ml_vtx + conn

        all_recon = _dequantize_global(global_codes, g_min, g_range, g_bits)
        errors = np.linalg.norm(all_recon - vn, axis=1) * scale
        bpv = total_bits / n_v; bpt = total_bits / n_t
        if self.verbose:
            pct = (errors <= self.precision_error).sum() / n_v * 100
            print(f"MeshletWaveletGlobalAMD (crack-free) mv={self.max_verts}:")
            print(f"  {len(meshlets)} meshlets, global bits=[{g_bits[0]},{g_bits[1]},{g_bits[2]}]")
            print(f"  Vertex: {total_vtx/8:,.0f}B  Conn: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Accuracy: max={errors.max():.6f}  %OK={pct:.1f}%")
        return CompressedModel(bytes(int(np.ceil(total_bits / 8))), bpv, bpt)


# ============================================================
# Boundary / interior split (scaffold for float & neural wavelets)
# Layout: global boundary table + per-meshlet [boundary refs | interior codes].
# Interior here uses per-meshlet integer codes (lossless vs. global grid),
# providing a baseline that pluggable float/neural transforms replace later.
# See utils/boundary_split.py for the layout spec.
# ============================================================

class MeshletSplitAMD(Encoder):
    """Boundary/interior split with AMD packed connectivity (crack-free).

    Boundary verts live in a single global table (bitwise identical across
    meshlets). Interior verts are per-meshlet, stored as lossless integer
    deltas from a meshlet-local origin on the global grid. This is the
    scaffold that later variants swap out for float/learned wavelet encoders
    on the interior stream only.
    """

    def __init__(self, max_verts=256, precision_error=0.0005,
                 conn="gts_v3", sort="morton", verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.conn = conn  # 'packed' or 'gts'
        self.sort = sort  # 'eb' or 'morton'
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        from utils.boundary_split import (
            identify_boundary_verts, build_boundary_table,
            split_meshlet_verts, boundary_table_bits, ref_bits_for,
            quantize_interior_local, verify_crack_free,
            sort_by_morton, gts_encode_meshlet,
        )

        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        boundary_set = identify_boundary_verts(meshlets, tris_np)
        boundary_list, gv_to_ref, _boundary_codes = build_boundary_table(
            boundary_set, global_codes)
        n_boundary = len(boundary_list)
        ref_bits = ref_bits_for(n_boundary)

        # Global header: g_min(12) + g_range(12) + g_bits(3) + n_meshlets(4)
        #                + n_boundary(4) + ref_bits(1) = 36B
        global_header_bits = 36 * 8
        bnd_table_bits = boundary_table_bits(n_boundary, g_bits)

        total_bits = global_header_bits + bnd_table_bits
        total_bnd_refs = 0
        total_interior = 0
        total_conn = 0
        total_ml_hdr = 0

        for ml_tris in meshlets:
            vert_order, _, _ = edgebreaker_vertex_order(
                ml_tris, tris_np, tri_adj)
            n_mv = len(vert_order)
            if n_mv < 1:
                continue

            local_to_global, bnd_local, int_local, _remap = split_meshlet_verts(
                vert_order, boundary_set)

            # Boundary always Morton-sorted; interior dispatches on self.sort.
            # Accepts "morton" | "eb" | "hilbert" | "pca" | "greedy_nn".
            if self.sort != "eb":
                from utils.interior_sorts import sort_interior as _si
                bnd_local = sort_by_morton(bnd_local, global_codes)
                int_local = _si(
                    self.sort, int_local,
                    global_codes=global_codes,
                    vert_pos_float=vn,
                )
                local_to_global = bnd_local + int_local

            n_bnd = len(bnd_local)
            n_int = len(int_local)

            # Per-meshlet header:
            #   n_bnd(1) + n_int(1) + interior_offset(12) + interior_bits(3) = 17B
            ml_hdr_bits = 17 * 8
            total_ml_hdr += ml_hdr_bits

            # Boundary refs
            ref_total = n_bnd * ref_bits
            total_bnd_refs += ref_total

            # Interior codes (lossless integer deltas on the global grid)
            int_codes_global = global_codes[int_local] if n_int > 0 else \
                np.zeros((0, 3), dtype=np.int64)
            _off, _lb, _codes, interior_bits = quantize_interior_local(
                int_codes_global)
            total_interior += interior_bits

            if self.conn in ("gts", "gts_v2", "gts_v3"):
                variant = {"gts": "v1", "gts_v2": "v2", "gts_v3": "v3"}[self.conn]
                conn_bits, _ = gts_encode_meshlet(
                    ml_tris, tris_np, tri_adj, local_to_global, variant=variant)
            else:
                conn_bits = _amd_packed_bits(ml_tris, tris_np, tri_adj)
            total_conn += conn_bits

            total_bits += ml_hdr_bits + ref_total + interior_bits + conn_bits

        # Crack verification on the integer codes (boundary dedup check)
        n_cracks, n_shared = verify_crack_free(
            meshlets, tris_np, global_codes, boundary_set)

        # Full-mesh error (interior is lossless vs global codes, so
        # reconstruction error equals global quantization error for all verts)
        all_recon = _dequantize_global(global_codes, g_min, g_range, g_bits)
        errors = np.linalg.norm(all_recon - vn, axis=1) * scale

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            pct = (errors <= self.precision_error).sum() / n_v * 100
            bnd_frac = n_boundary / n_v * 100 if n_v else 0.0
            print(f"MeshletSplitAMD (boundary/interior, crack-free) "
                  f"mv={self.max_verts}  conn={self.conn}  sort={self.sort}:")
            print(f"  {len(meshlets)} meshlets, {n_boundary} boundary verts "
                  f"({bnd_frac:.1f}% of total)")
            print(f"  Global bits=[{g_bits[0]},{g_bits[1]},{g_bits[2]}], "
                  f"ref_bits={ref_bits}")
            print(f"  Cracks: {n_cracks} / {n_shared} shared checks")
            print(f"  Boundary table: {bnd_table_bits/8:,.0f}B")
            print(f"  Boundary refs: {total_bnd_refs/8:,.0f}B")
            print(f"  Interior codes: {total_interior/8:,.0f}B")
            print(f"  Meshlet hdrs: {total_ml_hdr/8:,.0f}B  "
                  f"Conn: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  "
                  f"BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Accuracy: max={errors.max():.6f}  %OK={pct:.1f}%")

        return CompressedModel(bytes(int(np.ceil(total_bits / 8))), bpv, bpt)


# ============================================================
# Boundary/interior split with float Haar wavelet on interior.
# Boundary stays on the global int grid (crack-free); interior is lossy float
# Haar with uniform coefficient quantization tuned to per_coord_err.
# ============================================================

class MeshletSplitFloatWaveletAMD(Encoder):
    """Boundary/interior split + float wavelet on interior vertices.

    Boundary verts: global int grid (bitwise identical across meshlets).
    Interior verts: original floats transformed by lifting wavelet
    ('haar' or 'cdf53'), per-level uniformly quantized with a schedule
    ('uniform' or 'geometric') tuned so reconstruction error stays within
    per_coord_err.
    """

    def __init__(self, max_verts=256, precision_error=0.0005,
                 wavelet="haar", schedule="geometric", ratio=2.0,
                 conn="gts_v3", sort="greedy_nn", pack_meta=True,
                 target_base=32, dct_block_size=4,
                 dct_schedule="uniform", dct_ratio=2.0, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.wavelet = wavelet
        self.schedule = schedule
        self.ratio = ratio
        self.conn = conn  # 'packed' | 'gts' | 'gts_v2' | 'gts_v3'
        self.sort = sort  # 'eb' | 'morton' | 'hilbert' | 'pca' | 'greedy_nn'
        self.pack_meta = pack_meta  # True: 7B/level shared; False: 5B/stream
        self.target_base = target_base
        # Optional block-DCT alternative to the wavelet on the interior stream.
        # 0 disables (current baseline). If > 0, encodes each meshlet with
        # both the wavelet AND a block-DCT variant, picks whichever gives the
        # smaller bit cost, and prepends a 1-bit per-meshlet selector flag.
        self.dct_block_size = dct_block_size
        self.dct_schedule = dct_schedule
        self.dct_ratio = dct_ratio
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        from utils.boundary_split import (
            identify_boundary_verts, build_boundary_table,
            split_meshlet_verts, boundary_table_bits, ref_bits_for,
            verify_crack_free, sort_by_morton, gts_encode_meshlet,
        )
        from utils.interior_sorts import sort_interior
        from utils.float_wavelet import (
            quantize_interior_float_wavelet,
            quantize_interior_float_wavelet_packed,
        )

        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        boundary_set = identify_boundary_verts(meshlets, tris_np)
        boundary_list, gv_to_ref, _bnd_codes = build_boundary_table(
            boundary_set, global_codes)
        n_boundary = len(boundary_list)
        ref_bits = ref_bits_for(n_boundary)

        # Global header: g_min(12) + g_range(12) + g_bits(3) + n_meshlets(4)
        #                + n_boundary(4) + ref_bits(1) = 36B
        global_header_bits = 36 * 8
        bnd_table_bits = boundary_table_bits(n_boundary, g_bits)

        total_bits = global_header_bits + bnd_table_bits
        total_bnd_refs = 0
        total_interior = 0
        total_conn = 0
        total_ml_hdr = 0

        # Track actual interior reconstruction to measure real error
        interior_errors = []
        boundary_errors = []

        # Dequantized boundary (used for boundary reconstruction error)
        bnd_recon_global = _dequantize_global(
            global_codes, g_min, g_range, g_bits)

        for ml_tris in meshlets:
            vert_order, _, _ = edgebreaker_vertex_order(
                ml_tris, tris_np, tri_adj)
            n_mv = len(vert_order)
            if n_mv < 1:
                continue

            local_to_global, bnd_local, int_local, _remap = split_meshlet_verts(
                vert_order, boundary_set)

            # Boundary always Morton-sorted (existing behavior).
            # Interior dispatches on self.sort, which now accepts:
            #   "morton" | "eb" | "hilbert" | "pca" | "greedy_nn".
            if self.sort != "eb":
                bnd_local = sort_by_morton(bnd_local, global_codes)
                int_local = sort_interior(
                    self.sort, int_local,
                    global_codes=global_codes,
                    vert_pos_float=vn,
                )
                local_to_global = bnd_local + int_local

            n_bnd = len(bnd_local)
            n_int = len(int_local)

            # Per-meshlet header: n_bnd(1) + n_int(1) + reserved(2) = 4B.
            # Interior float wavelet carries its own per-axis offset + q_step.
            ml_hdr_bits = 4 * 8
            total_ml_hdr += ml_hdr_bits

            ref_total = n_bnd * ref_bits
            total_bnd_refs += ref_total

            # Interior float wavelet (optionally with block-DCT alternative)
            if n_int > 0:
                int_pts = vn[int_local]
                quant_fn = (quantize_interior_float_wavelet_packed
                            if self.pack_meta
                            else quantize_interior_float_wavelet)
                wave_recon, wave_bits, _meta = quant_fn(
                    int_pts, per_coord_err,
                    wavelet=self.wavelet,
                    schedule=self.schedule,
                    ratio=self.ratio,
                    target_base=self.target_base,
                )

                if self.dct_block_size > 0:
                    from utils.block_dct import quantize_interior_dct_packed
                    dct_recon, dct_bits, _ = quantize_interior_dct_packed(
                        int_pts, per_coord_err,
                        block_size=self.dct_block_size,
                        schedule=self.dct_schedule,
                        ratio=self.dct_ratio,
                    )
                    # Per-meshlet 1-bit flag picks the smaller branch.
                    if dct_bits + 1 < wave_bits + 1:
                        int_recon, int_bits = dct_recon, dct_bits + 1
                    else:
                        int_recon, int_bits = wave_recon, wave_bits + 1
                else:
                    int_recon, int_bits = wave_recon, wave_bits

                total_interior += int_bits
                # Measure real error in ORIGINAL (unnormalized) units
                err = np.linalg.norm(int_recon - int_pts, axis=1) * scale
                interior_errors.extend(err.tolist())

            # Boundary error (from global int grid dequantization)
            if n_bnd > 0:
                for gv in bnd_local:
                    err = np.linalg.norm(bnd_recon_global[gv] - vn[gv]) * scale
                    boundary_errors.append(err)

            if self.conn in ("gts", "gts_v2", "gts_v3"):
                variant = {"gts": "v1", "gts_v2": "v2", "gts_v3": "v3"}[self.conn]
                conn_bits, _ = gts_encode_meshlet(
                    ml_tris, tris_np, tri_adj, local_to_global, variant=variant)
            else:
                conn_bits = _amd_packed_bits(ml_tris, tris_np, tri_adj)
            total_conn += conn_bits

            total_bits += ml_hdr_bits + ref_total + \
                (int_bits if n_int > 0 else 0) + conn_bits

        # Crack verification (boundary is still on the global int grid)
        n_cracks, n_shared = verify_crack_free(
            meshlets, tris_np, global_codes, boundary_set)

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            import numpy as _np
            bnd_err = _np.array(boundary_errors) if boundary_errors else _np.zeros(1)
            int_err = _np.array(interior_errors) if interior_errors else _np.zeros(1)
            combined = _np.concatenate([bnd_err, int_err])
            pct = (combined <= self.precision_error).sum() / len(combined) * 100
            bnd_frac = n_boundary / n_v * 100 if n_v else 0.0
            print(f"MeshletSplitFloatWaveletAMD (crack-free) "
                  f"mv={self.max_verts}  wavelet={self.wavelet}  "
                  f"schedule={self.schedule}  ratio={self.ratio}  "
                  f"conn={self.conn}  sort={self.sort}  "
                  f"pack_meta={self.pack_meta}:")
            print(f"  {len(meshlets)} meshlets, {n_boundary} boundary verts "
                  f"({bnd_frac:.1f}% of total)")
            print(f"  Global bits=[{g_bits[0]},{g_bits[1]},{g_bits[2]}], "
                  f"ref_bits={ref_bits}")
            print(f"  Cracks: {n_cracks} / {n_shared} shared checks")
            print(f"  Boundary table: {bnd_table_bits/8:,.0f}B  "
                  f"Boundary refs: {total_bnd_refs/8:,.0f}B")
            print(f"  Interior float {self.wavelet}: {total_interior/8:,.0f}B")
            print(f"  Meshlet hdrs: {total_ml_hdr/8:,.0f}B  "
                  f"Conn: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  "
                  f"BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Boundary err: max={bnd_err.max():.6f}  "
                  f"Interior err: max={int_err.max():.6f}")
            print(f"  Combined %OK (<= {self.precision_error}): {pct:.1f}%")

        return CompressedModel(bytes(int(np.ceil(total_bits / 8))), bpv, bpt)


class MeshletSplitFloatHaarAMD(MeshletSplitFloatWaveletAMD):
    """Split + float Haar interior (geometric per-level quantization)."""

    def __init__(self, max_verts=256, precision_error=0.0005,
                 schedule="geometric", ratio=2.0,
                 conn="gts_v3", sort="greedy_nn", pack_meta=True,
                 target_base=32, dct_block_size=4,
                 dct_schedule="uniform", dct_ratio=2.0, verbose=False):
        super().__init__(max_verts=max_verts, precision_error=precision_error,
                         wavelet="haar", schedule=schedule, ratio=ratio,
                         conn=conn, sort=sort, pack_meta=pack_meta,
                         target_base=target_base,
                         dct_block_size=dct_block_size,
                         dct_schedule=dct_schedule, dct_ratio=dct_ratio,
                         verbose=verbose)


class MeshletSplitDiffQuantAMD(Encoder):
    """End-to-end differentiable interior encoder.

    Two-pass per mesh:
      1. Collect interior streams (one per meshlet × 3 axes).
      2. Train a NeuralCompressor (MLP predictors + per-level learnable
         μ-law curve α + learnable δ) with RD loss:
             loss = λ_rate · Σ var(q_level)  +  λ_max · ReLU(‖x − x̂‖_∞ − ε)²
         (variance proxy as the rate term, per user preference.)
      3. Export MLP weights + α/δ; encode each meshlet with the learned
         parameters via the numpy-side inference path (bit-exact at
         inference since rounding is deterministic).
    """

    def __init__(self, max_verts=256, precision_error=0.0005,
                 kernel_size=4, hidden=8,
                 epochs=300, lr=5e-3,
                 lambda_rate=1.0, lambda_max=1e5, seed=0,
                 predictor_type="mlp",
                 conn="gts_v3", sort="greedy_nn",
                 target_base=32, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.kernel_size = kernel_size
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.lambda_rate = lambda_rate
        self.lambda_max = lambda_max
        self.seed = seed
        self.predictor_type = predictor_type   # 'mlp' or 'dense'
        self.conn = conn
        self.sort = sort
        self.target_base = target_base
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        from utils.boundary_split import (
            identify_boundary_verts, build_boundary_table,
            split_meshlet_verts, boundary_table_bits, ref_bits_for,
            verify_crack_free, sort_by_morton, gts_encode_meshlet,
        )
        from utils.neural_compressor import (
            train_compressor, quantize_interior_diff,
        )

        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)
        center = verts_np.mean(axis=0); vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1)); vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)
        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np); fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        boundary_set = identify_boundary_verts(meshlets, tris_np)
        boundary_list, _, _ = build_boundary_table(boundary_set, global_codes)
        n_boundary = len(boundary_list)
        ref_bits = ref_bits_for(n_boundary)

        # -- Pass 1: collect meshlet data + training streams
        meshlet_data = []
        training_streams = []
        for ml_tris in meshlets:
            vert_order, _, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
            if len(vert_order) < 1:
                continue
            ltg, bnd_local, int_local, _ = split_meshlet_verts(
                vert_order, boundary_set)
            if self.sort != "eb":
                from utils.interior_sorts import sort_interior as _si
                bnd_local = sort_by_morton(bnd_local, global_codes)
                int_local = _si(self.sort, int_local,
                                global_codes=global_codes,
                                vert_pos_float=vn)
                ltg = bnd_local + int_local
            meshlet_data.append((ml_tris, ltg, bnd_local, int_local))
            if len(int_local) > 0:
                pts = vn[int_local]
                for d in range(3):
                    off = float(pts[:, d].min())
                    training_streams.append(pts[:, d] - off)

        # -- Pass 2: train neural compressor
        n_levels_max = max(1, int(np.ceil(np.log2(
            self.max_verts / self.target_base))))
        if self.verbose:
            print(f"MeshletSplitDiffQuantAMD: training on "
                  f"{len(training_streams):,} streams (eps={per_coord_err:.4e})")
        _model, params = train_compressor(
            training_streams,
            kernel_size=self.kernel_size, hidden=self.hidden,
            n_levels_max=n_levels_max, target_base=self.target_base,
            eps=per_coord_err, epochs=self.epochs, lr=self.lr,
            lambda_rate=self.lambda_rate, lambda_max=self.lambda_max,
            predictor_type=self.predictor_type,
            max_signal_len=self.max_verts,
            seed=self.seed, verbose=self.verbose,
        )

        # -- Header accounting (MLP or Dense)
        mlp_bytes = 0
        for pd in params["predictors"]:
            if pd["type"] == "dense":
                eW, eb = pd["enc"]
                dW, db = pd["dec"]
                mlp_bytes += (eW.size + eb.size + dW.size + db.size) * 4
            else:
                for W, b in pd["layers"]:
                    mlp_bytes += W.size * 4 + b.size * 4
        alpha_delta_bytes = (params["n_levels"] + 1) * 2 * 4
        global_header_bits = 36 * 8
        param_header_bits = (mlp_bytes + alpha_delta_bytes) * 8
        bnd_table_bits = boundary_table_bits(n_boundary, g_bits)

        total_bits = global_header_bits + param_header_bits + bnd_table_bits
        total_bnd_refs = 0
        total_interior = 0
        total_conn = 0
        total_ml_hdr = 0
        interior_errors = []
        boundary_errors = []
        bnd_recon_global = _dequantize_global(
            global_codes, g_min, g_range, g_bits)

        # -- Pass 3: encode each meshlet with learned params
        for ml_tris, local_to_global, bnd_local, int_local in meshlet_data:
            n_bnd = len(bnd_local)
            n_int = len(int_local)

            ml_hdr_bits = 4 * 8
            total_ml_hdr += ml_hdr_bits
            ref_total = n_bnd * ref_bits
            total_bnd_refs += ref_total

            int_bits = 0
            if n_int > 0:
                int_pts = vn[int_local]
                int_recon, int_bits, _meta = quantize_interior_diff(
                    int_pts, params, verbose=False)
                total_interior += int_bits
                err = np.linalg.norm(int_recon - int_pts, axis=1) * scale
                interior_errors.extend(err.tolist())

            if n_bnd > 0:
                for gv in bnd_local:
                    err = np.linalg.norm(bnd_recon_global[gv] - vn[gv]) * scale
                    boundary_errors.append(err)

            if self.conn in ("gts", "gts_v2", "gts_v3"):
                variant = {"gts": "v1", "gts_v2": "v2", "gts_v3": "v3"}[self.conn]
                conn_bits, _ = gts_encode_meshlet(
                    ml_tris, tris_np, tri_adj, local_to_global, variant=variant)
            else:
                conn_bits = _amd_packed_bits(ml_tris, tris_np, tri_adj)
            total_conn += conn_bits

            total_bits += ml_hdr_bits + ref_total + int_bits + conn_bits

        n_cracks, n_shared = verify_crack_free(
            meshlets, tris_np, global_codes, boundary_set)

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            import numpy as _np
            bnd_err = _np.array(boundary_errors) if boundary_errors else _np.zeros(1)
            int_err = _np.array(interior_errors) if interior_errors else _np.zeros(1)
            combined = _np.concatenate([bnd_err, int_err])
            pct = (combined <= self.precision_error).sum() / len(combined) * 100
            bnd_frac = n_boundary / n_v * 100 if n_v else 0.0
            print(f"MeshletSplitDiffQuantAMD (crack-free) "
                  f"mv={self.max_verts}  K={self.kernel_size}  H={self.hidden}  "
                  f"conn={self.conn}  sort={self.sort}:")
            print(f"  {len(meshlets)} meshlets, {n_boundary} boundary verts "
                  f"({bnd_frac:.1f}% of total)")
            print(f"  Learned α: {params['alphas']}")
            print(f"  Learned δ: {params['deltas']}")
            print(f"  Cracks: {n_cracks} / {n_shared} shared checks")
            print(f"  Boundary table: {bnd_table_bits/8:,.0f}B  "
                  f"Boundary refs: {total_bnd_refs/8:,.0f}B")
            print(f"  MLP+curve params: {mlp_bytes+alpha_delta_bytes:,} B")
            print(f"  Interior diff-quant: {total_interior/8:,.0f}B  "
                  f"Conn: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  "
                  f"BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Boundary err max={bnd_err.max():.6f}  "
                  f"Interior err max={int_err.max():.6f}")
            print(f"  Combined %OK (<= {self.precision_error}): {pct:.1f}%")

        return CompressedModel(bytes(int(np.ceil(total_bits / 8))), bpv, bpt)


class MeshletSplitMLPAMD(Encoder):
    """Boundary/interior split + non-linear MLP lifting predictor.

    - One tiny MLP per wavelet level, shared across all meshlets and axes.
    - MLPs are trained once per mesh by Adam on the predict-the-odd task.
    - Crack-free by construction (boundary on global int grid).
    - Max-error guarantee: inverse-lifting amplification bounded by
      Lip(MLP) * sqrt(K); passed to the δ allocator so
        δ_base + amp * Σ δ_k ≤ 2 * per_coord_err.
    - Quantization curve learned: the per-level δ ratio is chosen by a
      short grid search over candidates to minimise total interior bits
      under the error constraint.
    """

    def __init__(self, max_verts=256, precision_error=0.0005,
                 kernel_size=4, hidden=16,
                 ratio_candidates=(2.0, 4.0, 6.0, 8.0, 12.0, 16.0),
                 epochs=300, lr=1e-3, weight_decay=1e-4, seed=0,
                 conn="gts_v3", sort="greedy_nn",
                 target_base=32, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.kernel_size = kernel_size
        self.hidden = hidden
        self.ratio_candidates = tuple(ratio_candidates)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.conn = conn
        self.sort = sort
        self.target_base = target_base
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        from utils.boundary_split import (
            identify_boundary_verts, build_boundary_table,
            split_meshlet_verts, boundary_table_bits, ref_bits_for,
            verify_crack_free, sort_by_morton, gts_encode_meshlet,
        )
        from utils.learned_mlp_wavelet import (
            fit_mlps, mlp_lipschitz_bound, quantize_interior_mlp_wavelet,
        )

        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        boundary_set = identify_boundary_verts(meshlets, tris_np)
        boundary_list, _gv_to_ref, _bnd_codes = build_boundary_table(
            boundary_set, global_codes)
        n_boundary = len(boundary_list)
        ref_bits = ref_bits_for(n_boundary)

        # -- Pass 1: gather per-meshlet data + training streams
        meshlet_data = []
        training_streams = []
        for ml_tris in meshlets:
            vert_order, _, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
            if len(vert_order) < 1:
                continue
            ltg, bnd_local, int_local, _ = split_meshlet_verts(
                vert_order, boundary_set)
            if self.sort != "eb":
                from utils.interior_sorts import sort_interior as _si
                bnd_local = sort_by_morton(bnd_local, global_codes)
                int_local = _si(self.sort, int_local,
                                global_codes=global_codes,
                                vert_pos_float=vn)
                ltg = bnd_local + int_local
            meshlet_data.append((ml_tris, ltg, bnd_local, int_local))
            if len(int_local) > 0:
                pts = vn[int_local]
                for d in range(3):
                    off = float(pts[:, d].min())
                    training_streams.append(pts[:, d] - off)

        # -- Pass 2: train MLPs
        n_levels_max = max(1, int(np.ceil(np.log2(
            self.max_verts / self.target_base))))
        if self.verbose:
            print(f"MeshletSplitMLPAMD: training {n_levels_max} MLPs on "
                  f"{len(training_streams):,} streams...")
        mlp_weights, _modules = fit_mlps(
            training_streams, n_levels_max,
            kernel_size=self.kernel_size, hidden=self.hidden,
            target_base=self.target_base,
            epochs=self.epochs, lr=self.lr,
            weight_decay=self.weight_decay, seed=self.seed,
            verbose=self.verbose,
        )
        # Error-safe amp = max over levels of min(L2*sqrt(K), Linf) bound.
        lip_per_level = [mlp_lipschitz_bound(w, self.kernel_size)
                         for w in mlp_weights]
        amp = max(1.0, max(lip_per_level))

        # -- Pass 3: ratio sweep (learned quantization curve)
        # Score each candidate ratio by summed interior bits
        meshlets_with_int = [(i, md) for i, md in enumerate(meshlet_data)
                             if len(md[3]) > 0]
        best_ratio = self.ratio_candidates[0]
        best_bits = None
        sweep_results = {}
        for r in self.ratio_candidates:
            total = 0
            for _, (_, _, _, int_local) in meshlets_with_int:
                pts = vn[int_local]
                _, bits, _ = quantize_interior_mlp_wavelet(
                    pts, per_coord_err, mlp_weights, self.kernel_size,
                    amp=amp, schedule="geometric", ratio=r,
                    target_base=self.target_base,
                )
                total += bits
            sweep_results[r] = total
            if best_bits is None or total < best_bits:
                best_bits = total
                best_ratio = r

        # -- Header accounting
        # Standard global header (36 B) + MLP weights + learned ratio (4 B)
        mlp_bytes = 0
        for w_list in mlp_weights:
            for W, b in w_list:
                mlp_bytes += W.size * 4 + b.size * 4  # float32
        global_header_bits = 36 * 8
        ratio_bits = 32  # float32 learned ratio
        mlp_header_bits = mlp_bytes * 8
        bnd_table_bits = boundary_table_bits(n_boundary, g_bits)

        total_bits = (global_header_bits + ratio_bits
                      + mlp_header_bits + bnd_table_bits)
        total_bnd_refs = 0
        total_interior = 0
        total_conn = 0
        total_ml_hdr = 0
        interior_errors = []
        boundary_errors = []
        bnd_recon_global = _dequantize_global(
            global_codes, g_min, g_range, g_bits)

        # -- Pass 4: encode each meshlet with chosen ratio
        for ml_tris, local_to_global, bnd_local, int_local in meshlet_data:
            n_bnd = len(bnd_local)
            n_int = len(int_local)

            ml_hdr_bits = 4 * 8
            total_ml_hdr += ml_hdr_bits

            ref_total = n_bnd * ref_bits
            total_bnd_refs += ref_total

            int_bits = 0
            if n_int > 0:
                int_pts = vn[int_local]
                int_recon, int_bits, _meta = quantize_interior_mlp_wavelet(
                    int_pts, per_coord_err, mlp_weights, self.kernel_size,
                    amp=amp, schedule="geometric", ratio=best_ratio,
                    target_base=self.target_base,
                )
                total_interior += int_bits
                err = np.linalg.norm(int_recon - int_pts, axis=1) * scale
                interior_errors.extend(err.tolist())

            if n_bnd > 0:
                for gv in bnd_local:
                    err = np.linalg.norm(bnd_recon_global[gv] - vn[gv]) * scale
                    boundary_errors.append(err)

            if self.conn in ("gts", "gts_v2", "gts_v3"):
                variant = {"gts": "v1", "gts_v2": "v2", "gts_v3": "v3"}[self.conn]
                conn_bits, _ = gts_encode_meshlet(
                    ml_tris, tris_np, tri_adj, local_to_global, variant=variant)
            else:
                conn_bits = _amd_packed_bits(ml_tris, tris_np, tri_adj)
            total_conn += conn_bits

            total_bits += ml_hdr_bits + ref_total + int_bits + conn_bits

        n_cracks, n_shared = verify_crack_free(
            meshlets, tris_np, global_codes, boundary_set)

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            import numpy as _np
            bnd_err = _np.array(boundary_errors) if boundary_errors else _np.zeros(1)
            int_err = _np.array(interior_errors) if interior_errors else _np.zeros(1)
            combined = _np.concatenate([bnd_err, int_err])
            pct = (combined <= self.precision_error).sum() / len(combined) * 100
            bnd_frac = n_boundary / n_v * 100 if n_v else 0.0
            print(f"MeshletSplitMLPAMD (crack-free) "
                  f"mv={self.max_verts}  K={self.kernel_size}  "
                  f"H={self.hidden}  conn={self.conn}  sort={self.sort}:")
            print(f"  {len(meshlets)} meshlets, {n_boundary} boundary verts "
                  f"({bnd_frac:.1f}% of total)")
            lip_str = ", ".join(f"{l:.2f}" for l in lip_per_level)
            print(f"  MLP L_inf Lipschitz per level: [{lip_str}]  "
                  f"amp={amp:.3f}")
            print(f"  Ratio sweep (interior bytes):")
            for r in self.ratio_candidates:
                marker = " <--" if r == best_ratio else ""
                print(f"    ratio={r:>5.1f}: {sweep_results[r]/8:9,.0f} B{marker}")
            print(f"  Cracks: {n_cracks} / {n_shared} shared checks")
            print(f"  Boundary table: {bnd_table_bits/8:,.0f}B  "
                  f"Boundary refs: {total_bnd_refs/8:,.0f}B")
            print(f"  MLP storage: {mlp_bytes:,} B  "
                  f"(K={self.kernel_size}, H={self.hidden}, "
                  f"L={n_levels_max})")
            print(f"  Interior MLP: {total_interior/8:,.0f}B  "
                  f"Conn: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  "
                  f"BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Boundary err: max={bnd_err.max():.6f}  "
                  f"Interior err: max={int_err.max():.6f}")
            print(f"  Combined %OK (<= {self.precision_error}): {pct:.1f}%")

        return CompressedModel(bytes(int(np.ceil(total_bits / 8))), bpv, bpt)


class MeshletSplitLearnedAMD(Encoder):
    """Boundary/interior split + learned-lifting interior wavelet.

    Two-pass encoder:
      1. Collect all interior (per-axis) streams across meshlets.
      2. Fit per-level linear predictor kernels by ridge-regularised lstsq.
      3. Encode using the fitted kernels; store them once in the global header.

    Kernels are shared across all meshlets and all axes (one kernel per
    decomposition level). Storage overhead is negligible — e.g. 3 levels of
    K=4 weights at float32 = 48 B.
    """

    def __init__(self, max_verts=256, precision_error=0.0005,
                 schedule="geometric", ratio=2.0, kernel_size=4,
                 conn="gts_v3", sort="greedy_nn", pack_meta=True,
                 target_base=32, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.schedule = schedule
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.conn = conn
        self.sort = sort
        self.pack_meta = pack_meta  # kept for API symmetry; layout is always packed
        self.target_base = target_base
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        from utils.boundary_split import (
            identify_boundary_verts, build_boundary_table,
            split_meshlet_verts, boundary_table_bits, ref_bits_for,
            verify_crack_free, sort_by_morton, gts_encode_meshlet,
        )
        from utils.learned_wavelet import (
            fit_kernels, quantize_interior_learned_wavelet,
        )

        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        boundary_set = identify_boundary_verts(meshlets, tris_np)
        boundary_list, _gv_to_ref, _bnd_codes = build_boundary_table(
            boundary_set, global_codes)
        n_boundary = len(boundary_list)
        ref_bits = ref_bits_for(n_boundary)

        # -- Pass 1: collect meshlet metadata + interior streams for training
        meshlet_data = []  # per meshlet: (ml_tris, local_to_global, int_local, n_bnd)
        training_streams = []
        for ml_tris in meshlets:
            vert_order, _, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
            if len(vert_order) < 1:
                continue
            ltg, bnd_local, int_local, _ = split_meshlet_verts(
                vert_order, boundary_set)
            if self.sort != "eb":
                from utils.interior_sorts import sort_interior as _si
                bnd_local = sort_by_morton(bnd_local, global_codes)
                int_local = _si(self.sort, int_local,
                                global_codes=global_codes,
                                vert_pos_float=vn)
                ltg = bnd_local + int_local
            meshlet_data.append((ml_tris, ltg, bnd_local, int_local))
            if len(int_local) > 0:
                pts = vn[int_local]
                for d in range(3):
                    off = float(pts[:, d].min())
                    training_streams.append(pts[:, d] - off)

        # -- Pass 2: fit kernels
        n_levels_max = max(1, int(np.ceil(np.log2(self.max_verts / self.target_base))))
        kernels = fit_kernels(training_streams, n_levels_max,
                              self.kernel_size, self.target_base)
        # Conservative amp = max kernel L1 norm (floor at 1.0)
        amp = max(1.0, *(float(np.abs(k).sum()) for k in kernels))

        # -- Global header: standard 36 B + kernels
        global_header_bits = 36 * 8
        kernel_storage_bits = n_levels_max * self.kernel_size * 32  # float32 weights
        bnd_table_bits = boundary_table_bits(n_boundary, g_bits)

        total_bits = global_header_bits + kernel_storage_bits + bnd_table_bits
        total_bnd_refs = 0
        total_interior = 0
        total_conn = 0
        total_ml_hdr = 0
        interior_errors = []
        boundary_errors = []
        bnd_recon_global = _dequantize_global(global_codes, g_min, g_range, g_bits)

        # -- Pass 3: encode each meshlet using fitted kernels
        for ml_tris, local_to_global, bnd_local, int_local in meshlet_data:
            n_bnd = len(bnd_local)
            n_int = len(int_local)

            ml_hdr_bits = 4 * 8
            total_ml_hdr += ml_hdr_bits

            ref_total = n_bnd * ref_bits
            total_bnd_refs += ref_total

            int_bits = 0
            if n_int > 0:
                int_pts = vn[int_local]
                int_recon, int_bits, _meta = quantize_interior_learned_wavelet(
                    int_pts, per_coord_err, kernels,
                    schedule=self.schedule, ratio=self.ratio,
                    target_base=self.target_base, amp=amp,
                )
                total_interior += int_bits
                err = np.linalg.norm(int_recon - int_pts, axis=1) * scale
                interior_errors.extend(err.tolist())

            if n_bnd > 0:
                for gv in bnd_local:
                    err = np.linalg.norm(bnd_recon_global[gv] - vn[gv]) * scale
                    boundary_errors.append(err)

            if self.conn in ("gts", "gts_v2", "gts_v3"):
                variant = {"gts": "v1", "gts_v2": "v2", "gts_v3": "v3"}[self.conn]
                conn_bits, _ = gts_encode_meshlet(
                    ml_tris, tris_np, tri_adj, local_to_global, variant=variant)
            else:
                conn_bits = _amd_packed_bits(ml_tris, tris_np, tri_adj)
            total_conn += conn_bits

            total_bits += ml_hdr_bits + ref_total + int_bits + conn_bits

        n_cracks, n_shared = verify_crack_free(
            meshlets, tris_np, global_codes, boundary_set)

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            import numpy as _np
            bnd_err = _np.array(boundary_errors) if boundary_errors else _np.zeros(1)
            int_err = _np.array(interior_errors) if interior_errors else _np.zeros(1)
            combined = _np.concatenate([bnd_err, int_err])
            pct = (combined <= self.precision_error).sum() / len(combined) * 100
            bnd_frac = n_boundary / n_v * 100 if n_v else 0.0
            print(f"MeshletSplitLearnedAMD (crack-free) "
                  f"mv={self.max_verts}  K={self.kernel_size}  "
                  f"schedule={self.schedule}  ratio={self.ratio}  "
                  f"conn={self.conn}  sort={self.sort}:")
            print(f"  {len(meshlets)} meshlets, {n_boundary} boundary verts "
                  f"({bnd_frac:.1f}% of total)")
            print(f"  Fitted kernels (L1 norm, amp={amp:.3f}):")
            for i, k in enumerate(kernels):
                print(f"    level {i}: {np.array2string(k, precision=3)} "
                      f"L1={np.abs(k).sum():.3f}")
            print(f"  Cracks: {n_cracks} / {n_shared} shared checks")
            print(f"  Boundary table: {bnd_table_bits/8:,.0f}B  "
                  f"Boundary refs: {total_bnd_refs/8:,.0f}B  "
                  f"Kernel storage: {kernel_storage_bits/8:,.0f}B")
            print(f"  Interior learned: {total_interior/8:,.0f}B")
            print(f"  Meshlet hdrs: {total_ml_hdr/8:,.0f}B  "
                  f"Conn: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  "
                  f"BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Boundary err: max={bnd_err.max():.6f}  "
                  f"Interior err: max={int_err.max():.6f}")
            print(f"  Combined %OK (<= {self.precision_error}): {pct:.1f}%")

        return CompressedModel(bytes(int(np.ceil(total_bits / 8))), bpv, bpt)


class MeshletSplitFloatCDF53AMD(MeshletSplitFloatWaveletAMD):
    """Split + float CDF 5/3 interior (geometric per-level quantization)."""

    def __init__(self, max_verts=256, precision_error=0.0005,
                 schedule="geometric", ratio=2.0,
                 conn="gts_v3", sort="greedy_nn", pack_meta=True,
                 target_base=32, dct_block_size=4,
                 dct_schedule="uniform", dct_ratio=2.0, verbose=False):
        super().__init__(max_verts=max_verts, precision_error=precision_error,
                         wavelet="cdf53", schedule=schedule, ratio=ratio,
                         conn=conn, sort=sort, pack_meta=pack_meta,
                         target_base=target_base,
                         dct_block_size=dct_block_size,
                         dct_schedule=dct_schedule, dct_ratio=dct_ratio,
                         verbose=verbose)


# ============================================================
# Solution 2: Deduplicated boundary vertices
# ============================================================

def _identify_boundary_verts(meshlets, tris_np):
    vert_count = {}
    for ml_tris in meshlets:
        vs = set()
        for ti in ml_tris:
            for j in range(3): vs.add(int(tris_np[ti, j]))
        for v in vs:
            vert_count[v] = vert_count.get(v, 0) + 1
    return set(v for v, c in vert_count.items() if c >= 2)


class MeshletWaveletDedupEB(Encoder):
    """Solution 2: Boundary verts globally quantized (crack-free),
    interior verts per-meshlet float wavelet + EdgeBreaker."""

    def __init__(self, max_verts=256, precision_error=0.0005, verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        boundary = _identify_boundary_verts(meshlets, tris_np)
        boundary_list = sorted(boundary)
        n_boundary = len(boundary_list)
        ref_bits = max(1, int(np.ceil(np.log2(n_boundary + 1)))) if n_boundary > 0 else 1

        # Global boundary stream
        bnd_stream_bits = 0
        if n_boundary > 0:
            bnd_pts = vn[boundary_list]
            for d in range(3):
                vals = bnd_pts[:, d]
                rng = vals.max() - vals.min() if n_boundary > 1 else 0.001
                bits = _bits_for_error(rng, per_coord_err)
                codes = _quantize(vals, vals.min(), vals.max(), bits)
                bnd_stream_bits += _stream_bits(codes, bits)
            bnd_stream_bits += (6*4 + 3 + 4) * 8

        # Global header
        total_bits = (3*4 + 4 + 4 + 4) * 8 + bnd_stream_bits
        total_interior = 0; total_refs = 0; total_conn = 0

        for ml_tris in meshlets:
            vert_order, opcodes, n_roots = edgebreaker_vertex_order(
                ml_tris, tris_np, tri_adj)
            n_mv = len(vert_order)
            if n_mv < 1: continue

            interior = [v for v in vert_order if v not in boundary]
            n_bnd_local = n_mv - len(interior)

            # Interior wavelet
            int_bits = 0
            if len(interior) > 0:
                int_pts = vn[interior]
                for d in range(3):
                    w = estimate_wavelet_bits(int_pts[:, d], per_coord_err, 32)
                    int_bits += w["total_bits"]
            total_interior += int_bits

            # Boundary refs + flags + header
            ml_refs = n_bnd_local * ref_bits + n_mv + 64
            total_refs += ml_refs

            conn = _edgebreaker_bits(opcodes, n_roots, n_mv)
            total_conn += conn
            total_bits += int_bits + ml_refs + conn

        bpv = total_bits / n_v; bpt = total_bits / n_t
        errors_bnd = np.zeros(n_v)  # approximate: boundary has quant error, interior ~0

        if self.verbose:
            print(f"MeshletWaveletDedupEB (crack-free) mv={self.max_verts}:")
            print(f"  {len(meshlets)} meshlets, {n_boundary} boundary ({n_boundary/n_v*100:.1f}%)")
            print(f"  Boundary stream: {bnd_stream_bits/8:,.0f}B")
            print(f"  Interior wavelet: {total_interior/8:,.0f}B")
            print(f"  Refs+flags: {total_refs/8:,.0f}B  Conn: {total_conn/8:,.0f}B")
            print(f"  Total: {total_bits/8:,.0f}B  BPV={bpv:.2f}  BPT={bpt:.2f}")
        return CompressedModel(bytes(int(np.ceil(total_bits / 8))), bpv, bpt)


# ============================================================
# RECOMMENDED ENCODERS: GTS connectivity + various vertex encodings
# These match the AMD GPUOpen article architecture:
#   - GTS connectivity (parallel GPU decode via countbits/firstbithigh)
#   - Global quantization grid (crack-free)
#   - Vertex encoding options: plain / segmented delta / Haar wavelet
# ============================================================

class _MeshletGTSBase(Encoder):
    """Base class for GTS-connectivity meshlet encoders."""

    def __init__(self, max_verts=256, precision_error=0.0005,
                 vertex_mode='seg_delta', verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.vertex_mode = vertex_mode  # 'plain', 'seg_delta', 'haar'
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        from utils.wavelet import estimate_wavelet_bits_int

        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)

        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        # Global quantize (crack-free)
        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        # Global header
        total_bits = (3 * 4 + 3 * 4 + 3 + 4) * 8
        total_vtx = 0
        total_conn = 0
        total_hdr = total_bits
        vertex_recon = {}
        n_cracks = 0

        for ml_tris in meshlets:
            vert_order, _, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
            n_mv = len(vert_order)
            if n_mv < 1:
                continue

            int_pts = global_codes[vert_order]

            # Per-meshlet header: 37 bytes
            ml_hdr = 37 * 8
            total_hdr += ml_hdr

            # Vertex encoding
            vtx_bits = 0
            if self.vertex_mode == 'seg_delta':
                for d in range(3):
                    vtx_bits += estimate_wavelet_bits_int(
                        int_pts[:, d], 32, 'seg_delta')['total_bits']
            elif self.vertex_mode == 'haar':
                for d in range(3):
                    vtx_bits += estimate_wavelet_bits_int(
                        int_pts[:, d], 32, 'haar')['total_bits']
            else:  # plain
                for d in range(3):
                    vals = int_pts[:, d]
                    rng = int(vals.max() - vals.min()) if n_mv > 1 else 0
                    bits = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
                    vtx_bits += n_mv * bits + 5 * 8
            total_vtx += vtx_bits

            # GTS connectivity
            conn_bits = _amd_gts_bits(ml_tris, tris_np, tri_adj)
            total_conn += conn_bits

            total_bits += ml_hdr + vtx_bits + conn_bits

            # Crack check
            dq = _dequantize_global(int_pts, g_min, g_range, g_bits)
            for i, gv in enumerate(vert_order):
                pos = tuple(dq[i])
                if gv in vertex_recon:
                    if vertex_recon[gv] != pos:
                        n_cracks += 1
                else:
                    vertex_recon[gv] = pos

        # Accuracy
        all_recon = _dequantize_global(global_codes, g_min, g_range, g_bits)
        errors = np.linalg.norm(all_recon - vn, axis=1) * scale

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            pct = (errors <= self.precision_error).sum() / n_v * 100
            vtx_name = {'plain': 'Plain', 'seg_delta': 'SegDelta',
                        'haar': 'Haar'}[self.vertex_mode]
            print(f"MeshletGTS+{vtx_name} (crack-free) mv={self.max_verts}:")
            print(f"  {len(meshlets)} meshlets, global bits={g_bits}, cracks={n_cracks}")
            print(f"  Headers:      {total_hdr/8:>10,.0f} B ({total_hdr/total_bits*100:.1f}%)")
            print(f"  Vertex data:  {total_vtx/8:>10,.0f} B ({total_vtx/total_bits*100:.1f}%)")
            print(f"  Connectivity: {total_conn/8:>10,.0f} B ({total_conn/total_bits*100:.1f}%)")
            print(f"  Total: {total_bits/8:,.0f}B  BPV={bpv:.2f}  BPT={bpt:.2f}  "
                  f"Ratio={n_v*96+n_t*96:.0f}/{total_bits/8:.0f}="
                  f"{(n_v*96+n_t*96)/(total_bits/8):.1f}x")
            print(f"  Accuracy: max={errors.max():.6f}  %OK={pct:.1f}%")

        data = bytes(int(np.ceil(total_bits / 8)))
        return CompressedModel(data, bpv, bpt)


class MeshletGTSPlain(_MeshletGTSBase):
    """AMD original: GTS connectivity + plain per-meshlet quantized vertices.
    Crack-free. GPU-parallel decode. Baseline for comparison."""
    def __init__(self, max_verts=256, precision_error=0.0005, verbose=False):
        super().__init__(max_verts, precision_error, 'plain', verbose)


class MeshletGTSSegDelta(_MeshletGTSBase):
    """Our best: GTS connectivity + segmented delta vertex encoding.
    Crack-free. GPU-parallel decode. ~24% smaller than AMD baseline."""
    def __init__(self, max_verts=256, precision_error=0.0005, verbose=False):
        super().__init__(max_verts, precision_error, 'seg_delta', verbose)


class MeshletGTSHaar(_MeshletGTSBase):
    """GTS connectivity + Haar integer wavelet vertex encoding.
    Crack-free. GPU-parallel decode (log N steps)."""
    def __init__(self, max_verts=256, precision_error=0.0005, verbose=False):
        super().__init__(max_verts, precision_error, 'haar', verbose)


# ============================================================
# LOD Encoder: progressive wavelet LOD with QEM importance ordering
# ============================================================

def _build_lod_triangle_masks(meshlet_tris, tris_np, vert_order,
                               lod_vert_counts):
    """Build boolean masks of active triangles per LOD level.

    A triangle is active at LOD level L if all 3 of its vertices
    have local index < lod_vert_counts[L] in the importance-ordered
    vertex list.

    Returns list of np.ndarray bool masks, one per LOD level.
    """
    # Map global vid → local index in the importance order
    g2l = {gv: li for li, gv in enumerate(vert_order)}

    n_tris = len(meshlet_tris)
    masks = []
    for nv_cutoff in lod_vert_counts:
        mask = np.zeros(n_tris, dtype=bool)
        for ti_local, ti_global in enumerate(meshlet_tris):
            a, b, c = int(tris_np[ti_global, 0]), int(tris_np[ti_global, 1]), int(tris_np[ti_global, 2])
            if g2l.get(a, 999) < nv_cutoff and g2l.get(b, 999) < nv_cutoff and g2l.get(c, 999) < nv_cutoff:
                mask[ti_local] = True
        masks.append(mask)
    return masks


def _lod_mask_bits(masks):
    """Estimate bits for LOD triangle masks using delta-encoded bitmasks.

    LOD masks are hierarchical: mask[i] is a superset of mask[i-1].
    Encode mask[0] + delta(mask[1]-mask[0]) + delta(mask[2]-mask[1]) + ...
    Use entropy coding on each delta bitmask.
    """
    total_bits = 0
    prev = np.zeros(len(masks[0]), dtype=bool)
    for mask in masks:
        delta = mask & ~prev  # new triangles at this level
        n_ones = int(delta.sum())
        n = len(delta)
        if n == 0:
            continue
        # Entropy of binary stream
        if n_ones == 0 or n_ones == n:
            bits = n + 8  # 1 bpt + small header
        else:
            p = n_ones / n
            ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            bits = min(n, n * ent + 16)  # entropy vs plain
        total_bits += bits + 8  # 8 bits for delta header
        prev = mask
    return total_bits


def _identify_meshlet_boundary_verts(meshlets, tris_np):
    """Find vertices shared between ≥2 meshlets (meshlet boundary)."""
    vert_ml = {}
    for ml_idx, ml_tris in enumerate(meshlets):
        vs = set()
        for ti in ml_tris:
            for j in range(3):
                vs.add(int(tris_np[ti, j]))
        for v in vs:
            if v in vert_ml:
                vert_ml[v].add(ml_idx)
            else:
                vert_ml[v] = {ml_idx}
    return set(v for v, ms in vert_ml.items() if len(ms) >= 2)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _build_lod_masks_from_positions(meshlet_tris, tris_np, g2l, g2pos,
                                     lod_vertex_sets):
    """Build hierarchical LOD triangle masks based on VERTEX POSITIONS.

    lod_vertex_sets: list of sets; lod_vertex_sets[k] = positions present at LOD k.
    A triangle is active at LOD k iff all 3 of its vertex positions are in set k.
    """
    n_tris = len(meshlet_tris)
    masks = []
    for pos_set in lod_vertex_sets:
        mask = np.zeros(n_tris, dtype=bool)
        for ti_local, ti_global in enumerate(meshlet_tris):
            a = int(tris_np[ti_global, 0])
            b = int(tris_np[ti_global, 1])
            c = int(tris_np[ti_global, 2])
            if (g2pos.get(a, -1) in pos_set and
                g2pos.get(b, -1) in pos_set and
                g2pos.get(c, -1) in pos_set):
                mask[ti_local] = True
        masks.append(mask)
    return masks


class MeshletWaveletLOD(Encoder):
    """Crack-free progressive LOD encoder.

    Pipeline:
      1. Generate meshlets on original mesh
      2. Identify meshlet boundary vertices (shared between ≥2 meshlets)
      3. Run QEM with boundary verts PROTECTED (never collapsed)
      4. Per-meshlet:
         - target_base = max(32, next_pow2(n_boundary_in_meshlet))
         - Place boundary verts in LOD 0 position slots
         - Place interior verts (by QEM importance) in LOD 1+ slots
         - Haar wavelet with per-meshlet target_base
         - Triangle LOD masks based on decoded positions per level

    Crack-free guarantee:
      - Global integer quantization → shared verts have identical positions
      - Boundary verts NEVER collapsed by QEM → positions stable across meshlets
      - All boundary verts at LOD 0 positions → available at every LOD level
        in every meshlet that contains them
      - Topological crack-free: boundary verts decodable from LOD 0 upward
    """

    def __init__(self, max_verts=256, precision_error=0.0005,
                 connectivity='gts', verbose=False):
        self.max_verts = max_verts
        self.precision_error = precision_error
        self.connectivity = connectivity  # 'gts', 'edgebreaker', 'amd_packed'
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        from utils.wavelet import (
            estimate_wavelet_bits_int, lod_position_order,
        )
        from utils.qem import progressive_simplify

        verts_np, tris_np = _to_numpy(model)
        n_v, n_t = len(verts_np), len(tris_np)

        # --- Step 1: Global quantization (crack-free grid) ---
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)
        global_codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        # --- Step 2: Generate meshlets on original mesh ---
        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        from utils.meshlet_generator import generate_meshlets_by_verts
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=self.max_verts)

        # --- Step 3: Identify boundary verts (shared between meshlets) ---
        boundary_verts = _identify_meshlet_boundary_verts(meshlets, tris_np)

        # --- Step 4: QEM with boundary protection ---
        if self.verbose:
            print(f"  Meshlets: {len(meshlets)}, boundary verts: "
                  f"{len(boundary_verts)} ({len(boundary_verts)/n_v*100:.1f}%)")
            print("  QEM simplification (boundary protected)...")
        qem_result = progressive_simplify(
            verts_np, tris_np,
            target_ratios=[0.5],
            protected_vertices=boundary_verts)
        imp_order = qem_result["importance_order"]
        imp_rank = np.full(n_v, n_v, dtype=np.int64)
        for rank, vid in enumerate(imp_order):
            imp_rank[vid] = rank

        # --- Step 5: Per-meshlet encoding ---
        total_bits = (3 * 4 + 3 * 4 + 3 + 4 + 4) * 8  # global header
        total_vtx = 0
        total_conn = 0
        total_lod_mask = 0
        total_hdr = total_bits
        lod_tris_total = [0, 0, 0, 0]
        lod_verts_total = [0, 0, 0, 0]
        max_target_base = 0

        # Track topological cracks: boundary vert should be at LOD 0 in all its meshlets
        bnd_vert_min_lod = {}   # vid → min LOD level across all meshlets
        bnd_vert_max_lod = {}

        for ml_tris in meshlets:
            # All vertices in meshlet
            ml_vert_set = set()
            for ti in ml_tris:
                for j in range(3):
                    ml_vert_set.add(int(tris_np[ti, j]))
            n_mv = len(ml_vert_set)
            if n_mv < 1:
                continue

            # Split into boundary and interior
            bnd_in_ml = sorted(v for v in ml_vert_set if v in boundary_verts)
            int_in_ml = sorted((v for v in ml_vert_set if v not in boundary_verts),
                                key=lambda v: imp_rank[v])

            # Determine target_base.
            # LOD 0 has target_base positions at stride = n_padded/target_base.
            # Only positions < n_mv are "real" (rest are padding).
            # Count of real LOD 0 positions = ceil(n_mv / stride) = target_base * n_mv / n_padded.
            # We need this >= n_bnd, so target_base >= n_padded * n_bnd / n_mv.
            n_bnd = len(bnd_in_ml)
            n_padded = _next_pow2(max(n_mv, 32))
            if n_bnd > 0:
                min_tb = _next_pow2(max(1, int(np.ceil(n_padded * n_bnd / n_mv))))
            else:
                min_tb = 32
            target_base = max(32, min_tb)
            target_base = min(target_base, n_padded)
            max_target_base = max(max_target_base, target_base)

            # Importance-ordered list: boundary first (protected), then interior by rank
            ranked_verts = bnd_in_ml + int_in_ml  # length n_mv

            # Get LOD positions
            positions, lod_boundaries = lod_position_order(n_mv, target_base)
            # positions[k] = array position for rank k
            # lod_boundaries[L] = cumulative count up to and including LOD L

            # Map rank → array position
            rank_to_pos = positions  # already in this order
            # Build vid → position and vid → local-rank
            g2pos = {}
            g2l = {}
            for rank, vid in enumerate(ranked_verts):
                if rank < len(rank_to_pos):
                    g2pos[vid] = rank_to_pos[rank]
                    g2l[vid] = rank
                else:
                    # Shouldn't happen if n_mv matches
                    g2pos[vid] = rank_to_pos[-1]
                    g2l[vid] = rank

            # Build arranged array (padded to pow2) for wavelet input
            n_padded = _next_pow2(max(n_mv, target_base))
            arranged = np.zeros((n_padded, 3), dtype=np.int64)
            for rank, vid in enumerate(ranked_verts):
                p = rank_to_pos[rank]
                arranged[p] = global_codes[vid]
            # Pad unfilled positions: repeat last value
            used_positions = set(rank_to_pos[:n_mv])
            fill_val = global_codes[ranked_verts[-1]] if ranked_verts else [0, 0, 0]
            for p in range(n_padded):
                if p not in used_positions:
                    arranged[p] = fill_val

            # Per-meshlet header (approx): 45B
            ml_hdr = 45 * 8
            total_hdr += ml_hdr

            # Vertex encoding: Haar wavelet with meshlet's target_base
            vtx_bits = 0
            for d in range(3):
                w = estimate_wavelet_bits_int(
                    arranged[:, d], target_base, 'haar')
                vtx_bits += w["total_bits"]
            total_vtx += vtx_bits

            # LOD triangle masks based on position buckets
            # Positions available at LOD k = first lod_boundaries[k] entries
            # in the rank_to_pos list
            L = len(lod_boundaries)
            lod_position_sets = []
            for k in range(L):
                pos_set = set(rank_to_pos[:lod_boundaries[k]])
                lod_position_sets.append(pos_set)
            # Pad to 4 LOD levels for reporting consistency
            while len(lod_position_sets) < 4:
                lod_position_sets.append(lod_position_sets[-1])

            masks = _build_lod_masks_from_positions(
                ml_tris, tris_np, g2l, g2pos, lod_position_sets)
            mask_bits = _lod_mask_bits(masks)
            total_lod_mask += mask_bits

            for li, mask in enumerate(masks[:4]):
                lod_tris_total[li] += int(mask.sum())
                lod_verts_total[li] += len(lod_position_sets[li])

            # Track boundary vert LOD levels (for crack check)
            for vid in bnd_in_ml:
                rank = g2l[vid]
                # Which LOD level does this rank belong to?
                lod_level = 0
                for k, bnd in enumerate(lod_boundaries):
                    if rank < bnd:
                        lod_level = k
                        break
                bnd_vert_min_lod[vid] = min(bnd_vert_min_lod.get(vid, 99), lod_level)
                bnd_vert_max_lod[vid] = max(bnd_vert_max_lod.get(vid, -1), lod_level)

            # Connectivity (full-resolution)
            if self.connectivity == 'gts':
                conn_bits = _amd_gts_bits(ml_tris, tris_np, tri_adj)
            elif self.connectivity == 'edgebreaker':
                vo_eb, opcodes, n_roots = edgebreaker_vertex_order(
                    ml_tris, tris_np, tri_adj)
                conn_bits = _edgebreaker_bits(opcodes, n_roots, len(vo_eb))
            else:
                conn_bits = _amd_packed_bits(ml_tris, tris_np, tri_adj)
            total_conn += conn_bits

            total_bits += ml_hdr + vtx_bits + mask_bits + conn_bits

        # Crack-free verification: every boundary vert should be at LOD 0
        # in every meshlet that contains it
        n_bnd_not_lod0 = sum(1 for v, lvl in bnd_vert_max_lod.items() if lvl > 0)

        # Geometric crack check: same vert → same position across meshlets
        # (guaranteed by global quantization, but verify)
        all_recon = _dequantize_global(global_codes, g_min, g_range, g_bits)
        errors = np.linalg.norm(all_recon - vn, axis=1) * scale

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            pct = (errors <= self.precision_error).sum() / n_v * 100
            conn_name = self.connectivity.upper()
            print(f"MeshletWaveletLOD+{conn_name} (crack-free, progressive) "
                  f"mv={self.max_verts}:")
            print(f"  {len(meshlets)} meshlets, global bits={g_bits.tolist()}")
            print(f"  Max target_base: {max_target_base}")
            print(f"  Boundary verts at LOD>0 (topological cracks): "
                  f"{n_bnd_not_lod0} / {len(bnd_vert_max_lod)}")
            print(f"  Headers:      {total_hdr/8:>10,.0f} B "
                  f"({total_hdr/total_bits*100:.1f}%)")
            print(f"  Vertex data:  {total_vtx/8:>10,.0f} B "
                  f"({total_vtx/total_bits*100:.1f}%)")
            print(f"  LOD masks:    {total_lod_mask/8:>10,.0f} B "
                  f"({total_lod_mask/total_bits*100:.1f}%)")
            print(f"  Connectivity: {total_conn/8:>10,.0f} B "
                  f"({total_conn/total_bits*100:.1f}%)")
            print(f"  Total: {total_bits/8:,.0f}B  BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Ratio: {(n_v*96+n_t*96)/(total_bits/8):.1f}x  "
                  f"Accuracy: max={errors.max():.6f}  %OK={pct:.1f}%")
            print(f"  LOD triangle coverage:")
            for li in range(4):
                avg_v = lod_verts_total[li] / max(len(meshlets), 1)
                print(f"    LOD {li} (~{avg_v:>3.0f} v/ml avg): "
                      f"{lod_tris_total[li]:>8,} / {n_t:>8,} tris "
                      f"({lod_tris_total[li]/n_t*100:.1f}%)")

        data = bytes(int(np.ceil(total_bits / 8)))
        return CompressedModel(data, bpv, bpt)

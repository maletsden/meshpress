"""
Connectivity-free centroid+offsets encoder (Idea #21 in docs/compression_ideas.md).

Pipeline:
  1. Quantize all vertices to a global integer grid at per_coord_err.
  2. Compute integer centroid per triangle: centroid = round(sum(v_codes)/3)
     on the same v-grid (integer arithmetic).
  3. Sort triangles by Morton(centroid).
  4. Group sorted triangles into chunks (default 256). NO vertex meshlets;
     these are pure tri-chunks.
  5. Per chunk:
     - Centroid stream: anchor (absolute, sum(v_bits) bits) +
                        deltas along the Morton-sorted order (per-axis,
                        Shannon-entropy-estimated bits).
     - Offset stream:   r0 = v_codes[t,0] - centroid_codes[t]
                        r1 = v_codes[t,1] - centroid_codes[t]
                        each per-axis, Shannon-entropy-estimated.
     - Correction stream: ε = (sum(v_codes[t,k]) for k=0..2) - 3*centroid_codes[t]
                          ε ∈ {-2, -1, 0, +1, +2}^3 per axis (integer rounding error).
                          Encode as 3 bits/axis (8 levels — covers -3..+4).
     At decode:  r2 = ε - r0 - r1.    v_codes[t,k] = centroid_codes[t] + r_k.

Crack-free: every triangle containing vertex v reconstructs the same integer
v_code for v (because the input v_codes were the same integers). Spatial-hash
dedup at decode is exact.

NO connectivity bits. Vertex sharing is recovered at decode by a spatial-hash
de-duplication pass on the reconstructed integer codes (within zero tolerance —
exact integer match).

This is a Phase-1 estimation encoder: it computes the bit cost via Shannon
entropy + fixed-width fallback (matching the rest of MeshPress) and reports
BPV/BPT. No real bitstream is emitted; the returned bytes blob is a buffer of
the right size (consistent with other estimation encoders in this project).
"""

from __future__ import annotations

import numpy as np
from collections import Counter

from utils.types import Model, CompressedModel
from ..encoder import Encoder


# ----------------------------------------------------------------------
# Helpers (mirrors meshlet_wavelet.py utilities; duplicated here to keep
# this encoder dependency-light and self-contained for readers).
# ----------------------------------------------------------------------

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


def _bits_for_range(int_range):
    """Min bits to represent values in [0, int_range] inclusive."""
    if int_range <= 0:
        return 1
    return max(1, int(np.ceil(np.log2(int_range + 1))))


def _bits_for_error(val_range, max_err):
    if max_err <= 0 or val_range <= 0:
        return 1
    return max(1, int(np.ceil(np.log2(val_range / (2 * max_err) + 1))))


def _shannon(codes):
    if len(codes) == 0:
        return 0.0
    counts = Counter(codes.tolist() if hasattr(codes, "tolist") else list(codes))
    total = len(codes)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


def _stream_bits(codes, fixed_bits, hdr_bits=24):
    """Pick min(fixed-width, arith-coded estimate). hdr_bits accounts for
    per-stream entropy table fingerprint."""
    n = len(codes)
    if n == 0:
        return 0
    plain = n * fixed_bits
    arith = n * _shannon(codes) + hdr_bits
    return min(plain, arith)


def _spread3_21(v):
    v = v & 0x1FFFFF
    v = (v | (v << 32)) & 0x001F00000000FFFF
    v = (v | (v << 16)) & 0x001F0000FF0000FF
    v = (v | (v << 8))  & 0x100F00F00F00F00F
    v = (v | (v << 4))  & 0x10C30C30C30C30C3
    v = (v | (v << 2))  & 0x1249249249249249
    return v


def _morton3(int_xyz):
    """Vectorized 3D Morton code. int_xyz is (n, 3) int."""
    arr = np.asarray(int_xyz, dtype=np.int64)
    x = arr[:, 0]; y = arr[:, 1]; z = arr[:, 2]
    return _spread3_21(x) | (_spread3_21(y) << 1) | (_spread3_21(z) << 2)


def _global_quantize(vn, per_coord_err):
    """Quantize vn to a global int grid at per_coord_err per axis."""
    g_min = vn.min(axis=0)
    g_max = vn.max(axis=0)
    g_range = g_max - g_min
    g_range = np.where(g_range < 1e-12, 1e-12, g_range)
    g_bits = np.array([_bits_for_error(g_range[d], per_coord_err)
                       for d in range(3)], dtype=np.int64)
    codes = np.zeros_like(vn, dtype=np.int64)
    for d in range(3):
        mx = (1 << int(g_bits[d])) - 1
        codes[:, d] = np.round(
            (vn[:, d] - g_min[d]) / g_range[d] * mx
        ).clip(0, mx).astype(np.int64)
    return codes, g_min, g_range, g_bits


def _dequantize_global(codes, g_min, g_range, g_bits):
    out = np.empty(codes.shape, dtype=np.float64)
    for d in range(3):
        mx = (1 << int(g_bits[d])) - 1
        out[..., d] = codes[..., d].astype(np.float64) / max(1, mx) * g_range[d] + g_min[d]
    return out


# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------

class MeshletCentroidNoConn(Encoder):
    """Connectivity-free triangle-as-(centroid, offsets) encoder.

    Args:
        chunk_size: triangles per Morton-sorted tri-chunk.
        precision_error: physical-space max error budget (per coord, after
            scale-back from unit-sphere normalization).
        predictor: 'zero' | 'mirror' | 'neighbor' | 'best'.
            'zero':     r1 encoded directly (no prediction).
            'mirror':   pred(r1) = -r0 (mirror across centroid).
            'neighbor': pred(r0[t]) = r0[t-1], pred(r1[t]) = r1[t-1] from the
                        previous triangle in Morton-sorted chunk. Exploits
                        the assumption that neighboring triangles in 3D
                        Morton order have similar shapes.
            'best':     try all three per chunk, pick min, add 2-bit flag.
        verbose: print breakdown.
    """

    def __init__(self, chunk_size=256, precision_error=0.0005,
                 predictor='zero', sort='morton', verbose=False):
        self.chunk_size = chunk_size
        self.precision_error = precision_error
        self.predictor = predictor
        self.sort = sort  # 'morton' | 'hilbert'
        self.verbose = verbose

    # ------------------------------------------------------------------
    def encode(self, model: Model) -> CompressedModel:
        verts, tris = _to_numpy(model)
        n_v, n_t = len(verts), len(tris)
        if n_v == 0 or n_t == 0:
            return CompressedModel(b"", 0.0, 0.0)

        # Center & scale to unit sphere — same convention as the rest of
        # the MeshPress encoders (so per_coord_err is comparable).
        center = verts.mean(axis=0)
        vc = verts - center
        scale = float(np.max(np.linalg.norm(vc, axis=1)))
        if scale < 1e-12:
            scale = 1.0
        vn = vc / scale
        per_coord_err = self.precision_error / scale / np.sqrt(3)

        # 1. Global integer quantization.
        v_codes, g_min, g_range, v_bits = _global_quantize(vn, per_coord_err)

        # 2. Integer centroids on the v-grid: floor-div((sum)/3) is biased
        # toward -inf; round((sum + 1)/3 - 0.0) is symmetric. We use plain
        # numpy round on the float average.
        sum_codes = v_codes[tris[:, 0]] + v_codes[tris[:, 1]] + v_codes[tris[:, 2]]
        cen_codes = np.round(sum_codes.astype(np.float64) / 3.0).astype(np.int64)

        # 3. Sort triangles by Morton or Hilbert curve over centroids.
        if self.sort == 'hilbert':
            from utils.interior_sorts import hilbert3_codes
            keys = hilbert3_codes(cen_codes)
        else:
            keys = _morton3(cen_codes)
        order = np.argsort(keys, kind="stable")

        cen_sorted = cen_codes[order]              # (n_t, 3)
        v_sorted = v_codes[tris][order]            # (n_t, 3, 3)
        # 4. Chunking.
        n_chunks = (n_t + self.chunk_size - 1) // self.chunk_size

        # ----- bit accounting -----
        # Global header layout (bytes):
        #   g_min[3*4] + g_range[3*4] + v_bits[3*1] + n_t[4]
        #   + chunk_size[2] + predictor_id[1] = 30 B
        global_header_bits = 30 * 8

        total_anchor = 0
        total_cen_delta = 0
        total_off_r0 = 0
        total_off_r1 = 0
        total_correction = 0
        total_chunk_hdr = 0

        # Per-chunk header (bytes):
        #   n_chunk[2] + cen_axis_bits[3] + r0_axis_bits[3] + r1_axis_bits[3]
        #   + correction_axis_bits[3] + entropy_table_fingerprint[16] = 30 B
        chunk_header_bits = 30 * 8

        # Reconstruction-error tracking.
        all_err = []

        for ci in range(n_chunks):
            s = ci * self.chunk_size
            e = min(s + self.chunk_size, n_t)
            n_chunk = e - s
            if n_chunk == 0:
                continue
            total_chunk_hdr += chunk_header_bits

            cs = cen_sorted[s:e]                    # (n_chunk, 3)
            vs = v_sorted[s:e]                      # (n_chunk, 3, 3)

            # ---- Centroid stream ----
            anchor = cs[0]
            anchor_bits = int(v_bits.sum())
            total_anchor += anchor_bits

            if n_chunk > 1:
                # Per-axis deltas. Range can be negative; shift to non-negative.
                deltas = cs[1:] - cs[:-1]           # (n_chunk-1, 3)
                for d in range(3):
                    col = deltas[:, d].astype(np.int64)
                    lo = int(col.min())
                    hi = int(col.max())
                    rng = hi - lo
                    bits_axis = _bits_for_range(rng)
                    shifted = (col - lo).astype(np.int64)
                    # +20 b for storing (lo, rng) per-axis fingerprint.
                    sb = _stream_bits(shifted, bits_axis, hdr_bits=24)
                    total_cen_delta += sb

            # ---- Offsets r0, r1 ----
            r0 = vs[:, 0, :].astype(np.int64) - cs   # (n_chunk, 3)
            r1 = vs[:, 1, :].astype(np.int64) - cs   # (n_chunk, 3)

            def _stream_for(arr3):
                """Per-axis Shannon-estimated bits for a (n, 3) int array."""
                tot = 0
                for d in range(3):
                    col = arr3[:, d].astype(np.int64)
                    lo = int(col.min()) if len(col) else 0
                    hi = int(col.max()) if len(col) else 0
                    rng = hi - lo
                    bits_axis = _bits_for_range(rng)
                    shifted = (col - lo).astype(np.int64)
                    tot += _stream_bits(shifted, bits_axis, hdr_bits=24)
                return tot

            def _residuals_for_predictor(name):
                """Return (r0_resid, r1_resid) for the given predictor name."""
                if name == 'mirror':
                    return r0, r1 - (-r0)
                if name == 'neighbor':
                    # r0[t] - r0[t-1]; r0[0] left as-is.
                    r0r = r0.copy()
                    r0r[1:] = r0[1:] - r0[:-1]
                    r1r = r1.copy()
                    r1r[1:] = r1[1:] - r1[:-1]
                    return r0r, r1r
                # 'zero' default
                return r0, r1

            if self.predictor == 'best':
                cands = []
                for name in ('zero', 'mirror', 'neighbor'):
                    a, b = _residuals_for_predictor(name)
                    cands.append((_stream_for(a) + _stream_for(b), name, a, b))
                cands.sort(key=lambda x: x[0])
                _, _chosen, r0_resid, r1_resid = cands[0]
                # 2-bit selector flag per chunk
                sel_bits = 2
                total_off_r0 += _stream_for(r0_resid) + sel_bits
                total_off_r1 += _stream_for(r1_resid)
            else:
                r0_resid, r1_resid = _residuals_for_predictor(self.predictor)
                total_off_r0 += _stream_for(r0_resid)
                total_off_r1 += _stream_for(r1_resid)

            # ---- Correction: ε = sum(v_codes) - 3*cen_codes ----
            sum_v = vs.sum(axis=1).astype(np.int64)              # (n_chunk, 3)
            eps = sum_v - 3 * cs                                  # ∈ {-2..+2} per axis
            for d in range(3):
                col = eps[:, d].astype(np.int64)
                lo = int(col.min()); hi = int(col.max())
                rng = hi - lo
                bits_axis = _bits_for_range(rng)
                shifted = (col - lo).astype(np.int64)
                sb = _stream_bits(shifted, bits_axis, hdr_bits=16)
                total_correction += sb

            # ---- Reconstruction sanity check ----
            # decoder reconstructs v_codes[t, k] = cs + r_k
            # where r2 = eps - r0 - r1. Check end-to-end.
            r2 = eps - r0 - r1
            v_recon_codes = np.stack([cs + r0, cs + r1, cs + r2], axis=1)
            assert np.array_equal(v_recon_codes, vs), \
                f"Centroid encoder reconstruction mismatch at chunk {ci}"

            # Dequantize & measure error in physical space.
            v_recon_phys_unit = _dequantize_global(v_recon_codes, g_min, g_range, v_bits)
            v_orig_unit = vn[tris[order[s:e]]]
            err = np.linalg.norm(v_recon_phys_unit - v_orig_unit, axis=2) * scale
            all_err.append(err.flatten())

        total_bits = (
            global_header_bits + total_chunk_hdr +
            total_anchor + total_cen_delta +
            total_off_r0 + total_off_r1 + total_correction
        )

        bpv = total_bits / n_v
        bpt = total_bits / n_t

        if self.verbose:
            err_arr = np.concatenate(all_err) if all_err else np.array([0.0])
            pct_ok = float((err_arr <= self.precision_error).sum()) / max(1, len(err_arr)) * 100.0
            print(f"MeshletCentroidNoConn  chunk={self.chunk_size}  "
                  f"predictor={self.predictor}:")
            print(f"  n_v={n_v}  n_t={n_t}  n_chunks={n_chunks}")
            print(f"  v_bits={v_bits.tolist()}  scale={scale:.4f}")
            print(f"  Global hdr:        {global_header_bits/8:>10,.0f} B")
            print(f"  Chunk headers:     {total_chunk_hdr/8:>10,.0f} B")
            print(f"  Cen anchors:       {total_anchor/8:>10,.0f} B")
            print(f"  Cen deltas:        {total_cen_delta/8:>10,.0f} B")
            print(f"  Off r0:            {total_off_r0/8:>10,.0f} B")
            print(f"  Off r1:            {total_off_r1/8:>10,.0f} B")
            print(f"  Correction (eps):  {total_correction/8:>10,.0f} B")
            print(f"  Total:             {total_bits/8:>10,.0f} B  "
                  f"BPV={bpv:.2f}  BPT={bpt:.2f}")
            print(f"  Max-err: {err_arr.max():.6f}  %OK={pct_ok:.1f}%")

        return CompressedModel(bytes(int(np.ceil(total_bits / 8))), bpv, bpt)

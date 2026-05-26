"""STRIDE-dup: per-meshlet self-contained variant of ParaDelta v5.

Differences from v5:
  * No global boundary table — each meshlet stores ALL its verts inline.
  * No per-meshlet refs table — global vert IDs are dense (per-meshlet
    base + local id).
  * Vert prediction: integer parallelogram (a + b - c) on quantized codes.
  * Walk order classifies each new vert as:
      anchor (1 per meshlet)   — raw int triple at g_bits/axis,
      delta  (extra strip roots) — diff from previously decoded vert,
      para   (interior of strip) — parallelogram int residual.

Bitstream layout (little-endian within byte, MSB-first inside fields):

  GLOBAL HEADER
    magic        u32  'DPR4' = 0x44505234
    version      u8   1
    code_width   u8   0 (i16 SMEM hint) / 1 (i32)
    center       3×f32
    scale        f32
    per_coord_err f32 (normalized)
    g_min        3×f32
    g_range      3×f32
    g_bits       3×u8
    n_v          u32  (= sum of per-meshlet n_local)
    n_t          u32
    n_meshlets   u32
    pad to byte boundary
    offset_table n_meshlets × u32  (bit offsets into meshlet region)

  PER MESHLET
    n_local      u16
    n_tris       u16
    n_strips     u16

    CONNECTIVITY (same encoding as v5)
      idx_bits = ceil(log2(n_local + 1))
      for each strip:
        strip_len u16
        3 root verts: emit_vert each (flag + idx/fifo)
        per subsequent tri: edge_code (1 bit) + emit_vert(new)

    VERT SECTION (in walk order)
      ANCHOR (1 vert)
        x  g_bits[0]
        y  g_bits[1]
        z  g_bits[2]
      DELTA STREAM (extra strip-root verts, walk order)
        for d in 0..2:
          tag u8 (1 = Rice, 2 = EG)
          k   u8
          coded body × n_delta
      PARA STREAM (parallelogram-predicted verts, walk order)
        for d in 0..2:
          tag u8
          k   u8
          coded body × n_para

  Decoder reproduces walk order from connectivity (same _root_orient
  + strip step rules), so n_delta and n_para are implicit.
"""
from __future__ import annotations

import math
import numpy as np
from collections import deque

from utils.bit_codec import BitWriter, BitReader
from encoder.paradelta_codec import (
    REUSE_BUF_SIZE, _idx_bits_for, _emit_vert, _read_vert,
    _best_rice_k, _best_eg_k, _root_orient,
)
from utils.residual_entropy import _zigzag


MAGIC_DUP = 0x44505234  # 'DPR4' as little-endian C string
VERSION_DUP = 3  # v3: + 12 B generalized parallelogram predictor header


# =====================================================================
# Integer-Rational Linear Predictor (IRLP). Per-axis weight (n, K) stored
# in the per-mesh bitstream header. Canonical parallelogram is the special
# case (n, K) = ((1, 1, -1), 0), which the formula evaluates to a + b - c
# exactly — single unconditional decode path, no branch.
# =====================================================================

CANON_N = np.array([[1, 1, -1]] * 3, dtype=np.int64)
CANON_K = np.zeros(3, dtype=np.int64)


def _irlp_pred(a_codes, b_codes, c_codes, n_3x3, K_3):
    """Integer-rational linear predictor per axis. (a, b, c)_codes are
    length-3 int64 vectors. Returns length-3 int64 prediction."""
    out = np.empty(3, dtype=np.int64)
    for d in range(3):
        K = int(K_3[d])
        n0 = int(n_3x3[d, 0]); n1 = int(n_3x3[d, 1]); n2 = int(n_3x3[d, 2])
        s = n0 * int(a_codes[d]) + n1 * int(b_codes[d]) + n2 * int(c_codes[d])
        if K == 0:
            out[d] = s
        else:
            half = 1 << (K - 1)
            out[d] = (s + half) >> K
    return out


# =====================================================================
# Walk: classify each new vert encountered in strip order.
# Returns list of (local_id, kind, refs) where kind ∈ {'none', 'para'}.
# 'none' verts are split by the encoder into the first (anchor) and the
# rest (delta-from-prev). Decoder applies the same split.
# =====================================================================

def _walk_meshlet(plan):
    ml_tris_local = np.asarray(plan["ml_tris_local"], dtype=np.int64)
    strips = plan["strips"]
    n_local = plan["n_bnd"] + plan["n_int"]
    decoded = np.zeros(n_local, dtype=bool)
    order = []
    for strip in strips:
        root_id = strip[0]
        next_id = strip[1] if len(strip) > 1 else None
        root = _root_orient(ml_tris_local, root_id, next_id)
        for v in root:
            v = int(v)
            if not decoded[v]:
                order.append((v, 'none', None))
                decoded[v] = True
        prev_tri = list(root)
        for li in strip[1:]:
            tri_v = [int(x) for x in ml_tris_local[li]]
            tri_set = set(tri_v); prev_set = set(prev_tri)
            shared = tri_set & prev_set
            if len(shared) != 2:
                continue
            new_v = next(iter(tri_set - shared))
            pair_newest = frozenset((prev_tri[1], prev_tri[2]))
            pair_second = frozenset((prev_tri[0], prev_tri[2]))
            shared_fs = frozenset(shared)
            if shared_fs == pair_newest:
                a, b, c = prev_tri[1], prev_tri[2], prev_tri[0]
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            elif shared_fs == pair_second:
                a, b, c = prev_tri[0], prev_tri[2], prev_tri[1]
                new_prev = [prev_tri[0], prev_tri[2], new_v]
            else:
                continue
            if not decoded[new_v]:
                order.append((new_v, 'para', (a, b, c)))
                decoded[new_v] = True
            prev_tri = new_prev
    return order


# =====================================================================
# Encoder
# =====================================================================

def _pick_best_k(codes_arr: np.ndarray) -> tuple[list[int], list[int]]:
    """Per-axis best (tag, k) for codes_arr of shape (n, 3)."""
    tags, ks = [], []
    for d in range(3):
        arr = codes_arr[:, d].astype(np.int64) if codes_arr.size else np.zeros(0, np.int64)
        u_arr = _zigzag(arr)
        rice_k, rice_body = _best_rice_k(u_arr) if u_arr.size else (0, 0)
        eg_k, eg_body = _best_eg_k(u_arr) if u_arr.size else (0, 0)
        if rice_body <= eg_body:
            tags.append(1); ks.append(rice_k)
        else:
            tags.append(2); ks.append(eg_k)
    return tags, ks


def _emit_header(w: BitWriter, tags: list[int], ks: list[int]):
    for d in range(3):
        w.write_fixed(tags[d], 8)
        w.write_fixed(ks[d], 8)


def _emit_vert_code(w: BitWriter, code: np.ndarray, tags: list[int],
                     ks: list[int]):
    """Emit one vert's 3-axis residual using per-axis (tag, k)."""
    for d in range(3):
        u = int(code[d])
        u = (u << 1) ^ (u >> 63) if u < 0 else (u << 1)  # zigzag
        if tags[d] == 1:
            w.write_rice(int(u), ks[d])
        else:
            w.write_exp_golomb(int(u), ks[d])


def _emit_axis_stream(w: BitWriter, codes: np.ndarray, axis: int,
                       tag: int, k: int):
    """Emit one axis's residual stream from a (N,3) codes array."""
    if codes.size == 0:
        return
    for i in range(codes.shape[0]):
        u = int(codes[i, axis])
        u = (u << 1) ^ (u >> 63) if u < 0 else (u << 1)
        if tag == 1:
            w.write_rice(int(u), k)
        else:
            w.write_exp_golomb(int(u), k)


def _read_int_stream(r: BitReader, n: int) -> np.ndarray:
    """Compatibility (used by old decoder). For vert-major decode use
    _read_vert_code below."""
    out = np.zeros((n, 3), dtype=np.int64)
    if n == 0:
        return out
    for d in range(3):
        tag = r.read_fixed(8)
        k = r.read_fixed(8)
        if tag == 1:
            for i in range(n):
                u = r.read_rice(k)
                out[i, d] = (u >> 1) ^ -(u & 1)
        elif tag == 2:
            for i in range(n):
                u = r.read_exp_golomb(k)
                out[i, d] = (u >> 1) ^ -(u & 1)
        else:
            raise ValueError(f"bad tag {tag}")
    return out


def _read_vert_code(r: BitReader, tags: list[int], ks: list[int]) -> np.ndarray:
    out = np.zeros(3, dtype=np.int64)
    for d in range(3):
        if tags[d] == 1:
            u = r.read_rice(ks[d])
        else:
            u = r.read_exp_golomb(ks[d])
        out[d] = (u >> 1) ^ -(u & 1)
    return out


def _write_meshlet_dup(w: BitWriter, plan, anchor_code, delta_codes,
                       para_codes, walk_order, n_local, n_tris_m, n_strips,
                       g_bits):
    """Write one meshlet (vert-major residual layout v2).

    Bitstream order:
      n_local, n_tris, n_strips u16 each
      Connectivity (strips, same as v5)
      Anchor: g_bits[0]+g_bits[1]+g_bits[2] bits
      Delta header: 3 × (tag u8, k u8) = 48 bits
      Para header:  3 × (tag u8, k u8) = 48 bits
      Vert residuals (interleaved 3 axes, in walk order, anchor skipped):
        for each non-anchor vert: 3 × Rice/EG using its kind's header.
    """
    w.write_fixed(n_local, 16)
    w.write_fixed(n_tris_m, 16)
    w.write_fixed(n_strips, 16)

    # Connectivity (same as v5)
    idx_bits = _idx_bits_for(n_local)
    reuse_fifo: deque[int] = deque(maxlen=REUSE_BUF_SIZE)
    ml_tris_local = plan["ml_tris_local"]
    for strip in plan["strips"]:
        strip_len = len(strip)
        w.write_fixed(strip_len, 16)
        root_id = strip[0]
        next_id = strip[1] if len(strip) > 1 else None
        root = _root_orient(ml_tris_local, root_id, next_id)
        for v in root:
            _emit_vert(w, int(v), reuse_fifo, idx_bits)
        prev_tri = list(root)
        for li in strip[1:]:
            tri_v = [int(x) for x in ml_tris_local[li]]
            tri_set = set(tri_v); prev_set = set(prev_tri)
            shared = tri_set & prev_set
            new_v = next(iter(tri_set - shared))
            pair_newest = frozenset((prev_tri[1], prev_tri[2]))
            pair_second = frozenset((prev_tri[0], prev_tri[2]))
            shared_fs = frozenset(shared)
            if shared_fs == pair_newest:
                ec = 0
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            elif shared_fs == pair_second:
                ec = 1
                new_prev = [prev_tri[0], prev_tri[2], new_v]
            else:
                raise RuntimeError("oldest-edge share")
            w.write_bits(ec, 1)
            _emit_vert(w, new_v, reuse_fifo, idx_bits)
            prev_tri = new_prev

    # Anchor — raw int per axis at g_bits.
    resid_start_bit = w.bit_pos()
    for d in range(3):
        w.write_fixed(int(anchor_code[d]), int(g_bits[d]))
    # Per-axis (tag, k) headers for delta + para streams.
    d_tags, d_ks = _pick_best_k(delta_codes)
    p_tags, p_ks = _pick_best_k(para_codes)
    _emit_header(w, d_tags, d_ks)
    _emit_header(w, p_tags, p_ks)
    # v4 axis-separated emit. Sub-offsets are relative to resid_start_bit.
    # Order: delta_x, delta_y, delta_z, para_x, para_y, para_z.
    sub_offs = []   # 5 offsets: starts of y, z, para_x, para_y, para_z
    _emit_axis_stream(w, delta_codes, 0, d_tags[0], d_ks[0])
    sub_offs.append(w.bit_pos() - resid_start_bit)  # delta_y start
    _emit_axis_stream(w, delta_codes, 1, d_tags[1], d_ks[1])
    sub_offs.append(w.bit_pos() - resid_start_bit)  # delta_z start
    _emit_axis_stream(w, delta_codes, 2, d_tags[2], d_ks[2])
    sub_offs.append(w.bit_pos() - resid_start_bit)  # para_x start
    _emit_axis_stream(w, para_codes, 0, p_tags[0], p_ks[0])
    sub_offs.append(w.bit_pos() - resid_start_bit)  # para_y start
    _emit_axis_stream(w, para_codes, 1, p_tags[1], p_ks[1])
    sub_offs.append(w.bit_pos() - resid_start_bit)  # para_z start
    _emit_axis_stream(w, para_codes, 2, p_tags[2], p_ks[2])
    return resid_start_bit, sub_offs


def encode_dup(prep: dict, *, verbose: bool = False, return_meta: bool = False,
               predictor: str = "generalized"):
    """Encode a STRIDE-dup bitstream from a v5 prep dict.

    predictor:
      "generalized"  fit per-mesh int-rational weights, write 12 B header,
                     fall back to canonical per-axis when canonical is better.
      "canonical"    skip the fit; emit (1,1,-1)/K=0 header. Same bitstream
                     format as "generalized" — header is mandatory in v3.
    """
    center = prep["center"]; scale = prep["scale"]
    per_coord_err = prep["per_coord_err"]
    g_min = prep["g_min"]; g_range = prep["g_range"]; g_bits = prep["g_bits"]
    n_v = prep["n_v"]; n_t = prep["n_t"]
    n_meshlets = prep["n_meshlets"]
    global_codes = prep["global_codes"]
    plans = prep["plans"]

    # Predictor fit (or canonical).
    if predictor == "generalized":
        from encoder._irlp_fit import fit_predictor
        pred_n, pred_K = fit_predictor(prep)
    elif predictor == "canonical":
        pred_n = CANON_N.copy(); pred_K = CANON_K.copy()
    else:
        raise ValueError(f"predictor must be 'generalized' or 'canonical', got {predictor!r}")

    # Pass 1: walk every meshlet, compute residuals, gather stats.
    per_meshlet = []
    max_abs_code = 0
    total_local = 0
    for plan in plans:
        local_to_global = np.asarray(plan["local_to_global"], dtype=np.int64)
        n_local = len(local_to_global)
        true_codes = global_codes[local_to_global].astype(np.int64)
        order = _walk_meshlet(plan)
        anchor_code = None
        delta_codes = []
        para_codes  = []
        prev_decoded = None
        for v, kind, refs in order:
            tc = true_codes[v]
            if kind == 'none':
                if anchor_code is None:
                    anchor_code = tc.copy()
                else:
                    delta_codes.append(tc - prev_decoded)
            else:
                a, b, c = refs
                # Generalized integer-rational parallelogram.
                pred = _irlp_pred(true_codes[a], true_codes[b], true_codes[c],
                              pred_n, pred_K)
                para_codes.append(tc - pred)
            prev_decoded = tc
        if anchor_code is None:
            # Empty meshlet — should never happen.
            anchor_code = np.zeros(3, dtype=np.int64)
        delta_arr = (np.array(delta_codes, dtype=np.int64)
                     if delta_codes else np.zeros((0, 3), dtype=np.int64))
        para_arr  = (np.array(para_codes,  dtype=np.int64)
                     if para_codes  else np.zeros((0, 3), dtype=np.int64))
        if delta_arr.size:
            max_abs_code = max(max_abs_code, int(np.abs(delta_arr).max()))
        if para_arr.size:
            max_abs_code = max(max_abs_code, int(np.abs(para_arr).max()))
        per_meshlet.append({
            "n_local": n_local,
            "n_tris_m": plan["n_tris_m"],
            "n_strips": plan["n_strips"],
            "plan": plan,
            "anchor": anchor_code,
            "deltas": delta_arr,
            "paras":  para_arr,
            "order":  order,
        })
        total_local += n_local
    code_width_flag = 1 if max_abs_code > 32767 else 0

    # Pass 2: emit per-meshlet bytes into a temp buffer to record offsets.
    body_w = BitWriter()
    meshlet_offsets_rel = []
    resid_offs_rel = []      # absolute bit offset of anchor start (body-relative)
    n_kind0_arr    = []
    axis_sub_offs  = []      # 5 sub-offsets per meshlet (relative to resid start)
    for m in per_meshlet:
        meshlet_offsets_rel.append(body_w.bit_pos())
        resid_off, subs = _write_meshlet_dup(body_w, m["plan"],
                            m["anchor"], m["deltas"], m["paras"],
                            m["order"],
                            m["n_local"], m["n_tris_m"], m["n_strips"],
                            g_bits)
        resid_offs_rel.append(resid_off)
        n_kind0_arr.append(m["deltas"].shape[0] + 1)  # +1 for anchor
        axis_sub_offs.append(subs)
    body_bytes = body_w.finalize()

    # Globals.
    w = BitWriter()
    w.write_fixed(MAGIC_DUP, 32)
    w.write_fixed(VERSION_DUP, 8)
    w.write_fixed(code_width_flag, 8)
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
    # v3 predictor header: 9 int16 numerators (row-major 3x3) + 3 uint8 K = 21 B.
    for d in range(3):
        for j in range(3):
            n_val = int(pred_n[d, j]) & 0xFFFF   # int16 two's complement
            w.write_fixed(n_val, 16)
    for d in range(3):
        w.write_fixed(int(pred_K[d]) & 0xFF, 8)
    w.write_fixed(total_local, 32)
    w.write_fixed(n_t, 32)
    w.write_fixed(n_meshlets, 32)

    # Pad to byte; offset table.
    pad = (-w.bit_pos()) & 7
    if pad:
        w.write_bits(0, pad)
    offset_table_start = w.bit_pos()
    offset_table_bits = n_meshlets * 32
    meshlet_region_start = offset_table_start + offset_table_bits
    for rel in meshlet_offsets_rel:
        w.write_fixed(meshlet_region_start + rel, 32)
    header_bytes = w.finalize()

    if verbose:
        bpv_src = 8 * (len(header_bytes) + len(body_bytes)) / n_v
        print(f"  dup: n_v_src={n_v:,} n_v_dup={total_local:,}  "
              f"max|code|={max_abs_code:,}  width={'i32' if code_width_flag else 'i16'}  "
              f"BPV(src)={bpv_src:.2f}")

    data = header_bytes + body_bytes
    if return_meta:
        meta = {
            "meshlet_offsets_rel": meshlet_offsets_rel,
            "resid_offs_rel":      resid_offs_rel,
            "n_kind0":             n_kind0_arr,
            "axis_sub_offs":       axis_sub_offs,
        }
        return data, meta
    return data


# =====================================================================
# Decoder (CPU, for roundtrip verification)
# =====================================================================

def decode_dup(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    r = BitReader(data)
    magic = r.read_fixed(32)
    if magic != MAGIC_DUP:
        raise ValueError(f"bad magic 0x{magic:08X}")
    version = r.read_fixed(8)
    if version != VERSION_DUP:
        raise ValueError(f"version {version}")
    _code_width = r.read_fixed(8)
    center = np.array([r.read_f32() for _ in range(3)], dtype=np.float64)
    scale = float(r.read_f32())
    _per_coord_err = float(r.read_f32())
    g_min = np.array([r.read_f32() for _ in range(3)], dtype=np.float64)
    g_range = np.array([r.read_f32() for _ in range(3)], dtype=np.float64)
    g_bits = np.array([r.read_fixed(8) for _ in range(3)], dtype=np.int32)
    # v3 predictor header: 9 int16 numerators + 3 uint8 K.
    pred_n = np.zeros((3, 3), dtype=np.int64)
    for d in range(3):
        for j in range(3):
            v_u16 = r.read_fixed(16)
            pred_n[d, j] = v_u16 - 0x10000 if v_u16 & 0x8000 else v_u16
    pred_K = np.array([r.read_fixed(8) for _ in range(3)], dtype=np.int64)
    n_v_total = r.read_fixed(32)
    _n_t = r.read_fixed(32)
    n_meshlets = r.read_fixed(32)
    pad = (-r.bit_pos()) & 7
    if pad:
        r.read_bits(pad)
    # Skip offset table — CPU decoder is sequential.
    for _ in range(n_meshlets):
        r.read_fixed(32)

    all_codes_local: list[np.ndarray] = []
    all_tris: list[tuple[int, int, int]] = []
    v_base = 0

    for _ in range(n_meshlets):
        n_local = r.read_fixed(16)
        n_tris_m = r.read_fixed(16)
        n_strips = r.read_fixed(16)

        # Connectivity decode — also yields walk order classification.
        idx_bits = _idx_bits_for(n_local)
        reuse_fifo: deque[int] = deque(maxlen=REUSE_BUF_SIZE)
        local_tris: list[tuple[int, int, int]] = []
        walk: list[tuple[int, str, tuple]] = []
        decoded_set = set()
        for _s in range(n_strips):
            strip_len = r.read_fixed(16)
            v0 = _read_vert(r, reuse_fifo, idx_bits)
            v1 = _read_vert(r, reuse_fifo, idx_bits)
            v2 = _read_vert(r, reuse_fifo, idx_bits)
            local_tris.append((v0, v1, v2))
            for v in (v0, v1, v2):
                if v not in decoded_set:
                    walk.append((v, 'none', ()))
                    decoded_set.add(v)
            prev_tri = [v0, v1, v2]
            for _ in range(strip_len - 1):
                ec = r.read_bits(1)
                new_v = _read_vert(r, reuse_fifo, idx_bits)
                if ec == 0:
                    a, b, c = prev_tri[1], prev_tri[2], prev_tri[0]
                    new_prev = [prev_tri[1], prev_tri[2], new_v]
                else:
                    a, b, c = prev_tri[0], prev_tri[2], prev_tri[1]
                    new_prev = [prev_tri[0], prev_tri[2], new_v]
                local_tris.append((a, b, new_v))
                if new_v not in decoded_set:
                    walk.append((new_v, 'para', (a, b, c)))
                    decoded_set.add(new_v)
                prev_tri = new_prev

        # Anchor.
        anchor = np.array([r.read_fixed(int(g_bits[d])) for d in range(3)],
                          dtype=np.int64)
        # Delta per-axis (tag, k); then para per-axis (tag, k). Each
        # _emit_header writes 3 (tag, k) pairs interleaved as tag, k, tag,
        # k, tag, k.
        d_tags = []; d_ks = []
        for _ in range(3):
            d_tags.append(r.read_fixed(8))
            d_ks.append(r.read_fixed(8))
        p_tags = []; p_ks = []
        for _ in range(3):
            p_tags.append(r.read_fixed(8))
            p_ks.append(r.read_fixed(8))

        # v4: read axis-separated substreams in order:
        # delta_x, delta_y, delta_z, para_x, para_y, para_z.
        n_kind0 = sum(1 for _, k, _ in walk if k == 'none')
        n_kind1 = sum(1 for _, k, _ in walk if k == 'para')
        n_delta = max(0, n_kind0 - 1)
        delta_resids = np.zeros((n_delta, 3), dtype=np.int64)
        para_resids  = np.zeros((n_kind1, 3), dtype=np.int64)

        def _decode_one(tag, k):
            if tag == 1:
                u = r.read_rice(k)
            else:
                u = r.read_exp_golomb(k)
            return (u >> 1) ^ -(u & 1)

        for d in range(3):
            for i in range(n_delta):
                delta_resids[i, d] = _decode_one(d_tags[d], d_ks[d])
        for d in range(3):
            for i in range(n_kind1):
                para_resids[i, d] = _decode_one(p_tags[d], p_ks[d])

        codes_local = np.zeros((n_local, 3), dtype=np.int64)
        prev = None
        first = True
        di = 0
        pi = 0
        for v, kind, refs in walk:
            if kind == 'none':
                if first:
                    codes_local[v] = anchor
                    first = False
                else:
                    codes_local[v] = prev + delta_resids[di]
                    di += 1
            else:
                a, b, c = refs
                pred = _irlp_pred(codes_local[a], codes_local[b], codes_local[c],
                              pred_n, pred_K)
                codes_local[v] = pred + para_resids[pi]
                pi += 1
            prev = codes_local[v]

        all_codes_local.append(codes_local)
        for (a, b, c) in local_tris:
            all_tris.append((v_base + a, v_base + b, v_base + c))
        v_base += n_local

    # Stack + dequantize.
    codes = np.concatenate(all_codes_local, axis=0)
    V_norm = np.zeros_like(codes, dtype=np.float64)
    for d in range(3):
        mx = (1 << int(g_bits[d])) - 1
        if mx == 0:
            V_norm[:, d] = float(g_min[d])
        else:
            V_norm[:, d] = float(g_min[d]) + codes[:, d].astype(np.float64) / mx * float(g_range[d])
    V = V_norm * scale + center
    T = np.array(all_tris, dtype=np.int64)
    return V.astype(np.float32), T

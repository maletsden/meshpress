"""ParaDelta v5: strip-emit-order traversal. No decoder-side greedy.

v5 differs from v4 only in:
  - version byte = 5 (vs 4)
  - predictor: fixed to LIN3 (no 5-tap, no apex search at decode)
  - header carries lin3 weights only (no predictor_mode byte; no fallback w)
  - interior residuals in strip-emit order (matches order decoder walks)

Everything else (boundary table, offset table, GTS v3 connectivity, per-axis
fixed/Rice/EG residual encoding) is unchanged.
"""
from __future__ import annotations

import numpy as np
from collections import deque

from utils.bit_codec import BitWriter, BitReader
from utils.residual_entropy import _zigzag
from encoder.paradelta_codec import (
    MAGIC, REUSE_BUF_SIZE,
    _idx_bits_for, _fit_linear3,
    _best_rice_k,
    _root_orient, _read_vert, _write_meshlet,
)


VERSION_V5 = 5


# =====================================================================
# Strip-order traversal (encoder + decoder share this view)
# =====================================================================

def _strip_traversal(ml_tris_local, strips, n_bnd):
    """Walk strips in emit order. For each NEW interior vert encountered,
    return (v_local, kind, refs):
      kind='para', refs=(a, b, c)   — 2 shared edge verts + opposite c from
                                      prev_tri. Always available within a
                                      strip after root tri.
      kind='none', refs=()          — root-tri interior verts (no prev_tri).
    """
    decoded = set(range(n_bnd))
    order = []
    for strip in strips:
        root_id = strip[0]
        next_id = strip[1] if len(strip) > 1 else None
        root = _root_orient(ml_tris_local, root_id, next_id)
        for v in root:
            if v >= n_bnd and v not in decoded:
                order.append((v, 'none', ()))
                decoded.add(v)
            else:
                decoded.add(v)
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
                a, b = prev_tri[1], prev_tri[2]
                c = prev_tri[0]
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            elif shared_fs == pair_second:
                a, b = prev_tri[0], prev_tri[2]
                c = prev_tri[1]
                new_prev = [prev_tri[0], prev_tri[2], new_v]
            else:
                raise RuntimeError("oldest-edge share in strip")
            if new_v >= n_bnd and new_v not in decoded:
                order.append((new_v, 'para', (a, b, c)))
                decoded.add(new_v)
            else:
                decoded.add(new_v)
            prev_tri = new_prev
    return order


# =====================================================================
# Encoder
# =====================================================================

def _interior_pass_strip(plan, vn, bnd_recon_norm, delta, w3):
    """Numba-jitted strip-order predictor pass."""
    n_bnd = plan["n_bnd"]
    n_int = plan["n_int"]
    if n_int == 0:
        return np.zeros((0, 3), dtype=np.int64), []

    from encoder.paradelta_v5_nb import interior_pass_strip_nb, flatten_strips

    local_to_global_arr = np.asarray(plan["local_to_global"], dtype=np.int64)
    ml_tris_local = np.ascontiguousarray(plan["ml_tris_local"], dtype=np.int64)
    strip_flat, strip_offsets = flatten_strips(plan["strips"])
    vn_f64 = np.ascontiguousarray(vn, dtype=np.float64)
    bnd_recon_f64 = np.ascontiguousarray(bnd_recon_norm, dtype=np.float64)
    delta_f32 = np.float32(delta)

    use_w3 = w3 is not None
    if use_w3:
        w0_f32 = np.float32(w3[0])
        w1_f32 = np.float32(w3[1])
        w2_f32 = np.float32(w3[2])
    else:
        w0_f32 = np.float32(0.0)
        w1_f32 = np.float32(0.0)
        w2_f32 = np.float32(0.0)

    collect = not use_w3
    codes_out = np.zeros((n_int, 3), dtype=np.int64)
    sample_a = np.zeros((n_int, 3), dtype=np.float64) if collect else \
        np.zeros((0, 3), dtype=np.float64)
    sample_b = np.zeros((n_int, 3), dtype=np.float64) if collect else \
        np.zeros((0, 3), dtype=np.float64)
    sample_c = np.zeros((n_int, 3), dtype=np.float64) if collect else \
        np.zeros((0, 3), dtype=np.float64)
    sample_true = np.zeros((n_int, 3), dtype=np.float64) if collect else \
        np.zeros((0, 3), dtype=np.float64)

    n_emit, n_samp = interior_pass_strip_nb(
        ml_tris_local, strip_flat, strip_offsets,
        n_bnd, n_int, local_to_global_arr,
        vn_f64, bnd_recon_f64,
        delta_f32, w0_f32, w1_f32, w2_f32,
        use_w3, collect,
        codes_out, sample_a, sample_b, sample_c, sample_true,
    )

    samples = []
    if collect:
        for i in range(n_samp):
            samples.append((sample_a[i].copy(),
                            sample_b[i].copy(),
                            sample_c[i].copy(),
                            sample_true[i].copy()))

    return codes_out, samples


def encode_from_prepared_v5(prep: dict, *, verbose: bool = False) -> bytes:
    """Encode v5 bitstream: strip-order traversal + LIN3 predictor."""
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

    # Pass 1: TG baseline to collect lin3 fitting samples
    all_samples = []
    for plan in plans:
        _, samples = _interior_pass_strip(
            plan, vn, bnd_recon_norm, delta, w3=None)
        all_samples.extend(samples)
    lin3_w = _fit_linear3(all_samples)
    if verbose:
        print(f"  v5 lin3 samples={len(all_samples)}  "
              f"w=({float(lin3_w[0]):.4f}, "
              f"{float(lin3_w[1]):.4f}, "
              f"{float(lin3_w[2]):.4f})")

    # Quantize-to-fp32 to match decoder roundtrip
    w3_decoder = lin3_w.astype(np.float32).astype(np.float64)

    # Pass 2 — emit per-meshlet bytes via Numba writer into a shared buffer.
    from encoder.paradelta_v5_nb import (
        write_meshlet_nb, flatten_strips,
    )
    from encoder.paradelta_codec import _idx_bits_for, _REUSE_BITS, REUSE_BUF_SIZE

    # Worst-case ~2 KB per meshlet. Safety margin 8x.
    buf_cap = max(1024, n_meshlets * 2048)
    big_buf = np.zeros(buf_cap, dtype=np.uint8)
    byte_pos = 0
    bit_in_byte = 0
    meshlet_offsets_rel = []
    max_abs_code = 0
    for plan in plans:
        codes, _ = _interior_pass_strip(
            plan, vn, bnd_recon_norm, delta, w3_decoder)
        if codes.shape[0] > 0:
            ma = int(np.abs(codes).max())
            if ma > max_abs_code:
                max_abs_code = ma
        meshlet_offsets_rel.append(byte_pos * 8 + bit_in_byte)

        n_bnd_m = plan["n_bnd"]
        n_int_m = plan["n_int"]
        n_local_m = n_bnd_m + n_int_m
        n_tris_m = plan["n_tris_m"]
        n_strips_m = plan["n_strips"]
        ml_tris_local = np.ascontiguousarray(plan["ml_tris_local"], dtype=np.int64)
        refs_arr = np.ascontiguousarray(plan["refs"], dtype=np.int64)
        strip_flat, strip_offsets = flatten_strips(plan["strips"])
        codes_arr = np.ascontiguousarray(codes, dtype=np.int64)
        idx_bits = _idx_bits_for(n_local_m)
        # Grow buffer if needed (rare).
        if byte_pos + 16384 > buf_cap:
            new_cap = buf_cap * 2
            new_buf = np.zeros(new_cap, dtype=np.uint8)
            new_buf[:byte_pos + 1] = big_buf[:byte_pos + 1]
            big_buf = new_buf
            buf_cap = new_cap
        byte_pos, bit_in_byte = write_meshlet_nb(
            big_buf, byte_pos, bit_in_byte,
            ml_tris_local, strip_flat, strip_offsets,
            n_bnd_m, n_int_m, n_tris_m, n_strips_m,
            refs_arr, codes_arr,
            idx_bits, _REUSE_BITS, REUSE_BUF_SIZE,
        )
    # Finalize: trim and add partial byte
    if bit_in_byte > 0:
        meshlet_bytes = bytes(big_buf[:byte_pos + 1])
    else:
        meshlet_bytes = bytes(big_buf[:byte_pos])
    # 0 = decoder uses int16 SMEM codes (fast path); 1 = int32 (needed
    # when any meshlet has |residual| > 32767, e.g. scan-noisy meshes).
    code_width_flag = 1 if max_abs_code > 32767 else 0
    if verbose:
        print(f"  v5 max |code| across mesh = {max_abs_code:,}  "
              f"-> code_width_flag={code_width_flag} "
              f"({'i32' if code_width_flag else 'i16'})")

    # Globals
    w = BitWriter()
    w.write_fixed(MAGIC, 32)
    w.write_fixed(VERSION_V5, 8)
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
    w.write_fixed(n_v, 32)
    w.write_fixed(n_t, 32)
    w.write_fixed(n_boundary, 32)
    w.write_fixed(n_meshlets, 32)
    for v in lin3_w:
        w.write_f32(float(v))

    # Boundary table (same layout as v4)
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

    # Pad + offset table
    pad = (-w.bit_pos()) & 7
    if pad:
        w.write_bits(0, pad)
    offset_table_start = w.bit_pos()
    offset_table_bits = n_meshlets * 32
    meshlet_region_start = offset_table_start + offset_table_bits
    assert meshlet_region_start % 8 == 0
    for rel in meshlet_offsets_rel:
        w.write_fixed(meshlet_region_start + rel, 32)

    return w.finalize() + meshlet_bytes


# =====================================================================
# Decoder
# =====================================================================

def decode_paradelta_v5(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode v5 bitstream. Walks strip + applies residual chain inline.
    No greedy_order, no edge_to_tris build."""
    r = BitReader(data)
    magic = r.read_fixed(32)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X}")
    version = r.read_fixed(8)
    if version != VERSION_V5:
        raise ValueError(f"expected v5, got v{version}")
    _code_width = r.read_fixed(8)  # decoder-side SMEM hint; CPU ignores
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
    lin3_w = np.array([r.read_f32() for _ in range(3)], dtype=np.float64)

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
    bnd_pos_norm = np.zeros((n_boundary, 3), dtype=np.float64)
    for d in range(3):
        mx = (1 << int(g_bits[d])) - 1
        if mx == 0:
            bnd_pos_norm[:, d] = float(g_min[d])
        else:
            bnd_pos_norm[:, d] = (
                float(g_min[d])
                + bnd_codes[:, d].astype(np.float64) / mx * float(g_range[d])
            )

    # Skip offset table (CPU decoder doesn't need random access)
    pad = (-r.bit_pos()) & 7
    if pad:
        r.read_bits(pad)
    for _ in range(n_meshlets):
        r.read_fixed(32)

    all_verts_norm: list[np.ndarray] = [
        np.asarray(p, dtype=np.float64).copy() for p in bnd_pos_norm
    ]
    all_tris: list[tuple[int, int, int]] = []

    for _ml in range(n_meshlets):
        n_bnd = r.read_fixed(16)
        n_int = r.read_fixed(16)
        n_tris_m = r.read_fixed(16)
        n_strips = r.read_fixed(16)
        n_local = n_bnd + n_int

        # Boundary refs
        ref_indices = []
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

        local_to_global_dec = list(ref_indices)
        first_interior_global = len(all_verts_norm)
        for k in range(n_int):
            local_to_global_dec.append(first_interior_global + k)
        for _ in range(n_int):
            all_verts_norm.append(np.zeros(3, dtype=np.float64))

        # Connectivity + record strip-emit order for interior verts
        idx_bits = _idx_bits_for(n_local)
        reuse_fifo: deque[int] = deque(maxlen=REUSE_BUF_SIZE)
        local_tris: list[tuple[int, int, int]] = []
        emit_order: list[tuple[int, str, tuple]] = []
        decoded_set = set(range(n_bnd))
        for _s in range(n_strips):
            strip_len = r.read_fixed(16)
            v0 = _read_vert(r, reuse_fifo, idx_bits)
            v1 = _read_vert(r, reuse_fifo, idx_bits)
            v2 = _read_vert(r, reuse_fifo, idx_bits)
            local_tris.append((v0, v1, v2))
            for v in (v0, v1, v2):
                if v >= n_bnd and v not in decoded_set:
                    emit_order.append((v, 'none', ()))
                decoded_set.add(v)
            prev_tri = [v0, v1, v2]
            for _ in range(strip_len - 1):
                edge_code = r.read_bits(1)
                new_v = _read_vert(r, reuse_fifo, idx_bits)
                if edge_code == 0:
                    a, b = prev_tri[1], prev_tri[2]
                    c = prev_tri[0]
                    new_prev = [prev_tri[1], prev_tri[2], new_v]
                else:
                    a, b = prev_tri[0], prev_tri[2]
                    c = prev_tri[1]
                    new_prev = [prev_tri[0], prev_tri[2], new_v]
                local_tris.append((a, b, new_v))
                if new_v >= n_bnd and new_v not in decoded_set:
                    emit_order.append((new_v, 'para', (a, b, c)))
                decoded_set.add(new_v)
                prev_tri = new_prev

        # Residual codes (per-axis, in strip-emit order)
        if n_int > 0:
            codes = np.zeros((n_int, 3), dtype=np.int64)
            for d in range(3):
                tag = r.read_fixed(8)
                if tag == 0:
                    mn_u = r.read_fixed(16)
                    mn = mn_u - 0x10000 if (mn_u & 0x8000) else mn_u
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

            # Apply codes in emit order
            recon: dict[int, np.ndarray] = {}
            for lid in range(n_bnd):
                recon[lid] = np.asarray(
                    all_verts_norm[local_to_global_dec[lid]],
                    dtype=np.float64).copy()
            fallback = (np.mean([recon[i] for i in range(n_bnd)], axis=0)
                        if n_bnd > 0 else np.zeros(3))
            for i, (v_local, kind, refs) in enumerate(emit_order):
                if kind == 'para':
                    a, b, c = refs
                    pred = (lin3_w[0] * recon[a]
                            + lin3_w[1] * recon[b]
                            + lin3_w[2] * recon[c])
                else:
                    pred = fallback.copy()
                rec = pred + codes[i].astype(np.float64) * delta
                recon[v_local] = rec
                all_verts_norm[local_to_global_dec[v_local]] = rec

        for (a, b, c) in local_tris:
            all_tris.append((
                local_to_global_dec[a],
                local_to_global_dec[b],
                local_to_global_dec[c],
            ))

    V = np.array(all_verts_norm, dtype=np.float64) * scale + center
    T = np.array(all_tris, dtype=np.int64)
    return V.astype(np.float32), T

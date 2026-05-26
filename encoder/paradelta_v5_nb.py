"""Numba-JIT inner kernels for ParaDelta v5 encoder.

Currently provides:
  - interior_pass_strip_nb : per-meshlet strip-order vertex predictor +
    residual emission. Replaces the Python loop in
    `encoder.paradelta_v5._interior_pass_strip` (cProfile top-1 hotspot,
    ~20 s/Monkey).

The kernel walks each strip, mirrors the encoder's f32 reconstruction
chain bit-for-bit with the CUDA decoder, and (optionally) collects the
LSQ samples needed for the global linear-3 weight fit.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def interior_pass_strip_nb(
    ml_tris_local,        # int64[n_tris_m, 3]
    strip_tris_flat,      # int64[total_strip_tris]
    strip_offsets,        # int64[n_strips + 1]
    n_bnd,                # int
    n_int,                # int
    local_to_global,      # int64[n_local]
    vn,                   # float64[N, 3]
    bnd_recon_norm,       # float64[N, 3]
    delta_f32,            # float32
    w0_f32, w1_f32, w2_f32,
    use_w3,               # bool
    collect_samples,      # bool
    codes_out,            # int64[n_int, 3]
    sample_a,             # float64[n_int, 3]
    sample_b,             # float64[n_int, 3]
    sample_c,             # float64[n_int, 3]
    sample_true,          # float64[n_int, 3]
):
    """Returns (n_emitted, n_samples). Sample arrays are filled only for
    para-kind interior verts (root-tri interior verts use fallback predictor
    and contribute no sample to the LSQ fit)."""
    n_local = n_bnd + n_int
    recon = np.zeros((n_local, 3), dtype=np.float32)
    for lid in range(n_bnd):
        gid = local_to_global[lid]
        recon[lid, 0] = np.float32(bnd_recon_norm[gid, 0])
        recon[lid, 1] = np.float32(bnd_recon_norm[gid, 1])
        recon[lid, 2] = np.float32(bnd_recon_norm[gid, 2])

    fb0 = np.float32(0.0)
    fb1 = np.float32(0.0)
    fb2 = np.float32(0.0)
    if n_bnd > 0:
        s0 = np.float32(0.0); s1 = np.float32(0.0); s2 = np.float32(0.0)
        for lid in range(n_bnd):
            s0 = s0 + recon[lid, 0]
            s1 = s1 + recon[lid, 1]
            s2 = s2 + recon[lid, 2]
        nb = np.float32(n_bnd)
        fb0 = s0 / nb; fb1 = s1 / nb; fb2 = s2 / nb

    decoded = np.zeros(n_local, dtype=np.bool_)
    for lid in range(n_bnd):
        decoded[lid] = True

    out_i = 0
    samp_i = 0

    n_strips = strip_offsets.shape[0] - 1

    for s in range(n_strips):
        s_off = strip_offsets[s]
        s_end = strip_offsets[s + 1]
        strip_len = s_end - s_off

        root_id = strip_tris_flat[s_off]
        next_id = -1
        if strip_len > 1:
            next_id = strip_tris_flat[s_off + 1]

        # _root_orient — (y, z) sorted by value to match Python's sorted()
        r0 = ml_tris_local[root_id, 0]
        r1 = ml_tris_local[root_id, 1]
        r2 = ml_tris_local[root_id, 2]
        if next_id >= 0:
            n0 = ml_tris_local[next_id, 0]
            n1 = ml_tris_local[next_id, 1]
            n2 = ml_tris_local[next_id, 2]
            in_next_0 = (r0 == n0) or (r0 == n1) or (r0 == n2)
            in_next_1 = (r1 == n0) or (r1 == n1) or (r1 == n2)
            in_next_2 = (r2 == n0) or (r2 == n1) or (r2 == n2)
            shared_n = (1 if in_next_0 else 0) + (1 if in_next_1 else 0) + (1 if in_next_2 else 0)
            if shared_n == 2:
                if not in_next_0:
                    third = r0; y_ = r1; z_ = r2
                elif not in_next_1:
                    third = r1; y_ = r0; z_ = r2
                else:
                    third = r2; y_ = r0; z_ = r1
                if y_ > z_:
                    tmp = y_; y_ = z_; z_ = tmp
                p0_, p1_, p2_ = third, y_, z_
            else:
                p0_, p1_, p2_ = r0, r1, r2
        else:
            p0_, p1_, p2_ = r0, r1, r2

        # Root verts
        for k in range(3):
            if k == 0:
                v = p0_
            elif k == 1:
                v = p1_
            else:
                v = p2_
            if v >= n_bnd and not decoded[v]:
                gid = local_to_global[v]
                true_x = np.float32(vn[gid, 0])
                true_y = np.float32(vn[gid, 1])
                true_z = np.float32(vn[gid, 2])
                pred_x = fb0; pred_y = fb1; pred_z = fb2
                cx = np.int64(np.round((true_x - pred_x) / delta_f32))
                cy = np.int64(np.round((true_y - pred_y) / delta_f32))
                cz = np.int64(np.round((true_z - pred_z) / delta_f32))
                codes_out[out_i, 0] = cx
                codes_out[out_i, 1] = cy
                codes_out[out_i, 2] = cz
                recon[v, 0] = pred_x + np.float32(cx) * delta_f32
                recon[v, 1] = pred_y + np.float32(cy) * delta_f32
                recon[v, 2] = pred_z + np.float32(cz) * delta_f32
                decoded[v] = True
                out_i += 1
            else:
                decoded[v] = True

        prev0, prev1, prev2 = p0_, p1_, p2_

        for ii in range(1, strip_len):
            li = strip_tris_flat[s_off + ii]
            t0 = ml_tris_local[li, 0]
            t1 = ml_tris_local[li, 1]
            t2 = ml_tris_local[li, 2]

            t0_in_prev = (t0 == prev0) or (t0 == prev1) or (t0 == prev2)
            t1_in_prev = (t1 == prev0) or (t1 == prev1) or (t1 == prev2)
            t2_in_prev = (t2 == prev0) or (t2 == prev1) or (t2 == prev2)
            if not t0_in_prev:
                new_v = t0
            elif not t1_in_prev:
                new_v = t1
            else:
                new_v = t2

            p0_in_t = (prev0 == t0) or (prev0 == t1) or (prev0 == t2)
            p1_in_t = (prev1 == t0) or (prev1 == t1) or (prev1 == t2)
            p2_in_t = (prev2 == t0) or (prev2 == t1) or (prev2 == t2)

            if p1_in_t and p2_in_t and not p0_in_t:
                a = prev1; b = prev2; c = prev0
                new_p0 = prev1; new_p1 = prev2; new_p2 = new_v
            elif p0_in_t and p2_in_t and not p1_in_t:
                a = prev0; b = prev2; c = prev1
                new_p0 = prev0; new_p1 = prev2; new_p2 = new_v
            else:
                raise RuntimeError("oldest-edge share in strip")

            if new_v >= n_bnd and not decoded[new_v]:
                gid = local_to_global[new_v]
                if use_w3:
                    pred_x = w0_f32 * recon[a, 0] + w1_f32 * recon[b, 0] + w2_f32 * recon[c, 0]
                    pred_y = w0_f32 * recon[a, 1] + w1_f32 * recon[b, 1] + w2_f32 * recon[c, 1]
                    pred_z = w0_f32 * recon[a, 2] + w1_f32 * recon[b, 2] + w2_f32 * recon[c, 2]
                else:
                    pred_x = recon[a, 0] + recon[b, 0] - recon[c, 0]
                    pred_y = recon[a, 1] + recon[b, 1] - recon[c, 1]
                    pred_z = recon[a, 2] + recon[b, 2] - recon[c, 2]
                true_x = np.float32(vn[gid, 0])
                true_y = np.float32(vn[gid, 1])
                true_z = np.float32(vn[gid, 2])
                cx = np.int64(np.round((true_x - pred_x) / delta_f32))
                cy = np.int64(np.round((true_y - pred_y) / delta_f32))
                cz = np.int64(np.round((true_z - pred_z) / delta_f32))
                codes_out[out_i, 0] = cx
                codes_out[out_i, 1] = cy
                codes_out[out_i, 2] = cz
                recon[new_v, 0] = pred_x + np.float32(cx) * delta_f32
                recon[new_v, 1] = pred_y + np.float32(cy) * delta_f32
                recon[new_v, 2] = pred_z + np.float32(cz) * delta_f32

                if collect_samples:
                    sample_a[samp_i, 0] = np.float64(recon[a, 0])
                    sample_a[samp_i, 1] = np.float64(recon[a, 1])
                    sample_a[samp_i, 2] = np.float64(recon[a, 2])
                    sample_b[samp_i, 0] = np.float64(recon[b, 0])
                    sample_b[samp_i, 1] = np.float64(recon[b, 1])
                    sample_b[samp_i, 2] = np.float64(recon[b, 2])
                    sample_c[samp_i, 0] = np.float64(recon[c, 0])
                    sample_c[samp_i, 1] = np.float64(recon[c, 1])
                    sample_c[samp_i, 2] = np.float64(recon[c, 2])
                    sample_true[samp_i, 0] = vn[gid, 0]
                    sample_true[samp_i, 1] = vn[gid, 1]
                    sample_true[samp_i, 2] = vn[gid, 2]
                    samp_i += 1

                decoded[new_v] = True
                out_i += 1
            else:
                decoded[new_v] = True

            prev0, prev1, prev2 = new_p0, new_p1, new_p2

    return out_i, samp_i


@njit(cache=True, inline='always')
def _write_bits_into(buf, byte_pos, bit_in_byte, value, n_bits):
    if n_bits <= 0:
        return byte_pos, bit_in_byte
    value &= (np.int64(1) << n_bits) - 1
    i = n_bits
    while i > 0:
        free = 8 - bit_in_byte
        take = free if free < i else i
        chunk = (value >> (i - take)) & ((np.int64(1) << take) - 1)
        buf[byte_pos] = np.uint8(
            buf[byte_pos] | (chunk << (free - take)))
        bit_in_byte += take
        i -= take
        if bit_in_byte == 8:
            byte_pos += 1
            bit_in_byte = 0
    return byte_pos, bit_in_byte


@njit(cache=True, inline='always')
def _write_rice_nb(buf, byte_pos, bit_in_byte, u, k):
    q = u >> k
    zb = q
    while zb >= 8:
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(0), 8)
        zb -= 8
    if zb > 0:
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(0), zb)
    byte_pos, bit_in_byte = _write_bits_into(
        buf, byte_pos, bit_in_byte, np.int64(1), 1)
    if k > 0:
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, u & ((np.int64(1) << k) - 1), k)
    return byte_pos, bit_in_byte


@njit(cache=True, inline='always')
def _emit_vert_nb(buf, byte_pos, bit_in_byte, v, reuse_fifo, fifo_len,
                  reuse_buf_size, idx_bits, reuse_bits):
    """Emit one local-ID via FIFO MTF. Returns (byte_pos, bit_in_byte, new_fifo_len)."""
    hit_idx = -1
    for i in range(fifo_len):
        if reuse_fifo[i] == v:
            hit_idx = i
            break
    if hit_idx >= 0:
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(0), 1)
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(hit_idx), reuse_bits)
        # remove from current position
        for j in range(hit_idx, fifo_len - 1):
            reuse_fifo[j] = reuse_fifo[j + 1]
        fifo_len -= 1
    else:
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(1), 1)
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(v), idx_bits)
    # append to end (most-recent)
    if fifo_len == reuse_buf_size:
        # drop oldest
        for j in range(0, fifo_len - 1):
            reuse_fifo[j] = reuse_fifo[j + 1]
        reuse_fifo[fifo_len - 1] = v
    else:
        reuse_fifo[fifo_len] = v
        fifo_len += 1
    return byte_pos, bit_in_byte, fifo_len


@njit(cache=True, inline='always')
def _best_rice_k_nb(u_arr, k_max):
    n = u_arr.shape[0]
    best_bits = np.int64(0x7FFFFFFFFFFFFFFF)
    best_k = 0
    for k in range(k_max + 1):
        # bits = sum(u >> k) + n * (1 + k)
        s = np.int64(0)
        for i in range(n):
            s += u_arr[i] >> k
        b = s + n * (1 + k)
        if b < best_bits:
            best_bits = b
            best_k = k
    return best_k, best_bits


@njit(cache=True, inline='always')
def _best_eg_k_nb(u_arr, k_max):
    n = u_arr.shape[0]
    best_bits = np.int64(0x7FFFFFFFFFFFFFFF)
    best_k = 0
    for k in range(k_max + 1):
        s = np.int64(0)
        for i in range(n):
            shifted = u_arr[i] >> k
            x = shifted + 1
            lb = 0
            while x > 1:
                x >>= 1
                lb += 1
            s += 2 * lb + 1 + k
        if s < best_bits:
            best_bits = s
            best_k = k
    return best_k, best_bits


@njit(cache=True)
def write_meshlet_nb(
    buf, byte_pos, bit_in_byte,
    ml_tris_local,
    strip_tris_flat, strip_offsets,
    n_bnd, n_int, n_tris_m, n_strips,
    refs,                # int64[n_bnd] global boundary table refs
    codes_traversal,     # int64[n_int, 3] residual codes in strip order
    idx_bits,
    reuse_bits, reuse_buf_size,
):
    """Write one meshlet's full bitstream into shared buf. Returns new
    (byte_pos, bit_in_byte). Order matches the Python reference exactly."""
    # Header: 4× 16-bit
    byte_pos, bit_in_byte = _write_bits_into(buf, byte_pos, bit_in_byte, np.int64(n_bnd), 16)
    byte_pos, bit_in_byte = _write_bits_into(buf, byte_pos, bit_in_byte, np.int64(n_int), 16)
    byte_pos, bit_in_byte = _write_bits_into(buf, byte_pos, bit_in_byte, np.int64(n_tris_m), 16)
    byte_pos, bit_in_byte = _write_bits_into(buf, byte_pos, bit_in_byte, np.int64(n_strips), 16)

    # Boundary refs: first 32-bit, then delta-Rice
    if n_bnd > 0:
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(refs[0]), 32)
        if n_bnd > 1:
            n_diffs = n_bnd - 1
            diffs = np.empty(n_diffs, dtype=np.int64)
            for i in range(n_diffs):
                diffs[i] = refs[i + 1] - refs[i] - 1
            k_b, _ = _best_rice_k_nb(diffs, 11)
            byte_pos, bit_in_byte = _write_bits_into(
                buf, byte_pos, bit_in_byte, np.int64(k_b), 8)
            for i in range(n_diffs):
                byte_pos, bit_in_byte = _write_rice_nb(
                    buf, byte_pos, bit_in_byte, diffs[i], k_b)

    # FIFO state
    reuse_fifo = np.empty(reuse_buf_size, dtype=np.int64)
    fifo_len = 0

    # Walk strips
    for s in range(n_strips):
        s_off = strip_offsets[s]
        s_end = strip_offsets[s + 1]
        strip_len = s_end - s_off
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(strip_len), 16)

        root_id = strip_tris_flat[s_off]
        next_id = -1
        if strip_len > 1:
            next_id = strip_tris_flat[s_off + 1]

        r0 = ml_tris_local[root_id, 0]
        r1 = ml_tris_local[root_id, 1]
        r2 = ml_tris_local[root_id, 2]
        if next_id >= 0:
            n0 = ml_tris_local[next_id, 0]
            n1 = ml_tris_local[next_id, 1]
            n2 = ml_tris_local[next_id, 2]
            in_n_0 = (r0 == n0) or (r0 == n1) or (r0 == n2)
            in_n_1 = (r1 == n0) or (r1 == n1) or (r1 == n2)
            in_n_2 = (r2 == n0) or (r2 == n1) or (r2 == n2)
            shared_n = (1 if in_n_0 else 0) + (1 if in_n_1 else 0) + (1 if in_n_2 else 0)
            if shared_n == 2:
                if not in_n_0:
                    third = r0; y_ = r1; z_ = r2
                elif not in_n_1:
                    third = r1; y_ = r0; z_ = r2
                else:
                    third = r2; y_ = r0; z_ = r1
                if y_ > z_:
                    tmp = y_; y_ = z_; z_ = tmp
                p0_, p1_, p2_ = third, y_, z_
            else:
                p0_, p1_, p2_ = r0, r1, r2
        else:
            p0_, p1_, p2_ = r0, r1, r2

        # Emit 3 root verts
        byte_pos, bit_in_byte, fifo_len = _emit_vert_nb(
            buf, byte_pos, bit_in_byte, p0_, reuse_fifo, fifo_len,
            reuse_buf_size, idx_bits, reuse_bits)
        byte_pos, bit_in_byte, fifo_len = _emit_vert_nb(
            buf, byte_pos, bit_in_byte, p1_, reuse_fifo, fifo_len,
            reuse_buf_size, idx_bits, reuse_bits)
        byte_pos, bit_in_byte, fifo_len = _emit_vert_nb(
            buf, byte_pos, bit_in_byte, p2_, reuse_fifo, fifo_len,
            reuse_buf_size, idx_bits, reuse_bits)
        prev0, prev1, prev2 = p0_, p1_, p2_

        for ii in range(1, strip_len):
            li = strip_tris_flat[s_off + ii]
            t0 = ml_tris_local[li, 0]
            t1 = ml_tris_local[li, 1]
            t2 = ml_tris_local[li, 2]
            t0_in_prev = (t0 == prev0) or (t0 == prev1) or (t0 == prev2)
            t1_in_prev = (t1 == prev0) or (t1 == prev1) or (t1 == prev2)
            t2_in_prev = (t2 == prev0) or (t2 == prev1) or (t2 == prev2)
            if not t0_in_prev:
                new_v = t0
            elif not t1_in_prev:
                new_v = t1
            else:
                new_v = t2
            p0_in_t = (prev0 == t0) or (prev0 == t1) or (prev0 == t2)
            p1_in_t = (prev1 == t0) or (prev1 == t1) or (prev1 == t2)
            p2_in_t = (prev2 == t0) or (prev2 == t1) or (prev2 == t2)
            if p1_in_t and p2_in_t and not p0_in_t:
                edge_code = 0
                new_p0 = prev1; new_p1 = prev2; new_p2 = new_v
            elif p0_in_t and p2_in_t and not p1_in_t:
                edge_code = 1
                new_p0 = prev0; new_p1 = prev2; new_p2 = new_v
            else:
                raise RuntimeError("oldest-edge share")
            byte_pos, bit_in_byte = _write_bits_into(
                buf, byte_pos, bit_in_byte, np.int64(edge_code), 1)
            byte_pos, bit_in_byte, fifo_len = _emit_vert_nb(
                buf, byte_pos, bit_in_byte, new_v, reuse_fifo, fifo_len,
                reuse_buf_size, idx_bits, reuse_bits)
            prev0, prev1, prev2 = new_p0, new_p1, new_p2

    # Residual axes: pick best of fixed/Rice/EG per axis
    if n_int > 0:
        for d in range(3):
            # arr = codes_traversal[:, d]
            mn = codes_traversal[0, d]
            mx = codes_traversal[0, d]
            for i in range(1, n_int):
                v = codes_traversal[i, d]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
            rng = mx - mn
            if rng > 0:
                # ceil(log2(rng+2))
                x = rng + 1
                lb = 0
                while x > 0:
                    x >>= 1
                    lb += 1
                fixed_bw = lb if lb > 0 else 1
            else:
                fixed_bw = 1
            # zigzag
            u_arr = np.empty(n_int, dtype=np.int64)
            for i in range(n_int):
                v = codes_traversal[i, d]
                u_arr[i] = (v << 1) ^ (v >> 63)

            fixed_total = np.int64(8 + 16 + 8 + n_int * fixed_bw)
            if mn < -32768 or mn > 32767:
                fixed_total = np.int64(0x7FFFFFFFFFFFFFFF)
            rice_k, rice_body = _best_rice_k_nb(u_arr, 11)
            rice_total = np.int64(8 + 8) + rice_body
            eg_k, eg_body = _best_eg_k_nb(u_arr, 7)
            eg_total = np.int64(8 + 8) + eg_body

            tag = 0
            best_total = fixed_total
            if rice_total < best_total:
                best_total = rice_total
                tag = 1
            if eg_total < best_total:
                best_total = eg_total
                tag = 2

            if tag == 0:
                byte_pos, bit_in_byte = _write_bits_into(
                    buf, byte_pos, bit_in_byte, np.int64(0), 8)
                byte_pos, bit_in_byte = _write_bits_into(
                    buf, byte_pos, bit_in_byte, np.int64(mn) & 0xFFFF, 16)
                byte_pos, bit_in_byte = _write_bits_into(
                    buf, byte_pos, bit_in_byte, np.int64(fixed_bw), 8)
                for i in range(n_int):
                    byte_pos, bit_in_byte = _write_bits_into(
                        buf, byte_pos, bit_in_byte,
                        np.int64(codes_traversal[i, d] - mn), fixed_bw)
            elif tag == 1:
                byte_pos, bit_in_byte = _write_bits_into(
                    buf, byte_pos, bit_in_byte, np.int64(1), 8)
                byte_pos, bit_in_byte = _write_bits_into(
                    buf, byte_pos, bit_in_byte, np.int64(rice_k), 8)
                for i in range(n_int):
                    byte_pos, bit_in_byte = _write_rice_nb(
                        buf, byte_pos, bit_in_byte, u_arr[i], rice_k)
            else:
                byte_pos, bit_in_byte = _write_bits_into(
                    buf, byte_pos, bit_in_byte, np.int64(2), 8)
                byte_pos, bit_in_byte = _write_bits_into(
                    buf, byte_pos, bit_in_byte, np.int64(eg_k), 8)
                for i in range(n_int):
                    u = u_arr[i]
                    shifted = u >> eg_k
                    x = shifted + 1
                    lb = 0
                    while x > 1:
                        x >>= 1
                        lb += 1
                    if lb > 0:
                        byte_pos, bit_in_byte = _write_bits_into(
                            buf, byte_pos, bit_in_byte, np.int64(0), lb)
                    byte_pos, bit_in_byte = _write_bits_into(
                        buf, byte_pos, bit_in_byte, np.int64(1), 1)
                    if lb > 0:
                        offset = shifted - ((np.int64(1) << lb) - 1)
                        byte_pos, bit_in_byte = _write_bits_into(
                            buf, byte_pos, bit_in_byte, offset, lb)
                    if eg_k > 0:
                        byte_pos, bit_in_byte = _write_bits_into(
                            buf, byte_pos, bit_in_byte,
                            u & ((np.int64(1) << eg_k) - 1), eg_k)

    return byte_pos, bit_in_byte


def flatten_strips(strips):
    """Convert list-of-lists strips to (flat int64, offsets int64)."""
    if len(strips) == 0:
        return (np.zeros(0, dtype=np.int64),
                np.zeros(1, dtype=np.int64))
    offsets = np.zeros(len(strips) + 1, dtype=np.int64)
    total = 0
    for i, s in enumerate(strips):
        total += len(s)
        offsets[i + 1] = total
    flat = np.empty(total, dtype=np.int64)
    j = 0
    for s in strips:
        for v in s:
            flat[j] = int(v)
            j += 1
    return flat, offsets
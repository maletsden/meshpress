"""Numba batched bit-writer kernels.

Encode whole numpy arrays of fixed-width / Rice / Exp-Golomb codes in a
single JIT call instead of per-value Python BitWriter overhead.

Each kernel writes into a pre-allocated uint8 buffer starting from a given
(byte_pos, bit_in_byte) cursor, in MSB-first order — same convention as
`utils.bit_codec.BitWriter`.

Caller must size `out_bytes` large enough:
  fixed:    n * n_bits
  rice:     n * (q_max + 1 + k) — q_max bounded by value >> k, so callers
            pre-scan or oversize conservatively (e.g. 64 bits/value).
  EG:       similar with log2 bound

Returns the updated `(byte_pos, bit_in_byte)` cursor as a 2-element int64 array.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def _write_bits_into(buf, byte_pos, bit_in_byte, value, n_bits):
    """MSB-first append of `value` (n_bits) into `buf` starting at
    (byte_pos, bit_in_byte). Returns new (byte_pos, bit_in_byte)."""
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


@njit(cache=True)
def encode_fixed_array(values, n_bits, buf, byte_pos, bit_in_byte):
    n = values.shape[0]
    for i in range(n):
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(values[i]), n_bits)
    return byte_pos, bit_in_byte


@njit(cache=True)
def encode_rice_array(values, k, buf, byte_pos, bit_in_byte):
    n = values.shape[0]
    for i in range(n):
        u = np.int64(values[i])
        q = u >> k
        # Emit q zero bits (no-op since buffer pre-zeroed), then 1.
        # We need to ADVANCE the cursor by q bits — buf is pre-zeroed.
        zb = q
        while zb >= 8:
            # Skip 8 zero bits = one full byte advance
            byte_pos, bit_in_byte = _write_bits_into(
                buf, byte_pos, bit_in_byte, np.int64(0), 8)
            zb -= 8
        if zb > 0:
            byte_pos, bit_in_byte = _write_bits_into(
                buf, byte_pos, bit_in_byte, np.int64(0), zb)
        # Terminator
        byte_pos, bit_in_byte = _write_bits_into(
            buf, byte_pos, bit_in_byte, np.int64(1), 1)
        # Remainder k bits
        if k > 0:
            byte_pos, bit_in_byte = _write_bits_into(
                buf, byte_pos, bit_in_byte, u & ((np.int64(1) << k) - 1), k)
    return byte_pos, bit_in_byte


@njit(cache=True)
def encode_exp_golomb_array(values, k, buf, byte_pos, bit_in_byte):
    n = values.shape[0]
    for i in range(n):
        u = np.int64(values[i])
        shifted = u >> k
        # lb = floor(log2(shifted + 1))
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
        if k > 0:
            byte_pos, bit_in_byte = _write_bits_into(
                buf, byte_pos, bit_in_byte, u & ((np.int64(1) << k) - 1), k)
    return byte_pos, bit_in_byte
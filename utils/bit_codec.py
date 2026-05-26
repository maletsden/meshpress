"""Bit-level codec primitives for MeshletParaDelta byte stream.

BitWriter accumulates bits MSB-first into a bytearray. BitReader pulls
bits MSB-first. Both support fixed-width integers, Rice(k), Exp-Golomb(k),
zigzag-encoded signed ints, and IEEE-754 float32.

Layout convention: bits flushed left-to-right within each byte. First
written bit is bit 7 of byte 0.
"""

from __future__ import annotations

import struct


class BitWriter:
    __slots__ = ("_buf", "_cur", "_n_bits")

    def __init__(self):
        self._buf = bytearray()
        self._cur = 0
        self._n_bits = 0

    def write_bits(self, value: int, n_bits: int) -> None:
        if n_bits <= 0:
            return
        # Truncate value to n_bits
        value &= (1 << n_bits) - 1
        cur = self._cur
        nb = self._n_bits
        # Append bits MSB-first to cur, flushing whole bytes as we go.
        i = n_bits
        while i > 0:
            free = 8 - nb
            take = min(free, i)
            cur = (cur << take) | ((value >> (i - take)) & ((1 << take) - 1))
            nb += take
            i -= take
            if nb == 8:
                self._buf.append(cur)
                cur = 0
                nb = 0
        self._cur = cur
        self._n_bits = nb

    def write_fixed(self, value: int, n_bits: int) -> None:
        self.write_bits(value, n_bits)

    def write_zigzag(self, n: int, n_bits: int) -> None:
        u = (n << 1) ^ (n >> 63) if n >= 0 else ((-n) << 1) - 1
        # Equivalent to: u = (n << 1) ^ (n >> 31) for 32-bit; explicit branch is
        # cleaner for arbitrary-range ints.
        self.write_bits(u, n_bits)

    def write_rice(self, u: int, k: int) -> None:
        if u < 0:
            raise ValueError("Rice(k) expects non-negative input")
        q = u >> k
        # Unary prefix: q zeros then a terminating 1.
        # Common pathological case: q huge → emit q zeros one byte at a time.
        while q >= 8:
            self.write_bits(0, 8)
            q -= 8
        if q > 0:
            self.write_bits(0, q)
        self.write_bits(1, 1)
        if k > 0:
            self.write_bits(u & ((1 << k) - 1), k)

    def write_exp_golomb(self, u: int, k: int) -> None:
        if u < 0:
            raise ValueError("EG(k) expects non-negative input")
        shifted = u >> k
        # Find lb = floor(log2(shifted + 1))
        lb = 0
        x = shifted + 1
        while x > 1:
            x >>= 1
            lb += 1
        # Unary lb zeros then 1, then (shifted - (2^lb - 1)) on lb bits
        # plus the original low k bits.
        if lb > 0:
            self.write_bits(0, lb)
        self.write_bits(1, 1)
        if lb > 0:
            offset = shifted - ((1 << lb) - 1)
            self.write_bits(offset, lb)
        if k > 0:
            self.write_bits(u & ((1 << k) - 1), k)

    def write_f32(self, v: float) -> None:
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
        self.write_bits(bits, 32)

    # ---- Bulk array writers (numba-backed) ----
    # Each method packs an entire numpy array of values into the stream.
    # ~10-30x faster than per-element write_bits / write_rice on large n.

    def _bulk_emit(self, values, encoder_fn, *enc_args, max_bits_per_value):
        try:
            import numpy as np
            from utils.bit_codec_nb import (
                encode_fixed_array, encode_rice_array, encode_exp_golomb_array,
            )
        except ImportError:
            return False
        n = int(len(values))
        if n == 0:
            return True
        max_bits = int(max_bits_per_value * n) + 16
        cap_bytes = (max_bits + 7) // 8 + 8
        buf = np.zeros(cap_bytes, dtype=np.uint8)
        bp, bib = encoder_fn(np.asarray(values, dtype=np.int64),
                             *enc_args, buf, 0, int(self._n_bits))
        # Splice into self._buf:
        # If there were pending bits in self._cur, kernel started writing into
        # buf[0]'s lower bits (bit_in_byte = self._n_bits). The high bits of
        # buf[0] still need to OR with self._cur.
        if self._n_bits > 0:
            buf[0] |= (self._cur << (8 - self._n_bits)) & 0xFF
        # bp = full bytes written; bib = bits in tail byte (partial).
        # Append full bytes (buf[0..bp]) to _buf.
        if bp > 0:
            self._buf.extend(buf[:bp].tobytes())
        # New tail state
        if bib > 0:
            self._cur = int(buf[bp]) >> (8 - bib)
            self._n_bits = int(bib)
        else:
            self._cur = 0
            self._n_bits = 0
        return True

    def write_fixed_array(self, values, n_bits: int) -> None:
        from utils.bit_codec_nb import encode_fixed_array
        ok = self._bulk_emit(values, encode_fixed_array, int(n_bits),
                             max_bits_per_value=int(n_bits))
        if not ok:
            for v in values:
                self.write_bits(int(v), n_bits)

    def write_rice_array(self, values, k: int) -> None:
        # Upper bound per-value: floor(u/2^k) + 1 + k. For codes capped at 32 bits
        # of input, this is ≤ 65 bits.
        from utils.bit_codec_nb import encode_rice_array
        import numpy as np
        arr = np.asarray(values, dtype=np.int64)
        # Worst-case length per value: (max_val >> k) + 1 + k bits.
        if len(arr) > 0:
            worst = int(arr.max()) >> int(k)
        else:
            worst = 0
        per = worst + 1 + int(k)
        ok = self._bulk_emit(arr, encode_rice_array, int(k),
                             max_bits_per_value=per)
        if not ok:
            for v in values:
                self.write_rice(int(v), k)

    def write_exp_golomb_array(self, values, k: int) -> None:
        from utils.bit_codec_nb import encode_exp_golomb_array
        import numpy as np
        arr = np.asarray(values, dtype=np.int64)
        # Worst-case per value: ~2 * log2(value + 1) + k bits.
        if len(arr) > 0:
            mx = int(arr.max())
        else:
            mx = 0
        lb = 0
        x = (mx >> int(k)) + 1
        while x > 1:
            x >>= 1
            lb += 1
        per = 2 * lb + 1 + int(k)
        ok = self._bulk_emit(arr, encode_exp_golomb_array, int(k),
                             max_bits_per_value=per)
        if not ok:
            for v in values:
                self.write_exp_golomb(int(v), k)

    def bit_pos(self) -> int:
        return len(self._buf) * 8 + self._n_bits

    def finalize(self) -> bytes:
        if self._n_bits > 0:
            self._buf.append(self._cur << (8 - self._n_bits))
            self._cur = 0
            self._n_bits = 0
        return bytes(self._buf)


class BitReader:
    __slots__ = ("_buf", "_pos")

    def __init__(self, data: bytes):
        self._buf = data
        self._pos = 0

    def read_bits(self, n_bits: int) -> int:
        if n_bits <= 0:
            return 0
        v = 0
        i = n_bits
        pos = self._pos
        buf = self._buf
        while i > 0:
            byte_idx = pos >> 3
            bit_in_byte = pos & 7
            free = 8 - bit_in_byte
            take = min(free, i)
            byte = buf[byte_idx]
            chunk = (byte >> (free - take)) & ((1 << take) - 1)
            v = (v << take) | chunk
            pos += take
            i -= take
        self._pos = pos
        return v

    def read_fixed(self, n_bits: int) -> int:
        return self.read_bits(n_bits)

    def read_zigzag(self, n_bits: int) -> int:
        u = self.read_bits(n_bits)
        return (u >> 1) ^ -(u & 1)

    def read_rice(self, k: int) -> int:
        # Count leading zeros then read terminator + k low bits.
        q = 0
        while True:
            b = self.read_bits(1)
            if b == 1:
                break
            q += 1
        low = self.read_bits(k) if k > 0 else 0
        return (q << k) | low

    def read_exp_golomb(self, k: int) -> int:
        lb = 0
        while True:
            b = self.read_bits(1)
            if b == 1:
                break
            lb += 1
        if lb > 0:
            offset = self.read_bits(lb)
            shifted = (1 << lb) - 1 + offset
        else:
            shifted = 0
        low = self.read_bits(k) if k > 0 else 0
        return (shifted << k) | low

    def read_f32(self) -> float:
        bits = self.read_bits(32)
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    def bit_pos(self) -> int:
        return self._pos


# Inline self-test on import: round-trip a small mixed payload to catch
# regressions early.
def _self_test():
    w = BitWriter()
    w.write_fixed(0xABCD, 16)
    w.write_rice(0, 3)
    w.write_rice(1, 0)
    w.write_rice(42, 4)
    w.write_exp_golomb(0, 0)
    w.write_exp_golomb(7, 2)
    w.write_zigzag(-12345, 16)
    w.write_zigzag(12345, 16)
    w.write_f32(3.14159)
    data = w.finalize()

    r = BitReader(data)
    assert r.read_fixed(16) == 0xABCD
    assert r.read_rice(3) == 0
    assert r.read_rice(0) == 1
    assert r.read_rice(4) == 42
    assert r.read_exp_golomb(0) == 0
    assert r.read_exp_golomb(2) == 7
    assert r.read_zigzag(16) == -12345
    assert r.read_zigzag(16) == 12345
    f = r.read_f32()
    assert abs(f - 3.14159) < 1e-5, f


_self_test()
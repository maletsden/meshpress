"""GPU bitstream parser for ParaDelta v4 (Phase 2a).

Replaces the Python `decode_paradelta_to_struct` bit-level parse (≈3 s on
Monkey) with a CUDA kernel.

Strategy:
  - CPU pre-parses the GLOBAL header (cheap; one-time).
  - GPU kernel 1 (`sizes_kernel`) reads the first 4×u16 of each meshlet
    from its absolute bit offset (offset table from v4) → n_bnd, n_int,
    n_tris, n_strips arrays.
  - CPU prefix-sums to get ml_l2g_off, ml_tris_off, ml_codes_off.
  - GPU kernel 2 (`full_parse_kernel`) per-meshlet lane-0 sequential
    parse: boundary refs (delta-Rice), strip tokens (FIFO reuse),
    interior residual codes (3 per-axis: fixed/Rice/EG). Writes to flat
    ml_l2g, ml_tris, ml_codes.

Phase 2a leaves the traversal-order recompute (`_greedy_order`) on the
host. That stage is still the ~4 s bottleneck on Monkey and is the
target for Phase 2b.
"""
from __future__ import annotations

import math
import struct

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

from utils.bit_codec import BitReader
from encoder.paradelta_codec import (
    MAGIC, VERSION, PREDICTOR_PLAIN, PREDICTOR_LIN5,
)


# =====================================================================
# CUDA source: device BitReader + Rice/EG + FIFO + parser
# =====================================================================
_CUDA_SRC = r"""
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef long long          int64_t;
typedef int                int32_t;

// Read up to 32 bits MSB-first from absolute bit-position *bp into the
// uint8_t stream `buf`. Advances *bp. Loads 5 bytes to cover any
// alignment for up to 32-bit reads.
__device__ __forceinline__ uint32_t br_read_bits(
    const uint8_t* __restrict__ buf, uint64_t* bp, int n)
{
    uint64_t pos = *bp;
    *bp = pos + n;
    int64_t byte_idx = (int64_t)(pos >> 3);
    int bit_in_byte = (int)(pos & 7ULL);
    // Load 5 bytes into 40-bit register
    uint64_t w =
        ((uint64_t)buf[byte_idx + 0] << 32) |
        ((uint64_t)buf[byte_idx + 1] << 24) |
        ((uint64_t)buf[byte_idx + 2] << 16) |
        ((uint64_t)buf[byte_idx + 3] <<  8) |
        ((uint64_t)buf[byte_idx + 4] <<  0);
    int shift = 40 - bit_in_byte - n;
    uint64_t mask = (n >= 32) ? 0xFFFFFFFFULL : ((1ULL << n) - 1ULL);
    return (uint32_t)((w >> shift) & mask);
}

__device__ __forceinline__ uint32_t br_read_rice(
    const uint8_t* __restrict__ buf, uint64_t* bp, int k)
{
    uint32_t q = 0;
    while (br_read_bits(buf, bp, 1) == 0) ++q;
    uint32_t low = (k > 0) ? br_read_bits(buf, bp, k) : 0;
    return (q << k) | low;
}

__device__ __forceinline__ uint32_t br_read_exp_golomb(
    const uint8_t* __restrict__ buf, uint64_t* bp, int k)
{
    uint32_t lb = 0;
    while (br_read_bits(buf, bp, 1) == 0) ++lb;
    uint32_t shifted = 0;
    if (lb > 0) {
        uint32_t off = br_read_bits(buf, bp, lb);
        shifted = (1u << lb) - 1u + off;
    }
    uint32_t low = (k > 0) ? br_read_bits(buf, bp, k) : 0;
    return (shifted << k) | low;
}

// Convert unsigned zigzag → signed
__device__ __forceinline__ int32_t zz_to_signed(uint32_t u) {
    return (int32_t)((u >> 1) ^ -(int32_t)(u & 1));
}

// Round up log2(n+1)
__device__ __forceinline__ int idx_bits_for(int n_local) {
    int x = n_local + 1;
    int bits = 0;
    while ((1 << bits) < x) ++bits;
    return bits < 1 ? 1 : bits;
}

#define REUSE_BUF_SIZE 16
#define REUSE_BITS 5    // ceil(log2(16+1))

// LRU FIFO: insert v, removing prior position if present.
__device__ __forceinline__ void fifo_touch(
    int* fifo, int* fifo_n, int v)
{
    int n = *fifo_n;
    int found = -1;
    #pragma unroll
    for (int i = 0; i < REUSE_BUF_SIZE; ++i) {
        if (i < n && fifo[i] == v) { found = i; }
    }
    if (found >= 0) {
        for (int i = found; i < n - 1; ++i) fifo[i] = fifo[i+1];
        --n;
    }
    if (n < REUSE_BUF_SIZE) {
        fifo[n++] = v;
    } else {
        for (int i = 0; i < REUSE_BUF_SIZE - 1; ++i) fifo[i] = fifo[i+1];
        fifo[REUSE_BUF_SIZE - 1] = v;
    }
    *fifo_n = n;
}

__device__ __forceinline__ int read_vert(
    const uint8_t* buf, uint64_t* bp,
    int* fifo, int* fifo_n, int idx_bits)
{
    uint32_t flag = br_read_bits(buf, bp, 1);
    int v;
    if (flag == 0) {
        int fi = (int)br_read_bits(buf, bp, REUSE_BITS);
        v = fifo[fi];
    } else {
        v = (int)br_read_bits(buf, bp, idx_bits);
    }
    fifo_touch(fifo, fifo_n, v);
    return v;
}


// ============================================================
// Pass 1: read 4×u16 header at each meshlet's absolute offset.
// Outputs:  n_bnd, n_int, n_tris, n_strips  (n_meshlets each)
// ============================================================
extern "C" __global__ void paradelta_parse_sizes(
    const uint8_t* __restrict__ buf,
    const uint64_t* __restrict__ ml_off_bits,
    const int n_meshlets,
    int* __restrict__ out_n_bnd,
    int* __restrict__ out_n_int,
    int* __restrict__ out_n_tris,
    int* __restrict__ out_n_strips)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_meshlets) return;
    uint64_t bp = ml_off_bits[m];
    out_n_bnd[m]    = (int)br_read_bits(buf, &bp, 16);
    out_n_int[m]    = (int)br_read_bits(buf, &bp, 16);
    out_n_tris[m]   = (int)br_read_bits(buf, &bp, 16);
    out_n_strips[m] = (int)br_read_bits(buf, &bp, 16);
}


// ============================================================
// Pass 2: full per-meshlet parse on lane 0.
//   - boundary refs (delta-Rice prefix sum) → l2g[0..n_bnd]
//   - strips (FIFO reuse) → local_tris[0..n_tris]
//   - 3 per-axis residual streams → codes[0..n_int][3]
//
// Other lanes idle; output to flat arrays via offsets.
// ============================================================
extern "C" __global__ void paradelta_parse_full(
    const uint8_t* __restrict__ buf,
    const uint64_t* __restrict__ ml_off_bits,
    const int* __restrict__ ml_n_bnd,
    const int* __restrict__ ml_n_int,
    const int* __restrict__ ml_n_tris,
    const int* __restrict__ ml_n_strips,
    const int* __restrict__ ml_l2g_off,
    const int* __restrict__ ml_tris_off,
    const int* __restrict__ ml_codes_off,
    const int interior_global_base,   // = n_boundary
    const int* __restrict__ interior_cursor,  // prefix sum of n_int (per meshlet start)
    int* __restrict__ out_l2g,
    int* __restrict__ out_tris,
    int* __restrict__ out_codes)
{
    if (threadIdx.x != 0) return;  // lane 0 only
    const int m = blockIdx.x;

    const int n_bnd     = ml_n_bnd[m];
    const int n_int     = ml_n_int[m];
    const int n_tris    = ml_n_tris[m];
    const int n_strips  = ml_n_strips[m];
    const int n_local   = n_bnd + n_int;

    const int l2g_off   = ml_l2g_off[m];
    const int tris_off  = ml_tris_off[m];
    const int codes_off = ml_codes_off[m];
    const int int_base  = interior_global_base + interior_cursor[m];

    // Skip 4×16-bit header
    uint64_t bp = ml_off_bits[m] + 64ULL;

    // --- Boundary refs: delta-Rice prefix sum ---
    if (n_bnd > 0) {
        uint32_t first = br_read_bits(buf, &bp, 32);
        out_l2g[l2g_off + 0] = (int)first;
        if (n_bnd > 1) {
            int k = (int)br_read_bits(buf, &bp, 8);
            int prev = (int)first;
            for (int i = 1; i < n_bnd; ++i) {
                uint32_t u = br_read_rice(buf, &bp, k);
                prev = prev + (int)u + 1;
                out_l2g[l2g_off + i] = prev;
            }
        }
    }
    // Reserve interior global IDs
    for (int k = 0; k < n_int; ++k) {
        out_l2g[l2g_off + n_bnd + k] = int_base + k;
    }

    // --- Strips with FIFO reuse ---
    int idx_bits = idx_bits_for(n_local);
    int fifo[REUSE_BUF_SIZE];
    int fifo_n = 0;
    int tri_cursor = 0;
    int prev_tri_0 = 0, prev_tri_1 = 0, prev_tri_2 = 0;
    for (int s = 0; s < n_strips; ++s) {
        int strip_len = (int)br_read_bits(buf, &bp, 16);
        int v0 = read_vert(buf, &bp, fifo, &fifo_n, idx_bits);
        int v1 = read_vert(buf, &bp, fifo, &fifo_n, idx_bits);
        int v2 = read_vert(buf, &bp, fifo, &fifo_n, idx_bits);
        int rb = (tris_off + tri_cursor) * 3;
        out_tris[rb + 0] = v0;
        out_tris[rb + 1] = v1;
        out_tris[rb + 2] = v2;
        ++tri_cursor;
        prev_tri_0 = v0; prev_tri_1 = v1; prev_tri_2 = v2;
        for (int t = 1; t < strip_len; ++t) {
            int edge_code = (int)br_read_bits(buf, &bp, 1);
            int new_v = read_vert(buf, &bp, fifo, &fifo_n, idx_bits);
            int s1, s2;
            int np0, np1, np2;
            if (edge_code == 0) {
                s1 = prev_tri_1; s2 = prev_tri_2;
                np0 = prev_tri_1; np1 = prev_tri_2; np2 = new_v;
            } else {
                s1 = prev_tri_0; s2 = prev_tri_2;
                np0 = prev_tri_0; np1 = prev_tri_2; np2 = new_v;
            }
            int wb = (tris_off + tri_cursor) * 3;
            out_tris[wb + 0] = s1;
            out_tris[wb + 1] = s2;
            out_tris[wb + 2] = new_v;
            ++tri_cursor;
            prev_tri_0 = np0; prev_tri_1 = np1; prev_tri_2 = np2;
        }
    }

    // --- Interior residual codes (3 per-axis streams) ---
    if (n_int > 0) {
        for (int d = 0; d < 3; ++d) {
            int tag = (int)br_read_bits(buf, &bp, 8);
            if (tag == 0) {
                uint32_t mn_u = br_read_bits(buf, &bp, 16);
                int mn = (mn_u & 0x8000u) ? ((int)mn_u - 0x10000) : (int)mn_u;
                int bw = (int)br_read_bits(buf, &bp, 8);
                for (int i = 0; i < n_int; ++i) {
                    int v = mn + (int)br_read_bits(buf, &bp, bw);
                    out_codes[(codes_off + i) * 3 + d] = v;
                }
            } else if (tag == 1) {
                int k = (int)br_read_bits(buf, &bp, 8);
                for (int i = 0; i < n_int; ++i) {
                    uint32_t u = br_read_rice(buf, &bp, k);
                    out_codes[(codes_off + i) * 3 + d] = zz_to_signed(u);
                }
            } else {  // EG
                int k = (int)br_read_bits(buf, &bp, 8);
                for (int i = 0; i < n_int; ++i) {
                    uint32_t u = br_read_exp_golomb(buf, &bp, k);
                    out_codes[(codes_off + i) * 3 + d] = zz_to_signed(u);
                }
            }
        }
    }
}
"""


def parse_globals(data: bytes) -> dict:
    """Parse only the global header + boundary positions on CPU.

    Returns dict identical-keyed to `decode_paradelta_to_struct` for the
    globals + boundary block, plus the byte-aligned `meshlet_region_bit`
    and per-meshlet absolute bit-offsets (`ml_off_bits`).

    Per-meshlet count arrays and connectivity/codes are filled by the
    GPU kernels.
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
        lin5_w3 = np.array([r.read_f32() for _ in range(3)],
                           dtype=np.float32)
        lin5_w5 = np.array([r.read_f32() for _ in range(5)],
                           dtype=np.float32)
    elif predictor_mode != PREDICTOR_PLAIN:
        raise ValueError(
            f"GPU parser supports only PLAIN/LIN5 (got {predictor_mode})")

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
                + bnd_codes[:, d].astype(np.float64) / mx
                * float(g_range[d])
            )

    # Skip pad + offset table; capture absolute bit offsets per meshlet.
    pad = (-r.bit_pos()) & 7
    if pad:
        r.read_bits(pad)
    ml_off_bits = np.zeros(n_meshlets, dtype=np.uint64)
    for m in range(n_meshlets):
        ml_off_bits[m] = np.uint64(r.read_fixed(32))

    return {
        "center": center, "scale": np.float32(scale),
        "delta": np.float32(delta),
        "g_min": g_min, "g_range": g_range, "g_bits": g_bits,
        "predictor_mode": np.int32(predictor_mode),
        "lin5_w3": lin5_w3, "lin5_w5": lin5_w5,
        "n_v": int(n_v), "n_t": int(n_t),
        "n_boundary": int(n_boundary), "n_meshlets": int(n_meshlets),
        "bnd_pos_norm": bnd_pos_norm,
        "ml_off_bits": ml_off_bits,
        "_raw_bytes": data,
    }


class ParaDeltaGpuParser:
    """GPU bit-decode of one ParaDelta v4 bitstream → flat struct arrays.

    Usage:
        parser = ParaDeltaGpuParser(data)
        s_partial = parser.parse_to_struct()
        # s_partial has all keys of decode_paradelta_to_struct except
        # ml_order/ml_order_off (CPU-only step in Phase 2a).
    """

    _sizes_kernel = None
    _full_kernel = None

    @classmethod
    def _ensure_kernels(cls):
        if cls._sizes_kernel is None:
            mod = cp.RawModule(code=_CUDA_SRC,
                               options=("-std=c++14",))
            cls._sizes_kernel = mod.get_function("paradelta_parse_sizes")
            cls._full_kernel = mod.get_function("paradelta_parse_full")

    def __init__(self, data: bytes):
        if not _HAS_CUPY:
            raise RuntimeError("cupy not installed")
        self._ensure_kernels()
        self.globals = parse_globals(data)

        self.d_buf = cp.asarray(
            np.frombuffer(data, dtype=np.uint8), dtype=cp.uint8)
        self.d_off_bits = cp.asarray(
            self.globals["ml_off_bits"], dtype=cp.uint64)
        self.n_meshlets = self.globals["n_meshlets"]

    def parse_to_struct(self) -> dict:
        n_m = self.n_meshlets

        # Pass 1: sizes
        d_n_bnd    = cp.empty(n_m, dtype=cp.int32)
        d_n_int    = cp.empty(n_m, dtype=cp.int32)
        d_n_tris   = cp.empty(n_m, dtype=cp.int32)
        d_n_strips = cp.empty(n_m, dtype=cp.int32)

        threads = 128
        blocks = (n_m + threads - 1) // threads
        self._sizes_kernel(
            (blocks,), (threads,),
            (self.d_buf, self.d_off_bits, cp.int32(n_m),
             d_n_bnd, d_n_int, d_n_tris, d_n_strips))

        # Build prefix-sum offsets on GPU
        d_l2g_off = cp.empty(n_m + 1, dtype=cp.int32)
        d_tris_off = cp.empty(n_m + 1, dtype=cp.int32)
        d_codes_off = cp.empty(n_m + 1, dtype=cp.int32)
        d_int_cursor = cp.empty(n_m, dtype=cp.int32)  # prefix sum of n_int

        d_l2g_off[0] = 0
        cp.cumsum(d_n_bnd + d_n_int, out=d_l2g_off[1:])
        d_tris_off[0] = 0
        cp.cumsum(d_n_tris, out=d_tris_off[1:])
        d_codes_off[0] = 0
        cp.cumsum(d_n_int, out=d_codes_off[1:])
        # interior cursor: 0 at meshlet 0, then prefix sum (exclusive)
        d_int_cursor[0] = 0
        if n_m > 1:
            cp.cumsum(d_n_int[:-1], out=d_int_cursor[1:])

        # Total sizes
        total_l2g   = int(d_l2g_off[-1].get())
        total_tris  = int(d_tris_off[-1].get())
        total_codes = int(d_codes_off[-1].get())

        d_l2g   = cp.empty(total_l2g, dtype=cp.int32)
        d_tris  = cp.empty(total_tris * 3, dtype=cp.int32)
        d_codes = cp.empty(total_codes * 3, dtype=cp.int32)

        # Pass 2: full parse, one block per meshlet, lane 0 works
        self._full_kernel(
            (n_m,), (32,),
            (self.d_buf, self.d_off_bits,
             d_n_bnd, d_n_int, d_n_tris, d_n_strips,
             d_l2g_off, d_tris_off, d_codes_off,
             cp.int32(self.globals["n_boundary"]),
             d_int_cursor,
             d_l2g, d_tris, d_codes))

        return {
            **{k: v for k, v in self.globals.items()
               if k not in ("ml_off_bits", "_raw_bytes")},
            "ml_n_bnd":   cp.asnumpy(d_n_bnd),
            "ml_n_int":   cp.asnumpy(d_n_int),
            "ml_n_tris":  cp.asnumpy(d_n_tris),
            "ml_n_strips":cp.asnumpy(d_n_strips),
            "ml_l2g_off": cp.asnumpy(d_l2g_off),
            "ml_l2g":     cp.asnumpy(d_l2g),
            "ml_tris_off":cp.asnumpy(d_tris_off),
            "ml_tris":    cp.asnumpy(d_tris).reshape(-1, 3),
            "ml_codes_off":cp.asnumpy(d_codes_off),
            "ml_codes":   cp.asnumpy(d_codes).reshape(-1, 3),
            # GPU-resident handles for chaining into recon kernel
            "_d_n_bnd": d_n_bnd, "_d_n_int": d_n_int,
            "_d_n_tris": d_n_tris, "_d_n_strips": d_n_strips,
            "_d_l2g_off": d_l2g_off, "_d_l2g": d_l2g,
            "_d_tris_off": d_tris_off, "_d_tris": d_tris,
            "_d_codes_off": d_codes_off, "_d_codes": d_codes,
        }
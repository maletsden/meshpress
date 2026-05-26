"""ParaDelta v5 fused GPU decoder.

Single kernel per meshlet: 1 block (32 threads), lane 0 does sequential
bit-decode + strip walk + lin3 predict + apply residual. All work stays
on GPU. No greedy_order step.

Pipeline:
  CPU: parse_globals_v5(data) → header, boundary table, offset table
  GPU: sizes_kernel → n_bnd, n_int, n_tris, n_strips per meshlet
  GPU: cumsum → flat l2g/tris offsets
  GPU: fused_decode_v5 → verts + tris

Reuses Phase 2a sizes kernel (header layout matches between v4/v5 at
the per-meshlet level).
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

from utils.bit_codec import BitReader
from encoder.paradelta_codec import MAGIC
from encoder.paradelta_v5 import VERSION_V5
# Reuse the v4 sizes kernel — per-meshlet header is 4×u16 in both versions.
from utils.paradelta_cuda_parse import _CUDA_SRC as _V4_PARSE_SRC


_V5_CUDA_SRC = r"""
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef long long          int64_t;
typedef int                int32_t;


// ============================================================
// Device BitReader (mirrors Phase 2a)
// ============================================================
__device__ __forceinline__ uint32_t br_read_bits(
    const uint8_t* __restrict__ buf, uint64_t* bp, int n)
{
    uint64_t pos = *bp;
    *bp = pos + n;
    int64_t byte_idx = (int64_t)(pos >> 3);
    int bit_in_byte = (int)(pos & 7ULL);
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

// Fast Rice / Exp-Golomb decoders: use __clz on a 32-bit peek to count the
// unary prefix in one instruction. Original bit-by-bit loop did one full
// 5-byte br_read_bits per zero, so q=4 was already ~20 byte loads.
__device__ __forceinline__ uint32_t br_read_rice(
    const uint8_t* __restrict__ buf, uint64_t* bp, int k)
{
    uint64_t pos = *bp;
    int64_t byte_idx = (int64_t)(pos >> 3);
    int bit_in_byte = (int)(pos & 7ULL);
    uint64_t w =
        ((uint64_t)buf[byte_idx + 0] << 32) |
        ((uint64_t)buf[byte_idx + 1] << 24) |
        ((uint64_t)buf[byte_idx + 2] << 16) |
        ((uint64_t)buf[byte_idx + 3] <<  8) |
        ((uint64_t)buf[byte_idx + 4] <<  0);
    uint32_t peek = (uint32_t)(w >> (8 - bit_in_byte));
    uint32_t q;
    if (peek != 0u) {
        q = (uint32_t)__clz((int)peek);
        *bp = pos + q + 1;
    } else {
        // Fallback: unary > 32 bits (extremely rare). Slow scan.
        *bp = pos + 32;
        q = 32u;
        while (br_read_bits(buf, bp, 1) == 0u) ++q;
    }
    uint32_t low = (k > 0) ? br_read_bits(buf, bp, k) : 0u;
    return (q << k) | low;
}

__device__ __forceinline__ uint32_t br_read_exp_golomb(
    const uint8_t* __restrict__ buf, uint64_t* bp, int k)
{
    uint64_t pos = *bp;
    int64_t byte_idx = (int64_t)(pos >> 3);
    int bit_in_byte = (int)(pos & 7ULL);
    uint64_t w =
        ((uint64_t)buf[byte_idx + 0] << 32) |
        ((uint64_t)buf[byte_idx + 1] << 24) |
        ((uint64_t)buf[byte_idx + 2] << 16) |
        ((uint64_t)buf[byte_idx + 3] <<  8) |
        ((uint64_t)buf[byte_idx + 4] <<  0);
    uint32_t peek = (uint32_t)(w >> (8 - bit_in_byte));
    uint32_t lb;
    if (peek != 0u) {
        lb = (uint32_t)__clz((int)peek);
        *bp = pos + lb + 1;
    } else {
        *bp = pos + 32;
        lb = 32u;
        while (br_read_bits(buf, bp, 1) == 0u) ++lb;
    }
    uint32_t shifted = 0;
    if (lb > 0) {
        uint32_t off = br_read_bits(buf, bp, lb);
        shifted = (1u << lb) - 1u + off;
    }
    uint32_t low = (k > 0) ? br_read_bits(buf, bp, k) : 0u;
    return (shifted << k) | low;
}

__device__ __forceinline__ int32_t zz_to_signed(uint32_t u) {
    return (int32_t)((u >> 1) ^ -(int32_t)(u & 1));
}

__device__ __forceinline__ int idx_bits_for(int n_local) {
    int x = n_local + 1;
    int bits = 0;
    while ((1 << bits) < x) ++bits;
    return bits < 1 ? 1 : bits;
}

#define REUSE_BUF_SIZE 16
#define REUSE_BITS 5

// Register-resident FIFO. Pass int(&fifo)[16] by reference + #pragma unroll
// every access loop so the 16 entries live in lane-0 registers (16 named
// scalars, no array indexing in the emitted PTX). Smem-byte variant got
// replaced because lane-0 still pays ~25 cyc smem latency per touch
// uncovered (no ILP on a single lane). Register path: ~1 cyc per cmov.
__device__ __forceinline__ int read_vert(
    const uint8_t* buf, uint64_t* bp,
    int (&fifo)[REUSE_BUF_SIZE], int* fifo_n, int idx_bits)
{
    uint32_t flag = br_read_bits(buf, bp, 1);
    int v;
    if (flag == 0) {
        int fi = (int)br_read_bits(buf, bp, REUSE_BITS);
        // Unrolled lookup: 16 cmov, dynamic `fi` resolves to compile-time
        // indexed register selects after unroll.
        v = fifo[0];
        #pragma unroll
        for (int i = 0; i < REUSE_BUF_SIZE; ++i) {
            if (i == fi) v = fifo[i];
        }
    } else {
        v = (int)br_read_bits(buf, bp, idx_bits);
    }
    // Touch: find first match, shift left from match, append at tail.
    int n = *fifo_n;
    int found = -1;
    #pragma unroll
    for (int i = 0; i < REUSE_BUF_SIZE; ++i) {
        if (i < n && fifo[i] == v && found < 0) found = i;
    }
    if (found >= 0) {
        #pragma unroll
        for (int i = 0; i < REUSE_BUF_SIZE - 1; ++i) {
            if (i >= found && i < n - 1) fifo[i] = fifo[i+1];
        }
        --n;
    }
    if (n < REUSE_BUF_SIZE) {
        #pragma unroll
        for (int i = 0; i < REUSE_BUF_SIZE; ++i) {
            if (i == n) fifo[i] = v;
        }
        ++n;
    } else {
        #pragma unroll
        for (int i = 0; i < REUSE_BUF_SIZE - 1; ++i) {
            fifo[i] = fifo[i+1];
        }
        fifo[REUSE_BUF_SIZE - 1] = v;
    }
    *fifo_n = n;
    return v;
}


// ============================================================
// Pass 1: read per-meshlet 4×u16 sizes (same as Phase 2a).
// ============================================================
extern "C" __global__ void paradelta_v5_parse_sizes(
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
// Pass 2: fused parse + predict + write.
//   1 block per meshlet, 32 threads/block.
//   Lane 0 owns all sequential decode + predict.
//   All lanes do parallel output writes.
//
// Shared mem (per block):
//   recon[N_LOCAL_MAX * 3]      ~ 3 KB
//   l2g[N_LOCAL_MAX]            ~ 1 KB
//   local_tris[N_TRIS_MAX * 3]  ~ 3 KB
//   codes[N_INT_MAX * 3]        ~ 3 KB
//   emit_v[N_INT_MAX]           ~ 1 KB
//   emit_a/b/c[N_INT_MAX]       ~ 3 KB
//   total ~14 KB
// ============================================================
extern "C" __global__ void paradelta_v5_fused_decode(
    const uint8_t* __restrict__ buf,
    const uint64_t* __restrict__ ml_off_bits,
    const int* __restrict__ ml_n_bnd,
    const int* __restrict__ ml_n_int,
    const int* __restrict__ ml_n_tris,
    const int* __restrict__ ml_n_strips,
    const int* __restrict__ ml_tris_off,
    const int interior_global_base,    // = n_boundary
    const int* __restrict__ interior_cursor, // prefix sum of n_int (exclusive)
    const float* __restrict__ bnd_pos, // (n_boundary, 3) normalized
    const int n_boundary,
    const float lin3_w0, const float lin3_w1, const float lin3_w2,
    const float delta,
    float* __restrict__ verts_out,     // (n_v_total, 3) normalized
    int* __restrict__ tris_out,        // (n_t_total, 3)
    const int n_meshlets_total,
    const int per_warp_smem_bytes,
    const int meshlets_per_block,
    int* __restrict__ ml_counter   // global atomic work-queue head
)
{
    // PERSISTENT KERNEL: launch a fixed grid sized to fill the GPU
    // (~n_sm × max_blocks_per_sm). Each block atomically pulls meshlet IDs
    // from `ml_counter` until exhausted. Eliminates the partial-wave tail
    // effect (was ~12% of Monkey runtime per ncu).
    const int lane = threadIdx.x & 31;
    extern __shared__ unsigned char smem_pool[];
    (void)per_warp_smem_bytes;
    (void)meshlets_per_block;

    #define SEEN_TEST(v) (((seen_bm[(v) >> 5] >> ((v) & 31)) & 1u) != 0u)
    #define SEEN_SET(v)  (seen_bm[(v) >> 5] |= (1u << ((v) & 31)))

    while (true) {
        int m;
        if (lane == 0) {
            m = atomicAdd(ml_counter, 1);
        }
        m = __shfl_sync(0xFFFFFFFFu, m, 0);
        if (m >= n_meshlets_total) break;

        const int n_bnd    = ml_n_bnd[m];
        const int n_int    = ml_n_int[m];
        const int n_tris   = ml_n_tris[m];
        const int n_strips = ml_n_strips[m];
        const int n_local  = n_bnd + n_int;
        const int tris_off = ml_tris_off[m];
        const int int_base = interior_global_base + interior_cursor[m];

        // SMEM layout (rebuilt per meshlet — sizes vary). All pointers live
        // in the single warp's smem pool starting at smem_pool[0].
        //   recon      : float, stride 4 (pad slot .w) so each vert is a
        //                16-byte float4 → single 128-bit smem transaction
        //                per load/store in the predict hot path
        //   codes      : {CODE_T} (residuals; i16 fits common meshes,
        //                  i32 needed for scan-noisy meshes like Lucy
        //                  where one outlier per meshlet hits |code|~160K)
        //   local_tris : uint8 (local IDs ≤ 255)
        //   seen       : uint32 bitmap (8 words = 256 bits)
        float*        recon       = (float*)smem_pool;
        int*          l2g         = (int*)(recon + n_local * 4);
        unsigned int* seen_bm     = (unsigned int*)(l2g + n_local);
        {CODE_T}*     codes_s     = ({CODE_T}*)(seen_bm + 8);
        short*        emit_v      = (short*)(codes_s + n_int * 3);
        short*        emit_a      = emit_v + n_int;
        short*        emit_b      = emit_a + n_int;
        short*        emit_c      = emit_b + n_int;
        unsigned char* local_tris_s = (unsigned char*)(emit_c + n_int);
        // FIFO is now register-resident (declared inside lane-0 block).

        // Cooperative seen_bm init (lanes 0..7 own one word each).
        if (lane < 8) {
            int lo = lane * 32;
            int hi = lo + 32;
            unsigned int mask;
            if (n_bnd >= hi)      mask = 0xFFFFFFFFu;
            else if (n_bnd <= lo) mask = 0u;
            else                  mask = (1u << (n_bnd - lo)) - 1u;
            seen_bm[lane] = mask;
        }
        __syncwarp();

        // ----- Stage A: lane-0 bit decode (l2g, strip, codes). -----
        if (lane == 0) {
        uint64_t bp = ml_off_bits[m] + 64ULL;  // skip 4×u16 hdr (already read)

        // --- Boundary refs (delta-Rice prefix sum): l2g[] only ---
        if (n_bnd > 0) {
            uint32_t first = br_read_bits(buf, &bp, 32);
            l2g[0] = (int)first;
            if (n_bnd > 1) {
                int k = (int)br_read_bits(buf, &bp, 8);
                int prev = (int)first;
                for (int i = 1; i < n_bnd; ++i) {
                    uint32_t u = br_read_rice(buf, &bp, k);
                    prev = prev + (int)u + 1;
                    l2g[i] = prev;
                }
            }
        }
        // Interior global IDs are not stored — computed on-the-fly during
        // output writes via (int_base + (idx - n_bnd)). Saves smem for l2g[].

        // --- Strip parse: build local_tris and emit_order ---
        int idx_bits = idx_bits_for(n_local);
        int fifo[REUSE_BUF_SIZE];
        #pragma unroll
        for (int _i = 0; _i < REUSE_BUF_SIZE; ++_i) fifo[_i] = 0;
        int fifo_n = 0;
        int tri_cursor = 0;
        int emit_cursor = 0;

        for (int s = 0; s < n_strips; ++s) {
            int strip_len = (int)br_read_bits(buf, &bp, 16);
            int v0 = read_vert(buf, &bp, fifo, &fifo_n, idx_bits);
            int v1 = read_vert(buf, &bp, fifo, &fifo_n, idx_bits);
            int v2 = read_vert(buf, &bp, fifo, &fifo_n, idx_bits);
            int rb = tri_cursor * 3;
            local_tris_s[rb+0] = v0;
            local_tris_s[rb+1] = v1;
            local_tris_s[rb+2] = v2;
            ++tri_cursor;
            // Root verts: any not-yet-seen interior gets 'none' context.
            // Inlined 3-way instead of an indexed array to avoid local-mem spill.
            #define ROOT_EMIT(vj) \
                if ((vj) >= n_bnd && !SEEN_TEST(vj)) { \
                    emit_v[emit_cursor] = (short)(vj); \
                    emit_a[emit_cursor] = (short)-1; \
                    emit_b[emit_cursor] = 0; \
                    emit_c[emit_cursor] = 0; \
                    ++emit_cursor; \
                    SEEN_SET(vj); \
                }
            ROOT_EMIT(v0)
            ROOT_EMIT(v1)
            ROOT_EMIT(v2)
            #undef ROOT_EMIT
            int prev_tri_0 = v0, prev_tri_1 = v1, prev_tri_2 = v2;
            for (int t = 1; t < strip_len; ++t) {
                int edge_code = (int)br_read_bits(buf, &bp, 1);
                int new_v = read_vert(buf, &bp, fifo, &fifo_n, idx_bits);
                int a, b, c;
                int np0, np1, np2;
                if (edge_code == 0) {
                    a = prev_tri_1; b = prev_tri_2; c = prev_tri_0;
                    np0 = prev_tri_1; np1 = prev_tri_2; np2 = new_v;
                } else {
                    a = prev_tri_0; b = prev_tri_2; c = prev_tri_1;
                    np0 = prev_tri_0; np1 = prev_tri_2; np2 = new_v;
                }
                int wb = tri_cursor * 3;
                local_tris_s[wb+0] = a;
                local_tris_s[wb+1] = b;
                local_tris_s[wb+2] = new_v;
                ++tri_cursor;
                if (new_v >= n_bnd && !SEEN_TEST(new_v)) {
                    emit_v[emit_cursor] = (short)new_v;
                    emit_a[emit_cursor] = (short)a;
                    emit_b[emit_cursor] = (short)b;
                    emit_c[emit_cursor] = (short)c;
                    ++emit_cursor;
                    SEEN_SET(new_v);
                }
                prev_tri_0 = np0; prev_tri_1 = np1; prev_tri_2 = np2;
            }
        }

        // --- Interior residual codes (per-axis) ---
        if (n_int > 0) {
            for (int d = 0; d < 3; ++d) {
                int tag = (int)br_read_bits(buf, &bp, 8);
                if (tag == 0) {
                    uint32_t mn_u = br_read_bits(buf, &bp, 16);
                    int mn = (mn_u & 0x8000u) ? ((int)mn_u - 0x10000) : (int)mn_u;
                    int bw = (int)br_read_bits(buf, &bp, 8);
                    for (int i = 0; i < n_int; ++i) {
                        codes_s[i*3+d] = mn + (int)br_read_bits(buf, &bp, bw);
                    }
                } else if (tag == 1) {
                    int k = (int)br_read_bits(buf, &bp, 8);
                    for (int i = 0; i < n_int; ++i) {
                        uint32_t u = br_read_rice(buf, &bp, k);
                        codes_s[i*3+d] = zz_to_signed(u);
                    }
                } else {
                    int k = (int)br_read_bits(buf, &bp, 8);
                    for (int i = 0; i < n_int; ++i) {
                        uint32_t u = br_read_exp_golomb(buf, &bp, k);
                        codes_s[i*3+d] = zz_to_signed(u);
                    }
                }
            }
        }
    }
    __syncwarp();

    // ----- Stage B: cooperative boundary recon copy (32 lanes) -----
    // Replaces the prior lane-0 serial scatter from bnd_pos. Spreads the
    // uncoalesced gather across the warp so L1 sees concurrent requests.
    for (int i = lane; i < n_bnd; i += 32) {
        int g = l2g[i];
        float4 v;
        v.x = bnd_pos[g*3+0];
        v.y = bnd_pos[g*3+1];
        v.z = bnd_pos[g*3+2];
        v.w = 0.f;
        *(float4*)&recon[i*4] = v;
    }
    __syncwarp();

    // ----- Stage C: warp-reduce centroid (all 32 lanes) -----
    float fb_x = 0.f, fb_y = 0.f, fb_z = 0.f;
    for (int i = lane; i < n_bnd; i += 32) {
        float4 r = *(float4*)&recon[i*4];
        fb_x += r.x; fb_y += r.y; fb_z += r.z;
    }
    #pragma unroll
    for (int offs = 16; offs > 0; offs >>= 1) {
        fb_x += __shfl_xor_sync(0xFFFFFFFFu, fb_x, offs);
        fb_y += __shfl_xor_sync(0xFFFFFFFFu, fb_y, offs);
        fb_z += __shfl_xor_sync(0xFFFFFFFFu, fb_z, offs);
    }
    if (n_bnd > 0) {
        float inv = 1.0f / (float)n_bnd;
        fb_x *= inv; fb_y *= inv; fb_z *= inv;
    }

        // ----- Stage D: predict + apply (lane-0 serial; float4 SMEM I/O) -----
        if (lane == 0 && n_int > 0) {
            for (int i = 0; i < n_int; ++i) {
                int v_local = (int)emit_v[i];
                int a = (int)emit_a[i];
                float c0 = (float)codes_s[i*3+0];
                float c1 = (float)codes_s[i*3+1];
                float c2 = (float)codes_s[i*3+2];
                float4 out;
                if (a < 0) {
                    out.x = fb_x + c0 * delta;
                    out.y = fb_y + c1 * delta;
                    out.z = fb_z + c2 * delta;
                } else {
                    int b = (int)emit_b[i];
                    int c = (int)emit_c[i];
                    float4 ra = *(float4*)&recon[a*4];
                    float4 rb = *(float4*)&recon[b*4];
                    float4 rc = *(float4*)&recon[c*4];
                    out.x = lin3_w0*ra.x + lin3_w1*rb.x + lin3_w2*rc.x + c0 * delta;
                    out.y = lin3_w0*ra.y + lin3_w1*rb.y + lin3_w2*rc.y + c1 * delta;
                    out.z = lin3_w0*ra.z + lin3_w1*rb.z + lin3_w2*rc.z + c2 * delta;
                }
                out.w = 0.f;
                *(float4*)&recon[v_local*4] = out;
            }
        }
        __syncwarp();

    // --- Parallel write outputs ---
    // Boundary verts already in their global slots, but we re-write to
    // keep things consistent. (Cheap; same value, just over-write.)
    for (int i = lane; i < n_local; i += 32) {
        // Interior gid synthesized; only boundary needs l2g lookup.
        int gid = (i < n_bnd) ? l2g[i] : (int_base + i - n_bnd);
        float4 r = *(float4*)&recon[i*4];
        verts_out[gid*3+0] = r.x;
        verts_out[gid*3+1] = r.y;
        verts_out[gid*3+2] = r.z;
    }
    for (int t = lane; t < n_tris; t += 32) {
        int rb_in  = t * 3;
        int rb_out = (tris_off + t) * 3;
        int i0 = (int)local_tris_s[rb_in+0];
        int i1 = (int)local_tris_s[rb_in+1];
        int i2 = (int)local_tris_s[rb_in+2];
        tris_out[rb_out+0] = (i0 < n_bnd) ? l2g[i0] : (int_base + i0 - n_bnd);
        tris_out[rb_out+1] = (i1 < n_bnd) ? l2g[i1] : (int_base + i1 - n_bnd);
        tris_out[rb_out+2] = (i2 < n_bnd) ? l2g[i2] : (int_base + i2 - n_bnd);
    }
        __syncwarp();
    }  // end persistent fetch loop
}
"""


def parse_globals_v5(data: bytes) -> dict:
    """Parse v5 global header + boundary table on CPU."""
    r = BitReader(data)
    magic = r.read_fixed(32)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X}")
    version = r.read_fixed(8)
    if version != VERSION_V5:
        raise ValueError(f"expected v5, got v{version}")
    code_width = r.read_fixed(8)  # 0=i16 SMEM codes, 1=i32 SMEM codes
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
    lin3_w = np.array([r.read_f32() for _ in range(3)], dtype=np.float32)
    delta = 2.0 * per_coord_err

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
                * float(g_range[d]))

    pad = (-r.bit_pos()) & 7
    if pad:
        r.read_bits(pad)
    ml_off_bits = np.zeros(n_meshlets, dtype=np.uint64)
    for m in range(n_meshlets):
        ml_off_bits[m] = np.uint64(r.read_fixed(32))

    return {
        "code_width": int(code_width),
        "center": center, "scale": np.float32(scale),
        "delta": np.float32(delta),
        "lin3_w": lin3_w,
        "n_v": int(n_v), "n_t": int(n_t),
        "n_boundary": int(n_boundary), "n_meshlets": int(n_meshlets),
        "bnd_pos_norm": bnd_pos_norm,
        "ml_off_bits": ml_off_bits,
    }


class ParaDeltaV5GpuDecoder:
    _sizes_kernel = None
    _fused_kernel_i16 = None
    _fused_kernel_i32 = None

    @classmethod
    def _ensure_kernels(cls):
        if cls._sizes_kernel is None:
            for tag, code_t in (("i16", "short"), ("i32", "int")):
                src = _V5_CUDA_SRC.replace("{CODE_T}", code_t)
                mod = cp.RawModule(code=src, options=("-std=c++14",))
                if cls._sizes_kernel is None:
                    cls._sizes_kernel = mod.get_function(
                        "paradelta_v5_parse_sizes")
                setattr(cls, f"_fused_kernel_{tag}",
                         mod.get_function("paradelta_v5_fused_decode"))

    def __init__(self, data: bytes, meshlets_per_block: int = 1):
        if not _HAS_CUPY:
            raise RuntimeError("cupy not installed")
        if meshlets_per_block < 1 or meshlets_per_block > 16:
            raise ValueError("meshlets_per_block must be in [1, 16]")
        self.meshlets_per_block = int(meshlets_per_block)
        self._ensure_kernels()
        self.globals = parse_globals_v5(data)
        self.code_width = int(self.globals["code_width"])  # 0=i16, 1=i32
        self._fused_kernel = (self._fused_kernel_i32 if self.code_width
                               else self._fused_kernel_i16)

        self.d_buf = cp.asarray(
            np.frombuffer(data, dtype=np.uint8), dtype=cp.uint8)
        self.d_off_bits = cp.asarray(
            self.globals["ml_off_bits"], dtype=cp.uint64)
        self.d_bnd_pos = cp.asarray(
            self.globals["bnd_pos_norm"].reshape(-1), dtype=cp.float32)
        self.n_meshlets = self.globals["n_meshlets"]
        self.n_boundary = self.globals["n_boundary"]
        self.scale = float(self.globals["scale"])
        self.center = np.asarray(self.globals["center"], dtype=np.float32)
        self.lin3_w = self.globals["lin3_w"]
        self.delta = float(self.globals["delta"])

        # Pass 1: per-meshlet sizes
        n_m = self.n_meshlets
        self.d_n_bnd    = cp.empty(n_m, dtype=cp.int32)
        self.d_n_int    = cp.empty(n_m, dtype=cp.int32)
        self.d_n_tris   = cp.empty(n_m, dtype=cp.int32)
        self.d_n_strips = cp.empty(n_m, dtype=cp.int32)
        threads = 128
        blocks = (n_m + threads - 1) // threads
        self._sizes_kernel(
            (blocks,), (threads,),
            (self.d_buf, self.d_off_bits, cp.int32(n_m),
             self.d_n_bnd, self.d_n_int, self.d_n_tris, self.d_n_strips))

        # Prefix sums for tris offsets + interior cursor
        self.d_tris_off = cp.empty(n_m + 1, dtype=cp.int32)
        self.d_tris_off[0] = 0
        cp.cumsum(self.d_n_tris, out=self.d_tris_off[1:])
        self.d_int_cursor = cp.empty(n_m, dtype=cp.int32)
        self.d_int_cursor[0] = 0
        if n_m > 1:
            cp.cumsum(self.d_n_int[:-1], out=self.d_int_cursor[1:])

        # Total sizes for output buffers
        self.n_v_total = (self.n_boundary +
                          int(self.d_n_int.sum().get()))
        self.n_t_total = int(self.d_tris_off[-1].get())

        self.d_verts_norm = cp.zeros((self.n_v_total, 3), dtype=cp.float32)
        self.d_tris = cp.zeros((self.n_t_total, 3), dtype=cp.int32)

        # Shared mem sizing (worst case)
        n_local_max = int((self.d_n_bnd + self.d_n_int).max().get())
        n_bnd_max   = int(self.d_n_bnd.max().get())
        n_tris_max  = int(self.d_n_tris.max().get())
        n_int_max   = int(self.d_n_int.max().get())
        code_bytes = 4 if self.code_width else 2
        self._shared_bytes = (
            n_local_max * 4 * 4        # recon (float4, stride 4 incl. .w pad)
            + n_bnd_max * 4            # l2g (boundary-only; interior synth)
            + 8 * 4                    # seen_bm (uint32[8] = 256-bit bitmap)
            + n_int_max * 3 * code_bytes  # codes (int16 or int32)
            + n_int_max * 2 * 4        # emit_v + emit_a + emit_b + emit_c (4×short)
            + n_tris_max * 3 * 1       # local_tris (uint8)
            # fifo now register-resident (lane 0 only)
        )
        # Single warp per block — smem is reused, atomic counter feeds meshlets.
        self._per_warp_bytes = (self._shared_bytes + 15) & ~15
        self._shared_bytes = self._per_warp_bytes
        if self._shared_bytes > 48 * 1024:
            try:
                self._fused_kernel.max_dynamic_shared_size_bytes = \
                    self._shared_bytes
            except Exception:
                pass

        # Persistent kernel grid: launch ~n_sm × max_blocks_per_sm blocks so
        # the GPU is fully occupied. Each block atomic-fetches meshlet IDs
        # until exhausted. Ampere SM 8.6 caps at 16 blocks/SM; smem floor is
        # smaller (~14) per ncu — but extra blocks just spin on the counter.
        props = cp.cuda.runtime.getDeviceProperties(0)
        n_sm = int(props["multiProcessorCount"])
        self._persistent_blocks = n_sm * 16
        self.d_counter = cp.zeros(1, dtype=cp.int32)

    def decode(self):
        # Re-fill boundary in case earlier run wrote interior over it
        self.d_verts_norm[: self.n_boundary] = \
            self.d_bnd_pos.reshape(self.n_boundary, 3)
        self.d_counter.fill(0)
        # Cap grid at n_meshlets to avoid atomic contention from idle blocks
        # on small meshes (most blocks would otherwise just fetch and exit).
        n_blocks = min(self.n_meshlets, self._persistent_blocks)
        self._fused_kernel(
            (n_blocks,), (32,),
            (self.d_buf, self.d_off_bits,
             self.d_n_bnd, self.d_n_int, self.d_n_tris, self.d_n_strips,
             self.d_tris_off,
             cp.int32(self.n_boundary), self.d_int_cursor,
             self.d_bnd_pos, cp.int32(self.n_boundary),
             cp.float32(self.lin3_w[0]),
             cp.float32(self.lin3_w[1]),
             cp.float32(self.lin3_w[2]),
             cp.float32(self.delta),
             self.d_verts_norm, self.d_tris,
             cp.int32(self.n_meshlets),
             cp.int32(self._per_warp_bytes),
             cp.int32(self.meshlets_per_block),
             self.d_counter),
            shared_mem=self._shared_bytes,
        )
        return self.d_verts_norm, self.d_tris

    def decode_to_host(self):
        v_norm, tris = self.decode()
        verts_world = (v_norm * cp.float32(self.scale)
                       + cp.asarray(self.center, dtype=cp.float32))
        return cp.asnumpy(verts_world), cp.asnumpy(tris)
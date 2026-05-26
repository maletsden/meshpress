// STRIDE-dup C++/CUDA decode-throughput bench harness.
//
// Per meshlet (one CUDA block, 32 threads):
//   * Lane 0 reads the bitstream serially: connectivity (strips +
//     reuse FIFO), anchor (3 raw codes), delta stream, para stream.
//   * Walk classifies each new vert as anchor / delta / para.
//   * Codes reconstructed in SMEM via integer parallelogram (a+b-c).
//   * 32 lanes parallel-write verts (float3) + tris (uint32×3) to
//     global memory.
//
// No global boundary table, no per-meshlet refs table — each meshlet
// is fully self-contained.
//
// Build (Windows, CUDA 12.x, sm_86):
//   nvcc -O3 -arch=sm_86 -std=c++17 stride_dup_decode_bench.cu
//        -o stride_dup_decode_bench.exe
//
// Run:
//   stride_dup_decode_bench.exe blobs/dup/<mesh>.blob [warmup] [runs]
//
// Blob header (LE, fixed 64 B):
//   u32 magic   = 'DPRB' (0x42525044)
//   u32 version = 1
//   u32 n_meshlets
//   u32 n_v_total       (sum of per-meshlet n_local)
//   u32 n_t_total
//   u32 buf_size
//   u32 _resv[2]
//   f32 center[3]
//   f32 scale
//   f32 per_coord_err
//   f32 g_min[3]
//   f32 g_range[3]
//   u32 g_bits[3]
// Then sections (appended in order):
//   u8   buf[buf_size]                   raw bitstream
//   u64  ml_off_bits[n_meshlets]         bit offset of each meshlet
//                                        into buf
//   u32  ml_n_local[n_meshlets]
//   u32  ml_n_tris[n_meshlets]
//   u32  ml_n_strips[n_meshlets]
//   u32  ml_v_off[n_meshlets]            prefix sum of n_local
//   u32  ml_t_off[n_meshlets]            prefix sum of n_tris

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int16_t  i16;
typedef int32_t  i32;
typedef uint64_t u64;

// ===========================================================================
// Device bit-stream primitives — buffered reader.
//
// State holds top-aligned 64-bit window. Refill via single aligned u32 __ldg
// + __byte_perm BE-swap. 1 L1 transaction per refill vs 5 per call in v1.
// ===========================================================================

struct BR {
    u64 buf;         // top-aligned bits; MSB is next to read
    int valid;       // bits remaining in buf, 32..64 after any refill
    const u32* src;  // aligned u32 pointer for next refill
};

__device__ __forceinline__ u32 _ld_be32(const u32* p) {
    return __byte_perm(__ldg(p), 0u, 0x0123u);
}

__device__ __forceinline__ void br_init(
    BR& r, const u8* __restrict__ base, u64 bit_off)
{
    u64 byte_idx = bit_off >> 3;
    int bit_in_byte = (int)(bit_off & 7ULL);
    u64 word_idx = byte_idx >> 2;
    int byte_in_word = (int)(byte_idx & 3ULL);
    int skip = byte_in_word * 8 + bit_in_byte;   // 0..31
    r.src = (const u32*)base + word_idx;
    u32 w0 = _ld_be32(r.src++);
    u32 w1 = _ld_be32(r.src++);
    r.buf = (((u64)w0) << 32) | (u64)w1;
    r.buf <<= skip;
    r.valid = 64 - skip;
}

__device__ __forceinline__ void br_refill(BR& r) {
    if (r.valid <= 32) {
        u32 w = _ld_be32(r.src++);
        r.buf |= ((u64)w) << (32 - r.valid);
        r.valid += 32;
    }
}

__device__ __forceinline__ u32 br_read_bits(BR& r, int n) {
    br_refill(r);
    u32 out = (n == 0) ? 0u : (u32)(r.buf >> (64 - n));
    r.buf <<= n;
    r.valid -= n;
    return out;
}

__device__ __forceinline__ u32 br_read_rice(BR& r, int k) {
    br_refill(r);
    // unary: count leading zeros in r.buf top bits.
    u32 hi = (u32)(r.buf >> 32);
    u32 q;
    if (hi != 0u) {
        q = (u32)__clz((int)hi);
        // consume q+1 bits
        r.buf <<= (q + 1);
        r.valid -= (int)(q + 1);
    } else {
        // All 32 top bits were zero. Drain them, refill, count more.
        r.buf <<= 32;
        r.valid -= 32;
        q = 32u;
        // After this, we keep peeking 1-bit until 1 found. Could be slow but rare.
        while (true) {
            br_refill(r);
            if ((r.buf >> 63) & 1ULL) {
                r.buf <<= 1;
                r.valid -= 1;
                break;
            }
            r.buf <<= 1;
            r.valid -= 1;
            ++q;
        }
    }
    u32 low = (k > 0) ? br_read_bits(r, k) : 0u;
    return (q << k) | low;
}

__device__ __forceinline__ u32 br_read_exp_golomb(BR& r, int k) {
    br_refill(r);
    u32 hi = (u32)(r.buf >> 32);
    u32 q;
    if (hi != 0u) {
        q = (u32)__clz((int)hi);
        r.buf <<= (q + 1);
        r.valid -= (int)(q + 1);
    } else {
        r.buf <<= 32;
        r.valid -= 32;
        q = 32u;
        while (true) {
            br_refill(r);
            if ((r.buf >> 63) & 1ULL) {
                r.buf <<= 1;
                r.valid -= 1;
                break;
            }
            r.buf <<= 1;
            r.valid -= 1;
            ++q;
        }
    }
    u32 base = (q == 0) ? 0u : ((1u << q) - 1u);
    u32 low_q = (q > 0) ? br_read_bits(r, (int)q) : 0u;
    u32 unscaled = base + low_q;
    u32 low_k = (k > 0) ? br_read_bits(r, k) : 0u;
    return (unscaled << k) | low_k;
}

__device__ __forceinline__ i32 zz_to_signed(u32 u) {
    return (i32)((u >> 1) ^ (~(u & 1u) + 1u));
}

// ===========================================================================
// Kernel: one CUDA block per meshlet, 32 threads.
// ===========================================================================
//
// MAX_VERTS_PER_MESHLET = 256 (max_verts/max_tris config), but we
// template on this if needed. For now, hard-cap at 256.
constexpr int MAX_V = 256;
constexpr int MAX_T = 256;
constexpr int FIFO_SZ = 16;
constexpr int REUSE_BITS = 5;

// v4 axis-parallel: 1 warp = 1 meshlet, 4 warps/block. Lane 0 does conn;
// lanes 0/1/2 parallel-decode axes; lane 0 predictor; 32 lanes emit.
constexpr int WARPS_PER_BLOCK = 4;

__global__ void stride_dup_decode_kernel(
    const u8*  __restrict__ buf,
    const u64* __restrict__ ml_off_bits,
    const u32* __restrict__ ml_n_local,
    const u32* __restrict__ ml_n_tris,
    const u32* __restrict__ ml_n_strips,
    const u32* __restrict__ ml_v_off,
    const u32* __restrict__ ml_t_off,
    const u64* __restrict__ ml_resid_off_bits,
    const u32* __restrict__ ml_n_kind0,
    const u16* __restrict__ ml_axis_sub_offs,   // 5 × n_meshlets
    int n_meshlets,
    float g_min0, float g_min1, float g_min2,
    float g_range0, float g_range1, float g_range2,
    int   g_bits0, int   g_bits1, int   g_bits2,
    float center0, float center1, float center2,
    float scale,
    // Generalized parallelogram predictor: per-axis (n0, n1, n2, K).
    int   n_x0, int   n_x1, int   n_x2, int K_x,
    int   n_y0, int   n_y1, int   n_y2, int K_y,
    int   n_z0, int   n_z1, int   n_z2, int K_z,
    float* __restrict__ verts_out,
    u32*   __restrict__ tris_out,
    int*   __restrict__ ml_counter)   // persistent work-steal counter
{
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    // Per-warp SMEM slices (1 meshlet/warp).
    __shared__ u64 codes_s_pool [WARPS_PER_BLOCK * MAX_V];
    __shared__ u8  tris_s_pool  [WARPS_PER_BLOCK * MAX_T * 3];
    __shared__ u32 walk_pack_pool[WARPS_PER_BLOCK * MAX_V];
    __shared__ u32 walk_kind_bm_pool[WARPS_PER_BLOCK * 8];
    // Per-axis residual SMEM (axis-major within a warp for clean lane writes).
    // Layout: [warp_id][axis][slot]
    __shared__ i16 resid_s_pool [WARPS_PER_BLOCK * 3 * MAX_V];
    __shared__ i32 anchor_s_pool[WARPS_PER_BLOCK * 3];
    // Shared per-warp scalars (lane 0 → lanes 0/1/2 handoff).
    __shared__ u32 nkd_pool      [WARPS_PER_BLOCK];   // n_kind0
    __shared__ u32 tagk_pool     [WARPS_PER_BLOCK * 12]; // d_tag[3], d_k[3], p_tag[3], p_k[3]
    __shared__ int bi_smem_pool  [WARPS_PER_BLOCK];

    u64* codes_s        = codes_s_pool        + warp_id * MAX_V;
    u8*  tris_s         = tris_s_pool         + warp_id * MAX_T * 3;
    u32* walk_pack      = walk_pack_pool      + warp_id * MAX_V;
    u32* walk_kind_bm   = walk_kind_bm_pool   + warp_id * 8;
    i16* resid_s        = resid_s_pool        + warp_id * 3 * MAX_V;  // [axis * MAX_V + slot]
    i32* anchor_s       = anchor_s_pool       + warp_id * 3;
    u32& nkd_s          = nkd_pool[warp_id];
    u32* tagk_s         = tagk_pool           + warp_id * 12;
    int& bi_smem        = bi_smem_pool[warp_id];

    while (true) {
    if (lane == 0) bi_smem = atomicAdd(ml_counter, 1);
    __syncwarp();
    const int bi = bi_smem;
    if (bi >= n_meshlets) return;

    const u32 n_local  = ml_n_local[bi];
    const u32 n_tris_m = ml_n_tris[bi];
    const u32 n_strips = ml_n_strips[bi];

    // SMEM layout — codes packed into u64 (i16x4 with last lane unused).
    // One SMEM transaction per ref load. q12 max value 4095 fits in i16;
    // parallelogram math promotes to i32. Walk packed as u32 =
    // v | a | b | c (refs 0..255). Kind = separate bitmask.
    // SMEM slices set up above per warp.

    // -------- Phase A: lane 0 conn-decode + read anchor/tags --------------
    u32 walk_count = 0;   // lane-0 private, persists into predictor phase
    if (lane == 0) {
        BR r;
        br_init(r, buf, ml_off_bits[bi]);
        br_read_bits(r, 16);  // n_local
        br_read_bits(r, 16);  // n_tris
        br_read_bits(r, 16);  // n_strips

        u32 nl_plus1 = n_local + 1;
        int idx_bits = 1;
        while ((1u << idx_bits) < nl_plus1) ++idx_bits;

        u8 fifo[FIFO_SZ];
        int fifo_n = 0;
        #pragma unroll
        for (int i = 0; i < FIFO_SZ; ++i) fifo[i] = 0;

        auto read_vert_local = [&]() -> u8 {
            u32 flag = br_read_bits(r, 1);
            u8 v;
            if (flag == 0u) {
                u32 fi = br_read_bits(r, REUSE_BITS);
                v = fifo[fi];
                for (int k = fi; k + 1 < fifo_n; ++k) fifo[k] = fifo[k + 1];
                --fifo_n;
            } else {
                v = (u8)br_read_bits(r, idx_bits);
            }
            if (fifo_n < FIFO_SZ) {
                fifo[fifo_n++] = v;
            } else {
                for (int k = 0; k + 1 < FIFO_SZ; ++k) fifo[k] = fifo[k + 1];
                fifo[FIFO_SZ - 1] = v;
            }
            return v;
        };

        u32 seen[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) seen[i] = 0u;
        #pragma unroll
        for (int i = 0; i < 8; ++i) walk_kind_bm[i] = 0u;
        #define SEEN_TEST(v) ((seen[(v) >> 5] >> ((v) & 31)) & 1u)
        #define SEEN_SET(v)  (seen[(v) >> 5] |= (1u << ((v) & 31)))

        u32 tri_cursor = 0;
        for (u32 s = 0; s < n_strips; ++s) {
            u32 strip_len = br_read_bits(r, 16);
            u8 v0 = read_vert_local();
            u8 v1 = read_vert_local();
            u8 v2 = read_vert_local();
            tris_s[tri_cursor*3+0] = v0;
            tris_s[tri_cursor*3+1] = v1;
            tris_s[tri_cursor*3+2] = v2;
            ++tri_cursor;
            if (!SEEN_TEST(v0)) { walk_pack[walk_count++] = (u32)v0; SEEN_SET(v0); }
            if (!SEEN_TEST(v1)) { walk_pack[walk_count++] = (u32)v1; SEEN_SET(v1); }
            if (!SEEN_TEST(v2)) { walk_pack[walk_count++] = (u32)v2; SEEN_SET(v2); }
            u8 prev0 = v0, prev1 = v1, prev2 = v2;
            for (u32 t = 1; t < strip_len; ++t) {
                u32 ec = br_read_bits(r, 1);
                u8 new_v = read_vert_local();
                u8 a, b, c, np0, np1, np2;
                if (ec == 0u) { a = prev1; b = prev2; c = prev0;
                                np0 = prev1; np1 = prev2; np2 = new_v; }
                else          { a = prev0; b = prev2; c = prev1;
                                np0 = prev0; np1 = prev2; np2 = new_v; }
                tris_s[tri_cursor*3+0] = a;
                tris_s[tri_cursor*3+1] = b;
                tris_s[tri_cursor*3+2] = new_v;
                ++tri_cursor;
                if (!SEEN_TEST(new_v)) {
                    walk_pack[walk_count] = (u32)new_v |
                        ((u32)a << 8) | ((u32)b << 16) | ((u32)c << 24);
                    walk_kind_bm[walk_count >> 5] |= (1u << (walk_count & 31));
                    ++walk_count;
                    SEEN_SET(new_v);
                }
                prev0 = np0; prev1 = np1; prev2 = np2;
            }
        }
        #undef SEEN_TEST
        #undef SEEN_SET

        // Read anchor + tag headers (jump to ml_resid_off_bits[bi]).
        BR rr;
        br_init(rr, buf, ml_resid_off_bits[bi]);
        anchor_s[0] = (i32)br_read_bits(rr, g_bits0);
        anchor_s[1] = (i32)br_read_bits(rr, g_bits1);
        anchor_s[2] = (i32)br_read_bits(rr, g_bits2);
        #pragma unroll
        for (int d = 0; d < 3; ++d) {
            tagk_s[d*2+0]   = br_read_bits(rr, 8);   // d_tag[d]
            tagk_s[d*2+1]   = br_read_bits(rr, 8);   // d_k[d]
        }
        #pragma unroll
        for (int d = 0; d < 3; ++d) {
            tagk_s[6 + d*2+0] = br_read_bits(rr, 8); // p_tag[d]
            tagk_s[6 + d*2+1] = br_read_bits(rr, 8); // p_k[d]
        }
        nkd_s = ml_n_kind0[bi];
    }
    __syncwarp();

    // -------- Phase B: lanes 0/1/2 parallel axis Rice/EG decode ----------
    if (lane < 3) {
        const int axis = lane;
        const u32 n_k0    = nkd_s;
        const u32 n_delta = (n_k0 > 0) ? (n_k0 - 1) : 0;
        const u32 n_para  = n_local - n_k0;

        // delta-axis-start (relative to ml_resid_off_bits[bi]):
        //   axis 0: g_bits0 + g_bits1 + g_bits2 + 96 (after anchor + 6 tag bytes)
        //   axis 1: ml_axis_sub_offs[bi*5 + 0]
        //   axis 2: ml_axis_sub_offs[bi*5 + 1]
        // para-axis-start:
        //   axis 0: ml_axis_sub_offs[bi*5 + 2]
        //   axis 1: ml_axis_sub_offs[bi*5 + 3]
        //   axis 2: ml_axis_sub_offs[bi*5 + 4]
        u32 d_off_rel, p_off_rel;
        if (axis == 0) {
            d_off_rel = (u32)g_bits0 + (u32)g_bits1 + (u32)g_bits2 + 96u;
            p_off_rel = (u32)ml_axis_sub_offs[bi*5 + 2];
        } else if (axis == 1) {
            d_off_rel = (u32)ml_axis_sub_offs[bi*5 + 0];
            p_off_rel = (u32)ml_axis_sub_offs[bi*5 + 3];
        } else {
            d_off_rel = (u32)ml_axis_sub_offs[bi*5 + 1];
            p_off_rel = (u32)ml_axis_sub_offs[bi*5 + 4];
        }
        const u64 resid_off = ml_resid_off_bits[bi];
        const u32 d_tag = tagk_s[axis*2 + 0];
        const u32 d_k   = tagk_s[axis*2 + 1];
        const u32 p_tag = tagk_s[6 + axis*2 + 0];
        const u32 p_k   = tagk_s[6 + axis*2 + 1];

        BR rd; br_init(rd, buf, resid_off + (u64)d_off_rel);
        i16* dst = resid_s + axis * MAX_V;
        for (u32 i = 0; i < n_delta; ++i) {
            u32 u = (d_tag == 1u) ? br_read_rice(rd, d_k)
                                  : br_read_exp_golomb(rd, d_k);
            dst[i] = (i16)zz_to_signed(u);
        }
        BR rp; br_init(rp, buf, resid_off + (u64)p_off_rel);
        for (u32 i = 0; i < n_para; ++i) {
            u32 u = (p_tag == 1u) ? br_read_rice(rp, p_k)
                                  : br_read_exp_golomb(rp, p_k);
            dst[n_delta + i] = (i16)zz_to_signed(u);
        }
    }
    __syncwarp();

    // -------- Phase C: lane 0 predictor (axis-major SMEM reads) ----------
    if (lane == 0) {
        const i32 anchor_x = anchor_s[0];
        const i32 anchor_y = anchor_s[1];
        const i32 anchor_z = anchor_s[2];
        const u32 n_k0       = nkd_s;
        const u32 n_delta_slot = (n_k0 > 0) ? (n_k0 - 1) : 0;
        const i16* rx_arr = resid_s + 0 * MAX_V;
        const i16* ry_arr = resid_s + 1 * MAX_V;
        const i16* rz_arr = resid_s + 2 * MAX_V;

        i32 prev_x = 0, prev_y = 0, prev_z = 0;
        bool first = true;
        u32 cur_d = 0, cur_p = 0;
        for (u32 i = 0; i < walk_count; ++i) {
            u32 pk = walk_pack[i];
            u32 v = pk & 0xFFu;
            u32 kind = (walk_kind_bm[i >> 5] >> (i & 31)) & 1u;
            i32 cx, cy, cz;
            if (kind == 0u) {
                if (first) {
                    cx = anchor_x; cy = anchor_y; cz = anchor_z;
                    first = false;
                } else {
                    cx = prev_x + (i32)rx_arr[cur_d];
                    cy = prev_y + (i32)ry_arr[cur_d];
                    cz = prev_z + (i32)rz_arr[cur_d];
                    ++cur_d;
                }
            } else {
                u32 a = (pk >>  8) & 0xFFu;
                u32 b = (pk >> 16) & 0xFFu;
                u32 c = (pk >> 24) & 0xFFu;
                u64 av64 = codes_s[a];
                u64 bv64 = codes_s[b];
                u64 cv64 = codes_s[c];
                i32 ax = (i32)(i16)(av64 & 0xFFFFu);
                i32 ay = (i32)(i16)((av64 >> 16) & 0xFFFFu);
                i32 az = (i32)(i16)((av64 >> 32) & 0xFFFFu);
                i32 bx = (i32)(i16)(bv64 & 0xFFFFu);
                i32 by = (i32)(i16)((bv64 >> 16) & 0xFFFFu);
                i32 bz = (i32)(i16)((bv64 >> 32) & 0xFFFFu);
                i32 cxr = (i32)(i16)(cv64 & 0xFFFFu);
                i32 cyr = (i32)(i16)((cv64 >> 16) & 0xFFFFu);
                i32 czr = (i32)(i16)((cv64 >> 32) & 0xFFFFu);
                u32 slot = n_delta_slot + cur_p;
                // Generalized parallelogram: pred = (n0·a + n1·b + n2·c + half) >> K
                // (canonical = K=0, n=(1,1,-1) → pred = a + b - c)
                i32 sx = n_x0 * ax + n_x1 * bx + n_x2 * cxr;
                i32 sy = n_y0 * ay + n_y1 * by + n_y2 * cyr;
                i32 sz = n_z0 * az + n_z1 * bz + n_z2 * czr;
                i32 px = (K_x == 0) ? sx : ((sx + (1 << (K_x - 1))) >> K_x);
                i32 py = (K_y == 0) ? sy : ((sy + (1 << (K_y - 1))) >> K_y);
                i32 pz = (K_z == 0) ? sz : ((sz + (1 << (K_z - 1))) >> K_z);
                cx = px + (i32)rx_arr[slot];
                cy = py + (i32)ry_arr[slot];
                cz = pz + (i32)rz_arr[slot];
                ++cur_p;
            }
            codes_s[v] = ((u64)(u16)(i16)cx)
                         | (((u64)(u16)(i16)cy) << 16)
                         | (((u64)(u16)(i16)cz) << 32);
            prev_x = cx; prev_y = cy; prev_z = cz;
        }
    }
    __syncwarp();

    // -------- Phase 4: parallel emit verts + tris (32 lanes per warp) ----
    const float inv_mx0 = (g_bits0 > 0) ? (g_range0 / (float)((1u << g_bits0) - 1u)) : 0.0f;
    const float inv_mx1 = (g_bits1 > 0) ? (g_range1 / (float)((1u << g_bits1) - 1u)) : 0.0f;
    const float inv_mx2 = (g_bits2 > 0) ? (g_range2 / (float)((1u << g_bits2) - 1u)) : 0.0f;
    const u32 v_base = ml_v_off[bi];
    const u32 t_base = ml_t_off[bi];

    for (u32 i = lane; i < n_local; i += 32) {
        u64 v64 = codes_s[i];
        i32 cx = (i32)(i16)(v64 & 0xFFFFu);
        i32 cy = (i32)(i16)((v64 >> 16) & 0xFFFFu);
        i32 cz = (i32)(i16)((v64 >> 32) & 0xFFFFu);
        float fx = (g_min0 + (float)cx * inv_mx0) * scale + center0;
        float fy = (g_min1 + (float)cy * inv_mx1) * scale + center1;
        float fz = (g_min2 + (float)cz * inv_mx2) * scale + center2;
        verts_out[3 * (v_base + i) + 0] = fx;
        verts_out[3 * (v_base + i) + 1] = fy;
        verts_out[3 * (v_base + i) + 2] = fz;
    }
    for (u32 t = lane; t < n_tris_m; t += 32) {
        u32 i0 = tris_s[3 * t + 0];
        u32 i1 = tris_s[3 * t + 1];
        u32 i2 = tris_s[3 * t + 2];
        tris_out[3 * (t_base + t) + 0] = v_base + i0;
        tris_out[3 * (t_base + t) + 1] = v_base + i1;
        tris_out[3 * (t_base + t) + 2] = v_base + i2;
    }
    __syncwarp();
    } // persistent loop
}

// ===========================================================================
// Host harness.
// ===========================================================================

#define CK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); std::exit(1); \
    } \
} while (0)

struct DupBlobHeader {
    u32 magic;
    u32 version;
    u32 n_meshlets;
    u32 n_v_total;
    u32 n_t_total;
    u32 buf_size;
    u32 _resv[2];
    float center[3];
    float scale;
    float per_coord_err;
    float g_min[3];
    float g_range[3];
    u32   g_bits[3];
    // v4 predictor: 9 int16 numerators + 3 uint8 K + 3 pad = 24 B
    i16   pred_n[9];
    u8    pred_K[3];
    u8    _pad[3];
};
static_assert(sizeof(DupBlobHeader) == 88 + 24, "DupBlobHeader size mismatch");

struct DupBuffers {
    DupBlobHeader hdr;
    std::vector<u8>  buf;
    std::vector<u64> ml_off_bits;
    std::vector<u32> ml_n_local;
    std::vector<u32> ml_n_tris;
    std::vector<u32> ml_n_strips;
    std::vector<u32> ml_v_off;
    std::vector<u32> ml_t_off;
    std::vector<u64> ml_resid_off_bits;
    std::vector<u32> ml_n_kind0;
    std::vector<u16> ml_axis_sub_offs;   // 5 × n_meshlets, layout: meshlet-major
};

static bool read_dup_blob(const std::string& path, DupBuffers& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "open %s\n", path.c_str()); return false; }
    f.read((char*)&out.hdr, sizeof(out.hdr));
    if (out.hdr.magic != 0x42525044u) {
        fprintf(stderr, "bad magic 0x%08x\n", out.hdr.magic);
        return false;
    }
    if (out.hdr.version < 4u) {
        fprintf(stderr,
            "blob version %u not supported; need v4 (generalized parallelogram)\n",
            out.hdr.version);
        return false;
    }
    const auto& h = out.hdr;
    out.buf.resize(h.buf_size);
    f.read((char*)out.buf.data(), h.buf_size);
    out.ml_off_bits.resize(h.n_meshlets);
    f.read((char*)out.ml_off_bits.data(), h.n_meshlets * 8);
    out.ml_n_local.resize(h.n_meshlets);
    f.read((char*)out.ml_n_local.data(), h.n_meshlets * 4);
    out.ml_n_tris.resize(h.n_meshlets);
    f.read((char*)out.ml_n_tris.data(), h.n_meshlets * 4);
    out.ml_n_strips.resize(h.n_meshlets);
    f.read((char*)out.ml_n_strips.data(), h.n_meshlets * 4);
    out.ml_v_off.resize(h.n_meshlets);
    f.read((char*)out.ml_v_off.data(), h.n_meshlets * 4);
    out.ml_t_off.resize(h.n_meshlets);
    f.read((char*)out.ml_t_off.data(), h.n_meshlets * 4);
    out.ml_resid_off_bits.resize(h.n_meshlets);
    f.read((char*)out.ml_resid_off_bits.data(), h.n_meshlets * 8);
    out.ml_n_kind0.resize(h.n_meshlets);
    f.read((char*)out.ml_n_kind0.data(), h.n_meshlets * 4);
    out.ml_axis_sub_offs.resize(h.n_meshlets * 5);
    f.read((char*)out.ml_axis_sub_offs.data(), h.n_meshlets * 5 * 2);
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("usage: %s <blob> [warmup=20] [runs=100] [--dump <prefix>]\n",
               argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int warmup = 20, runs = 100;
    const char* dump_prefix = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_prefix = argv[++i];
        } else if (argv[i][0] != '-') {
            if (i == 2) warmup = std::atoi(argv[i]);
            else if (i == 3) runs = std::atoi(argv[i]);
        }
    }

    DupBuffers B;
    if (!read_dup_blob(path, B)) return 1;
    const auto& h = B.hdr;

    CK(cudaSetDevice(0));

    u8*  d_buf = nullptr;
    u64* d_off = nullptr;
    u32* d_n_local = nullptr;
    u32* d_n_tris  = nullptr;
    u32* d_n_strips = nullptr;
    u32* d_v_off  = nullptr;
    u32* d_t_off  = nullptr;
    u64* d_resid_off = nullptr;
    u32* d_n_kind0   = nullptr;
    u16* d_axis_off  = nullptr;
    float* d_verts = nullptr;
    u32*   d_tris  = nullptr;
    int*   d_counter = nullptr;

    CK(cudaMalloc(&d_buf, h.buf_size + 32));  // +32B tail pad: buffered reader prefetches u32s past meshlet end
    CK(cudaMemsetAsync(d_buf + h.buf_size, 0, 32, 0));
    CK(cudaMalloc(&d_off, h.n_meshlets * sizeof(u64)));
    CK(cudaMalloc(&d_counter, sizeof(int)));
    CK(cudaMalloc(&d_n_local,  h.n_meshlets * sizeof(u32)));
    CK(cudaMalloc(&d_n_tris,   h.n_meshlets * sizeof(u32)));
    CK(cudaMalloc(&d_n_strips, h.n_meshlets * sizeof(u32)));
    CK(cudaMalloc(&d_v_off,    h.n_meshlets * sizeof(u32)));
    CK(cudaMalloc(&d_t_off,    h.n_meshlets * sizeof(u32)));
    CK(cudaMalloc(&d_resid_off, h.n_meshlets * sizeof(u64)));
    CK(cudaMalloc(&d_n_kind0,   h.n_meshlets * sizeof(u32)));
    CK(cudaMalloc(&d_axis_off,  h.n_meshlets * 5 * sizeof(u16)));
    CK(cudaMalloc(&d_verts, (size_t)h.n_v_total * 3 * sizeof(float)));
    CK(cudaMalloc(&d_tris,  (size_t)h.n_t_total * 3 * sizeof(u32)));

    CK(cudaMemcpy(d_buf,      B.buf.data(),         h.buf_size,                cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_off,      B.ml_off_bits.data(), h.n_meshlets * 8,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_local,  B.ml_n_local.data(),  h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_tris,   B.ml_n_tris.data(),   h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_strips, B.ml_n_strips.data(), h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v_off,    B.ml_v_off.data(),    h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_t_off,    B.ml_t_off.data(),    h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_resid_off, B.ml_resid_off_bits.data(), h.n_meshlets * 8,   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_kind0,   B.ml_n_kind0.data(),       h.n_meshlets * 4,    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_axis_off,  B.ml_axis_sub_offs.data(), h.n_meshlets * 5 * 2, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int n_sm = prop.multiProcessorCount;
    int grid_blocks = std::min((int)h.n_meshlets, n_sm * 16);

    auto launch = [&]() {
        cudaMemsetAsync(d_counter, 0, sizeof(int), 0);
        dim3 grid(grid_blocks);
        dim3 block(32 * WARPS_PER_BLOCK);
        stride_dup_decode_kernel<<<grid, block>>>(
            d_buf, d_off, d_n_local, d_n_tris, d_n_strips,
            d_v_off, d_t_off, d_resid_off, d_n_kind0, d_axis_off,
            (int)h.n_meshlets,
            h.g_min[0], h.g_min[1], h.g_min[2],
            h.g_range[0], h.g_range[1], h.g_range[2],
            (int)h.g_bits[0], (int)h.g_bits[1], (int)h.g_bits[2],
            h.center[0], h.center[1], h.center[2], h.scale,
            (int)h.pred_n[0], (int)h.pred_n[1], (int)h.pred_n[2], (int)h.pred_K[0],
            (int)h.pred_n[3], (int)h.pred_n[4], (int)h.pred_n[5], (int)h.pred_K[1],
            (int)h.pred_n[6], (int)h.pred_n[7], (int)h.pred_n[8], (int)h.pred_K[2],
            d_verts, d_tris, d_counter);
    };

    for (int i = 0; i < warmup; ++i) launch();
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    CK(cudaEventRecord(s));
    for (int i = 0; i < runs; ++i) launch();
    CK(cudaEventRecord(e));
    CK(cudaEventSynchronize(e));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, s, e));
    double per_us = (double)ms * 1000.0 / runs;
    double mtps   = (double)h.n_t_total / per_us;

    CK(cudaEventRecord(s));
    launch();
    CK(cudaEventRecord(e));
    CK(cudaEventSynchronize(e));
    float ms1 = 0.f;
    CK(cudaEventElapsedTime(&ms1, s, e));
    double single_us = (double)ms1 * 1000.0;

    printf("%s,%u,%u,%u,%.3f,%.3f,%.3f,%d,%d\n",
           path, h.n_v_total, h.n_t_total, h.n_meshlets,
           single_us, per_us, mtps, warmup, runs);
    fprintf(stderr,
            "%-48s n_v=%-8u n_t=%-8u n_m=%-5u  single=%7.1f us  "
            "amortized=%7.1f us  %7.1f Mtris/s\n",
            path, h.n_v_total, h.n_t_total, h.n_meshlets,
            single_us, per_us, mtps);

    if (dump_prefix) {
        std::vector<float> h_verts((size_t)h.n_v_total * 3);
        std::vector<u32>   h_tris((size_t)h.n_t_total * 3);
        CK(cudaMemcpy(h_verts.data(), d_verts,
            h_verts.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_tris.data(),  d_tris,
            h_tris.size()  * sizeof(u32),   cudaMemcpyDeviceToHost));
        std::string vp = std::string(dump_prefix) + ".verts.f32";
        std::string tp = std::string(dump_prefix) + ".tris.u32";
        std::ofstream vf(vp, std::ios::binary);
        vf.write((const char*)h_verts.data(), h_verts.size() * sizeof(float));
        std::ofstream tf(tp, std::ios::binary);
        tf.write((const char*)h_tris.data(),  h_tris.size()  * sizeof(u32));
        fprintf(stderr, "Dumped %s and %s\n", vp.c_str(), tp.c_str());
    }

    CK(cudaFree(d_buf));     CK(cudaFree(d_off));
    CK(cudaFree(d_n_local)); CK(cudaFree(d_n_tris));
    CK(cudaFree(d_n_strips));
    CK(cudaFree(d_v_off));   CK(cudaFree(d_t_off));
    CK(cudaFree(d_resid_off)); CK(cudaFree(d_n_kind0));
    CK(cudaFree(d_axis_off));
    CK(cudaFree(d_verts));   CK(cudaFree(d_tris));
    CK(cudaFree(d_counter));
    return 0;
}

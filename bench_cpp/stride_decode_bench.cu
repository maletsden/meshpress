// STRIDE C++/CUDA decode-throughput bench harness.
//
// Loads a STRIDE blob (produced by scripts/dump_stride_blob.py), uploads
// buffers, JIT-launches the fused decode kernel, and times via CUDA Events.
// No Python, no CuPy — pure C/C++ launch overhead.
//
// Build (Windows, CUDA 12.x, sm_86):
//   nvcc -O3 -arch=sm_86 -std=c++17 stride_decode_bench.cu
//        -o stride_decode_bench.exe
//
// Run:
//   stride_decode_bench.exe blobs/bunny.blob [warmup] [runs]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>

// ===========================================================================
// Kernel source (copy of paradelta_v5_fused_decode from
// utils/paradelta_v5_cuda.py with {CODE_T} substituted explicitly).
// ===========================================================================

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// ---------- Device bit-stream primitives ----------
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

__device__ __forceinline__ int read_vert(
    const uint8_t* buf, uint64_t* bp,
    int (&fifo)[REUSE_BUF_SIZE], int* fifo_n, int idx_bits)
{
    uint32_t flag = br_read_bits(buf, bp, 1);
    int v;
    if (flag == 0) {
        int fi = (int)br_read_bits(buf, bp, REUSE_BITS);
        v = fifo[0];
        #pragma unroll
        for (int i = 0; i < REUSE_BUF_SIZE; ++i) {
            if (i == fi) v = fifo[i];
        }
    } else {
        v = (int)br_read_bits(buf, bp, idx_bits);
    }
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

// ---------- Fused decode kernel (template on code type) ----------
template <typename CODE_T>
__global__ void paradelta_v5_fused_decode(
    const uint8_t* __restrict__ buf,
    const uint64_t* __restrict__ ml_off_bits,
    const int* __restrict__ ml_n_bnd,
    const int* __restrict__ ml_n_int,
    const int* __restrict__ ml_n_tris,
    const int* __restrict__ ml_n_strips,
    const int* __restrict__ ml_tris_off,
    const int interior_global_base,
    const int* __restrict__ interior_cursor,
    const float* __restrict__ bnd_pos,
    const int n_boundary,
    const float lin3_w0, const float lin3_w1, const float lin3_w2,
    const float delta,
    float* __restrict__ verts_out,
    int* __restrict__ tris_out,
    const int n_meshlets_total,
    const int per_warp_smem_bytes,
    const int meshlets_per_block,
    int* __restrict__ ml_counter)
{
    const int lane = threadIdx.x & 31;
    extern __shared__ unsigned char smem_pool[];
    (void)per_warp_smem_bytes;
    (void)meshlets_per_block;

    #define SEEN_TEST(v) (((seen_bm[(v) >> 5] >> ((v) & 31)) & 1u) != 0u)
    #define SEEN_SET(v)  (seen_bm[(v) >> 5] |= (1u << ((v) & 31)))

    while (true) {
        int m;
        if (lane == 0) m = atomicAdd(ml_counter, 1);
        m = __shfl_sync(0xFFFFFFFFu, m, 0);
        if (m >= n_meshlets_total) break;

        const int n_bnd    = ml_n_bnd[m];
        const int n_int    = ml_n_int[m];
        const int n_tris   = ml_n_tris[m];
        const int n_strips = ml_n_strips[m];
        const int n_local  = n_bnd + n_int;
        const int tris_off = ml_tris_off[m];
        const int int_base = interior_global_base + interior_cursor[m];

        float*        recon       = (float*)smem_pool;
        int*          l2g         = (int*)(recon + n_local * 4);
        unsigned int* seen_bm     = (unsigned int*)(l2g + n_local);
        CODE_T*       codes_s     = (CODE_T*)(seen_bm + 8);
        short*        emit_v      = (short*)(codes_s + n_int * 3);
        short*        emit_a      = emit_v + n_int;
        short*        emit_b      = emit_a + n_int;
        short*        emit_c      = emit_b + n_int;
        unsigned char* local_tris_s = (unsigned char*)(emit_c + n_int);

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

        if (lane == 0) {
            uint64_t bp = ml_off_bits[m] + 64ULL;

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

                #define ROOT_EMIT(vj) \
                    if ((vj) >= n_bnd && !SEEN_TEST(vj)) { \
                        emit_v[emit_cursor] = (short)(vj); \
                        emit_a[emit_cursor] = (short)-1; \
                        emit_b[emit_cursor] = 0; \
                        emit_c[emit_cursor] = 0; \
                        ++emit_cursor; \
                        SEEN_SET(vj); \
                    }
                ROOT_EMIT(v0) ROOT_EMIT(v1) ROOT_EMIT(v2)
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

            if (n_int > 0) {
                for (int d = 0; d < 3; ++d) {
                    int tag = (int)br_read_bits(buf, &bp, 8);
                    if (tag == 0) {
                        uint32_t mn_u = br_read_bits(buf, &bp, 16);
                        int mn = (mn_u & 0x8000u) ? ((int)mn_u - 0x10000) : (int)mn_u;
                        int bw = (int)br_read_bits(buf, &bp, 8);
                        for (int i = 0; i < n_int; ++i) {
                            codes_s[i*3+d] = (CODE_T)(mn + (int)br_read_bits(buf, &bp, bw));
                        }
                    } else if (tag == 1) {
                        int k = (int)br_read_bits(buf, &bp, 8);
                        for (int i = 0; i < n_int; ++i) {
                            uint32_t u = br_read_rice(buf, &bp, k);
                            codes_s[i*3+d] = (CODE_T)zz_to_signed(u);
                        }
                    } else {
                        int k = (int)br_read_bits(buf, &bp, 8);
                        for (int i = 0; i < n_int; ++i) {
                            uint32_t u = br_read_exp_golomb(buf, &bp, k);
                            codes_s[i*3+d] = (CODE_T)zz_to_signed(u);
                        }
                    }
                }
            }
        }
        __syncwarp();

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

        for (int i = lane; i < n_local; i += 32) {
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
    }
}

// Template variants are instantiated implicitly at the host launch sites
// (paradelta_v5_fused_decode<short><<<...>>> / <int><<<...>>>).

// ===========================================================================
// Host harness
// ===========================================================================

#define CK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); std::exit(1); \
    } \
} while (0)

struct BlobHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t code_width;
    uint32_t n_meshlets;
    uint32_t n_boundary;
    uint32_t n_v_total;
    uint32_t n_t_total;
    uint32_t buf_size;
    float    lin3_w0;
    float    lin3_w1;
    float    lin3_w2;
    float    delta;
    uint32_t per_warp_bytes;
    uint32_t meshlets_per_block;
    uint32_t _resv0;
    uint32_t _resv1;
};
static_assert(sizeof(BlobHeader) == 64, "header size must be 64");

struct Buffers {
    BlobHeader hdr;
    std::vector<uint8_t>  buf;
    std::vector<uint64_t> ml_off_bits;
    std::vector<int32_t>  ml_n_bnd;
    std::vector<int32_t>  ml_n_int;
    std::vector<int32_t>  ml_n_tris;
    std::vector<int32_t>  ml_n_strips;
    std::vector<int32_t>  ml_tris_off;
    std::vector<int32_t>  ml_int_cursor;
    std::vector<float>    bnd_pos_norm;
};

static bool read_blob(const std::string& path, Buffers& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "open %s\n", path.c_str()); return false; }
    f.read((char*)&out.hdr, sizeof(out.hdr));
    if (out.hdr.magic != 0x424c4253u) { fprintf(stderr, "bad magic\n"); return false; }
    auto& h = out.hdr;
    out.buf.resize(h.buf_size);
    f.read((char*)out.buf.data(), h.buf_size);
    out.ml_off_bits.resize(h.n_meshlets);
    f.read((char*)out.ml_off_bits.data(), h.n_meshlets * 8);
    out.ml_n_bnd.resize(h.n_meshlets);    f.read((char*)out.ml_n_bnd.data(),    h.n_meshlets * 4);
    out.ml_n_int.resize(h.n_meshlets);    f.read((char*)out.ml_n_int.data(),    h.n_meshlets * 4);
    out.ml_n_tris.resize(h.n_meshlets);   f.read((char*)out.ml_n_tris.data(),   h.n_meshlets * 4);
    out.ml_n_strips.resize(h.n_meshlets); f.read((char*)out.ml_n_strips.data(), h.n_meshlets * 4);
    out.ml_tris_off.resize(h.n_meshlets + 1);
    f.read((char*)out.ml_tris_off.data(), (h.n_meshlets + 1) * 4);
    out.ml_int_cursor.resize(h.n_meshlets);
    f.read((char*)out.ml_int_cursor.data(), h.n_meshlets * 4);
    out.bnd_pos_norm.resize(h.n_boundary * 3);
    f.read((char*)out.bnd_pos_norm.data(), h.n_boundary * 12);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <blob> [warmup=20] [runs=100]\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int warmup = (argc > 2) ? std::atoi(argv[2]) : 20;
    int runs   = (argc > 3) ? std::atoi(argv[3]) : 100;

    Buffers B;
    if (!read_blob(path, B)) return 1;
    const auto& h = B.hdr;

    // Device props
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    int n_sm = prop.multiProcessorCount;
    int n_blocks = (int)h.n_meshlets;
    int cap = n_sm * 16;
    if (n_blocks > cap) n_blocks = cap;

    // Allocate GPU buffers + upload (one-shot, outside the timed loop)
    uint8_t*  d_buf = nullptr;
    uint64_t* d_off = nullptr;
    int*      d_n_bnd = nullptr;
    int*      d_n_int = nullptr;
    int*      d_n_tris = nullptr;
    int*      d_n_strips = nullptr;
    int*      d_tris_off = nullptr;
    int*      d_int_cur = nullptr;
    float*    d_bnd_pos = nullptr;
    float*    d_verts = nullptr;
    int*      d_tris  = nullptr;
    int*      d_counter = nullptr;

    CK(cudaMalloc(&d_buf,        h.buf_size));
    CK(cudaMalloc(&d_off,        h.n_meshlets * sizeof(uint64_t)));
    CK(cudaMalloc(&d_n_bnd,      h.n_meshlets * sizeof(int)));
    CK(cudaMalloc(&d_n_int,      h.n_meshlets * sizeof(int)));
    CK(cudaMalloc(&d_n_tris,     h.n_meshlets * sizeof(int)));
    CK(cudaMalloc(&d_n_strips,   h.n_meshlets * sizeof(int)));
    CK(cudaMalloc(&d_tris_off,   (h.n_meshlets + 1) * sizeof(int)));
    CK(cudaMalloc(&d_int_cur,    h.n_meshlets * sizeof(int)));
    CK(cudaMalloc(&d_bnd_pos,    h.n_boundary * 3 * sizeof(float)));
    CK(cudaMalloc(&d_verts,      h.n_v_total * 3 * sizeof(float)));
    CK(cudaMalloc(&d_tris,       h.n_t_total * 3 * sizeof(int)));
    CK(cudaMalloc(&d_counter,    sizeof(int)));

    CK(cudaMemcpy(d_buf,      B.buf.data(),          h.buf_size,                cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_off,      B.ml_off_bits.data(),  h.n_meshlets * 8,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_bnd,    B.ml_n_bnd.data(),     h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_int,    B.ml_n_int.data(),     h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_tris,   B.ml_n_tris.data(),    h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_strips, B.ml_n_strips.data(),  h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tris_off, B.ml_tris_off.data(),  (h.n_meshlets + 1) * 4,    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_int_cur,  B.ml_int_cursor.data(),h.n_meshlets * 4,          cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bnd_pos,  B.bnd_pos_norm.data(), h.n_boundary * 12,         cudaMemcpyHostToDevice));

    // Configure dynamic shared mem for both kernel variants (>48 KB needs opt-in)
    if (h.per_warp_bytes > 48 * 1024) {
        if (h.code_width == 0)
            CK(cudaFuncSetAttribute(paradelta_v5_fused_decode<short>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)h.per_warp_bytes));
        else
            CK(cudaFuncSetAttribute(paradelta_v5_fused_decode<int>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)h.per_warp_bytes));
    }

    // ---------------- Launch helper ----------------
    auto launch = [&]() {
        // Counter reset on GPU (no host roundtrip)
        cudaMemsetAsync(d_counter, 0, sizeof(int), 0);
        dim3 grid(n_blocks);
        dim3 block(32);
        size_t smem = h.per_warp_bytes;
        if (h.code_width == 0) {
            paradelta_v5_fused_decode<short><<<grid, block, smem>>>(
                d_buf, d_off, d_n_bnd, d_n_int, d_n_tris, d_n_strips,
                d_tris_off, (int)h.n_boundary, d_int_cur,
                d_bnd_pos, (int)h.n_boundary,
                h.lin3_w0, h.lin3_w1, h.lin3_w2, h.delta,
                d_verts, d_tris,
                (int)h.n_meshlets, (int)h.per_warp_bytes,
                (int)h.meshlets_per_block, d_counter);
        } else {
            paradelta_v5_fused_decode<int><<<grid, block, smem>>>(
                d_buf, d_off, d_n_bnd, d_n_int, d_n_tris, d_n_strips,
                d_tris_off, (int)h.n_boundary, d_int_cur,
                d_bnd_pos, (int)h.n_boundary,
                h.lin3_w0, h.lin3_w1, h.lin3_w2, h.delta,
                d_verts, d_tris,
                (int)h.n_meshlets, (int)h.per_warp_bytes,
                (int)h.meshlets_per_block, d_counter);
        }
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) launch();
    CK(cudaDeviceSynchronize());

    // Timed loop
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

    // Also single-launch timing (1 iteration event-bracketed) for direct
    // comparison against the amortized 100-iteration number.
    CK(cudaEventRecord(s));
    launch();
    CK(cudaEventRecord(e));
    CK(cudaEventSynchronize(e));
    float ms1 = 0.f;
    CK(cudaEventElapsedTime(&ms1, s, e));
    double single_us = (double)ms1 * 1000.0;

    // CSV-friendly single-line output + a human-readable line
    printf("%s,%u,%u,%u,%u,%.3f,%.3f,%.3f,%.1f,%d,%d\n",
           path, h.n_v_total, h.n_t_total, h.n_meshlets,
           h.code_width, single_us, per_us, mtps,
           (double)mtps, warmup, runs);
    fprintf(stderr,
            "%-48s n_v=%-8u n_t=%-8u n_m=%-5u  single=%7.1f us  "
            "amortized=%7.1f us  %7.1f Mtris/s\n",
            path, h.n_v_total, h.n_t_total, h.n_meshlets,
            single_us, per_us, mtps);

    CK(cudaFree(d_buf));      CK(cudaFree(d_off));
    CK(cudaFree(d_n_bnd));    CK(cudaFree(d_n_int));
    CK(cudaFree(d_n_tris));   CK(cudaFree(d_n_strips));
    CK(cudaFree(d_tris_off)); CK(cudaFree(d_int_cur));
    CK(cudaFree(d_bnd_pos));  CK(cudaFree(d_verts));
    CK(cudaFree(d_tris));     CK(cudaFree(d_counter));
    return 0;
}
// DGF C++/CUDA decode-throughput bench harness — faithful port of
// DGFLib::DecodeTriangleList + DecodeOffsetVerts to a CUDA kernel.
//
// Mirrors stride_decode_bench.cu's harness style (cudaEvent timing,
// warmup + amortized loop, single-launch line). Per-block work: one
// CUDA block per 128-byte DGF block, 32 threads (1 warp). Faithful to
// AMD HPG24 wave32 model: lane 0 serial-decodes the variable-length
// sections, then the warp parallel-writes float positions + uint32
// indices to global memory.
//
// Build (Windows, CUDA 12.x, sm_86):
//   nvcc -O3 -arch=sm_86 -std=c++17 dgf_decode_bench.cu
//        -o dgf_decode_bench.exe
//
// Run:
//   dgf_decode_bench.exe blobs/bunny.dgfblob [warmup=20] [runs=100]
//
// Blob file format (little-endian):
//   uint32 magic   = 0x44474642  ('DGFB')
//   uint32 version = 1
//   uint32 n_blocks
//   uint32 n_v_total
//   uint32 n_t_total
//   uint32 _resv0[3]
//   uint8  blocks[n_blocks * 128]
//   uint32 vert_offsets[n_blocks]    // running vertex prefix sum
//   uint32 tri_offsets[n_blocks]     // running triangle prefix sum

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int32_t  i32;
typedef uint64_t u64;

// ===========================================================================
// Device primitives — direct mirror of DGFLib::ReadBits.
// ===========================================================================

// Faithful ReadBits: reads `len` bits (len <= 48) starting at bit
// `start` from `bytes`, little-endian within byte. Used everywhere in
// the reference decoder. We deliberately keep this serial byte-by-byte
// to match the spec text, not micro-optimize.
__device__ __forceinline__ u64 dgf_read_bits(
    const u8* __restrict__ bytes, size_t start, size_t len)
{
    size_t firstByte = start / 8;
    size_t lastByte  = (start + (len - 1)) / 8;
    size_t numBytes  = 1 + lastByte - firstByte;
    u64 dst = 0;
    for (size_t i = 0; i < numBytes; ++i) {
        u64 b = bytes[firstByte + i];
        dst |= (b << (8 * i));
    }
    return (dst >> (start & 7)) & ((1ULL << len) - 1ULL);
}

// ===========================================================================
// DGF block header — bitfield layout from DGFLib::BlockHeader.
// We read raw little-endian dwords and unpack manually (host bitfields
// aren't portable to device).
// ===========================================================================

struct DGFMeta {
    i32 anchorX, anchorY, anchorZ;
    u8  exponent;
    u8  xBits, yBits, zBits;     // already +1
    u8  numTris, numVerts;       // already +1
    u8  bitsPerIndex;            // already +3
    u8  haveUserData;
    u8  ommDescCount;
    u8  geomIDMode;
    u32 geomIDMeta;
};

__device__ __forceinline__ i32 sign_extend_24(u32 v)
{
    return (v & 0x800000u) ? (i32)(v | 0xFF000000u) : (i32)v;
}

__device__ __forceinline__ void dgf_decode_meta(DGFMeta& m, const u8* block)
{
    // Read 5 LE dwords.
    u32 d0, d1, d2, d3, d4;
    memcpy(&d0, block +  0, 4);
    memcpy(&d1, block +  4, 4);
    memcpy(&d2, block +  8, 4);
    memcpy(&d3, block + 12, 4);
    memcpy(&d4, block + 16, 4);

    // DWORD 0: header_byte(8) | bits_per_index(2) | num_vertices(6) |
    //          num_triangles(6) | geom_id_meta(10).
    u32 bpi_raw     = (d0 >> 8)  & 0x3;
    u32 num_verts_m = (d0 >> 10) & 0x3F;
    u32 num_tris_m  = (d0 >> 16) & 0x3F;
    u32 geom_meta_lo= (d0 >> 22) & 0x3FF;

    // DWORD 1: exponent(8) | x_anchor(24 signed).
    m.exponent = (u8)(d1 & 0xFF);
    m.anchorX  = sign_extend_24((d1 >> 8) & 0xFFFFFF);

    // DWORD 2: x_bits(4) | y_bits(4) | y_anchor(24 signed).
    u32 x_bits_m = d2 & 0xF;
    u32 y_bits_m = (d2 >> 4) & 0xF;
    m.anchorY    = sign_extend_24((d2 >> 8) & 0xFFFFFF);

    // DWORD 3: z_bits(4) | omm_descriptor_count(3) | geom_id_mode(1) |
    //          z_anchor(24 signed).
    u32 z_bits_m = d3 & 0xF;
    m.ommDescCount = (u8)((d3 >> 4) & 0x7);
    m.geomIDMode   = (u8)((d3 >> 7) & 0x1);
    m.anchorZ      = sign_extend_24((d3 >> 8) & 0xFFFFFF);

    // DWORD 4: prim_id_base(29) | have_user_data(1) | unused(2).
    m.haveUserData = (u8)((d4 >> 29) & 0x1);

    m.bitsPerIndex = (u8)(bpi_raw + 3);
    m.numVerts     = (u8)(num_verts_m + 1);
    m.numTris      = (u8)(num_tris_m + 1);
    m.xBits        = (u8)(x_bits_m + 1);
    m.yBits        = (u8)(y_bits_m + 1);
    m.zBits        = (u8)(z_bits_m + 1);
    m.geomIDMeta   = geom_meta_lo;
}

// ===========================================================================
// Faithful CUDA decoder — one CUDA block per DGF block, 32 threads.
// Layout follows AMD HPG24 §4: lane 0 streams variable-length sections
// into SMEM, then the warp parallel-writes outputs to global.
// ===========================================================================

__global__ void dgf_decode_kernel(
    const u8*  __restrict__ blocks,         // n_blocks * 128 bytes
    const u32* __restrict__ vert_offsets,   // n_blocks: per-block vert base
    const u32* __restrict__ tri_offsets,    // n_blocks: per-block tri base
    int        n_blocks,
    float*     __restrict__ verts_out,      // [total_verts, 3]
    u32*       __restrict__ tris_out)       // [total_tris,  3]
{
    const int bi   = blockIdx.x;
    const int lane = threadIdx.x;
    if (bi >= n_blocks) return;

    const u8* block = blocks + (size_t)bi * 128;

    // -------- SMEM: decoded per-block intermediates --------
    __shared__ DGFMeta meta;
    __shared__ u16 verts_smem[64 * 3];      // OffsetVert::xyz
    __shared__ u8  ctrl_smem[64];           // TriControlValues
    __shared__ u8  idx_smem[3 * 64];        // strip index buffer (numStored + 3 ≤ 3*MAX_TRIS)
    __shared__ u8  tris_smem[64 * 3];       // expanded tri list (3 idx/tri)
    __shared__ u32 num_stored_indices_smem;
    __shared__ u32 user_data_bits_smem;     // 0 or 32
    __shared__ u32 vertex_data_bits_smem;
    __shared__ u32 omm_palette_bits_smem;
    __shared__ u32 geom_palette_bits_smem;

    // -------- Phase 1: header parse (lane 0) --------
    if (lane == 0) {
        dgf_decode_meta(meta, block);
        user_data_bits_smem = meta.haveUserData ? 32u : 0u;

        u32 bpv = (u32)meta.xBits + meta.yBits + meta.zBits;
        u32 vbits = bpv * (u32)meta.numVerts;
        vertex_data_bits_smem = (vbits + 7u) & ~7u;

        // OMM palette size.
        u32 omm_n = meta.ommDescCount;
        if (omm_n == 0) {
            omm_palette_bits_smem = 0;
        } else {
            u32 hot_dwords = 2 + omm_n;
            u32 idx_sz = (omm_n <= 1) ? 0 :
                         (omm_n == 2) ? 1 :
                         (omm_n <= 4) ? 2 : 3;
            u32 sz = 32u * hot_dwords + (((idx_sz * meta.numTris) + 7u) & ~7u);
            omm_palette_bits_smem = sz;
        }

        // GeomID palette size.
        if (meta.geomIDMode == 0) {
            geom_palette_bits_smem = 0;
        } else {
            u32 numIDs = (meta.geomIDMeta >> 5) + 1;
            u32 prefixBitSize = meta.geomIDMeta & 0x1f;
            u32 payloadBitSize = (24u + 1u) - prefixBitSize;
            // BitsNeeded(numIDs - 1)
            u32 v = numIDs - 1;
            u32 bn = 0;
            while (v) { bn++; v >>= 1; }
            u32 sz = numIDs * payloadBitSize + (u32)meta.numTris * bn + prefixBitSize;
            sz = (sz + 7u) & ~7u;
            geom_palette_bits_smem = sz;
        }
    }
    __syncwarp();

    const u32 numTris  = meta.numTris;
    const u32 numVerts = meta.numVerts;

    // -------- Phase 2: topology decode (lane 0, faithful sequential) --------
    if (lane == 0) {
        ctrl_smem[0] = 0;  // TC_RESTART
        u32 numStored = 0;
        // Control bits packed back-to-front, 2 bits each.
        for (u32 i = 1; i < numTris; ++i) {
            u64 ctrl = dgf_read_bits(block, 1024 - 2 * i, 2);
            ctrl_smem[i] = (u8)ctrl;
            numStored += (ctrl == 0) ? 3 : 1;
        }
        num_stored_indices_smem = numStored;

        // "is-first" bits live just before the control bits.
        u32 isFirstBitPos = 1024 - 2 * (numTris - 1) - 1;

        // Index buffer starts after header + user data + front buffer.
        u32 frontBufBits = vertex_data_bits_smem +
                           omm_palette_bits_smem +
                           geom_palette_bits_smem;
        u32 indexBitPos  = 8u * 20u + user_data_bits_smem + frontBufBits;

        idx_smem[0] = 0;
        idx_smem[1] = 1;
        idx_smem[2] = 2;
        u32 vertexCounter = 3;
        u32 bpi = meta.bitsPerIndex;
        for (u32 i = 0; i < numStored; ++i) {
            bool isFirst = (dgf_read_bits(block, isFirstBitPos - i, 1) != 0);
            u32 val;
            if (isFirst) {
                val = vertexCounter++;
            } else {
                val = (u32)dgf_read_bits(block, indexBitPos, bpi);
                indexBitPos += bpi;
            }
            idx_smem[i + 3] = (u8)val;
        }
    }
    __syncwarp();

    // -------- Phase 3: walk strip → expanded tri list (lane 0) --------
    // ConvertTopologyToTriangleList — inherently sequential per block
    // because each BACKTRACK references the previous tri's previous tri.
    if (lane == 0) {
        u32 indexPos = 0;
        u32 prev[3]     = {0, 0, 0};
        u32 prevPrev[3] = {0, 0, 0};
        for (u32 i = 0; i < numTris; ++i) {
            u32 v[3] = {0, 0, 0};
            u8 c = ctrl_smem[i];
            switch (c) {
            case 0: // TC_RESTART
                v[0] = indexPos++;
                v[1] = indexPos++;
                v[2] = indexPos++;
                break;
            case 1: // TC_EDGE1: 1,2 of prev
                v[0] = prev[2];
                v[1] = prev[1];
                v[2] = indexPos++;
                break;
            case 2: // TC_EDGE2: 2,0 of prev
                v[0] = prev[0];
                v[1] = prev[2];
                v[2] = indexPos++;
                break;
            case 3: // TC_BACKTRACK
                if (ctrl_smem[i - 1] == 1) {
                    v[0] = prevPrev[0];
                    v[1] = prevPrev[2];
                } else { // EDGE2
                    v[0] = prevPrev[2];
                    v[1] = prevPrev[1];
                }
                v[2] = indexPos++;
                break;
            }
            tris_smem[3 * i + 0] = idx_smem[v[0]];
            tris_smem[3 * i + 1] = idx_smem[v[1]];
            tris_smem[3 * i + 2] = idx_smem[v[2]];
            for (int j = 0; j < 3; ++j) {
                prevPrev[j] = prev[j];
                prev[j]     = v[j];
            }
        }
    }
    __syncwarp();

    // -------- Phase 4: vertex offsets (lane 0 reads packed bpv bits) ------
    if (lane == 0) {
        const u8* vertexData = block + 20 + (user_data_bits_smem / 8);
        u32 xb = meta.xBits, yb = meta.yBits, zb = meta.zBits;
        u32 bpv = xb + yb + zb;
        for (u32 i = 0; i < numVerts; ++i) {
            u64 packed = dgf_read_bits(vertexData, i * bpv, bpv);
            u32 x = (u32)(packed & ((1ULL << xb) - 1ULL));
            u32 y = (u32)((packed >> xb) & ((1ULL << yb) - 1ULL));
            u32 z = (u32)((packed >> (xb + yb)) & ((1ULL << zb) - 1ULL));
            verts_smem[3 * i + 0] = (u16)x;
            verts_smem[3 * i + 1] = (u16)y;
            verts_smem[3 * i + 2] = (u16)z;
        }
    }
    __syncwarp();

    // -------- Phase 5: parallel emit — verts dequant + tri indices --------
    // float = (anchor + offset) * 2^(exponent - 127).
    // Match DGFLib::ConvertOffsetsToFloat exactly: cast to float, add as
    // float, multiply by ldexp scale.
    const float scale = ldexpf(1.0f, (int)meta.exponent - 127);
    const i32 ax = meta.anchorX, ay = meta.anchorY, az = meta.anchorZ;
    const u32 v_base = vert_offsets[bi];
    const u32 t_base = tri_offsets[bi];

    for (u32 i = lane; i < numVerts; i += 32) {
        u32 ox = verts_smem[3 * i + 0];
        u32 oy = verts_smem[3 * i + 1];
        u32 oz = verts_smem[3 * i + 2];
        float fx = (float)((i32)ox + ax) * scale;
        float fy = (float)((i32)oy + ay) * scale;
        float fz = (float)((i32)oz + az) * scale;
        verts_out[3 * (v_base + i) + 0] = fx;
        verts_out[3 * (v_base + i) + 1] = fy;
        verts_out[3 * (v_base + i) + 2] = fz;
    }

    for (u32 t = lane; t < numTris; t += 32) {
        u32 i0 = tris_smem[3 * t + 0];
        u32 i1 = tris_smem[3 * t + 1];
        u32 i2 = tris_smem[3 * t + 2];
        tris_out[3 * (t_base + t) + 0] = v_base + i0;
        tris_out[3 * (t_base + t) + 1] = v_base + i1;
        tris_out[3 * (t_base + t) + 2] = v_base + i2;
    }
}

// ===========================================================================
// Host harness — mirrors stride_decode_bench.cu shape.
// ===========================================================================

#define CK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); std::exit(1); \
    } \
} while (0)

struct DGFBlobHeader {
    u32 magic;          // 'DGFB' = 0x42464744 LE? actually 'DGFB' in
                        // C-string order: D=44 G=47 F=46 B=42 → as
                        // u32 LE = 0x42464744. We use this value.
    u32 version;
    u32 n_blocks;
    u32 n_v_total;
    u32 n_t_total;
    u32 _resv[3];
};
static_assert(sizeof(DGFBlobHeader) == 32, "header size must be 32");

struct DGFBuffers {
    DGFBlobHeader hdr;
    std::vector<u8>  blocks;
    std::vector<u32> vert_offsets;
    std::vector<u32> tri_offsets;
};

static bool read_blob(const std::string& path, DGFBuffers& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "open %s\n", path.c_str()); return false; }
    f.read((char*)&out.hdr, sizeof(out.hdr));
    if (out.hdr.magic != 0x42464744u) {
        fprintf(stderr, "bad magic 0x%08x\n", out.hdr.magic);
        return false;
    }
    const auto& h = out.hdr;
    out.blocks.resize((size_t)h.n_blocks * 128);
    f.read((char*)out.blocks.data(), out.blocks.size());
    out.vert_offsets.resize(h.n_blocks);
    f.read((char*)out.vert_offsets.data(), h.n_blocks * 4);
    out.tri_offsets.resize(h.n_blocks);
    f.read((char*)out.tri_offsets.data(), h.n_blocks * 4);
    return f.good() || f.eof();
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("usage: %s <blob> [warmup=20] [runs=100] [--dump <prefix>]\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int warmup = 20;
    int runs   = 100;
    const char* dump_prefix = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_prefix = argv[++i];
        } else if (argv[i][0] != '-') {
            if (i == 2) warmup = std::atoi(argv[i]);
            else if (i == 3) runs = std::atoi(argv[i]);
        }
    }

    DGFBuffers B;
    if (!read_blob(path, B)) return 1;
    const auto& h = B.hdr;

    int dev = 0;
    CK(cudaSetDevice(dev));

    // Upload — one-shot, outside timed loop.
    u8*  d_blocks = nullptr;
    u32* d_voff   = nullptr;
    u32* d_toff   = nullptr;
    float* d_verts = nullptr;
    u32*   d_tris  = nullptr;

    CK(cudaMalloc(&d_blocks, B.blocks.size()));
    CK(cudaMalloc(&d_voff,   h.n_blocks * sizeof(u32)));
    CK(cudaMalloc(&d_toff,   h.n_blocks * sizeof(u32)));
    CK(cudaMalloc(&d_verts,  (size_t)h.n_v_total * 3 * sizeof(float)));
    CK(cudaMalloc(&d_tris,   (size_t)h.n_t_total * 3 * sizeof(u32)));

    CK(cudaMemcpy(d_blocks, B.blocks.data(),       B.blocks.size(),         cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_voff,   B.vert_offsets.data(), h.n_blocks * sizeof(u32),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_toff,   B.tri_offsets.data(),  h.n_blocks * sizeof(u32),cudaMemcpyHostToDevice));

    auto launch = [&]() {
        dim3 grid(h.n_blocks);
        dim3 block(32);
        dgf_decode_kernel<<<grid, block>>>(
            d_blocks, d_voff, d_toff, (int)h.n_blocks,
            d_verts, d_tris);
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
           path, h.n_v_total, h.n_t_total, h.n_blocks,
           single_us, per_us, mtps, warmup, runs);
    fprintf(stderr,
            "%-48s n_v=%-8u n_t=%-8u n_blk=%-6u  single=%7.1f us  "
            "amortized=%7.1f us  %7.1f Mtris/s\n",
            path, h.n_v_total, h.n_t_total, h.n_blocks,
            single_us, per_us, mtps);

    // Optional dump: write decoded verts + tris to disk for parity check.
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

    CK(cudaFree(d_blocks));  CK(cudaFree(d_voff)); CK(cudaFree(d_toff));
    CK(cudaFree(d_verts));   CK(cudaFree(d_tris));
    return 0;
}

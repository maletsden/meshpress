"""
CUDA benchmark: AMD vs EdgeBreaker meshlet decode + wavelet inverse.
1 CUDA block = 1 meshlet. Phase 1: connectivity (1 thread). Phase 2: wavelet (all threads).
"""

import numpy as np
import cupy as cp
import time
from reader import Reader
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts, edgebreaker_vertex_order, meshlet_bfs,
)
from utils.wavelet import haar_decompose
from utils.connectivity import VertexFIFO

CUDA_SRC = r'''
extern "C" {

#define MAX_VERTS 256
#define MAX_FIFO 32

// AMD FIFO decode (single thread): reads [type, edge, info] entries
__device__ void sim_amd(const int* ent, int n, int* out, int* cnt) {
    int fifo[MAX_FIFO], fc = 0, vc = 0, nd = 0;
    for (int e = 0; e < n; e++) {
        int t = ent[e*3], info = ent[e*3+2], nv;
        if (t == 0) { nv = vc++; }
        else if (t == 1) { int i = fc-1-info; nv = (i>=0&&i<fc) ? fifo[i] : 0; }
        else { nv = info; }
        out[nd++] = nv;
        // FIFO push with evict
        for (int i=0;i<fc;i++) if(fifo[i]==nv){for(int j=i;j<fc-1;j++)fifo[j]=fifo[j+1];fc--;break;}
        if(fc>=MAX_FIFO){for(int j=0;j<fc-1;j++)fifo[j]=fifo[j+1];fc--;}
        fifo[fc++]=nv;
    }
    *cnt = nd;
}

// EdgeBreaker decode (single thread): reads CLERS opcodes
__device__ void sim_eb(const int* ops, int n, int* out, int* cnt) {
    int bnd[MAX_VERTS], bs=3, vc=3, nd=0, g=0;
    bnd[0]=0; bnd[1]=1; bnd[2]=2;
    for (int i=0; i<n; i++) {
        int gv0=bnd[g%bs], gv1=bnd[(g+1)%bs], nv;
        if (ops[i]==0) { nv=vc++; if(bs<MAX_VERTS){for(int j=bs;j>(g+1)%bs+1;j--)bnd[j]=bnd[j-1];bnd[(g+1)%bs+1]=nv;bs++;}g=(g+1)%bs; }
        else if (ops[i]==1) { nv=bnd[(g+bs-1)%bs]; if(bs>3)bs--; g=g>0?g-1:bs-1; }
        else if (ops[i]==2) { nv=bnd[(g+2)%bs]; if(bs>3)bs--; }
        else { nv=gv0; }
        out[nd*3]=gv0; out[nd*3+1]=gv1; out[nd*3+2]=nv; nd++;
    }
    *cnt = nd;
}

// Wavelet inverse (all threads in block)
__device__ void inv_wav(float* ch, const float* gdata, int doff, int bsz, int nlvl, int nv) {
    int tid = threadIdx.x;
    __shared__ float tmp[MAX_VERTS];
    int csz = bsz;
    for (int l = nlvl-1; l >= 0; l--) {
        __syncthreads();
        if (tid < csz && csz+tid < MAX_VERTS)
            ch[csz+tid] = ch[tid] + gdata[doff+tid];
        __syncthreads();
        if (tid < csz*2 && tid < nv)
            tmp[tid] = ch[(tid&1) ? csz+tid/2 : tid/2];
        __syncthreads();
        if (tid < csz*2 && tid < nv)
            ch[tid] = tmp[tid];
        __syncthreads();
        doff += csz;
        csz *= 2;
    }
}

// ---- Kernel 1: Wavelet only (baseline) ----
__global__ void k_wavelet(
    const float* wdata, const int* woff, const int* wbsz, const int* wnlvl,
    const int* nverts, float* out, const int* voff, int nmesh
) {
    int m=blockIdx.x; if(m>=nmesh) return;
    int tid=threadIdx.x, nv=nverts[m], wo=woff[m], bs=wbsz[m], nl=wnlvl[m];
    __shared__ float sv[MAX_VERTS*3];
    for(int c=0;c<3;c++) if(tid<bs) sv[c*MAX_VERTS+tid]=wdata[wo+c*nv+tid];
    __syncthreads();
    for(int c=0;c<3;c++) inv_wav(&sv[c*MAX_VERTS], wdata, wo+c*nv+bs, bs, nl, nv);
    __syncthreads();
    int vo=voff[m];
    if(tid<nv){out[(vo+tid)*3]=sv[tid];out[(vo+tid)*3+1]=sv[MAX_VERTS+tid];out[(vo+tid)*3+2]=sv[2*MAX_VERTS+tid];}
}

// ---- Kernel 2: AMD + Wavelet ----
__global__ void k_amd_wav(
    const int* cent, const int* coff, const int* csz,
    const float* wdata, const int* woff, const int* wbsz, const int* wnlvl,
    const int* nverts, float* out, const int* voff, int nmesh
) {
    int m=blockIdx.x; if(m>=nmesh) return;
    int tid=threadIdx.x, nv=nverts[m];
    __shared__ float sv[MAX_VERTS*3];
    __shared__ int st[MAX_VERTS];
    // Phase 1: AMD decode (thread 0)
    if(tid==0){int cnt; sim_amd(&cent[coff[m]], csz[m], st, &cnt);}
    __syncthreads();
    // Phase 2: Wavelet
    int wo=woff[m],bs=wbsz[m],nl=wnlvl[m];
    for(int c=0;c<3;c++) if(tid<bs) sv[c*MAX_VERTS+tid]=wdata[wo+c*nv+tid];
    __syncthreads();
    for(int c=0;c<3;c++) inv_wav(&sv[c*MAX_VERTS], wdata, wo+c*nv+bs, bs, nl, nv);
    __syncthreads();
    int vo=voff[m];
    if(tid<nv){out[(vo+tid)*3]=sv[tid];out[(vo+tid)*3+1]=sv[MAX_VERTS+tid];out[(vo+tid)*3+2]=sv[2*MAX_VERTS+tid];}
}

// ---- Kernel 3: EdgeBreaker + Wavelet ----
__global__ void k_eb_wav(
    const int* ops, const int* ooff, const int* osz,
    const float* wdata, const int* woff, const int* wbsz, const int* wnlvl,
    const int* nverts, float* out, const int* voff, int nmesh
) {
    int m=blockIdx.x; if(m>=nmesh) return;
    int tid=threadIdx.x, nv=nverts[m];
    __shared__ float sv[MAX_VERTS*3];
    __shared__ int st[MAX_VERTS*3];
    // Phase 1: EdgeBreaker decode (thread 0)
    if(tid==0){int cnt; sim_eb(&ops[ooff[m]], osz[m], st, &cnt);}
    __syncthreads();
    // Phase 2: Wavelet
    int wo=woff[m],bs=wbsz[m],nl=wnlvl[m];
    for(int c=0;c<3;c++) if(tid<bs) sv[c*MAX_VERTS+tid]=wdata[wo+c*nv+tid];
    __syncthreads();
    for(int c=0;c<3;c++) inv_wav(&sv[c*MAX_VERTS], wdata, wo+c*nv+bs, bs, nl, nv);
    __syncthreads();
    int vo=voff[m];
    if(tid<nv){out[(vo+tid)*3]=sv[tid];out[(vo+tid)*3+1]=sv[MAX_VERTS+tid];out[(vo+tid)*3+2]=sv[2*MAX_VERTS+tid];}
}

// ---- Kernel: Global Int Wavelet + AMD Packed (Solution 1, crack-free) ----
// Integer wavelet coefficients → inverse wavelet (int) → global dequantize → float positions
__global__ void k_global_int_wav_amd(
    const unsigned char* tri_indices, const int* tri_offsets, const int* tri_counts,
    const int* wav_int_data,       // packed integer wavelet coefficients per meshlet
    const int* wav_offsets,
    const int* wav_base_sizes,
    const int* wav_n_levels,
    const float* global_dequant,   // [min_x, range_x, bits_x, min_y, range_y, bits_y, min_z, range_z, bits_z]
    const int* nverts,
    float* out, const int* voff,
    int* out_tris, const int* toff,
    int nmesh
) {
    int m = blockIdx.x;
    if (m >= nmesh) return;
    int tid = threadIdx.x;
    int nv = nverts[m];
    int ntri = tri_counts[m];

    __shared__ int si[MAX_VERTS * 3];    // integer vertex coords
    __shared__ int tmp_i[MAX_VERTS];
    __shared__ float sv[MAX_VERTS * 3];  // final float positions

    // Phase 1: Triangle decode (parallel)
    int t_base = tri_offsets[m];
    int t_out = toff[m];
    for (int t = tid; t < ntri; t += blockDim.x) {
        int off = (t_base + t) * 3;
        out_tris[(t_out + t) * 3 + 0] = tri_indices[off];
        out_tris[(t_out + t) * 3 + 1] = tri_indices[off + 1];
        out_tris[(t_out + t) * 3 + 2] = tri_indices[off + 2];
    }

    // Phase 2: Integer wavelet inverse (all threads, per channel)
    int wo = wav_offsets[m];
    int bsz = wav_base_sizes[m];
    int nlvl = wav_n_levels[m];

    for (int ch = 0; ch < 3; ch++) {
        // Load base into shared memory
        if (tid < bsz)
            si[ch * MAX_VERTS + tid] = wav_int_data[wo + ch * nv + tid];
        __syncthreads();

        int* channel = &si[ch * MAX_VERTS];
        int csz = bsz;
        int doff = wo + ch * nv + bsz;

        for (int l = nlvl - 1; l >= 0; l--) {
            __syncthreads();
            if (tid < csz && csz + tid < MAX_VERTS)
                channel[csz + tid] = channel[tid] + wav_int_data[doff + tid];
            __syncthreads();
            if (tid < csz * 2 && tid < nv)
                tmp_i[tid] = channel[(tid & 1) ? csz + tid/2 : tid/2];
            __syncthreads();
            if (tid < csz * 2 && tid < nv)
                channel[tid] = tmp_i[tid];
            __syncthreads();
            doff += csz;
            csz *= 2;
        }
    }
    __syncthreads();

    // Phase 3: Global dequantize (parallel)
    int vo = voff[m];
    if (tid < nv) {
        for (int ch = 0; ch < 3; ch++) {
            float mn = global_dequant[ch * 3 + 0];
            float rng = global_dequant[ch * 3 + 1];
            float mx_val = global_dequant[ch * 3 + 2];  // (1 << bits) - 1
            float val = (float)si[ch * MAX_VERTS + tid] / mx_val * rng + mn;
            out[(vo + tid) * 3 + ch] = val;
        }
    }
}

// ---- Kernel: S1 OPT v1 (warp sync + fused dequant + single buffer) ----
__global__ void k_s1_optimized(
    const unsigned int* packed_tris_u32, // triangles packed as uint32 (3 × uint8 in low 24 bits)
    const int* tri_offsets, const int* tri_counts,
    const int* wav_int_data, const int* wav_offsets,
    const int* wav_base_sizes, const int* wav_n_levels,
    const float* global_dequant,
    const int* nverts,
    float* out, const int* voff,
    int* out_tris, const int* toff,
    int nmesh
) {
    int m = blockIdx.x;
    if (m >= nmesh) return;
    int tid = threadIdx.x;
    int nv = nverts[m];
    int ntri = tri_counts[m];

    // Single shared buffer — process 1 channel at a time (saves 2KB shared mem → better occupancy)
    __shared__ int sd[MAX_VERTS];

    int wo = wav_offsets[m];
    int bsz = wav_base_sizes[m];
    int nlvl = wav_n_levels[m];
    int vo = voff[m];

    // Phase 1: Triangle decode (uint32 packed — 1 aligned read per tri)
    int t_base = tri_offsets[m];
    int t_out = toff[m];
    for (int t = tid; t < ntri; t += blockDim.x) {
        unsigned int p = packed_tris_u32[t_base + t];
        out_tris[(t_out + t) * 3 + 0] = p & 0xFF;
        out_tris[(t_out + t) * 3 + 1] = (p >> 8) & 0xFF;
        out_tris[(t_out + t) * 3 + 2] = (p >> 16) & 0xFF;
    }

    // Phase 2+3: Per-channel integer wavelet inverse + fused dequantize
    for (int ch = 0; ch < 3; ch++) {
        float dq_mn = global_dequant[ch * 3 + 0];
        float dq_rng = global_dequant[ch * 3 + 1];
        float dq_mx = global_dequant[ch * 3 + 2];

        // Load base values into shared
        if (tid < bsz)
            sd[tid] = wav_int_data[wo + ch * nv + tid];
        __syncthreads();

        int csz = bsz;
        int doff = wo + ch * nv + bsz;

        for (int l = nlvl - 1; l >= 0; l--) {
            // Step A: Expand — add residuals from global memory
            if (tid < csz && csz + tid < MAX_VERTS)
                sd[csz + tid] = sd[tid] + wav_int_data[doff + tid];
            __syncthreads();

            int new_size = csz * 2;

            // Step B: Interleave [e0,e1,...,o0,o1,...] → [e0,o0,e1,o1,...]
            if (new_size <= 32) {
                // WARP-LEVEL: no __syncthreads needed, use register exchange
                // All participating threads are in warp 0
                if (tid < new_size) {
                    int src = (tid & 1) ? (csz + tid / 2) : (tid / 2);
                    int val = sd[src];  // within-warp coherent read
                    __syncwarp(0xFFFFFFFF);
                    sd[tid] = val;
                    __syncwarp(0xFFFFFFFF);
                }
            } else {
                // Cross-warp: must use temp storage
                // Optimization: read into registers, sync, write back (avoids temp array)
                int my_val = 0;
                if (tid < new_size && tid < nv) {
                    int src = (tid & 1) ? (csz + tid / 2) : (tid / 2);
                    my_val = sd[src];
                }
                __syncthreads();
                if (tid < new_size && tid < nv)
                    sd[tid] = my_val;
                __syncthreads();
            }

            doff += csz;
            csz = new_size;
        }

        // Fused dequantize + write (no extra syncthreads for separate dequant phase)
        if (tid < nv) {
            out[(vo + tid) * 3 + ch] = (float)sd[tid] / dq_mx * dq_rng + dq_mn;
        }
        __syncthreads(); // sync before reusing sd for next channel
    }
}

// ---- Kernel: S1 OPT v2 (no syncwarp, 3ch parallel, __ldg) ----
// Opt 1: Remove __syncwarp (implicit on SM 7.0+)
// Opt 2: Process 3 channels in parallel — thread tid handles vertex (tid%nv), channel (tid/nv)
//         using 256 threads for max(nv,256) work items across 3 channels
// Opt 3: Use __ldg() for global memory reads
__global__ void k_s1_opt_v2(
    const unsigned int* packed_tris_u32,
    const int* tri_offsets, const int* tri_counts,
    const int* __restrict__ wav_int_data, const int* wav_offsets,
    const int* wav_base_sizes, const int* wav_n_levels,
    const float* __restrict__ global_dequant,
    const int* nverts,
    float* __restrict__ out, const int* voff,
    int* __restrict__ out_tris, const int* toff,
    int nmesh
) {
    int m = blockIdx.x;
    if (m >= nmesh) return;
    int tid = threadIdx.x;
    int nv = nverts[m];
    int ntri = tri_counts[m];

    __shared__ int sd[MAX_VERTS];

    int wo = wav_offsets[m];
    int bsz = wav_base_sizes[m];
    int nlvl = wav_n_levels[m];
    int vo = voff[m];

    // Phase 1: Triangle decode (uint32 packed, __ldg)
    int t_base = tri_offsets[m];
    int t_out = toff[m];
    for (int t = tid; t < ntri; t += blockDim.x) {
        unsigned int p = __ldg(&packed_tris_u32[t_base + t]);
        out_tris[(t_out + t) * 3 + 0] = p & 0xFF;
        out_tris[(t_out + t) * 3 + 1] = (p >> 8) & 0xFF;
        out_tris[(t_out + t) * 3 + 2] = (p >> 16) & 0xFF;
    }

    // Phase 2+3: Per-channel wavelet + dequant (sequential channels, optimized per-level)
    for (int ch = 0; ch < 3; ch++) {
        float dq_mn = __ldg(&global_dequant[ch * 3 + 0]);
        float dq_rng = __ldg(&global_dequant[ch * 3 + 1]);
        float dq_mx = __ldg(&global_dequant[ch * 3 + 2]);

        // Load base values with __ldg
        if (tid < bsz)
            sd[tid] = __ldg(&wav_int_data[wo + ch * nv + tid]);
        __syncthreads();

        int csz = bsz;
        int doff = wo + ch * nv + bsz;

        for (int l = nlvl - 1; l >= 0; l--) {
            // Expand: add residuals (use __ldg for global read)
            if (tid < csz && csz + tid < MAX_VERTS)
                sd[csz + tid] = sd[tid] + __ldg(&wav_int_data[doff + tid]);
            __syncthreads();

            int new_size = csz * 2;

            if (new_size <= 32) {
                // Warp-level: no syncwarp needed on SM 7.0+ (implicit warp sync)
                if (tid < new_size) {
                    int src = (tid & 1) ? (csz + tid / 2) : (tid / 2);
                    int val = sd[src];
                    // Implicit warp-level memory visibility on SM 7.0+
                    sd[tid] = val;
                }
                // No __syncwarp, no __syncthreads for intra-warp shared mem ops
            } else {
                // Cross-warp: register-based interleave
                int my_val = 0;
                if (tid < new_size && tid < nv)
                    my_val = sd[(tid & 1) ? (csz + tid / 2) : (tid / 2)];
                __syncthreads();
                if (tid < new_size && tid < nv)
                    sd[tid] = my_val;
                __syncthreads();
            }

            doff += csz;
            csz = new_size;
        }

        // Fused dequantize + write
        if (tid < nv)
            out[(vo + tid) * 3 + ch] = (float)sd[tid] / dq_mx * dq_rng + dq_mn;
        __syncthreads();
    }
}

// ---- Kernel: S1 OPT v3 (3 channels PARALLEL — 3 warps per channel) ----
// Each warp handles one channel independently. No cross-channel sync needed.
// Requires nv <= 256 (each warp processes up to 256/3 ≈ 85 values... too few)
// Better: split 256 threads into 3 groups of ~85 for 3 channels
// Actually: just use first 3 warps (96 threads) for 3 channels (32 threads each)
// and remaining threads idle. Only works well if base_size <= 32.
// SIMPLER: interleave channel work across ALL threads.
// Thread tid processes vertex (tid) for channel (0,1,2) sequentially but
// loads all 3 base values at once.
// Actually the simplest parallel approach: 3× data in shared, 3× the work.
__global__ void k_s1_opt_v3(
    const unsigned int* packed_tris_u32,
    const int* tri_offsets, const int* tri_counts,
    const int* __restrict__ wav_int_data, const int* wav_offsets,
    const int* wav_base_sizes, const int* wav_n_levels,
    const float* __restrict__ global_dequant,
    const int* nverts,
    float* __restrict__ out, const int* voff,
    int* __restrict__ out_tris, const int* toff,
    int nmesh
) {
    int m = blockIdx.x;
    if (m >= nmesh) return;
    int tid = threadIdx.x;
    int nv = nverts[m];
    int ntri = tri_counts[m];

    // 3 channels in parallel: sd[0..255] = ch0, sd[256..511] = ch1, sd[512..767] = ch2
    // Requires 3 × MAX_VERTS = 3KB shared memory
    __shared__ int sd[MAX_VERTS * 3];

    int wo = wav_offsets[m];
    int bsz = wav_base_sizes[m];
    int nlvl = wav_n_levels[m];
    int vo = voff[m];

    // Phase 1: Triangles
    int t_base = tri_offsets[m];
    int t_out = toff[m];
    for (int t = tid; t < ntri; t += blockDim.x) {
        unsigned int p = __ldg(&packed_tris_u32[t_base + t]);
        out_tris[(t_out + t) * 3 + 0] = p & 0xFF;
        out_tris[(t_out + t) * 3 + 1] = (p >> 8) & 0xFF;
        out_tris[(t_out + t) * 3 + 2] = (p >> 16) & 0xFF;
    }

    // Phase 2: Load ALL 3 channels' base values simultaneously
    for (int ch = 0; ch < 3; ch++) {
        if (tid < bsz)
            sd[ch * MAX_VERTS + tid] = __ldg(&wav_int_data[wo + ch * nv + tid]);
    }
    __syncthreads();

    // Phase 3: Wavelet inverse — all 3 channels at each level, single sync per step
    int csz = bsz;
    int doff0 = wo + 0 * nv + bsz;
    int doff1 = wo + 1 * nv + bsz;
    int doff2 = wo + 2 * nv + bsz;

    for (int l = nlvl - 1; l >= 0; l--) {
        // Expand all 3 channels (1 sync instead of 3)
        if (tid < csz && csz + tid < MAX_VERTS) {
            sd[0 * MAX_VERTS + csz + tid] = sd[0 * MAX_VERTS + tid] + __ldg(&wav_int_data[doff0 + tid]);
            sd[1 * MAX_VERTS + csz + tid] = sd[1 * MAX_VERTS + tid] + __ldg(&wav_int_data[doff1 + tid]);
            sd[2 * MAX_VERTS + csz + tid] = sd[2 * MAX_VERTS + tid] + __ldg(&wav_int_data[doff2 + tid]);
        }
        __syncthreads();

        int new_size = csz * 2;

        // Interleave all 3 channels (1 sync instead of 3)
        int my_v0 = 0, my_v1 = 0, my_v2 = 0;
        if (tid < new_size && tid < nv) {
            int src = (tid & 1) ? (csz + tid / 2) : (tid / 2);
            my_v0 = sd[0 * MAX_VERTS + src];
            my_v1 = sd[1 * MAX_VERTS + src];
            my_v2 = sd[2 * MAX_VERTS + src];
        }
        __syncthreads();
        if (tid < new_size && tid < nv) {
            sd[0 * MAX_VERTS + tid] = my_v0;
            sd[1 * MAX_VERTS + tid] = my_v1;
            sd[2 * MAX_VERTS + tid] = my_v2;
        }
        __syncthreads();

        doff0 += csz; doff1 += csz; doff2 += csz;
        csz = new_size;
    }

    // Phase 4: Fused dequantize + write (all 3 channels at once)
    if (tid < nv) {
        float mn0 = __ldg(&global_dequant[0]), rng0 = __ldg(&global_dequant[1]), mx0 = __ldg(&global_dequant[2]);
        float mn1 = __ldg(&global_dequant[3]), rng1 = __ldg(&global_dequant[4]), mx1 = __ldg(&global_dequant[5]);
        float mn2 = __ldg(&global_dequant[6]), rng2 = __ldg(&global_dequant[7]), mx2 = __ldg(&global_dequant[8]);
        int base_out = (vo + tid) * 3;
        out[base_out + 0] = (float)sd[0 * MAX_VERTS + tid] / mx0 * rng0 + mn0;
        out[base_out + 1] = (float)sd[1 * MAX_VERTS + tid] / mx1 * rng1 + mn1;
        out[base_out + 2] = (float)sd[2 * MAX_VERTS + tid] / mx2 * rng2 + mn2;
    }
}

// ---- Kernel: AMD Baseline (packed quantized verts, actual bit depth) ----
__global__ void k_amd_quant(
    const unsigned char* tri_indices, const int* tri_offsets, const int* tri_counts,
    const unsigned int* packed_verts,  // 1 × uint32 per vertex: bits packed xyz
    const int* pv_offsets,
    const float* dequant_params,       // global: [min_x, dq_factor_x, min_y, dq_factor_y, min_z, dq_factor_z]
    const int* meshlet_offsets_q,      // per-meshlet: [offset_x, offset_y, offset_z] as uint32
    const int* bits_per_axis,          // [bits_x, bits_y, bits_z]
    const int* nverts,
    float* out, const int* voff,
    int* out_tris, const int* toff,
    int nmesh
) {
    int m = blockIdx.x;
    if (m >= nmesh) return;
    int tid = threadIdx.x;
    int nv = nverts[m];
    int ntri = tri_counts[m];

    // Load dequant params + meshlet offsets into shared
    __shared__ float s_dq[6];    // min_x, factor_x, min_y, factor_y, min_z, factor_z
    __shared__ int s_off[3];     // quantized meshlet offset per axis
    __shared__ int s_bits[3];    // bits per axis
    if (tid < 6) s_dq[tid] = dequant_params[tid];
    if (tid < 3) { s_off[tid] = meshlet_offsets_q[m * 3 + tid]; s_bits[tid] = bits_per_axis[tid]; }
    __syncthreads();

    int bx = s_bits[0], by = s_bits[1], bz = s_bits[2];
    int mask_x = (1 << bx) - 1;
    int mask_y = (1 << by) - 1;
    int mask_z = (1 << bz) - 1;

    // Phase 1: Unpack + dequantize vertices (parallel)
    int pbase = pv_offsets[m];
    int vo = voff[m];
    if (tid < nv) {
        unsigned int packed = packed_verts[pbase + tid];
        int local_x = packed & mask_x;
        int local_y = (packed >> bx) & mask_y;
        int local_z = (packed >> (bx + by)) & mask_z;
        out[(vo + tid) * 3 + 0] = (s_off[0] + local_x) * s_dq[1] + s_dq[0];
        out[(vo + tid) * 3 + 1] = (s_off[1] + local_y) * s_dq[3] + s_dq[2];
        out[(vo + tid) * 3 + 2] = (s_off[2] + local_z) * s_dq[5] + s_dq[4];
    }

    // Phase 2: Triangle indices (parallel)
    int t_base = tri_offsets[m];
    int t_out = toff[m];
    for (int t = tid; t < ntri; t += blockDim.x) {
        int off = (t_base + t) * 3;
        out_tris[(t_out + t) * 3 + 0] = tri_indices[off];
        out_tris[(t_out + t) * 3 + 1] = tri_indices[off + 1];
        out_tris[(t_out + t) * 3 + 2] = tri_indices[off + 2];
    }
}

// ---- Kernel: S1 GlobalInt+Wav+EB (crack-free, sequential connectivity) ----
__global__ void k_global_int_wav_eb(
    const int* opcodes, const int* op_offsets, const int* op_sizes,
    const int* wav_int_data, const int* wav_offsets,
    const int* wav_base_sizes, const int* wav_n_levels,
    const float* global_dequant,
    const int* nverts,
    float* out, const int* voff,
    int nmesh
) {
    int m = blockIdx.x;
    if (m >= nmesh) return;
    int tid = threadIdx.x;
    int nv = nverts[m];

    __shared__ int si[MAX_VERTS * 3];
    __shared__ int tmp_i[MAX_VERTS];
    __shared__ int s_tris[MAX_VERTS * 3];

    // Phase 1: EdgeBreaker decode (thread 0, sequential)
    if (tid == 0) {
        int cnt;
        sim_eb(&opcodes[op_offsets[m]], op_sizes[m], s_tris, &cnt);
    }
    __syncthreads();

    // Phase 2: Integer wavelet inverse (all threads)
    int wo = wav_offsets[m];
    int bsz = wav_base_sizes[m];
    int nlvl = wav_n_levels[m];

    for (int ch = 0; ch < 3; ch++) {
        if (tid < bsz)
            si[ch * MAX_VERTS + tid] = wav_int_data[wo + ch * nv + tid];
        __syncthreads();
        int* channel = &si[ch * MAX_VERTS];
        int csz = bsz;
        int doff = wo + ch * nv + bsz;
        for (int l = nlvl - 1; l >= 0; l--) {
            __syncthreads();
            if (tid < csz && csz + tid < MAX_VERTS)
                channel[csz + tid] = channel[tid] + wav_int_data[doff + tid];
            __syncthreads();
            if (tid < csz * 2 && tid < nv)
                tmp_i[tid] = channel[(tid & 1) ? csz + tid/2 : tid/2];
            __syncthreads();
            if (tid < csz * 2 && tid < nv)
                channel[tid] = tmp_i[tid];
            __syncthreads();
            doff += csz;
            csz *= 2;
        }
    }
    __syncthreads();

    // Phase 3: Global dequantize
    int vo = voff[m];
    if (tid < nv) {
        for (int ch = 0; ch < 3; ch++) {
            float mn = global_dequant[ch * 3 + 0];
            float rng = global_dequant[ch * 3 + 1];
            float mx_val = global_dequant[ch * 3 + 2];
            out[(vo + tid) * 3 + ch] = (float)si[ch * MAX_VERTS + tid] / mx_val * rng + mn;
        }
    }
}

// ---- Kernel 5: AMD Packed (GPU-optimized) + Wavelet ----
// The encoder pre-packs triangles as 3 local vertex indices (uint8 each).
// Decode is trivially parallel: each thread reads 1 triangle (3 bytes).
// No FIFO, no state machine — just index lookup.
// This is how real AMD meshlet decode works on GPU.
__global__ void k_amd_packed_wav(
    const unsigned char* tri_indices,  // packed: 3 × uint8 per triangle
    const int* tri_offsets,            // offset per meshlet into tri_indices
    const int* tri_counts,             // n_tris per meshlet
    const float* wdata, const int* woff, const int* wbsz, const int* wnlvl,
    const int* nverts, float* out, const int* voff,
    int* out_tris, const int* toff,    // triangle output
    int nmesh
) {
    int m = blockIdx.x;
    if (m >= nmesh) return;
    int tid = threadIdx.x;
    int nv = nverts[m];
    int ntri = tri_counts[m];

    __shared__ float sv[MAX_VERTS * 3];
    __shared__ float tmp[MAX_VERTS];

    // Phase 1: Connectivity decode — fully parallel!
    // Each thread decodes 1-2 triangles by reading 3 bytes each
    int t_base = tri_offsets[m];
    int t_out = toff[m];
    for (int t = tid; t < ntri; t += blockDim.x) {
        int off = (t_base + t) * 3;
        int v0 = tri_indices[off];
        int v1 = tri_indices[off + 1];
        int v2 = tri_indices[off + 2];
        out_tris[(t_out + t) * 3 + 0] = v0;
        out_tris[(t_out + t) * 3 + 1] = v1;
        out_tris[(t_out + t) * 3 + 2] = v2;
    }
    __syncthreads();

    // Phase 2: Wavelet inverse (same as other kernels)
    int wo = woff[m], bs = wbsz[m], nl = wnlvl[m];
    for (int c = 0; c < 3; c++)
        if (tid < bs) sv[c * MAX_VERTS + tid] = wdata[wo + c * nv + tid];
    __syncthreads();

    for (int c = 0; c < 3; c++) {
        float* channel = &sv[c * MAX_VERTS];
        int csz = bs;
        int doff = wo + c * nv + bs;
        for (int l = nl - 1; l >= 0; l--) {
            __syncthreads();
            if (tid < csz && csz + tid < MAX_VERTS)
                channel[csz + tid] = channel[tid] + wdata[doff + tid];
            __syncthreads();
            if (tid < csz * 2 && tid < nv)
                tmp[tid] = channel[(tid & 1) ? csz + tid/2 : tid/2];
            __syncthreads();
            if (tid < csz * 2 && tid < nv)
                channel[tid] = tmp[tid];
            __syncthreads();
            doff += csz;
            csz *= 2;
        }
    }
    __syncthreads();

    int vo = voff[m];
    if (tid < nv) {
        out[(vo + tid) * 3 + 0] = sv[tid];
        out[(vo + tid) * 3 + 1] = sv[MAX_VERTS + tid];
        out[(vo + tid) * 3 + 2] = sv[2 * MAX_VERTS + tid];
    }
}

}
'''


def prepare_data(meshlets, verts, tris_np, tri_adj):
    """Pack all meshlet data for GPU."""
    all_wav = []; wav_off = []; wav_bsz = []; wav_nlv = []; nv_list = []
    all_conn = []; conn_off = []; conn_sz = []
    all_ops = []; ops_off = []; ops_sz = []
    ow = oc = oo = 0

    for ml_tris in meshlets:
        vs = set()
        for ti in ml_tris:
            for j in range(3): vs.add(int(tris_np[ti, j]))
        lv = sorted(vs); g2l = {g: l for l, g in enumerate(lv)}
        nv = len(lv); pts = verts[lv]
        tb = min(32, nv)

        # Wavelet
        wch = []
        nl = 0
        for c in range(3):
            base, levels, _ = haar_decompose(pts[:, c], tb)
            nl = len(levels)
            d = list(base)
            for l in reversed(levels): d.extend(l.tolist())
            while len(d) < nv: d.append(0.0)
            wch.extend(d)
        wav_off.append(ow); ow += len(wch)
        all_wav.extend(wch); wav_bsz.append(len(base)); wav_nlv.append(nl); nv_list.append(nv)

        # AMD connectivity
        bfs = meshlet_bfs(ml_tris, tri_adj)
        tri_map = {ti: li for li, ti in enumerate(ml_tris)}
        la = [[] for _ in range(len(ml_tris))]
        tl = np.zeros((len(ml_tris), 3), dtype=int)
        for li, ti in enumerate(ml_tris):
            for j in range(3): tl[li, j] = g2l[int(tris_np[ti, j])]
            for nb in tri_adj[ti]:
                if nb in tri_map: la[li].append(tri_map[nb])

        bo = [tri_map[ti] for ti, _ in bfs]
        proc = set(); vk = set(); cache = VertexFIFO(32); etv = {}; entries = []
        for step, li in enumerate(bo):
            tv = [int(tl[li, j]) for j in range(3)]
            pli = None
            for nb in sorted(la[li]):
                if nb in proc: pli = nb; break
            if pli is None:
                for v in tv: vk.add(v); cache.push(v)
                etv[li] = tv
            else:
                pv = etv[pli]; sh = set(tv) & set(pv)
                pe = [(pv[0],pv[1]),(pv[1],pv[2]),(pv[0],pv[2])]
                ec = 0
                for ei,(a,b) in enumerate(pe):
                    if {a,b}==sh: ec=ei; break
                for v in tv:
                    if v in sh: continue
                    if v not in vk:
                        entries.extend([0, ec, -1]); vk.add(v)
                    else:
                        ci = cache.index_of(v)
                        if ci >= 0: entries.extend([1, ec, ci])
                        else: entries.extend([2, ec, v])
                    cache.push(v)
                sv = sorted(sh); ns = [v for v in tv if v not in sh]
                etv[li] = sv + ns
            proc.add(li)

        conn_off.append(oc); conn_sz.append(len(entries)//3)
        all_conn.extend(entries); oc += len(entries)

        # EdgeBreaker opcodes
        _, opcodes, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
        om = {'C':0,'L':1,'R':2,'S':3,'E':4}
        o = [om.get(x,0) for x in opcodes]
        ops_off.append(oo); ops_sz.append(len(o))
        all_ops.extend(o); oo += len(o)

    # Also prepare packed triangle indices for AMD-packed kernel
    all_packed = []; packed_off = []; packed_cnt = []; packed_toff = []
    op = 0; ot = 0
    for ml_tris in meshlets:
        vs = set()
        for ti in ml_tris:
            for j in range(3): vs.add(int(tris_np[ti, j]))
        lv = sorted(vs); g2l = {g: l for l, g in enumerate(lv)}
        nf = len(ml_tris)
        packed_off.append(op); packed_cnt.append(nf); packed_toff.append(ot)
        for ti in ml_tris:
            for j in range(3):
                all_packed.append(g2l[int(tris_np[ti, j])])
        op += nf; ot += nf

    result = {k: np.array(v, dtype=np.float32 if k=='wavelet' else np.int32) for k,v in {
        'wavelet': all_wav, 'wav_off': wav_off, 'wav_bsz': wav_bsz,
        'wav_nlv': wav_nlv, 'nverts': nv_list,
        'conn': all_conn or [0], 'conn_off': conn_off, 'conn_sz': conn_sz,
        'ops': all_ops or [0], 'ops_off': ops_off, 'ops_sz': ops_sz,
    }.items()}
    result['packed_tri'] = np.array(all_packed, dtype=np.uint8)
    result['packed_off'] = np.array(packed_off, dtype=np.int32)
    result['packed_cnt'] = np.array(packed_cnt, dtype=np.int32)
    result['packed_toff'] = np.array(packed_toff, dtype=np.int32)

    # AMD Global Grid: proper bit-packed quantized vertices (1 uint32 per vertex)
    from utils.wavelet import _bits_for_error as bfe
    per_coord_err_amd = 0.0005 / np.max(np.linalg.norm(verts, axis=1).max()) / np.sqrt(3)

    # Global AABB
    g_min = verts.min(axis=0)
    g_max = verts.max(axis=0)
    g_delta = g_max - g_min
    g_delta[g_delta < 1e-12] = 1e-12

    # Largest meshlet extent per axis → target local bits
    largest_ml_delta = np.zeros(3)
    ml_verts_list = []
    for ml_tris in meshlets:
        vs = set()
        for ti in ml_tris:
            for j in range(3): vs.add(int(tris_np[ti, j]))
        lv = sorted(vs)
        pts = verts[lv]
        ml_verts_list.append((lv, pts))
        if len(pts) > 1:
            for d in range(3):
                delta = pts[:, d].max() - pts[:, d].min()
                largest_ml_delta[d] = max(largest_ml_delta[d], delta)

    target_bits = np.array([bfe(largest_ml_delta[d], per_coord_err_amd) for d in range(3)])
    # Global grid
    meshlet_step = np.array([largest_ml_delta[d] / ((1 << target_bits[d]) - 1)
                             if target_bits[d] > 0 else 1e-12 for d in range(3)])
    global_states = np.array([max(1, int(g_delta[d] / meshlet_step[d])) for d in range(3)])
    quant_factor = np.array([(global_states[d] - 1) / g_delta[d] for d in range(3)])
    dequant_factor = np.array([g_delta[d] / (global_states[d] - 1) for d in range(3)])

    # Pack vertices: per meshlet, compute offset + local codes packed into uint32
    all_packed_verts = []
    pv_off = []
    all_ml_offsets = []
    pvo = 0
    for lv, pts in ml_verts_list:
        pv_off.append(pvo)
        # Meshlet quantized offset
        ml_min = pts.min(axis=0) if len(pts) > 0 else g_min
        q_offset = np.array([int((ml_min[d] - g_min[d]) * quant_factor[d] + 0.5)
                              for d in range(3)], dtype=np.int64)
        all_ml_offsets.extend(q_offset.tolist())

        bx, by, bz = int(target_bits[0]), int(target_bits[1]), int(target_bits[2])
        for v_idx in lv:
            p = verts[v_idx]
            # Global quantize then subtract offset for local
            gqx = int((p[0] - g_min[0]) * quant_factor[0] + 0.5)
            gqy = int((p[1] - g_min[1]) * quant_factor[1] + 0.5)
            gqz = int((p[2] - g_min[2]) * quant_factor[2] + 0.5)
            lx = max(0, gqx - int(q_offset[0]))
            ly = max(0, gqy - int(q_offset[1]))
            lz = max(0, gqz - int(q_offset[2]))
            # Pack into uint32: x | (y << bx) | (z << (bx+by))
            packed = (lx & ((1 << bx) - 1)) | \
                     ((ly & ((1 << by) - 1)) << bx) | \
                     ((lz & ((1 << bz) - 1)) << (bx + by))
            all_packed_verts.append(packed)
        pvo += len(lv)

    result['packed_qverts'] = np.array(all_packed_verts, dtype=np.uint32)
    result['pv_off'] = np.array(pv_off, dtype=np.int32)
    # Dequant params: [min_x, dequant_factor_x, min_y, dq_factor_y, min_z, dq_factor_z]
    result['dequant_params'] = np.array([
        g_min[0], dequant_factor[0], g_min[1], dequant_factor[1],
        g_min[2], dequant_factor[2]], dtype=np.float32)
    result['ml_offsets_q'] = np.array(all_ml_offsets, dtype=np.int32)
    result['bits_per_axis'] = target_bits.astype(np.int32)

    # Global-quantized integer wavelet data (Solution 1)
    from encoder.implementation.meshlet_wavelet import _global_quantize
    from utils.wavelet import haar_decompose_int
    per_coord_err_global = 0.0005 / np.max(np.linalg.norm(verts, axis=1).max()) / np.sqrt(3)
    # Use same error as encoder
    gc, g_min, g_range, g_bits = _global_quantize(verts, per_coord_err_global)

    all_int_wav = []; int_wav_off = []; int_wav_bsz = []; int_wav_nlv = []
    oiw = 0
    for ml_tris in meshlets:
        vs = set()
        for ti in ml_tris:
            for j in range(3): vs.add(int(tris_np[ti, j]))
        lv = sorted(vs)
        nv_local = len(lv)
        int_pts = gc[lv]

        int_wav_off.append(oiw)
        tb = min(32, nv_local)
        nl = 0
        for ch in range(3):
            base, levels, _ = haar_decompose_int(int_pts[:, ch], tb)
            nl = len(levels)
            d = list(base)
            for l in reversed(levels): d.extend(l.tolist())
            while len(d) < nv_local: d.append(0)
            all_int_wav.extend(d)
        oiw += nv_local * 3
        int_wav_bsz.append(len(base))
        int_wav_nlv.append(nl)

    result['int_wav'] = np.array(all_int_wav, dtype=np.int32)
    result['int_wav_off'] = np.array(int_wav_off, dtype=np.int32)
    result['int_wav_bsz'] = np.array(int_wav_bsz, dtype=np.int32)
    result['int_wav_nlv'] = np.array(int_wav_nlv, dtype=np.int32)
    # Global dequant params: [min_x, range_x, max_code_x, min_y, range_y, max_code_y, ...]
    gdq = []
    for d_idx in range(3):
        gdq.extend([float(g_min[d_idx]), float(g_range[d_idx]),
                     float((1 << g_bits[d_idx]) - 1)])
    result['global_dequant'] = np.array(gdq, dtype=np.float32)

    # Packed uint32 triangles (3 × uint8 in low 24 bits of uint32)
    all_packed_tris_u32 = []
    for ml_tris in meshlets:
        vs = set()
        for ti in ml_tris:
            for j in range(3): vs.add(int(tris_np[ti, j]))
        lv = sorted(vs); g2l_map = {g: l for l, g in enumerate(lv)}
        for ti in ml_tris:
            v0 = g2l_map[int(tris_np[ti, 0])]
            v1 = g2l_map[int(tris_np[ti, 1])]
            v2 = g2l_map[int(tris_np[ti, 2])]
            all_packed_tris_u32.append((v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16))
    result['packed_tris_u32'] = np.array(all_packed_tris_u32, dtype=np.uint32)

    return result


def benchmark(obj_path, max_verts=256, n_warmup=50, n_runs=200):
    print(f"{'='*70}")
    print(f"CUDA Decode Benchmark — {obj_path}, max_verts={max_verts}")
    print(f"{'='*70}")

    mesh = Reader.read_from_file(obj_path)
    nv = len(mesh.vertices); nt = len(mesh.triangles)
    vn = np.empty((nv, 3), dtype=np.float64)
    for i, v in enumerate(mesh.vertices): vn[i] = (v.x, v.y, v.z)
    tn = np.empty((nt, 3), dtype=np.int64)
    for i, t in enumerate(mesh.triangles): tn[i] = (t.a, t.b, t.c)
    c = vn.mean(0); vn = ((vn - c) / np.max(np.linalg.norm(vn - c, axis=1))).astype(np.float32)

    ta = build_adjacency(tn)
    fn = compute_face_normals(vn, tn); fc = compute_face_centroids(vn, tn)
    ml = generate_meshlets_by_verts(tn, ta, fn, fc, max_verts=max_verts)
    nm = len(ml)
    print(f"  {nv:,} verts, {nt:,} tris, {nm} meshlets")

    print("  Packing data..."); t0 = time.time()
    d = prepare_data(ml, vn, tn, ta)
    print(f"  Packed in {time.time()-t0:.1f}s")

    nva = d['nverts']; vo = np.cumsum(np.concatenate([[0], nva[:-1]])).astype(np.int32)
    tv = int(nva.sum())

    # Upload
    gd = {k: cp.asarray(v) for k, v in d.items()}
    g_vo = cp.asarray(vo); g_out = cp.zeros(tv * 3, dtype=cp.float32)

    # Total triangles for throughput calc
    total_tris = sum(len(m) for m in ml)

    # Allocate triangle output for packed kernel
    g_out_tris = cp.zeros(total_tris * 3, dtype=cp.int32)
    tri_off = np.cumsum(np.concatenate([[0], d['packed_cnt'][:-1]])).astype(np.int32)
    g_tri_off = cp.asarray(tri_off)

    # Compile
    print("  Compiling kernels...")
    mod = cp.RawModule(code=CUDA_SRC)
    k_wav = mod.get_function('k_wavelet')
    k_amd = mod.get_function('k_amd_wav')
    k_eb = mod.get_function('k_eb_wav')
    k_packed = mod.get_function('k_amd_packed_wav')
    k_quant = mod.get_function('k_amd_quant')
    k_gint_amd = mod.get_function('k_global_int_wav_amd')
    k_gint_eb = mod.get_function('k_global_int_wav_eb')
    k_s1_opt = mod.get_function('k_s1_optimized')
    k_s1_v2 = mod.get_function('k_s1_opt_v2')
    k_s1_v3 = mod.get_function('k_s1_opt_v3')

    bs = 256

    def run_kernel(name, kernel, args):
        for _ in range(n_warmup):
            kernel((nm,), (bs,), args)
        cp.cuda.Stream.null.synchronize()
        s = cp.cuda.Event(); e = cp.cuda.Event()
        s.record()
        for _ in range(n_runs):
            kernel((nm,), (bs,), args)
        e.record(); e.synchronize()
        ms = cp.cuda.get_elapsed_time(s, e) / n_runs
        us = ms * 1000
        mvps = tv / ms / 1e6
        mtps = total_tris / ms / 1e6
        print(f"  {name:<35} {us:>8.1f} µs  {mvps:>8.1f} Mverts/s  {mtps:>8.1f} Mtris/s")
        return ms

    print(f"\n  Results ({n_runs} runs, {n_warmup} warmup):\n")

    t_quant = run_kernel("AMD Packed+GlobalGrid (baseline)",
        k_quant, (gd['packed_tri'], gd['packed_off'], gd['packed_cnt'],
                  gd['packed_qverts'], gd['pv_off'],
                  gd['dequant_params'], gd['ml_offsets_q'], gd['bits_per_axis'],
                  gd['nverts'], g_out, g_vo,
                  g_out_tris, g_tri_off, np.int32(nm)))

    t_gint_amd = run_kernel("S1: GlobalInt+Wav+AMD (ours)",
        k_gint_amd, (gd['packed_tri'], gd['packed_off'], gd['packed_cnt'],
                     gd['int_wav'], gd['int_wav_off'], gd['int_wav_bsz'], gd['int_wav_nlv'],
                     gd['global_dequant'],
                     gd['nverts'], g_out, g_vo,
                     g_out_tris, g_tri_off, np.int32(nm)))

    s1_opt_args = (gd['packed_tris_u32'], gd['packed_off'], gd['packed_cnt'],
                   gd['int_wav'], gd['int_wav_off'], gd['int_wav_bsz'], gd['int_wav_nlv'],
                   gd['global_dequant'],
                   gd['nverts'], g_out, g_vo,
                   g_out_tris, g_tri_off, np.int32(nm))

    t_s1_opt = run_kernel("S1: Opt v1 (warp+fused)", k_s1_opt, s1_opt_args)
    t_s1_v2 = run_kernel("S1: Opt v2 (no syncwarp+ldg)", k_s1_v2, s1_opt_args)
    t_s1_v3 = run_kernel("S1: Opt v3 (3ch parallel+ldg)", k_s1_v3, s1_opt_args)

    t_gint_eb = run_kernel("S1: GlobalInt+Wav+EB",
        k_gint_eb, (gd['ops'], gd['ops_off'], gd['ops_sz'],
                    gd['int_wav'], gd['int_wav_off'], gd['int_wav_bsz'], gd['int_wav_nlv'],
                    gd['global_dequant'],
                    gd['nverts'], g_out, g_vo, np.int32(nm)))

    t_wav = run_kernel("Wavelet only",
        k_wav, (gd['wavelet'], gd['wav_off'], gd['wav_bsz'], gd['wav_nlv'],
                gd['nverts'], g_out, g_vo, np.int32(nm)))

    t_packed = run_kernel("AMD Packed + Wavelet",
        k_packed, (gd['packed_tri'], gd['packed_off'], gd['packed_cnt'],
                   gd['wavelet'], gd['wav_off'], gd['wav_bsz'], gd['wav_nlv'],
                   gd['nverts'], g_out, g_vo,
                   g_out_tris, g_tri_off, np.int32(nm)))

    t_amd = run_kernel("AMD FIFO (naive seq.) + Wavelet",
        k_amd, (gd['conn'], gd['conn_off'], gd['conn_sz'],
                gd['wavelet'], gd['wav_off'], gd['wav_bsz'], gd['wav_nlv'],
                gd['nverts'], g_out, g_vo, np.int32(nm)))

    t_eb = run_kernel("EdgeBreaker (seq.) + Wavelet",
        k_eb, (gd['ops'], gd['ops_off'], gd['ops_sz'],
               gd['wavelet'], gd['wav_off'], gd['wav_bsz'], gd['wav_nlv'],
               gd['nverts'], g_out, g_vo, np.int32(nm)))

    print(f"\n  Comparison:")
    print(f"    {'Kernel':<42} {'Time':>10} {'vs Baseline':>12} {'vs S1 Orig':>11}")
    best_s1 = min(t_gint_amd, t_s1_opt, t_s1_v2, t_s1_v3)
    for name, t in [
        ("AMD GlobalGrid (baseline)", t_quant),
        ("S1: Original", t_gint_amd),
        ("S1: Opt v1 (warp+fused)", t_s1_opt),
        ("S1: Opt v2 (no syncwarp+ldg)", t_s1_v2),
        ("S1: Opt v3 (3ch parallel+ldg)", t_s1_v3),
        ("S1: EdgeBreaker", t_gint_eb),
    ]:
        r_base = t_quant / t if t > 0 else 0
        r_orig = t_gint_amd / t if t > 0 else 0
        marker = " <-- best S1" if abs(t - best_s1) < 0.001 else ""
        print(f"    {name:<42} {t*1000:>8.1f} µs  {r_base:>10.2f}x  {r_orig:>9.2f}x{marker}")

    print(f"\n  Key takeaway:")
    overhead = (best_s1 - t_quant) / t_quant * 100 if t_quant > 0 else 0
    speedup = (t_gint_amd - best_s1) / t_gint_amd * 100 if t_gint_amd > 0 else 0
    print(f"    Best S1 overhead vs AMD baseline: {overhead:+.1f}%")
    print(f"    Best S1 speedup vs S1 original: {speedup:+.1f}%")


if __name__ == "__main__":
    benchmark("assets/stanford-bunny.obj", max_verts=256)
    print()
    benchmark("assets/Monkey.obj", max_verts=256)

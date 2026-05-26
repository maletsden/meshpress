"""Optimized LIN5-only CUDA decoders for ParaDelta.

Three kernels for benchmarking:
  V2 (per_thread):  1 meshlet per thread, 32 threads/block.
                    Shared mem 96 KB/block (3 KB recon × 32) — opt-in to
                    the 100 KB dynamic shared limit on Ampere.
  V3 (warp):        1 meshlet per block, 32 threads/block.
                    Parallel boundary/tris over 32 lanes, sequential walk
                    on lane 0.

Both drop the predictor-mode branch (LIN5 always); kind branches for
non-para verts remain (mid/one/none) since the encoder still emits them.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


# =====================================================================
# V2: 1 meshlet / thread, 32 threads / block
# =====================================================================
_V2_SRC = r"""
extern "C" __global__ void paradelta_lin5_per_thread(
    const int* __restrict__ ml_n_bnd,
    const int* __restrict__ ml_n_int,
    const int* __restrict__ ml_n_tris,
    const int* __restrict__ ml_l2g_off,
    const int* __restrict__ l2g,
    const int* __restrict__ ml_tris_off,
    const int* __restrict__ local_tris,
    const int* __restrict__ ml_codes_off,
    const int* __restrict__ codes,
    const int* __restrict__ ml_order_off,
    const int* __restrict__ order,
    const float* __restrict__ bnd_pos,
    const float* __restrict__ lin5_w3,
    const float* __restrict__ lin5_w5,
    const float delta,
    const int n_meshlets,
    const int n_local_max,
    float* __restrict__ verts_out,
    int* __restrict__ tris_out
)
{
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_meshlets) return;

    const int n_bnd  = ml_n_bnd[m];
    const int n_int  = ml_n_int[m];
    const int n_tris = ml_n_tris[m];

    const int l2g_off   = ml_l2g_off[m];
    const int tris_off  = ml_tris_off[m];
    const int codes_off = ml_codes_off[m];
    const int order_off = ml_order_off[m];

    extern __shared__ float recon_pool[];
    float* recon = &recon_pool[threadIdx.x * n_local_max * 3];

    // Boundary copy (serial per thread)
    float fb_x = 0.f, fb_y = 0.f, fb_z = 0.f;
    for (int i = 0; i < n_bnd; ++i) {
        const int gid = __ldg(&l2g[l2g_off + i]);
        const float x = __ldg(&bnd_pos[gid*3+0]);
        const float y = __ldg(&bnd_pos[gid*3+1]);
        const float z = __ldg(&bnd_pos[gid*3+2]);
        recon[i*3+0] = x; recon[i*3+1] = y; recon[i*3+2] = z;
        fb_x += x; fb_y += y; fb_z += z;
    }
    if (n_bnd > 0) {
        const float inv = 1.0f / (float)n_bnd;
        fb_x *= inv; fb_y *= inv; fb_z *= inv;
    }

    const float w3_0 = lin5_w3[0], w3_1 = lin5_w3[1], w3_2 = lin5_w3[2];
    const float w5_0 = lin5_w5[0], w5_1 = lin5_w5[1], w5_2 = lin5_w5[2];
    const float w5_3 = lin5_w5[3], w5_4 = lin5_w5[4];

    for (int i = 0; i < n_int; ++i) {
        const int base = (order_off + i) * 7;
        const int v_local = order[base + 0];
        const int kind    = order[base + 1];
        const int a       = order[base + 2];
        const int b       = order[base + 3];
        const int c       = order[base + 4];
        const int d_ac    = order[base + 5];
        const int d_bc    = order[base + 6];

        float px, py, pz;
        if (kind == 0) {
            if (d_ac >= 0 && d_bc >= 0) {
                px = w5_0*recon[a*3+0]+w5_1*recon[b*3+0]+w5_2*recon[c*3+0]
                   + w5_3*recon[d_ac*3+0]+w5_4*recon[d_bc*3+0];
                py = w5_0*recon[a*3+1]+w5_1*recon[b*3+1]+w5_2*recon[c*3+1]
                   + w5_3*recon[d_ac*3+1]+w5_4*recon[d_bc*3+1];
                pz = w5_0*recon[a*3+2]+w5_1*recon[b*3+2]+w5_2*recon[c*3+2]
                   + w5_3*recon[d_ac*3+2]+w5_4*recon[d_bc*3+2];
            } else {
                px = w3_0*recon[a*3+0]+w3_1*recon[b*3+0]+w3_2*recon[c*3+0];
                py = w3_0*recon[a*3+1]+w3_1*recon[b*3+1]+w3_2*recon[c*3+1];
                pz = w3_0*recon[a*3+2]+w3_1*recon[b*3+2]+w3_2*recon[c*3+2];
            }
        } else if (kind == 1) {
            px = 0.5f*(recon[a*3+0]+recon[b*3+0]);
            py = 0.5f*(recon[a*3+1]+recon[b*3+1]);
            pz = 0.5f*(recon[a*3+2]+recon[b*3+2]);
        } else if (kind == 2) {
            px = recon[a*3+0]; py = recon[a*3+1]; pz = recon[a*3+2];
        } else {
            px = fb_x; py = fb_y; pz = fb_z;
        }

        const int cb = (codes_off + i) * 3;
        recon[v_local*3+0] = px + (float)codes[cb+0] * delta;
        recon[v_local*3+1] = py + (float)codes[cb+1] * delta;
        recon[v_local*3+2] = pz + (float)codes[cb+2] * delta;
    }

    // Write interior verts
    for (int i = 0; i < n_int; ++i) {
        const int lid = n_bnd + i;
        const int gid = __ldg(&l2g[l2g_off + lid]);
        verts_out[gid*3+0] = recon[lid*3+0];
        verts_out[gid*3+1] = recon[lid*3+1];
        verts_out[gid*3+2] = recon[lid*3+2];
    }
    // Write tris (with global remap)
    for (int t = 0; t < n_tris; ++t) {
        const int rb = (tris_off + t) * 3;
        tris_out[rb+0] = __ldg(&l2g[l2g_off + local_tris[rb+0]]);
        tris_out[rb+1] = __ldg(&l2g[l2g_off + local_tris[rb+1]]);
        tris_out[rb+2] = __ldg(&l2g[l2g_off + local_tris[rb+2]]);
    }
}
"""


# =====================================================================
# V3: 1 meshlet / block, 32 threads / block (one warp)
# =====================================================================
_V3_SRC = r"""
extern "C" __global__ void paradelta_lin5_warp(
    const int* __restrict__ ml_n_bnd,
    const int* __restrict__ ml_n_int,
    const int* __restrict__ ml_n_tris,
    const int* __restrict__ ml_l2g_off,
    const int* __restrict__ l2g,
    const int* __restrict__ ml_tris_off,
    const int* __restrict__ local_tris,
    const int* __restrict__ ml_codes_off,
    const int* __restrict__ codes,
    const int* __restrict__ ml_order_off,
    const int* __restrict__ order,
    const float* __restrict__ bnd_pos,
    const float* __restrict__ lin5_w3,
    const float* __restrict__ lin5_w5,
    const float delta,
    float* __restrict__ verts_out,
    int* __restrict__ tris_out
)
{
    const int m = blockIdx.x;
    const int tid = threadIdx.x;            // 0..31

    const int n_bnd  = ml_n_bnd[m];
    const int n_int  = ml_n_int[m];
    const int n_tris = ml_n_tris[m];
    const int n_local = n_bnd + n_int;

    const int l2g_off   = ml_l2g_off[m];
    const int tris_off  = ml_tris_off[m];
    const int codes_off = ml_codes_off[m];
    const int order_off = ml_order_off[m];

    extern __shared__ float recon[];  // [n_local * 3]

    // Phase 1: parallel boundary copy
    for (int i = tid; i < n_bnd; i += 32) {
        const int gid = __ldg(&l2g[l2g_off + i]);
        recon[i*3+0] = __ldg(&bnd_pos[gid*3+0]);
        recon[i*3+1] = __ldg(&bnd_pos[gid*3+1]);
        recon[i*3+2] = __ldg(&bnd_pos[gid*3+2]);
    }
    __syncwarp();

    // Phase 2: lane 0 sequential walk
    if (tid == 0 && n_int > 0) {
        float fb_x = 0.f, fb_y = 0.f, fb_z = 0.f;
        if (n_bnd > 0) {
            for (int i = 0; i < n_bnd; ++i) {
                fb_x += recon[i*3+0];
                fb_y += recon[i*3+1];
                fb_z += recon[i*3+2];
            }
            const float inv = 1.0f / (float)n_bnd;
            fb_x *= inv; fb_y *= inv; fb_z *= inv;
        }
        const float w3_0 = lin5_w3[0], w3_1 = lin5_w3[1], w3_2 = lin5_w3[2];
        const float w5_0 = lin5_w5[0], w5_1 = lin5_w5[1], w5_2 = lin5_w5[2];
        const float w5_3 = lin5_w5[3], w5_4 = lin5_w5[4];

        for (int i = 0; i < n_int; ++i) {
            const int base = (order_off + i) * 7;
            const int v_local = order[base + 0];
            const int kind    = order[base + 1];
            const int a       = order[base + 2];
            const int b       = order[base + 3];
            const int c       = order[base + 4];
            const int d_ac    = order[base + 5];
            const int d_bc    = order[base + 6];

            float px, py, pz;
            if (kind == 0) {
                if (d_ac >= 0 && d_bc >= 0) {
                    px = w5_0*recon[a*3+0]+w5_1*recon[b*3+0]+w5_2*recon[c*3+0]
                       + w5_3*recon[d_ac*3+0]+w5_4*recon[d_bc*3+0];
                    py = w5_0*recon[a*3+1]+w5_1*recon[b*3+1]+w5_2*recon[c*3+1]
                       + w5_3*recon[d_ac*3+1]+w5_4*recon[d_bc*3+1];
                    pz = w5_0*recon[a*3+2]+w5_1*recon[b*3+2]+w5_2*recon[c*3+2]
                       + w5_3*recon[d_ac*3+2]+w5_4*recon[d_bc*3+2];
                } else {
                    px = w3_0*recon[a*3+0]+w3_1*recon[b*3+0]+w3_2*recon[c*3+0];
                    py = w3_0*recon[a*3+1]+w3_1*recon[b*3+1]+w3_2*recon[c*3+1];
                    pz = w3_0*recon[a*3+2]+w3_1*recon[b*3+2]+w3_2*recon[c*3+2];
                }
            } else if (kind == 1) {
                px = 0.5f*(recon[a*3+0]+recon[b*3+0]);
                py = 0.5f*(recon[a*3+1]+recon[b*3+1]);
                pz = 0.5f*(recon[a*3+2]+recon[b*3+2]);
            } else if (kind == 2) {
                px = recon[a*3+0]; py = recon[a*3+1]; pz = recon[a*3+2];
            } else {
                px = fb_x; py = fb_y; pz = fb_z;
            }

            const int cb = (codes_off + i) * 3;
            recon[v_local*3+0] = px + (float)codes[cb+0] * delta;
            recon[v_local*3+1] = py + (float)codes[cb+1] * delta;
            recon[v_local*3+2] = pz + (float)codes[cb+2] * delta;
        }
    }
    __syncwarp();

    // Phase 3: parallel interior write
    for (int i = tid; i < n_int; i += 32) {
        const int lid = n_bnd + i;
        const int gid = __ldg(&l2g[l2g_off + lid]);
        verts_out[gid*3+0] = recon[lid*3+0];
        verts_out[gid*3+1] = recon[lid*3+1];
        verts_out[gid*3+2] = recon[lid*3+2];
    }
    // Phase 4: parallel tri write
    for (int t = tid; t < n_tris; t += 32) {
        const int rb = (tris_off + t) * 3;
        tris_out[rb+0] = __ldg(&l2g[l2g_off + local_tris[rb+0]]);
        tris_out[rb+1] = __ldg(&l2g[l2g_off + local_tris[rb+1]]);
        tris_out[rb+2] = __ldg(&l2g[l2g_off + local_tris[rb+2]]);
    }
}
"""


class _BaseLin5Decoder:
    """Shared upload of struct → GPU buffers."""

    def __init__(self, s: dict):
        if not _HAS_CUPY:
            raise RuntimeError("cupy not installed")
        self.n_v = int(s["n_boundary"]) + int(s["ml_n_int"].sum())
        self.n_t = int(s["n_t"])
        self.n_boundary = int(s["n_boundary"])
        self.n_meshlets = int(s["n_meshlets"])
        self.n_local_max = int((s["ml_n_bnd"] + s["ml_n_int"]).max())

        self.d_ml_n_bnd = cp.asarray(s["ml_n_bnd"], dtype=cp.int32)
        self.d_ml_n_int = cp.asarray(s["ml_n_int"], dtype=cp.int32)
        self.d_ml_n_tris = cp.asarray(s["ml_n_tris"], dtype=cp.int32)
        self.d_ml_l2g_off = cp.asarray(s["ml_l2g_off"], dtype=cp.int32)
        self.d_l2g = cp.asarray(s["ml_l2g"], dtype=cp.int32)
        self.d_ml_tris_off = cp.asarray(s["ml_tris_off"], dtype=cp.int32)
        self.d_local_tris = cp.asarray(
            s["ml_tris"].reshape(-1), dtype=cp.int32)
        self.d_ml_codes_off = cp.asarray(s["ml_codes_off"], dtype=cp.int32)
        self.d_codes = cp.asarray(s["ml_codes"].reshape(-1), dtype=cp.int32)
        self.d_ml_order_off = cp.asarray(s["ml_order_off"], dtype=cp.int32)
        self.d_order = cp.asarray(s["ml_order"].reshape(-1), dtype=cp.int32)
        self.d_bnd_pos = cp.asarray(
            s["bnd_pos_norm"].reshape(-1), dtype=cp.float32)
        self.d_lin5_w3 = cp.asarray(s["lin5_w3"], dtype=cp.float32)
        self.d_lin5_w5 = cp.asarray(s["lin5_w5"], dtype=cp.float32)
        self.delta = float(s["delta"])
        self.scale = float(s["scale"])
        self.center = np.asarray(s["center"], dtype=np.float32)

        self.d_verts_norm = cp.zeros((self.n_v, 3), dtype=cp.float32)
        self.d_tris = cp.zeros((self.n_t, 3), dtype=cp.int32)

    def _prefill_boundary(self):
        self.d_verts_norm[: self.n_boundary] = \
            self.d_bnd_pos.reshape(self.n_boundary, 3)

    def decode_to_host(self) -> tuple[np.ndarray, np.ndarray]:
        v_norm, tris = self.decode()
        verts_world = (v_norm * cp.float32(self.scale)
                       + cp.asarray(self.center, dtype=cp.float32))
        return cp.asnumpy(verts_world), cp.asnumpy(tris)


class ParaDeltaLin5PerThread(_BaseLin5Decoder):
    """V2 — 1 meshlet per thread, 32 threads per block."""

    def __init__(self, s: dict, threads_per_block: int = 32):
        super().__init__(s)
        self.tpb = threads_per_block
        self.n_blocks = (self.n_meshlets + self.tpb - 1) // self.tpb
        # Shared mem: tpb * n_local_max * 3 floats
        self.shared_bytes = int(self.tpb * self.n_local_max * 3 * 4)
        self._kernel = cp.RawKernel(_V2_SRC, "paradelta_lin5_per_thread")
        # Opt-in to >48 KB dynamic shared on Ampere (cap 100 KB).
        if self.shared_bytes > 48 * 1024:
            try:
                self._kernel.max_dynamic_shared_size_bytes = self.shared_bytes
            except Exception:
                pass

    def decode(self):
        self._prefill_boundary()
        self._kernel(
            (self.n_blocks,), (self.tpb,),
            (
                self.d_ml_n_bnd, self.d_ml_n_int, self.d_ml_n_tris,
                self.d_ml_l2g_off, self.d_l2g,
                self.d_ml_tris_off, self.d_local_tris,
                self.d_ml_codes_off, self.d_codes,
                self.d_ml_order_off, self.d_order,
                self.d_bnd_pos, self.d_lin5_w3, self.d_lin5_w5,
                cp.float32(self.delta),
                cp.int32(self.n_meshlets), cp.int32(self.n_local_max),
                self.d_verts_norm, self.d_tris,
            ),
            shared_mem=self.shared_bytes,
        )
        return self.d_verts_norm, self.d_tris


class ParaDeltaLin5Warp(_BaseLin5Decoder):
    """V3 — 1 meshlet per block, 32 threads per block (one warp)."""

    def __init__(self, s: dict):
        super().__init__(s)
        self.shared_bytes = int(self.n_local_max * 3 * 4)
        self._kernel = cp.RawKernel(_V3_SRC, "paradelta_lin5_warp")

    def decode(self):
        self._prefill_boundary()
        self._kernel(
            (self.n_meshlets,), (32,),
            (
                self.d_ml_n_bnd, self.d_ml_n_int, self.d_ml_n_tris,
                self.d_ml_l2g_off, self.d_l2g,
                self.d_ml_tris_off, self.d_local_tris,
                self.d_ml_codes_off, self.d_codes,
                self.d_ml_order_off, self.d_order,
                self.d_bnd_pos, self.d_lin5_w3, self.d_lin5_w5,
                cp.float32(self.delta),
                self.d_verts_norm, self.d_tris,
            ),
            shared_mem=self.shared_bytes,
        )
        return self.d_verts_norm, self.d_tris
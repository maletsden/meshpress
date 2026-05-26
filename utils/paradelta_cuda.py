"""CuPy CUDA decoder for ParaDelta L5 — Phase 1.

Operates on structured arrays from decode_paradelta_to_struct (CPU
bitstream parse). Kernel does only the math: predictor + residual
reconstruction per meshlet, parallel tri output.

Within meshlet:
  - parallel: copy boundary positions into shared recon buffer
  - sequential (thread 0): walk traversal order, apply linear5 + residual
  - parallel: write interior verts + tris to global memory

Across meshlets:
  - 1 CUDA block per meshlet, fully parallel

Phase 2 will move bitstream parsing onto GPU. For now we measure the math
portion to estimate the ceiling.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


_KERNEL_SRC = r"""
extern "C" __global__ void paradelta_decode_l5(
    // per-meshlet sizes
    const int* __restrict__ ml_n_bnd,
    const int* __restrict__ ml_n_int,
    const int* __restrict__ ml_n_tris,
    // per-meshlet offset arrays
    const int* __restrict__ ml_l2g_off,
    const int* __restrict__ l2g,
    const int* __restrict__ ml_tris_off,
    const int* __restrict__ local_tris,    // (sum_n_tris, 3)
    const int* __restrict__ ml_codes_off,
    const int* __restrict__ codes,         // (sum_n_int, 3)
    const int* __restrict__ ml_order_off,
    const int* __restrict__ order,         // (sum_n_int, 7)
    // globals
    const float* __restrict__ bnd_pos,     // (n_boundary, 3)
    const float* __restrict__ lin5_w3,     // (3,)
    const float* __restrict__ lin5_w5,     // (5,)
    const float delta,
    const int predictor_mode,
    // outputs
    float* __restrict__ verts_out,         // (n_v, 3) normalized
    int* __restrict__ tris_out             // (n_t, 3)
)
{
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    const int n_bnd  = ml_n_bnd[m];
    const int n_int  = ml_n_int[m];
    const int n_tris = ml_n_tris[m];
    const int n_local = n_bnd + n_int;

    const int l2g_off   = ml_l2g_off[m];
    const int tris_off  = ml_tris_off[m];
    const int codes_off = ml_codes_off[m];
    const int order_off = ml_order_off[m];

    extern __shared__ float recon[];  // [n_local * 3]

    // Phase 1: parallel copy boundary positions from global → shared
    for (int i = tid; i < n_bnd; i += bdim) {
        int gid = l2g[l2g_off + i];
        recon[i*3 + 0] = bnd_pos[gid*3 + 0];
        recon[i*3 + 1] = bnd_pos[gid*3 + 1];
        recon[i*3 + 2] = bnd_pos[gid*3 + 2];
    }
    __syncthreads();

    // Phase 2: thread 0 walks traversal order sequentially
    if (tid == 0 && n_int > 0) {
        float fb_x = 0.0f, fb_y = 0.0f, fb_z = 0.0f;
        if (n_bnd > 0) {
            for (int i = 0; i < n_bnd; ++i) {
                fb_x += recon[i*3 + 0];
                fb_y += recon[i*3 + 1];
                fb_z += recon[i*3 + 2];
            }
            float inv = 1.0f / (float)n_bnd;
            fb_x *= inv; fb_y *= inv; fb_z *= inv;
        }

        const float w3_0 = lin5_w3[0], w3_1 = lin5_w3[1], w3_2 = lin5_w3[2];
        const float w5_0 = lin5_w5[0], w5_1 = lin5_w5[1], w5_2 = lin5_w5[2];
        const float w5_3 = lin5_w5[3], w5_4 = lin5_w5[4];

        for (int i = 0; i < n_int; ++i) {
            const int  base = (order_off + i) * 7;
            const int  v_local = order[base + 0];
            const int  kind    = order[base + 1];
            const int  a       = order[base + 2];
            const int  b       = order[base + 3];
            const int  c       = order[base + 4];
            const int  d_ac    = order[base + 5];
            const int  d_bc    = order[base + 6];

            float px, py, pz;
            // predictor_mode==0 (PLAIN) or kind != para → Touma-Gotsman fallback
            if (predictor_mode == 0 || kind != 0) {
                if (kind == 0) {  // para
                    px = recon[a*3+0] + recon[b*3+0] - recon[c*3+0];
                    py = recon[a*3+1] + recon[b*3+1] - recon[c*3+1];
                    pz = recon[a*3+2] + recon[b*3+2] - recon[c*3+2];
                } else if (kind == 1) {  // mid
                    px = 0.5f * (recon[a*3+0] + recon[b*3+0]);
                    py = 0.5f * (recon[a*3+1] + recon[b*3+1]);
                    pz = 0.5f * (recon[a*3+2] + recon[b*3+2]);
                } else if (kind == 2) {  // one
                    px = recon[a*3+0]; py = recon[a*3+1]; pz = recon[a*3+2];
                } else {  // none
                    px = fb_x; py = fb_y; pz = fb_z;
                }
            } else {
                // LIN5 with para context
                if (d_ac >= 0 && d_bc >= 0) {
                    px = w5_0*recon[a*3+0] + w5_1*recon[b*3+0]
                       + w5_2*recon[c*3+0] + w5_3*recon[d_ac*3+0]
                       + w5_4*recon[d_bc*3+0];
                    py = w5_0*recon[a*3+1] + w5_1*recon[b*3+1]
                       + w5_2*recon[c*3+1] + w5_3*recon[d_ac*3+1]
                       + w5_4*recon[d_bc*3+1];
                    pz = w5_0*recon[a*3+2] + w5_1*recon[b*3+2]
                       + w5_2*recon[c*3+2] + w5_3*recon[d_ac*3+2]
                       + w5_4*recon[d_bc*3+2];
                } else {
                    px = w3_0*recon[a*3+0] + w3_1*recon[b*3+0]
                       + w3_2*recon[c*3+0];
                    py = w3_0*recon[a*3+1] + w3_1*recon[b*3+1]
                       + w3_2*recon[c*3+1];
                    pz = w3_0*recon[a*3+2] + w3_1*recon[b*3+2]
                       + w3_2*recon[c*3+2];
                }
            }

            const int  cbase = (codes_off + i) * 3;
            const float c0 = (float) codes[cbase + 0];
            const float c1 = (float) codes[cbase + 1];
            const float c2 = (float) codes[cbase + 2];
            recon[v_local*3 + 0] = px + c0 * delta;
            recon[v_local*3 + 1] = py + c1 * delta;
            recon[v_local*3 + 2] = pz + c2 * delta;
        }
    }
    __syncthreads();

    // Phase 3: parallel write interior verts (normalized) to output
    for (int i = tid; i < n_int; i += bdim) {
        int lid = n_bnd + i;
        int gid = l2g[l2g_off + lid];
        verts_out[gid*3 + 0] = recon[lid*3 + 0];
        verts_out[gid*3 + 1] = recon[lid*3 + 1];
        verts_out[gid*3 + 2] = recon[lid*3 + 2];
    }

    // Phase 4: parallel write triangles (decoder-global IDs)
    for (int t = tid; t < n_tris; t += bdim) {
        int rb = (tris_off + t) * 3;
        int la = local_tris[rb + 0];
        int lb = local_tris[rb + 1];
        int lc = local_tris[rb + 2];
        tris_out[rb + 0] = l2g[l2g_off + la];
        tris_out[rb + 1] = l2g[l2g_off + lb];
        tris_out[rb + 2] = l2g[l2g_off + lc];
    }
}
"""


class ParaDeltaCudaDecoder:
    """Holds GPU buffers built from a structured-arrays decode."""

    def __init__(self, s: dict, block_size: int = 128):
        if not _HAS_CUPY:
            raise RuntimeError("cupy not installed")
        # Header n_v counts ORIGINAL verts; meshlet gen may drop some. The
        # actually-encoded total is n_boundary + sum(ml_n_int).
        self.n_v = int(s["n_boundary"]) + int(s["ml_n_int"].sum())
        self.n_v_header = int(s["n_v"])
        self.n_t = s["n_t"]
        self.n_boundary = s["n_boundary"]
        self.n_meshlets = s["n_meshlets"]
        self.block_size = block_size
        # Max n_local across meshlets → shared mem size
        n_local_max = int((s["ml_n_bnd"] + s["ml_n_int"]).max())
        self.shared_bytes = int(n_local_max * 3 * 4)

        # Upload all per-meshlet arrays
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
        self.d_bnd_pos = cp.asarray(s["bnd_pos_norm"].reshape(-1),
                                    dtype=cp.float32)
        self.d_lin5_w3 = cp.asarray(s["lin5_w3"], dtype=cp.float32)
        self.d_lin5_w5 = cp.asarray(s["lin5_w5"], dtype=cp.float32)
        self.delta = float(s["delta"])
        self.predictor_mode = int(s["predictor_mode"])
        self.scale = float(s["scale"])
        self.center = np.asarray(s["center"], dtype=np.float32)

        # Output buffers (reused across calls)
        self.d_verts_norm = cp.zeros((self.n_v, 3), dtype=cp.float32)
        self.d_tris = cp.zeros((self.n_t, 3), dtype=cp.int32)
        # Pre-fill boundary portion of verts (normalized)
        self.d_verts_norm[: self.n_boundary] = \
            self.d_bnd_pos.reshape(self.n_boundary, 3)

        self._kernel = cp.RawKernel(_KERNEL_SRC, "paradelta_decode_l5")

    def decode(self) -> tuple[cp.ndarray, cp.ndarray]:
        """Run kernel. Returns (verts_norm, tris) on GPU.

        Boundary verts in `verts_norm` are pre-filled, kernel writes
        interior. Caller applies de-normalize separately.
        """
        # Re-fill boundary (cheap; ensures clean state across repeated runs)
        self.d_verts_norm[: self.n_boundary] = \
            self.d_bnd_pos.reshape(self.n_boundary, 3)
        self._kernel(
            (self.n_meshlets,), (self.block_size,),
            (
                self.d_ml_n_bnd, self.d_ml_n_int, self.d_ml_n_tris,
                self.d_ml_l2g_off, self.d_l2g,
                self.d_ml_tris_off, self.d_local_tris,
                self.d_ml_codes_off, self.d_codes,
                self.d_ml_order_off, self.d_order,
                self.d_bnd_pos, self.d_lin5_w3, self.d_lin5_w5,
                cp.float32(self.delta), cp.int32(self.predictor_mode),
                self.d_verts_norm, self.d_tris,
            ),
            shared_mem=self.shared_bytes,
        )
        return self.d_verts_norm, self.d_tris

    def decode_to_host(self) -> tuple[np.ndarray, np.ndarray]:
        """Decode + de-normalize + copy to host."""
        v_norm, tris = self.decode()
        verts_world = (v_norm * cp.float32(self.scale)
                       + cp.asarray(self.center, dtype=cp.float32))
        return cp.asnumpy(verts_world), cp.asnumpy(tris)
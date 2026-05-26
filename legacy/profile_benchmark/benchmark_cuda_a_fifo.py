"""
CUDA benchmark for Approach A (v2d+fifo) per-frame decompression.

Per-frame hot path (the operations that happen EVERY frame at target LOD):
  K1: Ancestor resolution (parallel per vertex) — walks compact chain
  K2: Boundary-interior ancestor composition
  K3: Triangle redirection + degenerate culling + emission (parallel per triangle)

One-time costs (NOT benchmarked here, done at mesh load):
  - FIFO-adjacency connectivity decode → produces per-meshlet triangle list
  - Root position decoding (chain-delta propagation)
  - Upload ancestor tables + triangle list + positions to GPU

The mesh shader equivalent would run K1+K3 per frame and emit to rasterizer.
"""

import numpy as np
import cupy as cp
import time

from reader import Reader
from encoder.implementation.meshlet_wavelet import (
    _to_numpy, _identify_meshlet_boundary_verts,
)
from encoder.implementation.meshlet_ancestry_lod_v2 import MeshletAncestryLODv2
from utils.qem import ancestors_at_lod_compact_batch


CUDA_SRC = r'''
extern "C" {

// ---------------------------------------------------------------------
// K1: Ancestor resolution (parallel per vertex)
//
// For each vertex V, walk collapse chain until we reach a vertex that is
// alive at the given threshold.
//   alive iff collapse_step[V] < 0 OR collapse_step[V] >= threshold
// ---------------------------------------------------------------------
__global__ void k1_resolve_ancestors(
    const int* __restrict__ collapse_step,   // [n_v]
    const int* __restrict__ direct_parent,   // [n_v]
    int n_v, int threshold,
    int* __restrict__ ancestors              // [n_v] output
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_v) return;

    int cur = v;
    // Cap walk depth to prevent pathological cycles.
    // Chain depth is O(log N) in practice for QEM output.
    for (int hop = 0; hop < 32; hop++) {
        int step = __ldg(&collapse_step[cur]);
        if (step < 0 || step >= threshold) break;
        cur = __ldg(&direct_parent[cur]);
    }
    ancestors[v] = cur;
}


// ---------------------------------------------------------------------
// K2: Compose boundary + interior ancestors (v2c/v2d)
//   combined[v] = boundary_ancestor[interior_ancestor[v]]
// ---------------------------------------------------------------------
__global__ void k2_compose_ancestors(
    const int* __restrict__ interior_anc,  // [n_v]
    const int* __restrict__ boundary_anc,  // [n_v]
    int n_v,
    int* __restrict__ combined              // [n_v] output
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_v) return;
    int ia = __ldg(&interior_anc[v]);
    combined[v] = __ldg(&boundary_anc[ia]);
}


// ---------------------------------------------------------------------
// K3: Triangle redirection + degenerate culling + emission
// (parallel per triangle, atomicAdd for output compaction)
// ---------------------------------------------------------------------
__global__ void k3_emit_triangles(
    const int* __restrict__ triangles,   // [n_tris, 3]
    const int* __restrict__ ancestors,   // [n_v]
    int n_tris,
    int* __restrict__ out_triangles,     // [max_tris, 3]
    int* __restrict__ out_count          // [1]
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= n_tris) return;

    int v0 = __ldg(&triangles[ti*3+0]);
    int v1 = __ldg(&triangles[ti*3+1]);
    int v2 = __ldg(&triangles[ti*3+2]);

    int a0 = __ldg(&ancestors[v0]);
    int a1 = __ldg(&ancestors[v1]);
    int a2 = __ldg(&ancestors[v2]);

    if (a0 != a1 && a1 != a2 && a0 != a2) {
        int idx = atomicAdd(out_count, 1);
        out_triangles[idx*3+0] = a0;
        out_triangles[idx*3+1] = a1;
        out_triangles[idx*3+2] = a2;
    }
}


// ---------------------------------------------------------------------
// K1 + K3 combined: single-pass per-triangle with inline ancestor walk
//
// Less GPU memory traffic but each thread walks 3 chains.
// Useful when ancestor table is not cached.
// ---------------------------------------------------------------------
__global__ void k_fused_emit(
    const int* __restrict__ triangles,      // [n_tris, 3]
    const int* __restrict__ collapse_step,  // [n_v]
    const int* __restrict__ direct_parent,  // [n_v]
    int threshold, int n_tris,
    int* __restrict__ out_triangles,
    int* __restrict__ out_count
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= n_tris) return;

    int vs[3];
    vs[0] = __ldg(&triangles[ti*3+0]);
    vs[1] = __ldg(&triangles[ti*3+1]);
    vs[2] = __ldg(&triangles[ti*3+2]);

    int anc[3];
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        int cur = vs[k];
        for (int hop = 0; hop < 32; hop++) {
            int step = __ldg(&collapse_step[cur]);
            if (step < 0 || step >= threshold) break;
            cur = __ldg(&direct_parent[cur]);
        }
        anc[k] = cur;
    }

    if (anc[0] != anc[1] && anc[1] != anc[2] && anc[0] != anc[2]) {
        int idx = atomicAdd(out_count, 1);
        out_triangles[idx*3+0] = anc[0];
        out_triangles[idx*3+1] = anc[1];
        out_triangles[idx*3+2] = anc[2];
    }
}

}
'''


def main():
    module = cp.RawModule(code=CUDA_SRC)
    k1_resolve = module.get_function("k1_resolve_ancestors")
    k2_compose = module.get_function("k2_compose_ancestors")
    k3_emit = module.get_function("k3_emit_triangles")
    k_fused = module.get_function("k_fused_emit")

    for model_path in ["assets/bunny.obj",
                       "assets/torus.obj",
                       "assets/stanford-bunny.obj",
                       "assets/Monkey.obj"]:
        try:
            model = Reader.read_from_file(model_path)
        except Exception as e:
            print(f"skip {model_path}: {e}")
            continue

        verts, tris = _to_numpy(model)
        n_v, n_t = len(verts), len(tris)
        model_name = model_path.split('/')[-1].replace('.obj', '')
        print(f"\n=== {model_name} ({n_v:,} v, {n_t:,} t) ===")

        # Encode with v2d+fifo
        enc = MeshletAncestryLODv2(variant='v2d', connectivity='fifo', verbose=False)
        result = enc.encode(model)
        print(f"  v2d+fifo BPV={result.bits_per_vertex:.2f}, "
              f"{len(result.data):,} B")

        # Extract data for GPU
        int_compact = result._interior_compact
        bnd_compact = result._boundary_compact   # v2d has this
        lod_int_thresh = result._lod_int_thresh
        lod_bnd_thresh = result._lod_bnd_thresh

        # Upload to GPU
        d_int_step = cp.asarray(int_compact["collapse_step"], dtype=cp.int32)
        d_int_parent = cp.asarray(int_compact["direct_parent"], dtype=cp.int32)
        d_bnd_step = cp.asarray(bnd_compact["collapse_step"], dtype=cp.int32)
        d_bnd_parent = cp.asarray(bnd_compact["direct_parent"], dtype=cp.int32)
        d_tris = cp.asarray(tris, dtype=cp.int32)

        d_int_anc = cp.empty(n_v, dtype=cp.int32)
        d_bnd_anc = cp.empty(n_v, dtype=cp.int32)
        d_combined = cp.empty(n_v, dtype=cp.int32)
        d_out_tris = cp.empty((n_t, 3), dtype=cp.int32)
        d_out_count = cp.zeros(1, dtype=cp.int32)

        BLOCK_V = 256
        BLOCK_T = 256
        grid_v = (n_v + BLOCK_V - 1) // BLOCK_V
        grid_t = (n_t + BLOCK_T - 1) // BLOCK_T

        # Warmup
        for _ in range(100):
            d_out_count[:] = 0
            k1_resolve((grid_v,), (BLOCK_V,),
                       (d_int_step, d_int_parent, n_v,
                        lod_int_thresh[2], d_int_anc))
            k1_resolve((grid_v,), (BLOCK_V,),
                       (d_bnd_step, d_bnd_parent, n_v,
                        lod_bnd_thresh[2], d_bnd_anc))
            k2_compose((grid_v,), (BLOCK_V,),
                       (d_int_anc, d_bnd_anc, n_v, d_combined))
            k3_emit((grid_t,), (BLOCK_T,),
                    (d_tris, d_combined, n_t, d_out_tris, d_out_count))
        cp.cuda.Stream.null.synchronize()

        # Measure per-LOD
        N_RUNS = 500
        print(f"  Per-LOD decompression timing (average over {N_RUNS} runs):")
        print(f"    {'LOD':>3} {'verts':>7} {'tris':>7} {'K1_int':>8} "
              f"{'K1_bnd':>8} {'K2':>8} {'K3':>8} {'total':>9} "
              f"{'fused':>9} {'Mtri/s':>9}")

        for lod in range(5):
            # Separate-kernel path
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                d_out_count[:] = 0
                k1_resolve((grid_v,), (BLOCK_V,),
                           (d_int_step, d_int_parent, n_v,
                            lod_int_thresh[lod], d_int_anc))
            cp.cuda.Stream.null.synchronize()
            t_k1_int = (time.perf_counter() - t0) / N_RUNS * 1e6

            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                k1_resolve((grid_v,), (BLOCK_V,),
                           (d_bnd_step, d_bnd_parent, n_v,
                            lod_bnd_thresh[lod], d_bnd_anc))
            cp.cuda.Stream.null.synchronize()
            t_k1_bnd = (time.perf_counter() - t0) / N_RUNS * 1e6

            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                k2_compose((grid_v,), (BLOCK_V,),
                           (d_int_anc, d_bnd_anc, n_v, d_combined))
            cp.cuda.Stream.null.synchronize()
            t_k2 = (time.perf_counter() - t0) / N_RUNS * 1e6

            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                d_out_count[:] = 0
                k3_emit((grid_t,), (BLOCK_T,),
                        (d_tris, d_combined, n_t, d_out_tris, d_out_count))
            cp.cuda.Stream.null.synchronize()
            t_k3 = (time.perf_counter() - t0) / N_RUNS * 1e6

            # Fused path (K1+K3 combined, single chain walk per triangle)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                d_out_count[:] = 0
                k_fused((grid_t,), (BLOCK_T,),
                        (d_tris, d_int_step, d_int_parent,
                         lod_int_thresh[lod], n_t,
                         d_out_tris, d_out_count))
            cp.cuda.Stream.null.synchronize()
            t_fused = (time.perf_counter() - t0) / N_RUNS * 1e6

            # Correctness: CPU reference
            cpu_anc = ancestors_at_lod_compact_batch(
                n_v, lod_int_thresh[lod],
                int_compact["collapse_step"], int_compact["direct_parent"])
            cpu_bnd = ancestors_at_lod_compact_batch(
                n_v, lod_bnd_thresh[lod],
                bnd_compact["collapse_step"], bnd_compact["direct_parent"])
            cpu_combined = cpu_bnd[cpu_anc]
            gpu_combined = cp.asnumpy(d_combined)
            assert np.array_equal(cpu_combined, gpu_combined), \
                f"LOD {lod}: CPU/GPU ancestor mismatch"

            # Read emitted count
            n_emitted = int(cp.asnumpy(d_out_count)[0])
            total_us = t_k1_int + t_k1_bnd + t_k2 + t_k3
            mtri_per_s = n_emitted / (total_us * 1e-6) / 1e6

            # Counts
            alive_mask = cpu_combined == np.arange(n_v)
            n_alive = int(alive_mask.sum())

            print(f"    {lod:>3} {n_alive:>7,} {n_emitted:>7,} "
                  f"{t_k1_int:>7.1f}µs {t_k1_bnd:>7.1f}µs "
                  f"{t_k2:>7.1f}µs {t_k3:>7.1f}µs "
                  f"{total_us:>8.1f}µs {t_fused:>8.1f}µs "
                  f"{mtri_per_s:>8.1f}")


if __name__ == "__main__":
    main()

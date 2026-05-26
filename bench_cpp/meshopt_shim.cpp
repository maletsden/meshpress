// Thin C wrapper around meshopt_buildMeshlets for ctypes use from Python.
// Build: see scripts/build_meshopt_shim.bat
#include "meshoptimizer.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Returns number of meshlets produced. Caller pre-allocates output buffers:
//   meshlets:           meshopt_buildMeshletsBound(...) entries (each 4 uint32: vo,to,vc,tc)
//   meshlet_vertices:   meshopt_buildMeshletsBound(...) * max_vertices uint32
//   meshlet_triangles:  meshopt_buildMeshletsBound(...) * max_triangles * 3 uint8
EXPORT size_t shim_build_meshlets(
    uint32_t* out_meshlets_packed,   // [bound*4]
    uint32_t* meshlet_vertices,
    uint8_t*  meshlet_triangles,
    const uint32_t* indices, size_t index_count,
    const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride,
    size_t max_vertices, size_t max_triangles, float cone_weight)
{
    size_t bound = meshopt_buildMeshletsBound(index_count, max_vertices, max_triangles);
    meshopt_Meshlet* mptr = (meshopt_Meshlet*) malloc(bound * sizeof(meshopt_Meshlet));
    size_t n = meshopt_buildMeshlets(mptr, meshlet_vertices, meshlet_triangles,
                                     indices, index_count,
                                     vertex_positions, vertex_count, vertex_positions_stride,
                                     max_vertices, max_triangles, cone_weight);
    for (size_t i = 0; i < n; ++i) {
        out_meshlets_packed[i*4 + 0] = mptr[i].vertex_offset;
        out_meshlets_packed[i*4 + 1] = mptr[i].triangle_offset;
        out_meshlets_packed[i*4 + 2] = mptr[i].vertex_count;
        out_meshlets_packed[i*4 + 3] = mptr[i].triangle_count;
    }
    free(mptr);
    return n;
}

EXPORT size_t shim_build_meshlets_bound(size_t index_count, size_t max_vertices, size_t max_triangles)
{
    return meshopt_buildMeshletsBound(index_count, max_vertices, max_triangles);
}

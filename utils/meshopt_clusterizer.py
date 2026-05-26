"""ctypes binding for meshopt_buildMeshlets via bench_cpp/meshopt_shim.dll.

Returns meshlets in our pipeline's expected format: a list of np.ndarray (int64)
of global-triangle ids per meshlet.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_DLL_PATH = _ROOT / "bench_cpp" / "meshopt_shim.dll"

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not _DLL_PATH.exists():
        raise FileNotFoundError(
            f"meshopt_shim.dll missing: {_DLL_PATH}. "
            "Run bench_cpp/build_meshopt_shim.bat first.")
    lib = ctypes.CDLL(str(_DLL_PATH))
    lib.shim_build_meshlets_bound.restype = ctypes.c_size_t
    lib.shim_build_meshlets_bound.argtypes = [
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.shim_build_meshlets.restype = ctypes.c_size_t
    lib.shim_build_meshlets.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # out_meshlets_packed
        ctypes.POINTER(ctypes.c_uint32),  # meshlet_vertices
        ctypes.POINTER(ctypes.c_uint8),   # meshlet_triangles
        ctypes.POINTER(ctypes.c_uint32),  # indices
        ctypes.c_size_t,                  # index_count
        ctypes.POINTER(ctypes.c_float),   # vertex_positions
        ctypes.c_size_t,                  # vertex_count
        ctypes.c_size_t,                  # vertex_positions_stride
        ctypes.c_size_t,                  # max_vertices
        ctypes.c_size_t,                  # max_triangles
        ctypes.c_float,                   # cone_weight
    ]
    _lib = lib
    return lib


def build_meshlets(verts: np.ndarray, tris: np.ndarray,
                   max_verts: int = 256, max_tris: int = 256,
                   cone_weight: float = 0.0) -> list[np.ndarray]:
    """Run meshopt_buildMeshlets; return list of arrays of global triangle ids.

    Args:
        verts: (V, 3) float32
        tris:  (T, 3) int (any int type)
        max_verts, max_tris: per-meshlet limits (<=256)
        cone_weight: 0.0 = pure spatial; >0 weights cone-fit for raster culling

    Returns:
        list of length n_meshlets; each entry is a 1D np.int64 array of triangle
        indices into the original `tris` array.
    """
    lib = _load()
    verts_f32 = np.ascontiguousarray(verts, dtype=np.float32)
    tris_u32 = np.ascontiguousarray(tris.reshape(-1), dtype=np.uint32)
    V = verts_f32.shape[0]
    T = tris.shape[0]
    ic = int(tris_u32.size)

    bound = int(lib.shim_build_meshlets_bound(ic, max_verts, max_tris))
    out_packed = np.zeros(bound * 4, dtype=np.uint32)
    ml_verts = np.zeros(bound * max_verts, dtype=np.uint32)
    ml_tris_local = np.zeros(bound * max_tris * 3, dtype=np.uint8)

    n = int(lib.shim_build_meshlets(
        out_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ml_verts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ml_tris_local.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        tris_u32.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ic,
        verts_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        V, 12,  # stride = sizeof(float)*3
        max_verts, max_tris, float(cone_weight)))

    # Build global-triangle lookup: sorted-3-tuple -> original index.
    tris_i64 = tris.astype(np.int64)
    sorted_tri = np.sort(tris_i64, axis=1)
    # Pack into a dict for O(1) lookup. ~ms-cheap; one-time cost per build.
    tri_lookup = {}
    for ti in range(T):
        key = (int(sorted_tri[ti, 0]),
               int(sorted_tri[ti, 1]),
               int(sorted_tri[ti, 2]))
        tri_lookup[key] = ti

    out = []
    used = np.zeros(T, dtype=np.bool_)
    for m in range(n):
        vo = int(out_packed[m * 4 + 0])
        to = int(out_packed[m * 4 + 1])
        # vc = int(out_packed[m*4 + 2])  # not needed for global-id recovery
        tc = int(out_packed[m * 4 + 3])
        verts_off = ml_verts[vo: vo + max_verts]  # local-vert -> global-vert
        local_tris = ml_tris_local[to: to + tc * 3].reshape(tc, 3)
        gids = np.empty(tc, dtype=np.int64)
        for k in range(tc):
            a = int(verts_off[int(local_tris[k, 0])])
            b = int(verts_off[int(local_tris[k, 1])])
            c = int(verts_off[int(local_tris[k, 2])])
            s = tuple(sorted((a, b, c)))
            gi = tri_lookup.get(s)
            if gi is None:
                raise RuntimeError(
                    f"meshopt produced triangle not in input: {s}")
            if used[gi]:
                # Each input triangle should land in exactly one meshlet.
                # If it doesn't, it's a duplicate triangle — pick any unused
                # match.
                raise RuntimeError(
                    f"triangle {s} assigned twice "
                    "(possible duplicate triangle in input)")
            used[gi] = True
            gids[k] = gi
        out.append(gids)
    return out

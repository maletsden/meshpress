"""Fast mesh loading bypassing per-vertex/triangle Python object construction.

`load_mesh_npy(path)` returns `(verts_np float64[N, 3], tris_np int64[M, 3])`:
  - First call parses the OBJ via Open3D (≈ 2x faster than the legacy
    line-by-line Vertex/Triangle reader), then writes a small `.cache.npz`
    sidecar next to the OBJ.
  - Subsequent calls load directly from the cache (≈ 100x faster than
    re-parsing the OBJ).
Cache invalidates on OBJ mtime change.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _cache_path(obj_path: Path) -> Path:
    return obj_path.with_suffix(obj_path.suffix + ".cache.npz")


def _read_obj_open3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(path))
    verts = np.asarray(m.vertices, dtype=np.float64)
    tris = np.asarray(m.triangles, dtype=np.int64)
    return verts, tris


def _read_obj_native(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Native OBJ parser preserving file vertex/triangle order.

    Vertex order matches the legacy line-by-line reader (essential for
    bit-exact BPV vs the published reference numbers). Uses regex
    extraction + the pandas C parser for the bulk text-to-number step
    (≈ 15x faster than numpy.fromstring on million-line OBJs).
    """
    import io
    import re
    import pandas as pd

    with open(path, "rb") as f:
        data = f.read()

    v_pat = re.compile(rb"^v ([^\n\r]*)", re.MULTILINE)
    v_lines = v_pat.findall(data)
    if not v_lines:
        verts = np.zeros((0, 3), dtype=np.float64)
    else:
        v_text = b"\n".join(v_lines)
        df = pd.read_csv(
            io.BytesIO(v_text), sep=r"\s+", header=None,
            dtype=np.float64, engine="c", usecols=[0, 1, 2])
        verts = df.values

    f_pat = re.compile(rb"^f ([^\n\r]*)", re.MULTILINE)
    f_lines = f_pat.findall(data)
    if not f_lines:
        return verts, np.zeros((0, 3), dtype=np.int64)

    sample = f_lines[0]
    has_slash = b"/" in sample
    arity = len(sample.split())
    uniform_arity = all(len(p.split()) == arity for p in f_lines[:512])

    if has_slash:
        # Strip "/vt/vn" → just first index. Pandas parses the result.
        f_text = b"\n".join(f_lines)
        f_text = re.sub(rb"/\d*/\d*", b"", f_text)
        f_text = re.sub(rb"/\d*", b"", f_text)
        f_lines = f_text.split(b"\n")
        sample = f_lines[0]
        arity = len(sample.split())
        uniform_arity = all(len(p.split()) == arity for p in f_lines[:512])

    if uniform_arity:
        f_text = b"\n".join(f_lines)
        df = pd.read_csv(
            io.BytesIO(f_text), sep=r"\s+", header=None,
            dtype=np.int64, engine="c",
            usecols=list(range(arity)))
        face = df.values - 1
        if arity == 3:
            return verts, face
        tris = np.empty((face.shape[0] * (arity - 2), 3), dtype=np.int64)
        for k in range(arity - 2):
            tris[k::arity - 2, 0] = face[:, 0]
            tris[k::arity - 2, 1] = face[:, k + 1]
            tris[k::arity - 2, 2] = face[:, k + 2]
        return verts, tris

    # Mixed arity polygons — per-line Python fallback (rare).
    tris_list: list[tuple[int, int, int]] = []
    for line in f_lines:
        parts = line.split()
        idxs = [int(p) - 1 for p in parts]
        for k in range(1, len(idxs) - 1):
            tris_list.append((idxs[0], idxs[k], idxs[k + 1]))
    tris = (np.asarray(tris_list, dtype=np.int64) if tris_list
            else np.zeros((0, 3), dtype=np.int64))
    return verts, tris


def load_mesh_npy(path: str | Path, use_cache: bool = True
                  ) -> tuple[np.ndarray, np.ndarray]:
    obj_path = Path(path)
    cache = _cache_path(obj_path)
    if use_cache and cache.exists():
        if cache.stat().st_mtime >= obj_path.stat().st_mtime:
            d = np.load(cache)
            return d["verts"], d["tris"]
    # Default to the native fast parser (preserves OBJ vertex order, which
    # is required for bit-exact BPV vs the published reference numbers).
    # Open3D can be ~2x faster on large clean OBJs but silently dedupes and
    # reorders vertices.
    verts, tris = _read_obj_native(obj_path)
    if use_cache:
        try:
            np.savez(cache, verts=verts, tris=tris)
        except OSError:
            pass
    return verts, tris


def clean_mesh_npy(verts: np.ndarray, tris: np.ndarray,
                   tol_rel: float = 1e-7
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-snap merge + drop degenerate / duplicate tris. Pure numpy."""
    if len(verts) == 0 or len(tris) == 0:
        return verts.astype(np.float64), tris.astype(np.int64)
    ext = float((verts.max(0) - verts.min(0)).max())
    tol = max(ext * tol_rel, 1e-12)
    keys = np.round(verts / tol).astype(np.int64)
    _, first_idx, inv = np.unique(
        keys, axis=0, return_index=True, return_inverse=True)
    new_verts = verts[first_idx]
    tris2 = inv[tris]
    mask = ((tris2[:, 0] != tris2[:, 1]) &
            (tris2[:, 1] != tris2[:, 2]) &
            (tris2[:, 0] != tris2[:, 2]))
    tris2 = tris2[mask]
    sorted_tris = np.sort(tris2, axis=1)
    _, u_idx = np.unique(sorted_tris, axis=0, return_index=True)
    u_idx = np.sort(u_idx)
    tris2 = tris2[u_idx]
    return new_verts.astype(np.float64), tris2.astype(np.int64)
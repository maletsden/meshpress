"""Per-axis bound check on STRIDE reconstruction."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import numpy as np
from scipy.spatial import cKDTree

from reader import Reader
from reader.fast_obj import clean_mesh_npy
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder


def diag(path: str, eps: float = 0.0005):
    print(f"\n=== {path} ===")
    m = Reader.read_from_file(path)
    src_raw = np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float64)
    src_faces = np.array([[t.a, t.b, t.c] for t in m.triangles], dtype=np.int64)
    src_clean, _ = clean_mesh_npy(src_raw, src_faces)

    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=eps,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    data = encode_from_prepared_v5(prep, verbose=False)
    dec = ParaDeltaV5GpuDecoder(data)
    v_dec, _ = dec.decode_to_host()
    dec_pts = np.asarray(v_dec, dtype=np.float64)

    print(f"  src_clean={len(src_clean):,}  dec={len(dec_pts):,}")
    print(f"  extent={(src_clean.max(0) - src_clean.min(0)).max():.4f}")
    print(f"  per-axis bound: eps/sqrt(3)={eps/math.sqrt(3):.6f}")
    print(f"  Euclidean bound: eps={eps:.6f}")

    # NN both directions
    t_dec = cKDTree(dec_pts)
    d_a, idx_a = t_dec.query(src_clean, k=1)
    delta_a = dec_pts[idx_a] - src_clean
    per_axis_a = np.abs(delta_a)

    print(f"  src->dec Euclidean: mean={d_a.mean():.4g}  max={d_a.max():.4g}")
    print(f"  src->dec per-axis max: x={per_axis_a[:,0].max():.4g}  "
          f"y={per_axis_a[:,1].max():.4g}  z={per_axis_a[:,2].max():.4g}")
    print(f"  src->dec verts with Euclidean > eps:       "
          f"{int((d_a>eps).sum()):,}")
    print(f"  src->dec verts with any-axis > eps/sqrt(3): "
          f"{int((per_axis_a.max(1) > eps/math.sqrt(3)).sum()):,}")
    print(f"  src->dec verts with any-axis > eps:         "
          f"{int((per_axis_a.max(1) > eps).sum()):,}")

    # Worst-axis-error sample
    j = int(np.argmax(per_axis_a.max(1)))
    print(f"  worst-axis sample: src={src_clean[j]}  dec_paired={dec_pts[idx_a[j]]}")
    print(f"                       per-axis abs={per_axis_a[j]}")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/Monkey.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/xyzrgb_dragon.obj",
        "D:/meshpress/assets/crab.obj",
    ]
    for p in paths:
        diag(p)

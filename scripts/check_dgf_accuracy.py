"""Check DGF GPU-decoded mesh accuracy vs source.

Runs CUDA DGF decode with --dump, loads the float positions, and
computes Hausdorff/RMSE/PSNR vs the source obj. Confirms precision is
in the same league as STRIDE (q12 bbox-relative target).
"""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.reader import Reader
from scripts.dump_dgf_blob import encode_to_dgfblob, BLOB_DIR

DGF_BENCH = ROOT / "bench_cpp" / "dgf_decode_bench.exe"

MESHES = [
    "assets/fandisk.obj",
    "assets/stanford-bunny.obj",
    "assets/horse.obj",
    "assets/Monkey.obj",
    "assets/happy_buddha.obj",
    "assets/crab.obj",
    "assets/tank.obj",
    "assets/xyzrgb_dragon.obj",
]


def metrics(src: np.ndarray, dec: np.ndarray) -> dict:
    from scipy.spatial import cKDTree
    bbox = float((src.max(0) - src.min(0)).max())
    # bidirectional NN since DGF reorders verts
    t1 = cKDTree(dec); d1, _ = t1.query(src, k=1)
    t2 = cKDTree(src); d2, _ = t2.query(dec, k=1)
    d_all = np.concatenate([d1, d2])
    rmse  = float(np.sqrt((d_all ** 2).mean()))
    hmean = float(d_all.mean())
    hmax  = float(d_all.max())
    psnr  = float(20 * np.log10(bbox / max(rmse, 1e-12)))
    return {
        "bbox": bbox, "rmse": rmse, "h_mean": hmean,
        "h_max": hmax, "psnr": psnr,
        "h_max_rel_bbox": hmax / bbox,
        "epsilon_q12": bbox / 4095,
    }


def main():
    paths = sys.argv[1:] or MESHES
    print(f"{'mesh':<22} {'bbox':>10} {'rmse':>10} {'h_mean':>10} "
          f"{'h_max':>10} {'PSNR':>8} {'h_max/eps':>10}")
    for p in paths:
        full = ROOT / p
        if not full.exists():
            print(f"missing: {p}"); continue
        m = Reader.read_from_file(str(full))
        src = np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float64)

        blob = BLOB_DIR / (full.stem + ".dgfblob")
        encode_to_dgfblob(full, blob, 12)
        with tempfile.TemporaryDirectory() as td:
            prefix = Path(td) / "out"
            subprocess.run(
                [str(DGF_BENCH), str(blob), "1", "1", "--dump", str(prefix)],
                capture_output=True, check=True)
            dec = np.fromfile(str(prefix) + ".verts.f32",
                                dtype=np.float32).reshape(-1, 3).astype(np.float64)
        r = metrics(src, dec)
        print(f"{full.name:<22} {r['bbox']:10.4g} {r['rmse']:10.4g} "
              f"{r['h_mean']:10.4g} {r['h_max']:10.4g} {r['psnr']:8.2f} "
              f"{r['h_max']/r['epsilon_q12']:>10.3f}")


if __name__ == "__main__":
    main()

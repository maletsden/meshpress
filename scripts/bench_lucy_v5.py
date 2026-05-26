"""Lucy v5-only bench with clean_mesh prepass.

Replaces the n/a¹ row in bench_competitors.csv. Mirrors
`bench_paradelta_v5` timing in bench_competitors.py.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.reader import Reader
from utils.mesh_clean import clean_mesh
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder

WARMUP = 20
RUNS = 100


def _max_err_nn(src_pts: np.ndarray, dec_pts: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    t1 = cKDTree(dec_pts)
    d1, _ = t1.query(src_pts, k=1)
    t2 = cKDTree(src_pts)
    d2, _ = t2.query(dec_pts, k=1)
    return float(max(d1.max(), d2.max()))


def main(path: str):
    p = Path(path)
    print(f"=== {p.name} ===")
    obj_mb = p.stat().st_size / (1024 * 1024)
    print(f"  .obj size: {obj_mb:.1f} MB")

    t0 = time.time()
    model = Reader.read_from_file(str(p))
    n_v0, n_t0 = len(model.vertices), len(model.triangles)
    print(f"  read: {time.time()-t0:.1f}s  raw verts={n_v0:,} tris={n_t0:,}")

    t0 = time.time()
    cleaned, stats = clean_mesh(model, verbose=True)
    n_v, n_t = stats["n_v_after"], stats["n_t_after"]
    print(f"  clean_mesh: {time.time()-t0:.1f}s")

    src_pts = np.asarray([[v.x, v.y, v.z] for v in cleaned.vertices],
                          dtype=np.float32)

    t0 = time.time()
    prep = load_or_prepare(str(p), max_verts=256, max_tris=256,
                            precision_error=0.0005,
                            gen_method="joint_learned",
                            strip_method="multiseed",
                            clean=True, verbose=True)
    print(f"  prep:  {time.time()-t0:.1f}s")

    t0 = time.time()
    data = encode_from_prepared_v5(prep, verbose=False)
    t_enc = time.time() - t0
    size = len(data)
    bpv = size * 8 / max(1, n_v)
    print(f"  encode: {t_enc:.1f}s  size={size:,} B  BPV={bpv:.2f}")

    dec = ParaDeltaV5GpuDecoder(data)
    for _ in range(WARMUP):
        dec.decode()
    cp.cuda.Device().synchronize()
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(RUNS):
        dec.decode()
    e.record(); e.synchronize()
    kernel_us = cp.cuda.get_elapsed_time(s, e) * 1000.0 / RUNS
    mtps = n_t / kernel_us
    print(f"  decode: {kernel_us:.1f} us  ({mtps:.0f} M tris/s)")

    v_dec, _ = dec.decode_to_host()
    err = _max_err_nn(src_pts, v_dec.astype(np.float32))
    print(f"  max NN err: {err:.4g}")

    row = dict(
        mesh=p.stem, obj_mb=f"{obj_mb:.2f}",
        n_v=n_v, n_t=n_t, name="ParaDelta v5 (ours)",
        size_b=size, bpv=f"{bpv:.4f}",
        max_err=f"{err:.6g}",
        enc_ms=f"{t_enc*1000:.1f}",
        dec_us=f"{kernel_us:.2f}",
        mtps=f"{mtps:.1f}", gpu=True,
        note="clean_mesh prepass (merged verts + degen drop)",
    )

    out_csv = ROOT / "bench_lucy_v5.csv"
    keys = ["mesh", "obj_mb", "n_v", "n_t", "name", "size_b", "bpv",
             "max_err", "enc_ms", "dec_us", "mtps", "gpu", "note"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerow(row)
    print(f"\nCSV written: {out_csv}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/lucy.obj"
    main(path)
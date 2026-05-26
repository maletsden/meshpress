"""Re-bench STRIDE only at q12-bbox, merge with cached competitor rows.

Competitors (Draco/meshopt/Corto/DGF/our legacy GlobalAMD/GlobalEB) are
mesh-precision-invariant from STRIDE's perspective — they encode their
own bbox-relative q12 grid regardless of what STRIDE does. Re-running
them is wasted compute. This script:
  1. Reads bench_competitors.csv (world mode, current paper numbers).
  2. Keeps every row whose `name` is NOT 'ParaDelta v5 (ours)'.
  3. Re-encodes STRIDE in q12-bbox mode on each mesh, with full decode
     timing + max-err verification (mirrors bench_paradelta_v5 in
     bench_competitors.py).
  4. Writes the combined CSV to bench_competitors_q12bbox.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import Reader  # noqa: E402
from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5 import encode_from_prepared_v5  # noqa: E402
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder  # noqa: E402

MESHES = [
    "fandisk.obj", "stanford-bunny.obj", "horse.obj",
    "Monkey.obj", "happy_buddha.obj", "crab.obj",
    "tank.obj", "xyzrgb_dragon.obj",
]

SRC_CSV = ROOT / "bench_competitors.csv"
OUT_CSV = ROOT / "bench_competitors_q12bbox.csv"


def _max_err_nn(src_pts, dec_pts):
    from scipy.spatial import cKDTree
    t1 = cKDTree(dec_pts)
    d1, _ = t1.query(src_pts, k=1)
    t2 = cKDTree(src_pts)
    d2, _ = t2.query(dec_pts, k=1)
    return float(max(d1.max(), d2.max()))


def _read_src_pts(path: Path):
    m = Reader.read_from_file(str(path))
    return np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float32)


def bench_stride_q12(path: Path, n_v: int, n_t: int,
                     warmup: int = 20, runs: int = 100) -> dict:
    prep = load_or_prepare(
        str(path), max_verts=256, max_tris=256,
        precision_error=1.0 / 4096.0, precision_mode="bbox_frac",
        gen_method="joint_learned", strip_method="multiseed",
        verbose=False)
    t0 = time.perf_counter()
    data = encode_from_prepared_v5(prep, verbose=False)
    t_enc = time.perf_counter() - t0
    size = len(data)
    dec = ParaDeltaV5GpuDecoder(data)
    for _ in range(warmup):
        dec.decode()
    cp.cuda.Device().synchronize()
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(runs):
        dec.decode()
    e.record(); e.synchronize()
    kernel_us = cp.cuda.get_elapsed_time(s, e) * 1000.0 / runs
    v_dec, _ = dec.decode_to_host()
    src_pts = _read_src_pts(path)
    err = _max_err_nn(src_pts, v_dec.astype(np.float32))
    return {
        "name": "ParaDelta v5 (ours)",
        "size_b": size,
        "bpv": size * 8 / max(1, n_v),
        "max_err": err,
        "enc_ms": t_enc * 1000,
        "dec_us": kernel_us,
        "mtps": n_t / kernel_us,
        "gpu": True,
        "note": "fused CUDA kernel (q12 bbox-relative)",
    }


def main():
    if not SRC_CSV.exists():
        print(f"[err] cached competitor CSV not found: {SRC_CSV}")
        sys.exit(1)

    # Read every row, group by mesh
    rows_by_mesh: dict[str, list[dict]] = {}
    fieldnames: list[str] = []
    with open(SRC_CSV, newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        for row in r:
            rows_by_mesh.setdefault(row["mesh"], []).append(row)

    out_rows = []
    for mesh in MESHES:
        rows = rows_by_mesh.get(mesh, [])
        if not rows:
            print(f"[skip] {mesh} missing from {SRC_CSV.name}")
            continue
        # Carry over every non-STRIDE row verbatim
        for row in rows:
            if row["name"] != "ParaDelta v5 (ours)":
                out_rows.append(row)
        # Re-bench STRIDE at q12-bbox
        first = rows[0]
        n_v = int(first["n_v"]); n_t = int(first["n_t"])
        path = ROOT / "assets" / mesh
        if not path.exists():
            print(f"[skip] {mesh}: asset missing")
            continue
        print(f"\n=== STRIDE q12-bbox: {mesh} ===")
        t0 = time.perf_counter()
        r = bench_stride_q12(path, n_v, n_t)
        print(f"  bpv={r['bpv']:.2f}  size={r['size_b']:,} B  "
              f"kernel={r['dec_us']:.1f} us  mtps={r['mtps']:.1f}  "
              f"max_err={r['max_err']:.4g}  ({time.perf_counter()-t0:.1f}s)")
        # Compose row with the same fieldnames as the source CSV
        merged = dict(first)  # carry mesh/obj_mb/n_v/n_t identical
        merged.update({k: ("" if v is None else str(v))
                       for k, v in r.items()})
        # Ensure all CSV fields present (skip unknown keys)
        out_rows.append({k: merged.get(k, "") for k in fieldnames})

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"\nWritten: {OUT_CSV}")


if __name__ == "__main__":
    main()

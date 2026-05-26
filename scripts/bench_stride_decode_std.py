"""Per-iter STRIDE GPU decode timing → mean + std across runs.

Differs from `bench_competitors.py`:
- Each iteration is timed individually via its own (start, end) event pair.
- Returns mean and sample std (ddof=1) in microseconds.
- 20 warmup, 100 measured runs (matches §5.1 methodology).

Output: bench_stride_decode_std{_q12bbox}.csv with columns
mesh, n_v, n_t, dec_us_mean, dec_us_std, mtps_mean, runs.
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

from reader import Reader
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder
from utils.bench_config import stride_precision, csv_suffix, mode_label
from utils.mesh_clean import clean_mesh


_PREC = stride_precision()
_SUFFIX = csv_suffix()

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

WARMUP = 20
RUNS = 100


def _bench_one(path: str) -> dict:
    model = Reader.read_from_file(path)
    model, _ = clean_mesh(model, verbose=False)
    n_v = len(model.vertices)
    n_t = len(model.triangles)

    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False,
                           **_PREC)
    data = encode_from_prepared_v5(prep, verbose=False)
    dec = ParaDeltaV5GpuDecoder(data)

    # Warmup
    for _ in range(WARMUP):
        dec.decode()
    cp.cuda.Device().synchronize()

    # Per-iter timing: queue all start/end events back-to-back, then sync once.
    starts = [cp.cuda.Event() for _ in range(RUNS)]
    ends   = [cp.cuda.Event() for _ in range(RUNS)]
    for i in range(RUNS):
        starts[i].record()
        dec.decode()
        ends[i].record()
    ends[-1].synchronize()
    times_us = np.array(
        [cp.cuda.get_elapsed_time(starts[i], ends[i]) * 1000.0
         for i in range(RUNS)],
        dtype=np.float64,
    )

    mean_us = float(times_us.mean())
    std_us  = float(times_us.std(ddof=1))
    return {
        "mesh":         Path(path).name,
        "n_v":          n_v,
        "n_t":          n_t,
        "dec_us_mean":  mean_us,
        "dec_us_std":   std_us,
        "mtps_mean":    n_t / mean_us,
        "runs":         RUNS,
    }


def main():
    print(f"[bench_stride_decode_std] precision = {mode_label()}  "
          f"warmup={WARMUP}  runs={RUNS}")
    out_path = ROOT / f"bench_stride_decode_std{_SUFFIX}.csv"
    rows = []
    for p in MESHES:
        if not (ROOT / p).exists() and not Path(p).exists():
            print(f"[skip missing] {p}")
            continue
        t0 = time.perf_counter()
        r = _bench_one(p)
        wall = time.perf_counter() - t0
        rows.append(r)
        print(f"  {r['mesh']:24s}  "
              f"n_t={r['n_t']:>10,}  "
              f"dec = {r['dec_us_mean']:>10.1f} +/- {r['dec_us_std']:>6.1f} us  "
              f"({r['mtps_mean']:>6.0f} M tris/s)  "
              f"[{wall:.1f}s wall]")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

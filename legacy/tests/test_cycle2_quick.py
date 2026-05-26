"""Cycle-2 sweep: polynomial trend degrees x meshes.

Default sort is greedy_nn (cycle-1 winner). Compares poly_degree in
{0, 1, 2, 3}; degree 0 = no trend = cycle-1 baseline. Per-meshlet 1-bit
flag picks whichever (with-trend or skip-trend) gives smaller bit cost,
so a degree never makes the encoder strictly worse than degree=0 + 1 bit
per meshlet of overhead.
"""

import time
import csv

from reader import Reader
from encoder.implementation.meshlet_wavelet import MeshletSplitFloatHaarAMD


DEGREES = [0, 1, 2, 3]
SORT = "greedy_nn"
RATIO = 4.0


def run(model, deg):
    enc = MeshletSplitFloatHaarAMD(
        max_verts=256, precision_error=0.0005,
        ratio=RATIO, sort=SORT, poly_degree=deg, verbose=False,
    )
    t0 = time.perf_counter()
    out = enc.encode(model)
    return len(out.data), out.bits_per_vertex, time.perf_counter() - t0


def main():
    rows = []
    for path in ["assets/bunny.obj", "assets/stanford-bunny.obj", "assets/Monkey.obj"]:
        print(f"\n=== {path} ===")
        model = Reader.read_from_file(path)
        n_v = len(model.vertices)
        for deg in DEGREES:
            try:
                tb, bpv, dt = run(model, deg)
                print(f"  deg={deg}  {tb:>10,} B  BPV={bpv:>6.2f}  t={dt:.2f}s")
                rows.append({
                    "model": path.split("/")[-1].rsplit(".", 1)[0],
                    "n_verts": n_v,
                    "sort": SORT,
                    "poly_degree": deg,
                    "total_bytes": tb,
                    "bpv": round(bpv, 2),
                    "encode_time_s": round(dt, 2),
                })
            except Exception as e:
                print(f"  deg={deg}  ERROR: {e}")
                raise

    with open("benchmarks_cycle2_poly.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        print(f"\nWrote benchmarks_cycle2_poly.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
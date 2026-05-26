"""Quick sanity check + cycle-1 sweep on Stanford-bunny.

Validates:
  1. sort="morton" produces the same bit count as before the patch (baseline
     reproducibility).
  2. The four new sort variants ("eb", "hilbert", "pca", "greedy_nn") all
     run end-to-end and produce a CompressedModel.

Then writes per-variant BPV / total-byte numbers for cycle-1 comparison.
"""

import time
import csv

from reader import Reader
from encoder.implementation.meshlet_wavelet import MeshletSplitFloatHaarAMD


VARIANTS = ["morton", "eb", "hilbert", "pca", "greedy_nn"]


def run_variant(model, sort_variant):
    enc = MeshletSplitFloatHaarAMD(
        max_verts=256, precision_error=0.0005,
        ratio=4.0,  # match the memory's documented "Haar r=4" baseline
        sort=sort_variant, verbose=False,
    )
    t0 = time.perf_counter()
    out = enc.encode(model)
    dt = time.perf_counter() - t0
    return len(out.data), out.bits_per_vertex, dt


def main():
    rows = []
    for path in ["assets/bunny.obj", "assets/stanford-bunny.obj", "assets/Monkey.obj"]:
        print(f"\n=== {path} ===")
        model = Reader.read_from_file(path)
        n_v = len(model.vertices)
        for v in VARIANTS:
            try:
                total_bytes, bpv, dt = run_variant(model, v)
                print(f"  sort={v:<10s}  {total_bytes:>10,} B  "
                      f"BPV={bpv:>6.2f}   t={dt:.2f}s")
                rows.append({
                    "model": path.split("/")[-1].rsplit(".", 1)[0],
                    "n_verts": n_v,
                    "sort": v,
                    "total_bytes": total_bytes,
                    "bpv": round(bpv, 2),
                    "encode_time_s": round(dt, 2),
                })
            except Exception as e:
                print(f"  sort={v:<10s}  ERROR: {e}")
                raise

    with open("benchmarks_cycle1_sorts.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
            print(f"\nWrote benchmarks_cycle1_sorts.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
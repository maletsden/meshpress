"""Cycle-3 sweep: block-DCT block sizes x schedules x meshes.

Default sort = greedy_nn (cycle-1 winner). Compares dct_block_size in
{0, 8, 16, 32} with uniform schedule, plus a geometric variant at B=16.
Block size 0 = no DCT = cycle-1 baseline. Per-meshlet 1-bit flag picks
whichever (Haar vs DCT) gives smaller bit cost per meshlet.
"""

import time
import csv

from reader import Reader
from encoder.implementation.meshlet_wavelet import MeshletSplitFloatHaarAMD


CONFIGS = [
    ("baseline_haar",     {}),
    ("dct_B8_uni",        {"dct_block_size": 8,  "dct_schedule": "uniform"}),
    ("dct_B16_uni",       {"dct_block_size": 16, "dct_schedule": "uniform"}),
    ("dct_B32_uni",       {"dct_block_size": 32, "dct_schedule": "uniform"}),
    ("dct_B16_geo_r2",    {"dct_block_size": 16, "dct_schedule": "geometric", "dct_ratio": 2.0}),
    ("dct_B16_geo_r4",    {"dct_block_size": 16, "dct_schedule": "geometric", "dct_ratio": 4.0}),
]


def run(model, kwargs):
    enc = MeshletSplitFloatHaarAMD(
        max_verts=256, precision_error=0.0005,
        ratio=4.0, sort="greedy_nn", verbose=False, **kwargs,
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
        for name, kwargs in CONFIGS:
            try:
                tb, bpv, dt = run(model, kwargs)
                print(f"  {name:<22s}  {tb:>10,} B  BPV={bpv:>6.2f}  t={dt:.2f}s")
                rows.append({
                    "model": path.split("/")[-1].rsplit(".", 1)[0],
                    "n_verts": n_v,
                    "config": name,
                    "total_bytes": tb,
                    "bpv": round(bpv, 2),
                    "encode_time_s": round(dt, 2),
                })
            except Exception as e:
                print(f"  {name:<22s}  ERROR: {e}")
                raise

    with open("benchmarks_cycle3_dct.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        print(f"\nWrote benchmarks_cycle3_dct.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()

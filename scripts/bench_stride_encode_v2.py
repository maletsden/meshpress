"""Full-corpus STRIDE encode timing (fast-reader pipeline).

Outputs per-mesh wall-clock breakdown: load, clean, prepare, encode.
Numba JIT warmed on bunny first.

Usage:
    python scripts/bench_stride_encode_v2.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.fast_obj import load_mesh_npy, clean_mesh_npy
from encoder.paradelta_codec import prepare_paradelta_arrays
from encoder.paradelta_v5 import encode_from_prepared_v5
from utils.bench_config import stride_precision, csv_suffix, mode_label

_PREC = stride_precision()
_SUFFIX = csv_suffix()
print(f"[bench_stride_encode_v2] precision = {mode_label()}")

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


def measure(path: str) -> dict:
    t0 = time.perf_counter()
    verts, tris = load_mesh_npy(path)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    verts, tris = clean_mesh_npy(verts, tris)
    t_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    prep = prepare_paradelta_arrays(
        verts, tris,
        max_verts=256, max_tris=256,
        gen_method="joint_learned", strip_method="multiseed",
        **_PREC)
    t_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    data = encode_from_prepared_v5(prep, verbose=False)
    t_enc = time.perf_counter() - t0

    return dict(
        path=path, n_v=int(prep["n_v"]), n_t=int(prep["n_t"]),
        n_meshlets=int(prep["n_meshlets"]),
        bytes=len(data), bpv=len(data) * 8 / prep["n_v"],
        t_load=t_load, t_clean=t_clean, t_prep=t_prep, t_enc=t_enc,
        t_total=t_load + t_clean + t_prep + t_enc,
    )


def main():
    # Warm JIT on bunny
    print("warming Numba JIT ...", flush=True)
    measure(str(ROOT / "assets/bunny.obj"))
    print("warm done\n", flush=True)

    paths = sys.argv[1:] or MESHES
    rows = []
    print(f"{'mesh':<22}{'n_v':>10}{'n_t':>11}  "
          f"{'load':>7} {'clean':>7} {'prep':>7} {'enc':>7} {'total':>8}  "
          f"{'BPV':>6}")
    print("-" * 95)
    for p in paths:
        full = str(ROOT / p) if not Path(p).is_absolute() else p
        if not Path(full).exists():
            print(f"missing: {full}")
            continue
        try:
            r = measure(full)
        except Exception as e:
            print(f"  ERR {Path(p).stem}: {e}")
            continue
        rows.append(r)
        name = Path(p).stem
        print(f"{name:<22}{r['n_v']:>10,}{r['n_t']:>11,}  "
              f"{r['t_load']:>6.2f}s {r['t_clean']:>6.2f}s "
              f"{r['t_prep']:>6.2f}s {r['t_enc']:>6.2f}s "
              f"{r['t_total']:>7.2f}s  {r['bpv']:>6.2f}",
              flush=True)

    csv_path = ROOT / f"bench_stride_encode_v2{_SUFFIX}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mesh", "n_v", "n_t", "n_meshlets", "bytes", "bpv",
                    "t_load_s", "t_clean_s", "t_prep_s", "t_enc_s",
                    "t_total_s"])
        for r in rows:
            w.writerow([Path(r["path"]).stem, r["n_v"], r["n_t"],
                        r["n_meshlets"], r["bytes"], f"{r['bpv']:.4f}",
                        f"{r['t_load']:.4f}", f"{r['t_clean']:.4f}",
                        f"{r['t_prep']:.4f}", f"{r['t_enc']:.4f}",
                        f"{r['t_total']:.4f}"])
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
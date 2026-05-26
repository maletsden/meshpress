"""DGF GPU decode bench sweep — mirrors bench_stride_decode_sweep.

Per mesh: encode with DGFTester.exe at tb12, repack to .dgfblob, run
dgf_decode_bench.exe (faithful CUDA port), record kernel µs + M tris/s.
Output: bench_dgf_decode_sweep.csv (suffix-aware via bench_config).

Usage:
  python scripts/bench_dgf_decode_sweep.py
  python scripts/bench_dgf_decode_sweep.py assets/stanford-bunny.obj
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.bench_config import csv_suffix
from scripts.dump_dgf_blob import encode_to_dgfblob, BLOB_DIR

DGF_BENCH_EXE = ROOT / "bench_cpp" / "dgf_decode_bench.exe"

DEFAULT_MESHES = [
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
RUNS   = 100


def run_one(mesh_path: Path) -> dict | None:
    if not mesh_path.exists():
        print(f"missing: {mesh_path}")
        return None
    blob = BLOB_DIR / (mesh_path.stem + ".dgfblob")
    n_blk, n_v, n_t, bin_sz = encode_to_dgfblob(mesh_path, blob, 12)
    bpv = bin_sz * 8 / max(1, n_v)

    r = subprocess.run(
        [str(DGF_BENCH_EXE), str(blob), str(WARMUP), str(RUNS)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        return None
    print(r.stderr.strip())  # human-readable line
    # CSV line is on stdout: path,n_v,n_t,n_blk,single_us,per_us,mtps,warmup,runs
    last = r.stdout.strip().splitlines()[-1]
    parts = last.split(",")
    single_us = float(parts[4])
    per_us    = float(parts[5])
    mtps      = float(parts[6])
    return {
        "mesh":    mesh_path.name,
        "n_v":     n_v,
        "n_t":     n_t,
        "n_blocks": n_blk,
        "bin_size_b": bin_sz,
        "bpv":     bpv,
        "single_us": single_us,
        "per_us":    per_us,
        "mtps":      mtps,
    }


def main():
    if not DGF_BENCH_EXE.exists():
        print(f"missing {DGF_BENCH_EXE}; build with bench_cpp/build_dgf_bench.bat",
              file=sys.stderr)
        sys.exit(1)

    meshes = sys.argv[1:] or DEFAULT_MESHES
    rows = []
    for m in meshes:
        r = run_one(ROOT / m)
        if r is not None:
            rows.append(r)

    out_csv = ROOT / f"bench_dgf_decode_sweep{csv_suffix()}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()

"""Per-mesh bit-budget breakdown for STRIDE v5.

Buckets (sum to real v5 bitstream size):
  header_bits          per-meshlet 4-u16 header
  boundary_bits        boundary refs table (u32 root + Rice deltas)
  connectivity_bits    strip headers + root + edge_code + reuse/global vert
  residual_bits        everything else (interior position residuals)

Wraps the diagnostic logic in diag_v5_boundary_cost.py: instead of
printing per-mesh tallies, accumulate them across the 8 paper meshes
and emit a CSV bench_bit_budget.csv.

Usage:
    python scripts/bench_bit_budget.py
"""
from __future__ import annotations

import csv
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_codec import (  # noqa: E402
    REUSE_BUF_SIZE, _idx_bits_for, _best_rice_k, _root_orient,
)
from encoder.paradelta_v5 import encode_from_prepared_v5  # noqa: E402
from utils.bench_config import stride_precision, csv_suffix, mode_label  # noqa: E402

_PREC = stride_precision()
_SUFFIX = csv_suffix()
print(f"[bench_bit_budget] precision = {mode_label()}")

_REUSE_BITS = max(1, int(math.ceil(math.log2(REUSE_BUF_SIZE + 1))))

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


def _strip_vert_cost(v: int, reuse_fifo: deque, idx_bits: int) -> int:
    if v in reuse_fifo:
        cost = 1 + _REUSE_BITS
        reuse_fifo.remove(v)
    else:
        cost = 1 + idx_bits
    reuse_fifo.append(v)
    return cost


def tally(path: str) -> dict | None:
    full = ROOT / path
    if not full.exists():
        return None
    prep = load_or_prepare(str(full), max_verts=256, max_tris=256,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False,
                           **_PREC)
    n_v = prep["n_v"]; n_t = prep["n_t"]
    plans = prep["plans"]
    n_meshlets = len(plans)

    data_v5 = encode_from_prepared_v5(prep, verbose=False)
    total_bits = len(data_v5) * 8

    header_bits = 0
    boundary_bits = 0
    conn_bits = 0

    for plan in plans:
        n_bnd = plan["n_bnd"]
        n_int = plan["n_int"]
        n_local = n_bnd + n_int
        refs = plan["refs"]
        strips = plan["strips"]
        ml_tris_local = plan["ml_tris_local"]
        idx_bits = _idx_bits_for(n_local)

        header_bits += 4 * 16

        if n_bnd > 0:
            boundary_bits += 32
            if n_bnd > 1:
                diffs = (refs[1:] - refs[:-1] - 1).astype(np.int64)
                _k, body = _best_rice_k(diffs)
                boundary_bits += 8 + body

        reuse_fifo: deque[int] = deque(maxlen=REUSE_BUF_SIZE)
        for strip in strips:
            strip_len = len(strip)
            conn_bits += 16  # strip-length header
            root_id = strip[0]
            next_id = strip[1] if strip_len > 1 else None
            root = _root_orient(ml_tris_local, root_id, next_id)
            for v in root:
                conn_bits += _strip_vert_cost(int(v), reuse_fifo, idx_bits)
            prev_tri = list(root)
            for li in strip[1:]:
                tri_v = [int(x) for x in ml_tris_local[li]]
                shared = set(tri_v) & set(prev_tri)
                new_v = next(iter(set(tri_v) - shared))
                pair_newest = frozenset((prev_tri[1], prev_tri[2]))
                if frozenset(shared) == pair_newest:
                    prev_tri = [prev_tri[1], prev_tri[2], new_v]
                else:
                    prev_tri = [prev_tri[0], prev_tri[2], new_v]
                conn_bits += 1                                     # edge_code
                conn_bits += _strip_vert_cost(new_v, reuse_fifo, idx_bits)

    accounted = header_bits + boundary_bits + conn_bits
    residual_bits = max(0, total_bits - accounted)

    return {
        "mesh": path,
        "n_v": n_v, "n_t": n_t, "n_meshlets": n_meshlets,
        "total_bits": total_bits,
        "header_bits": header_bits,
        "boundary_bits": boundary_bits,
        "connectivity_bits": conn_bits,
        "residual_bits": residual_bits,
        "bpv_total": total_bits / n_v,
        "bpv_header": header_bits / n_v,
        "bpv_boundary": boundary_bits / n_v,
        "bpv_connectivity": conn_bits / n_v,
        "bpv_residual": residual_bits / n_v,
    }


def main():
    rows = []
    for p in MESHES:
        print(f"\n=== {p} ===")
        r = tally(p)
        if r is None:
            print("  missing")
            continue
        print(f"  total={r['bpv_total']:.2f} BPV  "
              f"hdr={r['bpv_header']:.3f}  bnd={r['bpv_boundary']:.3f}  "
              f"conn={r['bpv_connectivity']:.3f}  "
              f"resid={r['bpv_residual']:.3f}")
        rows.append(r)
    out = ROOT / f"bench_bit_budget{_SUFFIX}.csv"
    with open(out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()

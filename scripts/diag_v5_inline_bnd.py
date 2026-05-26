"""Diag: hypothetical "inline boundary positions" scheme cost vs current.

Current v5:
  - Global boundary table: each unique bnd pos once (axis Morton + Rice-Δ)
  - Per-meshlet: u32 first ref + Rice-Δ on (ref[i]-ref[i-1]-1)

Alt:
  - No global bnd table, no per-meshlet refs
  - Per-meshlet: encode bnd pos codes directly (Morton-sorted within meshlet,
    axis-wise zigzag-delta + best Rice-k, same scheme as global table)

Each shared bnd vert gets encoded N times (N = meshlets using it). This
probes whether the duplication cost beats the ref-pointer cost.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import _best_rice_k
from encoder.paradelta_v5 import encode_from_prepared_v5
from utils.residual_entropy import _zigzag


def _axis_rice_bits(codes: np.ndarray, g_bits) -> int:
    """Cost of encoding (N,3) codes with: first per axis @ g_bits, then
    Rice-Δ zigzag on rest. Mirrors v5 global bnd table layout."""
    n = codes.shape[0]
    if n == 0:
        return 0
    total = 0
    for d in range(3):
        arr = codes[:, d].astype(np.int64)
        total += int(g_bits[d])  # first
        if n > 1:
            diffs = arr[1:] - arr[:-1]
            u = _zigzag(diffs)
            k, body = _best_rice_k(u)
            total += 8 + body
    return total


def _ref_table_bits(refs: np.ndarray) -> int:
    """Current scheme: u32 first + Rice on (refs[i+1]-refs[i]-1)."""
    n = len(refs)
    if n == 0:
        return 0
    total = 32
    if n > 1:
        diffs = (refs[1:] - refs[:-1] - 1).astype(np.int64)
        k, body = _best_rice_k(diffs)
        total += 8 + body
    return total


def tally(path: str) -> None:
    name = Path(path).name
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    n_v = prep["n_v"]; n_t = prep["n_t"]
    plans = prep["plans"]
    boundary_list = prep["boundary_list"]
    global_codes = prep["global_codes"]
    g_bits = prep["g_bits"]
    n_boundary = len(boundary_list)
    n_ml = len(plans)

    data_v5 = encode_from_prepared_v5(prep, verbose=False)
    real_bits = len(data_v5) * 8

    # --- Current scheme costs ---
    bnd_codes_global = global_codes[boundary_list]
    cur_global_table = _axis_rice_bits(bnd_codes_global, g_bits)
    cur_per_ml_refs = sum(_ref_table_bits(p["refs"]) for p in plans)
    cur_total_bnd_cost = cur_global_table + cur_per_ml_refs

    # --- Alt scheme: inline bnd positions per meshlet ---
    # For each meshlet, take its bnd-vert global codes, Morton-sort them
    # (already done: bnd_local order is Morton via sort_by_morton), encode
    # via the same axis-Rice scheme as the global table.
    alt_total = 0
    tot_n_bnd = 0
    for plan in plans:
        refs = plan["refs"]   # global indices into boundary_list (Morton-sorted)
        # Recover global vert ids → global codes
        bnd_codes_ml = bnd_codes_global[refs]
        alt_total += _axis_rice_bits(bnd_codes_ml, g_bits)
        tot_n_bnd += len(refs)

    print(f"\n[{name}]  n_v={n_v:,}  n_t={n_t:,}  meshlets={n_ml}")
    print(f"  real v5: {len(data_v5):,} B  ({real_bits/n_v:.2f} BPV)")
    print(f"  unique boundary verts: {n_boundary:,}")
    print(f"  per-meshlet bnd-vert sum: {tot_n_bnd:,}  "
          f"(duplication {tot_n_bnd/n_boundary:.2f}x)")
    print()
    print("  CURRENT scheme (global table + per-meshlet refs):")
    print(f"    global bnd table          : "
          f"{cur_global_table:>10,} bits ({cur_global_table/n_v:6.2f} BPV)")
    print(f"    per-meshlet refs (u32+Rice): "
          f"{cur_per_ml_refs:>10,} bits ({cur_per_ml_refs/n_v:6.2f} BPV)")
    print(f"    TOTAL boundary cost       : "
          f"{cur_total_bnd_cost:>10,} bits ({cur_total_bnd_cost/n_v:6.2f} BPV)")
    print()
    print("  ALT scheme (inline bnd positions per meshlet):")
    print(f"    no global table, no refs  : "
          f"{0:>10,} bits ({0/n_v:6.2f} BPV)")
    print(f"    inline pos (axis-Rice/ml) : "
          f"{alt_total:>10,} bits ({alt_total/n_v:6.2f} BPV)")
    print(f"    TOTAL boundary cost       : "
          f"{alt_total:>10,} bits ({alt_total/n_v:6.2f} BPV)")
    print()
    delta = alt_total - cur_total_bnd_cost
    print(f"  Δ = alt - cur = {delta:+,} bits  ({delta/n_v:+.2f} BPV)")
    print(f"  ratio alt/cur = {alt_total/max(cur_total_bnd_cost,1):.2f}x")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/bunny.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        try:
            tally(p)
        except FileNotFoundError:
            print(f"\n[{p}] not found, skipping")
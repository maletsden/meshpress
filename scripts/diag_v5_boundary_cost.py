"""Diag: per-meshlet bit accounting for v5.

Tally where the v5 bitstream spends bits per meshlet — focus on boundary
representation. Compares actual layout (implicit boundary via local-ID
ordering, n_bnd refs in u32+Rice table) against a hypothetical alternative
that adds a 1-bit boundary-flag per strip-vertex.

Conclusion target: show that the current scheme is much cheaper than any
per-vertex flag would be.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    REUSE_BUF_SIZE, _idx_bits_for, _best_rice_k, _root_orient, _fit_linear3,
)
from encoder.paradelta_v5 import (
    encode_from_prepared_v5, _interior_pass_strip,
)
from utils.residual_entropy import _zigzag


_REUSE_BITS = max(1, int(math.ceil(math.log2(REUSE_BUF_SIZE + 1))))


def _rice_body_bits(u_arr: np.ndarray, k: int) -> int:
    if len(u_arr) == 0:
        return 0
    return int(((u_arr >> k) + 1 + k).sum())


def _strip_vert_cost(v: int, reuse_fifo: deque, idx_bits: int) -> int:
    """Return bit cost of emitting v with current FIFO. Updates FIFO."""
    if v in reuse_fifo:
        cost = 1 + _REUSE_BITS
        reuse_fifo.remove(v)
    else:
        cost = 1 + idx_bits
    reuse_fifo.append(v)
    return cost


def tally(path: str) -> None:
    name = Path(path).name
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    n_v = prep["n_v"]; n_t = prep["n_t"]
    plans = prep["plans"]
    n_meshlets = len(plans)

    # Real v5 bitstream size for sanity
    data_v5 = encode_from_prepared_v5(prep, verbose=False)
    total_real_bits = len(data_v5) * 8

    tot_header = 0
    tot_bnd_refs = 0
    tot_conn_strip_root = 0
    tot_conn_strip_body = 0  # edge_code + vert (non-root)
    tot_edge_code_only = 0   # subset of body: just the 1-bit edge_code
    tot_strip_len = 0
    tot_n_bnd = 0
    tot_n_int = 0
    tot_n_strip_verts = 0    # how many vert-emits happen total
    tot_idx_bits_weighted = 0  # avg idx_bits * verts (alt baseline)

    for plan in plans:
        n_bnd = plan["n_bnd"]
        n_int = plan["n_int"]
        n_strips = plan["n_strips"]
        n_local = n_bnd + n_int
        refs = plan["refs"]
        strips = plan["strips"]
        ml_tris_local = plan["ml_tris_local"]
        idx_bits = _idx_bits_for(n_local)

        # Header
        tot_header += 4 * 16

        # Boundary refs
        bits = 0
        if n_bnd > 0:
            bits += 32
            if n_bnd > 1:
                diffs = (refs[1:] - refs[:-1] - 1).astype(np.int64)
                k, body = _best_rice_k(diffs)
                bits += 8 + body
        tot_bnd_refs += bits

        # Connectivity
        reuse_fifo: deque[int] = deque(maxlen=REUSE_BUF_SIZE)
        strip_len_bits = 0
        root_bits = 0
        body_bits = 0
        edge_bits = 0
        n_emits = 0
        for strip in strips:
            strip_len = len(strip)
            tot_strip_len += 1
            strip_len_bits += 16
            root_id = strip[0]
            next_id = strip[1] if strip_len > 1 else None
            root = _root_orient(ml_tris_local, root_id, next_id)
            for v in root:
                root_bits += _strip_vert_cost(int(v), reuse_fifo, idx_bits)
                n_emits += 1
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
                edge_bits += 1
                body_bits += 1 + _strip_vert_cost(new_v, reuse_fifo, idx_bits)
                n_emits += 1
        tot_conn_strip_root += strip_len_bits + root_bits
        tot_conn_strip_body += body_bits
        tot_edge_code_only += edge_bits
        tot_n_bnd += n_bnd
        tot_n_int += n_int
        tot_n_strip_verts += n_emits
        tot_idx_bits_weighted += idx_bits * n_emits

    # Hypothetical alt: 1 boundary-bit flag PER strip-vertex emit
    alt_extra_bits = tot_n_strip_verts  # 1 bit each

    print(f"\n[{name}]  n_v={n_v:,}  n_t={n_t:,}  meshlets={n_meshlets:,}")
    print(f"  real v5 size: {len(data_v5):,} B  ({total_real_bits/n_v:.2f} BPV)")
    print(f"  total strip-vert emits across all meshlets: {tot_n_strip_verts:,}")
    print(f"  total n_bnd across meshlets: {tot_n_bnd:,}  "
          f"(avg {tot_n_bnd/n_meshlets:.1f}/meshlet)")
    print(f"  total n_int across meshlets: {tot_n_int:,}  "
          f"(avg {tot_n_int/n_meshlets:.1f}/meshlet)")
    print()
    print("  Per-section bit accounting (per-meshlet portion of v5 stream):")
    print(f"    header (4×u16 × n_ml)       : "
          f"{tot_header:>12,} bits  ({tot_header/n_v:6.2f} bpv)")
    print(f"    boundary refs (u32 + Rice)  : "
          f"{tot_bnd_refs:>12,} bits  ({tot_bnd_refs/n_v:6.2f} bpv)")
    print(f"    strip headers + root verts  : "
          f"{tot_conn_strip_root:>12,} bits  ({tot_conn_strip_root/n_v:6.2f} bpv)")
    print(f"    strip body (edge_code+vert) : "
          f"{tot_conn_strip_body:>12,} bits  ({tot_conn_strip_body/n_v:6.2f} bpv)")
    print(f"      ↳ edge_code only (1b/tri) : "
          f"{tot_edge_code_only:>12,} bits  ({tot_edge_code_only/n_v:6.2f} bpv)")
    print()
    print("  Boundary representation cost summary:")
    print(f"    actual = bnd-refs table     : "
          f"{tot_bnd_refs:>12,} bits  "
          f"({tot_bnd_refs/max(tot_n_bnd,1):.2f} bits/bnd-vert)")
    print(f"    if we ADDED 1-bit flag per  : "
          f"{alt_extra_bits:>12,} bits  "
          f"(+{alt_extra_bits/n_v:.2f} BPV)")
    print(f"    ratio (alt extra / actual)  : "
          f"{alt_extra_bits/max(tot_bnd_refs,1):.2f}x")
    print()
    print("  Note: alt scheme would ADD bits on top of the existing refs.")
    print("        Current scheme uses ZERO per-vertex flag bits — boundary")
    print("        status is implicit from local-ID ordering (v < n_bnd).")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/stanford-bunny.obj",
    ]
    for p in paths:
        tally(p)
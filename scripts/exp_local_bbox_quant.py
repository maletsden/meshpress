"""Per-meshlet bbox quantization — bytes-only estimate.

For each meshlet:
  bbox (3 × float, 12 B) + 3 × 8-bit cell-count (3 B) overhead
  N_verts × 12-bit × 3 = 4.5 B/vert raw cell codes

Compare to v5 actual size and Shannon-on-codes.

This is a no-prediction baseline — answers "could we beat v5 by
dropping parallelogram and just raw-quant per meshlet?".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare


def _process(prep, name: str):
    plans = prep["plans"]
    vn = prep["vn"]
    n_v = prep["n_v"]
    n_meshlets = len(plans)
    scale = float(prep["scale"])

    overhead_bytes = 0
    cell_bytes = 0
    cell_bits_per_axis = 12  # match Draco / Corto / DGF baseline

    for plan in plans:
        l2g = plan["local_to_global"]
        n_local = plan["n_bnd"] + plan["n_int"]
        v_local = vn[l2g[:n_local]]
        ext = v_local.max(axis=0) - v_local.min(axis=0)
        # Header: 6 floats (bbox min/max) = 24 B
        overhead_bytes += 24
        # Raw cells: n_local * 12 bits * 3 axes
        cell_bytes += int(np.ceil(n_local * cell_bits_per_axis * 3 / 8))

    total = overhead_bytes + cell_bytes
    bpv = total * 8 / n_v
    print(f"\n=== {name}  n_v={n_v:,}  meshlets={n_meshlets:,} ===")
    print(f"  pure raw-bbox quant @ 12-bit/axis:")
    print(f"    overhead (per-meshlet bbox header): "
          f"{overhead_bytes:>10,} B")
    print(f"    raw cell payload:                   "
          f"{cell_bytes:>10,} B")
    print(f"    total:                              "
          f"{total:>10,} B")
    print(f"    BPV = {bpv:.2f}")
    print(f"  Note: NO connectivity included (would add 1.5-6 bpv).")
    print(f"  Note: NO prediction. NO residual compression.")
    print(f"  Note: This is the 'best possible' floor for non-predictive "
          f"per-meshlet codecs at q12.")


def main():
    meshes = [
        ("bunny",          "assets/bunny.obj"),
        ("Monkey",         "assets/Monkey.obj"),
        ("tank",           "assets/tank.obj"),
        ("xyzrgb_dragon",  "assets/xyzrgb_dragon.obj"),
    ]
    for name, path in meshes:
        prep = load_or_prepare(path, max_verts=256, max_tris=256,
                                precision_error=0.0005,
                                gen_method="joint_learned",
                                strip_method="multiseed",
                                clean=True, verbose=False)
        _process(prep, name)


if __name__ == "__main__":
    main()
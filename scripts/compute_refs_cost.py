"""Compute the EXACT cost of the per-meshlet boundary-refs table in v5.

For each mesh, walks every meshlet's refs (local→global mapping for
boundary verts, sorted) and computes the bit cost of v5's encoding:

  refs[0]            : u32  (32 bits)
  rice_k             : u8   (8 bits)  -- if n_bnd > 1
  rice on (delta-1)  : variable        -- one Rice code per remaining ref

Output per mesh: total refs BPV. With this we can compute the EXACT
projected dup BPV = v5_BPV - boundary_BPV - refs_BPV + dup_vert_BPV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_codec import _best_rice_k  # noqa: E402

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


def rice_total_bits(u_arr: np.ndarray, k: int) -> int:
    q = (u_arr >> k).astype(np.int64)
    return int((q + 1 + k).sum())


def refs_cost_meshlet(refs: np.ndarray) -> int:
    """Returns total bits of v5 refs encoding for one meshlet."""
    n = refs.shape[0]
    if n == 0:
        return 0
    bits = 32          # first ref u32
    if n > 1:
        diffs = (refs[1:] - refs[:-1] - 1).astype(np.int64)  # ≥ 0
        k, _ = _best_rice_k(diffs)
        bits += 8       # rice_k header
        bits += rice_total_bits(diffs, k)
    return bits


def main():
    paths = sys.argv[1:] or MESHES
    print(f"{'mesh':<22} {'n_v':>10} {'n_meshlets':>10} "
          f"{'refs_bits':>12} {'refs_BPV':>10}")
    rows = []
    for p in paths:
        full = ROOT / p
        prep = load_or_prepare(str(full), max_verts=256, max_tris=256,
                                precision_error=1.0/4096.0,
                                precision_mode="bbox_frac",
                                gen_method="joint_learned",
                                strip_method="multiseed", verbose=False)
        n_v = prep["n_v"]
        plans = prep["plans"]
        total = 0
        for plan in plans:
            refs = np.asarray(plan["refs"], dtype=np.int64)
            total += refs_cost_meshlet(refs)
        bpv = total / n_v
        print(f"{full.name:<22} {n_v:>10,} {len(plans):>10,} "
              f"{total:>12,} {bpv:>10.3f}")
        rows.append({"mesh": full.name, "n_v": n_v,
                     "n_meshlets": len(plans),
                     "refs_bits": total, "refs_bpv": bpv})

    out = ROOT / "bench_refs_cost.csv"
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()

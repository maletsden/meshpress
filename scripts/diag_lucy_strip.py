"""Diagnose Lucy v5 strip coverage.

Hypothesis: `_strip_traversal(plan)` produces an `order` list whose
length is < plan['n_int'] for some meshlets. That would mean some
interior vertices are never emitted as residuals — their positions
come out as zero in the bitstream and produce garbage verts on
decode (max NN err 29.74).

Walk every meshlet, count coverage gaps.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import _strip_traversal


def main(path: str = "assets/lucy.obj"):
    print(f"Loading prep for {path} ...")
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                            precision_error=0.0005,
                            gen_method="joint_learned",
                            strip_method="multiseed",
                            clean=True, verbose=False)
    plans = prep["plans"]
    n_meshlets = len(plans)
    print(f"  {n_meshlets:,} meshlets")

    n_bad = 0
    worst_gap = 0
    total_int = 0
    total_emitted = 0
    n_para = 0
    n_none = 0
    gap_distribution = {}
    bad_examples = []

    for mid, plan in enumerate(plans):
        n_bnd = plan["n_bnd"]
        n_int = plan["n_int"]
        total_int += n_int
        if n_int == 0:
            continue
        order = _strip_traversal(plan["ml_tris_local"],
                                   plan["strips"], n_bnd)
        n_emitted = len(order)
        total_emitted += n_emitted
        for _, kind, _ in order:
            if kind == 'para':
                n_para += 1
            else:
                n_none += 1
        gap = n_int - n_emitted
        if gap != 0:
            n_bad += 1
            worst_gap = max(worst_gap, abs(gap))
            gap_distribution[gap] = gap_distribution.get(gap, 0) + 1
            if len(bad_examples) < 5:
                bad_examples.append((mid, n_bnd, n_int, n_emitted, gap))

    print(f"\n=== Strip coverage diagnostic ===")
    print(f"  total n_int across meshlets: {total_int:,}")
    print(f"  total emitted by strip walk: {total_emitted:,}")
    print(f"  emitted-vs-n_int delta:      "
          f"{total_emitted - total_int:+,}")
    print(f"  meshlets with gap (n_int != n_emitted): "
          f"{n_bad:,} / {n_meshlets:,} "
          f"({100*n_bad/max(1,n_meshlets):.2f}%)")
    print(f"  worst single-meshlet gap (|diff|):       {worst_gap}")
    print(f"  para emits: {n_para:,}   none emits: {n_none:,}")
    if gap_distribution:
        print(f"\n  Gap histogram (n_int - n_emitted -> count):")
        for g in sorted(gap_distribution):
            print(f"    gap={g:+5d}: {gap_distribution[g]:,} meshlets")
    if bad_examples:
        print(f"\n  Sample bad meshlets:")
        for mid, n_bnd, n_int, n_em, gap in bad_examples:
            print(f"    meshlet {mid}: n_bnd={n_bnd}  n_int={n_int}  "
                  f"emitted={n_em}  gap={gap:+d}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "assets/lucy.obj"
    main(p)
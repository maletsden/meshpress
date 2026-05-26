"""Diagnostic: per-meshlet level distribution for wave-parallel predictor.

Computes level[v] using same rule kernel would:
  - anchor: level 0
  - delta (kind='none', non-anchor): level = level[prev_walk_vert] + 1
  - para (kind='para'): level = max(level[a], level[b], level[c]) + 1

Reports histogram per mesh: max_level, avg verts/level, distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5_dup import _walk_meshlet


def analyze(mesh_path: str):
    prep = load_or_prepare(mesh_path, max_verts=256, max_tris=256,
                           precision_error=1.0/4096.0,
                           precision_mode="bbox_frac",
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    plans = prep["plans"]
    all_max_lvl = []
    all_avg_per_lvl = []
    all_max_per_lvl = []
    all_n_local = []
    histogram = np.zeros(64, dtype=np.int64)  # bucket by verts/level
    for plan in plans:
        walk = _walk_meshlet(plan)
        n_local = plan["n_bnd"] + plan["n_int"]
        level = np.full(n_local, -1, dtype=np.int32)
        prev_v = None
        first_kind0 = True
        for v, kind, refs in walk:
            if kind == 'none':
                if first_kind0:
                    level[v] = 0
                    first_kind0 = False
                else:
                    level[v] = level[prev_v] + 1
            else:
                a, b, c = refs
                level[v] = max(level[a], level[b], level[c]) + 1
            prev_v = v
        max_lvl = int(level.max())
        all_max_lvl.append(max_lvl)
        # verts per level
        counts = np.bincount(level, minlength=max_lvl + 1)
        all_avg_per_lvl.append(len(level) / (max_lvl + 1))
        all_max_per_lvl.append(int(counts.max()))
        all_n_local.append(n_local)
        for c in counts:
            if c < 64:
                histogram[c] += 1
            else:
                histogram[63] += 1
    avg_n_local = float(np.mean(all_n_local))
    avg_max = float(np.mean(all_max_lvl))
    avg_per = float(np.mean(all_avg_per_lvl))
    max_per_max = float(np.max(all_max_per_lvl))
    print(f"\n=== {Path(mesh_path).stem} ({len(plans)} meshlets) ===")
    print(f"  avg n_local = {avg_n_local:.1f}")
    print(f"  avg max_level = {avg_max:.1f}")
    print(f"  avg verts/level = {avg_per:.2f}")
    print(f"  max single-level cluster (across meshlets) = {max_per_max:.0f}")
    print(f"  histogram (verts-per-level → count):")
    for c in range(min(16, 64)):
        if histogram[c] > 0:
            print(f"    {c:2d} verts: {histogram[c]:,}")


if __name__ == "__main__":
    meshes = sys.argv[1:] or ["assets/Monkey.obj", "assets/xyzrgb_dragon.obj"]
    for m in meshes:
        analyze(m)

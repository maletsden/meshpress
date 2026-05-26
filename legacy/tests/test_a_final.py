"""
Final comparison of Approach A variants + connectivity options.

Matrix:
  Variants: v1 (full table baseline), v2a, v2b, v2c, v2d
  Connectivity: gts_est (estimate), fifo (real FIFO-adj), amd_gts (real GTS)

Outputs:
  - benchmarks_a_final.csv
  - assets/plots/A_final_<model>.png
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from reader import Reader
from encoder.implementation.meshlet_wavelet import _to_numpy
from encoder.implementation.meshlet_ancestry_lod import (
    MeshletAncestryLOD, decode_at_lod,
)
from encoder.implementation.meshlet_ancestry_lod_v2 import MeshletAncestryLODv2
from utils.qem import ancestors_at_lod_compact_batch


CSV_PATH = "benchmarks_a_final.csv"


def decode_v2(verts_np, tris_np, result_v2, lod_level):
    interior = result_v2._interior_compact
    boundary = result_v2._boundary_compact
    int_thresh = result_v2._lod_int_thresh[lod_level]
    n_v = len(verts_np)
    int_anc = ancestors_at_lod_compact_batch(
        n_v, int_thresh,
        interior["collapse_step"], interior["direct_parent"])
    if boundary is not None:
        bnd_thresh = result_v2._lod_bnd_thresh[lod_level]
        bnd_anc = ancestors_at_lod_compact_batch(
            n_v, bnd_thresh,
            boundary["collapse_step"], boundary["direct_parent"])
        combined = bnd_anc[int_anc]
    else:
        combined = int_anc
    alive_ids = np.where(combined == np.arange(n_v))[0]
    g2c = {int(v): i for i, v in enumerate(alive_ids)}
    out_verts = verts_np[alive_ids]
    aa = combined[tris_np[:, 0]]; bb = combined[tris_np[:, 1]]; cc = combined[tris_np[:, 2]]
    keep = (aa != bb) & (bb != cc) & (aa != cc)
    tri_idx = np.where(keep)[0]
    out_tris = np.stack([aa[tri_idx], bb[tri_idx], cc[tri_idx]], axis=1)
    out_tris_compact = np.vectorize(g2c.get)(out_tris)
    return out_verts, out_tris_compact


def run_model(model_path, out_plot_path):
    model = Reader.read_from_file(model_path)
    verts, tris = _to_numpy(model)
    n_v, n_t = len(verts), len(tris)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    print(f"\n=== {model_name} ({n_v:,} v, {n_t:,} t) ===")

    rows = []
    # Always use v2d as the "best variant" with different connectivity
    configs = [
        ("v1",          lambda: MeshletAncestryLOD(verbose=False)),
        ("v2a",         lambda: MeshletAncestryLODv2(variant='v2a', connectivity='gts_est', verbose=False)),
        ("v2b",         lambda: MeshletAncestryLODv2(variant='v2b', connectivity='gts_est', verbose=False)),
        ("v2c",         lambda: MeshletAncestryLODv2(variant='v2c', connectivity='gts_est', verbose=False)),
        ("v2d",         lambda: MeshletAncestryLODv2(variant='v2d', connectivity='gts_est', verbose=False)),
        ("v2d+fifo",    lambda: MeshletAncestryLODv2(variant='v2d', connectivity='fifo',    verbose=False)),
        ("v2d+amd_gts", lambda: MeshletAncestryLODv2(variant='v2d', connectivity='amd_gts', verbose=False)),
    ]

    results = {}
    for name, factory in configs:
        enc = factory()
        t0 = time.perf_counter()
        r = enc.encode(model)
        t_ms = (time.perf_counter() - t0) * 1000
        results[name] = r
        print(f"  {name:<14}: BPV={r.bits_per_vertex:>6.2f}, "
              f"{len(r.data):>9,} B, {t_ms:>5.0f} ms")
        for k, s in enumerate(r._lod_stats) if hasattr(r, '_lod_stats') else []:
            rows.append({
                "model": model_name, "config": name, "lod": k,
                "bpv": round(r.bits_per_vertex, 2),
                "total_bytes": len(r.data),
                "encode_time_ms": round(t_ms, 0),
                "lod_verts": s["n_verts"],
                "lod_tris":  s["n_tris"],
            })

    # Plot: single row per model showing v2d+fifo at all LODs (best variant)
    if 'bunny' in model_name.lower():
        elev, azim = 20, -60
    elif 'torus' in model_name.lower():
        elev, azim = 30, -45
    else:
        elev, azim = 25, -55

    best = results['v2d+fifo']
    n_lod = len(best._lod_stats)

    fig, axes = plt.subplots(1, n_lod, figsize=(n_lod * 3.2, 3.5),
                              subplot_kw={'projection': '3d'})
    fig.suptitle(f'{model_name} — v2d+fifo (best): BPV={best.bits_per_vertex:.2f}, '
                 f'{len(best.data):,} B '
                 f'(orig {n_v:,}v / {n_t:,}t)', fontsize=12)

    for k in range(n_lod):
        ax = axes[k]
        v_out, t_out = decode_v2(verts, tris, best, k)
        if len(t_out) > 0:
            polys = [v_out[t_out[fi]] for fi in range(len(t_out))]
            col = Poly3DCollection(polys, alpha=0.85, linewidths=0.1,
                                    edgecolors='#333333')
            col.set_facecolor('#4a90d9')
            ax.add_collection3d(col)
        for d, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            setter(verts[:, d].min(), verts[:, d].max())
        ax.view_init(elev=elev, azim=azim)
        s = best._lod_stats[k]
        ax.set_title(f"LOD {k}\n{s['n_verts']:,}v {s['n_tris']:,}t", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_plot_path}")
    return rows


def main():
    os.makedirs('assets/plots', exist_ok=True)
    model_paths = [
        'assets/bunny.obj',
        'assets/torus.obj',
        'assets/stanford-bunny.obj',
    ]
    all_rows = []
    for mp in model_paths:
        if not os.path.exists(mp):
            print(f"[SKIP] {mp}"); continue
        name = os.path.splitext(os.path.basename(mp))[0]
        out = f'assets/plots/A_final_{name}.png'
        rows = run_model(mp, out)
        all_rows.extend(rows)

    if all_rows:
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults: {CSV_PATH} ({len(all_rows)} rows)")

    # Summary: BPV progression table
    print("\n== BPV by config ==")
    seen_model = {}
    for r in all_rows:
        m = r['model']
        seen_model.setdefault(m, {})
        seen_model[m][r['config']] = r['bpv']

    cfgs = ['v1', 'v2a', 'v2b', 'v2c', 'v2d', 'v2d+fifo', 'v2d+amd_gts']
    print(f"{'Model':<18}" + "".join(f"{c:>13}" for c in cfgs))
    print('-' * (18 + 13 * len(cfgs)))
    for m, bpvs in seen_model.items():
        print(f"{m:<18}" + "".join(f"{bpvs.get(c, 0):>13.2f}" for c in cfgs))


if __name__ == "__main__":
    main()

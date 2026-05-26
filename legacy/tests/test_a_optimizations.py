"""
Compare Approach A variants: v1 (full table), v2a (compact), v2b (+ local quant),
v2c (+ boundary LOD).

Outputs:
  - benchmarks_a_opt.csv
  - assets/plots/compare_A_variants_<model>.png  (4 rows × 5 LODs)
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


CSV_PATH = "benchmarks_a_opt.csv"
N_LOD = 5


def decode_v2(verts_np, tris_np, result_v2, lod_level):
    """Decode v2 compressed mesh at the given LOD."""
    interior = result_v2._interior_compact
    boundary = result_v2._boundary_compact
    int_thresh = result_v2._lod_int_thresh[lod_level]
    bnd_thresh = result_v2._lod_bnd_thresh[lod_level] if boundary is not None else 0

    n_v = len(verts_np)
    int_anc = ancestors_at_lod_compact_batch(
        n_v, int_thresh,
        interior["collapse_step"], interior["direct_parent"])

    if boundary is not None:
        bnd_anc = ancestors_at_lod_compact_batch(
            n_v, bnd_thresh,
            boundary["collapse_step"], boundary["direct_parent"])
        # Compose: interior first, then boundary (so boundary verts after interior
        # redirection also go through boundary LOD)
        combined = bnd_anc[int_anc]
    else:
        combined = int_anc

    alive_ids = np.where(combined == np.arange(n_v))[0]
    g2c = {int(v): i for i, v in enumerate(alive_ids)}
    out_verts = verts_np[alive_ids]

    aa = combined[tris_np[:, 0]]
    bb = combined[tris_np[:, 1]]
    cc = combined[tris_np[:, 2]]
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

    variants = [
        ("v1",  lambda: MeshletAncestryLOD(max_verts=256, precision_error=0.0005,
                                             n_lod_levels=N_LOD, verbose=False)),
        ("v2a", lambda: MeshletAncestryLODv2(max_verts=256, precision_error=0.0005,
                                              n_lod_levels=N_LOD, variant='v2a', verbose=False)),
        ("v2b", lambda: MeshletAncestryLODv2(max_verts=256, precision_error=0.0005,
                                              n_lod_levels=N_LOD, variant='v2b', verbose=False)),
        ("v2c", lambda: MeshletAncestryLODv2(max_verts=256, precision_error=0.0005,
                                              n_lod_levels=N_LOD, variant='v2c', verbose=False)),
    ]

    results = {}
    rows = []
    meshes = {}  # variant → list of (verts, tris) per LOD

    for name, factory in variants:
        enc = factory()
        t0 = time.perf_counter()
        res = enc.encode(model)
        t_ms = (time.perf_counter() - t0) * 1000
        results[name] = res
        print(f"  {name:<5s}: BPV={res.bits_per_vertex:>6.2f}, "
              f"{len(res.data):>7,} B, {t_ms:>5.0f} ms")

        # Decode at each LOD for visualization
        ms = []
        for k in range(N_LOD):
            if name == 'v1':
                v_out, t_out = decode_at_lod(verts, tris, res._ancestors, k)
            else:
                v_out, t_out = decode_v2(verts, tris, res, k)
            ms.append((v_out, t_out))
        meshes[name] = ms

        for k, s in enumerate(res._lod_stats):
            rows.append({
                "model": model_name,
                "variant": name,
                "lod": k,
                "bpv": round(res.bits_per_vertex, 2),
                "total_bytes": len(res.data),
                "encode_time_ms": round(t_ms, 0),
                "lod_verts": s["n_verts"],
                "lod_tris":  s["n_tris"],
            })

    # Visualization: 4 rows × 5 LODs
    if 'bunny' in model_name.lower():
        elev, azim = 20, -60
    elif 'torus' in model_name.lower():
        elev, azim = 30, -45
    else:
        elev, azim = 25, -55

    fig, axes = plt.subplots(4, N_LOD, figsize=(N_LOD * 3.2, 12.5),
                              subplot_kw={'projection': '3d'})
    colors = {'v1': '#4a90d9', 'v2a': '#5cb85c',
              'v2b': '#f0ad4e', 'v2c': '#d9534f'}

    fig.suptitle(f'{model_name} — Approach A variants comparison '
                 f'({n_v:,} verts, {n_t:,} tris)', fontsize=13)

    for r, (vname, _) in enumerate(variants):
        bpv = results[vname].bits_per_vertex
        for k in range(N_LOD):
            ax = axes[r, k]
            v_out, t_out = meshes[vname][k]
            if len(t_out) > 0 and len(v_out) > 0:
                polys = [v_out[t_out[fi]] for fi in range(len(t_out))]
                collection = Poly3DCollection(polys, alpha=0.85, linewidths=0.1,
                                              edgecolors='#333333')
                collection.set_facecolor(colors[vname])
                ax.add_collection3d(collection)

            for d, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                setter(verts[:, d].min(), verts[:, d].max())
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f'{vname} LOD{k}\n{len(v_out):,}v {len(t_out):,}t',
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

        # Row label
        axes[r, 0].text2D(-0.12, 0.5, f'{vname}\n{bpv:.1f} BPV',
                          transform=axes[r, 0].transAxes,
                          fontsize=11, fontweight='bold',
                          verticalalignment='center')

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
            print(f"[SKIP] {mp}")
            continue
        name = os.path.splitext(os.path.basename(mp))[0]
        out = f'assets/plots/compare_A_variants_{name}.png'
        rows = run_model(mp, out)
        all_rows.extend(rows)

    if all_rows:
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults written to {CSV_PATH} ({len(all_rows)} rows)")

    # Summary table
    print("\n== Summary: BPV by variant ==")
    print(f"{'Model':<20} {'v1':>8} {'v2a':>8} {'v2b':>8} {'v2c':>8}")
    print('-' * 60)
    seen = set()
    by_model = {}
    for r in all_rows:
        m = r['model']
        if m not in by_model:
            by_model[m] = {}
        by_model[m][r['variant']] = r['bpv']
    for m, bpvs in by_model.items():
        print(f"{m:<20} {bpvs.get('v1', 0):>8.2f} "
              f"{bpvs.get('v2a', 0):>8.2f} {bpvs.get('v2b', 0):>8.2f} "
              f"{bpvs.get('v2c', 0):>8.2f}")


if __name__ == "__main__":
    main()

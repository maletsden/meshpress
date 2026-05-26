"""
Compare Approach A (Ancestor Redirection) vs Approach C (Progressive Mesh).

Outputs:
  - benchmarks_ac.csv: BPV, ratio, LOD vert/tri counts for both approaches
  - assets/plots/compare_<model>.png: side-by-side LOD reconstructions
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
from encoder.implementation.meshlet_progressive_lod import (
    MeshletProgressiveLOD, decode_progressive,
)


CSV_PATH = "benchmarks_ac.csv"
N_LOD = 5


def compare_model(model_path, out_plot_path, n_lod=N_LOD):
    model = Reader.read_from_file(model_path)
    verts, tris = _to_numpy(model)
    n_v, n_t = len(verts), len(tris)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    print(f"\n=== {model_name} ({n_v:,} verts, {n_t:,} tris) ===")

    rows = []

    # --- Approach A ---
    print("Approach A: Ancestor Redirection...")
    t0 = time.perf_counter()
    enc_a = MeshletAncestryLOD(max_verts=256, precision_error=0.0005,
                                n_lod_levels=n_lod, verbose=False)
    result_a = enc_a.encode(model)
    t_a = time.perf_counter() - t0
    print(f"  BPV = {result_a.bits_per_vertex:.2f}, "
          f"size = {len(result_a.data):,} B, time = {t_a*1000:.0f}ms")

    # --- Approach C ---
    print("Approach C: Progressive Mesh...")
    t0 = time.perf_counter()
    enc_c = MeshletProgressiveLOD(max_verts=256, precision_error=0.0005,
                                   n_lod_levels=n_lod, verbose=False)
    result_c = enc_c.encode(model)
    t_c = time.perf_counter() - t0
    print(f"  BPV = {result_c.bits_per_vertex:.2f}, "
          f"size = {len(result_c.data):,} B, time = {t_c*1000:.0f}ms")

    # --- Per-LOD reconstruction ---
    # Approach A decoded meshes
    a_meshes = []
    orig_verts_q = result_a._qem["original_verts"]  # original positions
    for k in range(n_lod):
        v_out, t_out = decode_at_lod(orig_verts_q, tris, result_a._ancestors, k)
        a_meshes.append((v_out, t_out))

    # Approach C decoded meshes
    c_meshes = []
    base_pos = result_c._qem["base_positions"]
    base_vids = result_c._qem["base_vert_ids"]
    base_tris_ar = result_c._qem["base_tris"]
    records = result_c._qem["records"]
    for k in range(n_lod):
        n_splits = result_c._lod_n_splits[k]
        v_out, t_out, _ = decode_progressive(
            base_pos, base_vids, base_tris_ar, records, n_splits, n_v)
        c_meshes.append((v_out, t_out))

    raw_bytes = n_v * 12 + n_t * 12
    for k in range(n_lod):
        a_stats = result_a._lod_stats[k]
        c_stats = result_c._lod_stats[k]
        rows.append({
            "model": model_name,
            "lod": k,
            "n_verts_orig": n_v,
            "n_tris_orig": n_t,
            "approach_A_bpv": round(result_a.bits_per_vertex, 2),
            "approach_A_bytes": len(result_a.data),
            "approach_C_bpv": round(result_c.bits_per_vertex, 2),
            "approach_C_bytes": len(result_c.data),
            "lod_verts_A": a_stats["n_verts"],
            "lod_tris_A":  a_stats["n_tris"],
            "lod_verts_C": c_stats["n_verts"],
            "lod_tris_C":  c_stats["n_tris"],
            "encode_time_A_ms": round(t_a * 1000, 0),
            "encode_time_C_ms": round(t_c * 1000, 0),
        })

    # Visualization: 2 rows × n_lod cols. Top = Approach A, bottom = Approach C.
    if 'bunny' in model_name.lower():
        elev, azim = 20, -60
    elif 'torus' in model_name.lower():
        elev, azim = 30, -45
    else:
        elev, azim = 25, -55

    fig, axes = plt.subplots(2, n_lod, figsize=(n_lod * 3.5, 7.2),
                              subplot_kw={'projection': '3d'})
    fig.suptitle(f'{model_name} — Approach A (top, Ancestor) vs '
                 f'Approach C (bottom, Progressive Mesh)', fontsize=13)

    def plot_mesh(ax, verts_m, tris_m, title, color='#4a90d9'):
        if len(tris_m) > 0 and len(verts_m) > 0:
            polys = [verts_m[tris_m[fi]] for fi in range(len(tris_m))]
            collection = Poly3DCollection(polys, alpha=0.85, linewidths=0.12,
                                          edgecolors='#333333')
            collection.set_facecolor(color)
            ax.add_collection3d(collection)
        for d, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            setter(verts[:, d].min(), verts[:, d].max())
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    colors_a = '#4a90d9'   # blue
    colors_c = '#d9804a'   # orange

    for k in range(n_lod):
        va, ta = a_meshes[k]
        vc_, tc_ = c_meshes[k]
        plot_mesh(axes[0, k], va, ta,
                  f'A LOD{k}\n{len(va):,}v {len(ta):,}t', colors_a)
        plot_mesh(axes[1, k], vc_, tc_,
                  f'C LOD{k}\n{len(vc_):,}v {len(tc_):,}t', colors_c)

    # Row labels
    axes[0, 0].text2D(-0.12, 0.5, 'A', transform=axes[0, 0].transAxes,
                      fontsize=14, fontweight='bold')
    axes[1, 0].text2D(-0.12, 0.5, 'C', transform=axes[1, 0].transAxes,
                      fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=140, bbox_inches='tight')
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
        model_name = os.path.splitext(os.path.basename(mp))[0]
        out_path = f'assets/plots/compare_AC_{model_name}.png'
        rows = compare_model(mp, out_path)
        all_rows.extend(rows)

    # CSV
    if all_rows:
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults written to {CSV_PATH} ({len(all_rows)} rows)")

    # Summary
    print("\n== Summary (per-model BPV) ==")
    print(f"{'Model':<20} {'ApproachA':>12} {'ApproachC':>12} {'A/C ratio':>10}")
    print('-' * 60)
    seen = set()
    for row in all_rows:
        m = row['model']
        if m in seen:
            continue
        seen.add(m)
        a = row['approach_A_bpv']
        c = row['approach_C_bpv']
        print(f"{m:<20} {a:>12.2f} {c:>12.2f} {a/c:>10.2f}")


if __name__ == "__main__":
    main()

"""Meshlet generator comparison: greedy / tunneling / joint_learned.

Computes a wide set of structural metrics per (mesh, generator), so we can
later pick which ones to keep for the paper. Emits bench_meshlet_gen.csv
with every metric, plus a compact Markdown table to stdout.

Metrics (per row):
  - n_meshlets, v_mean, v_max, v_util_pct (mean v_per / 256)
  - t_mean, t_max, t_util_pct
  - strips_mean, strips_max
  - vert_replication = sum(v_per) / n_v          # boundary cost proxy
  - aspect_mean / med / p90                       # longest / shortest bbox axis
  - flatness_med = median(shortest / longest)     # 1 = cubic, 0 = flat
  - planarity_med = median(third_singular_val / longest_bbox_axis) # 0 = planar
  - bnd_frac_pct mean / p90
  - bpv (final STRIDE bits-per-vertex, encode_from_prepared)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import encode_from_prepared


MESHES = [
    ("s-bunny", "assets/stanford-bunny.obj"),
    ("tank",    "assets/tank.obj"),
    ("dragon",  "assets/xyzrgb_dragon.obj"),
]

GENERATORS = ["greedy", "tunneling", "joint_learned"]

GEN_LABEL = {
    "greedy":        "greedy",
    "tunneling":     "tunneling",
    "joint_learned": "ours",
}

CSV_PATH = ROOT / "bench_meshlet_gen.csv"

CSV_FIELDS = [
    "mesh", "generator",
    "n_meshlets",
    "v_mean", "v_max", "v_util_pct",
    "t_mean", "t_max", "t_util_pct",
    "strips_mean", "strips_max",
    "vert_replication",
    "aspect_mean", "aspect_med", "aspect_p90",
    "flatness_med",
    "planarity_med",
    "bnd_frac_pct_mean", "bnd_frac_pct_p90",
    "bpv",
]


def _row_for(mesh_name: str, mesh_path: str, gen: str) -> dict:
    prep = load_or_prepare(
        mesh_path,
        max_verts=256, max_tris=256,
        precision_error=0.0005,
        gen_method=gen,
        strip_method="multiseed",
        clean=True, verbose=False,
    )
    plans = prep["plans"]
    vn = prep["vn"]
    n_v = int(prep["n_v"])
    n_meshlets = len(plans)

    n_bnd    = np.empty(n_meshlets, dtype=np.int32)
    n_int    = np.empty(n_meshlets, dtype=np.int32)
    n_tris   = np.empty(n_meshlets, dtype=np.int32)
    n_strips = np.empty(n_meshlets, dtype=np.int32)
    aspect   = np.empty(n_meshlets, dtype=np.float64)
    flatness = np.empty(n_meshlets, dtype=np.float64)
    planar   = np.empty(n_meshlets, dtype=np.float64)

    for i, plan in enumerate(plans):
        nb = int(plan["n_bnd"])
        ni = int(plan["n_int"])
        n_bnd[i] = nb
        n_int[i] = ni
        n_tris[i] = int(plan["n_tris_m"])
        n_strips[i] = int(plan["n_strips"])
        l2g = plan["local_to_global"]
        pts = vn[l2g[:nb + ni]]            # normalized verts in [0, 1]
        ext = pts.max(0) - pts.min(0)
        e_max = float(ext.max())
        e_min = float(ext.min())
        if e_max > 0:
            aspect[i]   = e_max / max(e_min, 1e-9)
            flatness[i] = max(e_min, 0.0) / e_max
            # third singular value of centered point cloud / longest axis
            c = pts - pts.mean(0, keepdims=True)
            # cheap PCA via SVD on (k,3); k up to 256
            try:
                s = np.linalg.svd(c, compute_uv=False)
                sv3 = float(s[2]) if s.size >= 3 else 0.0
            except np.linalg.LinAlgError:
                sv3 = 0.0
            planar[i] = sv3 / e_max
        else:
            aspect[i]   = 0.0
            flatness[i] = 0.0
            planar[i]   = 0.0

    v_per = n_bnd + n_int
    bnd_frac = n_bnd / np.maximum(1, v_per)

    blob = encode_from_prepared(prep, predictor="linear3", verbose=False)
    bpv = 8.0 * len(blob) / n_v
    assert n_v > 0
    assert len(blob) > 0

    return {
        "mesh":               mesh_name,
        "generator":          GEN_LABEL[gen],
        "n_meshlets":         n_meshlets,
        "v_mean":             float(v_per.mean()),
        "v_max":              int(v_per.max()),
        "v_util_pct":         float(v_per.mean()) / 256.0 * 100.0,
        "t_mean":             float(n_tris.mean()),
        "t_max":              int(n_tris.max()),
        "t_util_pct":         float(n_tris.mean()) / 256.0 * 100.0,
        "strips_mean":        float(n_strips.mean()),
        "strips_max":         int(n_strips.max()),
        "vert_replication":   float(v_per.sum()) / float(n_v),
        "aspect_mean":        float(aspect.mean()),
        "aspect_med":         float(np.median(aspect)),
        "aspect_p90":         float(np.percentile(aspect, 90)),
        "flatness_med":       float(np.median(flatness)),
        "planarity_med":      float(np.median(planar)),
        "bnd_frac_pct_mean":  float(bnd_frac.mean()) * 100.0,
        "bnd_frac_pct_p90":   float(np.percentile(bnd_frac, 90)) * 100.0,
        "bpv":                bpv,
    }


def _print_md_table(rows: list[dict]) -> None:
    # Paper Table 1 column set (CSV retains every metric for future use).
    cols = [
        ("Mesh",        lambda r: r["mesh"],                          ""),
        ("Gen",         lambda r: r["generator"],                     ""),
        ("#M",          lambda r: f"{r['n_meshlets']:,}",             "---:"),
        ("V mean",      lambda r: f"{r['v_mean']:.0f}",               "---:"),
        ("T mean/max",  lambda r: f"{r['t_mean']:.0f} / {r['t_max']}", "---:"),
        ("T util%",     lambda r: f"{r['t_util_pct']:.0f}",           "---:"),
        ("Strips/M",    lambda r: f"{r['strips_mean']:.2f}",          "---:"),
        ("V repl",      lambda r: f"{r['vert_replication']:.2f}",     "---:"),
        ("Aspect",      lambda r: f"{r['aspect_med']:.1f}",           "---:"),
        ("Bnd%",        lambda r: f"{r['bnd_frac_pct_mean']:.1f}",    "---:"),
        ("BPV",         lambda r: f"{r['bpv']:.2f}",                  "---:"),
    ]
    print()
    print("| " + " | ".join(name for name, _, _ in cols) + " |")
    print("|" + "|".join((align or "---") for _, _, align in cols) + "|")
    for r in rows:
        print("| " + " | ".join(fn(r) for _, fn, _ in cols) + " |")


def main() -> None:
    rows: list[dict] = []
    for mesh_name, mesh_path in MESHES:
        for gen in GENERATORS:
            print(f"[{mesh_name:8s}] {gen} ...", flush=True)
            row = _row_for(mesh_name, mesh_path, gen)
            print(
                f"  n_meshlets={row['n_meshlets']:,}  "
                f"v_util={row['v_util_pct']:.0f}%  t_util={row['t_util_pct']:.0f}%  "
                f"strips_med={row['strips_mean']:.2f}  "
                f"v_repl={row['vert_replication']:.2f}  "
                f"aspect_med={row['aspect_med']:.1f}  "
                f"flat_med={row['flatness_med']:.2f}  "
                f"planar_med={row['planarity_med']:.3f}  "
                f"bnd_mean={row['bnd_frac_pct_mean']:.1f}%  "
                f"bpv={row['bpv']:.2f}"
            )
            rows.append(row)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {CSV_PATH.relative_to(ROOT)}")

    _print_md_table(rows)


if __name__ == "__main__":
    main()
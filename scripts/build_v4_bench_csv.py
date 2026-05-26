"""Build bench_competitors_q12bbox_v4.csv from the existing q12bbox CSV.

Changes vs the source CSV:
  * Replace `ParaDelta v5 (ours)` rows with `STRIDE-dup (ours)` — measured
    STRIDE-dup body BPV + side-table BPV and the v4 axis-parallel mtps.
  * Update DGF tb12 mtps with the measured CUDA-port numbers (per-mesh).
  * Drop GlobalAMD / GlobalEB / gltfpack rows we don't plot.

Inputs:
  bench_competitors_q12bbox.csv  (existing, has DGF/Draco/meshopt/Corto rows)
  bench_dup_opt_v4_axis.csv      (STRIDE-dup mtps + BPV)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "bench_competitors_q12bbox.csv"
DUP = ROOT / "bench_dup_opt_v4_axis.csv"
OUT = ROOT / "bench_competitors_q12bbox_v4.csv"

# Map "<mesh_stem>" -> mesh basename in the competitor CSV
MESH_FNAME = {
    "fandisk":         "fandisk.obj",
    "stanford-bunny":  "stanford-bunny.obj",
    "horse":           "horse.obj",
    "Monkey":          "Monkey.obj",
    "happy_buddha":    "happy_buddha.obj",
    "crab":            "crab.obj",
    "tank":            "tank.obj",
    "xyzrgb_dragon":   "xyzrgb_dragon.obj",
}


def main():
    df = pd.read_csv(SRC)
    dup = pd.read_csv(DUP)

    # Drop rows we won't show in v4 paper plots.
    df = df[~df["name"].isin(["GlobalAMD (ours)", "GlobalEB (ours)",
                                "gltfpack -cc"])].copy()

    # Fill in missing DGF rows (q12bbox CSV lacks tank + dragon).
    PATCH_DGF = {
        "tank.obj":          (1806639, 3514247, 272.350812, 11075456, 49.04336062710924),
        "xyzrgb_dragon.obj": (3609600, 7219045, 320.797199, 20929408, 46.386099290780145),
    }
    have_dgf = set(df.loc[df["name"] == "DGF tb12", "mesh"])
    add_rows = []
    for mname, (nv, nt, mb, sz, bpv) in PATCH_DGF.items():
        if mname not in have_dgf:
            add_rows.append({
                "mesh": mname, "obj_mb": mb, "n_v": nv, "n_t": nt,
                "name": "DGF tb12", "size_b": sz, "bpv": bpv,
                "max_err": 0.0, "enc_ms": 0.0, "dec_us": None, "mtps": None,
                "gpu": True, "note": "patched",
            })

    # Fill in missing Corto rows for tank + dragon (numbers from world csv;
    # q12bbox Corto runs were skipped). decode-side, world vs q12 makes a
    # negligible difference for Corto's CPU CLERS path.
    PATCH_CORTO = {
        # (n_v, n_t, obj_mb, size_b, bpv, dec_us, mtps)
        "tank.obj":          (1806639, 3514247, 272.350812, 2769616, 12.264170097069751,
                              117935.56, 29.798026990332687),
        "xyzrgb_dragon.obj": (3609600, 7219045, 320.797199, 4137684, 9.170398936170212,
                              235104.84, 30.705641789424668),
    }
    have_corto = set(df.loc[df["name"] == "Corto v12", "mesh"])
    for mname, (nv, nt, mb, sz, bpv, dus, mtps) in PATCH_CORTO.items():
        if mname not in have_corto:
            add_rows.append({
                "mesh": mname, "obj_mb": mb, "n_v": nv, "n_t": nt,
                "name": "Corto v12", "size_b": sz, "bpv": bpv,
                "max_err": 0.0, "enc_ms": 0.0, "dec_us": dus, "mtps": mtps,
                "gpu": False, "note": "world-bench (q12 Corto not measured)",
            })
    if add_rows:
        df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)

    # Update DGF tb12 mtps with measured per-mesh values.
    dgf_mtps_by_mesh = dict(zip(
        (MESH_FNAME[m] for m in dup["mesh"]),
        dup["dgf_mtps"].tolist(),
    ))
    is_dgf = df["name"] == "DGF tb12"
    df.loc[is_dgf, "mtps"] = df.loc[is_dgf, "mesh"].map(dgf_mtps_by_mesh)
    df.loc[is_dgf, "dec_us"] = df.loc[is_dgf, "mesh"].map(
        lambda m: (df.loc[df["mesh"] == m, "n_t"].iloc[0] /
                   dgf_mtps_by_mesh[m]) if m in dgf_mtps_by_mesh else None
    )
    df.loc[is_dgf, "note"] = "GPU decode (measured CUDA port on RTX 3090)"

    # Replace ParaDelta v5 rows with STRIDE-dup rows.
    df = df[df["name"] != "ParaDelta v5 (ours)"].copy()

    pd_rows = []
    for _, r in dup.iterrows():
        mesh_basename = MESH_FNAME[r["mesh"]]
        # Need n_v_src / n_t / obj_mb from existing CSV
        match = df[df["mesh"] == mesh_basename]
        if match.empty:
            continue
        obj_mb = float(match.iloc[0]["obj_mb"])
        n_v   = int(match.iloc[0]["n_v"])
        n_t   = int(match.iloc[0]["n_t"])
        bpv   = float(r["bpv_v4_total"])
        size_b = int(bpv * n_v / 8)
        mtps  = float(r["dup_v4_axis_mtps"])
        dec_us = n_t / mtps
        pd_rows.append({
            "mesh": mesh_basename, "obj_mb": obj_mb, "n_v": n_v, "n_t": n_t,
            "name": "STRIDE-dup (ours)",
            "size_b": size_b, "bpv": bpv,
            "max_err": 0.0,
            "enc_ms": 0.0,  # encode time not the focus
            "dec_us": dec_us,
            "mtps": mtps,
            "gpu": True,
            "note": "fused CUDA kernel, per-meshlet self-contained, axis-parallel",
        })
    df = pd.concat([df, pd.DataFrame(pd_rows)], ignore_index=True)

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(df)} rows)")
    print(df.groupby("name").size().to_string())


if __name__ == "__main__":
    main()

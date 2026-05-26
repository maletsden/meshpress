"""Re-bench tunneling with pack=True (consecutive-strip packing).

Same metrics as scripts/diag_dragon_meshlet_stats.py for tunneling row,
but bypasses cache (gen_method key doesn't capture pack flag) and uses
the tuned chunker.
"""
from __future__ import annotations

import sys
import time as _t
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import Reader
from encoder.paradelta_codec import prepare_paradelta, encode_from_prepared
from utils.mesh_clean import clean_mesh
from utils.meshlet_tunneling import generate_meshlets_tunneling
import utils.meshlet_generator as mg


MESHES = [
    ("s-bunny", "assets/stanford-bunny.obj"),
    ("tank",    "assets/tank.obj"),
    ("dragon",  "assets/xyzrgb_dragon.obj"),
]


def _patched_gen(*a, **kw):
    """Monkey-patch dispatcher to pass pack=True to tunneling."""
    method = kw.get("method", "greedy")
    if method == "tunneling":
        return generate_meshlets_tunneling(
            a[0], a[1],
            max_tris=kw.get("max_tris", 256),
            max_verts=kw.get("max_verts", 256),
            time_budget_s=300.0, verbose=False, pack=True)
    return _orig(*a, **kw)


_orig = mg.generate_meshlets
mg.generate_meshlets = _patched_gen


def main():
    rows = []
    for mesh_name, mesh_path in MESHES:
        print(f"[{mesh_name}] tunneling (tuned, pack=True) ...", flush=True)
        t0 = _t.time()
        model = Reader.read_from_file(mesh_path)
        model, _ = clean_mesh(model, verbose=False)
        prep = prepare_paradelta(
            model,
            max_verts=256, max_tris=256,
            precision_error=0.0005,
            precision_mode="world",
            gen_method="tunneling",
            strip_method="multiseed",
        )
        plans = prep["plans"]
        vn = prep["vn"]
        n_v = int(prep["n_v"])
        n_m = len(plans)

        n_bnd = np.empty(n_m, dtype=np.int32)
        n_int = np.empty(n_m, dtype=np.int32)
        n_tr  = np.empty(n_m, dtype=np.int32)
        n_st  = np.empty(n_m, dtype=np.int32)
        aspect = np.empty(n_m, dtype=np.float64)
        for i, plan in enumerate(plans):
            nb = int(plan["n_bnd"]); ni = int(plan["n_int"])
            n_bnd[i] = nb; n_int[i] = ni
            n_tr[i] = int(plan["n_tris_m"])
            n_st[i] = int(plan["n_strips"])
            l2g = plan["local_to_global"]
            pts = vn[l2g[:nb + ni]]
            ext = pts.max(0) - pts.min(0)
            e_max = float(ext.max()); e_min = float(ext.min())
            aspect[i] = e_max / max(e_min, 1e-9) if e_max > 0 else 0.0
        v_per = n_bnd + n_int
        bnd_frac = n_bnd / np.maximum(1, v_per)

        blob = encode_from_prepared(prep, predictor="linear3", verbose=False)
        bpv = 8.0 * len(blob) / n_v

        row = dict(
            mesh=mesh_name,
            n_m=n_m,
            v_mean=float(v_per.mean()),
            t_mean=float(n_tr.mean()),
            t_max=int(n_tr.max()),
            t_util=float(n_tr.mean())/256.0*100.0,
            strips_mean=float(n_st.mean()),
            v_repl=float(v_per.sum())/float(n_v),
            aspect_med=float(np.median(aspect)),
            bnd_pct=float(bnd_frac.mean())*100.0,
            bpv=bpv,
        )
        rows.append(row)
        print(f"  n_m={n_m:,}  v_mean={row['v_mean']:.0f}  "
              f"t_mean={row['t_mean']:.0f}/{row['t_max']}  "
              f"t_util={row['t_util']:.0f}%  strips/m={row['strips_mean']:.2f}  "
              f"v_repl={row['v_repl']:.2f}  aspect_med={row['aspect_med']:.1f}  "
              f"bnd%={row['bnd_pct']:.1f}  bpv={row['bpv']:.2f}  "
              f"[{_t.time()-t0:.1f}s]")

    print()
    print("| Mesh | Gen | #M | V mean | T mean/max | T util% | Strips/M | V repl | Aspect | Bnd% | BPV |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['mesh']} | tunneling-tuned | {r['n_m']:,} | "
              f"{r['v_mean']:.0f} | {r['t_mean']:.0f} / {r['t_max']} | "
              f"{r['t_util']:.0f} | {r['strips_mean']:.2f} | "
              f"{r['v_repl']:.2f} | {r['aspect_med']:.1f} | "
              f"{r['bnd_pct']:.1f} | {r['bpv']:.2f} |")


if __name__ == "__main__":
    main()

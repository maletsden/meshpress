"""Verify decoder f32 chain drift hypothesis for Lucy.

For each meshlet:
  1. Run encoder _interior_pass_strip (f64) to get codes
  2. Simulate decoder (f32 chain) replaying same strip walk with codes
  3. Compute per-vert error vs ground-truth vn

Report worst-case meshlet error and accumulated stats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import _interior_pass_strip, _strip_traversal
from encoder.paradelta_codec import _fit_linear3


def _decode_meshlet_f32(plan, codes, bnd_recon_norm, delta_f32,
                         w3_f32, n_bnd):
    """Decoder-faithful f32 simulator. Returns dict v_local -> f32 pos."""
    local_to_global = plan["local_to_global"]
    recon = {}
    for lid in range(n_bnd):
        recon[lid] = bnd_recon_norm[int(local_to_global[lid])]\
            .astype(np.float32).copy()
    if n_bnd > 0:
        s = np.zeros(3, dtype=np.float32)
        for lid in range(n_bnd):
            s += recon[lid]
        fallback = (s / np.float32(n_bnd)).astype(np.float32)
    else:
        fallback = np.zeros(3, dtype=np.float32)

    order = _strip_traversal(plan["ml_tris_local"], plan["strips"], n_bnd)
    for i, (v_local, kind, refs) in enumerate(order):
        c = codes[i].astype(np.float32)
        if kind == 'para':
            a, b, cc = refs
            pred = (np.float32(w3_f32[0]) * recon[a]
                    + np.float32(w3_f32[1]) * recon[b]
                    + np.float32(w3_f32[2]) * recon[cc])
        else:
            pred = fallback.copy()
        recon[v_local] = (pred + c * delta_f32).astype(np.float32)
    return recon


def main(path: str = "assets/lucy.obj"):
    print(f"Loading prep for {path} ...")
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                            precision_error=0.0005,
                            gen_method="joint_learned",
                            strip_method="multiseed",
                            clean=True, verbose=False)
    plans = prep["plans"]
    vn = prep["vn"]
    bnd_recon_norm = prep["bnd_recon_norm"]
    per_coord_err = float(prep["per_coord_err"])
    delta = 2.0 * per_coord_err
    scale = float(prep["scale"])
    print(f"  meshlets: {len(plans):,}  scale={scale:.2f}")

    print(f"\nPass 1: TG baseline ...")
    samples = []
    for plan in plans:
        _, sm = _interior_pass_strip(plan, vn, bnd_recon_norm, delta, w3=None)
        samples.extend(sm)
    lin3_w = _fit_linear3(samples)
    w3_decoder = lin3_w.astype(np.float32).astype(np.float64)
    print(f"  lin3 = ({float(lin3_w[0]):.4f}, "
          f"{float(lin3_w[1]):.4f}, {float(lin3_w[2]):.4f})")

    print(f"\nPass 2: encode f64 + decode f32 + measure ...")
    delta_f32 = np.float32(delta)
    w3_f32 = lin3_w.astype(np.float32)

    max_err_per_meshlet = np.zeros(len(plans), dtype=np.float32)
    n_with_drift = 0
    bad_examples = []

    for mid, plan in enumerate(plans):
        n_bnd = plan["n_bnd"]
        n_int = plan["n_int"]
        if n_int == 0:
            continue
        codes, _ = _interior_pass_strip(plan, vn, bnd_recon_norm,
                                          delta, w3=w3_decoder)
        recon_dec = _decode_meshlet_f32(plan, codes, bnd_recon_norm,
                                          delta_f32, w3_f32, n_bnd)
        local_to_global = plan["local_to_global"]
        worst = 0.0
        for v_local in range(n_bnd, n_bnd + n_int):
            true_norm = vn[int(local_to_global[v_local])]
            dec_norm = recon_dec[v_local]
            d = float(np.linalg.norm(dec_norm.astype(np.float64) - true_norm))
            if d > worst:
                worst = d
        # World units
        worst_world = worst * scale
        max_err_per_meshlet[mid] = worst_world
        if worst_world > 0.01:
            n_with_drift += 1
            if len(bad_examples) < 20:
                bad_examples.append((mid, worst_world, n_bnd, n_int,
                                     int(np.abs(codes).max())))

    print(f"\n=== Per-meshlet f32-decode err (world units) ===")
    arr = max_err_per_meshlet
    print(f"  p50  = {float(np.percentile(arr, 50)):.6f}")
    print(f"  p90  = {float(np.percentile(arr, 90)):.6f}")
    print(f"  p99  = {float(np.percentile(arr, 99)):.6f}")
    print(f"  p99.9= {float(np.percentile(arr, 99.9)):.6f}")
    print(f"  max  = {float(arr.max()):.6f}")
    print(f"  meshlets with worst > 0.01: {n_with_drift:,}")
    print(f"  meshlets with worst > 0.5:  "
          f"{int((arr > 0.5).sum()):,}")
    print(f"  meshlets with worst > 5.0:  "
          f"{int((arr > 5.0).sum()):,}")
    print(f"  precision target (0.0005 / sqrt(3)) world: "
          f"{0.0005/np.sqrt(3):.6f}")

    if bad_examples:
        print(f"\n  Worst meshlets (by f32-decode err vs vn truth):")
        bad_examples.sort(key=lambda t: -t[1])
        for mid, w, n_bnd, n_int, max_c in bad_examples[:10]:
            print(f"    meshlet {mid}: worst_err={w:.4f} mm  "
                  f"n_bnd={n_bnd}  n_int={n_int}  "
                  f"max|code|={max_c:,}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "assets/lucy.obj"
    main(p)

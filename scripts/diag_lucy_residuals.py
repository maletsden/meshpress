"""Diagnose Lucy v5 residual distribution.

Run TG-baseline pass 1 + lin3-fit + final pass over Lucy meshlets,
collect per-meshlet residual stats (per-axis mn, mx, range, abs-mean).
Identify meshlets with extreme residuals and their topology.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import _interior_pass_strip
from encoder.paradelta_codec import _fit_linear3


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
    print(f"  meshlets: {len(plans):,}  scale={scale:.2f}  "
          f"per_coord_err={per_coord_err:.3e}  delta={delta:.3e}")

    print(f"\nPass 1 (TG baseline) ...")
    all_samples = []
    for plan in plans:
        _, samples = _interior_pass_strip(plan, vn, bnd_recon_norm,
                                            delta, w3=None)
        all_samples.extend(samples)
    lin3_w = _fit_linear3(all_samples)
    print(f"  lin3 weights = ({float(lin3_w[0]):.6f}, "
          f"{float(lin3_w[1]):.6f}, "
          f"{float(lin3_w[2]):.6f})    sum={float(lin3_w.sum()):.6f}")

    print(f"\nPass 2 (fitted) — collect residual stats ...")
    per_mesh_max = []  # max abs code per meshlet
    per_mesh_mn = []
    per_mesh_mx = []
    per_mesh_rng = []
    huge_meshlets = []  # idx, max_abs
    HUGE_THR = 200_000  # zigzag-cost cliff

    for mid, plan in enumerate(plans):
        codes, _ = _interior_pass_strip(plan, vn, bnd_recon_norm,
                                          delta, w3=lin3_w)
        if codes.shape[0] == 0:
            per_mesh_max.append(0)
            per_mesh_mn.append(0); per_mesh_mx.append(0)
            per_mesh_rng.append(0)
            continue
        ma = int(np.abs(codes).max())
        mn = int(codes.min()); mx = int(codes.max())
        per_mesh_max.append(ma)
        per_mesh_mn.append(mn); per_mesh_mx.append(mx)
        per_mesh_rng.append(int(mx - mn))
        if ma > HUGE_THR:
            huge_meshlets.append((mid, ma, mn, mx, plan["n_int"]))

    per_mesh_max = np.asarray(per_mesh_max, dtype=np.int64)
    per_mesh_rng = np.asarray(per_mesh_rng, dtype=np.int64)

    print(f"\n=== Residual code stats (zigzagged absolute) ===")
    print(f"  per-meshlet max |code|:")
    print(f"    p50  = {int(np.percentile(per_mesh_max, 50)):,}")
    print(f"    p90  = {int(np.percentile(per_mesh_max, 90)):,}")
    print(f"    p99  = {int(np.percentile(per_mesh_max, 99)):,}")
    print(f"    p99.9= {int(np.percentile(per_mesh_max, 99.9)):,}")
    print(f"    max  = {int(per_mesh_max.max()):,}")
    print(f"  per-meshlet residual range (mx-mn):")
    print(f"    p50  = {int(np.percentile(per_mesh_rng, 50)):,}")
    print(f"    p99  = {int(np.percentile(per_mesh_rng, 99)):,}")
    print(f"    max  = {int(per_mesh_rng.max()):,}")

    # i16 fits: |mn| < 32768
    i16_overflow = int((np.abs(per_mesh_mn) >= 32768).sum() +
                       (np.abs(per_mesh_mx) >= 32768).sum())
    print(f"  meshlets where mn or mx exceeds i16: "
          f"{i16_overflow:,} / {len(plans):,}")

    # max code as f32 -> exact representable up to 2^24 = 16,777,216
    f32_overflow = int((per_mesh_max > (1 << 24)).sum())
    print(f"  meshlets where max |code| > 2^24 (f32 exact-int limit): "
          f"{f32_overflow:,}")

    if huge_meshlets:
        print(f"\n  Top 10 meshlets by max |code|:")
        huge_meshlets.sort(key=lambda t: -t[1])
        for mid, ma, mn, mx, n_int in huge_meshlets[:10]:
            print(f"    meshlet {mid}: max|code|={ma:,}  "
                  f"mn={mn:,}  mx={mx:,}  n_int={n_int}")

    # In world units: residual max * delta * scale
    world_max_residual = float(per_mesh_max.max()) * delta * scale
    print(f"\n  Worst-case residual in world units: "
          f"{world_max_residual:.4f}")
    print(f"  (target precision_error: {0.0005 * scale:.4f} world if "
          f"misinterpreted, or 0.0005 if absolute)")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "assets/lucy.obj"
    main(p)
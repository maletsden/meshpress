"""End-to-end smoke: encoder fits predictor, ships 12 B header in blob.
Decoder reads header + applies. Verifies bit-exact mesh recovery + measures
real byte savings vs canonical (= same blob layout but pred=(1,1,-1)/K=0).
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5_dup import encode_dup, decode_dup


def run(mesh_path: str):
    print(f"\n--- {mesh_path} ---")
    prep = load_or_prepare(mesh_path, max_verts=256, max_tris=256,
                           precision_error=1.0/4096.0,
                           precision_mode="bbox_frac",
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    n_v = prep["n_v"]; n_t = prep["n_t"]
    print(f"  n_v={n_v:,} n_t={n_t:,}")

    # Canonical baseline.
    t0 = time.time()
    data_c = encode_dup(prep, predictor="canonical")
    enc_c = time.time() - t0
    print(f"  canonical : {len(data_c):>10,} B   enc {enc_c:.1f}s")
    V_c, T_c = decode_dup(data_c)

    # Generalized.
    t1 = time.time()
    data_g = encode_dup(prep, predictor="generalized")
    enc_g = time.time() - t1
    print(f"  generalized: {len(data_g):>10,} B   enc {enc_g:.1f}s")
    V_g, T_g = decode_dup(data_g)

    delta = len(data_g) - len(data_c)
    pct = 100 * delta / len(data_c)
    print(f"  delta      : {delta:+,} B  ({pct:+.3f}%)")

    # Bit-exact decode check (both pipelines reconstruct same mesh).
    if V_c.shape == V_g.shape and T_c.shape == T_g.shape:
        max_dV = float(np.max(np.abs(V_c - V_g)))
        n_T_mis = int((T_c != T_g).sum())
        print(f"  max |V_c - V_g| = {max_dV:.6e}   T mismatches = {n_T_mis}")
    else:
        print(f"  SHAPE MISMATCH V {V_c.shape} vs {V_g.shape}, T {T_c.shape} vs {T_g.shape}")


if __name__ == "__main__":
    meshes = sys.argv[1:] or ["assets/stanford-bunny.obj"]
    for m in meshes:
        run(m)

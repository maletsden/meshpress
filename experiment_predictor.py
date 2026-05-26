"""Sweep predictor modes on cached prepared meshlets.

First run: caches meshlet+plans pickle per (model, gen_method, max_verts, ...).
Subsequent runs: skips meshlet generation, only re-runs interior encoding.
"""

import os
import sys
import time
import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import encode_from_prepared, decode_paradelta


def run_one(model_path: str, *, max_verts: int = 256,
            precision_error: float = 0.0005,
            gen_method: str = "joint_learned",
            strip_method: str = "multiseed",
            modes=("linear5",),
            verify: bool = True) -> None:
    print(f"\n=== {model_path} ===")
    prep = load_or_prepare(
        model_path, max_verts=max_verts, max_tris=max_verts,
        precision_error=precision_error,
        gen_method=gen_method, strip_method=strip_method,
        verbose=True)
    n_v = prep["n_v"]
    print(f"  verts={n_v}  tris={prep['n_t']}  "
          f"meshlets={prep['n_meshlets']}  "
          f"boundary={prep['n_boundary']}")
    for mode in modes:
        t0 = time.time()
        data = encode_from_prepared(prep, predictor=mode, verbose=True)
        t_enc = time.time() - t0
        bpv = len(data) * 8 / n_v
        line = (f"  [{mode:>7s}]  size={len(data):>7,} B  "
                f"BPV={bpv:6.2f}  enc={t_enc:5.2f}s")
        if verify:
            verts_d, tris_d = decode_paradelta(data)
            n_dec = len(verts_d)
            line += f"  decode_verts={n_dec}"
        print(line)


if __name__ == "__main__":
    args = sys.argv[1:]
    models = []
    max_verts = 256
    for a in args:
        if a.startswith("mv="):
            max_verts = int(a.split("=", 1)[1])
        else:
            models.append(a)
    if not models:
        models = ["assets/bunny.obj", "assets/stanford-bunny.obj"]
    for m in models:
        run_one(m, max_verts=max_verts)
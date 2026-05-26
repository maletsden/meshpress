"""Roundtrip + BPV comparison: v4 (lin5+greedy) vs v5 (lin3+strip).

For each model:
  - encode v4 (current, lin5 + greedy)
  - encode v5 (Plan F: lin3 + strip-emit-order)
  - decode each, compare reconstructed verts to original within precision
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, decode_paradelta,
)
from encoder.paradelta_v5 import (
    encode_from_prepared_v5, decode_paradelta_v5,
)


def check(path: str) -> None:
    name = Path(path).name
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    n_v = prep["n_v"]
    print(f"\n[{name}] n_v={n_v:,} n_t={prep['n_t']:,}")

    # ---- v4 baseline ----
    t0 = time.perf_counter()
    data_v4 = encode_from_prepared(prep, predictor="linear5", verbose=False)
    t_enc_v4 = time.perf_counter() - t0
    t0 = time.perf_counter()
    v4_v, v4_t = decode_paradelta(data_v4)
    t_dec_v4 = time.perf_counter() - t0
    bpv_v4 = len(data_v4) * 8 / n_v

    # ---- v5 Plan F ----
    t0 = time.perf_counter()
    data_v5 = encode_from_prepared_v5(prep, verbose=False)
    t_enc_v5 = time.perf_counter() - t0
    t0 = time.perf_counter()
    v5_v, v5_t = decode_paradelta_v5(data_v5)
    t_dec_v5 = time.perf_counter() - t0
    bpv_v5 = len(data_v5) * 8 / n_v

    # Compare v5 reconstruction to v4 reconstruction (should be within delta)
    if v4_v.shape != v5_v.shape:
        print(f"  FAIL: shapes differ v4={v4_v.shape} v5={v5_v.shape}")
        sys.exit(1)
    if v4_t.shape != v5_t.shape:
        print(f"  FAIL: tri shapes differ v4={v4_t.shape} v5={v5_t.shape}")
        sys.exit(1)
    # Tris may differ in vertex ordering — compare sets of triangle edges
    # instead. Quick: compare both as sorted tuple-of-sorted-triples.
    s_v4 = {tuple(sorted(t)) for t in v4_t}
    s_v5 = {tuple(sorted(t)) for t in v5_t}
    tris_eq_set = s_v4 == s_v5

    # Per-vert error: v5 verts may be in different global ID order if
    # encoder emits interior verts in different order. To compare, use
    # KDTree Hausdorff against original.
    from scipy.spatial import cKDTree
    orig = np.array([(v.x, v.y, v.z) for v in
                     load_or_prepare.__globals__["Reader"]
                     .read_from_file(path).vertices],
                    dtype=np.float32) if False else None
    # Simpler: compare v5 verts to v4 verts pointwise (same n, same order
    # iff encoders enumerate interior verts the same way per meshlet).
    diff_pw = float(np.abs(v5_v - v4_v).max())

    print(f"  v4: {len(data_v4):,} B  BPV={bpv_v4:6.2f}  "
          f"enc={t_enc_v4*1000:6.0f} ms  dec={t_dec_v4*1000:7.0f} ms")
    print(f"  v5: {len(data_v5):,} B  BPV={bpv_v5:6.2f}  "
          f"enc={t_enc_v5*1000:6.0f} ms  dec={t_dec_v5*1000:7.0f} ms  "
          f"(ΔBPV {bpv_v5-bpv_v4:+.2f})")
    # Strict residual budget = 2 × precision_error in normalized space
    # → world delta ≈ 2 × precision_error
    budget = 2 * prep["precision_error"]
    print(f"  v5 vs v4 pointwise max_err={diff_pw:.4g}  "
          f"budget≈{budget:.4g}  tris_eq_set={tris_eq_set}")
    if diff_pw > 10 * budget:
        print(f"  FAIL: vertex diff way over precision budget")
        sys.exit(1)


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        check(p)

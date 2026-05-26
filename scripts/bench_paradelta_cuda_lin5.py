"""Bench V1 (current) vs V2 (per-thread LIN5) vs V3 (warp LIN5)."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cupy as cp

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, decode_paradelta, decode_paradelta_to_struct,
)
from utils.paradelta_cuda import ParaDeltaCudaDecoder
from utils.paradelta_cuda_lin5 import ParaDeltaLin5PerThread, ParaDeltaLin5Warp


def bench_one(name: str, dec, n_t: int, warmup=5, runs=20) -> float:
    for _ in range(warmup):
        dec.decode()
    cp.cuda.Device().synchronize()
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(runs):
        dec.decode()
    end.record(); end.synchronize()
    us = cp.cuda.get_elapsed_time(start, end) * 1000.0 / runs
    print(f"    {name:24s} {us:7.1f} µs   {n_t/us:7.1f} M tris/s")
    return us


def check(path: str) -> None:
    name = Path(path).name
    prep = load_or_prepare(
        path, max_verts=256, max_tris=256, precision_error=0.0005,
        gen_method="joint_learned", strip_method="multiseed", verbose=False)
    data = encode_from_prepared(prep, predictor="linear5", verbose=False)
    print(f"\n[{name}] size={len(data):,} B  "
          f"BPV={len(data)*8/prep['n_v']:.2f}  n_t={prep['n_t']:,}")

    v_ref, t_ref = decode_paradelta(data)
    s = decode_paradelta_to_struct(data)

    decs = {
        "V1 (baseline 128/block)": ParaDeltaCudaDecoder(s, block_size=128),
        "V2 (1 ml/thread, 32/blk)": ParaDeltaLin5PerThread(s),
        "V3 (warp/meshlet)":        ParaDeltaLin5Warp(s),
    }
    # Correctness gate
    for label, d in decs.items():
        v_gpu, t_gpu = d.decode_to_host()
        if v_gpu.shape != v_ref.shape or t_gpu.shape != t_ref.shape:
            print(f"  FAIL shape {label}: v {v_gpu.shape} vs {v_ref.shape}")
            sys.exit(1)
        max_err = float(np.abs(v_gpu - v_ref).max())
        tris_eq = np.array_equal(t_gpu.astype(np.int64), t_ref)
        if max_err > 1e-3 or not tris_eq:
            print(f"  FAIL {label}: max_err={max_err:.3g} tris_eq={tris_eq}")
            sys.exit(1)
        print(f"  ok {label:28s} max_err={max_err:.3g}")

    print("  timed (kernel only):")
    for label, d in decs.items():
        bench_one(label, d, prep["n_t"])


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        check(p)
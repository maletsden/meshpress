"""Verify + bench v5 GPU fused decoder vs v4 GPU + CPU baselines."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cupy as cp

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, decode_paradelta,
)
from encoder.paradelta_v5 import (
    encode_from_prepared_v5, decode_paradelta_v5,
)
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder


def check(path: str, warmup=5, runs=20) -> None:
    name = Path(path).name
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    n_v = prep["n_v"]; n_t = prep["n_t"]
    print(f"\n[{name}] n_v={n_v:,} n_t={n_t:,}")

    data_v4 = encode_from_prepared(prep, predictor="linear5", verbose=False)
    data_v5 = encode_from_prepared_v5(prep, verbose=False)
    print(f"  v4: {len(data_v4):,} B  BPV={len(data_v4)*8/n_v:.2f}")
    print(f"  v5: {len(data_v5):,} B  BPV={len(data_v5)*8/n_v:.2f}  "
          f"(ΔBPV {(len(data_v5)-len(data_v4))*8/n_v:+.2f})")

    # CPU v5 reference
    v_cpu, t_cpu = decode_paradelta_v5(data_v5)

    # GPU v5 — build + decode + verify
    dec = ParaDeltaV5GpuDecoder(data_v5)
    v_gpu, t_gpu = dec.decode_to_host()
    if v_gpu.shape != v_cpu.shape:
        print(f"  FAIL v shape gpu={v_gpu.shape} cpu={v_cpu.shape}")
        sys.exit(1)
    if t_gpu.shape != t_cpu.shape:
        print(f"  FAIL t shape gpu={t_gpu.shape} cpu={t_cpu.shape}")
        sys.exit(1)
    err = float(np.abs(v_gpu - v_cpu).max())
    tris_eq = np.array_equal(t_gpu.astype(np.int64), t_cpu)
    print(f"  GPU v5 vs CPU v5: max_v_err={err:.4g}  tris_eq={tris_eq}")
    if err > 1e-3 or not tris_eq:
        print("  FAIL correctness")
        sys.exit(1)

    # Time kernel only
    for _ in range(warmup):
        dec.decode()
    cp.cuda.Device().synchronize()
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(runs):
        dec.decode()
    end.record(); end.synchronize()
    kernel_us = cp.cuda.get_elapsed_time(start, end) * 1000.0 / runs
    print(f"  GPU v5 fused kernel:   {kernel_us:7.1f} µs  "
          f"({n_t/kernel_us:7.1f} M tris/s)")

    # End-to-end (build + decode) time
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        d = ParaDeltaV5GpuDecoder(data_v5)
        d.decode()
        cp.cuda.Device().synchronize()
    t_e2e = (time.perf_counter() - t0) / 3
    print(f"  GPU v5 end-to-end (build+decode): {t_e2e*1000:7.1f} ms  "
          f"({n_t/t_e2e/1e6:.1f} M tris/s)")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        check(p)

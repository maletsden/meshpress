"""Where does v5 GPU decode time go?

Three timers per mesh:
  A) Pure kernel (CUDA Events around N kernel launches, single sync)
  B) Python decode() wall-clock with sync-per-call (Python + launch + kernel)
  C) Python decode() wall-clock with sync at end (Python + launch overlap)

(B - A)/runs ≈ Python+launch overhead per call.
(A) is what a C++/Vulkan/DX12 host would also pay (kernel only).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cupy as cp

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder


def bench(path: str, warmup=20, runs=100) -> None:
    name = Path(path).name
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    n_t = prep["n_t"]; n_v = prep["n_v"]
    data = encode_from_prepared_v5(prep, verbose=False)
    print(f"\n[{name}] n_v={n_v:,} n_t={n_t:,} bytes={len(data):,} "
          f"BPV={len(data)*8/n_v:.2f}")

    dec = ParaDeltaV5GpuDecoder(data)
    for _ in range(warmup):
        dec.decode()
    cp.cuda.Device().synchronize()

    # A) Pure kernel time (Events, no per-call sync)
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(runs):
        dec.decode()
    end.record(); end.synchronize()
    kernel_us = cp.cuda.get_elapsed_time(start, end) * 1000.0 / runs

    # B) Python wall-clock w/ sync per call
    t0 = time.perf_counter()
    for _ in range(runs):
        dec.decode()
        cp.cuda.Device().synchronize()
    py_sync_us = (time.perf_counter() - t0) * 1e6 / runs

    # C) Python wall-clock, sync once at end (queueing overlap)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        dec.decode()
    cp.cuda.Device().synchronize()
    py_batch_us = (time.perf_counter() - t0) * 1e6 / runs

    py_overhead = py_sync_us - kernel_us
    print(f"  A) kernel only (Events):         {kernel_us:7.1f} µs  "
          f"({n_t/kernel_us:7.1f} M tris/s)")
    print(f"  B) Python+sync per call:         {py_sync_us:7.1f} µs  "
          f"(overhead {py_overhead:+.1f} µs = {py_overhead/kernel_us*100:.1f}%)")
    print(f"  C) Python batched (sync at end): {py_batch_us:7.1f} µs  "
          f"({n_t/py_batch_us:7.1f} M tris/s)")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        bench(p)
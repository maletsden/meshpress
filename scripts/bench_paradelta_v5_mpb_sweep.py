"""Sweep meshlets_per_block on the fused v5 decoder."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cupy as cp

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5, decode_paradelta_v5
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder


def bench(path: str, mpb_values=(1, 2, 4, 8), warmup=5, runs=20) -> None:
    name = Path(path).name
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    n_t = prep["n_t"]
    data = encode_from_prepared_v5(prep, verbose=False)
    print(f"\n[{name}] n_t={n_t:,}")

    # CPU reference once
    v_cpu, t_cpu = decode_paradelta_v5(data)

    for mpb in mpb_values:
        try:
            dec = ParaDeltaV5GpuDecoder(data, meshlets_per_block=mpb)
        except Exception as e:
            print(f"  mpb={mpb}: ctor failed: {e}")
            continue
        # Correctness
        v_gpu, t_gpu = dec.decode_to_host()
        err = float(np.abs(v_gpu - v_cpu).max())
        tris_eq = np.array_equal(t_gpu.astype(np.int64), t_cpu)
        if not tris_eq or err > 1e-3:
            print(f"  mpb={mpb}: FAIL (err={err:.3g} tris_eq={tris_eq})")
            continue
        # Timing
        for _ in range(warmup):
            dec.decode()
        cp.cuda.Device().synchronize()
        start = cp.cuda.Event(); end = cp.cuda.Event()
        start.record()
        for _ in range(runs):
            dec.decode()
        end.record(); end.synchronize()
        kernel_us = cp.cuda.get_elapsed_time(start, end) * 1000.0 / runs
        smem = dec._shared_bytes
        per_warp = dec._per_warp_bytes
        print(f"  mpb={mpb}: {kernel_us:7.1f} µs  "
              f"({n_t/kernel_us:7.1f} M tris/s)  "
              f"smem/block={smem:>5} B (per-warp={per_warp})")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        bench(p)
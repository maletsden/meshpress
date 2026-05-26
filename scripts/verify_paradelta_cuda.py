"""Verify ParaDeltaCudaDecoder output matches decode_paradelta."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, decode_paradelta, decode_paradelta_to_struct,
)
from utils.paradelta_cuda import ParaDeltaCudaDecoder


def check(path: str, warmup: int = 5, runs: int = 20) -> None:
    name = Path(path).name
    prep = load_or_prepare(
        path, max_verts=256, max_tris=256, precision_error=0.0005,
        gen_method="joint_learned", strip_method="multiseed", verbose=False)
    data = encode_from_prepared(prep, predictor="linear5", verbose=False)
    print(f"\n[{name}] size={len(data):,} B  "
          f"BPV={len(data)*8/prep['n_v']:.2f}")

    # CPU reference
    t0 = time.time()
    v_ref, t_ref = decode_paradelta(data)
    t_cpu = time.time() - t0
    print(f"  decode_paradelta (CPU):  {t_cpu*1000:7.1f} ms  "
          f"verts={len(v_ref)} tris={len(t_ref)}")

    # Parse to struct
    t0 = time.time()
    s = decode_paradelta_to_struct(data)
    t_struct = time.time() - t0
    print(f"  struct parse (CPU):      {t_struct*1000:7.1f} ms")

    # Upload + build GPU decoder
    import cupy as cp
    t0 = time.time()
    dec = ParaDeltaCudaDecoder(s, block_size=128)
    cp.cuda.Device().synchronize()
    t_upload = time.time() - t0
    print(f"  GPU upload:              {t_upload*1000:7.1f} ms")

    # Correctness
    v_gpu, t_gpu = dec.decode_to_host()
    print(f"  shapes: v_gpu={v_gpu.shape} v_ref={v_ref.shape}  "
          f"t_gpu={t_gpu.shape} t_ref={t_ref.shape}")
    assert v_gpu.shape == v_ref.shape, \
        f"v shape {v_gpu.shape} != {v_ref.shape}"
    assert t_gpu.shape == t_ref.shape, \
        f"t shape {t_gpu.shape} != {t_ref.shape}"
    max_v_err = float(np.abs(v_gpu - v_ref).max())
    # tris come out int32; ref is int64
    tris_eq = np.array_equal(t_gpu.astype(np.int64), t_ref)
    print(f"  GPU vs CPU:              max_v_err={max_v_err:.4g}  "
          f"tris_match={tris_eq}")
    if max_v_err > 1e-3 or not tris_eq:
        print("  FAIL")
        sys.exit(1)

    # Timed runs (kernel only)
    for _ in range(warmup):
        dec.decode()
    cp.cuda.Device().synchronize()
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(runs):
        dec.decode()
    end.record()
    end.synchronize()
    kernel_ms = cp.cuda.get_elapsed_time(start, end) / runs
    print(f"  GPU kernel (avg of {runs}): {kernel_ms*1000:7.1f} µs  "
          f"({prep['n_t']/kernel_ms/1e3:6.1f} M tris/s)")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        check(p)
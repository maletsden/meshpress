"""Verify GPU bitstream parser matches CPU parser, then time it."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cupy as cp

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, decode_paradelta_to_struct,
)
from utils.paradelta_cuda_parse import ParaDeltaGpuParser


def check(path: str) -> None:
    name = Path(path).name
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    data = encode_from_prepared(prep, predictor="linear5", verbose=False)
    print(f"\n[{name}] size={len(data):,} B  n_t={prep['n_t']:,}")

    # CPU reference (only structural keys, skip ml_order)
    t0 = time.perf_counter()
    s_cpu = decode_paradelta_to_struct(data)
    t_cpu = time.perf_counter() - t0
    print(f"  CPU parse_to_struct (full): {t_cpu*1000:8.1f} ms")

    # GPU parser (no traversal). Time upload+kernel separately.
    cp.cuda.Device().synchronize()
    parser = ParaDeltaGpuParser(data)  # also runs parse_globals on CPU
    # warmup
    parser.parse_to_struct()
    cp.cuda.Device().synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        s_gpu = parser.parse_to_struct()
    cp.cuda.Device().synchronize()
    t_gpu = (time.perf_counter() - t0) / 5
    print(f"  GPU bit-decode (avg of 5):  {t_gpu*1000:8.1f} ms  "
          f"({prep['n_t']/t_gpu/1e6:.1f} M tris/s parser)")

    # Correctness check
    for key in ("ml_n_bnd", "ml_n_int", "ml_n_tris", "ml_n_strips",
                "ml_l2g_off", "ml_l2g",
                "ml_tris_off", "ml_codes_off"):
        if not np.array_equal(s_cpu[key], s_gpu[key]):
            print(f"  FAIL {key}: cpu_sum={int(s_cpu[key].sum())} "
                  f"gpu_sum={int(s_gpu[key].sum())}")
            # Show first 10 differing indices
            diff = np.where(s_cpu[key] != s_gpu[key])[0][:10]
            print(f"  first diffs at {diff}")
            for i in diff[:5]:
                print(f"    [{i}] cpu={s_cpu[key].flat[i]} "
                      f"gpu={s_gpu[key].flat[i]}")
            sys.exit(1)
    # ml_tris and ml_codes shapes match if offsets match; compare data
    for key in ("ml_tris", "ml_codes"):
        if not np.array_equal(s_cpu[key], s_gpu[key]):
            n_diff = int(np.sum(s_cpu[key] != s_gpu[key]))
            print(f"  FAIL {key}: {n_diff} entries differ")
            diff_rows = np.where(np.any(s_cpu[key] != s_gpu[key], axis=1))[0]
            for r in diff_rows[:5]:
                print(f"    row {r}: cpu={s_cpu[key][r]}  gpu={s_gpu[key][r]}")
            sys.exit(1)
    print(f"  OK  speedup_vs_cpu_full={t_cpu/t_gpu:.1f}x")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        check(p)
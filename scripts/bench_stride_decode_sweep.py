"""STRIDE-only decode sweep over all 8 paper meshes.

Same methodology for every mesh: 20 warmup + 100 measured iterations
bracketed by CUDA Events. Decode-only timing (kernel us), no encode.

Also measures VRAM footprint per mesh:
  - bitstream_b   compressed bitstream resident on device
  - vbuf_b        decoded vertex buffer (float32 * 3)
  - ibuf_b        decoded index buffer  (int32   * 3)
  - scratch_b     per-meshlet meta arrays + offset tables + counter
  - total_b       sum of the above (static footprint)
  - peak_b        cudaMemGetInfo delta around construction + decode
"""
import csv
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5 import encode_from_prepared_v5  # noqa: E402
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder  # noqa: E402
from utils.bench_config import stride_precision, csv_suffix, mode_label  # noqa: E402

_PREC = stride_precision()
_SUFFIX = csv_suffix()
print(f"[bench_stride_decode_sweep] precision = {mode_label()}")

MESHES = [
    "assets/fandisk.obj",
    "assets/stanford-bunny.obj",
    "assets/horse.obj",
    "assets/Monkey.obj",
    "assets/happy_buddha.obj",
    "assets/crab.obj",
    "assets/tank.obj",
    "assets/xyzrgb_dragon.obj",
]

WARMUP = 20
RUNS   = 100


def _arr_b(a) -> int:
    return int(a.nbytes) if a is not None else 0


def _vram_breakdown(dec: ParaDeltaV5GpuDecoder) -> dict:
    bitstream = _arr_b(dec.d_buf)
    vbuf      = _arr_b(dec.d_verts_norm)
    ibuf      = _arr_b(dec.d_tris)
    scratch   = (
        _arr_b(dec.d_off_bits)
        + _arr_b(dec.d_bnd_pos)
        + _arr_b(dec.d_n_bnd)
        + _arr_b(dec.d_n_int)
        + _arr_b(dec.d_n_tris)
        + _arr_b(dec.d_n_strips)
        + _arr_b(dec.d_tris_off)
        + _arr_b(dec.d_int_cursor)
        + _arr_b(dec.d_counter)
    )
    smem_per_block = int(dec._shared_bytes)
    n_blocks       = int(dec._persistent_blocks)
    smem_total     = smem_per_block * n_blocks
    return {
        "bitstream_b": bitstream,
        "vbuf_b":      vbuf,
        "ibuf_b":      ibuf,
        "scratch_b":   scratch,
        "smem_per_block_b": smem_per_block,
        "smem_total_b":     smem_total,
        "static_b":    bitstream + vbuf + ibuf + scratch,
    }


def _free_mem() -> int:
    free, _total = cp.cuda.runtime.memGetInfo()
    return int(free)


def time_one(p: str):
    full = ROOT / p
    if not full.exists():
        return None
    prep = load_or_prepare(str(full), max_verts=256, max_tris=256,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False,
                           **_PREC)
    data = encode_from_prepared_v5(prep, verbose=False)
    n_v = prep["n_v"] if isinstance(prep, dict) else prep.n_v
    n_t = prep["n_t"] if isinstance(prep, dict) else prep.n_t

    # Drain pool so memGetInfo reflects fresh allocations only.
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()
    free_before = _free_mem()

    dec = ParaDeltaV5GpuDecoder(data)
    # First decode also primes any lazy allocations.
    dec.decode()
    cp.cuda.Device().synchronize()
    free_after_first_decode = _free_mem()
    peak_b = max(0, free_before - free_after_first_decode)

    # Warmup
    for _ in range(WARMUP):
        dec.decode()
    cp.cuda.Device().synchronize()

    # CUDA Events
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(RUNS):
        dec.decode()
    e.record(); e.synchronize()
    kernel_us = cp.cuda.get_elapsed_time(s, e) * 1000.0 / RUNS
    mtps = n_t / kernel_us

    vram = _vram_breakdown(dec)
    row = {
        "mesh": p, "n_v": n_v, "n_t": n_t,
        "bytes": len(data), "bpv": 8 * len(data) / n_v,
        "kernel_us": kernel_us, "mtps": mtps,
        **vram,
        "peak_b": peak_b,
    }

    # Free before next mesh.
    del dec
    cp.get_default_memory_pool().free_all_blocks()
    return row


rows = []
for p in MESHES:
    print(f"\n=== {p} ===")
    t0 = time.perf_counter()
    r = time_one(p)
    if r is None:
        print("  missing")
        continue
    t_total = time.perf_counter() - t0
    print(f"  n_v={r['n_v']:,}  n_t={r['n_t']:,}  bpv={r['bpv']:.2f}")
    print(f"  kernel={r['kernel_us']:.1f} us   mtps={r['mtps']:.1f}   "
          f"bench_t={t_total:.1f}s")
    print(f"  VRAM: bitstream={r['bitstream_b']/1e6:.2f} MB  "
          f"vbuf={r['vbuf_b']/1e6:.2f} MB  ibuf={r['ibuf_b']/1e6:.2f} MB  "
          f"scratch={r['scratch_b']/1e6:.3f} MB")
    print(f"        smem/block={r['smem_per_block_b']/1024:.1f} KiB "
          f"x {r['smem_total_b']//max(1,r['smem_per_block_b'])} blocks  "
          f"static={r['static_b']/1e6:.2f} MB  peak={r['peak_b']/1e6:.2f} MB")
    rows.append(r)

out = ROOT / f"bench_stride_decode_sweep{_SUFFIX}.csv"
with open(out, "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
print(f"\nWritten: {out}")

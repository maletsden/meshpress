"""Diagnose where small-mesh STRIDE decode time goes.

For each mesh, measure:
  A. dec.decode() x 100 — current API (per-iter Python setup + kernel launch)
  B. Bare kernel launch x 100 (no per-iter Python setup)
  C. Single bare kernel launch (one event-bracketed call)
  D. CUDA graph capture of one decode, then replay x 100 (amortise host)

All timings are CUDA-Event-bracketed.
"""
import sys
from pathlib import Path

import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5 import encode_from_prepared_v5  # noqa: E402
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder  # noqa: E402

MESHES = [
    "assets/fandisk.obj",
    "assets/bunny.obj",
    "assets/eyeball.obj",
    "assets/horse.obj",
    "assets/stanford-bunny.obj",
]

WARMUP, RUNS = 20, 100


def launch_bare(dec):
    """Single kernel launch without the Python boundary-refresh / counter reset.

    Boundary refresh and counter reset are still required for correctness
    *between* logical decodes, but for timing the bare GPU cost we skip them.
    """
    n_blocks = min(dec.n_meshlets, dec._persistent_blocks)
    dec._fused_kernel(
        (n_blocks,), (32,),
        (dec.d_buf, dec.d_off_bits,
         dec.d_n_bnd, dec.d_n_int, dec.d_n_tris, dec.d_n_strips,
         dec.d_tris_off,
         cp.int32(dec.n_boundary), dec.d_int_cursor,
         dec.d_bnd_pos, cp.int32(dec.n_boundary),
         cp.float32(dec.lin3_w[0]),
         cp.float32(dec.lin3_w[1]),
         cp.float32(dec.lin3_w[2]),
         cp.float32(dec.delta),
         dec.d_verts_norm, dec.d_tris,
         cp.int32(dec.n_meshlets),
         cp.int32(dec._per_warp_bytes),
         cp.int32(dec.meshlets_per_block),
         dec.d_counter),
        shared_mem=dec._shared_bytes,
    )


def time_event_loop(fn, runs):
    cp.cuda.Device().synchronize()
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(runs):
        fn()
    e.record(); e.synchronize()
    return cp.cuda.get_elapsed_time(s, e) * 1000.0 / runs


def diagnose(p):
    full = ROOT / p
    if not full.exists():
        return
    prep = load_or_prepare(str(full), max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    data = encode_from_prepared_v5(prep, verbose=False)
    dec = ParaDeltaV5GpuDecoder(data)
    n_v = prep["n_v"] if isinstance(prep, dict) else prep.n_v
    n_t = prep["n_t"] if isinstance(prep, dict) else prep.n_t
    n_meshlets = dec.n_meshlets

    # Warmup once with the full API path
    for _ in range(WARMUP):
        dec.decode()

    # A. Full API loop
    a = time_event_loop(lambda: dec.decode(), RUNS)

    # B. Bare kernel loop, no per-iter Python setup
    b = time_event_loop(lambda: launch_bare(dec), RUNS)

    # C. Single bare kernel call (one event pair)
    c = time_event_loop(lambda: launch_bare(dec), 1)

    # D. CUDA graph capture + replay
    d_val = None
    try:
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            launch_bare(dec)
            graph = stream.end_capture()
        for _ in range(WARMUP):
            graph.launch(stream=stream)
        stream.synchronize()
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record(stream=stream)
        for _ in range(RUNS):
            graph.launch(stream=stream)
        e.record(stream=stream); e.synchronize()
        d_val = cp.cuda.get_elapsed_time(s, e) * 1000.0 / RUNS
    except Exception as ex:
        print(f"  graph fail: {ex}")

    print(f"\n{p}")
    print(f"  n_v={n_v:,}  n_t={n_t:,}  n_meshlets={n_meshlets}")
    print(f"  A. dec.decode() x100        = {a:7.1f} us  ({n_t/a:7.1f} mtps)")
    print(f"  B. bare kernel x100         = {b:7.1f} us  ({n_t/b:7.1f} mtps)")
    print(f"  C. bare kernel x1 (event)   = {c:7.1f} us  ({n_t/c:7.1f} mtps)")
    if d_val is not None:
        print(f"  D. CUDA graph replay x100   = {d_val:7.1f} us  ({n_t/d_val:7.1f} mtps)")


for p in MESHES:
    diagnose(p)
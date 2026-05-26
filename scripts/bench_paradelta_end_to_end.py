"""End-to-end bench: GPU parse + CPU greedy + GPU recon (V3) vs all-CPU.

Phase 2a has GPU bit-decode; greedy_order still on CPU. Reports the full
pipeline timings so the residual bottleneck is visible.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cupy as cp
from collections import deque

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, decode_paradelta, decode_paradelta_to_struct,
    _build_meshlet_local_topo, _KIND_MAP,
)
from utils.parallelogram_predictor import _greedy_order
from utils.paradelta_cuda_parse import ParaDeltaGpuParser
from utils.paradelta_cuda_lin5 import ParaDeltaLin5Warp


def gpu_parse_cpu_greedy(parser: ParaDeltaGpuParser) -> dict:
    s = parser.parse_to_struct()
    # CPU greedy_order pass (the residual bottleneck).
    n_meshlets = s["n_meshlets"]
    order_chunks = []
    for ml in range(n_meshlets):
        n_bnd = int(s["ml_n_bnd"][ml]); n_int = int(s["ml_n_int"][ml])
        if n_int == 0:
            order_chunks.append(np.zeros((0, 7), dtype=np.int32))
            continue
        n_local = n_bnd + n_int
        ts, te = int(s["ml_tris_off"][ml]), int(s["ml_tris_off"][ml + 1])
        tri_arr = s["ml_tris"][ts:te]
        edge_to_tris_local, vert_to_tris_local = \
            _build_meshlet_local_topo(tri_arr.astype(np.int64))
        order = _greedy_order(
            list(range(n_bnd, n_local)), list(range(n_bnd)),
            tri_arr.astype(np.int64),
            edge_to_tris_local, vert_to_tris_local)
        order_arr = np.full((n_int, 7), -1, dtype=np.int32)
        for i, (v_local, kind, refs) in enumerate(order):
            order_arr[i, 0] = v_local
            order_arr[i, 1] = _KIND_MAP[kind]
            for j, ref in enumerate(refs):
                if j >= 5:
                    break
                order_arr[i, 2 + j] = int(ref)
        order_chunks.append(order_arr)
    sizes = np.array([c.shape[0] for c in order_chunks], dtype=np.int32)
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    flat = np.concatenate(order_chunks, axis=0) if order_chunks else \
        np.zeros((0, 7), dtype=np.int32)
    s["ml_order"] = flat
    s["ml_order_off"] = offs
    return s


def check(path: str) -> None:
    name = Path(path).name
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    data = encode_from_prepared(prep, predictor="linear5", verbose=False)
    print(f"\n[{name}] size={len(data):,} B  "
          f"BPV={len(data)*8/prep['n_v']:.2f}  n_t={prep['n_t']:,}")

    v_ref, _ = decode_paradelta(data)  # warmup numba

    # Pipeline timings (1-shot end-to-end)
    parser = ParaDeltaGpuParser(data)
    # warmup parse + greedy
    s_test = gpu_parse_cpu_greedy(parser)

    runs = 3
    t_parse = 0.0; t_greedy = 0.0; t_recon = 0.0
    for _ in range(runs):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        s = parser.parse_to_struct()
        cp.cuda.Device().synchronize()
        t_parse += time.perf_counter() - t0

        t0 = time.perf_counter()
        # Greedy only
        n_meshlets = s["n_meshlets"]
        order_chunks = []
        for ml in range(n_meshlets):
            n_bnd = int(s["ml_n_bnd"][ml]); n_int = int(s["ml_n_int"][ml])
            if n_int == 0:
                order_chunks.append(np.zeros((0, 7), dtype=np.int32)); continue
            n_local = n_bnd + n_int
            ts, te = int(s["ml_tris_off"][ml]), int(s["ml_tris_off"][ml + 1])
            tri_arr = s["ml_tris"][ts:te]
            edge_to_tris_local, vert_to_tris_local = \
                _build_meshlet_local_topo(tri_arr.astype(np.int64))
            order = _greedy_order(
                list(range(n_bnd, n_local)), list(range(n_bnd)),
                tri_arr.astype(np.int64),
                edge_to_tris_local, vert_to_tris_local)
            order_arr = np.full((n_int, 7), -1, dtype=np.int32)
            for i, (v_local, kind, refs) in enumerate(order):
                order_arr[i, 0] = v_local
                order_arr[i, 1] = _KIND_MAP[kind]
                for j, ref in enumerate(refs):
                    if j >= 5: break
                    order_arr[i, 2 + j] = int(ref)
            order_chunks.append(order_arr)
        sizes = np.array([c.shape[0] for c in order_chunks], dtype=np.int32)
        offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
        flat = np.concatenate(order_chunks, axis=0)
        s["ml_order"] = flat; s["ml_order_off"] = offs
        t_greedy += time.perf_counter() - t0

        # Recon
        s_for_recon = dict(s)
        for k in list(s_for_recon.keys()):
            if k.startswith("_d_"):
                del s_for_recon[k]
        dec = ParaDeltaLin5Warp(s_for_recon)
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        v_gpu, t_gpu = dec.decode_to_host()
        cp.cuda.Device().synchronize()
        t_recon += time.perf_counter() - t0

    t_parse /= runs; t_greedy /= runs; t_recon /= runs
    total = t_parse + t_greedy + t_recon
    # Correctness vs reference
    if v_gpu.shape == v_ref.shape:
        err = float(np.abs(v_gpu - v_ref).max())
    else:
        err = float("nan")
    print(f"  pipeline (avg {runs}):")
    print(f"    GPU parse :   {t_parse*1000:8.1f} ms  "
          f"({t_parse/total*100:5.1f}%)")
    print(f"    CPU greedy:   {t_greedy*1000:8.1f} ms  "
          f"({t_greedy/total*100:5.1f}%)")
    print(f"    GPU recon :   {t_recon*1000:8.1f} ms  "
          f"({t_recon/total*100:5.1f}%)")
    print(f"    TOTAL     :   {total*1000:8.1f} ms  "
          f"({prep['n_t']/total/1e6:.1f} M tris/s)  err={err:.3g}")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        check(p)
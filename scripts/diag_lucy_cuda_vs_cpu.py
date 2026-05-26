"""Compare actual CUDA decoder output vs CPU f32 simulator.

End-to-end:
  1. Encode Lucy → bitstream
  2. Decode with CUDA → v_cuda (N_v, 3) world coords
  3. CPU-sim decode (mirror logic) → v_cpu (N_v, 3) world coords
  4. Diff per-vert; identify divergent verts + meshlets

Goal: pinpoint where CUDA differs from CPU model.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import (
    encode_from_prepared_v5, _interior_pass_strip, _strip_traversal,
)
from encoder.paradelta_codec import _fit_linear3
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder, parse_globals_v5


def _cpu_decode_full(prep, lin3_w_f32, bnd_pos_norm_f32):
    """Replay the f32 decoder logic for every meshlet using prep data."""
    plans = prep["plans"]
    vn_dim = prep["n_v"]
    out_norm = np.zeros((vn_dim, 3), dtype=np.float32)
    n_boundary = prep["n_boundary"]
    boundary_list = prep["boundary_list"]
    out_norm[:n_boundary] = bnd_pos_norm_f32

    delta_f32 = np.float32(2.0 * float(prep["per_coord_err"]))
    w0 = np.float32(lin3_w_f32[0])
    w1 = np.float32(lin3_w_f32[1])
    w2 = np.float32(lin3_w_f32[2])
    int_cursor = 0

    for plan in plans:
        n_bnd = plan["n_bnd"]
        n_int = plan["n_int"]
        if n_int == 0:
            continue
        local_to_global = plan["local_to_global"]

        # Boundary recon: pull from out_norm via local_to_global indirection
        # (matching CUDA Stage B's gather from bnd_pos[g*3+d]).
        recon = np.zeros((n_bnd + n_int, 3), dtype=np.float32)
        for lid in range(n_bnd):
            g = int(local_to_global[lid])
            recon[lid] = out_norm[g]

        # Fallback = mean(recon[0..n_bnd]).
        if n_bnd > 0:
            fallback = recon[:n_bnd].mean(axis=0).astype(np.float32)
        else:
            fallback = np.zeros(3, dtype=np.float32)

        # Re-encode-side codes (matches actual bitstream codes).
        # Pass w3 same as encoder uses (lin3_w_f32 promoted to f64).
        w3_for_enc = lin3_w_f32.astype(np.float64)
        codes, _ = _interior_pass_strip(plan,
                                          prep["vn"], prep["bnd_recon_norm"],
                                          2.0 * float(prep["per_coord_err"]),
                                          w3=w3_for_enc)
        order = _strip_traversal(plan["ml_tris_local"], plan["strips"],
                                   n_bnd)
        for i, (v_local, kind, refs) in enumerate(order):
            c = codes[i].astype(np.float32)
            if kind == 'para':
                a, b, cc = refs
                pred = w0 * recon[a] + w1 * recon[b] + w2 * recon[cc]
            else:
                pred = fallback
            recon[v_local] = pred + c * delta_f32

        # Write interior to global out array. The decoder writes interior
        # to gid = int_base + (v_local - n_bnd) with int_base = n_boundary
        # + interior_cursor[m]. interior_cursor[m] is prefix sum of n_ints.
        for v_local in range(n_bnd, n_bnd + n_int):
            gid = n_boundary + int_cursor + (v_local - n_bnd)
            out_norm[gid] = recon[v_local]
        int_cursor += n_int
    return out_norm


def main(path: str = "assets/lucy.obj"):
    print(f"Loading prep for {path} ...")
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                            precision_error=0.0005,
                            gen_method="joint_learned",
                            strip_method="multiseed",
                            clean=True, verbose=False)
    scale = float(prep["scale"])
    center = np.asarray(prep["center"], dtype=np.float32)
    print(f"  meshlets: {len(prep['plans']):,}  scale={scale:.2f}")

    print(f"\nEncoding (one-shot, slow) ...")
    t0 = time.time()
    data = encode_from_prepared_v5(prep, verbose=False)
    print(f"  encode: {time.time()-t0:.1f}s  size={len(data):,}")

    print(f"\nCUDA decode ...")
    dec = ParaDeltaV5GpuDecoder(data)
    v_cuda, _ = dec.decode_to_host()
    v_cuda = v_cuda.astype(np.float32)
    print(f"  v_cuda.shape={v_cuda.shape}")

    print(f"\nCPU simulator decode ...")
    g = parse_globals_v5(data)
    lin3_f32 = np.asarray(g["lin3_w"], dtype=np.float32)
    bnd_pos_norm = np.asarray(g["bnd_pos_norm"], dtype=np.float32)
    t0 = time.time()
    v_cpu_norm = _cpu_decode_full(prep, lin3_f32, bnd_pos_norm)
    v_cpu = (v_cpu_norm * np.float32(scale) + center).astype(np.float32)
    print(f"  cpu-sim: {time.time()-t0:.1f}s")

    diff = np.linalg.norm(
        v_cuda.astype(np.float64) - v_cpu.astype(np.float64), axis=1)
    print(f"\n=== CUDA vs CPU-sim per-vertex diff (world units) ===")
    print(f"  p50  = {float(np.percentile(diff, 50)):.6f}")
    print(f"  p99  = {float(np.percentile(diff, 99)):.6f}")
    print(f"  p99.9= {float(np.percentile(diff, 99.9)):.6f}")
    print(f"  max  = {float(diff.max()):.6f}")
    print(f"  n verts with diff > 1.0: {int((diff > 1.0).sum()):,}")
    print(f"  n verts with diff > 10.0: {int((diff > 10.0).sum()):,}")

    # Source-vs-CUDA NN err for reference
    from scipy.spatial import cKDTree
    src = np.asarray([[v.x, v.y, v.z]
                       for v in
                       __import__('reader.reader',
                                   fromlist=['Reader']).Reader
                       .read_from_file(path).vertices],
                      dtype=np.float32)
    from utils.mesh_clean import clean_mesh
    from reader.reader import Reader
    cleaned, _ = clean_mesh(Reader.read_from_file(path), verbose=False)
    src = np.array([[v.x, v.y, v.z] for v in cleaned.vertices],
                    dtype=np.float32)
    t1 = cKDTree(v_cuda); d1, _ = t1.query(src, k=1)
    t2 = cKDTree(src); d2, _ = t2.query(v_cuda, k=1)
    print(f"\n  CUDA-vs-src NN err: max={max(float(d1.max()), float(d2.max())):.4f}")

    t1 = cKDTree(v_cpu); d1, _ = t1.query(src, k=1)
    t2 = cKDTree(src); d2, _ = t2.query(v_cpu, k=1)
    print(f"  CPU -vs-src NN err: max={max(float(d1.max()), float(d2.max())):.4f}")

    # Locate worst diff vert
    if diff.max() > 1.0:
        worst = int(np.argmax(diff))
        print(f"\n  worst diff at global v_idx={worst}:")
        print(f"    v_cuda = {v_cuda[worst]}")
        print(f"    v_cpu  = {v_cpu[worst]}")
        print(f"    src    = {src[worst] if worst < len(src) else 'oob'}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "assets/lucy.obj"
    main(p)
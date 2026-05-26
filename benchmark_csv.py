"""
Benchmark all meshlet encoders and write results to CSV.

Produces benchmarks.csv with stable, reproducible results.
Run:
  python benchmark_csv.py
"""

import csv
import io
import os
import sys
import time
import contextlib
import numpy as np

from reader import Reader
from encoder import (
    BaselineEncoder,
    MeshletWaveletGlobalEB, MeshletWaveletGlobalAMD, MeshletPlainAMD,
    MeshletGTSPlain, MeshletGTSSegDelta, MeshletGTSHaar,
)
from encoder.implementation.meshlet_wavelet import (
    _to_numpy, _global_quantize, _dequantize_global,
)

# ---------- config ----------
MODELS = [
    "assets/bunny.obj",
    "assets/torus.obj",
    "assets/stanford-bunny.obj",
    "assets/Monkey.obj",
]
MAX_ERROR = 0.0005
MAX_VERTS = 256
N_RUNS = 3          # median of N encoding runs for timing
CSV_PATH = "benchmarks.csv"

ENCODERS = [
    ("Baseline",            lambda: BaselineEncoder()),
    ("GlobalEB",            lambda: MeshletWaveletGlobalEB(MAX_VERTS, MAX_ERROR, verbose=False)),
    ("GlobalAMD",           lambda: MeshletWaveletGlobalAMD(MAX_VERTS, MAX_ERROR, verbose=False)),
    ("PlainAMD",            lambda: MeshletPlainAMD(MAX_VERTS, MAX_ERROR, verbose=False)),
    ("GTS+Plain",           lambda: MeshletGTSPlain(MAX_VERTS, MAX_ERROR, verbose=False)),
    ("GTS+SegDelta",        lambda: MeshletGTSSegDelta(MAX_VERTS, MAX_ERROR, verbose=False)),
    ("GTS+Haar",            lambda: MeshletGTSHaar(MAX_VERTS, MAX_ERROR, verbose=False)),
]


def compute_metrics(model, max_error):
    """Compute reference metrics: n_verts, n_tris, raw_bytes, quantization max_err."""
    verts_np, tris_np = _to_numpy(model)
    n_v, n_t = len(verts_np), len(tris_np)
    raw_bytes = n_v * 12 + n_t * 12   # float32 x3 + uint32 x3
    # Quantization error (for crack-free encoders)
    center = verts_np.mean(axis=0)
    vc = verts_np - center
    scale = np.max(np.linalg.norm(vc, axis=1))
    vn = vc / scale
    per_coord_err = max_error / scale / np.sqrt(3)
    codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)
    recon = _dequantize_global(codes, g_min, g_range, g_bits)
    errs = np.linalg.norm(recon - vn, axis=1) * scale
    return n_v, n_t, raw_bytes, float(errs.max()), float((errs <= max_error).mean() * 100)


def run_encoder(encoder, model, n_runs):
    """Run encoder n_runs times, return (CompressedModel, median_time_ms)."""
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = encoder.encode(model)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return result, float(np.median(times))


def main():
    rows = []

    for model_path in MODELS:
        if not os.path.exists(model_path):
            print(f"[SKIP] {model_path} not found")
            continue

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"\n=== {model_name} ({model_path}) ===")
        model = Reader.read_from_file(model_path)

        n_v, n_t, raw_bytes, quant_max_err, quant_pct = compute_metrics(
            model, MAX_ERROR)
        print(f"  Vertices: {n_v:,}  Triangles: {n_t:,}  "
              f"Raw: {raw_bytes:,} B  QuantMaxErr: {quant_max_err:.6f}")

        for enc_name, enc_factory in ENCODERS:
            encoder = enc_factory()
            try:
                compressed, enc_time_ms = run_encoder(encoder, model, N_RUNS)
            except Exception as e:
                print(f"  {enc_name:<20s}  ERROR: {e}")
                continue

            total_bytes = len(compressed.data)
            bpv = compressed.bits_per_vertex
            bpt = compressed.bits_per_triangle
            ratio = raw_bytes / total_bytes if total_bytes > 0 else 0

            # For Baseline encoder, error metrics are 0/100%
            if enc_name == "Baseline":
                max_err = 0.0
                pct_ok = 100.0
            else:
                max_err = quant_max_err
                pct_ok = quant_pct

            row = {
                "model": model_name,
                "n_verts": n_v,
                "n_tris": n_t,
                "encoder": enc_name,
                "total_bytes": total_bytes,
                "raw_bytes": raw_bytes,
                "bpv": round(bpv, 2),
                "bpt": round(bpt, 2),
                "ratio": round(ratio, 2),
                "max_err": round(max_err, 6),
                "pct_within_target": round(pct_ok, 2),
                "encode_time_ms": round(enc_time_ms, 1),
            }
            rows.append(row)
            print(f"  {enc_name:<20s}  {total_bytes:>10,} B  "
                  f"BPV={bpv:>7.2f}  ratio={ratio:>5.1f}x  "
                  f"time={enc_time_ms:>7.1f} ms")

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {CSV_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

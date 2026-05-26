"""Parity check: STRIDE-dup CUDA decoder vs CPU decoder.

Encodes, runs CUDA kernel with --dump, runs the Python CPU decoder
(decode_dup) on the same bitstream, compares.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5_dup import encode_dup, decode_dup  # noqa: E402
from scripts.dump_stride_dup_blob import dump_blob  # noqa: E402

DUP_BENCH = ROOT / "bench_cpp" / "stride_dup_decode_bench.exe"
BLOB_DIR  = ROOT / "bench_cpp" / "blobs" / "dup"


def main():
    paths = sys.argv[1:] or ["assets/bunny.obj"]
    for p in paths:
        full = ROOT / p
        blob = BLOB_DIR / (full.stem + ".dup.blob")
        BLOB_DIR.mkdir(parents=True, exist_ok=True)
        dump_blob(full, blob)

        # CPU reference: re-encode + decode_dup on the same bitstream.
        prep = load_or_prepare(str(full), max_verts=256, max_tris=256,
                                precision_error=1.0/4096.0,
                                precision_mode="bbox_frac",
                                gen_method="joint_learned",
                                strip_method="multiseed", verbose=False)
        data = encode_dup(prep, verbose=False)
        cpu_v, cpu_t = decode_dup(data)

        with tempfile.TemporaryDirectory() as td:
            prefix = Path(td) / "gpu"
            r = subprocess.run([str(DUP_BENCH), str(blob), "1", "1",
                                 "--dump", str(prefix)],
                                capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout); print(r.stderr, file=sys.stderr)
                sys.exit(1)
            gpu_v = np.fromfile(str(prefix) + ".verts.f32",
                                 dtype=np.float32).reshape(-1, 3)
            gpu_t = np.fromfile(str(prefix) + ".tris.u32",
                                 dtype=np.uint32).reshape(-1, 3)

        print(f"\n=== {full.name} ===")
        print(f"  CPU verts: {cpu_v.shape}  GPU verts: {gpu_v.shape}")
        print(f"  CPU tris : {cpu_t.shape}   GPU tris : {gpu_t.shape}")
        # Tri parity (exact uint32 match).
        t_eq = np.array_equal(gpu_t.astype(np.int64), cpu_t)
        v_eq = np.array_equal(gpu_v.view(np.uint32), cpu_v.view(np.uint32))
        v_close = np.allclose(gpu_v, cpu_v, atol=1e-4, rtol=1e-3)
        print(f"  tri exact: {t_eq}")
        print(f"  vert bit-exact: {v_eq}   vert close (1e-4): {v_close}")
        if not t_eq:
            diff = np.where((gpu_t.astype(np.int64) != cpu_t).any(axis=1))[0]
            print(f"    tri mismatches: {len(diff)} / {len(cpu_t)}")
            if len(diff) > 0:
                k = int(diff[0])
                print(f"    first: tri {k} GPU={gpu_t[k]} CPU={cpu_t[k]}")
        if not v_close:
            diff = np.where(np.abs(gpu_v.astype(np.float64) - cpu_v.astype(np.float64)).max(axis=1) > 1e-4)[0]
            print(f"    vert mismatches: {len(diff)} / {len(cpu_v)}")
            if len(diff) > 0:
                k = int(diff[0])
                print(f"    first: vert {k} GPU={gpu_v[k]} CPU={cpu_v[k]}")


if __name__ == "__main__":
    main()

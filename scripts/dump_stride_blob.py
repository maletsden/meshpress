"""Dump a STRIDE decode-ready blob for the C++ bench harness.

Encodes the mesh with the Python STRIDE encoder, then writes all GPU-bound
buffers in one flat binary file. The C++ harness loads this blob, uploads
to GPU, and launches the fused decode kernel.

Blob layout (little-endian throughout):
  Header (fixed 64 B):
    u32 magic            = 'SBLB' (0x424c4253)
    u32 version          = 1
    u32 code_width       (0=i16 codes, 1=i32 codes)
    u32 n_meshlets
    u32 n_boundary
    u32 n_v_total
    u32 n_t_total
    u32 buf_size
    f32 lin3_w0
    f32 lin3_w1
    f32 lin3_w2
    f32 delta
    u32 per_warp_bytes
    u32 meshlets_per_block
    u32 _reserved[2]
  Then sections appended in order:
    buf            : u8  × buf_size           (the bitstream)
    ml_off_bits    : u64 × n_meshlets
    ml_n_bnd       : i32 × n_meshlets
    ml_n_int       : i32 × n_meshlets
    ml_n_tris      : i32 × n_meshlets
    ml_n_strips    : i32 × n_meshlets
    ml_tris_off    : i32 × (n_meshlets+1)
    ml_int_cursor  : i32 × n_meshlets
    bnd_pos_norm   : f32 × 3 × n_boundary
"""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5 import encode_from_prepared_v5  # noqa: E402
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder  # noqa: E402

MAGIC = 0x424c4253  # 'SBLB'
VERSION = 1


def dump_blob(obj_path: str, out_path: str, *,
              precision_error: float = 0.0005,
              precision_mode: str = "world",
              max_verts: int = 256,
              max_tris: int = 256):
    full = ROOT / obj_path
    if not full.exists():
        print(f"missing: {obj_path}")
        return
    prep = load_or_prepare(str(full), max_verts=max_verts, max_tris=max_tris,
                           precision_error=precision_error,
                           precision_mode=precision_mode,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    data = encode_from_prepared_v5(prep, verbose=False)
    dec = ParaDeltaV5GpuDecoder(data)

    # Pull buffers off GPU (cupy → host arrays)
    import cupy as cp
    ml_off_bits = cp.asnumpy(dec.d_off_bits).astype(np.uint64, copy=False)
    ml_n_bnd    = cp.asnumpy(dec.d_n_bnd).astype(np.int32, copy=False)
    ml_n_int    = cp.asnumpy(dec.d_n_int).astype(np.int32, copy=False)
    ml_n_tris   = cp.asnumpy(dec.d_n_tris).astype(np.int32, copy=False)
    ml_n_strips = cp.asnumpy(dec.d_n_strips).astype(np.int32, copy=False)
    ml_tris_off = cp.asnumpy(dec.d_tris_off).astype(np.int32, copy=False)
    ml_int_cur  = cp.asnumpy(dec.d_int_cursor).astype(np.int32, copy=False)
    bnd_pos     = cp.asnumpy(dec.d_bnd_pos).astype(np.float32, copy=False)

    assert ml_off_bits.size == dec.n_meshlets
    assert ml_tris_off.size == dec.n_meshlets + 1
    assert bnd_pos.size == 3 * dec.n_boundary

    header = struct.pack(
        "<IIIIIIIIffffII II",
        MAGIC, VERSION, dec.code_width, dec.n_meshlets,
        dec.n_boundary, dec.n_v_total, dec.n_t_total, len(data),
        float(dec.lin3_w[0]), float(dec.lin3_w[1]), float(dec.lin3_w[2]),
        float(dec.delta),
        dec._per_warp_bytes, dec.meshlets_per_block,
        0, 0,
    )
    assert len(header) == 64

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(bytes(data))
        f.write(ml_off_bits.tobytes())
        f.write(ml_n_bnd.tobytes())
        f.write(ml_n_int.tobytes())
        f.write(ml_n_tris.tobytes())
        f.write(ml_n_strips.tobytes())
        f.write(ml_tris_off.tobytes())
        f.write(ml_int_cur.tobytes())
        f.write(bnd_pos.tobytes())

    print(f"  {obj_path}: n_v={dec.n_v_total:,} n_t={dec.n_t_total:,} "
          f"n_meshlets={dec.n_meshlets} "
          f"buf={len(data)} smem/warp={dec._per_warp_bytes} "
          f"code_width={dec.code_width} → {out_path}")


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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--q12bbox", action="store_true",
                     help="use bbox-relative q12 quantization (matches DGF tb12)")
    ap.add_argument("--meshes", nargs="*", default=None)
    ap.add_argument("--max-verts", type=int, default=256)
    ap.add_argument("--max-tris", type=int, default=256)
    args = ap.parse_args()

    if args.q12bbox:
        prec = dict(precision_error=1.0 / 4096.0, precision_mode="bbox_frac")
        suffix = "_q12bbox"
    else:
        prec = dict(precision_error=0.0005, precision_mode="world")
        suffix = ""
    if args.max_verts != 256:
        suffix += f"_mv{args.max_verts}"

    out_dir = ROOT / "bench_cpp" / "blobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = args.meshes if args.meshes else MESHES
    for p in paths:
        stem = Path(p).stem
        dump_blob(p, str(out_dir / f"{stem}{suffix}.blob"),
                   max_verts=args.max_verts, max_tris=args.max_tris,
                   **prec)
    print(f"\nBlobs in: {out_dir}")
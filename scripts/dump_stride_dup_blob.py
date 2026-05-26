"""Dump a STRIDE-dup blob for stride_dup_decode_bench.exe.

Encodes the mesh via paradelta_v5_dup.encode_dup, then parses the
resulting bitstream to extract per-meshlet metadata (n_local, n_tris,
n_strips, bit offsets) + computes prefix sums for v_base / t_base.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5_dup import (  # noqa: E402
    encode_dup, MAGIC_DUP, VERSION_DUP,
)
from utils.bit_codec import BitReader  # noqa: E402

BLOB_MAGIC = 0x42525044  # 'DPRB' as C string LE
BLOB_VERSION = 4  # v4 blob: + 21 B generalized parallelogram predictor header


def _parse_blob_for_meta(data: bytes, n_meshlets: int):
    """Re-parse the encoded bitstream to recover meshlet bit offsets +
    per-meshlet header (n_local/n_tris/n_strips)."""
    r = BitReader(data)
    if r.read_fixed(32) != MAGIC_DUP:
        raise ValueError("bad magic")
    if r.read_fixed(8) != VERSION_DUP:
        raise ValueError("bad version")
    code_width = r.read_fixed(8)
    center = [r.read_f32() for _ in range(3)]
    scale = r.read_f32()
    per_coord_err = r.read_f32()
    g_min = [r.read_f32() for _ in range(3)]
    g_range = [r.read_f32() for _ in range(3)]
    g_bits = [r.read_fixed(8) for _ in range(3)]
    # v3 predictor header: 9 int16 numerators + 3 uint8 K
    pred_n = []
    for _ in range(9):
        v_u16 = r.read_fixed(16)
        pred_n.append(v_u16 - 0x10000 if v_u16 & 0x8000 else v_u16)
    pred_K = [r.read_fixed(8) for _ in range(3)]
    n_v_total = r.read_fixed(32)
    n_t_total = r.read_fixed(32)
    nm = r.read_fixed(32)
    if nm != n_meshlets:
        raise ValueError(f"n_meshlets {nm} vs {n_meshlets}")
    pad = (-r.bit_pos()) & 7
    if pad:
        r.read_bits(pad)
    abs_offsets = [r.read_fixed(32) for _ in range(n_meshlets)]
    meshlet_region_start = r.bit_pos()
    # Relative offsets into the meshlet region.
    rel_offsets = [int(a) - meshlet_region_start for a in abs_offsets]
    return {
        "code_width": code_width,
        "center": center, "scale": scale, "per_coord_err": per_coord_err,
        "g_min": g_min, "g_range": g_range, "g_bits": g_bits,
        "pred_n": pred_n, "pred_K": pred_K,
        "n_v_total": n_v_total, "n_t_total": n_t_total,
        "meshlet_region_start": meshlet_region_start,
        "rel_offsets": rel_offsets,
    }


def _read_per_meshlet_meta(body_bytes: bytes, rel_offsets: list[int],
                            n_meshlets: int):
    """For each meshlet, read its first 3×16-bit header words."""
    r = BitReader(body_bytes)
    n_local_arr = np.zeros(n_meshlets, dtype=np.uint32)
    n_tris_arr  = np.zeros(n_meshlets, dtype=np.uint32)
    n_strips_arr = np.zeros(n_meshlets, dtype=np.uint32)
    for i, off in enumerate(rel_offsets):
        r._pos = off  # absolute bit position into body_bytes
        n_local_arr[i] = r.read_fixed(16)
        n_tris_arr[i]  = r.read_fixed(16)
        n_strips_arr[i] = r.read_fixed(16)
    return n_local_arr, n_tris_arr, n_strips_arr


def dump_blob(mesh_path: Path, out_path: Path,
              max_verts: int = 256, max_tris: int = 256):
    prep = load_or_prepare(str(mesh_path),
                            max_verts=max_verts, max_tris=max_tris,
                            precision_error=1.0/4096.0,
                            precision_mode="bbox_frac",
                            gen_method="joint_learned",
                            strip_method="multiseed", verbose=False)
    data, enc_meta = encode_dup(prep, verbose=False, return_meta=True)
    n_meshlets = prep["n_meshlets"]
    meta = _parse_blob_for_meta(data, n_meshlets)

    # Split bytes into header + body. body_start = meshlet_region_start (bits).
    body_start_bit = meta["meshlet_region_start"]
    assert body_start_bit % 8 == 0
    body_start_byte = body_start_bit // 8
    body_bytes = data[body_start_byte:]

    n_local_arr, n_tris_arr, n_strips_arr = _read_per_meshlet_meta(
        body_bytes, meta["rel_offsets"], n_meshlets)
    v_off = np.concatenate([[0], np.cumsum(n_local_arr[:-1])]).astype(np.uint32)
    t_off = np.concatenate([[0], np.cumsum(n_tris_arr[:-1])]).astype(np.uint32)

    n_v_total = int(n_local_arr.sum())
    n_t_total = int(n_tris_arr.sum())
    assert n_v_total == meta["n_v_total"], f"{n_v_total} vs {meta['n_v_total']}"

    # Build new side tables from encoder meta.
    resid_off_bits_arr = np.asarray(enc_meta["resid_offs_rel"], dtype=np.uint64)
    n_kind0_arr        = np.asarray(enc_meta["n_kind0"],        dtype=np.uint32)
    axis_sub_offs_arr  = np.asarray(enc_meta["axis_sub_offs"],  dtype=np.uint16)
    assert resid_off_bits_arr.shape[0] == n_meshlets
    assert n_kind0_arr.shape[0]        == n_meshlets
    assert axis_sub_offs_arr.shape == (n_meshlets, 5)

    # Write blob.
    with open(out_path, "wb") as f:
        # Header (96 bytes, matches DupBlobHeader in .cu).
        f.write(struct.pack("<II", BLOB_MAGIC, BLOB_VERSION))     # magic, version
        f.write(struct.pack("<III", n_meshlets, n_v_total, n_t_total))
        f.write(struct.pack("<I", len(body_bytes)))               # buf_size
        f.write(struct.pack("<II", 0, 0))                          # _resv
        f.write(struct.pack("<fff", *meta["center"]))
        f.write(struct.pack("<f", meta["scale"]))
        f.write(struct.pack("<f", meta["per_coord_err"]))
        f.write(struct.pack("<fff", *meta["g_min"]))
        f.write(struct.pack("<fff", *meta["g_range"]))
        f.write(struct.pack("<III", *meta["g_bits"]))
        # v4 predictor header: 9 int16 + 3 uint8 + 3 pad → 24 B (4-aligned)
        f.write(struct.pack("<9h", *meta["pred_n"]))
        f.write(struct.pack("<3B", *meta["pred_K"]))
        f.write(b"\x00\x00\x00")  # 3 byte pad to 4-byte boundary
        f.write(body_bytes)
        # Absolute bit offsets INTO body_bytes (rel_offsets).
        f.write(np.array(meta["rel_offsets"], dtype=np.uint64).tobytes())
        f.write(n_local_arr.tobytes())
        f.write(n_tris_arr.tobytes())
        f.write(n_strips_arr.tobytes())
        f.write(v_off.tobytes())
        f.write(t_off.tobytes())
        # v2 additions
        f.write(resid_off_bits_arr.tobytes())   # u64 × n_meshlets
        f.write(n_kind0_arr.tobytes())          # u32 × n_meshlets
        # v3 additions
        f.write(axis_sub_offs_arr.tobytes())    # u16 × 5 × n_meshlets

    bpv = 8 * len(data) / prep["n_v"]
    print(f"  {mesh_path.name}: n_v_src={prep['n_v']:,} "
          f"n_v_dup={n_v_total:,} n_t={n_t_total:,} "
          f"n_meshlets={n_meshlets:,} buf={len(body_bytes):,}  "
          f"BPV(src)={bpv:.2f} → {out_path.name}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", nargs="*", default=None)
    ap.add_argument("--max-verts", type=int, default=256)
    ap.add_argument("--max-tris", type=int, default=256)
    args = ap.parse_args()

    out_dir = ROOT / "bench_cpp" / "blobs" / "dup"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.max_verts == 256 else f"_mv{args.max_verts}"
    paths = args.meshes if args.meshes else MESHES
    for p in paths:
        full = ROOT / p
        if not full.exists():
            print(f"missing: {p}"); continue
        out = out_dir / (full.stem + f"{suffix}.dup.blob")
        dump_blob(full, out, max_verts=args.max_verts, max_tris=args.max_tris)
    print(f"\nBlobs in: {out_dir}")


if __name__ == "__main__":
    main()

"""Encode mesh with DGFTester, repack to .dgfblob for CUDA bench.

Pipeline:
  .obj → DGFTester.exe --dump-bin → .bin (raw 128B blocks)
       → walk blocks, parse headers, compute vert/tri prefix sums
       → write .dgfblob (header + blocks + offsets)

Blob format (LE):
  uint32 magic = 0x42464744  ('DGFB' as C-string)
  uint32 version = 1
  uint32 n_blocks
  uint32 n_v_total
  uint32 n_t_total
  uint32 _resv[3]               # 32-byte aligned header
  uint8  blocks[n_blocks * 128]
  uint32 vert_offsets[n_blocks]
  uint32 tri_offsets[n_blocks]

Usage:
  python scripts/dump_dgf_blob.py assets/stanford-bunny.obj [out.dgfblob]
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DGFTESTER_EXE = (ROOT / "third_party" / "DGF-SDK" / "build" /
                 "DGFTester" / "Release" / "DGFTester.exe")
BLOB_DIR = ROOT / "bench_cpp" / "blobs" / "dgf"
BLOB_DIR.mkdir(parents=True, exist_ok=True)


def decode_block_header(block: bytes) -> tuple[int, int]:
    """Parse num_verts, num_triangles from a 128-byte DGF block.

    Header DWORD 0 layout (from DGFLib::BlockHeader):
      header_byte(8) | bits_per_index(2) | num_vertices(6) |
      num_triangles(6) | geom_id_meta(10).
    Verts/tris are stored as count-1.
    """
    d0 = struct.unpack_from("<I", block, 0)[0]
    num_verts = ((d0 >> 10) & 0x3F) + 1
    num_tris  = ((d0 >> 16) & 0x3F) + 1
    return num_verts, num_tris


def encode_to_dgfblob(mesh_path: Path, out_path: Path,
                       target_bits: int = 12) -> tuple[int, int, int, int]:
    """Run DGFTester on mesh_path → repack to out_path. Returns
    (n_blocks, n_v_total, n_t_total, bin_size_bytes)."""
    if not DGFTESTER_EXE.exists():
        raise RuntimeError(f"DGFTester not built: {DGFTESTER_EXE}")

    with tempfile.TemporaryDirectory() as td:
        bin_out = Path(td) / "out.bin"
        r = subprocess.run(
            [str(DGFTESTER_EXE), str(mesh_path),
             "--target-bits", str(target_bits),
             "--dump-bin", str(bin_out),
             "--skip-validation"],
            capture_output=True, text=True)
        if r.returncode != 0 or not bin_out.exists():
            print(r.stdout)
            print(r.stderr, file=sys.stderr)
            raise RuntimeError(
                f"DGFTester failed for {mesh_path} (rc={r.returncode})")
        bin_bytes = bin_out.read_bytes()

    if len(bin_bytes) % 128 != 0:
        raise RuntimeError(
            f"bin size {len(bin_bytes)} not a multiple of 128")
    n_blocks = len(bin_bytes) // 128

    # Walk headers → prefix sums.
    vert_offsets = []
    tri_offsets  = []
    v_cur = 0
    t_cur = 0
    for i in range(n_blocks):
        nv, nt = decode_block_header(bin_bytes[i * 128:(i + 1) * 128])
        vert_offsets.append(v_cur)
        tri_offsets.append(t_cur)
        v_cur += nv
        t_cur += nt
    n_v_total = v_cur
    n_t_total = t_cur

    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", 0x42464744, 1))                 # magic, version
        f.write(struct.pack("<III", n_blocks, n_v_total, n_t_total))
        f.write(struct.pack("<III", 0, 0, 0))                       # _resv
        f.write(bin_bytes)
        f.write(struct.pack(f"<{n_blocks}I", *vert_offsets))
        f.write(struct.pack(f"<{n_blocks}I", *tri_offsets))

    return n_blocks, n_v_total, n_t_total, len(bin_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--target-bits", type=int, default=12)
    args = ap.parse_args()
    mesh = Path(args.mesh)
    out = Path(args.out) if args.out else (BLOB_DIR / (mesh.stem + ".dgfblob"))

    n_blk, n_v, n_t, bin_sz = encode_to_dgfblob(mesh, out, args.target_bits)
    bpv = bin_sz * 8 / max(1, n_v)
    print(f"{mesh.name}: n_blocks={n_blk} n_v={n_v} n_t={n_t} "
          f"bin={bin_sz} B  BPV={bpv:.2f}  -> {out}")


if __name__ == "__main__":
    main()

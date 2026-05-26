"""Compute STRIDE-dup BPV including all side tables (honest reporting)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import struct

BLOB_DIR = ROOT / "bench_cpp" / "blobs" / "dup"

MESHES = [
    ("fandisk", 6475),
    ("stanford-bunny", 35947),
    ("horse", 48485),
    ("Monkey", 504482),
    ("happy_buddha", 543522),
    ("crab", 1079516),
    ("tank", 1790492),
    ("xyzrgb_dragon", 3609600),
]

# Per-meshlet side-table sizes (v3 blob format):
#   ml_off_bits         u64 = 8 B
#   ml_n_local          u32 = 4 B
#   ml_n_tris           u32 = 4 B
#   ml_n_strips         u32 = 4 B
#   ml_v_off            u32 = 4 B
#   ml_t_off            u32 = 4 B
#   ml_resid_off_bits   u64 = 8 B
#   ml_n_kind0          u32 = 4 B
#   ml_axis_sub_offs    5xu16 = 10 B
SIDE_BYTES_PER_MESHLET = 8+4+4+4+4+4+8+4+10  # = 50 B

# Fixed header bytes (DupBlobHeader, 88 B).
HEADER_BYTES = 88

print(f"mesh,n_v_src,n_meshlets,body_bytes,side_bytes,total_bytes,bpv_body,bpv_total")
for name, n_v_src in MESHES:
    blob = BLOB_DIR / f"{name}.dup.blob"
    if not blob.exists():
        print(f"# missing {blob}")
        continue
    with open(blob, "rb") as f:
        magic, version, n_meshlets, n_v_total, n_t_total, buf_size = struct.unpack(
            "<IIIIII", f.read(24))
    body_bytes = buf_size
    side_bytes = SIDE_BYTES_PER_MESHLET * n_meshlets
    total = HEADER_BYTES + body_bytes + side_bytes
    bpv_body  = 8 * (HEADER_BYTES + body_bytes) / n_v_src
    bpv_total = 8 * total / n_v_src
    print(f"{name},{n_v_src},{n_meshlets},{body_bytes},{side_bytes},{total},{bpv_body:.2f},{bpv_total:.2f}")

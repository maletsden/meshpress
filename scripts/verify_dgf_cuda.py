"""Parity check: CUDA DGF decoder vs CPU reference (Python port of DGFLib).

Encodes a mesh with DGFTester, runs CUDA kernel with --dump, runs pure-
Python port of DGFLib::DecodeTriangleList + DecodeOffsetVerts +
ConvertOffsetsToFloat on the SAME blob, and compares element-wise.

Pass criterion (faithful port):
  * Triangle indices: exact match per-tri (orientation preserved).
  * Vertex positions: bit-exact float32 match per coord.

Usage:
  python scripts/verify_dgf_cuda.py assets/bunny.obj
"""
from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dump_dgf_blob import encode_to_dgfblob, BLOB_DIR

DGF_BENCH = ROOT / "bench_cpp" / "dgf_decode_bench.exe"


# ---------- Pure-Python port of DGFLib bit reader + decoders ----------

def read_bits(buf: bytes, start: int, length: int) -> int:
    """Direct port of DGFLib::ReadBits — little-endian within byte."""
    first = start // 8
    last  = (start + length - 1) // 8
    n     = 1 + last - first
    dst = 0
    for i in range(n):
        dst |= buf[first + i] << (8 * i)
    return (dst >> (start & 7)) & ((1 << length) - 1)


def sign_extend_24(v: int) -> int:
    return v - (1 << 24) if (v & 0x800000) else v


def decode_meta(block: bytes) -> dict:
    d0, d1, d2, d3, d4 = struct.unpack_from("<5I", block, 0)
    m = {}
    m["bits_per_index_raw"] = (d0 >> 8) & 0x3
    m["num_verts_m"]   = (d0 >> 10) & 0x3F
    m["num_tris_m"]    = (d0 >> 16) & 0x3F
    m["geom_id_meta"]  = (d0 >> 22) & 0x3FF
    m["exponent"]      = d1 & 0xFF
    m["anchorX"]       = sign_extend_24((d1 >> 8) & 0xFFFFFF)
    m["xBits_m"]       = d2 & 0xF
    m["yBits_m"]       = (d2 >> 4) & 0xF
    m["anchorY"]       = sign_extend_24((d2 >> 8) & 0xFFFFFF)
    m["zBits_m"]       = d3 & 0xF
    m["omm_count"]     = (d3 >> 4) & 0x7
    m["geom_id_mode"]  = (d3 >> 7) & 0x1
    m["anchorZ"]       = sign_extend_24((d3 >> 8) & 0xFFFFFF)
    m["have_userdata"] = (d4 >> 29) & 0x1
    m["bitsPerIndex"]  = m["bits_per_index_raw"] + 3
    m["numVerts"]      = m["num_verts_m"] + 1
    m["numTris"]       = m["num_tris_m"] + 1
    m["xBits"]         = m["xBits_m"] + 1
    m["yBits"]         = m["yBits_m"] + 1
    m["zBits"]         = m["zBits_m"] + 1
    return m


def front_buffer_bits(m: dict) -> int:
    """Sum of vertex data + OMM palette + geomID palette, all 8-byte aligned."""
    bpv = m["xBits"] + m["yBits"] + m["zBits"]
    vbits = bpv * m["numVerts"]
    vbits = (vbits + 7) & ~7

    omm = 0
    if m["omm_count"]:
        hot = 2 + m["omm_count"]
        n = m["omm_count"]
        idx_sz = 0 if n <= 1 else (1 if n == 2 else (2 if n <= 4 else 3))
        omm = 32 * hot + (((idx_sz * m["numTris"]) + 7) & ~7)

    gp = 0
    if m["geom_id_mode"]:
        numIDs = (m["geom_id_meta"] >> 5) + 1
        prefix = m["geom_id_meta"] & 0x1f
        payload = 25 - prefix
        bn = max(1, (numIDs - 1).bit_length())
        gp = numIDs * payload + m["numTris"] * bn + prefix
        gp = (gp + 7) & ~7

    return vbits + omm + gp


def decode_topology(block: bytes, m: dict) -> tuple[list[int], list[int]]:
    """Returns (control[], indices[]). Mirrors DecodeTopology."""
    numTris = m["numTris"]
    ctrl = [0] * numTris  # first is RESTART (0)
    numStored = 0
    for i in range(1, numTris):
        c = read_bits(block, 1024 - 2 * i, 2)
        ctrl[i] = c
        numStored += 3 if c == 0 else 1

    is_first_pos = 1024 - 2 * (numTris - 1) - 1
    user_bits = 32 if m["have_userdata"] else 0
    index_bit_pos = 8 * 20 + user_bits + front_buffer_bits(m)

    indices = [0, 1, 2]
    vc = 3
    bpi = m["bitsPerIndex"]
    for i in range(numStored):
        is_first = read_bits(block, is_first_pos - i, 1)
        if is_first:
            val = vc
            vc += 1
        else:
            val = read_bits(block, index_bit_pos, bpi)
            index_bit_pos += bpi
        indices.append(val)
    return ctrl, indices


def convert_to_tri_list(ctrl: list[int], indices: list[int],
                         numTris: int) -> list[tuple[int, int, int]]:
    out = []
    pos = 0
    prev = [0, 0, 0]
    prevPrev = [0, 0, 0]
    for i in range(numTris):
        c = ctrl[i]
        v = [0, 0, 0]
        if c == 0:    # RESTART
            v = [pos, pos + 1, pos + 2]; pos += 3
        elif c == 1:  # EDGE1
            v = [prev[2], prev[1], pos]; pos += 1
        elif c == 2:  # EDGE2
            v = [prev[0], prev[2], pos]; pos += 1
        else:         # BACKTRACK
            if ctrl[i - 1] == 1:
                v = [prevPrev[0], prevPrev[2], pos]
            else:
                v = [prevPrev[2], prevPrev[1], pos]
            pos += 1
        out.append((indices[v[0]], indices[v[1]], indices[v[2]]))
        prevPrev = list(prev); prev = list(v)
    return out


def decode_offset_verts(block: bytes, m: dict) -> list[tuple[int, int, int]]:
    user_bits = 32 if m["have_userdata"] else 0
    vd_start_byte = 20 + user_bits // 8
    xb, yb, zb = m["xBits"], m["yBits"], m["zBits"]
    bpv = xb + yb + zb
    out = []
    for i in range(m["numVerts"]):
        v = read_bits(block, vd_start_byte * 8 + i * bpv, bpv)
        x = v & ((1 << xb) - 1)
        y = (v >> xb) & ((1 << yb) - 1)
        z = (v >> (xb + yb)) & ((1 << zb) - 1)
        out.append((x, y, z))
    return out


def cpu_decode_block(block: bytes) -> tuple[np.ndarray, np.ndarray]:
    m = decode_meta(block)
    ctrl, indices = decode_topology(block, m)
    tris = convert_to_tri_list(ctrl, indices, m["numTris"])
    offsets = decode_offset_verts(block, m)
    scale = math.ldexp(1.0, m["exponent"] - 127)
    ax, ay, az = m["anchorX"], m["anchorY"], m["anchorZ"]
    verts_f = np.array(
        [(float(ox + ax) * scale, float(oy + ay) * scale, float(oz + az) * scale)
         for ox, oy, oz in offsets],
        dtype=np.float32)
    tris_a = np.array(tris, dtype=np.uint32)
    return verts_f, tris_a


# ---------- Driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    args = ap.parse_args()
    mesh = Path(args.mesh)

    blob = BLOB_DIR / (mesh.stem + ".dgfblob")
    n_blk, n_v, n_t, _ = encode_to_dgfblob(mesh, blob, 12)
    print(f"encoded {mesh.name}: n_blocks={n_blk} n_v={n_v} n_t={n_t}")

    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "gpu"
        r = subprocess.run(
            [str(DGF_BENCH), str(blob), "1", "1", "--dump", str(prefix)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout); print(r.stderr, file=sys.stderr)
            sys.exit(1)
        gpu_verts = np.fromfile(str(prefix) + ".verts.f32",
                                  dtype=np.float32).reshape(-1, 3)
        gpu_tris  = np.fromfile(str(prefix) + ".tris.u32",
                                  dtype=np.uint32).reshape(-1, 3)

    # CPU-decode the same blob, block by block.
    with open(blob, "rb") as f:
        hdr = f.read(32)
        n_blk = struct.unpack_from("<I", hdr, 8)[0]
        n_v_total = struct.unpack_from("<I", hdr, 12)[0]
        n_t_total = struct.unpack_from("<I", hdr, 16)[0]
        all_blocks = f.read(n_blk * 128)
        voff = np.frombuffer(f.read(n_blk * 4), dtype=np.uint32)
        toff = np.frombuffer(f.read(n_blk * 4), dtype=np.uint32)

    cpu_verts = np.zeros((n_v_total, 3), dtype=np.float32)
    cpu_tris  = np.zeros((n_t_total, 3), dtype=np.uint32)
    for bi in range(n_blk):
        block = all_blocks[bi * 128:(bi + 1) * 128]
        v, t = cpu_decode_block(block)
        cpu_verts[voff[bi]:voff[bi] + v.shape[0]] = v
        cpu_tris [toff[bi]:toff[bi] + t.shape[0]] = t + voff[bi]

    # Compare.
    print(f"  GPU verts: {gpu_verts.shape}  CPU verts: {cpu_verts.shape}")
    print(f"  GPU tris : {gpu_tris.shape}   CPU tris : {cpu_tris.shape}")
    v_eq = np.array_equal(gpu_verts.view(np.uint32), cpu_verts.view(np.uint32))
    t_eq = np.array_equal(gpu_tris, cpu_tris)
    print(f"  vert bit-exact: {v_eq}")
    print(f"  tri  exact:     {t_eq}")

    if not (v_eq and t_eq):
        # Diagnostic.
        if not t_eq:
            diff_tri = np.where((gpu_tris != cpu_tris).any(axis=1))[0]
            print(f"    tri mismatches: {len(diff_tri)} / {n_t_total}")
            if len(diff_tri) > 0:
                k = diff_tri[0]
                print(f"    first: tri {k} GPU={gpu_tris[k]} CPU={cpu_tris[k]}")
        if not v_eq:
            diff_v = np.where(
                (gpu_verts.view(np.uint32) != cpu_verts.view(np.uint32)).any(axis=1))[0]
            print(f"    vert mismatches: {len(diff_v)} / {n_v_total}")
            if len(diff_v) > 0:
                k = diff_v[0]
                print(f"    first: vert {k} GPU={gpu_verts[k]} CPU={cpu_verts[k]}")
        sys.exit(2)
    print("PARITY: PASS")


if __name__ == "__main__":
    main()

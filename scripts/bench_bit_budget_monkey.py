"""Per-section bit-budget breakdown of the v5 bitstream on Monkey.

Reports: global header / boundary table / per-meshlet (header+refs,
connectivity, residual x/y/z) / offset table.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.bit_codec import BitWriter
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    REUSE_BUF_SIZE, _emit_vert, _idx_bits_for, _zigzag,
    _best_rice_k, _best_eg_k, _root_orient,
)
from encoder.paradelta_v5 import (
    encode_from_prepared_v5, _interior_pass_strip, _fit_linear3,
)
from collections import deque


def write_meshlet_track(w, plan, codes_traversal, totals):
    n_bnd = plan["n_bnd"]; n_int = plan["n_int"]
    n_tris_m = plan["n_tris_m"]; n_strips = plan["n_strips"]
    ml_tris_local = plan["ml_tris_local"]
    refs = plan["refs"]; strips = plan["strips"]
    n_local = n_bnd + n_int

    p0 = w.bit_pos()
    w.write_fixed(n_bnd, 16)
    w.write_fixed(n_int, 16)
    w.write_fixed(n_tris_m, 16)
    w.write_fixed(n_strips, 16)
    if n_bnd > 0:
        w.write_fixed(int(refs[0]), 32)
        if n_bnd > 1:
            diffs = refs[1:] - refs[:-1] - 1
            u = diffs.astype(np.int64)
            k, _ = _best_rice_k(u)
            w.write_fixed(k, 8)
            for x in u:
                w.write_rice(int(x), k)
    p1 = w.bit_pos()
    totals["hdr_refs"] += (p1 - p0)

    idx_bits = _idx_bits_for(n_local)
    reuse_fifo: deque[int] = deque(maxlen=REUSE_BUF_SIZE)
    for strip in strips:
        strip_len = len(strip)
        w.write_fixed(strip_len, 16)
        root_id = strip[0]
        next_id = strip[1] if len(strip) > 1 else None
        root = _root_orient(ml_tris_local, root_id, next_id)
        for v in root:
            _emit_vert(w, v, reuse_fifo, idx_bits)
        prev_tri = list(root)
        for li in strip[1:]:
            tri_v = [int(x) for x in ml_tris_local[li]]
            tri_set = set(tri_v); prev_set = set(prev_tri)
            shared = tri_set & prev_set
            new_v = next(iter(tri_set - shared))
            pair_newest = frozenset((prev_tri[1], prev_tri[2]))
            pair_second = frozenset((prev_tri[0], prev_tri[2]))
            shared_fs = frozenset(shared)
            if shared_fs == pair_newest:
                edge_code = 0
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            elif shared_fs == pair_second:
                edge_code = 1
                new_prev = [prev_tri[0], prev_tri[2], new_v]
            else:
                raise RuntimeError("oldest-edge share")
            w.write_bits(edge_code, 1)
            _emit_vert(w, new_v, reuse_fifo, idx_bits)
            prev_tri = new_prev
    p2 = w.bit_pos()
    totals["conn"] += (p2 - p1)

    if n_int == 0:
        return
    axis_labels = ("res_x", "res_y", "res_z")
    for d in range(3):
        p_axis_start = w.bit_pos()
        arr = codes_traversal[:, d]
        u_arr = _zigzag(arr)
        mn = int(arr.min()); rng = int(arr.max() - mn)
        fixed_bw = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
        fixed_total = 8 + 16 + 8 + n_int * fixed_bw
        if mn < -32768 or mn > 32767:
            fixed_total = float("inf")
        rice_k, rice_body = _best_rice_k(u_arr)
        rice_total = 8 + 8 + rice_body
        eg_k, eg_body = _best_eg_k(u_arr)
        eg_total = 8 + 8 + eg_body
        cands = [(fixed_total, 0), (rice_total, 1), (eg_total, 2)]
        _, tag = min(cands, key=lambda t: t[0])
        if tag == 0:
            w.write_fixed(0, 8); w.write_fixed(mn & 0xFFFF, 16)
            w.write_fixed(fixed_bw, 8)
            w.write_fixed_array((arr - mn).astype(np.int64), fixed_bw)
        elif tag == 1:
            w.write_fixed(1, 8); w.write_fixed(rice_k, 8)
            w.write_rice_array(u_arr.astype(np.int64), rice_k)
        else:
            w.write_fixed(2, 8); w.write_fixed(eg_k, 8)
            w.write_exp_golomb_array(u_arr.astype(np.int64), eg_k)
        totals[axis_labels[d]] += (w.bit_pos() - p_axis_start)


def run(path: str):
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    bnd_recon_norm = prep["bnd_recon_norm"]; vn = prep["vn"]
    plans = prep["plans"]
    delta = 2.0 * prep["per_coord_err"]

    all_samples = []
    for plan in plans:
        _, s = _interior_pass_strip(plan, vn, bnd_recon_norm, delta, w3=None)
        all_samples.extend(s)
    lin3_w = _fit_linear3(all_samples).astype(np.float32).astype(np.float64)

    w_ml = BitWriter()
    totals = {"hdr_refs": 0, "conn": 0, "res_x": 0, "res_y": 0, "res_z": 0}
    for plan in plans:
        codes, _ = _interior_pass_strip(
            plan, vn, bnd_recon_norm, delta, lin3_w)
        write_meshlet_track(w_ml, plan, codes, totals)
    meshlet_bits = w_ml.bit_pos()

    full = encode_from_prepared_v5(prep, verbose=False)
    full_bytes = len(full)

    n_v = prep["n_v"]; n_t = prep["n_t"]
    n_boundary = prep["n_boundary"]; n_meshlets = prep["n_meshlets"]
    global_codes = prep["global_codes"]
    g_bits = prep["g_bits"]

    w_hdr = BitWriter()
    p_hdr_start = w_hdr.bit_pos()
    w_hdr.write_fixed(0, 32)
    w_hdr.write_fixed(0, 8); w_hdr.write_fixed(0, 8)
    for _ in range(3): w_hdr.write_f32(0.0)
    w_hdr.write_f32(0.0); w_hdr.write_f32(0.0)
    for _ in range(3): w_hdr.write_f32(0.0)
    for _ in range(3): w_hdr.write_f32(0.0)
    for _ in range(3): w_hdr.write_fixed(0, 8)
    w_hdr.write_fixed(n_v, 32); w_hdr.write_fixed(n_t, 32)
    w_hdr.write_fixed(n_boundary, 32); w_hdr.write_fixed(n_meshlets, 32)
    for _ in range(3): w_hdr.write_f32(1.0)
    p_hdr_end = w_hdr.bit_pos()
    hdr_bits = p_hdr_end - p_hdr_start

    bnd_bits = 0
    if n_boundary > 0:
        bnd_codes = global_codes[prep["boundary_list"]]
        wb = BitWriter()
        for d in range(3):
            arr = bnd_codes[:, d].astype(np.int64)
            wb.write_fixed(int(arr[0]), int(g_bits[d]))
            if n_boundary > 1:
                diffs = arr[1:] - arr[:-1]
                u = _zigzag(diffs)
                k, _ = _best_rice_k(u)
                wb.write_fixed(k, 8)
                for x in u:
                    wb.write_rice(int(x), k)
        bnd_bits = wb.bit_pos()

    offset_table_bits = n_meshlets * 32
    accounted_bits = hdr_bits + bnd_bits + offset_table_bits + meshlet_bits
    accounted_bytes = accounted_bits // 8
    pad_bytes = full_bytes - accounted_bytes

    name = Path(path).name
    print(f"\n[{name}] n_v={n_v:,}  n_t={n_t:,}  meshlets={n_meshlets:,}")
    print(f"  total bytes (full encoder) = {full_bytes:,}  "
          f"BPV={full_bytes*8/n_v:.2f}")

    sections = [
        ("Global header (magic, AABB, fitted weights, counts)", hdr_bits),
        ("Boundary table (Morton-ordered, delta-Rice across all axes)", bnd_bits),
        ("Offset table (32-bit relative offsets per meshlet)", offset_table_bits),
        ("Per-meshlet headers + boundary refs", totals["hdr_refs"]),
        ("Connectivity (GTS v3 strip descriptors)", totals["conn"]),
        ("Interior residuals — axis x", totals["res_x"]),
        ("Interior residuals — axis y", totals["res_y"]),
        ("Interior residuals — axis z", totals["res_z"]),
    ]
    total_acc_bits = sum(b for _, b in sections)
    print()
    print(f"  {'Section':<60} {'Bytes':>10}  {'BPV':>7}  {'Share':>6}")
    for label, bits in sections:
        bytes_ = bits / 8.0
        bpv = bits / n_v
        share = bits / total_acc_bits * 100
        print(f"  {label:<60} {bytes_:>10,.0f}  {bpv:>7.3f}  {share:>5.2f}%")
    print(f"  {'(unaccounted padding to byte boundary)':<60} {pad_bytes:>10,d}")
    print(f"  {'Total':<60} {full_bytes:>10,d}  "
          f"{full_bytes*8/n_v:>7.2f}  100.00%")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        run(p)
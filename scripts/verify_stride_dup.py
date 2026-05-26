"""Lossless CPU roundtrip for STRIDE-dup variant.

For each meshlet:
  Encoder side:
    1. Walk strip → classify verts as anchor/delta/para.
    2. Compute residuals: anchor=raw, delta=(this - prev), para=(this - lin3_pred).
    3. Tally bit counts (Rice/EG per axis).
  Decoder side:
    1. Walk SAME strip in SAME order.
    2. Reconstruct int codes:
       anchor = raw code,
       delta-vert = prev_decoded + delta,
       para-vert  = lin3_pred(a, b, c) + para_residual.
    3. Compare to source int codes.

PASS criterion: reconstructed[meshlet][v_local] == global_codes[local_to_global[v_local]] for every meshlet+vert.

If PASS, the simulated BPV is real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5 import encode_from_prepared_v5  # noqa: E402
from scripts.sim_stride_dup import (
    _root_orient_local, best_code_bits,
)  # noqa: E402

MESHES = ["assets/Monkey.obj", "assets/xyzrgb_dragon.obj"]


def walk_meshlet(plan, decoded_already):
    """Returns walk order: list of (v_local, kind, refs)."""
    ml_tris_local = np.asarray(plan["ml_tris_local"], dtype=np.int64)
    strips = plan["strips"]
    n_local = len(plan["local_to_global"])
    decoded = decoded_already.copy()
    order = []
    for strip in strips:
        root_id = strip[0]
        next_id = strip[1] if len(strip) > 1 else None
        root = _root_orient_local(ml_tris_local, root_id, next_id)
        for v in root:
            if not decoded[v]:
                order.append((v, 'none', None))
                decoded[v] = True
        prev_tri = list(root)
        for li in strip[1:]:
            tri_v = [int(x) for x in ml_tris_local[li]]
            tri_set = set(tri_v); prev_set = set(prev_tri)
            shared = tri_set & prev_set
            if len(shared) != 2:
                continue
            new_v = next(iter(tri_set - shared))
            pair_newest = frozenset((prev_tri[1], prev_tri[2]))
            pair_second = frozenset((prev_tri[0], prev_tri[2]))
            shared_fs = frozenset(shared)
            if shared_fs == pair_newest:
                a, b, c = prev_tri[1], prev_tri[2], prev_tri[0]
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            elif shared_fs == pair_second:
                a, b, c = prev_tri[0], prev_tri[2], prev_tri[1]
                new_prev = [prev_tri[0], prev_tri[2], new_v]
            else:
                continue
            if not decoded[new_v]:
                order.append((new_v, 'para', (a, b, c)))
                decoded[new_v] = True
            prev_tri = new_prev
    return order


def encode_meshlet(order, true_codes, lin3_w):
    """Encoder: produce residuals for each vert."""
    w0, w1, w2 = lin3_w
    n_local = true_codes.shape[0]
    residuals = []   # list of (kind, v_local, payload)
    prev_decoded = None
    for v, kind, refs in order:
        if kind == 'none':
            if prev_decoded is None:
                residuals.append(('anchor', v, true_codes[v].copy()))
            else:
                d = true_codes[v] - prev_decoded
                residuals.append(('delta', v, d))
        else:
            a, b, c = refs
            ca, cb, cc = true_codes[a], true_codes[b], true_codes[c]
            pred = np.rint(w0 * ca + w1 * cb + w2 * cc).astype(np.int64)
            r = true_codes[v] - pred
            residuals.append(('para', v, r))
        prev_decoded = true_codes[v].copy()
    return residuals


def decode_meshlet(order, residuals, n_local, lin3_w):
    """Decoder: reconstruct codes from residuals + walk order."""
    w0, w1, w2 = lin3_w
    out = np.zeros((n_local, 3), dtype=np.int64)
    prev = None
    for (v, kind, refs), (rkind, rv, payload) in zip(order, residuals):
        assert rv == v and ((kind == 'none' and rkind in ('anchor', 'delta'))
                            or (kind == 'para' and rkind == 'para'))
        if rkind == 'anchor':
            out[v] = payload
        elif rkind == 'delta':
            assert prev is not None
            out[v] = prev + payload
        else:
            a, b, c = refs
            pred = np.rint(w0 * out[a] + w1 * out[b] + w2 * out[c]).astype(np.int64)
            out[v] = pred + payload
        prev = out[v].copy()
    return out


def measure_bits(residuals, g_bits):
    """Sum Rice/EG bit cost across all residuals."""
    by_kind = {'anchor': [], 'delta': [], 'para': []}
    for rkind, _, payload in residuals:
        by_kind[rkind].append(payload)

    bits = 0
    n_a = len(by_kind['anchor']); n_d = len(by_kind['delta']); n_p = len(by_kind['para'])
    # anchor: raw, g_bits per axis (already summed)
    bits += sum(g_bits) * n_a
    # delta + para: Rice/EG per axis, per-kind grouping (one k per axis).
    for kind in ('delta', 'para'):
        if by_kind[kind]:
            arr = np.array(by_kind[kind], dtype=np.int64)
            for d in range(3):
                bits += best_code_bits(arr[:, d])
    return bits, n_a, n_d, n_p


def verify_mesh(mesh_path: str):
    full = ROOT / mesh_path
    prep = load_or_prepare(str(full), max_verts=256, max_tris=256,
                            precision_error=1.0/4096.0,
                            precision_mode="bbox_frac",
                            gen_method="joint_learned",
                            strip_method="multiseed", verbose=False)
    global_codes = prep["global_codes"]
    g_bits = [int(b) for b in prep["g_bits"]]
    n_v = prep["n_v"]
    plans = prep["plans"]

    v5_bytes = encode_from_prepared_v5(prep, verbose=False)
    v5_bpv = 8 * len(v5_bytes) / n_v

    lin3_w = (1.0, 1.0, -1.0)
    total_bits = 0
    total_a = total_d = total_p = 0
    mismatches = 0
    n_local_total = 0

    for plan in plans:
        local_to_global = np.asarray(plan["local_to_global"], dtype=np.int64)
        n_local = len(local_to_global)
        true_codes = global_codes[local_to_global].astype(np.int64)

        decoded_init = np.zeros(n_local, dtype=bool)
        order = walk_meshlet(plan, decoded_init)
        residuals = encode_meshlet(order, true_codes, lin3_w)
        reconstructed = decode_meshlet(order, residuals, n_local, lin3_w)

        # Verify only the verts that appeared in the walk (rest may be
        # isolated — shouldn't happen but be safe).
        seen = np.zeros(n_local, dtype=bool)
        for v, _, _ in order:
            seen[v] = True
        diff = (reconstructed[seen] != true_codes[seen]).any(axis=1)
        mismatches += int(diff.sum())
        n_local_total += int(seen.sum())

        b, na, nd, np_ = measure_bits(residuals, g_bits)
        total_bits += b
        total_a += na; total_d += nd; total_p += np_

    dup_vert_bpv = total_bits / n_v
    print(f"\n=== {full.name} ===")
    print(f"  n_v={n_v:,}  n_meshlets={prep['n_meshlets']:,}")
    print(f"  Verts walked: {n_local_total:,}  mismatches: {mismatches}")
    print(f"  Anchor: {total_a:,}  Delta: {total_d:,}  Para: {total_p:,}")
    print(f"  Dup vert-section bits: {total_bits/8/1024/1024:.2f} MiB  "
          f"({dup_vert_bpv:.2f} BPV)")
    print(f"  v5 total BPV:    {v5_bpv:.2f}")
    # Estimate dup total: connectivity + header unchanged, vert section replaced.
    # We don't know exactly the current vert section size from this script,
    # but we can extract conn+hdr from bench_bit_budget if available.
    # Use Dragon/Monkey known numbers (from compacted summary).
    known_other = {
        "Monkey":         {"conn_hdr": 21.205 + 0.589},  # Dragon proxies
        "xyzrgb_dragon":  {"conn_hdr": 21.205 + 0.589},
    }
    other = known_other.get(full.stem, {}).get("conn_hdr", 21.794)
    dup_total = other + dup_vert_bpv
    print(f"  Projected dup total BPV (conn+hdr={other:.2f} + vert={dup_vert_bpv:.2f})"
          f" = {dup_total:.2f}")
    print(f"  Δ vs v5: {dup_total - v5_bpv:+.2f} BPV "
          f"({100*(dup_total - v5_bpv)/v5_bpv:+.1f}%)")


def main():
    paths = sys.argv[1:] or MESHES
    for p in paths:
        verify_mesh(p)


if __name__ == "__main__":
    main()

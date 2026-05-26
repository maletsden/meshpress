"""Simulate STRIDE-dup BPV cost without full encoder rewrite.

For each meshlet, re-walk the strip with all verts treated as "interior":
* First 3 verts of each strip's root tri → 'none' (raw int codes).
* All other verts → 'para' (parallelogram chain via prev_tri's lin3 predict).

Then Rice/EG-code all residuals + raw 'none' codes, sum bits. Compare
to current v5 BPV (which includes a global boundary table + per-meshlet
refs + interior residuals).

Output: bench_stride_dup_sim.csv with per-mesh projected BPV.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5 import encode_from_prepared_v5  # noqa: E402
from encoder.paradelta_codec import _fit_linear3, _best_rice_k  # noqa: E402

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


def _zigzag(arr: np.ndarray) -> np.ndarray:
    return ((arr << 1) ^ (arr >> 63)).astype(np.uint64)


def rice_bits(u_arr: np.ndarray, k: int) -> int:
    """Bit cost of Rice-coding nonneg u_arr at param k."""
    q = (u_arr >> k).astype(np.int64)
    bits = int((q + 1 + k).sum())
    return bits


def expgolomb_bits(u_arr: np.ndarray, k: int) -> int:
    """Bit cost of Exp-Golomb coding."""
    q_plus_1 = (u_arr.astype(np.int64) >> k) + 1
    leading = np.floor(np.log2(q_plus_1.astype(np.float64))).astype(np.int64) + 1
    bits = int((2 * leading - 1 + k).sum())
    return bits


def best_code_bits(values_int: np.ndarray) -> int:
    """Pick min(rice, EG) per axis-vector."""
    if values_int.size == 0:
        return 0
    u = _zigzag(values_int.astype(np.int64))
    best = 10**18
    for k in range(0, 16):
        b_rice = rice_bits(u, k) + 8  # 8b for k itself
        if b_rice < best:
            best = b_rice
    for k in range(0, 16):
        b_eg = expgolomb_bits(u, k) + 8
        if b_eg < best:
            best = b_eg
    return best


def _root_orient_local(ml_tris_local, root_id, next_id):
    """Orient root tri to share its (1,2) edge with next tri."""
    tri = list(ml_tris_local[root_id])
    if next_id is None:
        return tri
    nxt = set(ml_tris_local[next_id])
    shared = [v for v in tri if v in nxt]
    if len(shared) != 2:
        return tri
    s0, s1 = shared
    o = next(v for v in tri if v not in shared)
    return [o, s0, s1]


def simulate_meshlet(plan, global_codes, lin3_w, g_bits):
    """Returns (anchor_bits, delta_bits, para_bits, n_anchor, n_delta, n_para).
    * anchor: 1 raw vert per meshlet at g_bits per axis.
    * delta: every other 'none' vert (extra strip roots) as delta from
            the *previously decoded vert* in emit order — Rice-coded.
    * para: parallelogram residuals — Rice-coded.
    """
    ml_tris_local = np.asarray(plan["ml_tris_local"], dtype=np.int64)
    strips = plan["strips"]
    local_to_global = np.asarray(plan["local_to_global"], dtype=np.int64)
    n_local = len(local_to_global)

    # Pull int codes for this meshlet's verts (global → local).
    local_codes = global_codes[local_to_global]  # (n_local, 3) int64

    # Walk strips, classify each new vert.
    decoded = np.zeros(n_local, dtype=bool)
    order = []   # list of (v_local, kind, refs)
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
            tri_set = set(tri_v)
            prev_set = set(prev_tri)
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

    # Compute residuals + raw codes.
    w0, w1, w2 = float(lin3_w[0]), float(lin3_w[1]), float(lin3_w[2])
    delta_resid = []  # extra-root verts: delta from prev decoded
    para_resid  = []  # parallelogram residuals
    prev_decoded_code = None
    n_anchor = 0
    for v, kind, refs in order:
        true_c = local_codes[v]
        if kind == 'none':
            if prev_decoded_code is None:
                # Anchor vert — stored raw.
                n_anchor += 1
            else:
                # Delta from previously decoded vert.
                delta_resid.append(true_c - prev_decoded_code)
        else:
            a, b, c = refs
            ca, cb, cc = local_codes[a], local_codes[b], local_codes[c]
            pred = np.rint(w0 * ca + w1 * cb + w2 * cc).astype(np.int64)
            para_resid.append(true_c - pred)
        prev_decoded_code = true_c

    n_delta = len(delta_resid)
    n_para = len(para_resid)

    anchor_bits = sum(g_bits) * n_anchor  # 3 axes already in sum
    delta_bits = 0
    if n_delta > 0:
        d_arr = np.array(delta_resid, dtype=np.int64)
        for d in range(3):
            delta_bits += best_code_bits(d_arr[:, d])
    para_bits = 0
    if n_para > 0:
        p_arr = np.array(para_resid, dtype=np.int64)
        for d in range(3):
            para_bits += best_code_bits(p_arr[:, d])

    return anchor_bits, delta_bits, para_bits, n_anchor, n_delta, n_para


def simulate_mesh(mesh_path: Path) -> dict:
    full = ROOT / mesh_path if not Path(mesh_path).is_absolute() else Path(mesh_path)
    prep = load_or_prepare(str(full), max_verts=256, max_tris=256,
                            precision_error=1.0/4096.0,
                            precision_mode="bbox_frac",
                            gen_method="joint_learned",
                            strip_method="multiseed", verbose=False)
    global_codes = prep["global_codes"]
    g_bits = [int(b) for b in prep["g_bits"]]
    n_v = prep["n_v"]
    n_meshlets = prep["n_meshlets"]
    plans = prep["plans"]

    # Get current v5 bitstream size for comparison.
    v5_bytes = encode_from_prepared_v5(prep, verbose=False)
    v5_bpv = 8 * len(v5_bytes) / n_v

    # Estimate dup BPV:
    # - Per-meshlet header + connectivity: same as v5 minus the
    #   local→global refs table. Approximate: assume same connectivity
    #   share (~21 BPV from bench_bit_budget) and same header.
    # - Drop global boundary table (Dragon: 2.742 BPV).
    # - Drop per-meshlet refs (~2-3 BPV).
    # - Add: raw 'none' bits + parallelogram bits for ALL verts.

    # Lin3 weights from a quick first pass (or just use 0.5/0.5/-0 for now).
    lin3_w = np.array([1.0, 1.0, -1.0])  # classic parallelogram, good prior

    t_anchor_bits = 0; t_delta_bits = 0; t_para_bits = 0
    t_anchor = 0; t_delta = 0; t_para = 0
    for plan in plans:
        ab, db, pb, na, nd, np_ = simulate_meshlet(plan, global_codes, lin3_w, g_bits)
        t_anchor_bits += ab; t_delta_bits += db; t_para_bits += pb
        t_anchor += na; t_delta += nd; t_para += np_

    dup_vert_bpv = (t_anchor_bits + t_delta_bits + t_para_bits) / n_v

    # Current v5: vert section ≈ bnd (2.742) + refs (~2) + resid (~7.5).
    # Use bench_bit_budget for Dragon-like estimate.
    current_vert_bpv_approx = {
        "fandisk": 11.5,         # rough — high boundary fraction
        "stanford-bunny": 11.0,
        "horse": 11.0,
        "Monkey": 11.0,          # bnd 2.7 + refs 2 + resid 6.3 = 11
        "happy_buddha": 11.5,
        "crab": 13.5,
        "tank": 10.0,
        "xyzrgb_dragon": 10.9,
    }
    name = full.stem
    cur_v = current_vert_bpv_approx.get(name, 11.0)
    delta_bpv = dup_vert_bpv - cur_v
    projected_dup_bpv = v5_bpv + delta_bpv

    return {
        "mesh": name, "n_v": n_v, "n_meshlets": n_meshlets,
        "n_anchor": t_anchor, "n_delta": t_delta, "n_para": t_para,
        "anchor_bpv": t_anchor_bits / n_v,
        "delta_bpv":  t_delta_bits  / n_v,
        "para_bpv":   t_para_bits   / n_v,
        "dup_vert_bpv": dup_vert_bpv,
        "cur_vert_bpv_est": cur_v,
        "delta_vs_cur": delta_bpv,
        "v5_bpv": v5_bpv,
        "projected_dup_bpv": projected_dup_bpv,
        "pct_change": 100.0 * delta_bpv / v5_bpv,
    }


def main():
    paths = sys.argv[1:] or MESHES
    rows = []
    print(f"{'mesh':<22} {'n_m':>6} {'anc':>5} {'del':>5} {'para':>5} "
          f"{'a_bpv':>6} {'d_bpv':>6} {'p_bpv':>6} "
          f"{'dup_v':>6} {'cur_v':>6} {'v5':>6} {'dup':>6} {'pct':>6}")
    for p in paths:
        try:
            r = simulate_mesh(p)
            rows.append(r)
            print(f"{r['mesh']:<22} {r['n_meshlets']:>6} "
                  f"{r['n_anchor']:>5} {r['n_delta']:>5} {r['n_para']:>5} "
                  f"{r['anchor_bpv']:>6.2f} {r['delta_bpv']:>6.2f} "
                  f"{r['para_bpv']:>6.2f} "
                  f"{r['dup_vert_bpv']:>6.2f} {r['cur_vert_bpv_est']:>6.2f} "
                  f"{r['v5_bpv']:>6.2f} {r['projected_dup_bpv']:>6.2f} "
                  f"{r['pct_change']:>+6.1f}")
        except Exception as e:
            print(f"  {p}: ERR {e}")
            import traceback; traceback.print_exc()

    out_csv = ROOT / "bench_stride_dup_sim.csv"
    if rows:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()

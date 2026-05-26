"""Plan F ablation: strip-emit-order traversal vs greedy-order traversal.

For each meshlet, derive 2 traversals:
  baseline: _greedy_order (current encoder)
  strip:    walk strip in emit order, interior verts encoded in that order

Compute LIN5 residual codes both ways using same predictor + quantization.
Report:
  - total bytes when re-encoded (best-of fixed/Rice/EG per axis)
  - per-meshlet apex-availability degradation (% of para verts losing d_ac/d_bc)
  - net BPV delta

No bitstream change needed — only re-runs interior pass with different order.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, _root_orient, _best_rice_k, _best_eg_k,
    PREDICTOR_LIN5, _fit_linear5, _fit_linear3,
)
from utils.parallelogram_predictor import _greedy_order
from utils.residual_entropy import _zigzag
from encoder.paradelta_codec import _build_meshlet_local_topo


# -------------------------------------------------------------------
# Strip-order traversal builder (mirrors what the decoder would do).
# -------------------------------------------------------------------
def _strip_order_traversal(plan):
    """Walk strips, emit (v_local, kind, refs) for each NEW interior vert.

    refs = (a, b, c, d_ac, d_bc) with -1 sentinel when unavailable.
    kind = 'para' (always — strip guarantees 2 shared + opposite c) or
           'none' for root-tri interior verts (no prior context).
    """
    strips = plan["strips"]
    ml_tris_local = plan["ml_tris_local"]
    n_bnd = plan["n_bnd"]
    n_local = n_bnd + plan["n_int"]

    # Build edge_to_tris for 2nd-ring apex search.
    edge_to_tris, _ = _build_meshlet_local_topo(ml_tris_local)
    decoded = set(range(n_bnd))   # boundary decoded up front
    order = []

    def find_apex(u, v, exclude_tri):
        """Third vert of tri sharing edge (u,v) ≠ exclude_tri, if decoded."""
        key = (min(u, v), max(u, v))
        for t in edge_to_tris.get(key, []):
            if t == exclude_tri:
                continue
            tri = ml_tris_local[t]
            others = [int(x) for x in tri if int(x) != u and int(x) != v]
            if len(others) == 1 and others[0] in decoded:
                return others[0]
        return -1

    for strip in strips:
        root_id = strip[0]
        next_id = strip[1] if len(strip) > 1 else None
        root = _root_orient(ml_tris_local, root_id, next_id)
        # Root tri: any interior verts go in 'none' context (no preds yet).
        for v in root:
            if v >= n_bnd and v not in decoded:
                order.append((v, 'none', ()))
                decoded.add(v)
            else:
                decoded.add(v)
        prev_tri = list(root)
        prev_tri_local_id = root_id
        for li in strip[1:]:
            tri_v = [int(x) for x in ml_tris_local[li]]
            tri_set = set(tri_v)
            prev_set = set(prev_tri)
            shared = tri_set & prev_set
            new_v = next(iter(tri_set - shared))
            pair_newest = frozenset((prev_tri[1], prev_tri[2]))
            pair_second = frozenset((prev_tri[0], prev_tri[2]))
            shared_fs = frozenset(shared)
            if shared_fs == pair_newest:
                a, b = prev_tri[1], prev_tri[2]
                c = prev_tri[0]
                new_prev = [prev_tri[1], prev_tri[2], new_v]
            elif shared_fs == pair_second:
                a, b = prev_tri[0], prev_tri[2]
                c = prev_tri[1]
                new_prev = [prev_tri[0], prev_tri[2], new_v]
            else:
                raise RuntimeError("oldest-edge share in strip")
            if new_v >= n_bnd and new_v not in decoded:
                # 2nd-ring apexes across (a,c) and (b,c) of prev_tri.
                d_ac = find_apex(a, c, prev_tri_local_id)
                d_bc = find_apex(b, c, prev_tri_local_id)
                order.append((new_v, 'para', (a, b, c, d_ac, d_bc)))
                decoded.add(new_v)
            else:
                decoded.add(new_v)
            prev_tri = new_prev
            prev_tri_local_id = li
    return order


# -------------------------------------------------------------------
# Residual encoder (same best-of as paradelta_codec._write_meshlet).
# -------------------------------------------------------------------
def _bits_for_codes(codes: np.ndarray) -> int:
    """Return total bits (header + body) for a (n, 3) int code array using
    best-of (fixed / Rice / EG) per axis — matches encoder semantics."""
    n_int = codes.shape[0]
    if n_int == 0:
        return 0
    total = 0
    for d in range(3):
        arr = codes[:, d].astype(np.int64)
        u_arr = _zigzag(arr)
        mn = int(arr.min()); rng = int(arr.max() - mn)
        fixed_bw = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
        fixed_total = 8 + 16 + 8 + n_int * fixed_bw
        rice_k, rice_body = _best_rice_k(u_arr)
        rice_total = 8 + 8 + int(rice_body)
        eg_k, eg_body = _best_eg_k(u_arr)
        eg_total = 8 + 8 + int(eg_body)
        total += min(fixed_total, rice_total, eg_total)
    return total


def _predict_lin5(kind, refs, recon, fallback, w3, w5):
    if kind == 'para':
        a, b, c, d_ac, d_bc = refs
        if d_ac >= 0 and d_bc >= 0:
            return (w5[0]*recon[a] + w5[1]*recon[b] + w5[2]*recon[c]
                    + w5[3]*recon[d_ac] + w5[4]*recon[d_bc])
        return w3[0]*recon[a] + w3[1]*recon[b] + w3[2]*recon[c]
    if kind == 'mid':
        a, b = refs
        return 0.5 * (recon[a] + recon[b])
    if kind == 'one':
        return recon[refs[0]].copy()
    return fallback.copy()


def _interior_codes(plan, order, vn, bnd_recon_norm, delta, w3, w5):
    """Apply predictor walk over `order`, return (n_int, 3) int codes
    and apex stats (n_para_full, n_para_lin3, n_none)."""
    n_bnd = plan["n_bnd"]
    n_int = plan["n_int"]
    n_local = n_bnd + n_int
    if n_int == 0:
        return np.zeros((0, 3), dtype=np.int64), (0, 0, 0)
    local_to_global = plan["local_to_global"]
    recon = {}
    for lid in range(n_bnd):
        recon[lid] = bnd_recon_norm[int(local_to_global[lid])].copy()
    fallback = (np.mean([recon[i] for i in range(n_bnd)], axis=0)
                if n_bnd > 0 else np.zeros(3))
    codes = np.zeros((n_int, 3), dtype=np.int64)
    n_para_full = 0; n_para_lin3 = 0; n_none = 0
    for i, entry in enumerate(order):
        v_local, kind, refs = entry
        if kind == 'para':
            if refs[3] >= 0 and refs[4] >= 0:
                n_para_full += 1
            else:
                n_para_lin3 += 1
        elif kind == 'none':
            n_none += 1
        pred = _predict_lin5(kind, refs, recon, fallback, w3, w5)
        true = vn[int(local_to_global[v_local])].astype(np.float64)
        code = np.round((true - pred) / delta).astype(np.int64)
        rec = pred + code.astype(np.float64) * delta
        recon[v_local] = rec
        codes[i] = code
    return codes, (n_para_full, n_para_lin3, n_none)


def run(path: str) -> None:
    name = Path(path).name
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    # Get fitted LIN5 weights once via the standard encoder pipeline.
    data_baseline = encode_from_prepared(prep, predictor="linear5",
                                         verbose=False)
    n_v = prep["n_v"]
    bytes_baseline = len(data_baseline)
    print(f"\n[{name}] n_v={n_v:,} n_t={prep['n_t']:,}  "
          f"baseline={bytes_baseline:,} B  "
          f"BPV={bytes_baseline*8/n_v:.2f}")

    # Re-fit weights using a quick pass. Use the published default we
    # would actually ship: fit_linear5 over greedy-order training samples
    # (collected via the normal pipeline).
    # Simpler ablation: use identity weights ([1,1,-1,0,0] for lin5, [1,1,-1]
    # for lin3). Compare BPV impact.
    w3_ident = np.array([1.0, 1.0, -1.0], dtype=np.float64)
    w5_ident = np.array([1.0, 1.0, -1.0, 0.0, 0.0], dtype=np.float64)

    delta = 2.0 * prep["per_coord_err"]
    vn = prep["vn"]; bnd_recon = prep["bnd_recon_norm"]

    bits_greedy = 0
    bits_strip = 0
    n_para_full_g = 0; n_para_lin3_g = 0; n_none_g = 0
    n_para_full_s = 0; n_para_lin3_s = 0; n_none_s = 0
    total_int = 0
    for plan in prep["plans"]:
        n_int = plan["n_int"]
        n_bnd = plan["n_bnd"]; n_local = n_bnd + n_int
        if n_int == 0:
            continue
        total_int += n_int
        # Greedy order
        ord_g = _greedy_order(
            list(range(n_bnd, n_local)), list(range(n_bnd)),
            plan["ml_tris_local"], None, None)
        # Strip order
        ord_s = _strip_order_traversal(plan)
        # Sanity: same set of interior verts
        set_g = {e[0] for e in ord_g}
        set_s = {e[0] for e in ord_s}
        assert set_g == set_s, \
            f"vert set mismatch: |g|={len(set_g)} |s|={len(set_s)}"

        codes_g, st_g = _interior_codes(plan, ord_g, vn, bnd_recon, delta,
                                        w3_ident, w5_ident)
        codes_s, st_s = _interior_codes(plan, ord_s, vn, bnd_recon, delta,
                                        w3_ident, w5_ident)
        bits_greedy += _bits_for_codes(codes_g)
        bits_strip  += _bits_for_codes(codes_s)
        n_para_full_g += st_g[0]; n_para_lin3_g += st_g[1]; n_none_g += st_g[2]
        n_para_full_s += st_s[0]; n_para_lin3_s += st_s[1]; n_none_s += st_s[2]

    delta_bytes = (bits_strip - bits_greedy) / 8.0
    delta_bpv = delta_bytes * 8 / n_v
    print(f"  interior bits: greedy={bits_greedy:>10,}  "
          f"strip={bits_strip:>10,}  Δ={bits_strip-bits_greedy:+,} bits  "
          f"({delta_bpv:+.2f} BPV)")
    print(f"  greedy apex: full_para={n_para_full_g:>7,} "
          f"({n_para_full_g/total_int*100:5.1f}%)  "
          f"lin3_fallback={n_para_lin3_g:>7,} "
          f"({n_para_lin3_g/total_int*100:5.1f}%)  "
          f"none={n_none_g:>5,}")
    print(f"  strip  apex: full_para={n_para_full_s:>7,} "
          f"({n_para_full_s/total_int*100:5.1f}%)  "
          f"lin3_fallback={n_para_lin3_s:>7,} "
          f"({n_para_lin3_s/total_int*100:5.1f}%)  "
          f"none={n_none_s:>5,}  "
          f"(non-para+none = lost predictor power)")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/stanford-bunny.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        run(p)

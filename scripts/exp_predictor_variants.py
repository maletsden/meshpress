"""Predictor variant experiment.

For each meshlet, evaluate residual cost under several predictors:
  P0  parallelogram with global lin3 fit  (current v5)
  P1  Laplacian       pred = (a + b + c) / 3
  P2  edge midpoint   pred = (a + b) / 2
  P3  copy-a          pred = a
  P4  centroid only   pred = boundary mean (the 'none' fallback)

For each predictor, compute zigzagged Rice cost (matches what the
encoder picks). Then for each meshlet, pick the best-of-N → 'adaptive'
total. Add per-meshlet 3-bit selector overhead.

Reports per-mesh:
  - Per-predictor total bits
  - Adaptive total bits
  - Selector overhead
  - BPV impact

Usage:
    python scripts/exp_predictor_variants.py [mesh path] [mesh path ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import _interior_pass_strip, _strip_traversal
from encoder.paradelta_codec import _fit_linear3, _best_rice_k
from utils.residual_entropy import _zigzag


PREDS = ["parallelogram", "laplacian", "midpoint", "copy_a", "centroid"]


def _meshlet_residuals(plan, vn, bnd_recon, delta, pred_id, w3,
                        fallback):
    """Run strip walk in f32 chain (decoder-faithful) and return
    residual codes (n_int, 3) for the chosen predictor variant."""
    n_bnd = plan["n_bnd"]
    n_int = plan["n_int"]
    if n_int == 0:
        return np.zeros((0, 3), dtype=np.int64)
    local_to_global = plan["local_to_global"]
    order = _strip_traversal(plan["ml_tris_local"], plan["strips"], n_bnd)
    delta_f32 = np.float32(delta)

    recon = {}
    for lid in range(n_bnd):
        recon[lid] = bnd_recon[int(local_to_global[lid])].astype(np.float32)

    w0 = w1 = w2 = None
    if pred_id == "parallelogram" and w3 is not None:
        w0 = np.float32(w3[0]); w1 = np.float32(w3[1]); w2 = np.float32(w3[2])

    codes_out = np.zeros((n_int, 3), dtype=np.int64)
    for i, (v_local, kind, refs) in enumerate(order):
        if kind == 'para':
            a, b, c = refs
            if pred_id == "parallelogram":
                if w3 is None:
                    pred = (recon[a] + recon[b] - recon[c]).astype(np.float32)
                else:
                    pred = (w0*recon[a] + w1*recon[b] + w2*recon[c]).astype(np.float32)
            elif pred_id == "laplacian":
                pred = ((recon[a] + recon[b] + recon[c]) / np.float32(3.0)).astype(np.float32)
            elif pred_id == "midpoint":
                pred = ((recon[a] + recon[b]) / np.float32(2.0)).astype(np.float32)
            elif pred_id == "copy_a":
                pred = recon[a].astype(np.float32)
            elif pred_id == "centroid":
                pred = fallback.copy()
            else:
                raise ValueError(pred_id)
        else:
            pred = fallback.copy()
        true = vn[int(local_to_global[v_local])].astype(np.float32)
        code = np.round((true - pred) / delta_f32).astype(np.int64)
        rec = (pred + code.astype(np.float32) * delta_f32).astype(np.float32)
        recon[v_local] = rec
        codes_out[i] = code
    return codes_out


def _rice_cost_3axis(codes: np.ndarray) -> int:
    if codes.shape[0] == 0:
        return 0
    total = 0
    for d in range(3):
        u = _zigzag(codes[:, d])
        _, c = _best_rice_k(u)
        total += c + 16    # +k header (8) + body header (8) ≈ 16 bits / axis
    return total


def _fallback(plan, bnd_recon, local_to_global):
    n_bnd = plan["n_bnd"]
    if n_bnd == 0:
        return np.zeros(3, dtype=np.float32)
    s = np.zeros(3, dtype=np.float32)
    for lid in range(n_bnd):
        s = s + bnd_recon[int(local_to_global[lid])].astype(np.float32)
    return (s / np.float32(n_bnd)).astype(np.float32)


def _process(prep, name: str):
    plans = prep["plans"]
    vn = prep["vn"]
    bnd_recon = prep["bnd_recon_norm"]
    delta = 2.0 * float(prep["per_coord_err"])

    # Fit global lin3 weights for parallelogram
    samples = []
    for plan in plans:
        _, sm = _interior_pass_strip(plan, vn, bnd_recon, delta, w3=None)
        samples.extend(sm)
    lin3 = _fit_linear3(samples)

    per_pred_total = {p: 0 for p in PREDS}
    adaptive_total = 0
    n_vert = prep["n_v"]
    n_bnd_total = prep["n_boundary"]

    chosen_hist = {p: 0 for p in PREDS}

    for plan in plans:
        local_to_global = plan["local_to_global"]
        fb = _fallback(plan, bnd_recon, local_to_global)
        meshlet_costs = []
        meshlet_codes = []
        for p in PREDS:
            w3 = lin3 if p == "parallelogram" else None
            codes = _meshlet_residuals(plan, vn, bnd_recon, delta, p,
                                         w3, fb)
            cost = _rice_cost_3axis(codes)
            per_pred_total[p] += cost
            meshlet_costs.append(cost)
            meshlet_codes.append(codes)
        best_idx = int(np.argmin(meshlet_costs))
        adaptive_total += meshlet_costs[best_idx] + 3  # 3-bit selector
        chosen_hist[PREDS[best_idx]] += 1

    print(f"\n=== {name}  (n_v={n_vert:,}, n_meshlets={len(plans):,}) ===")
    print(f"  lin3 = ({float(lin3[0]):.3f}, {float(lin3[1]):.3f}, "
          f"{float(lin3[2]):.3f})")
    print(f"\n  Per-predictor (fixed across whole mesh):")
    base = per_pred_total["parallelogram"]
    for p in PREDS:
        bits = per_pred_total[p]
        bpv = bits / n_vert
        delta_pct = (bits - base) / base * 100
        print(f"    {p:14s}: {bits:>14,} bits  "
              f"BPV(int)={bpv:6.3f}  "
              f"{delta_pct:+6.1f}% vs parallelogram")

    print(f"\n  Adaptive (per-meshlet best): "
          f"{adaptive_total:,} bits  "
          f"BPV(int)={adaptive_total/n_vert:.3f}  "
          f"{(adaptive_total-base)/base*100:+.1f}% vs parallelogram")

    print(f"\n  Chosen predictor histogram:")
    total_m = sum(chosen_hist.values()) or 1
    for p in PREDS:
        print(f"    {p:14s}: {chosen_hist[p]:>7,} "
              f"({100*chosen_hist[p]/total_m:5.1f}%)")


def main():
    paths = sys.argv[1:] or ["assets/Monkey.obj"]
    for p in paths:
        print(f"\n[loading {p}]")
        prep = load_or_prepare(p, max_verts=256, max_tris=256,
                                precision_error=0.0005,
                                gen_method="joint_learned",
                                strip_method="multiseed",
                                clean=True, verbose=False)
        _process(prep, Path(p).stem)


if __name__ == "__main__":
    main()
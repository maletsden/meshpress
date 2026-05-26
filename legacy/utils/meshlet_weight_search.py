"""Stage-3: Bayesian weight search over joint-cost generator weights.

Objective: minimize MeshletParaDelta BPV on stanford-bunny.

Requires: scikit-optimize (pip install scikit-optimize).
"""

import json
import os
import numpy as np

from reader import Reader
from encoder import MeshletParaDelta


WEIGHT_KEYS = [
    "w1_plane_resid",
    "w2_boundary_perim",
    "w3_strip_cont",
    "w4_normal_sim",
    "w5_shared_edges",
    "w6_bfs_depth",
]


def _objective_for(model, max_tris=256, max_verts=256, use_nn=False, use_cuda=True):
    """Returns a function f(w_vec) -> bpv."""

    def f(w_vec):
        s = float(sum(w_vec))
        if s < 1e-9:
            return 1e9
        weights = {k: float(w) / s for k, w in zip(WEIGHT_KEYS, w_vec)}
        try:
            enc = MeshletParaDelta(
                max_verts=max_verts, max_tris=max_tris,
                use_nn=use_nn, use_cuda=use_cuda,
                gen_method="joint", gen_weights=weights, verbose=False)
            compressed = enc.encode(model)
            bpv = float(compressed.bits_per_vertex)
            print(f"  w={['%.2f' % x for x in w_vec]} -> BPV={bpv:.3f}")
            return bpv
        except Exception as e:
            print(f"  failed: {e}")
            return 1e9

    return f


def search(mesh_path="assets/stanford-bunny.obj", n_calls=30,
           n_initial_points=10, max_tris=256, max_verts=256,
           use_nn=False, use_cuda=True,
           out_json="docs/meshlet_gen_best_weights.json"):
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        raise SystemExit(
            "scikit-optimize not installed. Run: pip install scikit-optimize")

    print(f"loading {mesh_path}")
    model = Reader.read_from_file(mesh_path)
    f = _objective_for(model, max_tris, max_verts, use_nn, use_cuda)

    space = [Real(1e-3, 1.0, name=k) for k in WEIGHT_KEYS]

    print(f"running gp_minimize: n_calls={n_calls}, n_initial={n_initial_points}")
    result = gp_minimize(
        f, space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=False,
    )
    best_w = list(result.x)
    s = sum(best_w)
    weights = {k: w / s for k, w in zip(WEIGHT_KEYS, best_w)}

    out = {
        "weights": weights,
        "bpv": float(result.fun),
        "mesh": mesh_path,
        "n_calls": n_calls,
        "history": [
            {"w": list(map(float, x)), "bpv": float(y)}
            for x, y in zip(result.x_iters, result.func_vals)
        ],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nbest BPV={out['bpv']:.3f}")
    print("best weights:")
    for k, v in weights.items():
        print(f"  {k}: {v:.4f}")
    print(f"saved {out_json}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="assets/stanford-bunny.obj")
    ap.add_argument("--n-calls", type=int, default=30)
    ap.add_argument("--max-tris", type=int, default=256)
    ap.add_argument("--max-verts", type=int, default=256)
    ap.add_argument("--use-nn", action="store_true")
    ap.add_argument("--use-cuda", action="store_true")
    args = ap.parse_args()

    search(
        mesh_path=args.mesh,
        n_calls=args.n_calls,
        max_tris=args.max_tris,
        max_verts=args.max_verts,
        use_nn=args.use_nn,
        use_cuda=args.use_cuda,
    )

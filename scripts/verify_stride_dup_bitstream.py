"""Roundtrip STRIDE-dup bitstream end-to-end.

  prep → encode_dup() → bytes → decode_dup() → (verts, tris)

Compare against the source mesh's quantized positions (per-vert sets,
since dup duplicates verts). PASS criterion: every decoded vert equals
some source vert within Δ/2 (the quantizer's bound).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5_dup import encode_dup, decode_dup  # noqa: E402


def verify(mesh: str):
    full = ROOT / mesh
    prep = load_or_prepare(str(full), max_verts=256, max_tris=256,
                            precision_error=1.0/4096.0,
                            precision_mode="bbox_frac",
                            gen_method="joint_learned",
                            strip_method="multiseed", verbose=False)
    data = encode_dup(prep, verbose=True)
    V_dec, T_dec = decode_dup(data)

    n_v = prep["n_v"]; n_t = prep["n_t"]
    bpv = 8 * len(data) / n_v
    print(f"  Decoded: n_v_dup={len(V_dec):,} n_t={len(T_dec):,}  BPV={bpv:.2f}")
    print(f"  Source : n_v={n_v:,} n_t={n_t:,}")

    if len(T_dec) != n_t:
        print(f"  TRI COUNT MISMATCH: {len(T_dec)} vs {n_t}")
        return

    # Surface-distance NN against source.
    from reader.reader import Reader
    m = Reader.read_from_file(str(full))
    src = np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float64)

    from scipy.spatial import cKDTree
    tree = cKDTree(src)
    d, _ = tree.query(V_dec.astype(np.float64), k=1)
    print(f"  GPU-vs-source NN: max={d.max():.6g}  mean={d.mean():.6g}  "
          f"std={d.std():.6g}")
    bbox = float((src.max(0) - src.min(0)).max())
    eps_q12 = bbox / 4095
    print(f"  bbox={bbox:.4g}  eps_q12={eps_q12:.6g}  "
          f"max_err/eps={d.max()/eps_q12:.2f}")


def main():
    paths = sys.argv[1:] or ["assets/Monkey.obj"]
    for p in paths:
        print(f"\n=== {p} ===")
        verify(p)


if __name__ == "__main__":
    main()

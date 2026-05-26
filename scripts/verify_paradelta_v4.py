"""Verify v4 ParaDelta roundtrip preserves verts/tris and shapes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import encode_from_prepared, decode_paradelta


def check(path: str) -> None:
    prep = load_or_prepare(
        path, max_verts=256, max_tris=256, precision_error=0.0005,
        gen_method="joint_learned", strip_method="multiseed", verbose=False)
    data = encode_from_prepared(prep, predictor="linear5", verbose=False)
    print(f"[{Path(path).name}] encoded={len(data):,} B  "
          f"BPV={len(data)*8/prep['n_v']:.2f}")
    verts, tris = decode_paradelta(data)
    nv, nt = len(verts), len(tris)
    assert nv == prep["n_v"], f"verts mismatch {nv} vs {prep['n_v']}"
    assert nt == prep["n_t"], f"tris mismatch {nt} vs {prep['n_t']}"
    # Spot-check a triangle vertex max
    assert tris.max() < nv, "tri index out of range"
    print(f"  OK  verts={nv}  tris={nt}")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        check(p)
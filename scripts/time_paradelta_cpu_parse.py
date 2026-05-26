"""Break down CPU parse cost: bit decode vs greedy_order."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, decode_paradelta_to_struct,
)

for path in ["D:/meshpress/assets/fandisk.obj",
             "D:/meshpress/assets/stanford-bunny.obj",
             "D:/meshpress/assets/Monkey.obj"]:
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    data = encode_from_prepared(prep, predictor="linear5", verbose=False)
    name = Path(path).name
    # warmup
    decode_paradelta_to_struct(data)
    t0 = time.perf_counter()
    for _ in range(3):
        decode_paradelta_to_struct(data)
    dt = (time.perf_counter() - t0) / 3
    print(f"[{name:20s}] parse_to_struct: {dt*1000:7.1f} ms  "
          f"size={len(data):,} B  n_t={prep['n_t']:,}  "
          f"({prep['n_t']/dt/1e6:6.1f} M tris/s parser)")
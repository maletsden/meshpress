"""Minimal driver for ncu profiling: Monkey only, 1 warmup + 3 runs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder

path = sys.argv[1] if len(sys.argv) > 1 else "D:/meshpress/assets/Monkey.obj"
prep = load_or_prepare(path, max_verts=256, max_tris=256,
                       precision_error=0.0005,
                       gen_method="joint_learned",
                       strip_method="multiseed", verbose=False)
data = encode_from_prepared_v5(prep, verbose=False)
dec = ParaDeltaV5GpuDecoder(data)
# Warmup
for _ in range(2):
    dec.decode()
cp.cuda.Device().synchronize()
# Measured
for _ in range(3):
    dec.decode()
cp.cuda.Device().synchronize()
print("ok")
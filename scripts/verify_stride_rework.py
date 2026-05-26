"""Verify reworked kernel output matches CPU decoder bit-exactly."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5, decode_paradelta_v5
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder

MESHES = [
    "assets/fandisk.obj",
    "assets/bunny.obj",
    "assets/eyeball.obj",
    "assets/horse.obj",
    "assets/stanford-bunny.obj",
]

for p in MESHES:
    full = ROOT / p
    if not full.exists():
        continue
    prep = load_or_prepare(str(full), max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    data = encode_from_prepared_v5(prep, verbose=False)

    v_cpu, t_cpu = decode_paradelta_v5(data)
    dec = ParaDeltaV5GpuDecoder(data)
    v_gpu, t_gpu = dec.decode_to_host()

    n_v = min(len(v_cpu), len(v_gpu))
    n_t = min(len(t_cpu), len(t_gpu))

    if v_cpu.shape != v_gpu.shape or t_cpu.shape != t_gpu.shape:
        print(f"{p}: SHAPE MISMATCH cpu={v_cpu.shape}/{t_cpu.shape} gpu={v_gpu.shape}/{t_gpu.shape}")
        continue

    v_diff = np.abs(v_cpu.astype(np.float64) - v_gpu.astype(np.float64)).max()
    t_diff = (t_cpu != t_gpu).sum()
    print(f"{p}: v_maxabs={v_diff:.3e}  t_mismatches={t_diff}")
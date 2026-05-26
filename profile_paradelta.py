"""Profile ParaDelta joint+multiseed phases on s-bunny.

Measures: meshlet generation, strip generation per meshlet, parallelogram
predictor (CPU vs CUDA), bit accounting. Identifies dominant phase.
"""

import time
import cProfile
import pstats

from reader import Reader
from encoder import MeshletParaDelta


def time_full(use_cuda=False):
    model = Reader.read_from_file("assets/stanford-bunny.obj")
    enc = MeshletParaDelta(
        max_tris=256, max_verts=256,
        gen_method="joint_learned",
        strip_method="multiseed",
        use_nn=False,
        use_cuda=use_cuda,
    )
    t0 = time.time()
    cm = enc.encode(model)
    dt = time.time() - t0
    print(f"use_cuda={use_cuda}: {dt:.2f}s  BPV={cm.bits_per_vertex:.2f}")
    return dt


def profile_cpu():
    model = Reader.read_from_file("assets/stanford-bunny.obj")
    enc = MeshletParaDelta(
        max_tris=256, max_verts=256,
        gen_method="joint_learned",
        strip_method="multiseed",
        use_nn=False,
        use_cuda=False,
    )
    pr = cProfile.Profile()
    pr.enable()
    enc.encode(model)
    pr.disable()
    st = pstats.Stats(pr).sort_stats("cumulative")
    st.print_stats(25)


def time_phases():
    """Compare strip_method=v2 vs multiseed; greedy vs joint_learned gen."""
    model = Reader.read_from_file("assets/stanford-bunny.obj")

    configs = [
        ("greedy + v2",         "greedy",        "v2"),
        ("greedy + multiseed",  "greedy",        "multiseed"),
        ("joint_learned + v2",  "joint_learned", "v2"),
        ("joint+multiseed",     "joint_learned", "multiseed"),
    ]
    for name, gm, sm in configs:
        for cuda in (False, True):
            try:
                enc = MeshletParaDelta(
                    max_tris=256, max_verts=256,
                    gen_method=gm, strip_method=sm,
                    use_nn=False, use_cuda=cuda,
                )
                t0 = time.time()
                cm = enc.encode(model)
                dt = time.time() - t0
                print(f"  {name:<22s} cuda={cuda}  t={dt:>6.2f}s  "
                      f"BPV={cm.bits_per_vertex:.2f}")
            except Exception as e:
                print(f"  {name} cuda={cuda} FAILED: {e}")


if __name__ == "__main__":
    print("=== Phase sweep ===")
    time_phases()
    print("\n=== cProfile (CPU, joint+multiseed) ===")
    profile_cpu()

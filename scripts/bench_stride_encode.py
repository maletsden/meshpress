"""Per-stage STRIDE encode wall-clock benchmark.

Measures: read+clean, quantize, partition (joint_learned), boundary table,
per-meshlet plan, LSQ fit, bit-pack. Outputs CSV + console table.

Usage:
    python scripts/bench_stride_encode.py
    python scripts/bench_stride_encode.py assets/Monkey.obj
"""
from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import Reader
from utils.mesh_clean import clean_mesh
from utils.bit_codec import BitWriter
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets,
)
from utils.boundary_split import (
    identify_boundary_verts, build_boundary_table, verify_crack_free,
)
from utils.boundary_bvh import morton_permute_boundary
from encoder.paradelta_codec import (
    _quantize_global, _dequant_global, _plan_meshlet,
)
from encoder.paradelta_v5 import (
    encode_from_prepared_v5, _interior_pass_strip, _fit_linear3,
    _write_meshlet,
)

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


@dataclass
class Stages:
    read_clean: float = 0.0
    quantize: float = 0.0
    partition: float = 0.0
    boundary: float = 0.0
    plan: float = 0.0
    lsq_fit: float = 0.0
    bit_pack: float = 0.0
    n_v: int = 0
    n_t: int = 0
    n_meshlets: int = 0
    bytes_out: int = 0

    @property
    def prepare(self) -> float:
        return (self.read_clean + self.quantize + self.partition
                + self.boundary + self.plan)

    @property
    def encode(self) -> float:
        return self.lsq_fit + self.bit_pack

    @property
    def total(self) -> float:
        return self.prepare + self.encode


def measure(path: Path, *, max_verts: int = 256, max_tris: int = 256,
            precision_error: float = 0.0005) -> Stages:
    st = Stages()

    t0 = time.perf_counter()
    model = Reader.read_from_file(str(path))
    model, _ = clean_mesh(model, verbose=False)
    st.read_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    verts = np.asarray([(v.x, v.y, v.z) for v in model.vertices],
                       dtype=np.float64)
    tris_np = np.asarray([(t.a, t.b, t.c) for t in model.triangles],
                         dtype=np.int64)
    n_v, n_t = len(verts), len(tris_np)
    center = verts.mean(axis=0)
    vc = verts - center
    scale = float(np.max(np.linalg.norm(vc, axis=1)))
    vn = vc / scale
    per_coord_err = precision_error / scale / math.sqrt(3)
    global_codes, g_min, g_range, g_bits = _quantize_global(vn, per_coord_err)
    bnd_recon_norm = _dequant_global(global_codes, g_min, g_range, g_bits)
    st.quantize = time.perf_counter() - t0
    st.n_v = n_v
    st.n_t = n_t

    t0 = time.perf_counter()
    tri_adj = build_adjacency(tris_np)
    fn = compute_face_normals(vn, tris_np)
    fc = compute_face_centroids(vn, tris_np)
    meshlets = generate_meshlets(
        tris_np, tri_adj, fn, fc,
        method="joint_learned", max_tris=max_tris, max_verts=max_verts,
        verts_np=vn,
    )
    st.partition = time.perf_counter() - t0
    st.n_meshlets = len(meshlets)

    t0 = time.perf_counter()
    boundary_set = identify_boundary_verts(meshlets, tris_np)
    boundary_list, _, _ = build_boundary_table(boundary_set, global_codes)
    boundary_list, _ = morton_permute_boundary(boundary_list, global_codes)
    gv_to_ref = {gv: i for i, gv in enumerate(boundary_list)}
    n_boundary = len(boundary_list)
    n_cracks, _ = verify_crack_free(
        meshlets, tris_np, global_codes, boundary_set)
    if n_cracks > 0:
        raise RuntimeError(f"crack-free check failed: {n_cracks}")
    st.boundary = time.perf_counter() - t0

    t0 = time.perf_counter()
    plans = [
        _plan_meshlet(ml, tris_np, tri_adj, vn,
                      boundary_set, global_codes, gv_to_ref, "multiseed")
        for ml in meshlets
    ]
    st.plan = time.perf_counter() - t0

    prep = {
        "center": center, "scale": scale, "per_coord_err": per_coord_err,
        "g_min": g_min, "g_range": g_range, "g_bits": g_bits,
        "n_v": n_v, "n_t": n_t, "n_boundary": n_boundary,
        "n_meshlets": len(meshlets), "boundary_list": boundary_list,
        "global_codes": global_codes, "bnd_recon_norm": bnd_recon_norm,
        "vn": vn, "plans": plans, "strip_method": "multiseed",
        "max_verts": max_verts, "max_tris": max_tris,
        "precision_error": precision_error, "gen_method": "joint_learned",
    }

    delta = 2.0 * per_coord_err
    t0 = time.perf_counter()
    all_samples = []
    for plan in plans:
        _, samples = _interior_pass_strip(
            plan, vn, bnd_recon_norm, delta, w3=None)
        all_samples.extend(samples)
    lin3_w = _fit_linear3(all_samples)
    st.lsq_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = encode_from_prepared_v5(prep, verbose=False)
    st.bit_pack = time.perf_counter() - t0
    st.bytes_out = len(out)

    return st


def fmt_time(s: float) -> str:
    if s < 1.0:
        return f"{s*1000:.0f} ms"
    return f"{s:.2f} s"


def main():
    paths = sys.argv[1:] or MESHES
    rows = []
    print(f"{'mesh':<22}{'n_t':>10}  {'read':>8} {'quant':>8} "
          f"{'partn':>8} {'bnd':>8} {'plan':>8} {'lsq':>8} {'pack':>8} "
          f"{'total':>10}")
    print("-" * 110)
    for p in paths:
        path = ROOT / p if not Path(p).is_absolute() else Path(p)
        if not path.exists():
            print(f"missing: {path}")
            continue
        st = measure(path)
        rows.append((path.stem, st))
        print(f"{path.stem:<22}{st.n_t:>10,}  "
              f"{fmt_time(st.read_clean):>8} {fmt_time(st.quantize):>8} "
              f"{fmt_time(st.partition):>8} {fmt_time(st.boundary):>8} "
              f"{fmt_time(st.plan):>8} {fmt_time(st.lsq_fit):>8} "
              f"{fmt_time(st.bit_pack):>8} {fmt_time(st.total):>10}")

    csv_path = ROOT / "bench_stride_encode.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mesh", "n_v", "n_t", "n_meshlets",
                    "read_clean_s", "quantize_s", "partition_s",
                    "boundary_s", "plan_s", "lsq_fit_s", "bit_pack_s",
                    "prepare_s", "encode_s", "total_s", "bytes_out"])
        for name, st in rows:
            w.writerow([name, st.n_v, st.n_t, st.n_meshlets,
                        f"{st.read_clean:.4f}", f"{st.quantize:.4f}",
                        f"{st.partition:.4f}", f"{st.boundary:.4f}",
                        f"{st.plan:.4f}", f"{st.lsq_fit:.4f}",
                        f"{st.bit_pack:.4f}", f"{st.prepare:.4f}",
                        f"{st.encode:.4f}", f"{st.total:.4f}",
                        st.bytes_out])
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
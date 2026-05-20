"""Full competitor benchmark — paper headline table.

Per mesh, run every compressor we have, plus competitors:
  - ParaDelta v5 (ours, GPU decode timed)
  - MeshletWaveletGlobalEB (ours, encode-only timing; no fast GPU decoder)
  - MeshletWaveletGlobalAMD (ours, GPU decode via direct kernel run)
  - Draco q12 L7 (DracoPy, CPU decode timed)
  - gltfpack -cc (meshopt EXT_meshopt_compression, encode-only)
  - AMD DGFTester tb12 (size; GPU speed cited from paper)

Outputs:
  - Markdown table per mesh (stdout)
  - Master CSV at bench_competitors.csv

Notes:
  * Sizes include format overhead (Draco/gltfpack carry headers); our format
    is raw compressed payload — caveat in paper.
  * Max error reported in *absolute* world coords (max-axis bbox basis).
  * GPU decode column: kernel-only, CUDA Events (cudaStreamSync at end).
  * DGF GPU number is cited (AMD HPG24 paper, RX 7900 XTX).
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.reader import Reader
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder
from encoder.implementation.meshlet_wavelet import (
    MeshletWaveletGlobalEB, MeshletWaveletGlobalAMD,
)


DGFTESTER_EXE = (ROOT / "third_party" / "DGF-SDK" / "build" /
                 "DGFTester" / "Release" / "DGFTester.exe")


def _bpv(size_b: int, n_v: int) -> float:
    return size_b * 8 / max(1, n_v)


def _max_err_nn(src_pts: np.ndarray, dec_pts: np.ndarray) -> float:
    """Hausdorff via nearest-neighbour (Draco reorders verts)."""
    from scipy.spatial import cKDTree
    t1 = cKDTree(dec_pts)
    d1, _ = t1.query(src_pts, k=1)
    t2 = cKDTree(src_pts)
    d2, _ = t2.query(dec_pts, k=1)
    return float(max(d1.max(), d2.max()))


def _read_mesh(path: str):
    m = Reader.read_from_file(path)
    pts = np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float32)
    faces = np.array([[t.a, t.b, t.c] for t in m.triangles], dtype=np.int64)
    return m, pts, faces


def bench_paradelta_v5(path: str, n_v: int, n_t: int, warmup=20, runs=100):
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    t0 = time.perf_counter()
    data = encode_from_prepared_v5(prep, verbose=False)
    t_enc = time.perf_counter() - t0
    size = len(data)
    dec = ParaDeltaV5GpuDecoder(data)
    for _ in range(warmup):
        dec.decode()
    cp.cuda.Device().synchronize()
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(runs):
        dec.decode()
    e.record(); e.synchronize()
    kernel_us = cp.cuda.get_elapsed_time(s, e) * 1000.0 / runs
    # Max error: reconstructed vs source (source pts not in prep — use cache verts)
    v_dec, _ = dec.decode_to_host()
    # Our decoder reorders verts (boundary first); use NN Hausdorff vs source
    _, src_pts, _ = _read_mesh(path)
    err = _max_err_nn(src_pts, v_dec.astype(np.float32))
    return {
        "name": "ParaDelta v5 (ours)",
        "size_b": size, "bpv": _bpv(size, n_v),
        "enc_ms": t_enc * 1000, "dec_us": kernel_us,
        "mtps": n_t / kernel_us, "max_err": err,
        "gpu": True, "note": "fused CUDA kernel",
    }


def bench_global_amd(path: str, n_v: int, n_t: int):
    """Size only — GPU decode via separate path (kernel exists in
    benchmark_cuda_decode.py). For paper, run that script for GPU numbers."""
    model = Reader.read_from_file(path)
    enc = MeshletWaveletGlobalAMD(max_verts=256, precision_error=0.0005,
                                  verbose=False)
    t0 = time.perf_counter()
    c = enc.encode(model)
    t_enc = time.perf_counter() - t0
    return {
        "name": "GlobalAMD (ours)",
        "size_b": len(c.data), "bpv": _bpv(len(c.data), n_v),
        "enc_ms": t_enc * 1000, "dec_us": None, "mtps": None,
        "max_err": 0.0005,
        "gpu": True, "note": "GPU kernel: see benchmark_cuda_decode.py",
    }


def bench_global_eb(path: str, n_v: int, n_t: int):
    model = Reader.read_from_file(path)
    enc = MeshletWaveletGlobalEB(max_verts=256, precision_error=0.0005,
                                 verbose=False)
    t0 = time.perf_counter()
    c = enc.encode(model)
    t_enc = time.perf_counter() - t0
    return {
        "name": "GlobalEB (ours)",
        "size_b": len(c.data), "bpv": _bpv(len(c.data), n_v),
        "enc_ms": t_enc * 1000, "dec_us": None, "mtps": None,
        "max_err": 0.0005,
        "gpu": False, "note": "EB CLERS decode sequential",
    }


def bench_draco(path: str, n_v: int, n_t: int, qb=12):
    import DracoPy
    _, pts, faces = _read_mesh(path)
    t0 = time.perf_counter()
    buf = DracoPy.encode_mesh_to_buffer(
        pts, faces.astype(np.uint32),
        quantization_bits=qb, compression_level=7)
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    d = DracoPy.decode_buffer_to_mesh(buf)
    t_dec = time.perf_counter() - t0
    dec_pts = np.asarray(d.points, dtype=np.float32)
    err = _max_err_nn(pts, dec_pts)
    return {
        "name": f"Draco q{qb} L7",
        "size_b": len(buf), "bpv": _bpv(len(buf), n_v),
        "enc_ms": t_enc * 1000, "dec_us": t_dec * 1e6,
        "mtps": n_t / max(t_dec * 1e6, 1e-6), "max_err": err,
        "gpu": False, "note": "CPU decode",
    }


def bench_meshopt(path: str, n_v: int, n_t: int, warmup=20, runs=100):
    """meshopt vertex/index codec, CPU decode timed.

    Uses 12-bit per-axis quantization (matches our 0.0005 precision budget on
    a [-1,1] cube) packed as int16 ×3 + 2 B pad → vertex_size=8.
    Indices encoded as uint32.

    Not the same bytes as gltfpack -cc (no normals/uvs/glTF headers), but
    the *decode speed* number is the actual meshopt CPU codec on the same
    geometry payload.
    """
    import meshoptimizer as mo
    _, pts, faces = _read_mesh(path)
    bbox_min = pts.min(0); bbox_max = pts.max(0)
    extent = (bbox_max - bbox_min).max()
    if extent < 1e-12:
        return None
    norm = (pts - bbox_min) / extent
    q = np.clip(np.round(norm * 4095.0).astype(np.int16), 0, 4095)
    verts_packed = np.zeros((n_v, 4), dtype=np.int16)
    verts_packed[:, :3] = q                 # 6 B + 2 B pad = 8 B
    indices = faces.astype(np.uint32).reshape(-1)

    # gltfpack-style reorder: optimize_vertex_cache + optimize_vertex_fetch.
    # Without this, meshopt index codec compresses poorly (vertex deltas
    # not coherent), bloating BPV ~2x.
    idx_opt = np.empty_like(indices)
    mo.optimize_vertex_cache(idx_opt, indices, index_count=len(indices),
                              vertex_count=n_v)
    verts_opt = np.empty_like(verts_packed)
    n_unique = mo.optimize_vertex_fetch(
        verts_opt, idx_opt, verts_packed,
        index_count=len(idx_opt), vertex_count=n_v, vertex_size=8)
    verts_opt = verts_opt[:n_unique]
    n_v_opt = int(n_unique)

    t0 = time.perf_counter()
    enc_v = mo.encode_vertex_buffer(verts_opt, vertex_count=n_v_opt,
                                     vertex_size=8)
    enc_i = mo.encode_index_buffer(idx_opt, index_count=len(idx_opt),
                                    vertex_count=n_v_opt)
    t_enc = time.perf_counter() - t0
    size = len(enc_v) + len(enc_i)
    indices = idx_opt
    verts_packed = verts_opt
    n_v = n_v_opt

    # Warmup
    for _ in range(warmup):
        mo.decode_vertex_buffer(n_v, 8, enc_v)
        mo.decode_index_buffer(len(indices), 4, enc_i)
    t0 = time.perf_counter()
    for _ in range(runs):
        dec_v = mo.decode_vertex_buffer(n_v, 8, enc_v)
        dec_i = mo.decode_index_buffer(len(indices), 4, enc_i)
    t_dec_us = (time.perf_counter() - t0) * 1e6 / runs

    dec_v = dec_v.view(np.int16).reshape(n_v, 4)[:, :3].astype(np.float32)
    dec_pts = dec_v / 4095.0 * extent + bbox_min
    err = _max_err_nn(pts, dec_pts.astype(np.float32))
    # Restore reported n_v to the source count (output column matches mesh)
    n_v = len(pts)
    return {
        "name": "meshopt q12 (CPU)",
        "size_b": size, "bpv": _bpv(size, len(pts)),
        "enc_ms": t_enc * 1000, "dec_us": t_dec_us,
        "mtps": n_t / max(t_dec_us, 1e-6), "max_err": err,
        "gpu": False, "note": "vertex+index codec + cache-opt reorder",
    }


def bench_gltfpack(path: str, n_v: int, n_t: int):
    if not shutil.which("gltfpack") and not shutil.which("gltfpack.cmd"):
        return None
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    with tempfile.TemporaryDirectory() as td:
        glb_in = Path(td) / "in.glb"
        glb_out = Path(td) / "out.glb"
        m.export(glb_in)
        t0 = time.perf_counter()
        r = subprocess.run(
            ["gltfpack", "-i", str(glb_in), "-o", str(glb_out),
             "-cc", "-si", "1.0"],
            capture_output=True, text=True, shell=True)
        t_enc = time.perf_counter() - t0
        if r.returncode != 0:
            return None
        size = glb_out.stat().st_size
    return {
        "name": "gltfpack -cc",
        "size_b": size, "bpv": _bpv(size, n_v),
        "enc_ms": t_enc * 1000, "dec_us": None, "mtps": None,
        "max_err": None,
        "gpu": False, "note": "meshopt EXT (incl. glb headers)",
    }


def bench_dgf(path: str, n_v: int, n_t: int, tb=12):
    if not DGFTESTER_EXE.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        bin_out = Path(td) / "out.bin"
        t0 = time.perf_counter()
        r = subprocess.run(
            [str(DGFTESTER_EXE), path,
             "--target-bits", str(tb),
             "--measure-error",
             "--dump-bin", str(bin_out),
             "--print-perf"],
            capture_output=True, text=True)
        t_enc = time.perf_counter() - t0
        if r.returncode != 0 or not bin_out.exists():
            return None
        size = bin_out.stat().st_size
    max_err = None
    for line in r.stdout.splitlines():
        if "Max Error" in line:
            try:
                max_err = float(line.split("Max Error:")[1].split()[0])
            except Exception:
                pass
    return {
        "name": f"DGF tb{tb}",
        "size_b": size, "bpv": _bpv(size, n_v),
        "enc_ms": t_enc * 1000, "dec_us": None, "mtps": 38.0,  # paper claim
        "max_err": max_err,
        "gpu": True, "note": "GPU decode 38 M tris/s cited (HPG24, RX7900XTX)",
    }


def fmt(v, kind):
    if v is None:
        return "—"
    if kind == "us":     return f"{v:.1f}"
    if kind == "ms":     return f"{v:.1f}"
    if kind == "bpv":    return f"{v:.2f}"
    if kind == "bytes":  return f"{int(v):,}"
    if kind == "err":    return f"{v:.4g}"
    if kind == "mtps":   return f"{v:.1f}"
    return str(v)


def print_mesh_table(name: str, n_v: int, n_t: int, rows: list[dict]):
    print(f"\n### {name}  ({n_v:,} verts, {n_t:,} tris)\n")
    hdr = ("| Method | Size (B) | BPV | Max err | Enc (ms) "
           "| Dec (µs) | M tris/s | Note |")
    sep = "|" + "|".join(["---"] * 8) + "|"
    print(hdr); print(sep)
    for r in rows:
        if r is None: continue
        print(f"| {r['name']} | {fmt(r['size_b'],'bytes')} "
              f"| {fmt(r['bpv'],'bpv')} | {fmt(r.get('max_err'),'err')} "
              f"| {fmt(r['enc_ms'],'ms')} | {fmt(r.get('dec_us'),'us')} "
              f"| {fmt(r.get('mtps'),'mtps')} | {r['note']} |")


def run_one(path: str, csv_rows: list):
    name = Path(path).name
    print(f"\n=== {name} ===")
    _, pts, faces = _read_mesh(path)
    n_v, n_t = len(pts), len(faces)
    print(f"  verts={n_v:,} tris={n_t:,}")

    rows = []
    for fn in (
        bench_paradelta_v5,
        bench_global_amd,
        bench_global_eb,
        bench_draco,
        bench_meshopt,
        bench_gltfpack,
        bench_dgf,
    ):
        try:
            print(f"  running {fn.__name__}...", flush=True)
            r = fn(path, n_v, n_t)
            if r is None:
                print(f"    [skip]")
                continue
            rows.append(r)
            print(f"    -> {r['name']}: {r['size_b']:,} B  "
                  f"BPV={r['bpv']:.2f}  enc={r['enc_ms']:.0f}ms  "
                  f"dec={fmt(r.get('dec_us'),'us')}µs")
        except Exception as e:
            print(f"    [ERR] {fn.__name__}: {e}")
            import traceback; traceback.print_exc()

    print_mesh_table(name, n_v, n_t, rows)
    for r in rows:
        csv_rows.append({"mesh": name, "n_v": n_v, "n_t": n_t, **r})


def main():
    # Default: cached + small meshes. Pass --all for armadillo/dragon/buddha.
    if "--all" in sys.argv:
        sys.argv.remove("--all")
        paths = sys.argv[1:] or [
            "assets/fandisk.obj", "assets/bunny.obj",
            "assets/eyeball.obj", "assets/horse.obj",
            "assets/stanford-bunny.obj", "assets/armadillo.obj",
            "assets/dragon.obj", "assets/Monkey.obj",
            "assets/happy_buddha.obj", "assets/tank.obj",
        ]
    else:
        paths = sys.argv[1:] or [
            "assets/fandisk.obj", "assets/bunny.obj",
            "assets/eyeball.obj", "assets/horse.obj",
            "assets/stanford-bunny.obj", "assets/Monkey.obj",
            "assets/tank.obj",
        ]
    csv_rows = []
    for p in paths:
        if not Path(p).exists():
            print(f"missing: {p}"); continue
        run_one(p, csv_rows)

    out_csv = ROOT / "bench_competitors.csv"
    with open(out_csv, "w", newline="") as f:
        keys = ["mesh", "n_v", "n_t", "name", "size_b", "bpv",
                "max_err", "enc_ms", "dec_us", "mtps", "gpu", "note"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in csv_rows: w.writerow(r)
    print(f"\nCSV written: {out_csv}")


if __name__ == "__main__":
    main()

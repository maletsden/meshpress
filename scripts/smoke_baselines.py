"""Smoke test: run each encoder variant on a single small model.

Verifies API works, captures size/BPV/encode_time/decode_ok per variant.
Usage:
    python scripts/smoke_baselines.py [assets/fandisk.obj]
"""

import os
import sys
import time
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.reader import Reader


def _row(name, size_b, n_v, n_t, t_enc, decode_ok, extra=""):
    bpv = size_b * 8 / n_v if n_v else 0.0
    bpt = size_b * 8 / n_t if n_t else 0.0
    ok = "OK" if decode_ok else "----"
    print(f"  [{name:<22s}]  size={size_b:>9,} B  BPV={bpv:7.2f}  "
          f"BPT={bpt:7.2f}  enc={t_enc:6.2f}s  dec={ok}  {extra}")


def run_global_eb(path: str, n_v: int, n_t: int):
    from encoder.implementation.meshlet_wavelet import MeshletWaveletGlobalEB
    model = Reader.read_from_file(path)
    enc = MeshletWaveletGlobalEB(max_verts=256, precision_error=0.0005,
                                 verbose=False)
    t0 = time.time()
    c = enc.encode(model)
    t = time.time() - t0
    _row("MeshletWaveletGlobalEB", len(c.data), n_v, n_t, t,
         True, "crack-free EB")


def run_paradelta_linear5(path: str, n_v: int, n_t: int):
    from utils.paradelta_cache import load_or_prepare
    from encoder.paradelta_codec import encode_from_prepared, decode_paradelta
    prep = load_or_prepare(
        path, max_verts=256, max_tris=256, precision_error=0.0005,
        gen_method="joint_learned", strip_method="multiseed", verbose=False)
    t0 = time.time()
    data = encode_from_prepared(prep, predictor="linear5", verbose=False)
    t = time.time() - t0
    try:
        verts_d, tris_d = decode_paradelta(data)
        dec_ok = (len(verts_d) == n_v)
    except Exception as e:
        dec_ok = False
        print(f"    [decode err] {e}")
    _row("ParaDelta linear5", len(data), n_v, n_t, t, dec_ok,
         "gts_v3 conn")


def run_draco(path: str, n_v: int, n_t: int):
    """Sweep Draco quant bits.

    Our budget: precision_error=0.0005 in normalized [-1,1] coord space
    (range = 2). Step = 0.0005, so 2/0.0005 = 4000 positions → 12 bits.
    Draco normalizes to longest axis so q11 ≈ 0.00098 step,
    q12 ≈ 0.00049 step (closest match), q14 default, q16 high quality.
    """
    import DracoPy
    model = Reader.read_from_file(path)
    pts = np.array([[v.x, v.y, v.z] for v in model.vertices], dtype=np.float32)
    faces = np.array([[t.a, t.b, t.c] for t in model.triangles], dtype=np.uint32)
    # measure bbox to report effective max abs error per quant level
    bbox = pts.max(0) - pts.min(0)
    max_axis = float(bbox.max())

    for qb in (11, 12, 14, 16):
        t0 = time.time()
        buf = DracoPy.encode_mesh_to_buffer(
            pts, faces, quantization_bits=qb, compression_level=7)
        t = time.time() - t0
        try:
            d = DracoPy.decode_buffer_to_mesh(buf)
            n_dec_v = len(d.points) if hasattr(d.points, "__len__") else 0
            f = d.faces
            if hasattr(f, "ndim") and f.ndim == 2:
                n_dec_t = f.shape[0]
            else:
                n_dec_t = len(f) // 3
            dec_ok = (n_dec_v == n_v) and (n_dec_t == n_t)
            # Draco reorders verts; use nearest-neighbor Hausdorff
            d_pts = np.asarray(d.points, dtype=np.float32)
            from scipy.spatial import cKDTree
            tree = cKDTree(d_pts)
            d_src_to_dec, _ = tree.query(pts, k=1)
            tree2 = cKDTree(pts)
            d_dec_to_src, _ = tree2.query(d_pts, k=1)
            max_abs_err = float(max(d_src_to_dec.max(), d_dec_to_src.max()))
        except Exception as e:
            dec_ok = False
            max_abs_err = -1.0
            print(f"    [decode err q{qb}] {e}")
        step = max_axis / (2 ** qb)
        _row(f"Draco q{qb},L7", len(buf), n_v, n_t, t, dec_ok,
             f"step={step:.6f}  maxErr={max_abs_err:.6f}")


DGFTESTER_EXE = (Path(ROOT) / "third_party" / "DGF-SDK" / "build" /
                 "DGFTester" / "Release" / "DGFTester.exe")


def run_dgftester(path: str, n_v: int, n_t: int):
    """AMD DGF (Dense Geometry Format) — Knipping et al. 2024.

    target-bits 12 matches our 0.0005 precision budget.
    """
    if not DGFTESTER_EXE.exists():
        print(f"  [DGFTester] missing exe: {DGFTESTER_EXE}")
        return
    for tb in (12, 14, 16):
        with tempfile.TemporaryDirectory() as td:
            bin_out = Path(td) / "out.bin"
            t0 = time.time()
            r = subprocess.run(
                [str(DGFTESTER_EXE), path,
                 "--target-bits", str(tb),
                 "--measure-error",
                 "--dump-bin", str(bin_out)],
                capture_output=True, text=True)
            t = time.time() - t0
            if r.returncode != 0:
                print(f"  [DGFTester tb{tb}] code={r.returncode}")
                continue
            size = bin_out.stat().st_size if bin_out.exists() else 0
        # parse Max Error from stdout
        max_err = -1.0
        for line in r.stdout.splitlines():
            if "Max Error" in line:
                try:
                    max_err = float(line.split("Max Error:")[1].split()[0])
                except Exception:
                    pass
        _row(f"DGF tb{tb}", size, n_v, n_t, t, True,
             f"maxErr={max_err:.6f}")


def run_gltfpack(path: str, n_v: int, n_t: int):
    import trimesh
    gltfpack = shutil.which("gltfpack") or shutil.which("gltfpack.ps1")
    if not gltfpack:
        print("  [gltfpack] not on PATH, skip")
        return
    m = trimesh.load(path, process=False, force="mesh")
    with tempfile.TemporaryDirectory() as td:
        glb_in = Path(td) / "in.glb"
        glb_out = Path(td) / "out.glb"
        m.export(glb_in)
        t0 = time.time()
        # -cc = compress with meshopt EXT_meshopt_compression
        # -si 1.0 = no simplification
        r = subprocess.run(
            ["gltfpack", "-i", str(glb_in), "-o", str(glb_out),
             "-cc", "-si", "1.0"],
            capture_output=True, text=True, shell=True)
        t = time.time() - t0
        if r.returncode != 0:
            print(f"  [gltfpack] failed (code {r.returncode})")
            print(f"    stderr: {r.stderr[:200]}")
            return
        size = glb_out.stat().st_size
    _row("gltfpack -cc", size, n_v, n_t, t, True,
         "meshopt compressed glb (incl headers)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/fandisk.obj"
    if not os.path.exists(path):
        print(f"missing: {path}")
        sys.exit(1)
    model = Reader.read_from_file(path)
    n_v = len(model.vertices)
    n_t = len(model.triangles)
    print(f"\n=== {path}  verts={n_v:,}  tris={n_t:,} ===\n")

    variants = [
        ("MeshletWaveletGlobalEB", run_global_eb),
        ("ParaDelta linear5", run_paradelta_linear5),
        ("Draco", run_draco),
        ("DGFTester", run_dgftester),
        ("gltfpack", run_gltfpack),
    ]
    for name, fn in variants:
        try:
            fn(path, n_v, n_t)
        except Exception as e:
            import traceback
            print(f"  [{name}] ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
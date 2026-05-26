"""Reconstruction-quality measurement for STRIDE + competitors.

For each (mesh, codec) we encode -> decode -> measure four metrics
between the source mesh and the reconstruction:

  RMSE         sqrt(mean ||v_src - v_recon||^2)
  Hausdorff_mean   symmetric average nearest-point distance
  Hausdorff_max    symmetric max     nearest-point distance
  PSNR             20 * log10(peak / RMSE), peak = source bbox diagonal

All distances are in world coordinates. Lengths normalised by the
source longest-extent (so values are directly comparable to the
encoder's epsilon = 0.0005 target) are written as the *_norm columns.

Outputs:
  bench_recon_stride.csv             STRIDE on all 8 paper meshes
  bench_recon_competitors_monkey.csv 5 codecs on Monkey (paper Table 7b)
"""
from __future__ import annotations

import csv
import sys
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402

from reader import Reader  # noqa: E402
from utils.paradelta_cache import load_or_prepare  # noqa: E402
from encoder.paradelta_v5 import encode_from_prepared_v5  # noqa: E402
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder  # noqa: E402
from utils.bench_config import stride_precision, csv_suffix, mode_label  # noqa: E402

_PREC = stride_precision()
_SUFFIX = csv_suffix()
print(f"[bench_recon_quality] precision = {mode_label()}")


DGFTESTER_EXE = (ROOT / "third_party" / "DGF-SDK" / "build" /
                 "DGFTester" / "Release" / "DGFTester.exe")
CORTO_EXE = (ROOT / "third_party" / "corto" / "build" /
              "Release" / "corto.exe")

STRIDE_CORPUS = [
    "assets/fandisk.obj",
    "assets/stanford-bunny.obj",
    "assets/horse.obj",
    "assets/Monkey.obj",
    "assets/happy_buddha.obj",
    "assets/crab.obj",
    "assets/tank.obj",
    "assets/xyzrgb_dragon.obj",
]


# ---------------------------------------------------------------- metrics

def _read_mesh(path: str, clean: bool = True):
    """Read OBJ. When clean, apply the same clean_mesh_npy pre-pass the
    STRIDE encoder uses — merges coincident-quant verts, drops degenerate
    + duplicate triangles. That is the *actual* input the encoder sees,
    so it is the right comparison target for reconstruction-error metrics.
    Without this pre-pass, orphan verts referenced only by degenerate
    triangles end up in the KDTree and inflate H_max spuriously."""
    from reader.fast_obj import clean_mesh_npy
    m = Reader.read_from_file(path)
    pts   = np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float64)
    faces = np.array([[t.a, t.b, t.c] for t in m.triangles], dtype=np.int64)
    if clean:
        pts, faces = clean_mesh_npy(pts, faces)
    return pts, faces


def compute_metrics(src_pts: np.ndarray, src_faces: np.ndarray,
                    dec_pts: np.ndarray, dec_faces: np.ndarray,
                    n_samples: int = 200_000) -> dict:
    """Surface-to-surface metrics via trimesh `closest_point`.

    Standard MPEG / CGA mesh-quality methodology: sample ~`n_samples`
    points uniformly on each surface, compute point-to-surface distance
    against the other mesh (point-to-triangle, not vertex-NN). Robust
    to vertex count, order, and re-tessellation differences between
    codecs.
    """
    import trimesh
    src = np.asarray(src_pts, dtype=np.float64)
    dec = np.asarray(dec_pts, dtype=np.float64)
    src_f = np.asarray(src_faces, dtype=np.int64)
    dec_f = np.asarray(dec_faces, dtype=np.int64)

    m_src = trimesh.Trimesh(vertices=src, faces=src_f, process=False)
    m_dec = trimesh.Trimesh(vertices=dec, faces=dec_f, process=False)

    bbox = src.max(0) - src.min(0)
    peak = float(np.linalg.norm(bbox))      # bbox diagonal
    extent = float(bbox.max())              # longest axis

    rng = np.random.default_rng(0xC0DEBA5E)
    n = int(min(n_samples, max(50_000, n_samples)))
    pts_on_src, _ = trimesh.sample.sample_surface(m_src, n, seed=0xC0DE)
    pts_on_dec, _ = trimesh.sample.sample_surface(m_dec, n, seed=0xC0DE)

    # src surface samples vs dec surface
    _, d_a, _ = m_dec.nearest.on_surface(pts_on_src)
    # dec surface samples vs src surface
    _, d_b, _ = m_src.nearest.on_surface(pts_on_dec)

    rmse   = float(np.sqrt(np.mean(d_a ** 2)))   # one-sided RMSE
    h_mean = float(0.5 * (d_a.mean() + d_b.mean()))
    h_max  = float(max(d_a.max(),     d_b.max()))
    psnr   = 20.0 * np.log10(peak / rmse) if rmse > 0 else float("inf")

    return {
        "rmse":          rmse,
        "rmse_norm":     rmse  / extent,
        "h_mean":        h_mean,
        "h_mean_norm":   h_mean / extent,
        "h_max":         h_max,
        "h_max_norm":    h_max  / extent,
        "psnr_db":       float(psnr),
        "bbox_extent":   extent,
        "bbox_diag":     peak,
        "n_samples":     n,
    }


# ---------------------------------------------------------------- codecs

def recon_stride(path: str):
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False,
                           **_PREC)
    data = encode_from_prepared_v5(prep, verbose=False)
    dec = ParaDeltaV5GpuDecoder(data)
    v_dec, t_dec = dec.decode_to_host()
    cp.cuda.Device().synchronize()
    return (np.asarray(v_dec, dtype=np.float64),
            np.asarray(t_dec, dtype=np.int64),
            len(data))


def recon_draco(path: str, qb: int = 12):
    import DracoPy
    pts, faces = _read_mesh(path)
    buf = DracoPy.encode_mesh_to_buffer(
        pts.astype(np.float32), faces.astype(np.uint32),
        quantization_bits=qb, compression_level=7)
    d = DracoPy.decode_buffer_to_mesh(buf)
    dec_pts = np.asarray(d.points, dtype=np.float64)
    dec_faces = np.asarray(d.faces, dtype=np.int64).reshape(-1, 3)
    return dec_pts, dec_faces, len(buf)


def recon_meshopt(path: str, qb: int = 12):
    import meshoptimizer as mo
    pts, faces = _read_mesh(path)
    n_v = len(pts); n_t = len(faces)
    bbox_min = pts.min(0); bbox_max = pts.max(0)
    extent = (bbox_max - bbox_min).max()
    norm = (pts - bbox_min) / extent
    qmax = (1 << qb) - 1
    q = np.clip(np.round(norm * qmax).astype(np.int16), 0, qmax)
    verts_packed = np.zeros((n_v, 4), dtype=np.int16)
    verts_packed[:, :3] = q
    indices = faces.astype(np.uint32).reshape(-1)

    idx_opt = np.empty_like(indices)
    mo.optimize_vertex_cache(idx_opt, indices, index_count=len(indices),
                              vertex_count=n_v)
    verts_opt = np.empty_like(verts_packed)
    n_unique = mo.optimize_vertex_fetch(
        verts_opt, idx_opt, verts_packed,
        index_count=len(idx_opt), vertex_count=n_v, vertex_size=8)
    verts_opt = verts_opt[:n_unique]
    n_v_opt = int(n_unique)

    enc_v = mo.encode_vertex_buffer(verts_opt, vertex_count=n_v_opt,
                                     vertex_size=8)
    enc_i = mo.encode_index_buffer(idx_opt, index_count=len(idx_opt),
                                    vertex_count=n_v_opt)
    size = len(enc_v) + len(enc_i)
    dec_v = mo.decode_vertex_buffer(n_v_opt, 8, enc_v)
    dec_v = dec_v.view(np.int16).reshape(n_v_opt, 4)[:, :3].astype(np.float64)
    dec_pts = dec_v / qmax * extent + bbox_min
    dec_faces = idx_opt.astype(np.int64).reshape(-1, 3)
    return dec_pts, dec_faces, size


def recon_corto(path: str, qb: int = 12):
    if not CORTO_EXE.exists():
        return None, None, None
    from reader import Reader as _R
    m = _R.read_from_file(path)
    pts = np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float32)
    faces = np.array([[t.a, t.b, t.c] for t in m.triangles], dtype=np.int32)

    with tempfile.TemporaryDirectory() as td:
        ply_in  = Path(td) / "in.ply"
        crt_out = Path(td) / "out.crt"
        ply_rt  = Path(td) / "rt.ply"
        n_v, n_f = len(pts), len(faces)
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n_v}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {n_f}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        with open(ply_in, "wb") as f:
            f.write(header)
            f.write(np.asarray(pts, dtype="<f4").tobytes())
            face_rec = np.empty(n_f, dtype=[("c", "u1"), ("idx", "<i4", 3)])
            face_rec["c"] = 3
            face_rec["idx"] = faces.astype("<i4")
            f.write(face_rec.tobytes())

        common = ["-v", str(qb), "-n", "0", "-u", "0", "-c", "0"]
        subprocess.run([str(CORTO_EXE)] + common +
                       ["-o", str(crt_out), str(ply_in)],
                       capture_output=True, text=True)
        size = crt_out.stat().st_size
        subprocess.run([str(CORTO_EXE)] + common +
                       ["-P", str(ply_rt), str(ply_in)],
                       capture_output=True, text=True)
        if not ply_rt.exists():
            return None, None, None
        import trimesh
        m_rt = trimesh.load(str(ply_rt), process=False, force="mesh")
        return (np.asarray(m_rt.vertices, dtype=np.float64),
                np.asarray(m_rt.faces, dtype=np.int64), size)


def recon_dgf(path: str, tb: int = 12):
    if not DGFTESTER_EXE.exists():
        return None, None, None
    with tempfile.TemporaryDirectory() as td:
        bin_out = Path(td) / "out.bin"
        obj_out = Path(td) / "decoded.obj"
        r = subprocess.run(
            [str(DGFTESTER_EXE), path,
             "--target-bits", str(tb),
             "--dump-bin", str(bin_out),
             "--dump-obj", str(obj_out)],
            capture_output=True, text=True)
        if r.returncode != 0 or not obj_out.exists():
            return None, None, None
        size = bin_out.stat().st_size if bin_out.exists() else 0
        import trimesh
        m_dec = trimesh.load(str(obj_out), process=False, force="mesh")
        return (np.asarray(m_dec.vertices, dtype=np.float64),
                np.asarray(m_dec.faces, dtype=np.int64), size)


# ---------------------------------------------------------------- driver

def measure(path: str, codec: str):
    src_pts, src_faces = _read_mesh(path)
    t0 = time.perf_counter()
    if codec == "STRIDE":
        dec_pts, dec_faces, size = recon_stride(path)
    elif codec == "Draco":
        dec_pts, dec_faces, size = recon_draco(path)
    elif codec == "meshopt":
        dec_pts, dec_faces, size = recon_meshopt(path)
    elif codec == "Corto":
        dec_pts, dec_faces, size = recon_corto(path)
    elif codec == "DGF":
        dec_pts, dec_faces, size = recon_dgf(path)
    else:
        raise ValueError(codec)
    t_total = time.perf_counter() - t0
    if dec_pts is None:
        print(f"  [skip] {codec}")
        return None
    m = compute_metrics(src_pts, src_faces, dec_pts, dec_faces)
    m.update({
        "mesh": Path(path).name,
        "codec": codec,
        "n_v_src":  len(src_pts),
        "n_v_dec":  len(dec_pts),
        "size_b":   int(size),
        "wall_s":   t_total,
        "ratio_to_eps": m["h_max_norm"] / 0.0005,
    })
    print(f"  {codec:8s}  n_v_dec={len(dec_pts):,}  "
          f"RMSE={m['rmse']:.4g}  H_mean={m['h_mean']:.4g}  "
          f"H_max={m['h_max']:.4g}  PSNR={m['psnr_db']:.2f} dB  "
          f"({t_total:.1f}s)")
    return m


def run_stride_corpus():
    rows = []
    for p in STRIDE_CORPUS:
        full = ROOT / p
        if not full.exists():
            print(f"  [skip] {p} missing"); continue
        print(f"\n=== STRIDE on {p} ===")
        r = measure(str(full), "STRIDE")
        if r: rows.append(r)
    out = ROOT / f"bench_recon_stride{_SUFFIX}.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nWritten: {out}")


def run_monkey_competitors():
    p = ROOT / "assets" / "Monkey.obj"
    if not p.exists():
        print("[skip] Monkey.obj missing"); return
    print(f"\n=== Competitor head-to-head on Monkey ===")
    rows = []
    for codec in ["STRIDE", "DGF", "Draco", "meshopt", "Corto"]:
        r = measure(str(p), codec)
        if r: rows.append(r)
    out = ROOT / f"bench_recon_competitors_monkey{_SUFFIX}.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nWritten: {out}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "stride"):
        run_stride_corpus()
    if mode in ("all", "monkey"):
        run_monkey_competitors()

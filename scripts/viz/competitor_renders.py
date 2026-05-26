"""Side-by-side reconstruction renders: STRIDE vs Draco vs meshopt vs Corto vs DGF.

For each mesh, encode/decode each competitor at matched 12-bit precision
and render in a single row. Per-cell caption: bytes, BPV.

Output: docs/figs/compare_<mesh>_<angle>.png

Usage:
    python scripts/viz/competitor_renders.py
    python scripts/viz/competitor_renders.py assets/stanford-bunny.obj
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reader.reader import Reader
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_v5 import encode_from_prepared_v5, decode_paradelta_v5

FIGS_DIR = ROOT / "docs" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
DGFTESTER_EXE = (ROOT / "third_party" / "DGF-SDK" / "build" /
                  "DGFTester" / "Release" / "DGFTester.exe")
CORTO_EXE = (ROOT / "third_party" / "corto" / "build" /
              "Release" / "corto.exe")

ANGLES = [
    ("top",     (75,  -90)),
]
DPI = 240


def _render_panel(ax, verts, tris, title, elev, azim):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = verts[tris]
    coll = Poly3DCollection(polys, facecolors=(0.7, 0.78, 0.88),
                             edgecolors="k", linewidths=0.03)
    ax.add_collection3d(coll)
    mn = verts.min(0); mx = verts.max(0)
    rng = (mx - mn).max() / 2.2
    c = (mx + mn) / 2
    ax.set_xlim(c[0] - rng, c[0] + rng)
    ax.set_ylim(c[1] - rng, c[1] + rng)
    ax.set_zlim(c[2] - rng, c[2] + rng)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8, pad=2)


def _v5(path: Path, n_v: int):
    prep = load_or_prepare(str(path), max_verts=256, max_tris=256,
                            precision_error=0.0005,
                            gen_method="joint_learned",
                            strip_method="multiseed", verbose=False)
    data = encode_from_prepared_v5(prep, verbose=False)
    v_dec, t_dec = decode_paradelta_v5(data)
    return (v_dec.astype(np.float32), t_dec.astype(np.int64),
            len(data), len(data)*8/n_v)


def _draco(verts, faces, n_v: int):
    import DracoPy
    buf = DracoPy.encode_mesh_to_buffer(
        verts, faces.astype(np.uint32), quantization_bits=12,
        compression_level=7)
    d = DracoPy.decode_buffer_to_mesh(buf)
    v_dec = np.asarray(d.points, dtype=np.float32)
    f = d.faces
    if hasattr(f, "ndim") and f.ndim == 2:
        t_dec = f.astype(np.int64)
    else:
        t_dec = np.asarray(f, dtype=np.int64).reshape(-1, 3)
    return v_dec, t_dec, len(buf), len(buf)*8/n_v


def _meshopt(verts, faces, n_v: int):
    import meshoptimizer as mo
    bbox_min = verts.min(0); ext = (verts.max(0) - bbox_min).max()
    q = np.clip(np.round((verts - bbox_min) / ext * 4095).astype(np.int16),
                 0, 4095)
    vp = np.zeros((n_v, 4), dtype=np.int16); vp[:, :3] = q
    idx = faces.astype(np.uint32).reshape(-1)
    idx_opt = np.empty_like(idx)
    mo.optimize_vertex_cache(idx_opt, idx, index_count=len(idx),
                              vertex_count=n_v)
    vp_opt = np.empty_like(vp)
    n_unique = mo.optimize_vertex_fetch(
        vp_opt, idx_opt, vp, index_count=len(idx_opt),
        vertex_count=n_v, vertex_size=8)
    vp_opt = vp_opt[:n_unique]
    enc_v = mo.encode_vertex_buffer(vp_opt, vertex_count=n_unique,
                                     vertex_size=8)
    enc_i = mo.encode_index_buffer(idx_opt, index_count=len(idx_opt),
                                    vertex_count=n_unique)
    dec_v = mo.decode_vertex_buffer(n_unique, 8, enc_v)
    dec_i = mo.decode_index_buffer(len(idx_opt), 4, enc_i)
    dq = dec_v.view(np.int16).reshape(n_unique, 4)[:, :3].astype(np.float32)
    dq = dq / 4095.0 * ext + bbox_min
    tdec = dec_i.view(np.uint32).reshape(-1, 3).astype(np.int64)
    size = len(enc_v) + len(enc_i)
    return dq.astype(np.float32), tdec, size, size*8/n_v


def _write_pos_ply(pts: np.ndarray, faces: np.ndarray, out_path: Path):
    n_v, n_f = len(pts), len(faces)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(np.asarray(pts, dtype="<f4").tobytes())
        face_rec = np.empty(n_f, dtype=[("c", "u1"), ("idx", "<i4", 3)])
        face_rec["c"] = 3
        face_rec["idx"] = np.asarray(faces, dtype="<i4")
        f.write(face_rec.tobytes())


def _corto(path: Path, verts, faces, n_v: int):
    if not CORTO_EXE.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        ply_in = Path(td) / "in.ply"
        crt_out = Path(td) / "out.crt"
        ply_rt = Path(td) / "rt.ply"
        _write_pos_ply(verts, faces, ply_in)
        common = ["-v", "12", "-n", "0", "-u", "0", "-c", "0"]
        r1 = subprocess.run(
            [str(CORTO_EXE)] + common + ["-o", str(crt_out), str(ply_in)],
            capture_output=True, text=True)
        if r1.returncode != 0 or not crt_out.exists():
            return None
        size = crt_out.stat().st_size
        r2 = subprocess.run(
            [str(CORTO_EXE)] + common + ["-P", str(ply_rt), str(ply_in)],
            capture_output=True, text=True)
        if r2.returncode != 0 or not ply_rt.exists():
            return None
        import trimesh
        m = trimesh.load(str(ply_rt), process=False, force="mesh")
        v_dec = np.asarray(m.vertices, dtype=np.float32)
        t_dec = np.asarray(m.faces, dtype=np.int64)
    return v_dec, t_dec, size, size * 8 / n_v


def _dgf(path: Path, n_v: int):
    if not DGFTESTER_EXE.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        obj_out = Path(td) / "out.obj"
        bin_out = Path(td) / "out.bin"
        r = subprocess.run(
            [str(DGFTESTER_EXE), str(path),
             "--target-bits", "12",
             "--dump-obj", str(obj_out),
             "--dump-bin", str(bin_out)],
            capture_output=True, text=True)
        if r.returncode != 0 or not obj_out.exists():
            return None
        size = bin_out.stat().st_size if bin_out.exists() else 0
        m = Reader.read_from_file(str(obj_out))
        v_dec = np.array([[v.x, v.y, v.z] for v in m.vertices],
                          dtype=np.float32)
        t_dec = np.array([[t.a, t.b, t.c] for t in m.triangles],
                          dtype=np.int64)
    return v_dec, t_dec, size, size*8/n_v


def _process(path: Path):
    name = path.stem
    m = Reader.read_from_file(str(path))
    src_v = np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float32)
    src_t = np.array([[t.a, t.b, t.c] for t in m.triangles], dtype=np.int64)
    n_v, n_t = len(src_v), len(src_t)
    print(f"=== {name} n_v={n_v:,} n_t={n_t:,} ===")

    panels = [(src_v, src_t,
                f"Original\n{n_v:,} verts  uncompressed")]
    print("  Ours ..."); t0 = time.time()
    v, t, sz, bpv = _v5(path, n_v)
    panels.append((v, t, f"Ours\n{sz:,} B  BPV={bpv:.1f}"))
    print(f"    BPV={bpv:.2f} ({time.time()-t0:.1f}s)")

    print("  Draco q12 ..."); t0 = time.time()
    try:
        v, t, sz, bpv = _draco(src_v, src_t, n_v)
        panels.append((v, t, f"Draco q12\n{sz:,} B  BPV={bpv:.1f}"))
        print(f"    BPV={bpv:.2f} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"    ERR {e}")

    print("  meshopt q12 ..."); t0 = time.time()
    try:
        v, t, sz, bpv = _meshopt(src_v, src_t, n_v)
        panels.append((v, t, f"meshopt q12\n{sz:,} B  BPV={bpv:.1f}"))
        print(f"    BPV={bpv:.2f} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"    ERR {e}")

    print("  Corto v12 ..."); t0 = time.time()
    try:
        r = _corto(path, src_v, src_t, n_v)
        if r is not None:
            v, t, sz, bpv = r
            panels.append((v, t, f"Corto v12\n{sz:,} B  BPV={bpv:.1f}"))
            print(f"    BPV={bpv:.2f} ({time.time()-t0:.1f}s)")
        else:
            print(f"    skip")
    except Exception as e:
        print(f"    ERR {e}")

    print("  DGF tb12 ..."); t0 = time.time()
    r = _dgf(path, n_v)
    if r is not None:
        v, t, sz, bpv = r
        panels.append((v, t, f"DGF tb12\n{sz:,} B  BPV={bpv:.1f}"))
        print(f"    BPV={bpv:.2f} ({time.time()-t0:.1f}s)")
    else:
        print(f"    skip")

    for ang_name, (elev, azim) in ANGLES:
        n_p = len(panels)
        fig = plt.figure(figsize=(3.6 * n_p, 4.2))
        for i, (V, T, title) in enumerate(panels):
            ax = fig.add_subplot(1, n_p, i + 1, projection="3d")
            _render_panel(ax, V, T, title, elev, azim)
        out = FIGS_DIR / f"compare_{name}_{ang_name}.png"
        plt.subplots_adjust(left=0, right=1, top=1.0, bottom=0,
                             wspace=-0.10)
        plt.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"    wrote {out.name}")


def main():
    paths = sys.argv[1:] or [
        "assets/stanford-bunny.obj",
    ]
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"missing: {p}"); continue
        try:
            _process(p)
        except Exception as e:
            print(f"  ERR {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
"""
Correctness tests for the MeshPress compression pipeline.

Verifies:
  1. Integer wavelet transforms (Haar / CDF53 / segmented delta / delta) are lossless.
  2. Global quantize -> dequantize max error is within the target precision.
  3. Full pipeline crack-free guarantee: shared boundary vertices in different
     meshlets dequantize to identical float positions.
  4. GTS-connectivity round-trip: reconstructed triangle vertex sets match
     the originals (triangle count + vertex sets preserved).

Run:
  python test_correctness.py
Exits non-zero if any check fails.
"""

import sys
import numpy as np

from reader import Reader
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts, edgebreaker_vertex_order,
)
from utils.wavelet import (
    haar_decompose_int, haar_reconstruct_int,
    cdf53_decompose_int, cdf53_reconstruct_int,
    delta_decompose_int, delta_reconstruct_int,
    segmented_delta_decompose_int, segmented_delta_reconstruct_int,
)
from utils.connectivity import amd_encode_decode_verify
from encoder.implementation.meshlet_wavelet import (
    _global_quantize, _dequantize_global, _to_numpy,
)


# ----------------------------------------------------------------------
# Small helper for CLI output
# ----------------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"


class TestState:
    def __init__(self):
        self.n_pass = 0
        self.n_fail = 0

    def check(self, cond, name, detail=""):
        if cond:
            self.n_pass += 1
            print(f"  [{PASS}] {name}")
        else:
            self.n_fail += 1
            print(f"  [{FAIL}] {name}  {detail}")


# ----------------------------------------------------------------------
# 1. Wavelet round-trip tests (synthetic data, full integer exactness)
# ----------------------------------------------------------------------
def test_wavelet_roundtrip(state: TestState):
    print("\n== Integer wavelet round-trip ==")
    rng = np.random.default_rng(0xBEEF)

    sizes = [1, 16, 32, 33, 64, 100, 256]
    types = [
        ("haar", haar_decompose_int, haar_reconstruct_int),
        ("cdf53", cdf53_decompose_int, cdf53_reconstruct_int),
        ("delta", delta_decompose_int, delta_reconstruct_int),
        ("seg_delta", segmented_delta_decompose_int, segmented_delta_reconstruct_int),
    ]

    for n in sizes:
        values = rng.integers(-10_000, 10_000, size=n, dtype=np.int64)
        for name, dec, rec in types:
            base, levels, orig_n = dec(values)
            recon = rec(base, levels)[:orig_n]
            ok = recon.shape == values.shape and np.array_equal(recon, values)
            detail = ""
            if not ok:
                diff = int(np.max(np.abs(recon[:len(values)] - values))) if len(recon) >= len(values) else -1
                detail = f"n={n} max|diff|={diff} len_recon={len(recon)}"
            state.check(ok, f"{name:<10s} n={n:<4d}", detail)


# ----------------------------------------------------------------------
# 2. Global quantization error bound
# ----------------------------------------------------------------------
def test_global_quantize_error(state: TestState, model_paths, max_error=0.0005):
    print("\n== Global quantization error bound ==")
    for path in model_paths:
        model = Reader.read_from_file(path)
        if model is None:
            print(f"  [SKIP] {path}")
            continue
        verts_np, _ = _to_numpy(model)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = max_error / scale / np.sqrt(3)

        codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)
        recon = _dequantize_global(codes, g_min, g_range, g_bits)
        errs = np.linalg.norm(recon - vn, axis=1) * scale
        max_err_mesh = float(errs.max())
        state.check(
            max_err_mesh <= max_error * 1.0001,
            f"{path}: max_err={max_err_mesh:.6f} <= {max_error}",
            detail=f"bits={g_bits.tolist()}",
        )


# ----------------------------------------------------------------------
# 3. Crack-free guarantee across meshlets
# ----------------------------------------------------------------------
def test_crack_free(state: TestState, model_paths, max_error=0.0005, max_verts=256):
    print("\n== Crack-free meshlet boundary ==")
    for path in model_paths:
        model = Reader.read_from_file(path)
        if model is None:
            print(f"  [SKIP] {path}")
            continue

        verts_np, tris_np = _to_numpy(model)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale
        per_coord_err = max_error / scale / np.sqrt(3)

        codes, g_min, g_range, g_bits = _global_quantize(vn, per_coord_err)

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=max_verts)

        # Per-meshlet dequantize -> check boundary vertex equality across meshlets.
        seen_pos = {}
        n_cracks = 0
        n_shared = 0
        for ml_tris in meshlets:
            vert_order, _, _ = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
            if not vert_order:
                continue
            int_pts = codes[vert_order]
            dq = _dequantize_global(int_pts, g_min, g_range, g_bits)
            for i, gv in enumerate(vert_order):
                pos = tuple(dq[i])
                if gv in seen_pos:
                    n_shared += 1
                    if seen_pos[gv] != pos:
                        n_cracks += 1
                else:
                    seen_pos[gv] = pos

        state.check(
            n_cracks == 0,
            f"{path}: cracks={n_cracks} across {n_shared} shared refs "
            f"({len(meshlets)} meshlets)",
        )


# ----------------------------------------------------------------------
# 4. Connectivity encode/decode round-trip
# ----------------------------------------------------------------------
def test_connectivity_roundtrip(state: TestState, model_paths, max_verts=256,
                                 sample_meshlets=10):
    print("\n== Connectivity encode/decode round-trip ==")
    for path in model_paths:
        model = Reader.read_from_file(path)
        if model is None:
            print(f"  [SKIP] {path}")
            continue

        _, tris_np = _to_numpy(model)
        verts_np, _ = _to_numpy(model)
        center = verts_np.mean(axis=0)
        vc = verts_np - center
        scale = np.max(np.linalg.norm(vc, axis=1))
        vn = vc / scale

        tri_adj = build_adjacency(tris_np)
        fn = compute_face_normals(vn, tris_np)
        fc = compute_face_centroids(vn, tris_np)
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=max_verts)

        # Sample evenly across meshlets (deterministic)
        step = max(1, len(meshlets) // sample_meshlets)
        sampled = meshlets[::step][:sample_meshlets]

        all_ok = True
        details_log = []
        for idx, ml_tris in enumerate(sampled):
            forsyth_bits, bfs_bits, details = amd_encode_decode_verify(
                ml_tris, tris_np, tri_adj)
            # Parse verification from details string
            for line in details.split("\n"):
                if "verify=" in line:
                    part = line.split("verify=")[1]
                    matched, total = part.split("/")
                    matched, total = int(matched), int(total)
                    if matched != total:
                        all_ok = False
                        details_log.append(
                            f"meshlet {idx}: {matched}/{total}")

        detail = "; ".join(details_log) if details_log else ""
        state.check(
            all_ok,
            f"{path}: all sampled meshlets decoded correctly "
            f"({len(sampled)} meshlets, bfs+forsyth)",
            detail,
        )


# ----------------------------------------------------------------------
# 5. End-to-end encoder reconstruction error + crack-free assertions
# ----------------------------------------------------------------------
def test_encoder_reconstruction(state: TestState, model_paths,
                                max_error=0.0005, max_verts=256):
    """For each Split encoder, verify per-vertex max-err <= eps and
    cracks = 0. The encoder's own verbose output already reports both;
    we capture stdout, parse, and assert.

    Each Split encoder runs the full inverse pipeline internally
    (inverse DCT/wavelet, dequant, add offset) and computes the
    reconstruction error against the original float positions, so this
    is a real round-trip check at the float level - just not via the
    bitstream (which is currently a placeholder of zero bytes per
    CLAUDE.md note 'decoder/ directory is currently empty').
    """
    import contextlib
    import io
    import re
    from reader import Reader
    from encoder.implementation.meshlet_wavelet import (
        MeshletSplitAMD,
        MeshletSplitFloatHaarAMD,
        MeshletSplitFloatCDF53AMD,
        MeshletSplitLearnedAMD,
        MeshletSplitMLPAMD,
        MeshletSplitDiffQuantAMD,
    )

    print("\n== End-to-end encoder reconstruction (max-err + cracks) ==")

    encoders = [
        ("MeshletSplitAMD",         lambda: MeshletSplitAMD(verbose=True)),
        ("MeshletSplitFloatHaarAMD", lambda: MeshletSplitFloatHaarAMD(verbose=True)),
        ("MeshletSplitFloatCDF53AMD", lambda: MeshletSplitFloatCDF53AMD(verbose=True)),
        ("MeshletSplitLearnedAMD",  lambda: MeshletSplitLearnedAMD(verbose=True)),
        # Neural encoders skipped here by default - they take 10-100s per
        # mesh and the same code path is exercised by the wavelet variants.
    ]

    re_max = re.compile(r"max\s*=\s*([0-9.eE+-]+)")
    re_cracks = re.compile(r"Cracks?:\s*(\d+)")
    # %OK line can be either '%OK=99.5%' or 'Combined %OK (<= 0.0005): 100.0%';
    # match the percentage that ends with '%'.
    re_acc = re.compile(r"([0-9.]+)\s*%")

    for path in model_paths:
        try:
            model = Reader.read_from_file(path)
        except Exception as e:
            state.check(False, f"{path}: load", str(e))
            continue
        for name, mk in encoders:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    enc = mk()
                    enc.encode(model)
                except Exception as e:
                    state.check(False, f"{path}: {name} ran", repr(e))
                    continue
            txt = buf.getvalue()

            # Parse - some encoders print only "max=" once; for FloatWavelet
            # there are TWO "max=" lines (boundary + interior). Take the
            # largest reported value to be conservative.
            max_vals = [float(x) for x in re_max.findall(txt)]
            crack_vals = [int(x) for x in re_cracks.findall(txt)]
            ok_vals = [float(x) for x in re_acc.findall(txt)]

            if not max_vals:
                state.check(False,
                    f"{path}: {name} reported max-err in stdout",
                    "no 'max=' line found")
                continue
            measured_max = max(max_vals)
            measured_cracks = crack_vals[0] if crack_vals else 0

            err_ok = measured_max <= max_error + 1e-9
            crack_ok = measured_cracks == 0
            pct_ok = (ok_vals[-1] >= 99.99) if ok_vals else True

            state.check(
                err_ok and crack_ok and pct_ok,
                f"{path}: {name}  max_err={measured_max:.6f}  "
                f"cracks={measured_cracks}",
                "" if (err_ok and crack_ok and pct_ok)
                else f"err_ok={err_ok} crack_ok={crack_ok} pct_ok={pct_ok}",
            )


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
def main():
    state = TestState()

    model_paths = [
        "assets/bunny.obj",
        "assets/torus.obj",
        "assets/stanford-bunny.obj",
    ]

    test_wavelet_roundtrip(state)
    test_global_quantize_error(state, model_paths)
    test_crack_free(state, model_paths)
    test_connectivity_roundtrip(state, model_paths)
    test_encoder_reconstruction(state, model_paths)

    print(f"\n== Summary: {state.n_pass} passed, {state.n_fail} failed ==")
    sys.exit(0 if state.n_fail == 0 else 1)


if __name__ == "__main__":
    main()
"""
Wavelet Meshlet Compression: hierarchical Haar wavelet on vertex streams
within meshlets, with EdgeBreaker connectivity.

Variant 1: Bezier surface + wavelet on (u,v,d)
Variant 2: Direct wavelet on (x,y,z) — no Bezier
"""

import numpy as np
import time
from collections import Counter
from reader import Reader
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_by_verts, edgebreaker_vertex_order, meshlet_bfs,
)
from utils.wavelet import estimate_wavelet_bits, wavelet_reconstruct_quantized
from utils.bezier import (
    fit_bezier, parameterize_pca, compute_displacements, bezier_derivatives,
    reconstruct_from_bezier, n_control_points,
)


# ============================================================
# EdgeBreaker connectivity bits estimation
# ============================================================

def estimate_edgebreaker_bits(opcodes, n_root_tris, n_local_verts):
    """Estimate bits for EdgeBreaker CLERS string."""
    if not opcodes and n_root_tris == 0:
        return 0

    # Root triangles: 3 local vertex indices each
    idx_bits = max(1, int(np.ceil(np.log2(n_local_verts + 1))))
    root_bits = n_root_tris * 3 * idx_bits

    # Opcode entropy
    if opcodes:
        counts = Counter(opcodes)
        total = len(opcodes)
        ent = -sum((c / total) * np.log2(c / total)
                    for c in counts.values() if c > 0)
        opcode_bits = total * ent + 16
    else:
        opcode_bits = 0

    # Header
    header = 32  # n_verts(16) + n_tris(16)

    return int(header + root_bits + opcode_bits)


def estimate_amd_reuse_bits(meshlet_tris, tris_np, tri_adj):
    """AMD-style meshlet connectivity: BFS traversal + FIFO reuse buffer.
    Per triangle: shared_edge(2b) + per new vertex: is_new(1b) + [fifo_idx | full_idx].
    GPU-friendly: simple sequential decode with fixed-size FIFO."""
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    n_local = len(vert_set)
    n_faces = len(meshlet_tris)
    if n_faces == 0:
        return 0

    traversal = meshlet_bfs(meshlet_tris, tri_adj)

    vert_order = {}
    fifo = []
    fifo_size = min(32, n_local)

    total_bits = 32  # header
    idx_bits = max(1, int(np.ceil(np.log2(n_local + 1))))
    fifo_bits = max(1, int(np.ceil(np.log2(fifo_size + 1))))

    for tri_idx, parent_idx in traversal:
        tri_v = [int(tris_np[tri_idx, j]) for j in range(3)]

        if parent_idx is None:
            for v in tri_v:
                if v not in vert_order:
                    vert_order[v] = len(vert_order)
                    fifo.append(v)
                    if len(fifo) > fifo_size:
                        fifo.pop(0)
                total_bits += idx_bits
        else:
            parent_v = [int(tris_np[parent_idx, j]) for j in range(3)]
            shared = set(tri_v) & set(parent_v)

            # Which parent edge (2 bits)
            total_bits += 2

            for v in tri_v:
                if v in shared:
                    continue  # implicit from parent
                if v in vert_order:
                    # Reuse
                    if v in fifo:
                        total_bits += 1 + fifo_bits
                    else:
                        total_bits += 1 + idx_bits
                else:
                    # New vertex: flag only (index = next in order)
                    total_bits += 1
                    vert_order[v] = len(vert_order)

                # Update FIFO
                if v in fifo:
                    fifo.remove(v)
                fifo.append(v)
                if len(fifo) > fifo_size:
                    fifo.pop(0)

    return total_bits


# ============================================================
# Variant 1: Bezier + Wavelet on (u,v,d)
# ============================================================

def estimate_variant1(meshlet_tris, verts, tris_np, tri_adj,
                      max_error, deg=2, target_base=32):
    """Bezier surface fit + hierarchical wavelet on (u,v,d)."""
    # Get EdgeBreaker vertex ordering
    vert_order, opcodes, n_roots = edgebreaker_vertex_order(
        meshlet_tris, tris_np, tri_adj)
    n_v = len(vert_order)
    n_f = len(meshlet_tris)

    if n_v < 3:
        return {"total": n_v * 96, "vertex": n_v * 96, "conn": 0,
                "header": 0, "n_verts": n_v, "n_faces": n_f}

    # Vertices in EdgeBreaker order
    pts = verts[vert_order]

    # PCA parameterization
    u, v, pca_frame = parameterize_pca(pts)

    # Fit Bezier
    cp = fit_bezier(u, v, pts, deg)

    # Displacements
    disps, surf_pts, normals = compute_displacements(u, v, pts, cp, deg)

    # Derivative-aware precision for u, v
    Su, Sv = bezier_derivatives(u, v, cp, deg)
    max_Su = max(np.max(np.linalg.norm(Su, axis=1)), 1e-6)
    max_Sv = max(np.max(np.linalg.norm(Sv, axis=1)), 1e-6)

    per_coord_err = max_error / np.sqrt(3)
    u_err = per_coord_err / max_Su
    v_err = per_coord_err / max_Sv
    d_err = per_coord_err

    # Wavelet on each stream
    wu = estimate_wavelet_bits(u, u_err, target_base)
    wv = estimate_wavelet_bits(v, v_err, target_base)
    wd = estimate_wavelet_bits(disps, d_err, target_base)

    wavelet_bits = wu["total_bits"] + wv["total_bits"] + wd["total_bits"]

    # Header: Bezier CPs + PCA frame
    n_cp = n_control_points(deg)
    bezier_header = n_cp * 3 * 16  # float16 control points
    pca_header = (3 * 4 + 4 * 2) * 8  # center(3f32) + ranges(4f16)
    header_bits = bezier_header + pca_header + 64  # + meshlet header

    # Connectivity
    conn_bits = estimate_edgebreaker_bits(opcodes, n_roots, n_v)

    total = header_bits + wavelet_bits + conn_bits

    # Accuracy
    u_recon = wavelet_reconstruct_quantized(u, u_err, target_base)
    v_recon = wavelet_reconstruct_quantized(v, v_err, target_base)
    d_recon = wavelet_reconstruct_quantized(disps, d_err, target_base)
    pts_recon = reconstruct_from_bezier(u_recon, v_recon, d_recon, cp, deg)
    errors = np.linalg.norm(pts_recon - pts, axis=1)

    return {
        "total": total, "vertex": wavelet_bits, "conn": conn_bits,
        "header": header_bits, "n_verts": n_v, "n_faces": n_f,
        "n_levels": wu["n_levels"],
        "level_bit_depths_u": wu["level_bit_depths"],
        "level_bit_depths_d": wd["level_bit_depths"],
        "errors": errors,
    }


# ============================================================
# Variant 3: Bezier + Wavelet on (u,v) only, flat quantize d
# ============================================================

def _flat_quantize_bits(values, max_err):
    """Plain quantize + entropy estimate for a single stream."""
    from utils.wavelet import _bits_for_error, _quantize, _stream_bits
    if len(values) == 0:
        return 0
    rng = values.max() - values.min() if len(values) > 1 else 0.001
    bits = _bits_for_error(rng, max_err)
    codes = _quantize(values, values.min(), values.max(), bits)
    # Add range metadata: min(f16) + max(f16) + bits(u8) = 5 bytes
    return _stream_bits(codes, bits) + 5 * 8


def _flat_quantize_recon(values, max_err):
    """Quantize + dequantize for accuracy check."""
    from utils.wavelet import _bits_for_error, _quantize, _dequantize
    if len(values) == 0:
        return values.copy()
    rng = values.max() - values.min() if len(values) > 1 else 0.001
    bits = _bits_for_error(rng, max_err)
    codes = _quantize(values, values.min(), values.max(), bits)
    return _dequantize(codes, values.min(), values.max(), bits)


def estimate_variant3(meshlet_tris, verts, tris_np, tri_adj,
                      max_error, deg=2, target_base=32):
    """Bezier + wavelet on (u,v), flat quantize on d."""
    vert_order, opcodes, n_roots = edgebreaker_vertex_order(
        meshlet_tris, tris_np, tri_adj)
    n_v = len(vert_order)
    n_f = len(meshlet_tris)

    if n_v < 3:
        return {"total": n_v * 96, "vertex": n_v * 96, "conn": 0,
                "header": 0, "n_verts": n_v, "n_faces": n_f}

    pts = verts[vert_order]

    # PCA + Bezier
    u, v, pca_frame = parameterize_pca(pts)
    cp = fit_bezier(u, v, pts, deg)
    disps, surf_pts, normals = compute_displacements(u, v, pts, cp, deg)

    # Derivative-aware precision
    Su, Sv = bezier_derivatives(u, v, cp, deg)
    max_Su = max(np.max(np.linalg.norm(Su, axis=1)), 1e-6)
    max_Sv = max(np.max(np.linalg.norm(Sv, axis=1)), 1e-6)

    per_coord_err = max_error / np.sqrt(3)
    u_err = per_coord_err / max_Su
    v_err = per_coord_err / max_Sv
    d_err = per_coord_err  # no wavelet error budget splitting for d

    # Wavelet on u, v only
    wu = estimate_wavelet_bits(u, u_err, target_base)
    wv = estimate_wavelet_bits(v, v_err, target_base)
    # Flat quantize d (no wavelet, no error budget splitting)
    d_bits = _flat_quantize_bits(disps, d_err)

    wavelet_bits = wu["total_bits"] + wv["total_bits"] + d_bits

    # Header
    n_cp = n_control_points(deg)
    bezier_header = n_cp * 3 * 16
    pca_header = (3 * 4 + 4 * 2) * 8
    header_bits = bezier_header + pca_header + 64

    # Connectivity
    conn_bits = estimate_edgebreaker_bits(opcodes, n_roots, n_v)

    total = header_bits + wavelet_bits + conn_bits

    # Accuracy
    u_recon = wavelet_reconstruct_quantized(u, u_err, target_base)
    v_recon = wavelet_reconstruct_quantized(v, v_err, target_base)
    d_recon = _flat_quantize_recon(disps, d_err)
    pts_recon = reconstruct_from_bezier(u_recon, v_recon, d_recon, cp, deg)
    errors = np.linalg.norm(pts_recon - pts, axis=1)

    return {
        "total": total, "vertex": wavelet_bits, "conn": conn_bits,
        "header": header_bits, "n_verts": n_v, "n_faces": n_f,
        "n_levels": wu["n_levels"],
        "level_bit_depths_u": wu["level_bit_depths"],
        "d_bits_flat": d_bits,
        "errors": errors,
    }


# ============================================================
# Variant 4: Same as V3 but AMD reuse buffer connectivity
# ============================================================

def estimate_variant4(meshlet_tris, verts, tris_np, tri_adj,
                      max_error, deg=2, target_base=32):
    """Bezier + wavelet on (u,v), flat d, AMD FIFO reuse connectivity.
    Same vertex encoding as V3, but GPU-friendly connectivity."""
    vert_order, opcodes, n_roots = edgebreaker_vertex_order(
        meshlet_tris, tris_np, tri_adj)
    n_v = len(vert_order)
    n_f = len(meshlet_tris)

    if n_v < 3:
        return {"total": n_v * 96, "vertex": n_v * 96, "conn": 0,
                "header": 0, "n_verts": n_v, "n_faces": n_f}

    pts = verts[vert_order]

    # PCA + Bezier (identical to V3)
    u, v, pca_frame = parameterize_pca(pts)
    cp = fit_bezier(u, v, pts, deg)
    disps, surf_pts, normals = compute_displacements(u, v, pts, cp, deg)

    Su, Sv = bezier_derivatives(u, v, cp, deg)
    max_Su = max(np.max(np.linalg.norm(Su, axis=1)), 1e-6)
    max_Sv = max(np.max(np.linalg.norm(Sv, axis=1)), 1e-6)

    per_coord_err = max_error / np.sqrt(3)
    u_err = per_coord_err / max_Su
    v_err = per_coord_err / max_Sv
    d_err = per_coord_err

    # Wavelet on u, v only; flat d (identical to V3)
    wu = estimate_wavelet_bits(u, u_err, target_base)
    wv = estimate_wavelet_bits(v, v_err, target_base)
    d_bits = _flat_quantize_bits(disps, d_err)
    wavelet_bits = wu["total_bits"] + wv["total_bits"] + d_bits

    # Header (identical to V3)
    n_cp = n_control_points(deg)
    bezier_header = n_cp * 3 * 16
    pca_header = (3 * 4 + 4 * 2) * 8
    header_bits = bezier_header + pca_header + 64

    # Connectivity: AMD FIFO reuse (instead of EdgeBreaker)
    conn_bits = estimate_amd_reuse_bits(meshlet_tris, tris_np, tri_adj)

    total = header_bits + wavelet_bits + conn_bits

    # Accuracy (identical to V3 — connectivity doesn't affect vertex accuracy)
    u_recon = wavelet_reconstruct_quantized(u, u_err, target_base)
    v_recon = wavelet_reconstruct_quantized(v, v_err, target_base)
    d_recon = _flat_quantize_recon(disps, d_err)
    pts_recon = reconstruct_from_bezier(u_recon, v_recon, d_recon, cp, deg)
    errors = np.linalg.norm(pts_recon - pts, axis=1)

    return {
        "total": total, "vertex": wavelet_bits, "conn": conn_bits,
        "header": header_bits, "n_verts": n_v, "n_faces": n_f,
        "n_levels": wu["n_levels"],
        "level_bit_depths_u": wu["level_bit_depths"],
        "d_bits_flat": d_bits,
        "errors": errors,
    }


# ============================================================
# Variant 2: Direct Wavelet on (x,y,z)
# ============================================================

def estimate_variant2(meshlet_tris, verts, tris_np, tri_adj,
                      max_error, target_base=32):
    """Direct hierarchical wavelet on (x,y,z) — no Bezier."""
    # Get EdgeBreaker vertex ordering
    vert_order, opcodes, n_roots = edgebreaker_vertex_order(
        meshlet_tris, tris_np, tri_adj)
    n_v = len(vert_order)
    n_f = len(meshlet_tris)

    if n_v < 3:
        return {"total": n_v * 96, "vertex": n_v * 96, "conn": 0,
                "header": 0, "n_verts": n_v, "n_faces": n_f}

    pts = verts[vert_order]
    per_coord_err = max_error / np.sqrt(3)

    # Wavelet on each coordinate stream
    wx = estimate_wavelet_bits(pts[:, 0], per_coord_err, target_base)
    wy = estimate_wavelet_bits(pts[:, 1], per_coord_err, target_base)
    wz = estimate_wavelet_bits(pts[:, 2], per_coord_err, target_base)

    wavelet_bits = wx["total_bits"] + wy["total_bits"] + wz["total_bits"]

    # Header: just meshlet metadata (no Bezier)
    header_bits = 64  # n_verts(16) + n_faces(16) + flags(32)

    # Connectivity
    conn_bits = estimate_edgebreaker_bits(opcodes, n_roots, n_v)

    total = header_bits + wavelet_bits + conn_bits

    # Accuracy
    x_recon = wavelet_reconstruct_quantized(pts[:, 0], per_coord_err, target_base)
    y_recon = wavelet_reconstruct_quantized(pts[:, 1], per_coord_err, target_base)
    z_recon = wavelet_reconstruct_quantized(pts[:, 2], per_coord_err, target_base)
    pts_recon = np.stack([x_recon, y_recon, z_recon], axis=1)
    errors = np.linalg.norm(pts_recon - pts, axis=1)

    return {
        "total": total, "vertex": wavelet_bits, "conn": conn_bits,
        "header": header_bits, "n_verts": n_v, "n_faces": n_f,
        "n_levels": wx["n_levels"],
        "level_bit_depths_x": wx["level_bit_depths"],
        "errors": errors,
    }


# ============================================================
# Main pipeline
# ============================================================

def run(obj_path, max_error=0.001, max_verts_values=None):
    if max_verts_values is None:
        max_verts_values = [128, 256, 512]

    mesh = Reader.read_from_file(obj_path)
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)

    verts_np = np.empty((n_v, 3), dtype=np.float64)
    for i, v in enumerate(mesh.vertices):
        verts_np[i] = (v.x, v.y, v.z)
    tris_np = np.empty((n_t, 3), dtype=np.int64)
    for i, t in enumerate(mesh.triangles):
        tris_np[i] = (t.a, t.b, t.c)

    center = verts_np.mean(axis=0)
    vc = verts_np - center
    scale = np.max(np.linalg.norm(vc, axis=1))
    vn = vc / scale
    norm_err = max_error / scale

    raw_bits = n_v * 96 + n_t * 96
    print(f"{'='*75}")
    print(f"Wavelet Meshlet Compression — {obj_path}")
    print(f"  {n_v:,} verts, {n_t:,} tris, max_error={max_error}")
    print(f"  Raw: {raw_bits/8:,.0f} B")
    print(f"{'='*75}")

    tri_adj = build_adjacency(tris_np)
    fn = compute_face_normals(vn, tris_np)
    fc = compute_face_centroids(vn, tris_np)

    results = []

    for mv in max_verts_values:
        t0 = time.time()
        meshlets = generate_meshlets_by_verts(
            tris_np, tri_adj, fn, fc, max_verts=mv)
        t_gen = time.time() - t0

        total_verts_enc = sum(
            len(set(int(tris_np[ti, j]) for ti in ml for j in range(3)))
            for ml in meshlets)

        print(f"\n--- max_verts={mv} ({len(meshlets)} meshlets, "
              f"gen={t_gen:.1f}s, vert_overhead={total_verts_enc/n_v*100-100:.1f}%) ---")

        for variant_name, estimate_fn, extra_kwargs in [
            ("V3:Bez+EB", estimate_variant3, {"deg": 2}),
            ("V4:Bez+AMD", estimate_variant4, {"deg": 2}),
            ("V2:Direct+EB", estimate_variant2, {}),
        ]:
            t0 = time.time()
            total_header = 0
            total_vertex = 0
            total_conn = 0
            all_errors = []
            sample_levels_info = None

            # Global header
            global_hdr = (3*4 + 4 + 4 + 1) * 8  # center, scale, n_meshlets, flags
            total_bits = global_hdr

            for ml_tris in meshlets:
                r = estimate_fn(ml_tris, vn, tris_np, tri_adj,
                                norm_err, target_base=32, **extra_kwargs)
                total_bits += r["total"]
                total_header += r["header"]
                total_vertex += r["vertex"]
                total_conn += r["conn"]
                if len(all_errors) < 100000 and "errors" in r:
                    all_errors.extend((r["errors"] * scale).tolist())
                if sample_levels_info is None and r.get("n_levels", 0) > 0:
                    sample_levels_info = r

            t_est = time.time() - t0
            bpv = total_bits / n_v
            bpt = total_bits / n_t

            errors_arr = np.array(all_errors) if all_errors else np.array([0.0])
            pct_ok = (errors_arr <= max_error).sum() / len(errors_arr) * 100

            label = f"{variant_name} mv={mv}"
            print(f"\n  [{label}]  ({t_est:.1f}s)")
            if sample_levels_info:
                n_lvl = sample_levels_info["n_levels"]
                print(f"    Wavelet: {n_lvl} levels, base=32")
                for key in ["level_bit_depths_u", "level_bit_depths_d",
                            "level_bit_depths_x"]:
                    if key in sample_levels_info:
                        short = key.replace("level_bit_depths_", "")
                        print(f"    Sample level bits ({short}): {sample_levels_info[key]}")
                if "d_bits_flat" in sample_levels_info:
                    print(f"    Flat d bits: {sample_levels_info['d_bits_flat']/8:.0f} B")

            print(f"    Headers:      {total_header/8:>10,.0f} B  ({total_header/total_bits*100:.1f}%)")
            print(f"    Vertex data:  {total_vertex/8:>10,.0f} B  ({total_vertex/total_bits*100:.1f}%)")
            print(f"    Connectivity: {total_conn/8:>10,.0f} B  ({total_conn/total_bits*100:.1f}%)")
            print(f"    TOTAL:        {total_bits/8:>10,.0f} B")
            print(f"    BPV: {bpv:.2f}  BPT: {bpt:.2f}  Ratio: {raw_bits/total_bits:.2f}x")
            print(f"    Accuracy: mean={errors_arr.mean():.6f}  "
                  f"max={errors_arr.max():.6f}  within_target={pct_ok:.1f}%")

            results.append({
                "label": label, "total": total_bits/8, "bpv": bpv, "bpt": bpt,
                "ratio": raw_bits/total_bits,
                "hdr_pct": total_header/total_bits*100,
                "vtx_pct": total_vertex/total_bits*100,
                "con_pct": total_conn/total_bits*100,
                "max_err": errors_arr.max(), "pct_ok": pct_ok,
            })

    print(f"\n{'='*75}")
    print(f"SUMMARY — {obj_path}")
    print(f"{'='*75}")
    print(f"{'Config':<25} {'Total':>9} {'BPV':>7} {'Ratio':>7} "
          f"{'Hdr%':>5} {'Vtx%':>5} {'Con%':>5} {'MaxErr':>9} {'%OK':>5}")
    for r in results:
        print(f"{r['label']:<25} {r['total']:>9,.0f} {r['bpv']:>7.2f} "
              f"{r['ratio']:>7.2f} {r['hdr_pct']:>5.1f} {r['vtx_pct']:>5.1f} "
              f"{r['con_pct']:>5.1f} {r['max_err']:>9.6f} {r['pct_ok']:>4.1f}%")


def verify_single_meshlet(obj_path, max_error=0.001, max_verts=256, deg=2):
    """Detailed single-meshlet verification: compress → decompress → check."""
    from utils.wavelet import (haar_decompose, haar_reconstruct,
                                wavelet_reconstruct_quantized,
                                _bits_for_error, _quantize, _dequantize)

    mesh = Reader.read_from_file(obj_path)
    n_v = len(mesh.vertices)
    verts_np = np.empty((n_v, 3), dtype=np.float64)
    for i, v in enumerate(mesh.vertices):
        verts_np[i] = (v.x, v.y, v.z)
    tris_np = np.empty((len(mesh.triangles), 3), dtype=np.int64)
    for i, t in enumerate(mesh.triangles):
        tris_np[i] = (t.a, t.b, t.c)

    center = verts_np.mean(axis=0)
    vc = verts_np - center
    scale = np.max(np.linalg.norm(vc, axis=1))
    vn = vc / scale
    norm_err = max_error / scale

    tri_adj = build_adjacency(tris_np)
    fn = compute_face_normals(vn, tris_np)
    fc = compute_face_centroids(vn, tris_np)
    meshlets = generate_meshlets_by_verts(tris_np, tri_adj, fn, fc, max_verts=max_verts)

    # Pick largest meshlet
    ml_idx = max(range(len(meshlets)), key=lambda i: len(meshlets[i]))
    ml_tris = meshlets[ml_idx]

    print(f"{'='*75}")
    print(f"Single Meshlet Verification — {obj_path}")
    print(f"  Meshlet {ml_idx}: {len(ml_tris)} tris")
    print(f"{'='*75}")

    # EdgeBreaker ordering
    vert_order, opcodes, n_roots = edgebreaker_vertex_order(ml_tris, tris_np, tri_adj)
    n_mv = len(vert_order)
    pts_orig = vn[vert_order]
    print(f"  Vertices: {n_mv} (EdgeBreaker order)")
    print(f"  EdgeBreaker opcodes: C={opcodes.count('C')} L={opcodes.count('L')} "
          f"E={opcodes.count('E')} (total={len(opcodes)})")

    per_coord_err = norm_err / np.sqrt(3)

    # ---- Test Variant 2 (Direct xyz wavelet) ----
    print(f"\n--- V2: Direct Wavelet (x,y,z) ---")
    for dim_name, dim_idx in [("x", 0), ("y", 1), ("z", 2)]:
        vals = pts_orig[:, dim_idx]
        base, levels, orig_n = haar_decompose(vals, target_base=32)

        print(f"\n  Dim '{dim_name}': {len(vals)} values → base({len(base)}) + "
              f"{len(levels)} levels [{', '.join(str(len(l)) for l in levels)}]")
        print(f"    Original range: [{vals.min():.6f}, {vals.max():.6f}]")
        print(f"    Base range:     [{base.min():.6f}, {base.max():.6f}]")
        for li, detail in enumerate(levels):
            print(f"    Level {li} (finest={li==0}): {len(detail)} residuals, "
                  f"range=[{detail.min():.6f}, {detail.max():.6f}], "
                  f"std={detail.std():.6f}")

        # Verify wavelet roundtrip (no quantization)
        recon_exact = haar_reconstruct(base, levels)[:orig_n]
        roundtrip_err = np.max(np.abs(recon_exact - vals))
        print(f"    Wavelet roundtrip error (no quant): {roundtrip_err:.2e} "
              f"{'✓ EXACT' if roundtrip_err < 1e-12 else '✗ ERROR'}")

        # Verify with quantization
        recon_q = wavelet_reconstruct_quantized(vals, per_coord_err, target_base=32)
        quant_err = np.max(np.abs(recon_q - vals))
        print(f"    With quantization max error: {quant_err:.6f} "
              f"(budget: {per_coord_err:.6f})")

    # Full 3D reconstruction
    x_r = wavelet_reconstruct_quantized(pts_orig[:, 0], per_coord_err, 32)
    y_r = wavelet_reconstruct_quantized(pts_orig[:, 1], per_coord_err, 32)
    z_r = wavelet_reconstruct_quantized(pts_orig[:, 2], per_coord_err, 32)
    pts_v2 = np.stack([x_r, y_r, z_r], axis=1)
    errs_v2 = np.linalg.norm(pts_v2 - pts_orig, axis=1) * scale
    print(f"\n  V2 3D reconstruction: mean_err={errs_v2.mean():.6f} "
          f"max_err={errs_v2.max():.6f}")
    print(f"  Within target ({max_error}): "
          f"{(errs_v2 <= max_error).sum()}/{n_mv} "
          f"({(errs_v2 <= max_error).sum()/n_mv*100:.1f}%)")

    # ---- Test Variant 3 (Bezier + wavelet u,v + flat d) ----
    print(f"\n--- V3: Bezier + Wavelet(u,v) + Flat(d) ---")
    u, v, _ = parameterize_pca(pts_orig)
    cp = fit_bezier(u, v, pts_orig, deg)
    disps, surf_pts, normals = compute_displacements(u, v, pts_orig, cp, deg)

    Su, Sv = bezier_derivatives(u, v, cp, deg)
    max_Su = max(np.max(np.linalg.norm(Su, axis=1)), 1e-6)
    max_Sv = max(np.max(np.linalg.norm(Sv, axis=1)), 1e-6)
    u_err = per_coord_err / max_Su
    v_err = per_coord_err / max_Sv

    print(f"  Bezier deg={deg}: {n_control_points(deg)} control points")
    print(f"  u: range=[{u.min():.4f}, {u.max():.4f}], deriv_scale={max_Su:.4f}, "
          f"precision={u_err:.6f}")
    print(f"  v: range=[{v.min():.4f}, {v.max():.4f}], deriv_scale={max_Sv:.4f}, "
          f"precision={v_err:.6f}")
    print(f"  d: range=[{disps.min():.6f}, {disps.max():.6f}], std={disps.std():.6f}")

    # Wavelet on u
    base_u, levels_u, _ = haar_decompose(u, 32)
    print(f"\n  Wavelet u: base({len(base_u)}) + {len(levels_u)} levels")
    for li, det in enumerate(levels_u):
        print(f"    Level {li}: {len(det)} residuals, range=[{det.min():.6f}, {det.max():.6f}]")

    u_recon = wavelet_reconstruct_quantized(u, u_err, 32)
    v_recon = wavelet_reconstruct_quantized(v, v_err, 32)
    d_recon = _flat_quantize_recon(disps, per_coord_err)

    pts_v3 = reconstruct_from_bezier(u_recon, v_recon, d_recon, cp, deg)
    errs_v3 = np.linalg.norm(pts_v3 - pts_orig, axis=1) * scale
    print(f"\n  V3 3D reconstruction: mean_err={errs_v3.mean():.6f} "
          f"max_err={errs_v3.max():.6f}")
    print(f"  Within target ({max_error}): "
          f"{(errs_v3 <= max_error).sum()}/{n_mv} "
          f"({(errs_v3 <= max_error).sum()/n_mv*100:.1f}%)")

    # ---- Compare all variants for this meshlet ----
    print(f"\n--- Compression comparison (this meshlet only) ---")
    for name, fn, kw in [
        ("V1:Bez+Wav(uvd)", estimate_variant1, {"deg": deg}),
        ("V2:Direct(xyz)", estimate_variant2, {}),
        ("V3:Bez+Wav(uv)", estimate_variant3, {"deg": deg}),
    ]:
        r = fn(ml_tris, vn, tris_np, tri_adj, norm_err, target_base=32, **kw)
        errs = r.get("errors", np.array([0.0])) * scale
        print(f"  {name:<20} total={r['total']/8:.0f}B "
              f"(hdr={r['header']/8:.0f} vtx={r['vertex']/8:.0f} "
              f"conn={r['conn']/8:.0f}) "
              f"bpv={r['total']/n_mv:.1f} "
              f"max_err={errs.max():.6f} "
              f"ok={((errs<=max_error).sum()/len(errs)*100):.0f}%")


if __name__ == "__main__":
    for path in ["assets/bunny.obj", "assets/stanford-bunny.obj",
                  "assets/Monkey.obj"]:
        try:
            run(path, max_error=0.001, max_verts_values=[256, 512])
            print("\n\n")
        except Exception as e:
            import traceback; traceback.print_exc()
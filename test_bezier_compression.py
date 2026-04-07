"""
Bezier Meshlet Compression: per-meshlet Bezier surface fitting with
displacement encoding + triangle strip connectivity.

Each meshlet is a self-contained unit:
  [Bezier control points] + [u, v, displacement streams] + [strip connectivity]
"""

import numpy as np
import time
from collections import Counter
from reader import Reader
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_greedy, meshlet_bfs,
)
from utils.bezier import (
    fit_bezier, evaluate_bezier, bezier_normals, bezier_derivatives,
    compute_displacements, reconstruct_from_bezier,
    parameterize_pca, n_control_points,
)


# ============================================================
# Quantization & entropy
# ============================================================

def quantize(vals, lo, hi, bits):
    mx = (1 << bits) - 1
    norm = np.clip((vals - lo) / (hi - lo + 1e-15), 0, 1)
    return np.round(norm * mx).astype(np.int64)

def dequantize(codes, lo, hi, bits):
    return codes.astype(np.float64) / ((1 << bits) - 1) * (hi - lo) + lo

def bits_for_error(val_range, max_err):
    if max_err <= 0 or val_range <= 0:
        return 1
    return max(1, int(np.ceil(np.log2(val_range / (2 * max_err) + 1))))

def shannon_entropy(codes):
    if len(codes) == 0:
        return 0.0
    counts = Counter(codes.tolist() if hasattr(codes, 'tolist') else list(codes))
    total = len(codes)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())

def stream_bits(codes, fixed_bits):
    n = len(codes)
    if n == 0:
        return 0
    plain = n * fixed_bits
    ent = shannon_entropy(codes)
    arith = n * ent + 32
    return min(plain, arith)


# ============================================================
# Connectivity encoding within meshlet (3 methods)
# ============================================================

def _edgebreaker_encode(meshlet_tris, tris_np, tri_adj):
    """EdgeBreaker encoding: classify each triangle as C/L/R/S/E.
    Returns list of opcodes ('C','L','R','S','E')."""
    ml_set = set(meshlet_tris)

    # Build local vertex set and adjacency
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))

    # Build half-edge-like boundary tracking
    # Use corner table approach: for each triangle, track which vertices are visited
    visited_tris = set()
    visited_verts = set()
    boundary = []  # ordered list of boundary vertices (active boundary loop)
    opcodes = []

    # Find a seed triangle (prefer one at the edge of the meshlet)
    seed = meshlet_tris[0]
    visited_tris.add(seed)
    tri_v = [int(tris_np[seed, j]) for j in range(3)]
    for v in tri_v:
        visited_verts.add(v)
    # Initial boundary: the 3 edges of the seed triangle
    boundary = list(tri_v)  # clockwise or counterclockwise loop

    # BFS-based traversal, classifying each triangle
    stack = []
    gate_idx = 0  # index into boundary for the current gate edge

    def get_gate():
        """Get the two vertices of the current gate edge."""
        n = len(boundary)
        if n < 2:
            return None, None
        return boundary[gate_idx % n], boundary[(gate_idx + 1) % n]

    # Simple EdgeBreaker simulation using BFS ordering
    # For estimation, we classify based on the BFS traversal
    traversal = meshlet_bfs(meshlet_tris, tri_adj)

    visited_verts2 = set()
    for tri_idx, parent_idx in traversal:
        tri_v = [int(tris_np[tri_idx, j]) for j in range(3)]
        if parent_idx is None:
            # Root triangle — not an opcode, just initialization
            for v in tri_v:
                visited_verts2.add(v)
            continue

        parent_v = [int(tris_np[parent_idx, j]) for j in range(3)]
        shared = set(tri_v) & set(parent_v)
        new_verts = [v for v in tri_v if v not in shared]

        if len(new_verts) == 1 and new_verts[0] not in visited_verts2:
            # New vertex: C opcode (most common)
            opcodes.append('C')
            visited_verts2.add(new_verts[0])
        elif len(new_verts) == 1 and new_verts[0] in visited_verts2:
            # Vertex already visited — could be L, R, S, or E
            # Simplified: classify as L/R (most common for boundary vertices)
            # In a real implementation, we'd check the boundary position
            opcodes.append('L')  # approximate
        elif len(new_verts) == 0:
            # All vertices already visited — E (closing a hole)
            opcodes.append('E')
        else:
            opcodes.append('C')
            for v in new_verts:
                visited_verts2.add(v)

    return opcodes


def estimate_edgebreaker_bits(meshlet_tris, tris_np, tri_adj):
    """Estimate bits using EdgeBreaker encoding."""
    opcodes = _edgebreaker_encode(meshlet_tris, tris_np, tri_adj)
    n_faces = len(meshlet_tris)

    if n_faces == 0:
        return 0, 0

    # Count opcode frequencies
    counts = Counter(opcodes)
    n_ops = len(opcodes)

    # Entropy-coded opcodes (variable-length prefix codes)
    # C≈60-70% → ~1 bit, L/R≈10-15% → ~3 bits, S/E≈1-5% → ~4 bits
    # Use actual entropy for estimation
    if n_ops > 0:
        total_opcodes = sum(counts.values())
        ent = -sum((c / total_opcodes) * np.log2(c / total_opcodes)
                    for c in counts.values() if c > 0)
        opcode_bits = n_ops * ent + 16  # 16 bits overhead
    else:
        opcode_bits = 0

    # Root triangle: 3 vertex indices (not encoded by opcodes)
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    n_local = len(vert_set)
    idx_bits = max(1, int(np.ceil(np.log2(n_local + 1))))
    root_bits = 3 * idx_bits

    # S opcodes need an offset to identify the boundary vertex
    n_s = counts.get('S', 0)
    s_extra = n_s * idx_bits  # worst case

    # Header
    header_bits = 32  # n_verts + n_faces

    return int(header_bits + root_bits + opcode_bits + s_extra), n_faces


def estimate_amd_strip_bits(meshlet_tris, tris_np, tri_adj):
    """AMD-style meshlet encoding: local indices with FIFO reuse buffer.
    Triangle strip with: new_vertex_flag(1 bit) + local_index or FIFO_index."""
    ml_set = set(meshlet_tris)
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    n_local = len(vert_set)
    n_faces = len(meshlet_tris)
    if n_faces == 0:
        return 0, 0

    # BFS traversal
    traversal = meshlet_bfs(meshlet_tris, tri_adj)

    # Track vertex ordering and FIFO buffer
    vert_order = {}  # global_idx -> local_idx (order of first appearance)
    fifo = []  # recent vertex FIFO (size 16-32)
    fifo_size = min(32, n_local)

    total_bits = 32  # header: n_verts(16) + n_faces(16)
    idx_bits = max(1, int(np.ceil(np.log2(n_local + 1))))
    fifo_bits = max(1, int(np.ceil(np.log2(fifo_size + 1))))

    for tri_idx, parent_idx in traversal:
        tri_v = [int(tris_np[tri_idx, j]) for j in range(3)]

        if parent_idx is None:
            # Root: store 3 full local indices
            for v in tri_v:
                if v not in vert_order:
                    vert_order[v] = len(vert_order)
                    if len(fifo) >= fifo_size:
                        fifo.pop(0)
                    fifo.append(v)
                total_bits += idx_bits
        else:
            parent_v = [int(tris_np[parent_idx, j]) for j in range(3)]
            shared = set(tri_v) & set(parent_v)
            new_verts = [v for v in tri_v if v not in shared]

            # Shared edge: implicit from parent (2 bits for which edge)
            total_bits += 2

            for v in new_verts:
                if v in vert_order:
                    # Reuse: check FIFO
                    if v in fifo:
                        total_bits += 1 + fifo_bits  # flag(1) + fifo_index
                    else:
                        total_bits += 1 + idx_bits  # flag(1) + full local index
                else:
                    # New vertex: just flag (index is implicit = next in order)
                    total_bits += 1
                    vert_order[v] = len(vert_order)

                # Update FIFO
                if v in fifo:
                    fifo.remove(v)
                fifo.append(v)
                if len(fifo) > fifo_size:
                    fifo.pop(0)

    return total_bits, n_faces


def estimate_connectivity_bits(meshlet_tris, tris_np, tri_adj, method="best"):
    """Estimate connectivity bits using the best of available methods."""
    eb, _ = estimate_edgebreaker_bits(meshlet_tris, tris_np, tri_adj)
    amd, _ = estimate_amd_strip_bits(meshlet_tris, tris_np, tri_adj)

    if method == "edgebreaker":
        return eb, len(meshlet_tris)
    elif method == "amd":
        return amd, len(meshlet_tris)
    else:  # "best"
        return min(eb, amd), len(meshlet_tris)


# ============================================================
# Per-meshlet compression estimation
# ============================================================

def estimate_meshlet(meshlet_tris, verts, tris_np, tri_adj,
                     max_error, deg, cp_bits=16):
    """
    Estimate total bits for one meshlet: header + vertex data + connectivity.
    Returns dict with breakdown.
    """
    # Collect meshlet vertices
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    local_verts = sorted(vert_set)
    n_v = len(local_verts)
    n_f = len(meshlet_tris)

    if n_v < 3:
        # Degenerate meshlet, store raw
        return {"total": n_v * 96, "n_verts": n_v, "n_faces": n_f,
                "header": 0, "vertex": n_v * 96, "connectivity": 0,
                "disp_std": 0, "disp_range": 0}

    pts = verts[local_verts]  # (n_v, 3)

    # 1. Parameterize
    u, v, pca_frame = parameterize_pca(pts)

    # 2. Fit Bezier surface
    cp = fit_bezier(u, v, pts, deg)

    # 3. Compute displacements
    disps, surf_pts, normals = compute_displacements(u, v, pts, cp, deg)

    # 4. Header: control points (quantized)
    n_cp = n_control_points(deg)
    # Control points: each coord quantized to cp_bits
    # Also store PCA frame: center(3f) + u_range(2f) + v_range(2f) = 7 floats
    header_bits = n_cp * 3 * cp_bits + 7 * 32

    # 5. Vertex data: quantize u, v, d
    # Key insight: quantization error in u,v maps to world-space error
    # through the Bezier surface derivative: world_err ≈ Δu × ||∂S/∂u||
    # So: Δu ≤ max_error / (√3 × max||∂S/∂u||)
    per_coord_err = max_error / np.sqrt(3)

    Su, Sv = bezier_derivatives(u, v, cp, deg)
    max_Su = np.max(np.linalg.norm(Su, axis=1)) if n_v > 0 else 1.0
    max_Sv = np.max(np.linalg.norm(Sv, axis=1)) if n_v > 0 else 1.0

    # u: allowed quantization step = per_coord_err / max_Su
    u_range = u.max() - u.min()
    u_precision = per_coord_err / max(max_Su, 1e-6)
    bits_u = bits_for_error(u_range, u_precision)
    codes_u = quantize(u, u.min(), u.max(), bits_u)
    u_bits = stream_bits(codes_u, bits_u)

    # v: allowed quantization step = per_coord_err / max_Sv
    v_range = v.max() - v.min()
    v_precision = per_coord_err / max(max_Sv, 1e-6)
    bits_v = bits_for_error(v_range, v_precision)
    codes_v = quantize(v, v.min(), v.max(), bits_v)
    v_bits = stream_bits(codes_v, bits_v)

    # displacement: direct world-space error, no derivative scaling needed
    d_range = disps.max() - disps.min() if n_v > 1 else 0.001
    bits_d = bits_for_error(d_range, per_coord_err)
    codes_d = quantize(disps, disps.min(), disps.max(), bits_d)
    d_bits = stream_bits(codes_d, bits_d)

    # Per-stream ranges (min/max): 2 floats each × 3 streams = 6 × 32 bits
    # Plus bit counts: 3 bytes
    vertex_meta = (6 * 4 + 3) * 8
    vertex_bits = vertex_meta + u_bits + v_bits + d_bits

    # 6. Connectivity (best of EdgeBreaker vs AMD-style)
    conn_bits, _ = estimate_connectivity_bits(meshlet_tris, tris_np, tri_adj)

    total = header_bits + vertex_bits + conn_bits

    return {
        "total": total,
        "n_verts": n_v,
        "n_faces": n_f,
        "header": header_bits,
        "vertex": vertex_bits,
        "connectivity": conn_bits,
        "disp_std": disps.std(),
        "disp_range": d_range,
        "bits_u": bits_u, "bits_v": bits_v, "bits_d": bits_d,
        "codes_u": codes_u, "codes_v": codes_v, "codes_d": codes_d,
    }


# ============================================================
# Accuracy measurement
# ============================================================

def measure_accuracy_meshlet(meshlet_tris, verts, tris_np, deg, max_error):
    """Simulate quantize → dequantize → reconstruct for one meshlet."""
    vert_set = set()
    for ti in meshlet_tris:
        for j in range(3):
            vert_set.add(int(tris_np[ti, j]))
    local_verts = sorted(vert_set)
    n_v = len(local_verts)
    if n_v < 3:
        return np.zeros(n_v)

    pts = verts[local_verts]
    per_coord_err = max_error / np.sqrt(3)

    # Parameterize & fit
    u, v, _ = parameterize_pca(pts)
    cp = fit_bezier(u, v, pts, deg)
    disps, _, _ = compute_displacements(u, v, pts, cp, deg)

    # Derivative-aware precision
    Su, Sv = bezier_derivatives(u, v, cp, deg)
    max_Su = np.max(np.linalg.norm(Su, axis=1)) if n_v > 0 else 1.0
    max_Sv = np.max(np.linalg.norm(Sv, axis=1)) if n_v > 0 else 1.0

    u_precision = per_coord_err / max(max_Su, 1e-6)
    v_precision = per_coord_err / max(max_Sv, 1e-6)

    u_rng = u.max() - u.min()
    bits_u = bits_for_error(u_rng, u_precision)
    u_r = dequantize(quantize(u, u.min(), u.max(), bits_u), u.min(), u.max(), bits_u)

    v_rng = v.max() - v.min()
    bits_v = bits_for_error(v_rng, v_precision)
    v_r = dequantize(quantize(v, v.min(), v.max(), bits_v), v.min(), v.max(), bits_v)

    d_rng = disps.max() - disps.min() if n_v > 1 else 0.001
    bits_d = bits_for_error(d_rng, per_coord_err)
    d_r = dequantize(quantize(disps, disps.min(), disps.max(), bits_d),
                     disps.min(), disps.max(), bits_d)

    # Reconstruct
    recon = reconstruct_from_bezier(u_r, v_r, d_r, cp, deg)
    errors = np.linalg.norm(recon - pts, axis=1)
    return errors


# ============================================================
# Main pipeline
# ============================================================

def run(obj_path, max_error=0.001, deg_values=None, max_tris_values=None):
    if deg_values is None:
        deg_values = [2, 3]
    if max_tris_values is None:
        max_tris_values = [128, 256]

    mesh = Reader.read_from_file(obj_path)
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)

    verts_np = np.empty((n_v, 3), dtype=np.float64)
    for i, v in enumerate(mesh.vertices):
        verts_np[i] = (v.x, v.y, v.z)
    tris_np = np.empty((n_t, 3), dtype=np.int64)
    for i, t in enumerate(mesh.triangles):
        tris_np[i] = (t.a, t.b, t.c)

    # Normalize
    center = verts_np.mean(axis=0)
    vc = verts_np - center
    scale = np.max(np.linalg.norm(vc, axis=1))
    vn = vc / scale
    norm_err = max_error / scale

    raw_bits = n_v * 3 * 32 + n_t * 3 * 32  # raw vertices + raw triangles
    raw_v_bits = n_v * 3 * 32
    raw_t_bits = n_t * 3 * 32

    print(f"{'='*70}")
    print(f"Bezier Meshlet Compression — {obj_path}")
    print(f"  {n_v:,} verts, {n_t:,} tris, max_error={max_error}")
    print(f"  Raw: {raw_bits/8:,.0f} B (verts: {raw_v_bits/8:,.0f} + tris: {raw_t_bits/8:,.0f})")
    print(f"{'='*70}")

    # Build adjacency
    t0 = time.time()
    tri_adj = build_adjacency(tris_np)
    face_normals = compute_face_normals(vn, tris_np)
    face_centroids = compute_face_centroids(vn, tris_np)
    print(f"  Adjacency: {time.time()-t0:.1f}s")

    results = []

    for max_tris in max_tris_values:
        # Generate meshlets
        t0 = time.time()
        meshlets = generate_meshlets_greedy(
            tris_np, tri_adj, face_normals, face_centroids,
            max_tris=max_tris, max_verts=max_tris * 3)
        t_gen = time.time() - t0

        for deg in deg_values:
            label = f"deg={deg} mt={max_tris}"
            print(f"\n--- {label} ({len(meshlets)} meshlets, gen={t_gen:.1f}s) ---")
            t0 = time.time()

            total_header = 0
            total_vertex = 0
            total_conn = 0
            total_verts_encoded = 0
            total_faces_encoded = 0
            disp_stds = []
            all_errors = []

            # Global header: center(3f) + scale(1f) + n_meshlets(4B) + deg(1B)
            global_header = (3*4 + 4 + 4 + 1) * 8
            total_bits = global_header

            # Collect global code streams for cross-meshlet entropy analysis
            global_codes_u = []
            global_codes_v = []
            global_codes_d = []

            for ml_tris in meshlets:
                r = estimate_meshlet(ml_tris, vn, tris_np, tri_adj,
                                     norm_err, deg, cp_bits=16)
                total_bits += r["total"]
                total_header += r["header"]
                total_vertex += r["vertex"]
                total_conn += r["connectivity"]
                total_verts_encoded += r["n_verts"]
                total_faces_encoded += r["n_faces"]
                if r["disp_std"] > 0:
                    disp_stds.append(r["disp_std"])
                # Collect codes for global entropy analysis
                if "codes_u" in r:
                    global_codes_u.extend(r["codes_u"].tolist())
                    global_codes_v.extend(r["codes_v"].tolist())
                    global_codes_d.extend(r["codes_d"].tolist())

                # Accuracy for sample of meshlets
                if len(all_errors) < 50000:
                    errs = measure_accuracy_meshlet(ml_tris, vn, tris_np, deg, norm_err)
                    all_errors.extend(errs.tolist())

            t_total = time.time() - t0
            bpv = total_bits / n_v
            bpt = total_bits / n_t
            avg_disp_std = np.mean(disp_stds) if disp_stds else 0

            errors_arr = np.array(all_errors)
            # Scale errors back to world space
            errors_world = errors_arr * scale
            pct_ok = (errors_world <= max_error).sum() / len(errors_world) * 100 if len(errors_world) > 0 else 0

            # Global entropy analysis: if we concatenated all meshlet codes
            # into single streams, would entropy coding help?
            if global_codes_u:
                gu = np.array(global_codes_u, dtype=np.int64)
                gv = np.array(global_codes_v, dtype=np.int64)
                gd = np.array(global_codes_d, dtype=np.int64)
                ent_u = shannon_entropy(gu)
                ent_v = shannon_entropy(gv)
                ent_d = shannon_entropy(gd)
                n_total_codes = len(gu)
                global_arith = n_total_codes * (ent_u + ent_v + ent_d) + 3 * 32
                # Compare to per-meshlet plain vertex bits (no headers/meta)
                print(f"  Global entropy analysis ({n_total_codes:,} total codes):")
                print(f"    u: {ent_u:.2f} bps  v: {ent_v:.2f} bps  d: {ent_d:.2f} bps")
                print(f"    Global arith total: {global_arith/8:,.0f} B "
                      f"({global_arith/n_total_codes:.1f} bits/vertex)")
                print(f"    Per-meshlet plain:  {total_vertex/8:,.0f} B "
                      f"({total_vertex/total_verts_encoded:.1f} bits/vertex)")
                saving = (1 - global_arith / total_vertex) * 100
                print(f"    Global entropy saving: {saving:.1f}%"
                      f" {'← helps!' if saving > 0 else ''}")
                print()

            # Print per-meshlet breakdown
            n_cp = n_control_points(deg)
            print(f"  Control points per meshlet: {n_cp} ({n_cp*3*2} B at 16-bit)")
            print(f"  Avg displacement std: {avg_disp_std:.6f}")
            print(f"  Verts encoded: {total_verts_encoded:,} "
                  f"(overhead: {total_verts_encoded/n_v*100-100:.1f}% from shared)")
            print(f"  Faces encoded: {total_faces_encoded:,}")
            print()
            print(f"  Breakdown:")
            print(f"    Headers:      {total_header/8:>10,.0f} B  "
                  f"({total_header/total_bits*100:.1f}%)")
            print(f"    Vertex data:  {total_vertex/8:>10,.0f} B  "
                  f"({total_vertex/total_bits*100:.1f}%)")
            print(f"    Connectivity: {total_conn/8:>10,.0f} B  "
                  f"({total_conn/total_bits*100:.1f}%)")
            print(f"    TOTAL:        {total_bits/8:>10,.0f} B")
            print()
            print(f"  BPV: {bpv:.2f}  BPT: {bpt:.2f}  "
                  f"Ratio: {raw_bits/total_bits:.2f}x  [{t_total:.1f}s]")
            if len(errors_world) > 0:
                print(f"  Accuracy: mean={errors_world.mean():.6f}  "
                      f"max={errors_world.max():.6f}  within_target={pct_ok:.1f}%")

            results.append({
                "label": label, "deg": deg, "mt": max_tris,
                "total_bytes": total_bits / 8, "bpv": bpv, "bpt": bpt,
                "ratio": raw_bits / total_bits,
                "header_pct": total_header / total_bits * 100,
                "vertex_pct": total_vertex / total_bits * 100,
                "conn_pct": total_conn / total_bits * 100,
                "disp_std": avg_disp_std,
                "max_err": errors_world.max() if len(errors_world) > 0 else 0,
                "pct_ok": pct_ok,
            })

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — {obj_path}")
    print(f"{'='*70}")
    print(f"{'Config':<16} {'Total':>10} {'BPV':>7} {'BPT':>7} {'Ratio':>7} "
          f"{'Hdr%':>5} {'Vtx%':>5} {'Con%':>5} {'MaxErr':>9} {'%OK':>5}")
    for r in results:
        print(f"{r['label']:<16} {r['total_bytes']:>10,.0f} {r['bpv']:>7.2f} "
              f"{r['bpt']:>7.2f} {r['ratio']:>7.2f} "
              f"{r['header_pct']:>5.1f} {r['vertex_pct']:>5.1f} "
              f"{r['conn_pct']:>5.1f} {r['max_err']:>9.6f} {r['pct_ok']:>4.1f}%")


if __name__ == "__main__":
    for path in ["assets/bunny.obj", "assets/torus.obj",
                  "assets/stanford-bunny.obj"]:
        try:
            run(path, max_error=0.001, deg_values=[2, 3], max_tris_values=[128, 256])
            print("\n\n")
        except Exception as e:
            import traceback; traceback.print_exc()
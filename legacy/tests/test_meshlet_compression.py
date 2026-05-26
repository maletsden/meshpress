"""
Meshlet-based mesh compression estimation with parallelogram prediction.
Tests two meshlet generation methods:
  Option 3: Greedy DFS strips, cut into meshlets
  Option 4: Greedy region growing with geometric cost
Compression pipeline:
  1. Generate meshlets
  2. BFS within each meshlet for traversal order
  3. Parallelogram prediction: root triangle direct, rest predicted
  4. Global streams: direct coords + delta coords
  5. Quantize, estimate entropy, measure accuracy with error propagation
"""

import numpy as np
import time
from collections import Counter
from reader import Reader
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_greedy, generate_strips_greedy, meshlets_from_strips,
    meshlet_bfs,
)





# ============================================================
# Quantization & entropy (shared with C3)
# ============================================================

def quantize(vals, lo, hi, bits):
    mx = (1 << bits) - 1
    norm = np.clip((vals - lo) / (hi - lo + 1e-15), 0, 1)
    return np.round(norm * mx).astype(np.int64)


def dequantize(codes, lo, hi, bits):
    return codes.astype(np.float64) / ((1 << bits) - 1) * (hi - lo) + lo


def quantize_scalar(val, lo, hi, bits):
    mx = (1 << bits) - 1
    norm = np.clip((val - lo) / (hi - lo + 1e-15), 0, 1)
    code = int(round(norm * mx))
    return code


def dequantize_scalar(code, lo, hi, bits):
    return float(code) / ((1 << bits) - 1) * (hi - lo) + lo


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


def tree_entropy_1d(vals, codes, bits, max_d=12, min_leaf=4):
    """Tree partitioning entropy: split on value medians, compute actual
    code entropy at each leaf. Tree benefit = within-leaf codes are more
    uniform than the global distribution."""
    hdr = [0.0]; pay = [0.0]
    def go(idx, d):
        n_ = len(idx)
        if n_ == 0: return
        if d >= max_d or n_ <= min_leaf:
            # Use FULL codes — no modulo. The tree path doesn't encode bits.
            pay[0] += n_ * shannon_entropy(codes[idx]); return
        hdr[0] += np.ceil(np.log2(n_ + 1))
        v_ = vals[idx]; med = np.median(v_); m = v_ < med
        if m.sum() == 0 or m.sum() == n_:
            pay[0] += n_ * shannon_entropy(codes[idx]); return
        go(idx[m], d + 1); go(idx[~m], d + 1)
    go(np.arange(len(codes)), 0)
    return hdr[0] + pay[0]


def best_stream_bits(vals, fixed_bits, max_error_per_coord):
    """Quantize a 1D stream and return best of plain/arith.
    NOTE: tree partitioning is NOT valid here because the delta stream
    must be decoded in BFS traversal order (parallelogram prediction
    depends on previously reconstructed vertices). Tree would reorder
    values, breaking the sequential dependency."""
    if len(vals) == 0:
        return 0, np.array([], dtype=np.int64), 0
    rng = vals.max() - vals.min()
    bits = bits_for_error(rng, max_error_per_coord)
    codes = quantize(vals, vals.min(), vals.max(), bits)
    best = stream_bits(codes, bits)  # plain or arithmetic only
    return best, codes, bits


# ============================================================
# Parallelogram prediction pass (first pass: collect statistics)
# ============================================================

def collect_direct_and_deltas(verts, tris_np, meshlets, tri_adj):
    """
    First pass: process all meshlets, collect direct vertex coords and
    parallelogram delta coords. Uses original vertex positions for prediction.
    Returns: direct_values (N_direct, 3), delta_values (N_delta, 3),
             n_root_tris (count of root triangles across all meshlets)
    """
    n_verts = len(verts)
    encoded = set()
    direct_values = []
    delta_values = []
    n_root_tris = 0

    for ml_tris in meshlets:
        traversal = meshlet_bfs(ml_tris, tri_adj)

        for tri_idx, parent_idx in traversal:
            tri = [int(v) for v in tris_np[tri_idx]]

            if parent_idx is None:
                n_root_tris += 1
                for v in tri:
                    if v not in encoded:
                        direct_values.append(verts[v])
                        encoded.add(v)
            else:
                parent = [int(v) for v in tris_np[parent_idx]]
                shared = set(tri) & set(parent)
                if len(shared) != 2:
                    # Non-standard adjacency, encode directly
                    for v in tri:
                        if v not in encoded:
                            direct_values.append(verts[v])
                            encoded.add(v)
                    continue

                new_verts = [v for v in tri if v not in shared]
                for v in new_verts:
                    if v not in encoded:
                        va, vb = sorted(shared)
                        opp = [x for x in parent if x not in shared][0]
                        pred = verts[va] + verts[vb] - verts[opp]
                        delta = verts[v] - pred
                        delta_values.append(delta)
                        encoded.add(v)

    # Encode any remaining vertices (isolated, not in any triangle)
    for v in range(n_verts):
        if v not in encoded:
            direct_values.append(verts[v])
            encoded.add(v)

    return np.array(direct_values), np.array(delta_values), n_root_tris


# ============================================================
# Accuracy pass (with quantization error propagation)
# ============================================================

def measure_accuracy(verts, tris_np, meshlets, tri_adj,
                     direct_ranges, direct_bits_arr,
                     delta_ranges, delta_bits_arr):
    """
    Second pass: simulate encode/decode with error propagation.
    Returns per-vertex errors.
    """
    n_verts = len(verts)
    decoded = np.full((n_verts, 3), np.nan)
    encoded = set()

    def q_dq(val, lo, hi, bits):
        code = quantize_scalar(val, lo, hi, bits)
        return dequantize_scalar(code, lo, hi, bits)

    def encode_direct(v):
        for d in range(3):
            decoded[v, d] = q_dq(verts[v, d],
                                 direct_ranges[d, 0], direct_ranges[d, 1],
                                 direct_bits_arr[d])
        encoded.add(v)

    def encode_delta(v, pred):
        delta = verts[v] - pred
        recon_delta = np.zeros(3)
        for d in range(3):
            recon_delta[d] = q_dq(delta[d],
                                  delta_ranges[d, 0], delta_ranges[d, 1],
                                  delta_bits_arr[d])
        decoded[v] = pred + recon_delta
        encoded.add(v)

    for ml_tris in meshlets:
        traversal = meshlet_bfs(ml_tris, tri_adj)

        for tri_idx, parent_idx in traversal:
            tri = [int(v) for v in tris_np[tri_idx]]

            if parent_idx is None:
                for v in tri:
                    if v not in encoded:
                        encode_direct(v)
            else:
                parent = [int(v) for v in tris_np[parent_idx]]
                shared = set(tri) & set(parent)
                if len(shared) != 2:
                    for v in tri:
                        if v not in encoded:
                            encode_direct(v)
                    continue

                for v in tri:
                    if v not in encoded and v not in shared:
                        va, vb = sorted(shared)
                        opp = [x for x in parent if x not in shared][0]
                        # Use DECODED positions for prediction (error propagation)
                        pred = decoded[va] + decoded[vb] - decoded[opp]
                        encode_delta(v, pred)

    # Remaining vertices
    for v in range(n_verts):
        if v not in encoded:
            encode_direct(v)

    errors = np.linalg.norm(decoded - verts, axis=1)
    return errors


# ============================================================
# Main estimation pipeline
# ============================================================

def estimate(verts, tris_np, meshlets, tri_adj, max_error, label=""):
    """Full compression estimation for a set of meshlets."""
    n_verts = len(verts)
    per_coord_err = max_error / np.sqrt(3)

    # First pass: collect direct and delta values
    direct_arr, delta_arr, n_roots = collect_direct_and_deltas(
        verts, tris_np, meshlets, tri_adj)

    n_direct = len(direct_arr)
    n_delta = len(delta_arr)
    n_meshlets = len(meshlets)

    # Header: center(3f) + scale(1f) + n_verts(4B) + n_meshlets(4B)
    #         + per-stream ranges (direct: 6f, delta: 6f) + bit counts (6B)
    header_bits = (3*4 + 4 + 4 + 4 + 12*4 + 6) * 8

    # Quantize and estimate bits for direct stream (3 dims)
    total_bits = header_bits
    direct_ranges = np.zeros((3, 2))
    direct_bits_arr = np.zeros(3, dtype=int)
    for d in range(3):
        if n_direct > 0:
            vals = direct_arr[:, d]
            direct_ranges[d] = [vals.min(), vals.max()]
            bits_d, _, b = best_stream_bits(vals, 16, per_coord_err)
            direct_bits_arr[d] = b
            total_bits += bits_d
        else:
            direct_bits_arr[d] = 1

    # Quantize and estimate bits for delta stream (3 dims)
    delta_ranges = np.zeros((3, 2))
    delta_bits_arr = np.zeros(3, dtype=int)
    for d in range(3):
        if n_delta > 0:
            vals = delta_arr[:, d]
            delta_ranges[d] = [vals.min(), vals.max()]
            bits_d, _, b = best_stream_bits(vals, 16, per_coord_err)
            delta_bits_arr[d] = b
            total_bits += bits_d
        else:
            delta_bits_arr[d] = 1

    bpv = total_bits / n_verts

    # Accuracy: simulate encode/decode with error propagation
    errors = measure_accuracy(verts, tris_np, meshlets, tri_adj,
                              direct_ranges, direct_bits_arr,
                              delta_ranges, delta_bits_arr)
    pct_ok = (errors <= max_error).sum() / n_verts * 100

    # Stats
    raw_bits = n_verts * 96
    direct_range_avg = np.mean([direct_arr[:, d].max() - direct_arr[:, d].min()
                                for d in range(3)]) if n_direct > 0 else 0
    delta_range_avg = np.mean([delta_arr[:, d].max() - delta_arr[:, d].min()
                               for d in range(3)]) if n_delta > 0 else 0

    print(f"  [{label}]")
    print(f"    Meshlets: {n_meshlets}, root triangles: {n_roots}")
    print(f"    Direct: {n_direct} ({n_direct/n_verts*100:.1f}%), "
          f"avg range: {direct_range_avg:.4f}")
    print(f"    Predicted: {n_delta} ({n_delta/n_verts*100:.1f}%), "
          f"avg range: {delta_range_avg:.4f}")
    print(f"    Delta range reduction: {delta_range_avg/(direct_range_avg+1e-12)*100:.1f}% of direct")
    print(f"    Bits: direct=[{','.join(str(b) for b in direct_bits_arr)}] "
          f"delta=[{','.join(str(b) for b in delta_bits_arr)}]")
    print(f"    Total: {total_bits/8:.0f} B  ({bpv:.2f} bpv)  "
          f"ratio={raw_bits/total_bits:.2f}x")
    print(f"    Error: mean={errors.mean():.6f}  max={errors.max():.6f}  "
          f"within_target={pct_ok:.1f}%")

    return {
        "label": label, "total_bytes": total_bits / 8, "bpv": bpv,
        "ratio": raw_bits / total_bits, "n_meshlets": n_meshlets,
        "n_direct": n_direct, "n_delta": n_delta,
        "delta_range_pct": delta_range_avg / (direct_range_avg + 1e-12) * 100,
        "mean_err": errors.mean(), "max_err": errors.max(), "pct_ok": pct_ok,
    }


# ============================================================
# Main
# ============================================================

def run(obj_path, max_error=0.001, max_tris_values=None):
    if max_tris_values is None:
        max_tris_values = [64, 128, 256]

    mesh = Reader.read_from_file(obj_path)
    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)

    # Convert to numpy
    verts_np = np.empty((n_verts, 3), dtype=np.float64)
    for i, v in enumerate(mesh.vertices):
        verts_np[i] = (v.x, v.y, v.z)
    tris_np = np.empty((n_tris, 3), dtype=np.int64)
    for i, t in enumerate(mesh.triangles):
        tris_np[i] = (t.a, t.b, t.c)

    # Normalize
    center = verts_np.mean(axis=0)
    vc = verts_np - center
    scale = np.max(np.linalg.norm(vc, axis=1))
    vn = vc / scale
    norm_max_err = max_error / scale

    print(f"{'='*70}")
    print(f"Meshlet Compression — {obj_path}")
    print(f"  {n_verts} verts, {n_tris} tris, max_error={max_error}")
    print(f"{'='*70}")

    raw_bits = n_verts * 96
    bpc = bits_for_error(2 * scale, max_error)
    baseline_bits = n_verts * 3 * bpc
    print(f"  Raw:      {raw_bits/8:>10.0f} B  ({raw_bits/n_verts:.1f} bpv)")
    print(f"  Baseline: {baseline_bits/8:>10.0f} B  ({baseline_bits/n_verts:.1f} bpv)")

    # Build adjacency
    t0 = time.time()
    tri_adj = build_adjacency(tris_np)
    t_adj = time.time() - t0
    print(f"  Adjacency built in {t_adj:.1f}s")

    # Face normals & centroids (for greedy growing)
    face_normals = compute_face_normals(vn, tris_np)
    face_centroids = compute_face_centroids(vn, tris_np)

    results = []

    for max_t in max_tris_values:
        print(f"\n--- max_tris={max_t} ---")

        # Option 4: Greedy region growing
        t0 = time.time()
        ml_greedy = generate_meshlets_greedy(
            tris_np, tri_adj, face_normals, face_centroids,
            max_tris=max_t, max_verts=max_t * 3)
        t_greedy = time.time() - t0
        print(f"  Greedy: {len(ml_greedy)} meshlets in {t_greedy:.1f}s")

        r = estimate(vn, tris_np, ml_greedy, tri_adj, norm_max_err,
                     label=f"greedy mt={max_t}")
        results.append(r)

        # Option 3: Strip-first, then cut
        t0 = time.time()
        strips = generate_strips_greedy(tris_np, tri_adj)
        ml_strip = meshlets_from_strips(strips, max_tris=max_t)
        t_strip = time.time() - t0
        n_strips = len(strips)
        avg_len = np.mean([len(s) for s in strips])
        print(f"  Strips: {n_strips} strips (avg len {avg_len:.0f}), "
              f"{len(ml_strip)} meshlets in {t_strip:.1f}s")

        r = estimate(vn, tris_np, ml_strip, tri_adj, norm_max_err,
                     label=f"strip mt={max_t}")
        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY — {obj_path}")
    print(f"{'='*70}")
    print(f"{'Method':<22} {'MLets':>6} {'Direct':>7} {'Pred':>7} "
          f"{'DeltaRng':>9} {'Total':>8} {'BPV':>6} {'Ratio':>6} "
          f"{'MaxErr':>9} {'%OK':>5}")
    for r in results:
        print(f"{r['label']:<22} {r['n_meshlets']:>6} {r['n_direct']:>7} "
              f"{r['n_delta']:>7} {r['delta_range_pct']:>8.1f}% "
              f"{r['total_bytes']:>8.0f} {r['bpv']:>6.2f} {r['ratio']:>6.2f} "
              f"{r['max_err']:>9.6f} {r['pct_ok']:>4.1f}%")


if __name__ == "__main__":
    for path in ["assets/bunny.obj", "assets/torus.obj",
                  "assets/stanford-bunny.obj"]:
        try:
            run(path, max_error=0.001)
            print("\n\n")
        except Exception as e:
            import traceback; traceback.print_exc()
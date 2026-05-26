"""
Combined compression: C3 adaptive patches + meshlet parallelogram prediction.
Stage 1 (C3): Fit primitive per patch → local coords → residual ≈ 0
Stage 2 (Meshlet): Parallelogram prediction on local coords → small deltas
"""

import numpy as np
import time
from collections import Counter
from reader import Reader
from sklearn.cluster import KMeans
from utils.meshlet_generator import (
    build_adjacency, compute_face_normals, compute_face_centroids,
    generate_meshlets_greedy, meshlet_bfs,
)


# ============================================================
# Primitive fitting & local coords (from C3)
# ============================================================

PLANE, SPHERE, CYLINDER = 0, 1, 2

def fit_plane(pts):
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, Vt[2], Vt[0], Vt[1]

def fit_sphere(pts):
    A = np.column_stack([2 * pts, np.ones(len(pts))])
    b = np.sum(pts ** 2, axis=1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    c = x[:3]; r = np.sqrt(max(x[3] + c @ c, 1e-6))
    return c, r

def fit_cylinder(pts):
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = Vt[0]
    h = (pts - c) @ axis
    radial = (pts - c) - np.outer(h, axis)
    r = np.mean(np.linalg.norm(radial, axis=1))
    return c, axis, r

def local_plane(pts, c, n, au, av):
    d = pts - c
    return np.stack([d @ au, d @ av, d @ n], axis=1)

def inv_plane(local, c, n, au, av):
    return c + local[:, 0:1]*au + local[:, 1:2]*av + local[:, 2:3]*n

def local_sphere(pts, c, r):
    d = pts - c
    rr = np.linalg.norm(d, axis=1)
    th = np.arccos(np.clip(d[:, 2] / (rr + 1e-12), -1, 1))
    ph = np.arctan2(d[:, 1], d[:, 0])
    ph[ph < 0] += 2 * np.pi
    return np.stack([th, ph, rr - r], axis=1)

def inv_sphere(local, c, r):
    th, ph, dr = local[:, 0], local[:, 1], local[:, 2]
    rr = r + dr
    x = rr * np.sin(th) * np.cos(ph)
    y = rr * np.sin(th) * np.sin(ph)
    z = rr * np.cos(th)
    return np.stack([x, y, z], axis=1) + c

def _cyl_ref(axis):
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    ref -= np.dot(ref, axis) * axis
    ref /= np.linalg.norm(ref)
    return ref, np.cross(axis, ref)

def local_cylinder(pts, c, axis, r):
    d = pts - c
    h = d @ axis
    rad = d - np.outer(h, axis)
    rr = np.linalg.norm(rad, axis=1)
    ref, ref2 = _cyl_ref(axis)
    ang = np.arctan2(rad @ ref2 / (rr+1e-12), rad @ ref / (rr+1e-12))
    ang[ang < 0] += 2 * np.pi
    return np.stack([ang, h, rr - r], axis=1)

def inv_cylinder(local, c, axis, r):
    ang, h, dr = local[:, 0], local[:, 1], local[:, 2]
    ref, ref2 = _cyl_ref(axis)
    rr = r + dr
    xy = rr[:, None] * (np.cos(ang)[:, None]*ref + np.sin(ang)[:, None]*ref2)
    return c + np.outer(h, axis) + xy


def fit_best_primitive(pts):
    """Try all 3 primitives, return (type, params, to_local_fn, inv_fn)."""
    results = []
    # Plane
    c, n, au, av = fit_plane(pts)
    lc = local_plane(pts, c, n, au, av)
    res_std = np.std(lc[:, 2])
    results.append((PLANE, (c, n, au, av), res_std, 24))
    # Sphere
    cs, rs = fit_sphere(pts)
    lc = local_sphere(pts, cs, rs)
    res_std = np.std(lc[:, 2])
    results.append((SPHERE, (cs, rs), res_std, 16))
    # Cylinder
    cc, ax, rc = fit_cylinder(pts)
    lc = local_cylinder(pts, cc, ax, rc)
    res_std = np.std(lc[:, 2])
    results.append((CYLINDER, (cc, ax, rc), res_std, 28))
    # Pick smallest residual (simple heuristic)
    best = min(results, key=lambda x: x[2])
    return best[0], best[1], best[3]


def to_local(pts, ptype, params):
    if ptype == PLANE:
        return local_plane(pts, *params)
    elif ptype == SPHERE:
        return local_sphere(pts, *params)
    else:
        return local_cylinder(pts, *params)


def from_local(lc, ptype, params):
    if ptype == PLANE:
        return inv_plane(lc, *params)
    elif ptype == SPHERE:
        return inv_sphere(lc, *params)
    else:
        return inv_cylinder(lc, *params)


def to_local_single(pt, ptype, params):
    """Transform a single point (3,) -> (3,)."""
    return to_local(pt.reshape(1, 3), ptype, params)[0]


def from_local_single(lc, ptype, params):
    return from_local(lc.reshape(1, 3), ptype, params)[0]


# ============================================================
# Quantization & entropy
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
    return int(round(norm * mx))

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

def compute_vertex_normals(verts, tris):
    normals = np.zeros_like(verts)
    for i in range(len(tris)):
        a, b, c = int(tris[i, 0]), int(tris[i, 1]), int(tris[i, 2])
        e1 = verts[b] - verts[a]
        e2 = verts[c] - verts[a]
        n = np.cross(e1, e2)
        normals[a] += n; normals[b] += n; normals[c] += n
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / (lens + 1e-12)


# ============================================================
# Combined pipeline
# ============================================================

def run(obj_path, max_error=0.001, K_values=None, max_tris=128):
    if K_values is None:
        K_values = [2, 4]

    mesh = Reader.read_from_file(obj_path)
    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)

    verts_np = np.empty((n_verts, 3), dtype=np.float64)
    for i, v in enumerate(mesh.vertices):
        verts_np[i] = (v.x, v.y, v.z)
    tris_np = np.empty((n_tris, 3), dtype=np.int64)
    for i, t in enumerate(mesh.triangles):
        tris_np[i] = (t.a, t.b, t.c)

    center = verts_np.mean(axis=0)
    vc = verts_np - center
    scale = np.max(np.linalg.norm(vc, axis=1))
    vn = vc / scale
    norm_err = max_error / scale
    per_coord_err = norm_err / np.sqrt(3)

    raw_bits = n_verts * 96

    print(f"{'='*70}")
    print(f"Combined C3+Meshlet — {obj_path}")
    print(f"  {n_verts} verts, {n_tris} tris, max_error={max_error}")
    print(f"{'='*70}")
    print(f"  Raw: {raw_bits/8:.0f} B ({raw_bits/n_verts:.1f} bpv)")

    # Build adjacency
    t0 = time.time()
    tri_adj = build_adjacency(tris_np)
    print(f"  Adjacency: {time.time()-t0:.1f}s")

    # Vertex normals for segmentation
    vert_normals = compute_vertex_normals(vn, tris_np)
    face_normals = compute_face_normals(vn, tris_np)
    face_centroids = compute_face_centroids(vn, tris_np)

    results = []

    for K in K_values:
        print(f"\n--- K={K}, max_tris={max_tris} ---")
        t0 = time.time()

        # 1. K-means segment on vertex normals
        vert_labels = KMeans(n_clusters=K, n_init=10, random_state=42).fit(vert_normals).labels_

        # 2. Assign triangles to patches (majority vote)
        tri_labels = np.zeros(n_tris, dtype=int)
        for i in range(n_tris):
            votes = [vert_labels[int(tris_np[i, j])] for j in range(3)]
            tri_labels[i] = max(set(votes), key=votes.count)

        # Global header: center(3f) + scale(1f) + K(1B) + patch_sizes(K*2B)
        global_header_bits = (3*4 + 4 + 1 + K*2) * 8
        total_bits = global_header_bits

        # Track global encoding state
        global_encoded = set()
        local_coords_map = {}  # v -> np.array([c1, c2, c3]) in patch's PCA frame
        patch_primitives = {}  # patch_idx -> (ptype, pparams)
        patch_info = []

        for p in range(K):
            patch_tri_idx = np.where(tri_labels == p)[0]
            if len(patch_tri_idx) == 0:
                patch_info.append({"n_verts": 0, "n_direct": 0, "n_delta": 0})
                continue

            # Vertices referenced and owned by this patch
            patch_vert_set = set()
            for ti in patch_tri_idx:
                for j in range(3):
                    patch_vert_set.add(int(tris_np[ti, j]))
            owned = set(v for v in patch_vert_set if vert_labels[v] == p)

            # Always use PLANE (PCA) for local coords — gives Cartesian frame
            # where parallelogram prediction works correctly
            owned_list = sorted(owned)
            owned_pts = vn[owned_list]
            c, normal, au, av = fit_plane(owned_pts)
            header_bytes = 24  # 6 floats
            patch_primitives[p] = (PLANE, (c, normal, au, av))

            # Transform ALL referenced vertices to this patch's PCA frame
            all_ref = sorted(patch_vert_set)
            all_ref_local = local_plane(vn[all_ref], c, normal, au, av)
            local_map = {}
            for idx, v in enumerate(all_ref):
                local_map[v] = all_ref_local[idx]

            # Build LOCAL adjacency restricted to this patch's triangles
            patch_tri_set = set(int(t) for t in patch_tri_idx)
            local_tri_adj = [[] for _ in range(n_tris)]
            for ti in patch_tri_idx:
                ti = int(ti)
                for nb in tri_adj[ti]:
                    if nb in patch_tri_set:
                        local_tri_adj[ti].append(nb)

            # Generate meshlets WITHIN this patch using local adjacency
            patch_fn = face_normals[patch_tri_idx]
            patch_fc = face_centroids[patch_tri_idx]
            # Map local meshlet indices back to global
            patch_meshlets_raw = generate_meshlets_greedy(
                tris_np, local_tri_adj, face_normals, face_centroids,
                max_tris=max_tris, max_verts=max_tris*3)
            # Keep only meshlets that contain patch triangles
            patch_meshlets = [
                ml for ml in patch_meshlets_raw
                if any(t in patch_tri_set for t in ml)
            ]
            patch_meshlets = [
                [t for t in ml if t in patch_tri_set]
                for ml in patch_meshlets
            ]
            patch_meshlets = [ml for ml in patch_meshlets if ml]

            # Parallelogram prediction in PCA local coords
            direct_vals = []
            delta_vals = []

            for ml_tris in patch_meshlets:
                traversal = meshlet_bfs(ml_tris, local_tri_adj)

                for tri_idx, parent_idx in traversal:
                    tri = [int(v) for v in tris_np[tri_idx]]

                    if parent_idx is None:
                        for v in tri:
                            if v not in global_encoded and v in owned:
                                direct_vals.append(local_map[v])
                                local_coords_map[v] = local_map[v]
                                global_encoded.add(v)
                    else:
                        parent = [int(v) for v in tris_np[parent_idx]]
                        shared = set(tri) & set(parent)
                        if len(shared) != 2:
                            for v in tri:
                                if v not in global_encoded and v in owned:
                                    direct_vals.append(local_map[v])
                                    local_coords_map[v] = local_map[v]
                                    global_encoded.add(v)
                            continue

                        for v in tri:
                            if v not in global_encoded and v in owned and v not in shared:
                                va, vb = sorted(shared)
                                opp = [x for x in parent if x not in shared][0]
                                lc_va = local_coords_map.get(va, local_map.get(va))
                                lc_vb = local_coords_map.get(vb, local_map.get(vb))
                                lc_opp = local_coords_map.get(opp, local_map.get(opp))
                                if lc_va is not None and lc_vb is not None and lc_opp is not None:
                                    ppred = lc_va + lc_vb - lc_opp
                                    delta = local_map[v] - ppred
                                    delta_vals.append(delta)
                                else:
                                    direct_vals.append(local_map[v])
                                local_coords_map[v] = local_map[v]
                                global_encoded.add(v)

            # Remaining owned vertices
            for v in owned:
                if v not in global_encoded:
                    direct_vals.append(local_map[v])
                    local_coords_map[v] = local_map[v]
                    global_encoded.add(v)

            direct_arr = np.array(direct_vals) if direct_vals else np.empty((0, 3))
            delta_arr = np.array(delta_vals) if delta_vals else np.empty((0, 3))

            # Per-patch header: plane params + ranges + bit counts
            patch_header_bits = header_bytes * 8 + (6*4 + 3) * 8
            if len(delta_arr) > 0:
                patch_header_bits += (6*4 + 3) * 8
            total_bits += patch_header_bits

            # Quantize and estimate
            patch_payload = 0
            for d in range(3):
                if len(direct_arr) > 0:
                    vals = direct_arr[:, d]
                    rng = vals.max() - vals.min() if len(vals) > 1 else 0.001
                    b = bits_for_error(rng, per_coord_err)
                    codes = quantize(vals, vals.min(), vals.max(), b)
                    patch_payload += stream_bits(codes, b)
                if len(delta_arr) > 0:
                    vals = delta_arr[:, d]
                    rng = vals.max() - vals.min() if len(vals) > 1 else 0.001
                    b = bits_for_error(rng, per_coord_err)
                    codes = quantize(vals, vals.min(), vals.max(), b)
                    patch_payload += stream_bits(codes, b)
            total_bits += patch_payload

            n_owned = len(owned)
            n_direct = len(direct_arr)
            n_delta = len(delta_arr)
            dir_range = np.mean([direct_arr[:,d].max()-direct_arr[:,d].min()
                                 for d in range(3)]) if n_direct > 1 else 0
            del_range = np.mean([delta_arr[:,d].max()-delta_arr[:,d].min()
                                 for d in range(3)]) if n_delta > 1 else 0
            ratio_pct = del_range / (dir_range + 1e-12) * 100

            print(f"  Patch {p}: {n_owned} verts, plane, "
                  f"direct={n_direct} delta={n_delta}, "
                  f"dir_rng={dir_range:.4f} del_rng={del_range:.4f} ({ratio_pct:.1f}%)")

            patch_info.append({
                "n_verts": n_owned, "n_direct": n_direct, "n_delta": n_delta,
                "header": patch_header_bits, "payload": patch_payload,
            })

        # Handle unencoded vertices
        n_missed = 0
        for v in range(n_verts):
            if v not in global_encoded:
                n_missed += 1
                total_bits += 3 * 12
                global_encoded.add(v)
        if n_missed > 0:
            print(f"  Warning: {n_missed} vertices not reached by any patch")

        bpv = total_bits / n_verts
        t_total = time.time() - t0

        n_total_direct = sum(pi.get("n_direct", 0) for pi in patch_info)
        n_total_delta = sum(pi.get("n_delta", 0) for pi in patch_info)
        print(f"  Total: {total_bits/8:.0f} B  ({bpv:.2f} bpv)  "
              f"ratio={raw_bits/total_bits:.2f}x  [{t_total:.1f}s]")
        print(f"  Direct: {n_total_direct} ({n_total_direct/n_verts*100:.1f}%), "
              f"Predicted: {n_total_delta} ({n_total_delta/n_verts*100:.1f}%)")

        results.append({
            "K": K, "mt": max_tris,
            "total_bytes": total_bits / 8, "bpv": bpv,
            "ratio": raw_bits / total_bits,
            "n_direct": n_total_direct, "n_delta": n_total_delta,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — {obj_path}")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Total':>8} {'BPV':>7} {'Ratio':>7} "
          f"{'Direct':>7} {'Predicted':>9}")
    for r in results:
        print(f"K={r['K']} mt={r['mt']:<13} {r['total_bytes']:>8.0f} "
              f"{r['bpv']:>7.2f} {r['ratio']:>7.2f} "
              f"{r['n_direct']:>7} {r['n_delta']:>9}")


if __name__ == "__main__":
    for path in ["assets/bunny.obj", "assets/torus.obj",
                  "assets/stanford-bunny.obj"]:
        try:
            run(path, max_error=0.001, K_values=[2, 4], max_tris=128)
            print("\n\n")
        except Exception as e:
            import traceback; traceback.print_exc()
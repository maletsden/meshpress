"""
Visualize one meshlet: Bezier surface fit + displacement vectors.
Also: compare plain vs entropy-coded stream sizes.
"""

import numpy as np
import pyvista as pv
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
from collections import Counter


# ============================================================
# Entropy helpers
# ============================================================

def quantize(vals, lo, hi, bits):
    mx = (1 << bits) - 1
    norm = np.clip((vals - lo) / (hi - lo + 1e-15), 0, 1)
    return np.round(norm * mx).astype(np.int64)

def bits_for_error(val_range, max_err):
    if max_err <= 0 or val_range <= 0:
        return 1
    return max(1, int(np.ceil(np.log2(val_range / (2 * max_err) + 1))))

def shannon_entropy(codes):
    if len(codes) == 0:
        return 0.0
    counts = Counter(codes.tolist())
    total = len(codes)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


# ============================================================
# Main
# ============================================================

mesh = Reader.read_from_file('assets/stanford-bunny.obj')
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
max_error = 0.001
norm_err = max_error / scale
per_coord_err = norm_err / np.sqrt(3)

# Build adjacency & meshlets
tri_adj = build_adjacency(tris_np)
fn = compute_face_normals(vn, tris_np)
fc = compute_face_centroids(vn, tris_np)
meshlets = generate_meshlets_greedy(tris_np, tri_adj, fn, fc, max_tris=128, max_verts=384)

# Pick the largest meshlet for visualization
ml_idx = max(range(len(meshlets)), key=lambda i: len(meshlets[i]))
ml_tris = meshlets[ml_idx]

# Collect meshlet vertices
vert_set = set()
for ti in ml_tris:
    for j in range(3):
        vert_set.add(int(tris_np[ti, j]))
local_verts = sorted(vert_set)
pts = vn[local_verts]

deg = 2
print(f"Meshlet {ml_idx}: {len(ml_tris)} faces, {len(local_verts)} verts")

# Parameterize and fit
u, v, pca_frame = parameterize_pca(pts)
cp = fit_bezier(u, v, pts, deg)
disps, surf_pts, normals = compute_displacements(u, v, pts, cp, deg)

# Compute derivative-aware bits
Su, Sv = bezier_derivatives(u, v, cp, deg)
max_Su = np.max(np.linalg.norm(Su, axis=1))
max_Sv = np.max(np.linalg.norm(Sv, axis=1))
u_prec = per_coord_err / max(max_Su, 1e-6)
v_prec = per_coord_err / max(max_Sv, 1e-6)

u_range = u.max() - u.min()
v_range = v.max() - v.min()
d_range = disps.max() - disps.min() if len(disps) > 1 else 0.001

bits_u = bits_for_error(u_range, u_prec)
bits_v = bits_for_error(v_range, v_prec)
bits_d = bits_for_error(d_range, per_coord_err)

codes_u = quantize(u, u.min(), u.max(), bits_u)
codes_v = quantize(v, v.min(), v.max(), bits_v)
codes_d = quantize(disps, disps.min(), disps.max(), bits_d)

n = len(local_verts)
print(f"\n--- Compression Analysis for this meshlet ---")
print(f"  u: range={u_range:.4f}, bits={bits_u}, derivative_scale={max_Su:.4f}")
print(f"  v: range={v_range:.4f}, bits={bits_v}, derivative_scale={max_Sv:.4f}")
print(f"  d: range={d_range:.6f}, bits={bits_d}, std={disps.std():.6f}")

# Compare plain vs entropy coding
for name, codes, bits in [("u", codes_u, bits_u), ("v", codes_v, bits_v), ("d", codes_d, bits_d)]:
    plain = n * bits
    ent = shannon_entropy(codes)
    arith = n * ent + 32
    unique = len(set(codes.tolist()))
    print(f"\n  Stream '{name}': {n} values, {unique} unique codes")
    print(f"    Plain:     {plain:>6} bits ({plain/n:.1f} bpv)")
    print(f"    Entropy:   {ent:.2f} bits/symbol")
    print(f"    Arith:     {arith:>6.0f} bits ({arith/n:.1f} bpv)")
    print(f"    Saving:    {(1-arith/plain)*100:.1f}% {'← entropy helps!' if arith < plain else '← entropy not helpful'}")

# Total comparison
plain_total = n * (bits_u + bits_v + bits_d)
arith_total = sum(n * shannon_entropy(c) + 32 for c in [codes_u, codes_v, codes_d])
header = n_control_points(deg) * 3 * 16 + 7 * 32  # CP + PCA frame
print(f"\n  TOTAL (vertex data only):")
print(f"    Plain:   {plain_total/8:.0f} B  ({plain_total/n:.1f} bpv)")
print(f"    Arith:   {arith_total/8:.0f} B  ({arith_total/n:.1f} bpv)")
print(f"    Header:  {header/8:.0f} B  ({header/n:.1f} bpv)")
print(f"    Combined plain:  {(plain_total+header)/8:.0f} B  ({(plain_total+header)/n:.1f} bpv)")
print(f"    Combined arith:  {(arith_total+header)/8:.0f} B  ({(arith_total+header)/n:.1f} bpv)")

# ============================================================
# Visualization
# ============================================================

print("\n--- Opening 3D visualization ---")

# Generate a dense Bezier surface grid for visualization
u_grid = np.linspace(0, 1, 30)
v_grid = np.linspace(0, 1, 30)
uu, vv = np.meshgrid(u_grid, v_grid)
uu_flat = uu.ravel()
vv_flat = vv.ravel()

surf_grid = evaluate_bezier(uu_flat, vv_flat, cp, deg)

# Scale back to world
pts_world = pts * scale + center
surf_pts_world = surf_pts * scale + center
surf_grid_world = surf_grid * scale + center
normals_world = normals  # unit normals, no scaling

# Build meshlet face list for PyVista
local_idx_map = {v: i for i, v in enumerate(local_verts)}
faces_pv = []
for ti in ml_tris:
    a, b, c = int(tris_np[ti, 0]), int(tris_np[ti, 1]), int(tris_np[ti, 2])
    if a in local_idx_map and b in local_idx_map and c in local_idx_map:
        faces_pv.append([3, local_idx_map[a], local_idx_map[b], local_idx_map[c]])
if faces_pv:
    faces_flat = np.array(faces_pv).flatten()
    meshlet_mesh = pv.PolyData(pts_world, faces_flat)
else:
    meshlet_mesh = pv.PolyData(pts_world)

# Bezier surface as a structured grid
surf_grid_pv = pv.StructuredGrid(
    surf_grid_world[:, 0].reshape(30, 30),
    surf_grid_world[:, 1].reshape(30, 30),
    surf_grid_world[:, 2].reshape(30, 30),
)

# Displacement vectors
disp_vectors = disps[:, np.newaxis] * normals_world * scale  # scale to world

# Create plotter
pl = pv.Plotter(shape=(1, 2))

# Left: meshlet mesh + Bezier surface
pl.subplot(0, 0)
pl.add_text("Meshlet (blue) + Bezier Surface (orange)", font_size=10)
pl.add_mesh(meshlet_mesh, color="lightblue", show_edges=True, edge_color="gray", opacity=0.7)
pl.add_mesh(surf_grid_pv, color="orange", opacity=0.4, style="wireframe", line_width=2)
pl.add_points(pts_world, color="red", point_size=5, render_points_as_spheres=True)

# Right: displacement visualization
pl.subplot(0, 1)
pl.add_text("Displacement from Bezier surface", font_size=10)
pl.add_mesh(meshlet_mesh, scalars=np.abs(disps) * scale, cmap="coolwarm",
            show_edges=True, edge_color="gray")
# Displacement arrows (scaled up for visibility)
arrow_scale = max(1.0, 0.01 / (np.abs(disps).max() * scale + 1e-10))
pl.add_arrows(surf_pts_world, disp_vectors * arrow_scale,
              mag=1.0, color="black")
pl.add_scalar_bar("Displacement (world units)")

pl.link_views()
pl.show()
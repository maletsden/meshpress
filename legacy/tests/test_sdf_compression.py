"""
SDF-based mesh compression: estimation of compression rate and accuracy.

Pipeline:
  1. Center mesh to origin, normalize to unit sphere
  2. Fit parametric SDF (smooth union of 2 spheres) to mesh surface
  3. Convert vertices to spherical coords (r, u, v)
  4. Ray-march SDF along (u,v) direction -> r_predicted
  5. r_residual = r - r_predicted
  6. Quantize u, v, r_residual to bits needed for desired max error
  7. Estimate entropy (plain vs tree-partitioned + arithmetic coding)
  8. Dequantize and measure reconstruction accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import Counter
from reader import Reader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# 1. Parametric SDF Model (2 spheres with smooth union)
# ============================================================

def smin_torch(a, b, k):
    """Polynomial smooth minimum (differentiable)."""
    h = torch.clamp((k - torch.abs(a - b)) / k, min=0.0)
    return torch.min(a, b) - h * h * h * k * (1.0 / 6.0)


class SDFModel(nn.Module):
    """Smooth union of 2 spheres. Learnable: centers, radii, smoothness."""

    def __init__(self):
        super().__init__()
        # Initialize: two spheres offset along x-axis
        self.center0 = nn.Parameter(torch.tensor([0.3, 0.0, 0.0]))
        self.center1 = nn.Parameter(torch.tensor([-0.3, 0.0, 0.0]))
        self.radius0 = nn.Parameter(torch.tensor(0.7))
        self.radius1 = nn.Parameter(torch.tensor(0.7))
        self.smoothness = nn.Parameter(torch.tensor(0.3))

    def forward(self, points):
        """Compute SDF value at each point. points: (N, 3)"""
        k = torch.clamp(self.smoothness, 0.01, 2.0)
        r0 = torch.clamp(self.radius0, 0.05, 5.0)
        r1 = torch.clamp(self.radius1, 0.05, 5.0)

        d0 = torch.norm(points - self.center0, dim=1) - r0
        d1 = torch.norm(points - self.center1, dim=1) - r1
        return smin_torch(d0, d1, k)

    def header_size_bits(self):
        """Size of SDF parameters in the compressed stream."""
        # 2 centers (3 floats each) + 2 radii + 1 smoothness = 9 floats = 9*32 bits
        return 9 * 32

    def __repr__(self):
        with torch.no_grad():
            return (f"SDFModel(\n"
                    f"  center0={self.center0.cpu().numpy()}, r0={self.radius0.item():.4f}\n"
                    f"  center1={self.center1.cpu().numpy()}, r1={self.radius1.item():.4f}\n"
                    f"  smoothness={self.smoothness.item():.4f}\n)")


# ============================================================
# 2. Training
# ============================================================

def train_sdf(model, vertices_np, epochs=2000, lr=0.01):
    """Fit SDF so that surface passes through mesh vertices (SDF ≈ 0)."""
    points = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        sdf_vals = model(points)
        loss = torch.mean(sdf_vals ** 2)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d}: loss = {loss.item():.8f}")

    print(f"  Final:       loss = {loss.item():.8f}")
    return model


# ============================================================
# 3. Spherical coordinates & ray marching
# ============================================================

def cartesian_to_spherical(xyz):
    """(N,3) -> (r, u, v) where u=theta/pi in [0,1], v=phi/(2pi) in [0,1]."""
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(xyz[:, 2] / (r + 1e-12), -1, 1))  # [0, pi]
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])  # [-pi, pi]
    phi[phi < 0] += 2 * np.pi  # [0, 2pi]
    u = theta / np.pi       # [0, 1]
    v = phi / (2 * np.pi)   # [0, 1]
    return r, u, v


def spherical_to_cartesian(r, u, v):
    """Inverse: (r, u, v) -> (N,3) xyz."""
    theta = u * np.pi
    phi = v * 2 * np.pi
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


def ray_march_sdf(model, directions_torch, max_steps=64):
    """
    Find surface intersection along each direction using bisection.
    Assumes origin is inside the surface (SDF < 0) and far away is outside (SDF > 0).
    Returns predicted radius for each direction.
    """
    with torch.no_grad():
        n = directions_torch.shape[0]
        # Bracket: t_lo (inside, SDF<0), t_hi (outside, SDF>0)
        t_lo = torch.zeros(n, device=device)
        t_hi = torch.ones(n, device=device) * 3.0  # start far enough outside

        for _ in range(max_steps):
            t_mid = (t_lo + t_hi) / 2
            points = directions_torch * t_mid.unsqueeze(1)
            dist = model(points)
            # If SDF < 0 (inside), move lo up; if SDF > 0 (outside), move hi down
            inside = dist < 0
            t_lo = torch.where(inside, t_mid, t_lo)
            t_hi = torch.where(inside, t_hi, t_mid)

        return ((t_lo + t_hi) / 2).cpu().numpy()


# ============================================================
# 4. Quantization
# ============================================================

def quantize(values, min_val, max_val, bits):
    """Uniform quantization to integer codes."""
    max_int = (1 << bits) - 1
    normalized = (values - min_val) / (max_val - min_val + 1e-12)
    normalized = np.clip(normalized, 0.0, 1.0)
    codes = np.round(normalized * max_int).astype(np.int64)
    return codes


def dequantize(codes, min_val, max_val, bits):
    """Inverse of quantize."""
    max_int = (1 << bits) - 1
    normalized = codes.astype(np.float64) / max_int
    return normalized * (max_val - min_val) + min_val


def bits_for_max_error(value_range, max_error):
    """Minimum bits to achieve max_error over a given range."""
    if max_error <= 0 or value_range <= 0:
        return 16
    # Quantization step = range / (2^bits - 1), max error = step/2
    # So 2^bits - 1 >= range / (2 * max_error)
    n_levels = value_range / (2 * max_error)
    return max(1, int(np.ceil(np.log2(n_levels + 1))))


# ============================================================
# 5. Entropy estimation (plain and tree-based)
# ============================================================

def shannon_entropy(codes):
    """Bits per symbol via Shannon entropy."""
    if len(codes) == 0:
        return 0.0
    counts = Counter(codes.tolist())
    total = len(codes)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * np.log2(p)
    return ent


def estimate_stream_bits(codes, fixed_bits):
    """
    Compare plain fixed-width vs arithmetic coding estimate.
    Returns (plain_bits, arithmetic_bits, chosen_bits, method_name).
    """
    n = len(codes)
    plain = n * fixed_bits
    ent = shannon_entropy(codes)
    arith = n * ent + 32  # 32 bits overhead for arithmetic coder state
    if arith < plain:
        return plain, arith, arith, "arithmetic"
    else:
        return plain, arith, plain, "fixed"


def tree_partition_entropy(values_2d, codes_u, codes_v, bits_u, bits_v, max_depth=12, min_leaf=4):
    """
    Build binary tree over (u,v) space, estimate entropy of leaf-local codes.
    Returns total estimated bits for the tree-based approach.
    """
    header_bits = [0.0]
    payload_bits = [0.0]

    def _recurse(indices, depth, axis):
        n = len(indices)
        if n == 0:
            return
        if depth >= max_depth or n <= min_leaf:
            # Leaf: estimate entropy of codes within this cell
            leaf_u = codes_u[indices]
            leaf_v = codes_v[indices]

            # Local re-quantization: the tree path already narrows the range
            # so local codes have fewer effective bits
            local_bits_u = max(1, bits_u - (depth + 1) // 2)
            local_bits_v = max(1, bits_v - depth // 2)

            # Local entropy
            ent_u = shannon_entropy(leaf_u % (1 << local_bits_u))
            ent_v = shannon_entropy(leaf_v % (1 << local_bits_v))

            payload_bits[0] += n * (ent_u + ent_v)
            return

        # Internal node: store count of left children
        header_bits[0] += np.ceil(np.log2(n + 1))

        # Split along current axis
        vals = values_2d[indices, axis]
        median = np.median(vals)
        left_mask = vals < median
        # Avoid empty splits
        if left_mask.sum() == 0 or left_mask.sum() == n:
            # Can't split further, treat as leaf
            leaf_u = codes_u[indices]
            leaf_v = codes_v[indices]
            local_bits_u = max(1, bits_u - (depth + 1) // 2)
            local_bits_v = max(1, bits_v - depth // 2)
            ent_u = shannon_entropy(leaf_u % (1 << local_bits_u))
            ent_v = shannon_entropy(leaf_v % (1 << local_bits_v))
            payload_bits[0] += n * (ent_u + ent_v)
            return

        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]
        _recurse(left_idx, depth + 1, 1 - axis)
        _recurse(right_idx, depth + 1, 1 - axis)

    indices = np.arange(len(codes_u))
    _recurse(indices, 0, 0)
    return header_bits[0], payload_bits[0]


def tree_partition_entropy_1d(values_1d, codes, bits, max_depth=12, min_leaf=4):
    """
    Binary tree over 1D values (for r_residual), estimate entropy.
    """
    header_bits = [0.0]
    payload_bits = [0.0]

    def _recurse(indices, depth):
        n = len(indices)
        if n == 0:
            return
        if depth >= max_depth or n <= min_leaf:
            local_bits = max(1, bits - depth)
            leaf_codes = codes[indices] % (1 << local_bits)
            ent = shannon_entropy(leaf_codes)
            payload_bits[0] += n * ent
            return

        header_bits[0] += np.ceil(np.log2(n + 1))
        vals = values_1d[indices]
        median = np.median(vals)
        left_mask = vals < median
        if left_mask.sum() == 0 or left_mask.sum() == n:
            local_bits = max(1, bits - depth)
            leaf_codes = codes[indices] % (1 << local_bits)
            ent = shannon_entropy(leaf_codes)
            payload_bits[0] += n * ent
            return

        _recurse(indices[left_mask], depth + 1)
        _recurse(indices[~left_mask], depth + 1)

    _recurse(np.arange(len(codes)), 0)
    return header_bits[0], payload_bits[0]


# ============================================================
# 6. Main pipeline
# ============================================================

def run_compression_estimation(obj_path, max_error=0.001):
    print(f"{'='*60}")
    print(f"SDF Compression Estimation")
    print(f"Model: {obj_path}")
    print(f"Target max error: {max_error}")
    print(f"{'='*60}")

    # --- Load and center mesh ---
    mesh = Reader.read_from_file(obj_path)
    verts_np = np.array([[v.x, v.y, v.z] for v in mesh.vertices])
    n_verts = len(verts_np)

    center = verts_np.mean(axis=0)
    verts_centered = verts_np - center

    # Normalize to roughly unit sphere
    scale = np.max(np.linalg.norm(verts_centered, axis=1))
    verts_normalized = verts_centered / scale

    print(f"\nVertices: {n_verts}")
    print(f"Center: {center}")
    print(f"Scale: {scale:.6f}")

    # --- Train SDF ---
    print(f"\n--- Training SDF (2 spheres) ---")
    sdf_model = SDFModel().to(device)
    sdf_model = train_sdf(sdf_model, verts_normalized, epochs=2000, lr=0.01)
    print(sdf_model)

    # --- Convert to spherical ---
    r, u, v = cartesian_to_spherical(verts_normalized)
    print(f"\n--- Spherical Coords ---")
    print(f"r:  [{r.min():.4f}, {r.max():.4f}], mean={r.mean():.4f}")
    print(f"u:  [{u.min():.4f}, {u.max():.4f}]")
    print(f"v:  [{v.min():.4f}, {v.max():.4f}]")

    # --- Ray march to get r_predicted ---
    print(f"\n--- Ray Marching SDF ---")
    directions = spherical_to_cartesian(np.ones_like(r), u, v)
    dir_torch = torch.tensor(directions, dtype=torch.float32, device=device)
    r_pred = ray_march_sdf(sdf_model, dir_torch)
    r_residual = r - r_pred

    print(f"r_predicted: [{r_pred.min():.4f}, {r_pred.max():.4f}]")
    print(f"r_residual:  [{r_residual.min():.4f}, {r_residual.max():.4f}], "
          f"std={r_residual.std():.6f}")
    print(f"Variance reduction: {(1 - r_residual.var() / r.var()) * 100:.1f}%")

    # --- Determine bits per stream ---
    # u range is [0, 1], max angular error -> bits
    # For u: error in u maps to angular error = max_error * pi (approx)
    # We want spatial error < max_error at radius ~1, so angular error ≈ max_error
    u_range = u.max() - u.min()
    v_range = v.max() - v.min()
    r_res_range = r_residual.max() - r_residual.min()

    bits_u = bits_for_max_error(u_range, max_error)
    bits_v = bits_for_max_error(v_range, max_error)
    bits_r = bits_for_max_error(r_res_range, max_error)

    print(f"\n--- Quantization ---")
    print(f"bits_u: {bits_u} (range={u_range:.4f})")
    print(f"bits_v: {bits_v} (range={v_range:.4f})")
    print(f"bits_r: {bits_r} (range={r_res_range:.4f})")

    # --- Quantize ---
    codes_u = quantize(u, u.min(), u.max(), bits_u)
    codes_v = quantize(v, v.min(), v.max(), bits_v)
    codes_r = quantize(r_residual, r_residual.min(), r_residual.max(), bits_r)

    # --- Entropy estimation ---
    print(f"\n--- Entropy Estimation ---")

    # u,v: plain vs arithmetic
    plain_uv, arith_uv, chosen_uv, method_uv = estimate_stream_bits(
        np.stack([codes_u, codes_v], axis=1).ravel(),
        (bits_u + bits_v) // 2
    )
    # Better: estimate u and v separately
    plain_u, arith_u, chosen_u, method_u = estimate_stream_bits(codes_u, bits_u)
    plain_v, arith_v, chosen_v, method_v = estimate_stream_bits(codes_v, bits_v)
    plain_r, arith_r, chosen_r, method_r = estimate_stream_bits(codes_r, bits_r)

    print(f"u: plain={plain_u/8:.0f}B, arith={arith_u/8:.0f}B -> {method_u}")
    print(f"v: plain={plain_v/8:.0f}B, arith={arith_v/8:.0f}B -> {method_v}")
    print(f"r: plain={plain_r/8:.0f}B, arith={arith_r/8:.0f}B -> {method_r}")

    # Tree-based estimation for u,v
    uv_2d = np.stack([u, v], axis=1)
    tree_hdr_uv, tree_pay_uv = tree_partition_entropy(
        uv_2d, codes_u, codes_v, bits_u, bits_v)
    tree_total_uv = tree_hdr_uv + tree_pay_uv

    # Tree-based estimation for r
    tree_hdr_r, tree_pay_r = tree_partition_entropy_1d(
        r_residual, codes_r, bits_r)
    tree_total_r = tree_hdr_r + tree_pay_r

    print(f"\nTree-based u,v: header={tree_hdr_uv/8:.0f}B, "
          f"payload={tree_pay_uv/8:.0f}B, total={tree_total_uv/8:.0f}B")
    print(f"Tree-based r:   header={tree_hdr_r/8:.0f}B, "
          f"payload={tree_pay_r/8:.0f}B, total={tree_total_r/8:.0f}B")

    # --- Pick best method for each stream ---
    best_uv = min(chosen_u + chosen_v, tree_total_uv)
    best_uv_method = "tree" if tree_total_uv < chosen_u + chosen_v else f"plain({method_u}/{method_v})"
    best_r = min(chosen_r, tree_total_r)
    best_r_method = "tree" if tree_total_r < chosen_r else method_r

    print(f"\nBest u,v: {best_uv/8:.0f}B ({best_uv_method})")
    print(f"Best r:   {best_r/8:.0f}B ({best_r_method})")

    # --- Total compressed size ---
    sdf_header = sdf_model.header_size_bits()
    # Additional header: center (3*32), scale (32), ranges for u,v,r (6*32),
    # bits_u, bits_v, bits_r (3*8)
    meta_header = 3 * 32 + 32 + 6 * 32 + 3 * 8  # center, scale, min/max ranges, bit counts

    total_compressed_bits = sdf_header + meta_header + best_uv + best_r

    # Raw size: 3 floats * 32 bits per vertex
    raw_bits = n_verts * 3 * 32
    # Quantized baseline: 3 coords * bits_for_max_error
    baseline_bits_per_coord = bits_for_max_error(2 * scale, max_error)
    baseline_bits = n_verts * 3 * baseline_bits_per_coord

    print(f"\n{'='*60}")
    print(f"--- COMPRESSION SUMMARY ---")
    print(f"{'='*60}")
    print(f"Raw (float32):          {raw_bits/8:>10.0f} bytes  ({raw_bits/n_verts:.1f} bpv)")
    print(f"Quantized baseline:     {baseline_bits/8:>10.0f} bytes  ({baseline_bits/n_verts:.1f} bpv)")
    print(f"SDF compressed:")
    print(f"  SDF header:           {sdf_header/8:>10.0f} bytes")
    print(f"  Meta header:          {meta_header/8:>10.0f} bytes")
    print(f"  u,v payload:          {best_uv/8:>10.0f} bytes  ({best_uv_method})")
    print(f"  r payload:            {best_r/8:>10.0f} bytes  ({best_r_method})")
    print(f"  TOTAL:                {total_compressed_bits/8:>10.0f} bytes  "
          f"({total_compressed_bits/n_verts:.2f} bpv)")
    print(f"")
    print(f"Ratio vs raw:           {raw_bits/total_compressed_bits:.2f}x")
    print(f"Ratio vs quantized:     {baseline_bits/total_compressed_bits:.2f}x")

    # --- Dequantize and measure accuracy ---
    print(f"\n--- RECONSTRUCTION ACCURACY ---")

    u_recon = dequantize(codes_u, u.min(), u.max(), bits_u)
    v_recon = dequantize(codes_v, v.min(), v.max(), bits_v)
    r_res_recon = dequantize(codes_r, r_residual.min(), r_residual.max(), bits_r)

    # Reconstruct r from residual + SDF prediction
    r_recon = r_pred + r_res_recon

    # Back to Cartesian
    xyz_recon = spherical_to_cartesian(r_recon, u_recon, v_recon)

    # Un-normalize
    xyz_recon = xyz_recon * scale + center
    xyz_original = verts_np

    # Per-vertex error
    errors = np.linalg.norm(xyz_recon - xyz_original, axis=1)

    print(f"Mean error:    {errors.mean():.8f}")
    print(f"Max error:     {errors.max():.8f}")
    print(f"RMS error:     {np.sqrt((errors**2).mean()):.8f}")
    print(f"Target error:  {max_error}")
    print(f"% within target: {(errors <= max_error).sum() / n_verts * 100:.1f}%")

    return {
        'n_verts': n_verts,
        'raw_bits': raw_bits,
        'compressed_bits': total_compressed_bits,
        'bpv': total_compressed_bits / n_verts,
        'ratio': raw_bits / total_compressed_bits,
        'mean_error': errors.mean(),
        'max_error': errors.max(),
    }


# ============================================================
# Run on test models
# ============================================================

if __name__ == "__main__":
    for model_path in ['assets/bunny.obj', 'assets/torus.obj']:
        try:
            result = run_compression_estimation(model_path, max_error=0.001)
            print(f"\n\n")
        except Exception as e:
            print(f"Error processing {model_path}: {e}")
            import traceback; traceback.print_exc()
            print()
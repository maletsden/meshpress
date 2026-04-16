"""
Tensor-product Bezier surface utilities for mesh compression.
Supports fitting, evaluation, and displacement computation.
"""

import numpy as np
from scipy.special import comb as _comb


def bernstein_basis(n, t):
    """Bernstein basis polynomials B_{i,n}(t) for i=0..n.
    t: (N,) array. Returns: (N, n+1) matrix."""
    t = np.clip(t, 0, 1)
    N = len(t)
    B = np.zeros((N, n + 1))
    for i in range(n + 1):
        B[:, i] = _comb(n, i, exact=True) * t ** i * (1 - t) ** (n - i)
    return B


def bernstein_deriv(n, t):
    """Derivative of Bernstein basis: dB_{i,n}/dt.
    Returns: (N, n+1) matrix."""
    if n == 0:
        return np.zeros((len(t), 1))
    B_prev = bernstein_basis(n - 1, t)  # (N, n)
    dB = np.zeros((len(t), n + 1))
    for i in range(n + 1):
        if i > 0:
            dB[:, i] += n * B_prev[:, i - 1]
        if i < n:
            dB[:, i] -= n * B_prev[:, i]
    return dB


def bezier_basis_2d(deg, u, v):
    """Tensor-product Bezier basis matrix for degree (deg, deg).
    u, v: (N,) arrays. Returns: (N, (deg+1)^2) matrix."""
    Bu = bernstein_basis(deg, u)
    Bv = bernstein_basis(deg, v)
    d1 = deg + 1
    N = len(u)
    B = np.zeros((N, d1 * d1))
    for i in range(d1):
        for j in range(d1):
            B[:, i * d1 + j] = Bu[:, i] * Bv[:, j]
    return B


def fit_bezier(u, v, vertices, deg):
    """Fit tensor-product Bezier surface (deg × deg) to vertices.
    Returns control points: ((deg+1)^2, 3)."""
    B = bezier_basis_2d(deg, u, v)
    cp, _, _, _ = np.linalg.lstsq(B, vertices, rcond=None)
    return cp


def evaluate_bezier(u, v, cp, deg):
    """Evaluate Bezier surface at parameters. Returns: (N, 3)."""
    B = bezier_basis_2d(deg, u, v)
    return B @ cp


def bezier_normals(u, v, cp, deg):
    """Compute surface normals via cross product of partial derivatives.
    Returns: (N, 3) unit normals."""
    d1 = deg + 1
    N = len(u)

    # Partial derivative w.r.t. u
    dBu = bernstein_deriv(deg, u)
    Bv = bernstein_basis(deg, v)
    dB_du = np.zeros((N, d1 * d1))
    for i in range(d1):
        for j in range(d1):
            dB_du[:, i * d1 + j] = dBu[:, i] * Bv[:, j]
    Su = dB_du @ cp  # (N, 3)

    # Partial derivative w.r.t. v
    Bu = bernstein_basis(deg, u)
    dBv = bernstein_deriv(deg, v)
    dB_dv = np.zeros((N, d1 * d1))
    for i in range(d1):
        for j in range(d1):
            dB_dv[:, i * d1 + j] = Bu[:, i] * dBv[:, j]
    Sv = dB_dv @ cp  # (N, 3)

    normals = np.cross(Su, Sv)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / (lens + 1e-12)


def bezier_derivatives(u, v, cp, deg):
    """Compute partial derivatives ∂S/∂u and ∂S/∂v.
    Returns: Su (N,3), Sv (N,3)."""
    d1 = deg + 1
    N = len(u)

    dBu = bernstein_deriv(deg, u)
    Bv = bernstein_basis(deg, v)
    dB_du = np.zeros((N, d1 * d1))
    for i in range(d1):
        for j in range(d1):
            dB_du[:, i * d1 + j] = dBu[:, i] * Bv[:, j]
    Su = dB_du @ cp

    Bu = bernstein_basis(deg, u)
    dBv = bernstein_deriv(deg, v)
    dB_dv = np.zeros((N, d1 * d1))
    for i in range(d1):
        for j in range(d1):
            dB_dv[:, i * d1 + j] = Bu[:, i] * dBv[:, j]
    Sv = dB_dv @ cp

    return Su, Sv


def parameterize_pca(vertices):
    """PCA-based parameterization of a vertex set → (u, v) in [0, 1]².
    Returns u, v arrays and the PCA frame (center, axis_u, axis_v, normal)."""
    center = vertices.mean(axis=0)
    centered = vertices - center
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Project onto first 2 principal components
    u_raw = centered @ Vt[0]
    v_raw = centered @ Vt[1]

    # Normalize to [0, 1] with small margin to avoid boundary issues
    eps = 1e-6
    u_min, u_max = u_raw.min(), u_raw.max()
    v_min, v_max = v_raw.min(), v_raw.max()
    u_range = u_max - u_min if u_max > u_min else 1.0
    v_range = v_max - v_min if v_max > v_min else 1.0

    u = (u_raw - u_min) / u_range * (1 - 2 * eps) + eps
    v = (v_raw - v_min) / v_range * (1 - 2 * eps) + eps

    return u, v, (center, Vt[0], Vt[1], Vt[2], u_min, u_max, v_min, v_max)


def compute_displacements(u, v, vertices, cp, deg):
    """Compute scalar displacement along surface normal for each vertex.
    Returns: displacements (N,), surface_points (N,3), normals (N,3)."""
    surface_pts = evaluate_bezier(u, v, cp, deg)
    normals = bezier_normals(u, v, cp, deg)
    diff = vertices - surface_pts
    displacements = np.sum(diff * normals, axis=1)
    return displacements, surface_pts, normals


def reconstruct_from_bezier(u, v, d, cp, deg):
    """Reconstruct vertices from (u, v, d) + Bezier surface."""
    surface_pts = evaluate_bezier(u, v, cp, deg)
    normals = bezier_normals(u, v, cp, deg)
    return surface_pts + d[:, np.newaxis] * normals


def n_control_points(deg):
    """Number of control points for degree (deg, deg) tensor-product patch."""
    return (deg + 1) ** 2

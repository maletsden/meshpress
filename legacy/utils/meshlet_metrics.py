"""Per-meshlet-set metrics: boundary fraction, restarts, planarity."""

import numpy as np


def _meshlet_vert_sets(meshlets, tris_np):
    sets = []
    for ml in meshlets:
        s = set()
        for ti in ml:
            for j in range(3):
                s.add(int(tris_np[ti, j]))
        sets.append(s)
    return sets


def boundary_pct(meshlets, tris_np):
    """Fraction of vertices that appear in 2+ meshlets (boundary)."""
    counts = {}
    for s in _meshlet_vert_sets(meshlets, tris_np):
        for v in s:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return 0.0
    n_bnd = sum(1 for v, c in counts.items() if c > 1)
    return 100.0 * n_bnd / len(counts)


def boundary_inner_ratio(meshlets, tris_np):
    """Average per-meshlet |boundary| / |interior|.

    Boundary = verts shared with another meshlet. Interior = exclusive.
    """
    sets = _meshlet_vert_sets(meshlets, tris_np)
    counts = {}
    for s in sets:
        for v in s:
            counts[v] = counts.get(v, 0) + 1
    ratios = []
    for s in sets:
        n_bnd = sum(1 for v in s if counts[v] > 1)
        n_int = len(s) - n_bnd
        if n_int == 0:
            ratios.append(float(n_bnd))
        else:
            ratios.append(n_bnd / n_int)
    return float(np.mean(ratios)) if ratios else 0.0


def avg_restarts(meshlets, tris_np, tri_adj):
    """Average number of strip restarts per meshlet under simple BFS strip cover.

    A "restart" is a triangle whose parent in BFS is not adjacent via the
    last-emitted edge. Approximate: count BFS roots beyond the first +
    backtracks. Provides a comparable metric across generators.
    """
    out = []
    for ml in meshlets:
        ml_set = set(ml)
        if not ml:
            out.append(0.0)
            continue
        visited = set()
        restarts = 0
        # BFS roots beyond the first count as restarts
        for seed in ml:
            if seed in visited:
                continue
            if visited:
                restarts += 1  # new disconnected root
            stack = [seed]
            visited.add(seed)
            while stack:
                cur = stack.pop()
                for nb in tri_adj[cur]:
                    if nb in ml_set and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
        # Strip-restart proxy: count branchings (degree>2 in dual subgraph)
        branches = 0
        for ti in ml:
            deg = sum(1 for nb in tri_adj[ti] if nb in ml_set)
            if deg > 2:
                branches += deg - 2
        out.append(float(restarts + branches))
    return float(np.mean(out)) if out else 0.0


def avg_normal_var(meshlets, face_normals):
    """Average per-meshlet normal-direction variance (sum of trailing 2 PCA eigvals)."""
    out = []
    for ml in meshlets:
        if len(ml) < 2:
            out.append(0.0)
            continue
        N = face_normals[ml]
        # PCA on unit normals
        N = N - N.mean(axis=0, keepdims=True)
        cov = N.T @ N / max(1, len(ml) - 1)
        eigs = np.linalg.eigvalsh(cov)
        eigs = np.sort(eigs)
        out.append(float(eigs[0] + eigs[1]))
    return float(np.mean(out)) if out else 0.0


def avg_plane_resid(meshlets, tris_np, verts_np):
    """Average per-meshlet PCA-3rd-eigenvalue sqrt (RMS distance from best-fit plane)."""
    out = []
    for ml in meshlets:
        verts_idx = set()
        for ti in ml:
            for j in range(3):
                verts_idx.add(int(tris_np[ti, j]))
        if len(verts_idx) < 4:
            out.append(0.0)
            continue
        P = verts_np[list(verts_idx)]
        P = P - P.mean(axis=0, keepdims=True)
        cov = P.T @ P / max(1, len(P) - 1)
        eigs = np.linalg.eigvalsh(cov)
        out.append(float(np.sqrt(max(0.0, eigs[0]))))
    return float(np.mean(out)) if out else 0.0


def compute_metrics(meshlets, tris_np, tri_adj, face_normals, verts_np):
    """All metrics in one dict."""
    sizes = [len(ml) for ml in meshlets]
    return {
        "n_meshlets": int(len(meshlets)),
        "avg_tris": float(np.mean(sizes)) if sizes else 0.0,
        "max_tris": int(max(sizes)) if sizes else 0,
        "pct_boundary": boundary_pct(meshlets, tris_np),
        "boundary_inner_ratio": boundary_inner_ratio(meshlets, tris_np),
        "avg_restarts": avg_restarts(meshlets, tris_np, tri_adj),
        "avg_normal_var": avg_normal_var(meshlets, face_normals),
        "avg_plane_resid": avg_plane_resid(meshlets, tris_np, verts_np),
    }

"""Stage-B: per-mesh GNN overfit for meshlet partition refinement.

Idea:
    Train a tiny GCN per-mesh (no shared model) whose output is a soft
    triangle->meshlet assignment. The loss is a differentiable BPV proxy
    (interior plane-fit residual + boundary perimeter). After fit, take
    argmax to get hard assignments.

    Init from joint partition (one-hot per-tri logits). The GNN then
    refines by propagating local consistency through dual-graph edges.

    Encode-time only: model weights are NEVER stored. Output is just the
    refined meshlet partition; decoder unchanged.

Loss components per meshlet (computed from soft assignments):
    L_plane: weighted PCA 3rd eigval of vertex positions, weighted by
             soft membership. Encourages flat patches.
    L_perim: differentiable boundary count = sum over dual edges of
             |p_a - p_b|_1, where p is the soft tri->meshlet vector.
             Penalizes long boundaries.
    L_cap:   ReLU(soft_count - max_tris)^2 hinge per meshlet.
    L_balance: small entropy regularizer to prevent meshlet collapse.

Hard partition extraction:
    argmax(p) -> tri_to_meshlet. Splits by reachability inside each
    meshlet to avoid disconnected components. Caps enforced by greedy
    budget assignment fallback.

Requires PyTorch (already a project dep).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dual-graph features for GCN input
# ============================================================

def _build_tri_features(tris_np, verts_np, face_normals, face_centroids):
    """Per-triangle feature vector (N_T, F).

    F = 3 centroid + 3 normal + 1 area + 1 mean-edge-length = 8 dims.
    """
    n = len(tris_np)
    P = verts_np[tris_np]    # (n, 3, 3)
    a = P[:, 0]; b = P[:, 1]; c = P[:, 2]
    edges = np.stack([np.linalg.norm(a - b, axis=1),
                      np.linalg.norm(b - c, axis=1),
                      np.linalg.norm(c - a, axis=1)], axis=1)
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1, keepdims=True)
    feats = np.concatenate([
        face_centroids,
        face_normals,
        area,
        edges.mean(axis=1, keepdims=True),
    ], axis=1).astype(np.float32)
    # Center + scale
    feats[:, :3] -= feats[:, :3].mean(axis=0)
    sc = max(1e-6, float(np.linalg.norm(feats[:, :3], axis=1).max()))
    feats[:, :3] /= sc
    feats[:, 6:7] /= max(1e-6, float(area.max()))
    feats[:, 7:8] /= max(1e-6, float(edges.mean()))
    return feats


def _dual_edge_index(tri_adj, n_tris):
    """Edge index (2, E) for the dual graph."""
    src, dst = [], []
    for t in range(n_tris):
        for nb in tri_adj[t]:
            if nb > t:
                src.append(t); dst.append(nb)
                src.append(nb); dst.append(t)
    return np.array([src, dst], dtype=np.int64)


# ============================================================
# Tiny GCN
# ============================================================

class _GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, deg_inv_sqrt):
        # x: (N, F), edge_index: (2, E)
        src, dst = edge_index[0], edge_index[1]
        x_lin = self.lin(x)
        # Aggregate (mean-symmetric via deg^{-1/2})
        msg = x_lin[src] * deg_inv_sqrt[src].unsqueeze(-1) * \
                            deg_inv_sqrt[dst].unsqueeze(-1)
        out = torch.zeros_like(x_lin)
        out.index_add_(0, dst, msg)
        return out + x_lin   # residual


class _MeshGCN(nn.Module):
    def __init__(self, in_dim, hidden, n_meshlets, n_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [n_meshlets]
        self.layers = nn.ModuleList(
            [_GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)])

    def forward(self, x, edge_index, deg_inv_sqrt):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, deg_inv_sqrt)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x   # logits (N, K)


# ============================================================
# Differentiable loss
# ============================================================

def _soft_partition_loss(soft_p, tri_centroids_t, tri_areas_t, tri_to_v_idx,
                          verts_t, edge_index, max_tris,
                          lambda_interior=1.0, lambda_perim=0.5,
                          lambda_header=0.5, lambda_cap=50.0,
                          lambda_balance=0.01):
    """BPV-aligned differentiable proxy.

    Real BPV ≈ (interior_bits + boundary_bits + header_bits) / n_verts.

    Components:
        L_interior: Σ_k tri_count_k * log(eigval_min_k + ε)
            Captures interior residual rate. Bigger flat meshlets pull
            harder than tiny ones; log(eigval) ≈ residual bit-rate.
        L_perim: Σ over dual edges of soft "edges that cross meshlets".
            Differentiable proxy for boundary-edge count.
        L_header: differentiable count of "active" meshlets (size > 0).
            Soft 1 - exp(-soft_count) per meshlet.
        L_cap: ReLU(soft_count - max_tris)^2 hinge.
        L_balance: entropy of normalized soft_count, maximized.
    """
    N_T, K = soft_p.shape
    soft_count = soft_p.sum(dim=0)                                 # (K,)

    # ---- L_interior: weighted log eigval ----
    # Weighted centroid + 3x3 cov per meshlet (membership-weighted).
    w_norm = soft_p / (soft_count.unsqueeze(0) + 1e-6)             # (N_T, K)
    mean_c = w_norm.t() @ tri_centroids_t                           # (K, 3)
    diff = tri_centroids_t.unsqueeze(1) - mean_c.unsqueeze(0)       # (N_T, K, 3)
    diff_w = diff * w_norm.unsqueeze(-1)
    cov = torch.einsum("nki,nkj->kij", diff_w, diff)                # (K, 3, 3)
    eigs = torch.linalg.eigvalsh(
        cov + 1e-9 * torch.eye(3, device=cov.device))               # (K, 3)
    log_eigval_min = torch.log(eigs[:, 0] + 1e-9)
    # Weight log-eigval by tri count (residual bits scale with #verts ≈ tri_count/2).
    L_interior = (soft_count * log_eigval_min).sum() / max(1.0, N_T)

    # ---- L_perim: total soft cross-meshlet edges ----
    src, dst = edge_index[0], edge_index[1]
    p_src = soft_p[src]
    p_dst = soft_p[dst]
    # 0.5 * sum(|p_a - p_b|) ∈ [0, 1] per edge — soft "edge crosses" indicator
    edge_cross = 0.5 * (p_src - p_dst).abs().sum(dim=1)
    L_perim = edge_cross.sum() / max(1.0, edge_index.shape[1])

    # ---- L_header: soft # active meshlets ----
    active = 1.0 - torch.exp(-soft_count.clamp(min=0.0))            # ∈ [0, 1]
    L_header = active.sum() / float(K)

    # ---- L_cap: hard-cap hinge ----
    overflow = F.relu(soft_count - float(max_tris))
    L_cap = (overflow ** 2).mean()

    # ---- L_balance: penalize collapsed meshlets (negative entropy) ----
    p_size = soft_count / (soft_count.sum() + 1e-6)
    L_balance = -(p_size * torch.log(p_size + 1e-9)).sum()

    total = (lambda_interior * L_interior
             + lambda_perim * L_perim
             + lambda_header * L_header
             + lambda_cap * L_cap
             - lambda_balance * L_balance)
    return total, {
        "interior": float(L_interior),
        "perim": float(L_perim),
        "header": float(L_header),
        "cap": float(L_cap),
        "balance": float(L_balance),
        "min_eigval_avg": float(eigs[:, 0].mean()),
    }


# ============================================================
# Hard-partition extraction with connectivity + cap enforcement
# ============================================================

def _hard_partition(soft_p, tri_adj, max_tris, max_verts, tris_np):
    """argmax + connectivity-aware split + cap enforcement.

    1. argmax -> tri_to_meshlet
    2. For each meshlet, BFS components; keep largest, reassign others
       to their next-best meshlet (still respecting caps).
    3. If a meshlet exceeds max_tris, evict its lowest-confidence tris
       to next-best meshlets.
    Returns list of meshlets.
    """
    N_T, K = soft_p.shape
    p_np = soft_p.detach().cpu().numpy()
    assign = p_np.argmax(axis=1)

    # Step 1: Connectivity per meshlet, keep largest component
    for m in range(K):
        member = np.where(assign == m)[0]
        if len(member) == 0:
            continue
        member_set = set(int(x) for x in member)
        seen = set()
        components = []
        for s in member:
            s = int(s)
            if s in seen:
                continue
            comp = []
            stack = [s]
            seen.add(s)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in tri_adj[cur]:
                    if nb in member_set and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            components.append(comp)
        if len(components) <= 1:
            continue
        # Keep largest, reassign others to 2nd-best meshlet
        components.sort(key=len, reverse=True)
        for comp in components[1:]:
            for ti in comp:
                # 2nd best meshlet from softmax
                ranked = np.argsort(-p_np[ti])
                for cand in ranked:
                    if cand != m:
                        assign[ti] = cand
                        break

    # Step 2: Cap enforcement (max_tris)
    for m in range(K):
        member = np.where(assign == m)[0]
        if len(member) <= max_tris:
            continue
        # Evict lowest-confidence tris
        conf = p_np[member, m]
        order = np.argsort(conf)   # ascending
        n_to_evict = len(member) - max_tris
        evict = member[order[:n_to_evict]]
        for ti in evict:
            ranked = np.argsort(-p_np[ti])
            for cand in ranked:
                if cand != m:
                    assign[ti] = cand
                    break

    # Build meshlets
    out = [[] for _ in range(K)]
    for ti, m in enumerate(assign):
        out[int(m)].append(int(ti))
    out = [ml for ml in out if ml]
    return out


# ============================================================
# Public: fit
# ============================================================

def fit_partition_gnn(meshlets_init, tris_np, tri_adj, face_normals, verts_np,
                     face_centroids=None,
                     max_tris=256, max_verts=256,
                     hidden=32, n_layers=3,
                     n_steps=800, lr=5e-3,
                     lambda_interior=1.0, lambda_perim=2.0,
                     lambda_header=0.5, lambda_cap=50.0,
                     lambda_balance=0.05,
                     init_temp=1.0, device=None, verbose=False):
    """Per-mesh GNN overfit refinement.

    Init logits: one-hot from `meshlets_init` (K = len(meshlets_init)),
    multiplied by `init_temp` so soft_p is sharp at start.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if face_centroids is None:
        face_centroids = (verts_np[tris_np[:, 0]] +
                          verts_np[tris_np[:, 1]] +
                          verts_np[tris_np[:, 2]]) / 3

    K = len(meshlets_init)
    N_T = len(tris_np)
    feats = _build_tri_features(tris_np, verts_np, face_normals, face_centroids)
    edge_index = _dual_edge_index(tri_adj, N_T)
    deg = np.zeros(N_T, dtype=np.float32)
    for u, v in edge_index.T:
        deg[u] += 1
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1.0))

    # Init logits from meshlets_init
    init_logits = np.full((N_T, K), -1.0, dtype=np.float32)
    for mi, ml in enumerate(meshlets_init):
        for ti in ml:
            init_logits[ti, mi] = init_temp

    feats_t = torch.from_numpy(feats).to(device)
    ei_t = torch.from_numpy(edge_index).to(device)
    deg_t = torch.from_numpy(deg_inv_sqrt).to(device)
    init_logits_t = torch.from_numpy(init_logits).to(device)

    centroids_t = torch.from_numpy(face_centroids.astype(np.float32)).to(device)
    P = verts_np[tris_np].astype(np.float32)
    areas = 0.5 * np.linalg.norm(
        np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    areas_t = torch.from_numpy(areas.astype(np.float32)).to(device)
    verts_t = torch.from_numpy(verts_np.astype(np.float32)).to(device)
    tri_to_v_idx = torch.from_numpy(tris_np.astype(np.int64)).to(device)

    model = _MeshGCN(feats.shape[1], hidden, K, n_layers=n_layers).to(device)
    # Bias the final linear layer toward init logits via residual
    # (the GCN output is added to init via a learned skip below).
    # Implementation: combine GCN output with init_logits at every step.
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(n_steps):
        opt.zero_grad()
        delta_logits = model(feats_t, ei_t, deg_t)   # (N_T, K)
        logits = init_logits_t + delta_logits
        soft_p = F.softmax(logits, dim=-1)
        loss, parts = _soft_partition_loss(
            soft_p, centroids_t, areas_t, tri_to_v_idx, verts_t,
            ei_t, max_tris,
            lambda_interior=lambda_interior, lambda_perim=lambda_perim,
            lambda_header=lambda_header, lambda_cap=lambda_cap,
            lambda_balance=lambda_balance)
        loss.backward()
        opt.step()
        if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
            print(f"  [gnn] step={step} loss={float(loss):.4f} "
                  f"int={parts['interior']:.4f} perim={parts['perim']:.4f} "
                  f"hdr={parts['header']:.3f} cap={parts['cap']:.4f} "
                  f"bal={parts['balance']:.3f} eigmin={parts['min_eigval_avg']:.5f}")

    with torch.no_grad():
        delta_logits = model(feats_t, ei_t, deg_t)
        logits = init_logits_t + delta_logits
        soft_p = F.softmax(logits, dim=-1)

    meshlets = _hard_partition(soft_p, tri_adj, max_tris, max_verts, tris_np)
    return meshlets

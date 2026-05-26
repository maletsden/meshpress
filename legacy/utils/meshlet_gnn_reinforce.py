"""Straight-through REINFORCE GNN refinement.

Approach:
    Per-mesh GNN outputs categorical logits over K meshlets per tri.
    Forward: sample hard partition (Gumbel argmax). Use it via
        encoder to get true BPV. Backward: pretend the assignment is
        differentiable via straight-through (gradient of hard sample
        ≈ gradient of softmax).
    Reward = -BPV_sampled. Update via REINFORCE with a moving-mean
    baseline to reduce variance:
        loss = -log_prob(sampled_assignment) * (R - baseline)

Each step does ONE real encode → expensive but truthful. Use small
n_steps (~30-50) with EMA baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.meshlet_gnn_fit import (
    _MeshGCN, _build_tri_features, _dual_edge_index,
    _hard_partition,
)


def _meshlets_from_assign(assign, K):
    out = [[] for _ in range(K)]
    for ti, m in enumerate(assign):
        out[int(m)].append(int(ti))
    return [ml for ml in out if ml]


def _enforce_caps_and_connectivity(assign, K, tri_adj, tris_np, max_tris, max_verts,
                                    soft_p):
    """Clip assignment to respect caps + connectivity, mirroring _hard_partition."""
    p_np = soft_p.detach().cpu().numpy() if hasattr(soft_p, "detach") else soft_p
    # Simple version: use existing _hard_partition logic
    soft_t = torch.from_numpy(p_np) if not isinstance(soft_p, torch.Tensor) else soft_p
    return _hard_partition(soft_t, tri_adj, max_tris, max_verts, tris_np)


def fit_partition_reinforce(meshlets_init, tris_np, tri_adj, face_normals, verts_np,
                            face_centroids,
                            encode_fn,
                            max_tris=256, max_verts=256,
                            hidden=32, n_layers=3,
                            n_steps=40, lr=1e-3,
                            n_samples_per_step=1,
                            entropy_coef=0.001,
                            init_temp=8.0,
                            device=None, verbose=False):
    """REINFORCE-train GNN with real BPV reward.

    encode_fn(meshlets) -> bpv_float : encoder callback.
    Returns: best meshlets found (lowest BPV).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    K = len(meshlets_init)
    N_T = len(tris_np)

    feats = _build_tri_features(tris_np, verts_np, face_normals, face_centroids)
    edge_index = _dual_edge_index(tri_adj, N_T)
    deg = np.zeros(N_T, dtype=np.float32)
    for u, v in edge_index.T:
        deg[u] += 1
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1.0))

    init_logits = np.full((N_T, K), 0.0, dtype=np.float32)
    for mi, ml in enumerate(meshlets_init):
        for ti in ml:
            init_logits[ti, mi] = init_temp

    feats_t = torch.from_numpy(feats).to(device)
    ei_t = torch.from_numpy(edge_index).to(device)
    deg_t = torch.from_numpy(deg_inv_sqrt).to(device)
    init_logits_t = torch.from_numpy(init_logits).to(device)

    model = _MeshGCN(feats.shape[1], hidden, K, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Baseline = init BPV (fixed anchor). Prevents drift up when samples
    # are bad, which would otherwise reinforce sub-init policy.
    best_bpv = float("inf")
    best_meshlets = meshlets_init

    bpv_init = encode_fn(meshlets_init)
    baseline = bpv_init
    if verbose:
        print(f"  [rl] init bpv={bpv_init:.3f} (baseline fixed at this)")

    for step in range(n_steps):
        delta_logits = model(feats_t, ei_t, deg_t)
        logits = init_logits_t + delta_logits
        log_probs = F.log_softmax(logits, dim=-1)            # (N_T, K)
        probs = log_probs.exp()

        # Sample N hard partitions; pick one for the gradient step
        rewards = []
        sampled_assigns = []
        log_probs_list = []
        for _s in range(n_samples_per_step):
            # Categorical sample
            cat = torch.distributions.Categorical(probs=probs)
            assign_t = cat.sample()                          # (N_T,)
            assign = assign_t.detach().cpu().numpy().astype(np.int64)

            # Enforce caps + connectivity (uses soft_p for tie-breaking)
            meshlets = _enforce_caps_and_connectivity(
                assign, K, tri_adj, tris_np, max_tris, max_verts, probs)
            if not meshlets:
                continue

            bpv = encode_fn(meshlets)
            if bpv < best_bpv:
                best_bpv = bpv
                best_meshlets = meshlets

            log_p = log_probs.gather(1, assign_t.unsqueeze(-1)).squeeze(-1).sum()
            rewards.append(bpv)
            sampled_assigns.append(assign_t)
            log_probs_list.append(log_p)

        if not rewards:
            continue

        mean_r = float(np.mean(rewards))

        opt.zero_grad()
        # Advantage: positive if BPV lower than init; negative if worse.
        # No EMA — fixed baseline at init prevents drift-reinforcement.
        loss = 0.0
        for log_p, r in zip(log_probs_list, rewards):
            adv = baseline - r          # >0 = better than init, reinforce
            loss = loss + (-log_p * adv)
        loss = loss / len(rewards)

        # Entropy bonus (encourages exploration)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = loss - entropy_coef * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if verbose:
            print(f"  [rl] step={step} bpv_mean={mean_r:.3f} "
                  f"baseline={baseline:.3f} best={best_bpv:.3f} "
                  f"loss={float(loss):.4f} ent={float(entropy):.3f}")

    if verbose:
        print(f"  [rl] done init={bpv_init:.3f} best={best_bpv:.3f} "
              f"delta={best_bpv - bpv_init:+.3f}")
    return best_meshlets

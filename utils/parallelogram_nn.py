"""Neural bias correction on top of parallelogram prediction (idea #11).

Tiny MLP per mesh: predicts the curvature bias between the linear
parallelogram prediction (a + b - c) and the true vertex position,
expressed in a local frame anchored on the predicting triangle.

Frame for triangle (a, b, c) -> apex v:
    origin = a + b - c (the parallelogram point)
    t      = (b - a) / |b - a|
    n      = (b - a) x (c - a), normalized   (triangle normal)
    s      = n x t                            (binormal)

Features per sample (rotation- and translation-invariant scalars):
    [d_ab, d_ac, d_bc, cos_apex_at_c, area_T']
plus their pairwise products / log scales when needed.

Output: 3 scalars (bias_t, bias_n, bias_s).
World-space bias = bias_t * t + bias_n * n + bias_s * s.

Training: encode-time, ~300 Adam steps on per-vertex (features, true_bias)
pairs. Weights quantized to int8 + per-tensor float32 scale, shipped in
the per-mesh header (~200 B for the architecture below).

Inference: pure numpy matmul. Decoder replicates bit-exactly via dequant
weights, so no float-vs-quant drift between encoder and decoder.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Feature + frame extraction
# ============================================================

N_FEATURES = 13
N_HIDDEN = 16
N_OUT = 3


def _safe_norm(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps), n.squeeze(-1)


def build_frame_and_features(a, b, c, d_ac=None, d_bc=None):
    """Vectorised over leading dim. a, b, c : (..., 3) float64.

    If d_ac and d_bc are provided (both required), feature vector has 13
    entries: 7 base + 6 second-ring (each apex projected into local frame).
    Otherwise feats has 7 entries — caller is responsible for not feeding
    these to a 13-feature MLP.

    Returns:
        origin : (..., 3)   parallelogram base point a + b - c
        T      : (..., 3, 3) frame columns [t, n, s] (world<-local matrix)
        feats  : (..., 7) or (..., 13)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    e_ab = b - a
    e_ac = c - a
    e_bc = c - b
    d_ab_n = np.linalg.norm(e_ab, axis=-1)
    d_ac_n = np.linalg.norm(e_ac, axis=-1)
    d_bc_n = np.linalg.norm(e_bc, axis=-1)
    t, _ = _safe_norm(e_ab)
    nrm = np.cross(e_ab, e_ac)
    n, _ = _safe_norm(nrm)
    s = np.cross(n, t)
    cos_apex = np.einsum('...i,...i->...', -e_ac, -e_bc) / np.maximum(
        d_ac_n * d_bc_n, 1e-12)
    area = 0.5 * np.linalg.norm(nrm, axis=-1)
    base_feats = [
        d_ab_n, d_ac_n, d_bc_n,
        d_ac_n / np.maximum(d_ab_n, 1e-12),
        d_bc_n / np.maximum(d_ab_n, 1e-12),
        cos_apex,
        area,
    ]
    origin = a + b - c
    T = np.stack([t, n, s], axis=-1)  # columns
    if d_ac is None or d_bc is None:
        feats = np.stack(base_feats, axis=-1)
        return origin, T, feats

    d_ac_p = np.asarray(d_ac, dtype=np.float64)
    d_bc_p = np.asarray(d_bc, dtype=np.float64)
    # Project second-ring apexes into local frame, anchored at origin.
    rel_ac = d_ac_p - origin
    rel_bc = d_bc_p - origin
    # T columns are world<-local, so local = T.T @ world.
    proj_ac = np.einsum('...ji,...j->...i', T, rel_ac)
    proj_bc = np.einsum('...ji,...j->...i', T, rel_bc)
    extra = [proj_ac[..., 0], proj_ac[..., 1], proj_ac[..., 2],
             proj_bc[..., 0], proj_bc[..., 1], proj_bc[..., 2]]
    feats = np.stack(base_feats + extra, axis=-1)
    return origin, T, feats


# ============================================================
# MLP (numpy inference)
# ============================================================

class TinyMLP:
    """Single hidden layer, ReLU, no bias on output."""

    def __init__(self, W1, b1, W2, b2):
        # W1 : (N_HIDDEN, N_FEATURES)  b1 : (N_HIDDEN,)
        # W2 : (N_OUT, N_HIDDEN)        b2 : (N_OUT,)
        self.W1 = W1.astype(np.float64)
        self.b1 = b1.astype(np.float64)
        self.W2 = W2.astype(np.float64)
        self.b2 = b2.astype(np.float64)

    def forward(self, X):
        h = X @ self.W1.T + self.b1
        h = np.maximum(h, 0.0)
        return h @ self.W2.T + self.b2

    def n_params(self):
        return (self.W1.size + self.b1.size +
                self.W2.size + self.b2.size)


def quantize_weights(mlp):
    """fp32 storage. fp16 overflows when feature-normalisation is baked
    into W1 (per-feature inv_std can hit 1e6 for nearly-constant features),
    and int8 per-tensor was too coarse for the small target scales. fp32
    is exact at 4 B/weight which is still negligible vs the per-mesh
    bitstream (~780 B total header).
    """
    out = {}
    total_bits = 0
    for name, W in (('W1', mlp.W1), ('b1', mlp.b1),
                    ('W2', mlp.W2), ('b2', mlp.b2)):
        q = W.astype(np.float32)
        out[name] = q
        total_bits += q.size * 32
    return out, total_bits


def dequantize_weights(qdict):
    W1 = qdict['W1'].astype(np.float64)
    b1 = qdict['b1'].astype(np.float64)
    W2 = qdict['W2'].astype(np.float64)
    b2 = qdict['b2'].astype(np.float64)
    return TinyMLP(W1, b1, W2, b2)


# ============================================================
# Training (PyTorch)
# ============================================================

def train_bias_mlp(features_np, targets_np, n_steps=300, lr=1e-2,
                    seed=0, verbose=False):
    """features_np : (N, N_FEATURES)  targets_np : (N, 3) bias in local frame.
    Returns a numpy TinyMLP."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(features_np) == 0:
        # Zero-init, will produce zero bias -> reduces to plain parallelogram.
        return TinyMLP(
            np.zeros((N_HIDDEN, N_FEATURES)),
            np.zeros(N_HIDDEN),
            np.zeros((N_OUT, N_HIDDEN)),
            np.zeros(N_OUT),
        )

    X = torch.tensor(features_np, dtype=torch.float64)
    Y = torch.tensor(targets_np, dtype=torch.float64)

    # Normalize features per-dim to unit variance to keep the MLP well-scaled.
    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0).clamp(min=1e-6)
    X_norm = (X - x_mean) / x_std

    # Normalize targets per-dim so loss + grads are O(1). Inverse scale is
    # baked back into the output layer at the end so inference produces
    # bias in the original (un-normalized) frame.
    y_std = Y.std(dim=0).clamp(min=1e-12)
    Y_norm = Y / y_std

    # Zero-init output layer -> initial bias = 0 -> initial behaviour is
    # identical to plain parallelogram. Training only moves away from
    # zero if it actually reduces loss, so the worst case is "no NN".
    W1 = torch.zeros(N_HIDDEN, N_FEATURES, dtype=torch.float64,
                     requires_grad=True)
    b1 = torch.zeros(N_HIDDEN, dtype=torch.float64, requires_grad=True)
    W2 = torch.zeros(N_OUT, N_HIDDEN, dtype=torch.float64,
                     requires_grad=True)
    b2 = torch.zeros(N_OUT, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        W1.normal_(0, np.sqrt(2.0 / N_FEATURES))
        # W2, b2 stay zero: starts as identity.

    opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
    for it in range(n_steps):
        opt.zero_grad()
        h = (X_norm @ W1.T + b1).clamp(min=0)
        out = h @ W2.T + b2
        loss = ((out - Y_norm) ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and (it == 0 or it == n_steps - 1):
            print(f"    NN step {it}: loss={float(loss):.6f}")

    # Bake feature normalization into W1 / b1, target std into W2 / b2.
    inv_std = (1.0 / x_std).numpy()
    mean_np = x_mean.numpy()
    W1_np = W1.detach().numpy() * inv_std[None, :]
    b1_np = b1.detach().numpy() - W1.detach().numpy() @ (mean_np / x_std.numpy())
    y_std_np = y_std.numpy()
    W2_np = W2.detach().numpy() * y_std_np[:, None]
    b2_np = b2.detach().numpy() * y_std_np
    return TinyMLP(W1_np, b1_np, W2_np, b2_np)

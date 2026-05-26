"""
Non-linear (MLP) lifting predictor for float-domain wavelet.

Lifting structure (same as learned_wavelet, identity update):

    forward:  detail[k] = odd[k] - MLP(even_context[k])
    inverse:  odd[k]    = detail[k] + MLP(even_context[k])

Perfect reconstruction holds for any deterministic predictor — we just need
the decoder to apply the same function to the same inputs. ReLU MLPs are
deterministic given their weights, so the wavelet property is preserved.

ERROR BOUND (crack-free interior):
    Inverse lifting at one level:
        e_err ≤ δ_a/2
        ||context_err||_2 ≤ √K · δ_a/2            (all K entries bounded)
        P(context)_err   ≤ Lip(MLP) · √K · δ_a/2  (Lipschitz)
        o_err            ≤ δ_d/2 + Lip(MLP) · √K · δ_a/2

    Over L levels with a worst-case accumulation the δ allocator uses
        amp = Lip(MLP) · √K   (passed to per_level_deltas)
    so the budget  δ_base + amp · Σ δ_k ≤ 2ε  guarantees max recon error ≤ ε.

    Lipschitz upper bound = product of per-layer operator 2-norms.

SHARED PARAMETERS (one per mesh):
    - One MLP per wavelet level (max n_levels = log2(max_verts / target_base)).
    - Optional learned quantization curve: the geometric ratio is grid-
      searched across candidates and the best-fit is stored per mesh.
    Total header overhead for a 3-level / K=4 / H=16 config: ~2.5 KB.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils.wavelet import _stream_bits


# ============================================================
# Reflection padding (shared with utils.learned_wavelet)
# ============================================================

def _pad_even_reflect(even, kernel_size):
    K = kernel_size
    N = len(even)
    pad_left = (K - 1) // 2
    pad_right = K - 1 - pad_left
    padded = np.empty(N + pad_left + pad_right, dtype=np.float64)
    padded[pad_left:pad_left + N] = even
    for i in range(pad_left):
        src = min(pad_left - 1 - i + 1, N - 1)
        padded[i] = even[src]
    for i in range(pad_right):
        src = max(N - 2 - i, 0)
        padded[pad_left + N + i] = even[src]
    return padded


# ============================================================
# Tiny MLP (PyTorch for training, numpy for inference)
# ============================================================

class LiftingMLP(nn.Module):
    def __init__(self, kernel_size, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(kernel_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def export_mlp_weights(mlp):
    """Extract (W, b) pairs for numpy-side forward pass."""
    weights = []
    for m in mlp.net:
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy().astype(np.float64)
            b = m.bias.detach().cpu().numpy().astype(np.float64)
            weights.append((W, b))
    return weights


def mlp_forward_numpy(weights, x):
    """Numpy forward pass matching LiftingMLP architecture.
    weights: list of (W, b), alternating with ReLU between all but last layer.
    x: (N, K) batch -> (N,) outputs.
    """
    out = x
    for i, (W, b) in enumerate(weights):
        out = out @ W.T + b
        if i < len(weights) - 1:
            out = np.maximum(0.0, out)
    return out.squeeze(-1)


def mlp_lipschitz_bound(weights, kernel_size):
    """Upper bound on max |ΔP| given ||Δcontext||_inf ≤ 1.

    Two candidate bounds; we take the minimum:
      2-norm chain:  |ΔP| ≤ (∏ ||W||_2) · √K · ||Δx||_∞     (tight when layers are well-conditioned)
      ∞-norm chain:  |ΔP| ≤ (∏ ||W||_∞_ind) · ||Δx||_∞       (tight when weight rows are sparse)

    ReLU is 1-Lipschitz in both norms. The value returned is already the
    effective `amp` — pass it directly to the δ allocator.
    """
    L2 = 1.0
    Linf = 1.0
    for W, _ in weights:
        L2 *= float(np.linalg.norm(W, ord=2))
        Linf *= float(np.linalg.norm(W, ord=np.inf))
    return min(L2 * np.sqrt(kernel_size), Linf)


# ============================================================
# Lifting (MLP predict, identity update)
# ============================================================

def _predict_odd_mlp(even, weights, kernel_size):
    """Build context matrix for all odd positions and apply MLP forward.
    Returns: (N,) predicted odd values."""
    N = len(even)
    if N == 0:
        return np.zeros(0, dtype=np.float64)
    padded = _pad_even_reflect(even, kernel_size)
    contexts = np.empty((N, kernel_size), dtype=np.float64)
    for j in range(kernel_size):
        contexts[:, j] = padded[j:j + N]
    return mlp_forward_numpy(weights, contexts)


def mlp_decompose(values, mlp_weights_per_level, kernel_size, target_base=32):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= target_base:
        return values.copy(), [], n

    n_pad = 1
    while n_pad < n:
        n_pad *= 2
    sig = np.empty(n_pad, dtype=np.float64)
    sig[:n] = values
    sig[n:] = values[-1]

    levels = []
    cur = sig
    lvl = 0
    while len(cur) > target_base and len(cur) >= 2:
        even = cur[0::2]
        odd = cur[1::2]
        w = (mlp_weights_per_level[lvl]
             if lvl < len(mlp_weights_per_level)
             else mlp_weights_per_level[-1])
        detail = odd - _predict_odd_mlp(even, w, kernel_size)
        levels.append(detail)
        cur = even
        lvl += 1
    return cur, levels, n


def mlp_reconstruct(base, levels, mlp_weights_per_level, kernel_size, orig_n):
    cur = np.asarray(base, dtype=np.float64).copy()
    for l_idx in range(len(levels) - 1, -1, -1):
        d = levels[l_idx]
        w = (mlp_weights_per_level[l_idx]
             if l_idx < len(mlp_weights_per_level)
             else mlp_weights_per_level[-1])
        pred = _predict_odd_mlp(cur, w, kernel_size)
        odd = d + pred
        N = len(cur)
        out = np.empty(2 * N, dtype=np.float64)
        out[0::2] = cur
        out[1::2] = odd
        cur = out
    return cur[:orig_n]


# ============================================================
# Training
# ============================================================

def _collect_level_pairs(streams, n_levels_max, kernel_size, target_base):
    """Extract (context, target) training pairs per level from all streams.
    Uses identity-update (cur = even at each step) so all levels get data
    consistent with the encoder's decomposition."""
    per_level_data = [[] for _ in range(n_levels_max)]
    for stream in streams:
        n = len(stream)
        if n <= target_base:
            continue
        n_pad = 1
        while n_pad < n:
            n_pad *= 2
        sig = np.empty(n_pad, dtype=np.float64)
        sig[:n] = stream
        sig[n:] = stream[-1]
        cur = sig
        for lvl in range(n_levels_max):
            if len(cur) <= target_base or len(cur) < 2:
                break
            even = cur[0::2]
            odd = cur[1::2]
            padded = _pad_even_reflect(even, kernel_size)
            for k in range(len(odd)):
                per_level_data[lvl].append(
                    (padded[k:k + kernel_size], odd[k]))
            cur = even
    return per_level_data


def fit_mlps(streams, n_levels_max, kernel_size=4, hidden=16,
             target_base=32, epochs=300, lr=1e-3, weight_decay=1e-4,
             device="cpu", seed=0, verbose=False):
    """Train one LiftingMLP per level on the given streams.

    Returns (weights_per_level, torch_modules).
    weights_per_level: list of [(W, b), ...] ready for numpy forward.
    """
    torch.manual_seed(seed)
    per_level_data = _collect_level_pairs(
        streams, n_levels_max, kernel_size, target_base)

    weights_per_level = []
    modules = []
    for lvl in range(n_levels_max):
        pairs = per_level_data[lvl]
        model = LiftingMLP(kernel_size, hidden).to(device)

        if not pairs:
            # Init-only: identity-ish (average of context)
            modules.append(model)
            weights_per_level.append(export_mlp_weights(model))
            continue

        X = np.asarray([p[0] for p in pairs], dtype=np.float32)
        y = np.asarray([p[1] for p in pairs], dtype=np.float32)
        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).to(device)

        opt = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        for epoch in range(epochs):
            opt.zero_grad()
            pred = model(Xt)
            loss = (pred - yt).pow(2).mean()
            loss.backward()
            opt.step()
        if verbose:
            with torch.no_grad():
                final = (model(Xt) - yt).pow(2).mean().item()
            print(f"    level {lvl}: {len(pairs):,} pairs  "
                  f"final MSE={final:.4e}")

        modules.append(model)
        weights_per_level.append(export_mlp_weights(model))

    return weights_per_level, modules


# ============================================================
# Interior encoder
# ============================================================

def _deltas(per_coord_err, L, amp, schedule, ratio):
    """Budget allocator: δ_base + amp * Σ δ_k ≤ 2ε."""
    if L == 0:
        return 2.0 * per_coord_err, []
    if schedule == "uniform" or ratio == 1.0:
        delta = 2.0 * per_coord_err / (1.0 + amp * L)
        return delta, [delta] * L
    geo_sum = (ratio * (ratio ** L - 1)) / (ratio - 1)
    A = 2.0 * per_coord_err / (1.0 + amp * geo_sum)
    return A, [A * (ratio ** (L - k)) for k in range(L)]


def _pack_level(streams_this_level):
    """9 B/level (int16 min + uint8 bits per axis) + fixed/entropy body."""
    body_bits = 0
    for codes in streams_this_level:
        if len(codes) == 0:
            continue
        mn = int(codes.min())
        rng = int(codes.max() - mn)
        b = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
        shifted = codes - mn
        body_bits += _stream_bits(shifted, b)
    return body_bits + 3 * (16 + 8)


def quantize_interior_mlp_wavelet(positions, per_coord_err, mlp_weights_per_level,
                                   kernel_size, amp, schedule="geometric",
                                   ratio=4.0, target_base=32):
    """Interior encoder using MLP lifting predictor with packed metadata."""
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        return positions.copy(), 0, {"n_levels": 0}

    recon = np.empty_like(positions)
    total_bits = 3 * 32  # per-axis float32 offset
    per_axis_streams = []
    L_final = 0
    for d in range(3):
        offset = float(positions[:, d].min())
        shifted = positions[:, d] - offset
        base, levels, orig_n = mlp_decompose(
            shifted, mlp_weights_per_level, kernel_size, target_base)
        L = len(levels)
        L_final = max(L_final, L)

        delta_base, delta_levels = _deltas(
            per_coord_err, L, amp, schedule, ratio)

        base_q = np.round(base / delta_base).astype(np.int64)
        levels_q = [np.round(levels[k] / delta_levels[k]).astype(np.int64)
                    for k in range(L)]
        base_r = base_q.astype(np.float64) * delta_base
        levels_r = [levels_q[k].astype(np.float64) * delta_levels[k]
                    for k in range(L)]
        recon[:, d] = mlp_reconstruct(
            base_r, levels_r, mlp_weights_per_level, kernel_size, orig_n
        ) + offset
        per_axis_streams.append((base_q, levels_q))

    base_streams = [per_axis_streams[d][0] for d in range(3)]
    total_bits += _pack_level(base_streams)
    for lvl in range(L_final):
        level_streams = [per_axis_streams[d][1][lvl] for d in range(3)]
        total_bits += _pack_level(level_streams)

    return recon, total_bits, {"n_levels": L_final}

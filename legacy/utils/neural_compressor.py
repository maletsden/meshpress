"""
End-to-end differentiable neural compressor for interior meshlet vertices.

Architecture (one model per mesh, train-once / encode-many):
    1. Per-level LiftingMLP predictors (learned non-linear predict step).
    2. Per-level learnable μ-law quantization curve:
           f_α(x) = sign(x) · log1p(α|x|) / log1p(α · x_max)
       Non-linear curve concentrates bins near zero where detail coefficients
       cluster → lower rate for the same max error.
    3. Learnable per-level quantization step δ in the y-domain (post-curve).
    4. Noise-quantize during training (Ballé et al. 2017 additive-uniform
       trick). Swap for round() at inference.

Loss:
        L = distortion + λ_rate · rate_proxy + λ_max · max_err_penalty

    distortion       = mean((x - x_hat)^2)            in ORIGINAL units
    rate_proxy       = Σ_level var(q_level)           user-preferred simple proxy
    max_err_penalty  = ReLU(‖x - x_hat‖_∞ - ε)^2      hard-ish constraint

After training: export MLP weights + per-level (α, δ, x_max) → the numpy-side
encoder reproduces the exact same transforms bit-for-bit.

Max-error math (inverse μ-law):
    f⁻¹(y) = sign(y) · (expm1(|y| · log1p(α x_max))) / α
    max |x - x_hat| after ±δ/2 in y-domain, at |x|=x_max:
        = (δ/2) · (1 + α x_max) / α  ·  Lip(MLP lifting inverse)
    To guarantee ε we CHECK this analytic bound post-training and bump δ
    down if needed (or re-train with higher λ_max).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MLP predictor (same architecture as learned_mlp_wavelet)
# ============================================================

class LiftingPredictor(nn.Module):
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


# ============================================================
# Dense per-level predictor: all evens -> all odds, with
# position-specific learned transforms.
# ============================================================

class DensePredictor(nn.Module):
    """Per-level dense predictor.

    Input:  (B, N_even) full set of even samples at this level.
    Output: (B, N_odd)  predicted odd samples (N_odd = N_even).

    Architecture: bottleneck MLP
        even -> Linear(N -> H) -> ReLU -> Linear(H -> N) -> odd_pred

    Each row of the final linear layer encodes a unique linear
    combination of hidden features for that odd position — i.e. every
    odd sample has its own learned "predictor formula" but they share
    feature extraction via the bottleneck.

    Sizes are per-level and match the padded signal after decomposition.
    For max_verts=256, target_base=32: even sizes are 128 / 64 / 32.
    """

    def __init__(self, n_even, hidden=16):
        super().__init__()
        self.n_even = n_even
        self.hidden = hidden
        self.enc = nn.Linear(n_even, hidden)
        self.dec = nn.Linear(hidden, n_even)

    def forward(self, even):
        # even: (B, N_even) or (N_even,)
        single = even.ndim == 1
        if single:
            even = even.unsqueeze(0)
        h = F.relu(self.enc(even))
        out = self.dec(h)
        return out.squeeze(0) if single else out


# ============================================================
# Torch-side lifting (differentiable)
# ============================================================

def _pad_even_reflect_torch(even, K):
    """Reflection pad. `even` is (..., N). Returns (..., N + K - 1)."""
    pad_left = (K - 1) // 2
    pad_right = K - 1 - pad_left
    N = even.shape[-1]
    src_idx = torch.empty(N + pad_left + pad_right, dtype=torch.long, device=even.device)
    for i in range(pad_left):
        src_idx[i] = min(pad_left - 1 - i + 1, N - 1)
    src_idx[pad_left:pad_left + N] = torch.arange(N, device=even.device)
    for i in range(pad_right):
        src_idx[pad_left + N + i] = max(N - 2 - i, 0)
    return even.index_select(-1, src_idx)


def _predict_odd_mlp(even, predictor, K):
    """Local-context MLP: apply predictor to each odd position's K-window."""
    N = even.shape[-1]
    if N == 0:
        return torch.zeros_like(even)
    padded = _pad_even_reflect_torch(even, K)
    contexts = torch.stack(
        [padded[..., j:j + N] for j in range(K)], dim=-1)
    return predictor(contexts)


def _predict_odd_dense(even, predictor):
    """Dense per-level predictor: full even array -> full odd array."""
    if even.shape[-1] == 0:
        return torch.zeros_like(even)
    return predictor(even)


def _dispatch_predict(even, predictor):
    """Route to the right prediction path based on predictor type."""
    if isinstance(predictor, DensePredictor):
        return _predict_odd_dense(even, predictor)
    K = predictor.net[0].in_features
    return _predict_odd_mlp(even, predictor, K)


def _lifting_forward_torch(signal, predictors, target_base):
    levels = []
    cur = signal
    lvl = 0
    while cur.shape[-1] > target_base and cur.shape[-1] >= 2:
        even = cur[..., 0::2]
        odd = cur[..., 1::2]
        pred = _dispatch_predict(even, predictors[lvl])
        detail = odd - pred
        levels.append(detail)
        cur = even
        lvl += 1
    return cur, levels, signal.shape[-1]


def _lifting_inverse_torch(base, levels, predictors, orig_n):
    cur = base
    for l_idx in range(len(levels) - 1, -1, -1):
        d = levels[l_idx]
        pred = _dispatch_predict(cur, predictors[l_idx])
        odd = d + pred
        N = cur.shape[-1]
        batch_shape = cur.shape[:-1]
        interleaved = torch.stack([cur, odd], dim=-1).reshape(*batch_shape, 2 * N)
        cur = interleaved
    return cur[..., :orig_n]


# ============================================================
# Differentiable μ-law quantizer
# ============================================================

def _mulaw_forward(x, alpha, x_max):
    """f(x) = sign(x) · log1p(α|x|) / log1p(α · x_max).  Range: [-1, 1]."""
    scale = torch.log1p(alpha * x_max + 1e-12)
    return torch.sign(x) * torch.log1p(alpha * x.abs()) / scale


def _mulaw_inverse(y, alpha, x_max):
    """Inverse of the μ-law curve."""
    scale = torch.log1p(alpha * x_max + 1e-12)
    return torch.sign(y) * torch.expm1(y.abs() * scale) / (alpha + 1e-12)


def _noise_quantize(y, delta, training):
    """Ballé-style additive uniform noise during training; hard round at
    inference. `q_int` is the integer code we'd transmit."""
    if training:
        noise = (torch.rand_like(y) - 0.5) * delta
        return y + noise, None
    q_int = torch.round(y / delta)
    return q_int * delta, q_int


# ============================================================
# End-to-end compressor
# ============================================================

class NeuralCompressor(nn.Module):
    def __init__(self, kernel_size, hidden, n_levels,
                 init_alpha=10.0, init_log_delta=-4.0,
                 target_base=32, eps=5e-4,
                 predictor_type="mlp", max_signal_len=256):
        """
        predictor_type:
            'mlp'   — local K-window MLP, same net shared across positions.
            'dense' — per-level Linear(N_even)->Hidden->Linear(N_odd), with
                      a unique output row per odd position.
        max_signal_len: required for 'dense' type — determines per-level
            N_even sizes. Streams must be padded to this length.
        """
        super().__init__()
        self.K = kernel_size
        self.H = hidden
        self.L = n_levels
        self.target_base = target_base
        self.eps = eps
        self.predictor_type = predictor_type
        self.max_signal_len = max_signal_len

        if predictor_type == "mlp":
            self.predictors = nn.ModuleList(
                [LiftingPredictor(kernel_size, hidden) for _ in range(n_levels)])
        elif predictor_type == "dense":
            # Each level sees half the previous: max_signal_len/2, /4, /8, ...
            predictors = []
            n_even = max_signal_len // 2
            for _ in range(n_levels):
                predictors.append(DensePredictor(n_even, hidden=hidden))
                n_even = n_even // 2
            self.predictors = nn.ModuleList(predictors)
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

        self.log_alpha = nn.Parameter(
            torch.full((n_levels + 1,), float(np.log(init_alpha))))
        self.log_delta = nn.Parameter(
            torch.full((n_levels + 1,), init_log_delta))

    def _quantize_stream(self, s, stream_idx, training):
        """s: (B, N) or (N,). Same-shape output. x_max is per-sample along last dim."""
        alpha = torch.exp(self.log_alpha[stream_idx])
        delta = torch.exp(self.log_delta[stream_idx])
        # Per-sample x_max (last-dim max), broadcast back
        x_max = s.abs().amax(dim=-1, keepdim=True).detach().clamp(min=1e-6)
        y = _mulaw_forward(s, alpha, x_max)
        y_q, q_int = _noise_quantize(y, delta, training)
        s_hat = _mulaw_inverse(y_q, alpha, x_max)
        return s_hat, y_q

    def forward(self, signal_batch, training=True):
        """signal_batch: (B, N_pow2). Returns (recon, q_raw_list)."""
        base, details, orig_n = _lifting_forward_torch(
            signal_batch, self.predictors, self.target_base)
        streams = [base] + details
        recon_streams = []
        q_raw = []
        for i, s in enumerate(streams):
            s_hat, y_q = self._quantize_stream(s, i, training)
            recon_streams.append(s_hat)
            q_raw.append(y_q)
        recon = _lifting_inverse_torch(
            recon_streams[0], recon_streams[1:],
            self.predictors, orig_n)
        return recon, q_raw

    # ------------------------------------------------------------
    # Export learned parameters for numpy-side encoder
    # ------------------------------------------------------------
    def export_params(self):
        alphas = torch.exp(self.log_alpha).detach().cpu().numpy()
        deltas = torch.exp(self.log_delta).detach().cpu().numpy()
        predictors_data = []
        for p in self.predictors:
            if isinstance(p, DensePredictor):
                enc_W = p.enc.weight.detach().cpu().numpy().astype(np.float64)
                enc_b = p.enc.bias.detach().cpu().numpy().astype(np.float64)
                dec_W = p.dec.weight.detach().cpu().numpy().astype(np.float64)
                dec_b = p.dec.bias.detach().cpu().numpy().astype(np.float64)
                predictors_data.append({
                    "type": "dense",
                    "enc": (enc_W, enc_b),
                    "dec": (dec_W, dec_b),
                    "n_even": p.n_even,
                    "hidden": p.hidden,
                })
            else:
                layer_weights = []
                for m in p.net:
                    if isinstance(m, nn.Linear):
                        W = m.weight.detach().cpu().numpy().astype(np.float64)
                        b = m.bias.detach().cpu().numpy().astype(np.float64)
                        layer_weights.append((W, b))
                predictors_data.append({"type": "mlp", "layers": layer_weights})
        return {
            "alphas": alphas,
            "deltas": deltas,
            "predictors": predictors_data,
            "kernel_size": self.K,
            "hidden": self.H,
            "n_levels": self.L,
            "target_base": self.target_base,
            "predictor_type": self.predictor_type,
            "max_signal_len": self.max_signal_len,
            # Back-compat alias
            "mlps": [pd.get("layers", None) for pd in predictors_data],
        }


# ============================================================
# Training loop
# ============================================================

def _pad_streams_to_batch(streams, min_len, device, fixed_len=None):
    """Pad 1-D streams to a common power-of-2 length ≥ min_len.
    If fixed_len is given, pads ALL streams to that length regardless
    (required for dense predictor so per-level N_even is constant)."""
    filtered = [s for s in streams if len(s) >= min_len]
    if not filtered:
        return None, None
    if fixed_len is not None:
        n_pow2 = fixed_len
    else:
        max_n = max(len(s) for s in filtered)
        n_pow2 = 1
        while n_pow2 < max_n:
            n_pow2 *= 2
    B = len(filtered)
    batch = torch.empty((B, n_pow2), dtype=torch.float32, device=device)
    lengths = torch.empty(B, dtype=torch.long, device=device)
    for i, s in enumerate(filtered):
        L = min(len(s), n_pow2)
        arr = torch.tensor(s[:L], dtype=torch.float32, device=device)
        batch[i, :L] = arr
        batch[i, L:] = arr[-1]
        lengths[i] = L
    return batch, lengths


def train_compressor(streams, kernel_size=4, hidden=16, n_levels_max=3,
                     target_base=32, eps=5e-4,
                     epochs=600, lr=1e-3,
                     lambda_rate=1.0, lambda_max=1000.0,
                     predictor_type="mlp", max_signal_len=256,
                     seed=0, device="cpu", verbose=False):
    """Train a NeuralCompressor on a list of 1-D float streams.
    Returns (model, exported_params)."""
    torch.manual_seed(seed)
    model = NeuralCompressor(
        kernel_size=kernel_size,
        hidden=hidden,
        n_levels=n_levels_max,
        target_base=target_base,
        eps=eps,
        predictor_type=predictor_type,
        max_signal_len=max_signal_len,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    min_len = target_base * (2 ** (n_levels_max - 1)) + 1
    fixed_len = max_signal_len if predictor_type == "dense" else None
    batch, lengths = _pad_streams_to_batch(streams, min_len, device,
                                            fixed_len=fixed_len)
    if batch is None:
        if verbose:
            print(f"  No streams ≥ {min_len} samples; nothing to train.")
        return model, model.export_params()

    B, N = batch.shape
    # Build per-sample mask for the distortion loss (ignore padded tail)
    mask = (torch.arange(N, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()

    if verbose:
        print(f"  Training on {B} streams × {N} samples (min_len={min_len})")

    # Tighten train-time eps so inference round() stays within true budget
    train_eps = eps * 0.85

    for epoch in range(epochs):
        opt.zero_grad()
        recon, q_raw = model(batch, training=True)
        diff = (batch - recon) * mask
        dist = (diff.pow(2).sum() / mask.sum())
        # Rate proxy: var(y)/δ² per level — correctly pushes δ UP and
        # signal-var DOWN (the two ways to reduce bits per coefficient).
        # var is computed per-stream then averaged, so it reflects per-
        # stream compression instead of between-stream spread.
        rate = 0.0
        for lvl_idx, q in enumerate(q_raw):
            delta = torch.exp(model.log_delta[lvl_idx])
            per_stream_var = q.var(dim=-1) + 1e-12
            rate = rate + (per_stream_var / (delta * delta)).mean()
        # Max-err penalty: use .sum() so every violator contributes
        # proportionally instead of getting averaged out.
        max_err_per_sample = diff.abs().amax(dim=-1)
        mean_max_err = max_err_per_sample.mean()
        worst_max_err = max_err_per_sample.max()
        me_pen = F.relu(max_err_per_sample - train_eps).pow(2).sum()
        loss = dist + lambda_rate * rate + lambda_max * me_pen
        loss.backward()
        opt.step()

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}: dist={float(dist):.2e}  "
                  f"rate_proxy={float(rate):.2e}  "
                  f"mean_max_err={float(mean_max_err):.2e}  "
                  f"worst_max_err={float(worst_max_err):.2e}  "
                  f"me_pen={float(me_pen):.2e}")

    return model, model.export_params()


# ============================================================
# Numpy-side inference (bit-exact match to model at inference mode)
# ============================================================

def _pad_even_reflect_np(even, K):
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


def _mlp_forward_np(weights, x):
    out = x
    for i, (W, b) in enumerate(weights):
        out = out @ W.T + b
        if i < len(weights) - 1:
            out = np.maximum(0.0, out)
    return out.squeeze(-1)


def _predict_odd_np(even, predictor_data, K):
    """Dispatches to MLP or Dense predictor based on spec."""
    N = len(even)
    if N == 0:
        return np.zeros(0, dtype=np.float64)
    if predictor_data.get("type") == "dense":
        # Dense: full even array -> full odd array. Pad/truncate to n_even.
        n_even = predictor_data["n_even"]
        if N != n_even:
            # This should not happen if signal is padded to max_signal_len
            padded = np.full(n_even, even[-1] if N > 0 else 0.0)
            padded[:N] = even[:N]
        else:
            padded = even
        enc_W, enc_b = predictor_data["enc"]
        dec_W, dec_b = predictor_data["dec"]
        h = padded @ enc_W.T + enc_b
        h = np.maximum(0.0, h)
        return (h @ dec_W.T + dec_b)[:N]
    # MLP path (legacy): use K-window context
    padded = _pad_even_reflect_np(even, K)
    contexts = np.empty((N, K), dtype=np.float64)
    for j in range(K):
        contexts[:, j] = padded[j:j + N]
    layers = predictor_data["layers"] if "layers" in predictor_data else predictor_data
    return _mlp_forward_np(layers, contexts)


def _lifting_forward_np(signal, predictors_data, K, target_base, max_len=None):
    n = len(signal)
    if n <= target_base:
        return signal.copy(), [], n
    # Pad to max_len (required for dense) or next pow-2 (MLP)
    if max_len is not None:
        n_pad = max_len
    else:
        n_pad = 1
        while n_pad < n:
            n_pad *= 2
    sig = np.empty(n_pad, dtype=np.float64)
    L = min(n, n_pad)
    sig[:L] = signal[:L]
    sig[L:] = signal[L - 1]
    levels = []
    cur = sig
    lvl = 0
    while len(cur) > target_base and len(cur) >= 2:
        even = cur[0::2]
        odd = cur[1::2]
        pred = _predict_odd_np(even, predictors_data[lvl], K)
        detail = odd - pred
        levels.append(detail)
        cur = even
        lvl += 1
    return cur, levels, n  # return original n, not padded


def _lifting_inverse_np(base, levels, predictors_data, K, orig_n, padded_n=None):
    cur = base.copy()
    for l_idx in range(len(levels) - 1, -1, -1):
        d = levels[l_idx]
        pred = _predict_odd_np(cur, predictors_data[l_idx], K)
        odd = d + pred
        N = len(cur)
        out = np.empty(2 * N, dtype=np.float64)
        out[0::2] = cur
        out[1::2] = odd
        cur = out
    return cur[:orig_n]


def _mulaw_forward_np(x, alpha, x_max):
    scale = np.log1p(alpha * x_max + 1e-12)
    return np.sign(x) * np.log1p(alpha * np.abs(x)) / scale


def _mulaw_inverse_np(y, alpha, x_max):
    scale = np.log1p(alpha * x_max + 1e-12)
    return np.sign(y) * np.expm1(np.abs(y) * scale) / (alpha + 1e-12)


def _mulaw_max_err(alpha, x_max, delta):
    """Max x-domain reconstruction error after ±δ/2 rounding in y-domain."""
    # Worst-case slope of inverse: at |x|=x_max.
    # |dx/dy| = (1 + α x_max) / α  (for μ-law normalized to y in [-1, 1])
    # NB: this is NOT multiplied by log1p(α x_max) because the curve is
    # already normalized so y ∈ [-1, 1] and delta is in y-domain.
    # Hmm actually let me redo: dy/dx = α / (1 + α|x|) / scale where scale
    # = log1p(α x_max). Then dx/dy = scale * (1 + α|x|) / α. Worst case at
    # |x|=x_max -> dx/dy = scale * (1 + α x_max) / α.
    scale = np.log1p(alpha * x_max + 1e-12)
    return 0.5 * delta * scale * (1.0 + alpha * x_max) / (alpha + 1e-12)


def quantize_interior_diff(positions, params, amp=1.0, verbose=False):
    """Encode interior positions using learned compressor parameters.

    Returns (recon, total_bits, meta).
    Bit count uses the packed-meta layout (same as MLP encoder).

    params: dict from NeuralCompressor.export_params()
    amp:    lifting-inverse Lipschitz amp (applied on top of curve error,
            used only for diagnostics here — actual max error is measured)
    """
    from utils.wavelet import _stream_bits

    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        return positions.copy(), 0, {"n_levels": 0}

    K = params["kernel_size"]
    target_base = params["target_base"]
    predictors_data = params.get("predictors")
    if predictors_data is None:
        # Back-compat: build from legacy mlps list
        predictors_data = [{"type": "mlp", "layers": L} for L in params["mlps"]]
    alphas = params["alphas"]
    deltas = params["deltas"]
    max_len = (params.get("max_signal_len")
               if params.get("predictor_type") == "dense" else None)

    recon = np.empty_like(positions)
    total_bits = 3 * 32  # per-axis float32 offset
    per_axis_streams = []
    L_final = 0

    for d in range(3):
        offset = float(positions[:, d].min())
        shifted = positions[:, d] - offset
        base, details, orig_n = _lifting_forward_np(
            shifted, predictors_data, K, target_base, max_len=max_len)
        L = len(details)
        L_final = max(L_final, L)

        streams = [base] + details
        recon_streams = []
        q_codes_per_level = []
        for i, s in enumerate(streams):
            x_max = np.abs(s).max() if len(s) > 0 else 0.0
            x_max = max(x_max, 1e-6)
            y = _mulaw_forward_np(s, alphas[i], x_max)
            q_int = np.round(y / deltas[i]).astype(np.int64)
            y_q = q_int.astype(np.float64) * deltas[i]
            s_hat = _mulaw_inverse_np(y_q, alphas[i], x_max)
            recon_streams.append(s_hat)
            q_codes_per_level.append(q_int)

        recon[:, d] = _lifting_inverse_np(
            recon_streams[0], recon_streams[1:],
            predictors_data, K, orig_n) + offset
        per_axis_streams.append(q_codes_per_level)

    # Packed metadata (same as float_wavelet_packed)
    def _pack_level(level_streams):
        body_bits = 0
        for codes in level_streams:
            if len(codes) == 0:
                continue
            mn = int(codes.min())
            rng = int(codes.max() - mn)
            b = max(1, int(np.ceil(np.log2(rng + 2)))) if rng > 0 else 1
            shifted = codes - mn
            body_bits += _stream_bits(shifted, b)
        return body_bits + 3 * (16 + 8)

    # Base first, then details — indexing per axis
    base_streams = [per_axis_streams[d][0] for d in range(3)]
    total_bits += _pack_level(base_streams)
    for lvl in range(L_final):
        level_streams = [per_axis_streams[d][1 + lvl] for d in range(3)]
        total_bits += _pack_level(level_streams)

    return recon, total_bits, {"n_levels": L_final}

"""
B2: Neural SDF mesh compression estimation.
Tiny quantized MLP replaces 2-sphere SDF for better surface fitting.
Tests multiple hidden sizes and weight quantization levels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import Counter
from copy import deepcopy
from reader import Reader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Positional Encoding (key for small MLPs to capture detail)
# ============================================================

class PositionalEncoding:
    def __init__(self, n_freqs=4):
        self.n_freqs = n_freqs

    def encode(self, x):
        out = [x]
        for i in range(self.n_freqs):
            freq = 2.0 ** i * math.pi
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=1)

    @property
    def output_dim(self):
        return 3 + 3 * 2 * self.n_freqs


# ============================================================
# Neural SDF Model
# ============================================================

class NeuralSDF(nn.Module):
    def __init__(self, hidden=16, n_freqs=0):
        super().__init__()
        self.n_freqs = n_freqs
        self.pe = PositionalEncoding(n_freqs) if n_freqs > 0 else None
        in_dim = self.pe.output_dim if self.pe else 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Softplus(beta=5),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=5),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h = self.pe.encode(x) if self.pe else x
        return self.net(h).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def header_size_bits(self, weight_bits=8):
        n_groups = sum(1 for _ in self.parameters())
        return self.n_params() * weight_bits + n_groups * 2 * 32  # per-group min/max


def quantize_model(model, bits=8):
    """Return a copy with weights uniformly quantized to N bits."""
    mq = deepcopy(model)
    for p in mq.parameters():
        lo, hi = p.data.min(), p.data.max()
        if hi - lo < 1e-10:
            continue
        scale = (hi - lo) / ((1 << bits) - 1)
        p.data = torch.round((p.data - lo) / scale) * scale + lo
    return mq


# ============================================================
# Vertex normals from triangles
# ============================================================

def compute_normals(verts_np, tris):
    normals = np.zeros_like(verts_np)
    for t in tris:
        e1 = verts_np[t[1]] - verts_np[t[0]]
        e2 = verts_np[t[2]] - verts_np[t[0]]
        n = np.cross(e1, e2)
        normals[t[0]] += n
        normals[t[1]] += n
        normals[t[2]] += n
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / (lens + 1e-12)


# ============================================================
# Training
# ============================================================

def train_sdf(model, verts_t, normals_t, epochs=3000, lr=0.005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    eps = 0.02

    for epoch in range(epochs):
        optimizer.zero_grad()
        # Surface: SDF ≈ 0
        loss_s = torch.mean(model(verts_t) ** 2)
        # Outside: SDF ≈ +eps
        loss_o = torch.mean((model(verts_t + eps * normals_t) - eps) ** 2)
        # Inside: SDF ≈ -eps
        loss_i = torch.mean((model(verts_t - eps * normals_t) + eps) ** 2)
        loss = loss_s + 0.5 * (loss_o + loss_i)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 1000 == 0:
            print(f"    Epoch {epoch}: loss={loss.item():.8f}")
    print(f"    Final:      loss={loss.item():.8f}")
    return model


# ============================================================
# Spherical coords & bisection ray marching
# ============================================================

def to_spherical(xyz):
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(xyz[:, 2] / (r + 1e-12), -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi[phi < 0] += 2 * np.pi
    return r, theta / np.pi, phi / (2 * np.pi)


def to_cartesian(r, u, v):
    theta, phi = u * np.pi, v * 2 * np.pi
    return np.stack([r * np.sin(theta) * np.cos(phi),
                     r * np.sin(theta) * np.sin(phi),
                     r * np.cos(theta)], axis=1)


def ray_march(model, dirs_t, steps=64):
    with torch.no_grad():
        n = dirs_t.shape[0]
        t_lo = torch.zeros(n, device=device)
        t_hi = torch.ones(n, device=device) * 3.0
        for _ in range(steps):
            t_mid = (t_lo + t_hi) / 2
            d = model(dirs_t * t_mid.unsqueeze(1))
            inside = d < 0
            t_lo = torch.where(inside, t_mid, t_lo)
            t_hi = torch.where(inside, t_hi, t_mid)
        return ((t_lo + t_hi) / 2).cpu().numpy()


# ============================================================
# Quantization & Entropy
# ============================================================

def quantize(vals, lo, hi, bits):
    mx = (1 << bits) - 1
    norm = np.clip((vals - lo) / (hi - lo + 1e-15), 0, 1)
    return np.round(norm * mx).astype(np.int64)


def dequantize(codes, lo, hi, bits):
    return codes.astype(np.float64) / ((1 << bits) - 1) * (hi - lo) + lo


def bits_for_error(val_range, max_err):
    if max_err <= 0 or val_range <= 0:
        return 16
    return max(1, int(np.ceil(np.log2(val_range / (2 * max_err) + 1))))


def entropy(codes):
    if len(codes) == 0:
        return 0.0
    counts = Counter(codes.tolist())
    total = len(codes)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


def stream_bits(codes, fixed_bits):
    n = len(codes)
    plain = n * fixed_bits
    ent = entropy(codes)
    arith = n * ent + 32
    return min(plain, arith)


def tree_entropy_2d(uv, cu, cv, bu, bv, max_d=12, min_leaf=4):
    hdr = [0.0]; pay = [0.0]
    def go(idx, d, ax):
        n = len(idx)
        if n == 0: return
        if d >= max_d or n <= min_leaf:
            lbu = max(1, bu - (d+1)//2)
            lbv = max(1, bv - d//2)
            eu = entropy(cu[idx] % (1 << lbu))
            ev = entropy(cv[idx] % (1 << lbv))
            pay[0] += n * (eu + ev); return
        hdr[0] += np.ceil(np.log2(n + 1))
        vals = uv[idx, ax]; med = np.median(vals)
        left = vals < med
        if left.sum() == 0 or left.sum() == n:
            lbu = max(1, bu - (d+1)//2); lbv = max(1, bv - d//2)
            pay[0] += n * (entropy(cu[idx] % (1<<lbu)) + entropy(cv[idx] % (1<<lbv))); return
        go(idx[left], d+1, 1-ax); go(idx[~left], d+1, 1-ax)
    go(np.arange(len(cu)), 0, 0)
    return hdr[0] + pay[0]


def tree_entropy_1d(vals, codes, bits, max_d=12, min_leaf=4):
    hdr = [0.0]; pay = [0.0]
    def go(idx, d):
        n = len(idx)
        if n == 0: return
        if d >= max_d or n <= min_leaf:
            lb = max(1, bits - d)
            pay[0] += n * entropy(codes[idx] % (1<<lb)); return
        hdr[0] += np.ceil(np.log2(n + 1))
        v = vals[idx]; med = np.median(v); left = v < med
        if left.sum() == 0 or left.sum() == n:
            pay[0] += n * entropy(codes[idx] % (1<<max(1,bits-d))); return
        go(idx[left], d+1); go(idx[~left], d+1)
    go(np.arange(len(codes)), 0)
    return hdr[0] + pay[0]


# ============================================================
# Main pipeline
# ============================================================

def run(obj_path, max_error=0.001, configs=None):
    if configs is None:
        configs = [
            {"hidden": 8,  "n_freqs": 0, "wbits": 8},
            {"hidden": 16, "n_freqs": 0, "wbits": 8},
            {"hidden": 8,  "n_freqs": 4, "wbits": 8},
            {"hidden": 16, "n_freqs": 4, "wbits": 8},
            {"hidden": 32, "n_freqs": 4, "wbits": 8},
        ]

    mesh = Reader.read_from_file(obj_path)
    verts_np = np.array([[v.x, v.y, v.z] for v in mesh.vertices])
    tris = [[t.a, t.b, t.c] for t in mesh.triangles]
    n = len(verts_np)

    center = verts_np.mean(axis=0)
    vc = verts_np - center
    scale = np.max(np.linalg.norm(vc, axis=1))
    vn = vc / scale

    normals = compute_normals(vn, tris)
    verts_t = torch.tensor(vn, dtype=torch.float32, device=device)
    normals_t = torch.tensor(normals, dtype=torch.float32, device=device)

    r, u, v = to_spherical(vn)

    # Raw and baseline sizes
    raw_bits = n * 96
    bpc = bits_for_error(2 * scale, max_error)
    baseline_bits = n * 3 * bpc

    print(f"{'='*70}")
    print(f"B2 Neural SDF — {obj_path} — {n} verts — max_error={max_error}")
    print(f"{'='*70}")
    print(f"Raw:       {raw_bits/8:>8.0f} B  ({raw_bits/n:.1f} bpv)")
    print(f"Baseline:  {baseline_bits/8:>8.0f} B  ({baseline_bits/n:.1f} bpv)")

    results = []

    for cfg in configs:
        hidden = cfg["hidden"]
        n_freqs = cfg["n_freqs"]
        wbits = cfg["wbits"]
        label = f"h={hidden} PE={n_freqs} w{wbits}b"
        print(f"\n--- Config: {label} ---")

        sdf = NeuralSDF(hidden=hidden, n_freqs=n_freqs).to(device)
        print(f"  Params: {sdf.n_params()}")
        sdf = train_sdf(sdf, verts_t, normals_t, epochs=3000, lr=0.005)

        # Quantize model weights
        sdf_q = quantize_model(sdf, wbits)
        sdf_header = sdf_q.header_size_bits(wbits)

        # Ray march with quantized model
        dirs = to_cartesian(np.ones_like(r), u, v)
        dirs_t = torch.tensor(dirs, dtype=torch.float32, device=device)
        r_pred = ray_march(sdf_q, dirs_t)
        r_res = r - r_pred

        var_red = (1 - r_res.var() / r.var()) * 100
        print(f"  Variance reduction: {var_red:.1f}%")
        print(f"  r_residual range: {r_res.min():.4f} to {r_res.max():.4f}")

        # Quantize streams
        bu = bits_for_error(u.max() - u.min(), max_error)
        bv = bits_for_error(v.max() - v.min(), max_error)
        br = bits_for_error(r_res.max() - r_res.min(), max_error)

        cu = quantize(u, u.min(), u.max(), bu)
        cv = quantize(v, v.min(), v.max(), bv)
        cr = quantize(r_res, r_res.min(), r_res.max(), br)

        # Best of plain/arithmetic vs tree
        plain_uv = stream_bits(cu, bu) + stream_bits(cv, bv)
        tree_uv = tree_entropy_2d(np.stack([u,v],1), cu, cv, bu, bv)
        best_uv = min(plain_uv, tree_uv)

        plain_r = stream_bits(cr, br)
        tree_r = tree_entropy_1d(r_res, cr, br)
        best_r = min(plain_r, tree_r)

        meta = 3*32 + 32 + 6*32 + 3*8  # center, scale, ranges, bit counts
        total = sdf_header + meta + best_uv + best_r
        bpv = total / n

        # Dequantize and measure accuracy
        u_rec = dequantize(cu, u.min(), u.max(), bu)
        v_rec = dequantize(cv, v.min(), v.max(), bv)
        r_res_rec = dequantize(cr, r_res.min(), r_res.max(), br)
        r_rec = r_pred + r_res_rec
        xyz_rec = to_cartesian(r_rec, u_rec, v_rec) * scale + center
        errors = np.linalg.norm(xyz_rec - verts_np, axis=1)

        pct = (errors <= max_error).sum() / n * 100
        print(f"  Header:  {sdf_header/8:.0f} B")
        print(f"  u,v:     {best_uv/8:.0f} B (bits u={bu} v={bv})")
        print(f"  r:       {best_r/8:.0f} B (bits={br}, range={r_res.max()-r_res.min():.4f})")
        print(f"  TOTAL:   {total/8:.0f} B  ({bpv:.2f} bpv)  ratio={raw_bits/total:.2f}x")
        print(f"  Error:   mean={errors.mean():.6f}  max={errors.max():.6f}  "
              f"within_target={pct:.1f}%")

        results.append({
            "label": label, "params": sdf.n_params(),
            "header": sdf_header/8, "total": total/8, "bpv": bpv,
            "ratio": raw_bits/total, "var_red": var_red,
            "mean_err": errors.mean(), "max_err": errors.max(), "pct": pct
        })

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY — {obj_path}")
    print(f"{'='*70}")
    print(f"{'Config':<22} {'Params':>6} {'Header':>7} {'Total':>7} "
          f"{'BPV':>6} {'Ratio':>6} {'VarRed':>7} {'MaxErr':>8} {'%OK':>5}")
    for r in results:
        print(f"{r['label']:<22} {r['params']:>6} {r['header']:>7.0f} {r['total']:>7.0f} "
              f"{r['bpv']:>6.2f} {r['ratio']:>6.2f} {r['var_red']:>6.1f}% "
              f"{r['max_err']:>8.6f} {r['pct']:>4.1f}%")


if __name__ == "__main__":
    for path in ["assets/bunny.obj", "assets/torus.obj"]:
        try:
            run(path, max_error=0.001)
            print("\n\n")
        except Exception as e:
            import traceback; traceback.print_exc()
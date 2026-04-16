# MeshPress

MeshPress is a 3D mesh compression library for real-time GPU decompression. It extends the AMD GPUOpen meshlet compression approach with segmented delta vertex encoding, achieving **24% better compression** while maintaining crack-free rendering and GPU-parallel decode.

## Features

- **Crack-free** meshlet compression via global quantization grid
- **GPU-parallel** decompression using GTS strip decode (countbits/firstbithigh intrinsics)
- **Segmented delta** vertex encoding: 24% smaller vertices than AMD baseline
- Controlled precision: guaranteed maximum reconstruction error
- Scales to millions of polygons (tested on 1.8M vertex models)

## Installation

```bash
git clone https://github.com/maletsden/meshpress.git
cd meshpress
pip install -r requirements.txt
pip install scikit-learn cupy-cuda12x  # for meshlet encoders and GPU benchmarks
```

## Quick Usage

```python
from reader import Reader
from encoder import MeshletGTSSegDelta, MeshletGTSPlain

model = Reader.read_from_file('assets/stanford-bunny.obj')

# Our best: GTS connectivity + segmented delta vertices (crack-free, GPU-parallel)
encoder = MeshletGTSSegDelta(max_verts=256, precision_error=0.0005, verbose=True)
compressed = encoder.encode(model)

# AMD baseline: GTS connectivity + plain quantized vertices (crack-free, GPU-parallel)
encoder = MeshletGTSPlain(max_verts=256, precision_error=0.0005, verbose=True)
compressed = encoder.encode(model)
```

## Compression Benchmarks

All encoders use **AMD GTS connectivity** (Generalized Triangle Strip with L/R flags + inc/reuse packing, ~6 bpt, GPU-parallel decode via bit intrinsics). The difference is vertex encoding only.

### Full Pipeline (GTS connectivity + vertices, max_error=0.0005)

| Model | Verts | Tris | AMD Baseline BPV | SegDelta BPV | Haar BPV | SegDelta Improvement |
|-------|------:|-----:|-----------------:|-------------:|---------:|---------------------:|
| bunny | 2,503 | 4,968 | 47.34 | **41.53** | 42.28 | +12.3% |
| torus | 3,840 | 7,680 | 56.48 | **39.07** | 38.56 | +30.8% |
| stanford-bunny | 35,947 | 69,451 | 37.90 | **32.49** | 34.47 | +14.3% |
| Monkey | 504,482 | 1,007,616 | 41.54 | **31.69** | 33.77 | +23.7% |

All crack-free (0 shared vertex mismatches verified). 100% within error target.

### Bits Breakdown (Monkey, GTS+SegDelta)

| Component | Size | % | BPV | Description |
|-----------|-----:|--:|----:|-------------|
| Headers | 117 KB | 5.9% | 1.86 | Global + per-meshlet metadata |
| Vertex data | 1,083 KB | 54.2% | 17.18 | Segmented delta encoded |
| Connectivity | 798 KB | 39.9% | 12.66 | GTS: L/R flags + inc/reuse |
| **Total** | **1,998 KB** | **100%** | **31.69** | **9.1x compression** |

### Vertex Encoding Comparison (vertex data only, same GTS connectivity)

| Method | S-Bunny | Monkey | vs Plain | GPU Parallel |
|--------|--------:|-------:|---------:|:-------------|
| Plain (AMD baseline) | 104 KB | 1,704 KB | — | Trivial (just read) |
| **Segmented Delta** | **80 KB** | **1,083 KB** | **-36%** | **32 independent prefix sums** |
| Haar Wavelet | 89 KB | 1,215 KB | -29% | log(N) steps, 6 syncs |
| CDF 5/3 Wavelet | 91 KB | 1,228 KB | -28% | log(N) steps, 6 syncs |

Segmented delta gives the best compression AND the fastest GPU decode: 32 independent segments decoded via prefix sum (2 syncs total vs 6 for wavelets).

### GPU Decode Speed (RTX 3090, isolated timing)

| Kernel | Stanford-Bunny (36K) | Monkey (504K) |
|--------|---------------------:|------------------:|
| AMD GTS+Plain (baseline) | 11.2 µs | 36.7 µs |
| **GTS+SegDelta Opt v3 (ours)** | **9.8 µs** | **44.3 µs** |
| GTS+Haar Opt v3 (ours) | ~10 µs | ~45 µs |

Our segmented delta adds only **+21% decode overhead** on large meshes vs the plain AMD baseline, while providing **24% better compression**. On small meshes it's actually faster.

## Architecture

### Encoding Pipeline
```
Mesh → Global Quantization (crack-free integer grid)
     → Meshlet Generation (greedy region growing, max 256 verts)
     → Per meshlet:
         BFS vertex ordering (spatial locality for delta encoding)
         GTS strip generation (L/R flags + inc/reuse packing)
         Vertex encoding: segmented delta on globally-quantized integers
     → Compressed stream
```

### Per-Meshlet Compressed Format
```
┌── Meshlet Header (37 bytes)
│   n_verts, n_tris, per-channel: base_min, base_bits, delta_min, delta_bits
├── GTS Connectivity
│   L/R flags:     1 bit per triangle (which edge reused)
│   Inc flags:     1 bit per strip entry (new vertex vs reused)
│   Reuse buffer:  uint8 per reused vertex (local index)
├── Vertex Data (per channel x,y,z)
│   Base stream:   32 anchor values × base_bits
│   Delta stream:  (n_verts-32) deltas × delta_bits
└── GPU decode: countbits + firstbithigh for connectivity
               32 parallel prefix sums for vertices
```

### GPU Decode (1 CUDA block per meshlet, 256 threads)
```
Phase 1: Read header (1 thread → broadcast)               [1 sync]
Phase 2: GTS connectivity decode (parallel, bit intrinsics) [0 syncs]
Phase 3: Vertex base values (32 threads)                    [1 sync]
Phase 4: Segment prefix sums (96 parallel: 32 segs × 3 ch) [1 sync]
Phase 5: Global dequantize + write output (parallel)        [0 syncs]
Total: 3 __syncthreads
```

### Crack-Free Guarantee

All vertices are quantized to a **global integer grid** at encode time. The segmented delta encoding is **lossless on integers** (prefix sum exactly reconstructs the original sequence). Shared boundary vertices in different meshlets get the **same integer code** → identical float reconstruction → zero cracks.

## Running Benchmarks

```bash
# Full encoder comparison on all models
python test_final_benchmark.py

# CUDA GPU decode speed benchmark (requires CuPy)
python benchmark_cuda_decode.py

# Wavelet/delta comparison
python test_wavelet_meshlet_compression.py

# Connectivity encode/decode verification
python test_connectivity_verify.py
```

## Available Encoders

| Encoder | Connectivity | Vertices | LOD | Crack-free | GPU Decode |
|---------|-------------|----------|:---:|:----------:|:----------:|
| `MeshletGTSSegDelta` | GTS | Segmented delta | No | **Yes** | **Parallel** |
| `MeshletGTSHaar` | GTS | Haar wavelet | No | **Yes** | Parallel |
| `MeshletGTSPlain` | GTS | Plain quantized | No | **Yes** | **Parallel** |
| `MeshletWaveletGlobalEB` | EdgeBreaker | Int wavelet | No | **Yes** | Sequential |
| `MeshletPlainAMD` | Raw packed | Global grid | No | **Yes** | Parallel |
| **`MeshletLOD`** | FIFO-adjacency | Chain-delta + entropy | **Yes (5 levels)** | **Yes** | **Parallel** |

## Progressive LOD Compression (`MeshletLOD`)

`MeshletLOD` is a progressive Level-of-Detail encoder built on QEM simplification
with synchronized boundary protection. A single compressed bitstream can be
decoded at any of 5 resolution levels without re-encoding; every level produces
a valid, crack-free triangulation covering the full mesh area.

### Quick usage

```python
from reader import Reader
from encoder import MeshletLOD, decode_lod
from encoder.implementation.meshlet_wavelet import _to_numpy

model = Reader.read_from_file('assets/stanford-bunny.obj')
enc = MeshletLOD(max_verts=256, precision_error=0.0005, n_lod_levels=5, verbose=True)
compressed = enc.encode(model)

# Decode at a chosen LOD (0 = coarsest, 4 = full resolution)
verts, tris = _to_numpy(model)
out_verts, out_tris = decode_lod(verts, tris, compressed, lod_level=2)
```

### Algorithm

```
Mesh → Generate meshlets (greedy region growing, max 256 verts)
     → Identify boundary vertices (shared between ≥2 meshlets)
     → QEM on full mesh with boundary PROTECTED from collapse
     → QEM on boundary subgraph alone (boundary-to-boundary collapses)
     → Per meshlet:
         Local AABB quantization for interior vertices
         Chain-delta encoding of interior positions (delta from collapse parent)
         FIFO-adjacency bitstream for connectivity
     → Compact ancestor tables:
         (collapse_step, direct_parent) per vertex
         Entropy/exp-Golomb coded parent deltas
```

### Crack-free guarantees

1. **Geometric** — boundary vertices use a global quantization grid, so the
   same boundary vertex yields bit-identical positions in every meshlet.
2. **Topological** — triangles are **redirected**, not dropped. When a vertex
   collapses into its ancestor at low LOD, any triangle containing it gets
   its vertex replaced by the ancestor. Triangles that become degenerate
   disappear into neighbouring triangles; full area coverage is preserved.

### Compression (all crack-free, 5 LOD levels, max_error=0.0005)

| Model | Verts | Tris | BPV | Size | Non-LOD `GTSSegDelta` BPV | LOD overhead |
|-------|------:|-----:|----:|-----:|------------------------:|-------------:|
| bunny | 2,503 | 4,968 | 52.32 | 16 KB | 41.53 | +26% |
| torus | 3,840 | 7,680 | 50.25 | 24 KB | 39.07 | +29% |
| stanford-bunny | 35,947 | 69,451 | **49.80** | 224 KB | 32.49 | +53% |
| Monkey | 504,482 | 1,007,616 | 50.69 | 3.1 MB | 31.69 | +60% |

The ~25–60% BPV overhead vs a non-LOD encoder is the cost of progressive
decode capability — a single bitstream covering 5 resolution levels instead
of one fixed resolution.

### Per-component breakdown (stanford-bunny)

| Component | Size | % | Description |
|-----------|-----:|--:|-------------|
| Boundary positions | 24 KB | 11% | Global-quant, shared across meshlets |
| Interior positions | 43 KB | 19% | Local-AABB quant + chain-delta |
| Interior ancestors | 88 KB | 39% | Compact (collapse_step, parent) + entropy |
| Boundary ancestors | 20 KB | 9% | Same format, separate chain |
| Connectivity (FIFO) | 48 KB | 22% | Real AMD-style FIFO-adjacency bitstream |
| **Total** | **224 KB** | **100%** | **5 LODs in one bitstream** |

### LOD progression (stanford-bunny)

| LOD | Verts | Tris | % of full |
|----:|------:|-----:|----------:|
| 0 | 3,004 | 3,727 | 5% |
| 1 | 11,241 | 20,056 | 29% |
| 2 | 19,476 | 36,520 | 53% |
| 3 | 27,712 | 52,989 | 76% |
| 4 | 35,947 | 69,451 | 100% |

### GPU per-frame decompression (RTX 3090, CuPy kernels)

Per-frame hot path: ancestor resolution (K1) → boundary composition (K2) →
triangle redirection + emission (K3). All three kernels are embarrassingly
parallel (1 thread per vertex or per triangle).

| Model | LOD 0 total | LOD 4 total | Fused kernel | Throughput |
|-------|------------:|------------:|-------------:|-----------:|
| bunny (5K tris) | 65 µs | 80 µs | 24 µs | — |
| torus (7.7K tris) | 58 µs | 70 µs | 23 µs | — |
| stanford-bunny (70K tris) | 112 µs | 76 µs | 34 µs | ~900 M tri/s |
| Monkey (1M tris) | 268 µs | 78 µs | 37 µs | **~12.8 G tri/s** |

One-time mesh-load costs (connectivity decode + ancestor table upload) are
separate and amortized over all frames.

```bash
python benchmark_cuda_a_fifo.py   # run the CUDA decompression benchmark
```

### GPU decode pipeline

```
Per-frame (parallel):
  K1  1 thread / vertex    Walk compact collapse chain → interior ancestor
  K1' 1 thread / vertex    Walk boundary chain → boundary ancestor
  K2  1 thread / vertex    Compose: combined = bnd_anc[int_anc]
  K3  1 thread / triangle  Redirect 3 verts, skip degenerate, emit

One-time (at mesh load):
  - Decode FIFO-adjacency connectivity into per-meshlet triangle lists
  - Propagate chain-delta interior positions (roots → children)
  - Upload ancestor tables, triangle list, positions to GPU
```

### Tradeoffs vs non-LOD `MeshletGTSSegDelta`

| Aspect | `MeshletGTSSegDelta` | `MeshletLOD` |
|--------|:-----:|:-----:|
| BPV | 32 (stanford-bunny) | 50 |
| LOD levels | 1 (fixed) | 5 (progressive) |
| Crack-free | Yes | Yes (geometric + topological) |
| GPU decode per frame | ~10 µs | ~80 µs |
| GPU throughput | ~35 G tri/s | ~12 G tri/s |
| Re-encoding for different LOD | Required | Not needed |
| Use case | Static draws | Distance-based LOD |

Choose `MeshletGTSSegDelta` when you only need one resolution level and
absolute minimum size/fastest decode. Choose `MeshletLOD` when the camera
distance varies and you want smooth transition between detail levels from
a single compressed representation.

## Legacy Encoders (vertex-only, bunny model)

| Encoder Model | Bytes/tri | Bytes/vert | Ratio |
|---------------|----------:|-----------:|------:|
| BaselineEncoder | 18.05 | 35.82 | 1.00 |
| PackedGTSQuantizator (radix) | 4.18 | 8.29 | 4.32 |
| PackedGTSEllipsoidFitter(0.0005, 4) | **2.75** | **5.47** | **6.55** |

![bytes_per_triangle_bar_plot](images/bytes_per_triangle_bar_plot.png) ![compression_rate_bar_plot](images/compression_rate_bar_plot.png)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

# MeshPress

MeshPress is a 3D mesh compression library that offers various encoding techniques to reduce the storage size of 3D models. The project aims to provide efficient compression of 3D meshes while preserving the geometric integrity of the models. This is especially useful in applications such as graphics rendering, AR/VR, and game development where memory efficiency and loading times are critical.

## Features

- Crack-free meshlet compression with global quantization grid + integer Haar wavelet
- GPU-friendly parallel decompression (CUDA, <45µs for 504K vertices)
- Two connectivity strategies: EdgeBreaker (best compression, 22 bpv) and AMD Packed (fastest GPU decode)
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
from encoder import MeshletWaveletGlobalEB, MeshletWaveletGlobalAMD

model = Reader.read_from_file('assets/stanford-bunny.obj')

# Best compression (crack-free, EdgeBreaker connectivity)
encoder = MeshletWaveletGlobalEB(max_verts=256, precision_error=0.0005, verbose=True)
compressed = encoder.encode(model)

# Fastest GPU decode (crack-free, AMD packed connectivity)
encoder = MeshletWaveletGlobalAMD(max_verts=256, precision_error=0.0005, verbose=True)
compressed = encoder.encode(model)
```

## Compression Benchmarks

### Crack-Free Meshlet Encoders (full mesh: vertices + connectivity)

All encoders are **crack-free** — shared boundary vertices between meshlets reconstruct to identical positions via global quantization grid.

| Model | Verts | Tris | GlobalEB BPV | Ratio | GlobalAMD BPV | PlainAMD BPV |
|-------|------:|-----:|-------------:|------:|--------------:|-------------:|
| bunny | 2,503 | 4,968 | 30.43 | 9.4x | 74.65 | 78.91 |
| torus | 3,840 | 7,680 | 27.41 | 10.5x | 73.13 | 72.39 |
| stanford-bunny | 35,947 | 69,451 | **23.12** | **12.2x** | 66.78 | 72.39 |
| Monkey | 504,482 | 1,007,616 | **22.24** | **12.9x** | 67.81 | 82.88 |

- **GlobalEB**: Global int wavelet + EdgeBreaker (~1.5 bpt connectivity). Best compression.
- **GlobalAMD**: Global int wavelet + AMD packed micro-indices (~5.7 bpt). Fastest GPU decode.
- **PlainAMD**: AMD global-grid quantization baseline (no wavelet). Reference for comparison.

### GPU Decode Speed (RTX 3090, isolated timing)

| Kernel | Stanford-Bunny (36K) | Monkey (504K) | Notes |
|--------|---------------------:|------------------:|-------|
| AMD GlobalGrid (baseline) | 11.2 µs | 36.7 µs | uint32 unpack + dequant |
| **S1 Opt v3 + AMD (ours)** | **9.8 µs** | **44.3 µs** | 3ch parallel int wavelet |
| S1 Original + AMD | 11.0 µs | 63.5 µs | Sequential channel wavelet |
| S1 + EdgeBreaker | 106.2 µs | ~600 µs | Sequential CLERS decode |

Our optimized wavelet decoder (S1 Opt v3) adds only **21% overhead** vs the AMD baseline on large meshes, while providing **18% better compression**. On small meshes it's actually **faster** than the baseline. EdgeBreaker provides **3.7x better compression** but at 10x slower decode.

### Compression vs Speed Tradeoff

| Method | Monkey BPV | Compression | Decode Time | Overhead | Cracks |
|--------|----------:|-----------:|------------:|---------:|:------:|
| AMD GlobalGrid (baseline) | 82.88 | 3.5x | 36.7 µs | — | **0** |
| **GlobalInt Wavelet + AMD** | **67.81** | **4.2x** | 44.3 µs | +21% | **0** |
| **GlobalInt Wavelet + EB** | **22.24** | **12.9x** | ~600 µs | +16x | **0** |

### Legacy Encoders (vertex-only, bunny model)

| Encoder Model | Bytes/tri | Bytes/vert | Ratio |
|---------------|----------:|-----------:|------:|
| BaselineEncoder | 18.05 | 35.82 | 1.00 |
| SimpleQuantizator (radix, reordered) | 5.54 | 11.00 | 3.26 |
| PackedGTSQuantizator (radix) | 4.18 | 8.29 | 4.32 |
| GTSParallelogramPredictor (adaptive arith.) | 4.90 | 9.73 | 3.68 |
| PackedGTSEllipsoidFitter(0.0005, 4) | **2.75** | **5.47** | **6.55** |

![bytes_per_triangle_bar_plot](images/bytes_per_triangle_bar_plot.png) ![compression_rate_bar_plot](images/compression_rate_bar_plot.png)

## Architecture

```
OBJ file → Reader.read() → Model → Encoder.encode() → CompressedModel
```

### Crack-Free Meshlet Pipeline
```
Mesh → Global Quantization (integer grid, ensures crack-free shared vertices)
     → Meshlet Generation (greedy region growing, max 256 verts)
     → Per meshlet:
         Connectivity: EdgeBreaker CLERS opcodes OR AMD packed micro-indices
         Vertices: Integer Haar wavelet (lossless on quantized input)
     → CompressedModel
```

### GPU Decode Pipeline (1 CUDA block per meshlet)
```
Phase 1: Connectivity decode
  - AMD: fully parallel (each thread reads 1 packed uint32 triangle)
  - EdgeBreaker: sequential (1 thread, CLERS state machine)
Phase 2: Inverse integer Haar wavelet
  - Optimized: 3 channels processed in parallel (single sync per level)
  - log(N) steps in shared memory, ~4 __syncthreads total
Phase 3: Global dequantize (int → float, fused with wavelet output)
```

### Key Design Decisions

- **Global quantization grid** (from AMD GPUOpen paper): all meshlets share the same quantization step. Per-meshlet offset stored as uint32 per axis. Guarantees identical reconstruction for shared boundary vertices → zero cracks.
- **Integer Haar wavelet**: applied to globally-quantized integer values. The wavelet is lossless on integers (lazy Haar: `detail = odd - even`, `reconstruct: odd = even + detail`). All compression error comes from the initial quantization, not the wavelet.
- **EdgeBreaker vertex ordering**: defines the vertex processing order within each meshlet. Consecutive vertices in this order are spatially adjacent → excellent wavelet compression.

## Running Benchmarks

```bash
# Full encoder comparison on all models
python test_final_benchmark.py

# CUDA GPU decode speed benchmark (requires CuPy)
python benchmark_cuda_decode.py

# Wavelet meshlet compression (all variants)
python test_wavelet_meshlet_compression.py

# Connectivity encode/decode verification
python test_connectivity_verify.py

# Bezier surface compression
python test_bezier_compression.py
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have improvements or suggestions.
# MeshPress — STRIDE

**STRIDE** (STRIp-walked Triangulated Residual Integer Decoder) is a
per-meshlet, GPU-decodable mesh compression format. One fused CUDA kernel
decodes upward of **1.8 G triangles per second** on a consumer NVIDIA RTX 3090,
producing vertex and index buffers in the layout expected by modern mesh-shader
pipelines (DX12 / Vulkan / DirectX12 Ultimate).

Compared head-to-head against AMD's Dense Geometry Format (DGF) on the same
hardware, STRIDE is **strictly smaller on every test mesh** (1.38× to 1.93×
fewer bytes) while remaining **within 6–17 %** of DGF's decode throughput on
million-triangle inputs.

The implementation, full benchmark harness, and reproducibility recipe live in
this repository. The design is documented in the paper at
[`docs/paper_visual_computer_v6.md`](docs/paper_visual_computer_v6.md) (LaTeX
submission bundle under [`docs/paper_tex/`](docs/paper_tex/)). The end-to-end
reproduction recipe is in [`ARTIFACT.md`](ARTIFACT.md).

## Headline numbers

Eight-mesh corpus at uniform 12-bit per-axis (bbox-relative) quantization,
NVIDIA RTX 3090:

| Mesh           | Tris    | STRIDE BPV | STRIDE GPU (M tris/s) | DGF BPV | DGF GPU (M tris/s) |
|----------------|--------:|-----------:|----------------------:|--------:|-------------------:|
| fandisk        | 13 K    |   45.68    |             67        |  71.96  |       474         |
| stanford-bunny | 69 K    |   42.53    |            320        |  64.21  |     1,266         |
| horse          | 97 K    |   41.61    |            456        |  59.66  |     1,225         |
| Monkey         | 1.01 M  |   33.35    |          1,412        |  49.05  |     1,790         |
| Happy Buddha   | 1.09 M  |   43.25    |          1,513        |  51.26  |     1,749         |
| Crab           | 2.14 M  |   42.16    |          1,639        |  48.36  |     1,751         |
| tank           | 3.51 M  |   34.07    |          1,708        |  49.04  |     2,001         |
| xyz-dragon     | 7.22 M  |   33.62    |          1,810        |  46.39  |     2,169         |

On the 7.2 M-triangle Stanford XYZ RGB Dragon: STRIDE 13.5 MB / 3.99 ms decode
vs. DGF 18.7 MB / 3.33 ms. Full multi-codec comparison (Draco, meshoptimizer,
Corto, gltfpack) is in paper §5.

## Quick start

```bash
git clone --recurse-submodules https://github.com/maletsden/meshpress.git
cd meshpress
pip install -r requirements.txt
python scripts/download_models.py --paper        # auto-fetch 5 of 8 paper meshes
python scripts/bench_stride_decode_sweep.py      # GPU decode timing per mesh
python scripts/bench_competitors.py              # full multi-codec sweep
```

Encode an OBJ into STRIDE bytes:

```python
from reader.fast_obj import load_mesh_npy, clean_mesh_npy
from encoder.paradelta_codec import prepare_paradelta_arrays
from encoder.paradelta_v5 import encode_from_prepared_v5

verts, tris = load_mesh_npy("assets/stanford-bunny.obj")
verts, tris = clean_mesh_npy(verts, tris)
prep = prepare_paradelta_arrays(
    verts, tris,
    max_verts=256, max_tris=256, precision_error=0.0005,
    gen_method="joint_learned", strip_method="multiseed",
)
data = encode_from_prepared_v5(prep, verbose=False)
print(f"{len(data)} B, {len(data) * 8 / prep['n_v']:.2f} bpv")
```

Decode on the GPU (CuPy CUDA):

```python
from utils.paradelta_v5_cuda import ParaDeltaV5GpuDecoder
dec = ParaDeltaV5GpuDecoder(data)
v, t = dec.decode_to_host()    # numpy float32 positions + uint32 indices
```

## Repository layout

| Path | Contents |
|---|---|
| `encoder/paradelta_v5.py`        | Reference STRIDE encoder (bit-exact with paper §3.6). |
| `encoder/paradelta_codec.py`     | `prepare_paradelta_arrays` — partition + strip-walk + plan. |
| `encoder/paradelta_v5_nb.py`     | Numba-JIT hot kernels for the encoder. |
| `encoder/_irlp_fit.py`           | Integer-rational linear predictor (IRLP) per-mesh fit (paper §3.5). |
| `utils/paradelta_v5_cuda.py`     | Fused CUDA decoder (paper §4). |
| `utils/meshlet_gen_joint*.py`    | Joint-learned meshlet partitioner (paper §3.2). |
| `utils/meshlet_plan_nb.py`       | Strip-emit traversal + AMD GTS-style connectivity. |
| `reader/fast_obj.py`             | pandas-backed OBJ reader + `.cache.npz` sidecar. |
| `scripts/bench_*.py`             | Bench harnesses. Each writes a CSV. |
| `scripts/verify_*.py`            | Round-trip + crack-free verifiers. |
| `scripts/viz/`                   | Paper figure generators (matplotlib). |
| `bench_cpp/`                     | C++/CUDA bench binaries for STRIDE, DGF, and meshopt. See [`bench_cpp/README.md`](bench_cpp/README.md). |
| `docs/paper_visual_computer_v6.md` | Paper (markdown). |
| `docs/paper_tex/`                | Visual Computer LaTeX submission bundle. |
| `docs/bitstream_spec.md`         | Standalone bitstream layout reference. |
| `models/meshlet_gen_weights.json`| Joint-learned partitioner weights (paper §3.2). |
| `ARTIFACT.md`                    | Reproducibility recipe (tables + figures). |
| `legacy/`                        | Pre-STRIDE encoders (wavelet, LOD, ellipsoid). Not on the paper path. |
| `third_party/DGF-SDK`            | AMD DGF reference (submodule). |
| `third_party/corto`              | Corto codec (submodule). |

## Building the C++ bench harness

Required for the DGF GPU comparison and the C++ meshopt timings. See
[`bench_cpp/README.md`](bench_cpp/README.md). Builds against CUDA 12 and either
MSVC (Windows) or gcc / clang (Linux).

## Citation

Paper accepted-pending at **The Visual Computer** (Springer). Until publication:

```bibtex
@unpublished{stride2026,
  author = {Maletskyi, Denys and Vyklyuk, Yaroslav and Li, Fengping},
  title  = {{STRIDE}: {STRIp-walked} {Triangulated} {Residual} {Integer} {Decoder}
            for {Per-Meshlet} {GPU} {Mesh} {Compression}},
  year   = {2026},
  note   = {Submitted to The Visual Computer (Springer)},
  url    = {https://github.com/maletsden/meshpress}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

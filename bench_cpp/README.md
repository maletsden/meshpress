# `bench_cpp` — native bench harness

C++/CUDA bench drivers for the head-to-head decode comparison reported in
the STRIDE paper §5.3. Three benchmarks live here:

| Source | Output | Used in |
|---|---|---|
| `stride_dup_decode_bench.cu` | `stride_dup_decode_bench` | STRIDE GPU decode throughput (Table 6). |
| `dgf_decode_bench.cu`        | `dgf_decode_bench`        | DGF GPU decode throughput (Table 6). |
| `meshopt_shim.cpp`           | `meshopt_shim.{dll,so}`   | meshopt CPU decode timings (Table 6). |
| `stride_decode_bench.cu`     | `stride_decode_bench`     | Earlier (non-dup) STRIDE variant; kept for reproducibility. |

## Build requirements

- CUDA Toolkit 12.x with `nvcc` on `PATH`.
- A host compiler compatible with your CUDA toolkit:
  - Windows: MSVC 2019 or 2022 (`cl.exe`) via Visual Studio Build Tools.
  - Linux:   gcc ≥ 11 or clang ≥ 14.
- The two third-party submodules populated:
  ```bash
  git submodule update --init --recursive
  cmake -B third_party/DGF-SDK/build -S third_party/DGF-SDK && \
    cmake --build third_party/DGF-SDK/build --config Release
  cmake -B third_party/corto/build -S third_party/corto && \
    cmake --build third_party/corto/build --config Release
  ```

## Building on Windows

```cmd
cd bench_cpp
build_dgf_bench.bat
build_meshopt_shim.bat
```

`stride_dup_decode_bench` and `stride_decode_bench` build alongside DGF.

## Building on Linux

```bash
cd bench_cpp
./build.sh
```

The script invokes `nvcc` directly for the CUDA binaries and `g++` for the
meshopt shim. Adjust `CUDA_HOME` and `CXX` at the top of `build.sh` if your
toolchain lives elsewhere.

## Running

Each benchmark accepts a list of pre-encoded blobs as input. The Python
drivers (`scripts/bench_stride_decode_sweep.py`, `scripts/bench_dgf_decode_sweep.py`)
encode the eight-mesh corpus first, dump the blobs under `bench_cpp/blobs/`,
and then invoke the native binaries. See the per-script docstring for the
exact CLI.

```bash
python scripts/bench_stride_decode_sweep.py
# emits   bench_stride_decode_sweep_q12bbox.csv at the repo root
python scripts/bench_dgf_decode_sweep.py
# emits   bench_dgf_decode_sweep.csv
```

## Outputs not committed

Compiled binaries, the `blobs/` cache, and `dump_out/` are excluded from the
repository via the top-level `.gitignore`. Re-running the build scripts
regenerates them.

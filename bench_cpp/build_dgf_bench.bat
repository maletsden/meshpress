@echo off
setlocal
REM Build dgf_decode_bench.exe — faithful CUDA DGF decoder for paper-honest
REM measurement vs STRIDE. Run from any dir; uses %~dp0 as anchor.

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul

pushd "%~dp0"
nvcc -O3 -arch=sm_86 -std=c++17 ^
     dgf_decode_bench.cu ^
     -o dgf_decode_bench.exe
popd

if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
echo Built dgf_decode_bench.exe

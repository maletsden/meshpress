@echo off
setlocal
REM Build meshopt_shim.dll using MSVC. Run from project root or any dir.
set MO=%~dp0..\third_party\DGF-SDK\external\meshoptimizer
set OUT=%~dp0meshopt_shim.dll

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul

pushd "%~dp0"
cl /nologo /O2 /EHsc /LD /MD ^
   /I "%MO%\src" ^
   meshopt_shim.cpp ^
   "%MO%\src\clusterizer.cpp" ^
   "%MO%\src\allocator.cpp" ^
   /Fe:meshopt_shim.dll
popd

if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
echo Built %OUT%

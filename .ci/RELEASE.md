# Internal wheel releases

This fork reuses PyTorch's binary build infrastructure with a reduced release
matrix. Generated workflows come from
`.github/scripts/generate_ci_workflows.py`; do not edit generated workflow
files directly.

## Supported matrix

| Platform | Backend | Variant | Python |
|---|---|---|---|
| Linux x86-64-v3 | CUDA | CUDA runtime version | 3.14, 3.15 |
| Linux x86-64-v3 | ROCm | MI300X and ROCm runtime version | 3.14, 3.15 |
| Linux x86-64-v3 | XPU | Intel BMG | 3.14, 3.15 |
| macOS arm64 | MPS | Apple Silicon | 3.14, 3.15 |

Linux CPU-only wheels, Windows wheels, libtorch archives, and free-threaded
Python ABIs are not release targets.

## Package identity

The distribution name remains `torch` so packages declaring
`Requires-Dist: torch` continue to resolve correctly. The backend, hardware or
runtime variant, and source commit are encoded in the PEP 440 local version:

```text
2.15.0.dev20260906+cuda.13.4.g0123abcd
2.15.0.dev20260906+rocm.7.14.mi300x.g0123abcd
2.15.0.dev20260906+xpu.bmg.g0123abcd
2.15.0.dev20260906+mps.apple.silicon.g0123abcd
```

The commit component is the first eight hexadecimal digits of the checked-out
commit used for the build.

## Wheel contents

Release builds set `BUILD_TEST=0` and `INSTALL_TEST=0`. Packaging also excludes
standalone Python test modules and C++ helper headers. It retains
`torch.testing._internal` because production modules import utilities from that
package. Internal `_test_*` operator implementations are not selectively
removed because doing so would change the dispatcher surface and ABI.
The public `torch.utils.benchmark` API remains available, but its examples and
operator-fuzzer workloads are not packaged.

The base wheel does not install `fsspec`. Local filesystem distributed
checkpoints remain available; users of fsspec-backed checkpoint storage install
the `torch[distributed-checkpoint]` extra.

Static archives are excluded from wheels. XPU wheels contain the four PyTorch
ELF objects `_C`, `libtorch_python`, `libtorch_cpu`, and `libtorch_xpu`; Intel
and system runtimes remain external.

## XPU runtime dependencies

The XPU wheel declares only runtime packages used by the BMG build:

- Intel compiler and Unified Runtime support
- SYCL runtime
- oneCCL runtime
- oneMKL BLAS, DFT, LAPACK, and classic MKL runtime
- Intel OpenMP, TBB, tcmlib, UMF, and PTI

Development-only oneCCL files, oneMKL RNG and sparse components, MPI, OpenCL,
Python Level Zero tooling, and redundant meta or license packages are not
declared as top-level wheel dependencies.

Some SYCL and Unified Runtime components are loaded dynamically and therefore
do not appear in `DT_NEEDED`. Dependency removal must be validated in a clean
runtime image without `/opt/intel/oneapi`.

## Bundled libuv

Release builds enable `USE_BUNDLED_LIBUV=1`. CMake downloads libuv v1.49.2,
builds only its static archive, and links it into `libtorch_cpu`. Failure to
download or build libuv is fatal; this mode does not fall back to a system
shared library.

Bundled libuv is compiled with hidden visibility and linked with
`--exclude-libs,libuv.a` on Linux. A completed release wheel must not contain
`DT_NEEDED: libuv.so`.

## Local XPU wheel

Use the existing `build-release` directory and repository virtual environment.
The relevant configuration is:

```bash
source /opt/intel/oneapi/setvars.sh --force
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CFLAGS=-march=x86-64-v3
export CXXFLAGS=-march=x86-64-v3
export CMAKE_ARGS=-DTORCH_X86_BASELINE=x86-64-v3
export USE_BUNDLED_LIBUV=1
export USE_XPU=1
export USE_CUDA=0
export USE_ROCM=0
export USE_NCCL=0
export USE_XCCL=1
export USE_C10D_XCCL=1
export BUILD_TEST=0
export INSTALL_TEST=0
export SKBUILD_BUILD_DIR=build-release
export MAX_JOBS=48
```

The resulting wheel is a local `linux_x86_64` artifact. A publishable wheel
must be built and repaired in the repository's manylinux 2.28 pipeline.

## Publishing

Generated workflows currently build, test, and retain wheel artifacts only.
They do not invoke PyTorch's official S3 or R2 upload workflow. Configure the
company private index separately, require human approval for promotion, and do
not overwrite an existing version.

Set the `BINARY_RELEASE_REPOSITORY` GitHub repository variable to the canonical
`owner/repository` name. Release jobs compare `github.repository` with this
variable so forks do not run them, without embedding an organization name in
the workflow sources.

# Internal wheel releases

This fork reuses PyTorch's binary build infrastructure with a reduced release
matrix. Generated workflows come from
`.github/scripts/generate_ci_workflows.py`; do not edit generated workflow
files directly.

## Supported matrix

| Platform | Backend | Variant | Python |
|---|---|---|---|
| Linux x86-64-v3 | CUDA | sm120 and CUDA runtime version | 3.14, 3.15 |
| Linux x86-64-v3 | ROCm | gfx1201 and ROCm runtime version | 3.14, 3.15 |
| Linux x86-64-v3 | XPU | Intel BMG | 3.14, 3.15 |
| macOS arm64 | MPS | Apple Silicon | 3.14, 3.15 |

Linux CPU-only wheels, Windows wheels, libtorch archives, and free-threaded
Python ABIs are not release targets.

## Package identity

The distribution name remains `torch` so packages declaring
`Requires-Dist: torch` continue to resolve correctly. The backend, hardware or
runtime variant, and source commit are encoded in the PEP 440 local version:

```text
2.15.0.dev20260906+cuda.13.3.g0123abcd
2.15.0.dev20260906+rocm.10.0.gfx1201.g0123abcd
2.15.0.dev20260906+xpu.bmg.g0123abcd
2.15.0.dev20260906+mps.apple.silicon.g0123abcd
```

The commit component is the first eight hexadecimal digits of the checked-out
commit used for the build.

## Accelerator-specific indexes

Internal releases use one package-index channel per accelerator while retaining
the `torch` distribution name:

```text
$ACCELERATOR_INDEX_URL-cuda
$ACCELERATOR_INDEX_URL-rocm
$ACCELERATOR_INDEX_URL-xpu
$ACCELERATOR_INDEX_URL-mps
```

The base URL is not recorded in this repository: naming the host would publish
it, and the organisation that owns it, in every clone and in the history. Set
`ACCELERATOR_INDEX_URL` in the environment, or pass `--base-url` to the shared
Breadkernel `accelerator-wheel-pipeline.py` tool.

Each channel exposes its PEP 503 install endpoint under `/+simple/`. Do not put
multiple accelerator variants on the same `torch` simple page: pip cannot use
CUDA, ROCm, XPU, or MPS hardware to resolve wheel versions and may select the
wrong local version.

Install `torch` through the repository tool so users do not need to select or
know the accelerator-specific index URL:

```bash
python /path/to/breadkernel/accelerator-wheel-pipeline.py install torch
```

Detection uses `/dev/kfd` for ROCm, the NVIDIA device or driver interface for
CUDA, Intel DRM devices for XPU, and macOS arm64 for MPS. Set
`ACCELERATOR=cuda|rocm|xpu|mps` when provisioning needs to override
automatic detection. Detection failure and ambiguous CUDA/ROCm environments are
fatal; the tool never falls back to a CPU or another accelerator package. The
`configure` command remains available when an image should persist the selected
index in pip configuration.

Publish wheels with credentials supplied through twine's standard environment
or keyring configuration:

```bash
python /path/to/breadkernel/accelerator-wheel-pipeline.py publish \
  dist/torch-2.15.0+rocm.10.0.gfx1201.g0123abcd-cp314-cp314-linux_x86_64.whl
```

The publish command derives the channel from the wheel's local version,
rejects mixed-accelerator uploads, and uploads to the channel repository URL.
It does not overwrite existing releases or embed credentials. Use `--base-url`
for another internal repository.

Pass `--prune-old` to delete superseded development wheels after the upload
succeeds. Pruning is limited to wheels with the same release line, accelerator
variant, Python ABI, and platform tag, and only removes wheels from earlier
dates. Stable releases, same-day commit builds, and unmatched wheel tags are
retained. Deletion requires `TWINE_USERNAME` and `TWINE_PASSWORD`.

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

Static archives are excluded from wheels. XPU implementations live in
`libtorch_xpu`, while CPU and common runtime implementations live in
`libtorch_cpu`. Small `libc10`, `libtorch`, and `libc10_xpu` compatibility DSOs
preserve the library names used by existing extensions and forward them to
their owner DSOs. CUDA and ROCm builds use the same layout with `libc10_cuda`
and `libc10_hip`; MPS remains part of `libtorch_cpu`. Intel and system runtimes
remain external.

## XPU runtime dependencies

XPU wheels target the Arch Linux hosts used by this fork and use the oneAPI
installation under `/opt/intel/oneapi`. The wheel does not install Intel
runtime packages from PyPI. XPU targets use transitive `DT_RPATH` entries so
indirect dependencies such as the Intel compiler runtime resolve without
sourcing `setvars.sh`.

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
They do not invoke PyTorch's official S3 or R2 upload workflow. Promotion to
the accelerator-specific company index is a separate, human-approved step
using the shared Breadkernel `accelerator-wheel-pipeline.py publish` tool. Do not
overwrite an existing version.

Set the `BINARY_RELEASE_REPOSITORY` GitHub repository variable to the canonical
`owner/repository` name. Release jobs compare `github.repository` with this
variable so forks do not run them, without embedding an organization name in
the workflow sources.

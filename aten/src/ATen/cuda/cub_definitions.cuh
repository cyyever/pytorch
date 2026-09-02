#pragma once

#if !defined(USE_ROCM)
#include <cuda.h> // for CUDA_VERSION
#include <cub/version.cuh>
#else
#include <hipcub/hipcub_version.hpp>
#define CUB_VERSION HIPCUB_CCCL_VERSION
#endif

// CCCL 3.0 brought many bc-breaking changes and 3.4 the iterator spellings this
// tree uses; see
// https://github.com/NVIDIA/cccl/blob/main/docs/cccl/3.0_migration_guide.rst
// CUDA 13.3, the floor cmake enforces, ships CCCL 3.4.2. hipCUB reports its own
// CCCL version through HIPCUB_CCCL_VERSION.
#if !defined(CUB_VERSION) || CUB_VERSION < 300400
#error "PyTorch requires CCCL 3.4 or newer (CUDA 13.3+, or hipCUB reporting HIPCUB_CCCL_VERSION >= 300400)."
#endif

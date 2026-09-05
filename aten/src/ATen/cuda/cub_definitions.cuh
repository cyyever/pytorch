#pragma once

#if !defined(USE_ROCM)
#include <cuda.h> // for CUDA_VERSION
#include <cub/version.cuh>
#else
#include <hipcub/hipcub_version.hpp>
#if !defined(HIPCUB_CCCL_VERSION) || HIPCUB_CCCL_VERSION < 200800
#error "PyTorch requires hipCUB with CCCL 2.8 compatibility or newer."
#endif
#define CUB_VERSION HIPCUB_CCCL_VERSION
#endif

#define USE_GLOBAL_CUB_WRAPPED_NAMESPACE() true

// There were many bc-breaking changes in major version release of CCCL v3.0.0
// Please see https://github.com/NVIDIA/cccl/blob/main/docs/cccl/3.0_migration_guide.rst
#if CUB_VERSION >= 300400
#define CUB_V3_4_PLUS() true
#else
#define CUB_V3_4_PLUS() false
#endif

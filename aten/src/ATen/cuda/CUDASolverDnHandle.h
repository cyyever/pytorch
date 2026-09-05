#pragma once
// Split out of CUDAContextLight.h; see CUDABlasHandle.h. cusolverDn.h is the
// most expensive of the four, and it includes cublas_v2.h, so cuSOLVER callers
// need not include CUDABlasHandle.h as well.

#ifdef CUDART_VERSION
#include <cusolverDn.h>
#endif

#if defined(USE_ROCM)
#include <hipsolver/hipsolver.h>
#endif

#include <torch/headeronly/macros/Export.h>

namespace at::cuda {

TORCH_CUDA_CPP_API cusolverDnHandle_t getCurrentCUDASolverDnHandle();

} // namespace at::cuda

#pragma once
// Split out of CUDAContextLight.h; see CUDABlasHandle.h. cublasLt.h pulls in
// cublas_api.h but not the cublas_v2.h renaming layer, so this is not a
// superset of CUDABlasHandle.h.

#include <cublasLt.h>

#include <torch/headeronly/macros/Export.h>

namespace at::cuda {

TORCH_CUDA_CPP_API cublasLtHandle_t getCurrentCUDABlasLtHandle();

} // namespace at::cuda

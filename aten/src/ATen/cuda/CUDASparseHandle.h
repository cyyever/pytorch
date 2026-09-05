#pragma once
// Split out of CUDAContextLight.h; see CUDABlasHandle.h. cuDSS lives here
// rather than in its own header because it is the direct sparse solver and
// shares this header's callers.

#include <cusparse.h>

#if defined(USE_CUDSS)
#include <cudss.h>
#endif

#include <torch/headeronly/macros/Export.h>

namespace at::cuda {

TORCH_CUDA_CPP_API cusparseHandle_t getCurrentCUDASparseHandle();

#if defined(USE_CUDSS)
TORCH_CUDA_CPP_API cudssHandle_t getCurrentCudssHandle();
#endif

} // namespace at::cuda

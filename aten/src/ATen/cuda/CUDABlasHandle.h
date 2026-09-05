#pragma once
// Split out of CUDAContextLight.h. Adding cublas_v2.h back to that header
// would cost every translation unit reaching it 20,331 preprocessed lines --
// its marginal cost, the rest of its 45k being cuda_runtime_api.h and friends
// already present -- so only the files that name a cuBLAS handle pay it here.

#include <cublas_v2.h>

#include <c10/core/Allocator.h>
#include <torch/headeronly/macros/Export.h>

namespace at::cuda {

// On CUDA, the public handle uses cuBLAS's default workspace unless ATen
// workspace caching is explicitly enabled. ROCm caches workspaces by default.
TORCH_CUDA_CPP_API cublasHandle_t getCurrentCUDABlasHandle(bool setup = true);

// Internal scoped handle for ATen operations. When caching is disabled, it owns
// an eager workspace and restores the default before releasing it; otherwise it
// wraps the cache-backed handle. External users should continue to use
// getCurrentCUDABlasHandle(). Eager scopes for the same underlying handle must
// not overlap because restoring an inner scope replaces the outer workspace.
class TORCH_CUDA_CPP_API CUDABlasHandleWithWorkspace {
 public:
  CUDABlasHandleWithWorkspace(
      cublasHandle_t handle,
      cudaStream_t stream,
      at::DataPtr workspace,
      bool restore_default_workspace);
  ~CUDABlasHandleWithWorkspace();

  CUDABlasHandleWithWorkspace(CUDABlasHandleWithWorkspace&& other) noexcept;
  CUDABlasHandleWithWorkspace(const CUDABlasHandleWithWorkspace&) = delete;
  CUDABlasHandleWithWorkspace& operator=(
      const CUDABlasHandleWithWorkspace&) = delete;
  CUDABlasHandleWithWorkspace& operator=(
      CUDABlasHandleWithWorkspace&&) = delete;

  operator cublasHandle_t() const noexcept {
    return handle_;
  }

 private:
  cublasHandle_t handle_{nullptr};
  cudaStream_t stream_{nullptr};
  at::DataPtr workspace_;
  bool restore_default_workspace_{false};
};

TORCH_CUDA_CPP_API CUDABlasHandleWithWorkspace
getCurrentCUDABlasHandleWithWorkspace();

} // namespace at::cuda

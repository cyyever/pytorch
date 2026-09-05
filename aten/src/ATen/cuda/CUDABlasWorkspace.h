#pragma once
// The cuBLAS and cuBLASLt workspace pool, split out of CUDAContextLight.h.
// Nothing here names a vendor type -- the map is keyed on a (handle, stream)
// pair laundered through void* -- so this header stays cheap, and the ~200
// translation units that reach CUDAContextLight.h through CUDAContext.h stop
// carrying <map>, <shared_mutex> and c10/core/Allocator.h for a pool six files
// use. Measured at 6,128 preprocessed lines each, 4% of that header's cost.

#include <map>
#include <shared_mutex>

#include <cuda_runtime_api.h>

#include <c10/core/Allocator.h>
#include <torch/headeronly/macros/Export.h>

namespace at::cuda {

struct WorkspaceMapWithMutex {
  std::map<std::tuple<void*, void*>, std::pair<at::DataPtr, size_t>> map;
  std::shared_mutex mutex;
};

TORCH_CUDA_CPP_API WorkspaceMapWithMutex& cublas_handle_stream_to_workspace();
TORCH_CUDA_CPP_API WorkspaceMapWithMutex& cublaslt_handle_stream_to_workspace();

TORCH_CUDA_CPP_API void clearCublasWorkspaces();
TORCH_CUDA_CPP_API void clearCublasWorkspacesForStream(cudaStream_t stream);

TORCH_CUDA_CPP_API bool isCUDABlasWorkspaceCachingEnabled();
TORCH_CUDA_CPP_API at::DataPtr allocateCUDABlasWorkspace(size_t size);

TORCH_CUDA_CPP_API size_t getChosenWorkspaceSize();
TORCH_CUDA_CPP_API void setChosenWorkspaceSize(size_t size);
TORCH_CUDA_CPP_API void resetChosenWorkspaceSize();

TORCH_CUDA_CPP_API size_t getCUDABlasLtWorkspaceSize();
TORCH_CUDA_CPP_API void setCUDABlasLtWorkspaceSize(size_t size);
TORCH_CUDA_CPP_API void resetCUDABlasLtWorkspaceSize();
// These return cache-owned allocations even when ATen workspace caching is
// disabled. Use allocateCUDABlasWorkspace for operation-scoped storage.
TORCH_CUDA_CPP_API void* getCUDABlasLtWorkspace();
TORCH_CUDA_CPP_API void* getCUDABlasLtWorkspace(size_t workspace_size);

} // namespace at::cuda

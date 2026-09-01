#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/cuda/CUDAStream.h>

namespace at::cuda {

inline TORCH_CUDA_CPP_API at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kCUDA);
}
} // namespace at::cuda

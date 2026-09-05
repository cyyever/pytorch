#pragma once
// Light-weight version of CUDAContext.h with fewer transitive includes

#include <cuda_runtime_api.h>

#if defined(USE_ROCM) && defined(__noinline__)
#undef __noinline__
#endif

#include <c10/cuda/CUDAFunctions.h>

// getCUDADeviceAllocator returns a pointer, so declaring it needs only the
// name. c10/cuda/CUDAFunctions.h already supplies most of Allocator.h
// transitively, so dropping the explicit include saves 598 lines per
// translation unit rather than the 123k the header costs on its own.
namespace c10 {
struct Allocator;
}

namespace at::cuda {

/*
A common CUDA interface for ATen.

This interface is distinct from CUDAHooks, which defines an interface that links
to both CPU-only and CUDA builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
CUDA builds.

CUDAContext, on the other hand, should be preferred by files only included in
CUDA builds. It is intended to expose CUDA functionality in a consistent
manner.

This means there is some overlap between the CUDAContext and CUDAHooks, but
the choice of which to use is simple: use CUDAContext when in a CUDA-only file,
use CUDAHooks otherwise.

Note that CUDAContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single CUDA context/state.
*/

/**
 * CUDA is available if we compiled with CUDA, and there are one or more
 * devices.  If we compiled with CUDA but there is a driver problem, etc.,
 * this function will report CUDA is not available (rather than raise an error.)
 */
inline bool is_available() {
    return c10::cuda::device_count() > 0;
}

TORCH_CUDA_CPP_API cudaDeviceProp* getCurrentDeviceProperties();

TORCH_CUDA_CPP_API int warp_size();

TORCH_CUDA_CPP_API cudaDeviceProp* getDeviceProperties(c10::DeviceIndex device);

TORCH_CUDA_CPP_API bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device);

TORCH_CUDA_CPP_API c10::Allocator* getCUDADeviceAllocator();

} // namespace at::cuda

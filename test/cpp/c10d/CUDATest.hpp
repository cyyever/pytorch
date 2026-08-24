#pragma once

#include <c10/cuda/CUDAStream.h>

namespace c10d {
namespace test {

#define EXPORT_TEST_API

EXPORT_TEST_API void cudaSleep(at::cuda::CUDAStream& stream, uint64_t clocks);

EXPORT_TEST_API int cudaNumDevices();

} // namespace test
} // namespace c10d

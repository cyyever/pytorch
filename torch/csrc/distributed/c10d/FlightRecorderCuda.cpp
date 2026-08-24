#ifdef USE_C10D_NCCL
#include <cuda_runtime.h>

#include <fstream>

#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

namespace c10d {
/* Helper used by work::getDuration() and nccl flight recorder */
template <>
float getDurationFromEvent<at::cuda::CUDAEvent>(
    at::cuda::CUDAEvent& ncclStartEvent,
    at::cuda::CUDAEvent& ncclEndEvent) {
  TORCH_CHECK(
      ncclEndEvent.query(),
      "getDuration can only be called after work is succeeded.")
  return ncclStartEvent.elapsed_time(ncclEndEvent);
}

template struct FlightRecorder<at::cuda::CUDAEvent>;
} // namespace c10d
#endif // USE_C10D_NCCL

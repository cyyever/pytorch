

#include <nvtx3/nvtx3.hpp>
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

struct RangeHandle {
  nvtxRangeId_t id;
  const char* msg;
};

static void device_callback_range_end(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  nvtxRangeEnd(handle->id);
  free((void*)handle->msg);
  free((void*)handle);
}

static void device_nvtxRangeEnd(void* handle, std::intptr_t stream) {
  C10_CUDA_CHECK(cudaLaunchHostFunc(
      (cudaStream_t)stream, device_callback_range_end, handle));
}

static void device_callback_range_start(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  handle->id = nvtxRangeStartA(handle->msg);
}

static void* device_nvtxRangeStart(const char* msg, std::intptr_t stream) {
  auto handle = static_cast<RangeHandle*>(calloc(1, sizeof(RangeHandle)));
  handle->msg = strdup(msg);
  handle->id = 0;
  TORCH_CHECK(
      cudaLaunchHostFunc(
          (cudaStream_t)stream, device_callback_range_start, (void*)handle) ==
      cudaSuccess);
  return handle;
}

 void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvtx = m.def_submodule("_nvtx", "nvtx3 bindings");
  nvtx.def("rangePushA", nvtxRangePushA);
  nvtx.def("rangePop", nvtxRangePop);
  nvtx.def("rangeStartA", nvtxRangeStartA);
  nvtx.def("rangeEnd", nvtxRangeEnd);
  nvtx.def("markA", nvtxMarkA);
  nvtx.def("deviceRangeStart", device_nvtxRangeStart);
  nvtx.def("deviceRangeEnd", device_nvtxRangeEnd);
}


} // namespace torch::cuda::shared

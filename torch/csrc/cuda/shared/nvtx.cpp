

#include <roctracer/roctx.h>
#include <c10/hip/HIPException.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

struct RangeHandle {
  int id;
  const char* msg;
};

static void device_callback_range_end(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  roctxRangeStop(handle->id);
  free((void*)handle->msg);
  free((void*)handle);
}

static void device_nvtxRangeEnd(void* handle, std::intptr_t stream) {
  C10_CUDA_CHECK(hipLaunchHostFunc(
      (hipStream_t)stream, device_callback_range_end, handle));
}

static void device_callback_range_start(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  handle->id = roctxRangeStartA(handle->msg);
}

static void* device_nvtxRangeStart(const char* msg, std::intptr_t stream) {
  auto handle = static_cast<RangeHandle*>(calloc(1, sizeof(RangeHandle)));
  handle->msg = strdup(msg);
  handle->id = 0;
  TORCH_CHECK(
      hipLaunchHostFunc(
          (hipStream_t)stream, device_callback_range_start, (void*)handle) ==
      hipSuccess);
  return handle;
}

 void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvtx = m.def_submodule("_nvtx", "nvtx3 bindings");
  nvtx.def("rangePushA", roctxRangePushA);
  nvtx.def("rangePop", roctxRangePop);
  nvtx.def("rangeStartA", roctxRangeStartA);
  nvtx.def("rangeEnd", roctxRangeStop);
  nvtx.def("markA", roctxMarkA);
  nvtx.def("deviceRangeStart", device_nvtxRangeStart);
  nvtx.def("deviceRangeEnd", device_nvtxRangeEnd);
}


} // namespace torch::cuda::shared

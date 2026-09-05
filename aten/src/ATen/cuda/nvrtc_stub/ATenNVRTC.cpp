#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

namespace at::cuda {

NVRTC* load_nvrtc() {
  auto self = new NVRTC();
#ifdef USE_ROCM
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#define CREATE_ASSIGN(name) self->name = name;
  AT_FORALL_NVRTC(CREATE_ASSIGN)
#ifdef USE_ROCM
#pragma clang diagnostic pop
#endif
  return self;
}

} // at::cuda

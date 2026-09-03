#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cuda/ScaledBlasDeviceUtils.h>

#include <string>
#include <vector>

namespace at::native::scaled {

#ifdef USE_ROCM
// On ROCm, sm90_only/sm100_only are ignored.
bool scaled_mm_arch_allowed(bool /*sm90_only*/, bool /*sm100_only*/) {
  static const std::vector<std::string> archs = {
      "gfx1200",
      "gfx1201",
      "gfx950",
      "gfx1250",
  };
  return at::detail::getCUDAHooks().isGPUArch(archs);
}
#else
bool scaled_mm_arch_allowed(bool sm90_only, bool sm100_only) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (sm90_only || sm100_only) {
    return (sm90_only && dprops->major == 9) || (sm100_only && dprops->major == 10);
  }
  return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
}
#endif

} // namespace at::native::scaled

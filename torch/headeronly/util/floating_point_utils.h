#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <bit>
#include <cstdint>
#include <type_traits>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, detail)

C10_HOST_DEVICE constexpr float fp32_from_bits(uint32_t w) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float((unsigned int)w);
#else
  return std::bit_cast<float>(w);
#endif
}

C10_HOST_DEVICE constexpr uint32_t fp32_to_bits(float f) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return (uint32_t)__float_as_uint(f);
#else
  return std::bit_cast<uint32_t>(f);
#endif
}

// std::countl_zero is not callable from device code without
// --expt-relaxed-constexpr, which an out-of-tree extension need not pass.
C10_HOST_DEVICE constexpr uint32_t count_leading_zeros(uint32_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (!std::is_constant_evaluated()) {
    return __clz(x);
  }
#endif
  return std::countl_zero(x);
}

HIDDEN_NAMESPACE_END(torch, headeronly, detail)

namespace c10::detail {
using torch::headeronly::detail::count_leading_zeros;
using torch::headeronly::detail::fp32_from_bits;
using torch::headeronly::detail::fp32_to_bits;
} // namespace c10::detail

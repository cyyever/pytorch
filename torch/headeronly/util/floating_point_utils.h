#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <bit>
#include <cstdint>
#include <type_traits>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, detail)

// The device intrinsics are not constexpr, so they have to sit behind
// is_constant_evaluated or a constant expression reaching one is a hard error
// during device compilation rather than merely ill-formed-no-diagnostic.
C10_HOST_DEVICE constexpr float fp32_from_bits(uint32_t w) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (!std::is_constant_evaluated()) {
    return __uint_as_float((unsigned int)w);
  }
#endif
  return std::bit_cast<float>(w);
}

C10_HOST_DEVICE constexpr uint32_t fp32_to_bits(float f) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (!std::is_constant_evaluated()) {
    return (uint32_t)__float_as_uint(f);
  }
#endif
  return std::bit_cast<uint32_t>(f);
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

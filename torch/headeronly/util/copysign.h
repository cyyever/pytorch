#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cmath>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

// Note: the explicit Half and BFloat16 overloads below avoid the float
// round-trip std::copysign would take, which is also faster
template <typename T, typename U>
inline auto copysign(T a, U b) {
  return std::copysign(a, b);
}

// Implement copysign for half precision floats using bit ops
// Sign is the most significant bit for both half and bfloat16 types
inline Half copysign(Half a, Half b) {
  return Half((a.x & 0x7fff) | (b.x & 0x8000), Half::from_bits());
}

inline BFloat16 copysign(BFloat16 a, BFloat16 b) {
  return BFloat16((a.x & 0x7fff) | (b.x & 0x8000), BFloat16::from_bits());
}

HIDDEN_NAMESPACE_END(torch, headeronly)

namespace c10 {
using torch::headeronly::copysign;
} // namespace c10

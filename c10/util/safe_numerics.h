#pragma once
#include <c10/macros/Macros.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace c10 {

template <typename T>
C10_ALWAYS_INLINE bool add_overflows(T a, T b, T* out) requires (std::is_integral_v<T>) {
  return __builtin_add_overflow(a, b, out);
}

C10_ALWAYS_INLINE bool add_overflows(uint64_t a, uint64_t b, uint64_t* out) {
  return add_overflows<uint64_t>(a, b, out);
}

template <typename T>
C10_ALWAYS_INLINE bool mul_overflows(T a, T b, T* out) requires (std::is_integral_v<T>) {
  return __builtin_mul_overflow(a, b, out);
}

C10_ALWAYS_INLINE bool mul_overflows(uint64_t a, uint64_t b, uint64_t* out) {
  return mul_overflows<uint64_t>(a, b, out);
}

template <typename It>
bool safe_multiplies_u64(It first, It last, uint64_t* out) {
  uint64_t prod = 1;
  bool overflow = false;
  for (; first != last; ++first) {
    overflow |= c10::mul_overflows(prod, *first, &prod);
  }
  *out = prod;
  return overflow;
}

template <typename Container>
bool safe_multiplies_u64(const Container& c, uint64_t* out) {
  return safe_multiplies_u64(c.begin(), c.end(), out);
}

} // namespace c10

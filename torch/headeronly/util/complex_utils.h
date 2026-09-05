#pragma once

#if !defined(C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "torch/headeronly/util/complex_utils.h is not meant to be individually included. Include torch/headeronly/util/complex.h instead."
#endif

#include <limits>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

template <typename T>
struct is_complex : public std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

template <typename T>
struct is_complex<c10::complex<T>> : public std::true_type {};

// Extract double from std::complex<double>; is identity otherwise
template <typename T>
struct scalar_value_type {
  using type = T;
};
template <typename T>
struct scalar_value_type<std::complex<T>> {
  using type = T;
};
template <typename T>
struct scalar_value_type<c10::complex<T>> {
  using type = T;
};

HIDDEN_NAMESPACE_END(torch, headeronly)

// numeric_limits below is a specialization and fine; isnan is an overload,
// which [namespace.std]/1 does not allow. It stays for the same reason as the
// reduced-float math overloads: callers spell std::isnan.
// NOLINTBEGIN(bugprone-std-namespace-modification)
namespace std {

template <typename T>
class numeric_limits<c10::complex<T>> : public numeric_limits<T> {};

template <typename T>
bool isnan(c10::complex<T> v) {
  return std::isnan(v.real()) || std::isnan(v.imag());
}

// NOLINTEND(bugprone-std-namespace-modification)
} // namespace std

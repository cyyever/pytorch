#pragma once

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/complex.h>

#include <cmath>
#include <type_traits>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

// std::isnan isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <typename T>
  requires std::is_integral_v<T>
inline C10_HOST_DEVICE bool _isnan(T /*val*/) {
  return false;
}

template <typename T>
  requires std::is_floating_point_v<T>
inline C10_HOST_DEVICE bool _isnan(T val) {
#if defined(__HIPCC__) || defined(__HIPCC__)
  return ::isnan(val);
#else
  return std::isnan(val);
#endif
}

template <typename T>
  requires is_complex<T>::value
inline C10_HOST_DEVICE bool _isnan(T val) {
  return std::isnan(val.real()) || std::isnan(val.imag());
}

template <typename T>
  requires(std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>)
inline C10_HOST_DEVICE bool _isnan(T val) {
  return _isnan(static_cast<float>(val));
}

// Float8_e8m0fnu has an isnan() of its own but is deliberately not handled
// here, so the types are named rather than detected.
template <typename T>
  requires(
      std::is_same_v<T, Float8_e5m2> || std::is_same_v<T, Float8_e4m3fn> ||
      std::is_same_v<T, Float8_e5m2fnuz> || std::is_same_v<T, Float8_e4m3fnuz>)
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

// std::isinf isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <typename T>
  requires std::is_integral_v<T>
inline C10_HOST_DEVICE bool _isinf(T /*val*/) {
  return false;
}

template <typename T>
  requires std::is_floating_point_v<T>
inline C10_HOST_DEVICE bool _isinf(T val) {
#if defined(__HIPCC__) || defined(__HIPCC__)
  return ::isinf(val);
#else
  return std::isinf(val);
#endif
}

template <typename T>
  requires(std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>)
inline C10_HOST_DEVICE bool _isinf(T val) {
  return _isinf(static_cast<float>(val));
}

template <typename T>
  requires std::is_same_v<T, Float8_e5m2>
inline C10_HOST_DEVICE bool _isinf(T val) {
  return val.isinf();
}

// e4m3fn, e5m2fnuz and e4m3fnuz have no infinity to report.
template <typename T>
  requires(
      std::is_same_v<T, Float8_e4m3fn> ||
      std::is_same_v<T, Float8_e5m2fnuz> ||
      std::is_same_v<T, Float8_e4m3fnuz>)
inline C10_HOST_DEVICE bool _isinf(T /*val*/) {
  return false;
}

HIDDEN_NAMESPACE_END(torch, headeronly)

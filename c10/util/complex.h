#pragma once

#include <complex>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <torch/headeronly/util/complex.h>

// These follow the design of the C++20 std::complex free functions, but they
// live in c10 rather than std: [namespace.std] lets a program add a template
// specialization for a program-defined type and nothing else, so defining
// function templates in std -- which is where these used to be -- was
// undefined. The using-declarations below put them at global scope, so
// unqualified calls resolve exactly as they did.

namespace c10 {

template <typename T>
constexpr T real(c10::complex<T> z) {
  return z.real();
}

template <typename T>
constexpr T imag(c10::complex<T> z) {
  return z.imag();
}

template <typename T>
C10_HOST_DEVICE T abs(c10::complex<T> z) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return thrust::abs(static_cast<thrust::complex<T>>(z));
#else
  return std::abs(static_cast<std::complex<T>>(z));
#endif
}

template <typename T>
C10_HOST_DEVICE T arg(c10::complex<T> z) {
  return std::atan2(imag(z), real(z));
}

template <typename T>
constexpr T norm(c10::complex<T> z) {
  return z.real() * z.real() + z.imag() * z.imag();
}

// For std::conj, there are other versions of it:
//   constexpr std::complex<float> conj( float z );
//   template< class DoubleOrInteger >
//   constexpr std::complex<double> conj( DoubleOrInteger z );
//   constexpr std::complex<long double> conj( long double z );
// These are not implemented
// TODO(@zasdfgbnm): implement them as c10::conj
template <typename T>
constexpr c10::complex<T> conj(c10::complex<T> z) {
  return c10::complex<T>(z.real(), -z.imag());
}

// Thrust does not have complex --> complex version of thrust::proj,
// so this function is not implemented at c10 right now.
// TODO(@zasdfgbnm): implement it by ourselves

} // namespace c10

using c10::abs;
using c10::arg;
using c10::conj;
using c10::imag;
using c10::norm;
using c10::real;

#define C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
// math functions are included in a separate file
#include <c10/util/complex_math.h> // IWYU pragma: keep
#undef C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H

namespace c10 {
using torch::headeronly::is_complex;
using torch::headeronly::scalar_value_type;
} // namespace c10

// Code that is generic over the real scalar types and c10::complex needs both
// overload sets, and neither namespace has all of them: the real overloads are
// in std, the complex ones are in c10 and at global scope. Naming this
// namespace at the call site gets both, and unlike an unqualified call it is
// not shadowed by a member function of the same name -- Vectorized<T>::abs()
// and Scalar::log() would each find themselves.
//
// Do not reach for :: here. Global lookup finds C's int abs(int) for a float
// argument, which truncates, and pow computes in double before narrowing.
namespace c10::math {
using std::abs;
using ::abs;
using std::arg;
using ::arg;
using std::conj;
using ::conj;
using std::norm;
using ::norm;
using std::real;
using ::real;
using std::imag;
using ::imag;
using std::acos;
using ::acos;
using std::acosh;
using ::acosh;
using std::asin;
using ::asin;
using std::asinh;
using ::asinh;
using std::atan;
using ::atan;
using std::atanh;
using ::atanh;
using std::cos;
using ::cos;
using std::cosh;
using ::cosh;
using std::exp;
using ::exp;
using std::expm1;
using ::expm1;
using std::log;
using ::log;
using std::log10;
using ::log10;
using std::log1p;
using ::log1p;
using std::log2;
using ::log2;
using std::pow;
using ::pow;
using std::sin;
using ::sin;
using std::sinh;
using ::sinh;
using std::sqrt;
using ::sqrt;
using std::tan;
using ::tan;
using std::tanh;
using ::tanh;
using std::erf;
using ::erf;
using std::erfc;
using ::erfc;
using std::lgamma;
using ::lgamma;
} // namespace c10::math

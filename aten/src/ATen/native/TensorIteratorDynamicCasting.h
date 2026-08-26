#pragma once

#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>
#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>

// This file includes utilities for dynamic_casting done by TensorIterator, see
// CUDALoops.cuh and Loops.h.

// dynamic_casting handles when the types expected by the iterator do not match
// the types of the arguments to the function that is being called. On CUDA, the
// cast is currently pushed down into the kernel (for performance reasons). On
// CPU, there is currently an internal assert that a dynamic_cast is not needed.

namespace at::native {

// `needs_dynamic_casting` compares the types expected by iterator
// (i.e. dtypes of the operands) with the actual type of the arguments
// (and returns) of func_t
template <typename func_t, int nargs = function_traits<func_t>::arity>
struct needs_dynamic_casting {
  static bool check(TensorIteratorBase& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::template arg<nargs - 1>::type;
    using cpp_map = c10::CppTypeToScalarType<cpp_type>;

    if (iter.input_dtype(nargs - 1) != cpp_map::value) {
      return true;
    }
    return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
  }
};

namespace dynamic_casting_detail {

// Multiple-output kernels return the outputs packed together: std::tuple on
// CPU, thrust::tuple on CUDA. Both answer std::tuple_size, which is all this
// needs to walk the outputs.
template <typename T, typename = void>
struct is_tuple_like : std::false_type {};

template <typename T>
struct is_tuple_like<T, std::void_t<decltype(std::tuple_size<T>::value)>>
    : std::true_type {};

template <typename tuple_t, size_t... I>
bool tuple_needs_dynamic_casting(
    TensorIteratorBase& iter,
    std::index_sequence<I...>) {
  return ((iter.dtype(static_cast<int64_t>(I)) !=
           c10::CppTypeToScalarType<std::tuple_element_t<I, tuple_t>>::value) ||
          ...);
}

} // namespace dynamic_casting_detail

template <typename func_t>
struct needs_dynamic_casting<func_t, 0> {
  static bool check(TensorIteratorBase& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::result_type;

    // we could assert output numbers are correct here, but checks
    // (including arity) are currently pushed outside of this struct.
    if constexpr (std::is_void_v<cpp_type>) {
      return false;
    } else if constexpr (dynamic_casting_detail::is_tuple_like<cpp_type>::value) {
      return dynamic_casting_detail::tuple_needs_dynamic_casting<cpp_type>(
          iter, std::make_index_sequence<std::tuple_size_v<cpp_type>>{});
    } else {
      return iter.dtype(0) != c10::CppTypeToScalarType<cpp_type>::value;
    }
  }
};

} // namespace at::native

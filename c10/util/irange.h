// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <algorithm>
#include <type_traits>

#if defined(__CUDACC__)
#include <cuda/std/ranges>
#else
#include <ranges>
#endif

namespace c10 {

namespace detail {
#if defined(__CUDACC__)
namespace views = ::cuda::std::views;
#else
namespace views = ::std::views;
#endif
} // namespace detail

/// Creates an integer range for the half-open interval [begin, end)
/// If end<=begin, then the range is empty.
/// The range has the type of the `end` integer; `begin` integer is
/// cast to this type.
template <typename Integer1, typename Integer2>
  requires std::is_integral_v<Integer1> && std::is_integral_v<Integer2>
constexpr auto irange(Integer1 begin, Integer2 end) {
  const auto b = static_cast<Integer2>(begin);
  return detail::views::iota(b, std::max(b, end));
}

/// Creates an integer range for the half-open interval [0, end)
/// If end<=0, then the range is empty
template <typename Integer>
  requires std::is_integral_v<Integer>
constexpr auto irange(Integer end) {
  return detail::views::iota(Integer{}, std::max(Integer{}, end));
}

} // namespace c10

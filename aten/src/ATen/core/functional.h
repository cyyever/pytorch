#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace c10 {

namespace detail {

// Prefer consuming an owned element when the callable accepts an rvalue.
// This covers callables taking T, T&&, or const T&. Fall back to T& for
// callables that can only operate on an lvalue.
template <typename F, typename T>
inline decltype(auto) fmap_invoke_owned(T& input, const F& fn) {
  if constexpr (std::is_invocable_v<const F&, T&&>) {
    return fn(std::move(input));
  } else {
    static_assert(
        std::is_invocable_v<const F&, T&>,
        "fmap callable must accept the input element");
    return fn(input);
  }
}

} // namespace detail

// A non-consuming input is iterated as const, so a callable that can only bind
// a mutable lvalue fails this constraint and is rejected at the call rather
// than deep inside std::invoke_result. std::ranges::to reserves for a sized
// range, so this keeps the single allocation the hand-written loop had.
template <std::ranges::input_range R, class F>
  requires std::invocable<const F&, std::ranges::range_reference_t<const R&>>
inline auto fmap(const R& inputs, const F& fn) {
  return inputs | std::views::transform(std::cref(fn)) |
      std::ranges::to<std::vector>();
}

// Consuming overload for an owned vector. Prefer passing elements as rvalues
// when the callable supports it. If the map produces the same element type by
// value, transform in place so the result can reuse the vector's allocation.
template<class F, class T>
inline auto fmap(std::vector<T>&& inputs, const F& fn) {
  using raw_result_type = decltype(detail::fmap_invoke_owned(
      std::declval<T&>(), std::declval<const F&>()));
  using result_type = std::remove_cvref_t<raw_result_type>;

  static_assert(
      !std::is_void_v<result_type>,
      "fmap callable must return a value");

  // Do not reuse storage for reference results. In particular, a T&& result
  // may alias input, turning assignment back into the same slot into a
  // self-move.
  if constexpr (
      std::is_same_v<result_type, T> &&
      !std::is_reference_v<raw_result_type> &&
      std::is_assignable_v<T&, raw_result_type>) {
    for (auto&& input : inputs) {
      input = detail::fmap_invoke_owned(input, fn);
    }
    return std::move(inputs);
  } else {
    std::vector<result_type> r;
    r.reserve(inputs.size());
    for (auto&& input : inputs) {
      r.emplace_back(detail::fmap_invoke_owned(input, fn));
    }
    return r;
  }
}

// C++ forbids taking an address of a constructor, so here's a workaround...
// Overload for constructor (R) application
template <typename R, std::ranges::input_range T>
inline std::vector<R> fmap(const T& inputs) {
  return fmap(inputs, [](const auto& input) { return R(input); });
}

// Consuming overload for constructor application. Move from each element when
// R can be constructed from T&&; otherwise preserve the existing lvalue path.
template<typename R, typename T>
inline std::vector<R> fmap(std::vector<T>&& inputs) {
  if constexpr (std::is_same_v<R, T>) {
    return std::move(inputs);
  } else {
    std::vector<R> r;
    r.reserve(inputs.size());
    for (auto&& input : inputs) {
      if constexpr (std::is_constructible_v<R, T&&>) {
        r.emplace_back(std::move(input));
      } else {
        static_assert(
            std::is_constructible_v<R, T&>,
            "fmap result must be constructible from the input element");
        r.emplace_back(input);
      }
    }
    return r;
  }
}

// Deliberately not `inputs | views::filter | ranges::to`: a filter_view is
// forward but not sized, so vector's from_range constructor measures it with
// ranges::distance before filling, running the predicate over every element
// twice (measured: 1999 calls versus 999 for 1000 inputs). Reserving the input
// size and pushing keeps the single pass, at the cost of an exact capacity.
template <std::ranges::input_range R, typename F>
  requires std::predicate<const F&, std::ranges::range_reference_t<const R&>>
inline auto filter(const R& inputs, const F& fn) {
  std::vector<std::ranges::range_value_t<R>> r;
  if constexpr (std::ranges::sized_range<const R&>) {
    r.reserve(std::ranges::size(inputs));
  }
  for (const auto& input : inputs) {
    if (fn(input)) {
      r.push_back(input);
    }
  }
  return r;
}

// Erase in place when T allows it; a T that cannot be move-assigned drops this
// overload from the set and lands on the const& one above.
template <typename F, typename T>
  requires std::is_move_assignable_v<T>
inline std::vector<T> filter(std::vector<T>&& inputs, const F& fn) {
  inputs.erase(
      std::remove_if(
          inputs.begin(),
          inputs.end(),
          [&](auto&& input) { return !fn(input); }),
      inputs.end());
  return std::move(inputs);
}

} // namespace c10

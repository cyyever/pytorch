#pragma once

#include <utility>
namespace c10 {
namespace detail {

template <class... Ts>
struct overloaded_t : Ts... {
  using Ts::operator()...;
};

} // namespace detail

// Construct an overloaded callable combining multiple callables, e.g. lambdas
template <class... Ts>
detail::overloaded_t<Ts...> overloaded(Ts... ts) {
  return {std::move(ts)...};
}

} // namespace c10

#pragma once

#include <cstdint>
#include <functional>

#include <c10/util/hash.h>
#include <c10/util/intrusive_ptr.h>

namespace torch::autograd {

struct Node;

/// Represents a particular input of a function.
struct Edge {
  Edge() noexcept : function(nullptr), input_nr(0) {}

  Edge(c10::intrusive_ptr<Node> function_, uint32_t input_nr_) noexcept
      : function(std::move(function_)), input_nr(input_nr_) {}

  /// Convenience method to test if an edge is valid.
  bool is_valid() const noexcept {
    return function != nullptr;
  }

  // Required for use in associative containers.
  bool operator==(const Edge& other) const noexcept {
    return this->function == other.function && this->input_nr == other.input_nr;
  }

  bool operator!=(const Edge& other) const noexcept {
    return !(*this == other);
  }

  /// The function this `Edge` points to.
  c10::intrusive_ptr<Node> function;

  /// The identifier of a particular input to the function.
  uint32_t input_nr;
};
} // namespace torch::autograd

// Lets Edge be a key of the unordered containers without passing a custom
// hasher to each of them.
namespace std {
template <>
struct hash<torch::autograd::Edge> {
  size_t operator()(const torch::autograd::Edge& edge) const noexcept {
    return c10::get_hash(edge.function, edge.input_nr);
  }
};
} // namespace std

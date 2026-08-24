#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <memory>

namespace torch::jit {

struct Resolver;
using ResolverPtr = std::shared_ptr<Resolver>;

/**
 * class Resolver
 *
 * Represents an "outer environment" in which we can look up names during
 * schema parsing. Only type resolution survives the TorchScript removal.
 */
struct Resolver {
  virtual ~Resolver() = default;

  // Resolve `name` to a TypePtr.
  virtual at::TypePtr resolveType(const std::string& name, const SourceRange& loc) {
    return nullptr;
  }
};

} // namespace torch::jit

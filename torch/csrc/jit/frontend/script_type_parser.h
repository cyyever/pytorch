#pragma once
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/tree_views.h>

namespace torch::jit {

/**
 * class ScriptTypeParser
 *
 * Parses expressions in our typed AST format (TreeView) into types and
 * typenames.
 */
class TORCH_API ScriptTypeParser {
 public:
  explicit ScriptTypeParser() = default;

  c10::TypePtr parseTypeFromExpr(const Expr& expr) const;

  std::optional<std::pair<c10::TypePtr, int32_t>> parseBroadcastList(
      const Expr& expr) const;

  c10::TypePtr parseType(const std::string& str);

 private:
  c10::TypePtr parseTypeFromExprImpl(const Expr& expr) const;

  std::optional<std::string> parseBaseTypeName(const Expr& expr) const;
  at::TypePtr subscriptToType(
      const std::string& typeName,
      const Subscript& subscript) const;
};
} // namespace torch::jit

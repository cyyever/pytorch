#pragma once
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/named_value.h>

#include <ATen/core/function_schema.h>

namespace torch::jit {

// Try to match a list of inputs and keyword 'attributes' to this
// schema. Return the flat list of positional inputs to the call or
// `std::nullopt` on failure (`failure_messages` contains a good error
// report in this case)

struct MatchedSchema {
  std::vector<Value*> inputs;
  std::vector<TypePtr> return_types;
  c10::OptNameList return_field_names;
  std::string schema_name;
};

TORCH_API bool isBlockListedSchema(const FunctionSchema& schema);

TORCH_API Value* emitBuiltinCall(
    const SourceRange& loc,
    Graph& graph,
    Symbol name,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const std::optional<NamedValue>& self = std::nullopt);

// applies implicit conversion from value trying to turn it into type
// concrete_type it succeeds if the return_value->isSubtypeOf(concrete_type)
} // namespace torch::jit

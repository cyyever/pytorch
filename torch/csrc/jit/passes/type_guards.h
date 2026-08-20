#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch::jit {

// Profiling-executor type utilities: the profiling executor annotates a graph
// with prim::profile nodes and specialized TensorTypes, and consumers either
// bake those types in behind a guard or strip them back out again.

TORCH_API void RemoveProfileNodesAndSpecializeTypes(
    std::shared_ptr<Graph>& graph);
TORCH_API bool hasTensorTypeSpecialization(Value* v);
TORCH_API void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph);
TORCH_API void removeTensorTypeSpecializations(Block* block);

using tensor_type_converter_t =
    c10::function_ref<TensorTypePtr(const TensorTypePtr& t)>;

// inserts a TypeCheck pattern
//
// around the guarded node that has a Subgraph attribute, this inserts a pattern
//
//   if TypeCheck(...):
//     guarded_node
//   else:
//     FallbackGraph(...)
//
// The TypeCheck includes the types of all Tensor inputs to the guarded_node,
// as processed by the type_converter, a lambda
// TensorTypePtr(const TensorTypePtr& t). This allows erasing irrelevant
// aspects of the type.
//
// The Fallback graph will have the same subgraph as the guarded node (with the
// expectation that the guarded_node's subgraph will then be optimized.
TORCH_API void insertTypeGuard(
    Node* guarded_node,
    tensor_type_converter_t type_converter,
    c10::Symbol kind);

} // namespace torch::jit

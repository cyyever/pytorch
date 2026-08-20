#include <torch/csrc/jit/passes/type_guards.h>

#include <ATen/core/symbol.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch::jit {

static bool hasTensorTypeSpecialization(Value* v) {
  if (!v->type()->cast<TensorType>()) {
    return false;
  }
  // Constants always produce a specialized tensor type, and TypeCheck nodes
  // are inserted by insertTypeGuard itself and only used by fusion groups that
  // insert proper guards
  if (v->node()->kind() == prim::Constant ||
      v->node()->kind() == prim::TypeCheck) {
    return false;
  }
  if (v->type() == TensorType::get()) {
    return false;
  }
  return true;
}

static void removeTensorTypeSpecialization(Value* v) {
  if (hasTensorTypeSpecialization(v)) {
    v->setType(TensorType::get());
  }
}

static void removeTensorTypeSpecializations(Block* block) {
  for (Value* v : block->inputs()) {
    removeTensorTypeSpecialization(v);
  }
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      removeTensorTypeSpecializations(b);
    }
    for (Value* v : n->outputs()) {
      removeTensorTypeSpecialization(v);
    }
  }
}

void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph) {
  removeTensorTypeSpecializations(graph->block());
}

void insertTypeGuard(
    Node* guarded_node,
    tensor_type_converter_t type_converter,
    Symbol kind) {
  GRAPH_DEBUG("Inserting a typecheck guard for a node", *guarded_node);
  auto subgraph = SubgraphUtils::getSubgraph(guarded_node);

  // Fixup types of the subgraph inputs
  std::vector<Value*> inputs_to_check;
  std::vector<TypePtr> guard_types;
  for (Value* input : guarded_node->inputs()) {
    // We only check inputs of the guarded nodes and expect user to infer
    // intermediates and outputs shapes
    if (!input->type()->cast<TensorType>()) {
      continue;
    }

    // fusion outputs are already guarded
    if (input->node()->kind() == prim::Constant ||
        input->node()->kind() == prim::FusionGroup) {
      continue;
    }
    inputs_to_check.push_back(input);
    guard_types.emplace_back(
        type_converter(input->type()->expect<TensorType>()));
  }
  if (inputs_to_check.empty()) {
    return;
  }

  // Add prim::TypeCheck node
  //
  // TypeCheck nodes  look like the following:
  //   %out1 : Float(2, 3), %out2 : Int(10, 30), %types_match : bool =
  //   prim::TypeCheck(%inp1 : Tensor, %inp2 : Tensor)
  //
  // They have N inputs whose types we are going to check and N+1 outputs. The
  // first N outputs specify expected types and N+1-th output holds the result
  // of the check (bool).
  Node* typecheck_node =
      guarded_node->owningGraph()
          ->create(kind, inputs_to_check, inputs_to_check.size() + 1)
          ->insertBefore(guarded_node);
  typecheck_node->tys_(attr::types, std::move(guard_types));
  Value* typecheck_result = typecheck_node->output(inputs_to_check.size());

  std::unordered_map<Value*, Value*> typechecked_inputs;
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typechecked_inputs[typecheck_node->input(i)] = typecheck_node->output(i);
  }

  // Fixup types of the typecheck node outputs, which are used by the op in
  // execution
  typecheck_node->output(inputs_to_check.size())->setType(BoolType::get());
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typecheck_node->output(i)->setType(typecheck_node->input(i)->type());
  }

  // Insert if
  auto versioning_if =
      guarded_node->owningGraph()
          ->create(prim::If, {typecheck_result}, guarded_node->outputs().size())
          ->insertAfter(typecheck_node);
  for (size_t idx = 0; idx < guarded_node->outputs().size(); ++idx) {
    versioning_if->output(idx)->setType(guarded_node->output(idx)->type());
    guarded_node->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // Fill in the false block. It should contain the unoptimized
  // copy of the fused subgraph.
  WithInsertPoint guard(false_block->return_node());
  const auto subgraph_outputs = insertGraph(
      *guarded_node->owningGraph(), *subgraph, guarded_node->inputs());
  for (Value* output : subgraph_outputs) {
    false_block->registerOutput(output);
  }

  // types get copied to the fallback graph, so remove specializations before
  // replacing
  removeTensorTypeSpecializations(false_block);
  replaceBlockWithFallbackGraph(false_block, guarded_node->inputs());

  // Fill in the true block. It has all inputs type-checked and its
  // body should be the fusion group node.
  guarded_node->moveBefore(true_block->return_node());
  for (size_t idx = 0; idx < guarded_node->inputs().size(); ++idx) {
    if (typechecked_inputs.contains(guarded_node->input(idx))) {
      guarded_node->replaceInput(
          idx, typechecked_inputs.at(guarded_node->input(idx)));
    }
  }
  for (Value* output : guarded_node->outputs()) {
    true_block->registerOutput(output);
  }
}

} // namespace torch::jit

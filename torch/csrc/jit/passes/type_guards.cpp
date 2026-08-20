#include <torch/csrc/jit/passes/type_guards.h>

#include <ATen/core/symbol.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch::jit {

static void removeProfileNodesAndSpecializeTypes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile) {
      GRAPH_DEBUG("Removing prim::profile: %", it->output()->debugName());
      it->output()->replaceAllUsesWith(it->input());
      auto profiled_type = it->ty(attr::profiled_type)->expect<TensorType>();

      TensorTypePtr input_tensor_type = nullptr;
      bool input_is_optional = false;
      if (it->input()->type()->kind() == c10::TypeKind::TensorType) {
        input_tensor_type = it->input()->type()->expect<TensorType>();
      } else {
        auto element_type = it->input()
                              ->type();
        if (element_type->cast<OptionalType>()) {
          input_tensor_type = element_type->expectRef<OptionalType>()
                                          .getElementType()
                                          ->expect<TensorType>();
        } else {
          // This handles the following scenario:
          // 1. profiling nodes are inserted
          // 2. optimizations simplify a Tensor? -> None type
          // 3. Now the input to the prim::profile() is actually a None type.
          element_type->expect<NoneType>();
        }

        input_is_optional = true;
      }

      if (input_is_optional) {
        it.destroyCurrent();
        continue;
      }

      // A value can be profiled with differently typed uses.
      // This can occur from:
      // - having a use which is not executed, so the type will be
      // TensorType::get()
      // - control-flow that depends on tensor type:
      //   if x.size() == 2 op(x) else op(x)
      // - mutation of the value on a field represented in the tensor type
      //   op(x); x.resize_([...]); op(x)

      // The most common case today with num_profiles = 1 is from the first
      // case. Here we can just ignore non-profiled uses, and choose any of the
      // profiled uses. Because we guard all tensor types in the runtime, even
      // if we set a Value to have a profiled type from one use and then execute
      // a use with a different profiled type, we will still be correct.
      // In the future we could consider unifying the types of uses, or adding a
      // type refinement node so uses can have the correct corresponding type.
      if (profiled_type == TensorType::get()) {
        continue;
      }

      // If we encounter non-identical profiled types for the same value, merge
      // them.  This situation can happen if, e.g., loop unrolling duplicates
      // profiled types in a loop body in a manner that isn't logically
      // consistent (see TestTEFuser.test_unrolled_cat).
      if (input_tensor_type == TensorType::get()) {
        it->input()->setType(profiled_type);
      } else {
        it->input()->setType(input_tensor_type->merge(*profiled_type));
      }

      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        removeProfileNodesAndSpecializeTypes(ib);
      }
    }
  }
}

void RemoveProfileNodesAndSpecializeTypes(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG("Before removeProfileNodesAndSpecializeTypes:\n", *graph);
  removeProfileNodesAndSpecializeTypes(graph->block());
  GRAPH_DEBUG("After removeProfileNodesAndSpecializeTypes:\n", *graph);
}

bool hasTensorTypeSpecialization(Value* v) {
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

void removeTensorTypeSpecializations(Block* block) {
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

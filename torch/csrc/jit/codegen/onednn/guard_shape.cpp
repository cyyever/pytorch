#include <torch/csrc/jit/codegen/onednn/guard_shape.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/type_guards.h>

namespace torch::jit::fuser::onednn {

//! [ Note -- prepareFusionGroupAndGuardOutputs implementation ]
//! shamelessly copying code from the old NNC fuser with very little
//! modification, original code at:
//! prepareFusionGroupAndGuardOutputs
//!
//! We have the assumption that LLGA does not have operators
//! depending on the content of the tensor.
void prepareFusionGroupAndGuardOutputs(Block* block) {
  std::vector<Node*> fusion_groups;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      prepareFusionGroupAndGuardOutputs(b);
    }
    if (n->kind() == prim::oneDNNFusionGroup) {
      fusion_groups.push_back(n);
    }
  }
  for (Node* fusion_group : fusion_groups) {
    // TODO: add further optimization pass to removeOutputsUsedOnlyInSize,
    // refer to
    // removeOutputsUsedOnlyInSize
    // removeOutputsUsedOnlyInSize(fusion_group);
    insertTypeGuard(
        fusion_group,
        [](const TensorTypePtr& t) { return t; },
        prim::oneDNNFusionGuard);
  }
}

} // namespace torch::jit::fuser::onednn

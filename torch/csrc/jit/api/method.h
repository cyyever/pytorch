#pragma once

#include <ATen/core/function.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/api/object.h>

namespace torch::jit {

// A bound method of a torchbind (torch::class_) object: the owning object
// plus the unbound Function. Execution goes through Function::run, which for
// torchbind methods is a BuiltinOpFunction that directly invokes the
// registered C++ callable.
struct TORCH_API Method {
  Method(ObjectPtr owner, Function* function)
      : owner_(std::move(owner)), function_(function) {
    TORCH_INTERNAL_ASSERT(function_);
  }

  // The object that owns this method.
  ObjectPtr raw_owner() const {
    return owner_;
  }

  const std::string& name() const {
    return function_->name();
  }

  size_t num_inputs() const {
    return function_->num_inputs();
  }

  Function& function() const {
    return *function_;
  }

 private:
  ObjectPtr owner_;
  Function* function_;
};

} // namespace torch::jit

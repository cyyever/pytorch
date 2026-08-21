#pragma once

#include <ATen/core/function.h>
#include <torch/csrc/jit/ir/ir.h>

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace torch::jit {

struct TORCH_API GraphFunction : public Function {
  GraphFunction(
      c10::QualifiedName name,
      std::shared_ptr<Graph> graph,
      std::function<void(GraphFunction&)> function_creator)
      : name_(std::move(name)),
        graph_(std::move(graph)),
        function_creator_(std::move(function_creator)) {}

  bool isGraphFunction() const override {
    return true;
  }

  void run(Stack& stack) override;

  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch) override;

  std::function<void(GraphFunction&)> function_creator() const {
    return function_creator_;
  }

  std::shared_ptr<Graph> graph() const {
    return graph_;
  }

  const c10::QualifiedName& qualname() const override {
    return name_;
  }

  // if this isn't yet defined, run its method_creator function
  void ensure_defined() override;

  size_t num_inputs() const override {
    return graph()->inputs().size();
  }

  Function& setSchema(FunctionSchema schema) override {
    schema_ = std::make_unique<FunctionSchema>(std::move(schema));
    return *this;
  }

  const FunctionSchema& getSchema() const override;

  void check_single_output() {
    TORCH_CHECK(
        graph()->outputs().size() == 1,
        "Method (but not graphs in general) require a single output. Use None/Tuple for 0 or 2+ outputs");
  }

 private:
  c10::QualifiedName name_;
  // The original, non-optimized graph
  std::shared_ptr<Graph> graph_;

  // an optional function that actually creates the method when
  // ensure_defined() is called. This is used by the compiler so
  // that it can construct methods out of order
  std::function<void(GraphFunction&)> function_creator_;

  // if absent, then we generate a default schema based on the graph
  // mutable because getSchema caches the default schema if one is requested
  // before a call to setSchema
  mutable std::unique_ptr<FunctionSchema> schema_;
};

// Short hands for dynamic_cast<GraphFunction*>.
TORCH_API GraphFunction* tryToGraphFunction(Function& /*function*/) noexcept;
TORCH_API GraphFunction& toGraphFunction(Function& /*function*/);
TORCH_API const GraphFunction& toGraphFunction(const Function& /*function*/);
} // namespace torch::jit

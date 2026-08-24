#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/Exception.h>
#include <c10/util/FunctionRef.h>

namespace c10 {
struct FunctionSchema;
}

namespace at {
TORCH_API void launch(std::function<void()> func);
}

namespace torch::jit {

using Stack = std::vector<at::IValue>;
using Kwargs = std::unordered_map<std::string, at::IValue>;
struct RecursiveMethodCallError : public std::exception {};
using TaskLauncher = std::function<void(std::function<void()>)>;

// A Function has no implicit `self` object bound.
// It contains schema information and the executor that manages the
// execution of the function. Method is a wrapper around an
// underlying Function that also provides a `self` object.
struct TORCH_API Function {
  Function() = default;
  Function(const Function&) = default;
  Function& operator=(const Function&) = default;
  Function(Function&&) noexcept = default;
  Function& operator=(Function&&) noexcept = default;
  virtual std::string_view doc_string() const {
    static constexpr std::string_view no_doc_string;
    return no_doc_string;
  }

  virtual void run(Stack& stack) = 0;

  virtual c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& /*stack*/,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      [[maybe_unused]] TaskLauncher taskLauncher = at::launch) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
    return {};
  }

  at::IValue operator()(Stack stack, const Kwargs& kwargs = Kwargs()) {
    getSchema().checkAndNormalizeInputs(stack, kwargs);
    run(stack);
    return stack.front();
  }

  virtual const c10::QualifiedName& qualname() const = 0;

  const std::string& name() const {
    return qualname().name();
  }

  // if this isn't yet defined, run its method_creator function
  virtual void ensure_defined() = 0;

  virtual const c10::FunctionSchema& getSchema() const = 0;

  virtual size_t num_inputs() const = 0;

  virtual Function& setSchema(c10::FunctionSchema schema) = 0;

  virtual ~Function() = default;
};
} // namespace torch::jit

#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace torch::jit {

namespace {

c10::FunctionSchema defaultSchemaFor(const GraphFunction& function) {
  std::vector<c10::Argument> args;
  std::vector<c10::Argument> returns;
  const Graph& g = *function.graph();
  size_t num_inputs = function.num_inputs();
  for (const auto i : c10::irange(num_inputs)) {
    const Value* v = g.inputs().at(i);
    std::string name = v->hasDebugName() ? v->debugNameBase()
                                         : ("argument_" + std::to_string(i));
    args.emplace_back(std::move(name), unshapedType(g.inputs()[i]->type()));
  }
  for (const auto i : c10::irange(g.outputs().size())) {
    returns.emplace_back("", unshapedType(g.outputs()[i]->type()));
  }
  return {function.name(), "", std::move(args), std::move(returns)};
}

template <typename T, typename F>
T* tryToGraphFunctionImpl(F& function) noexcept {
  if (!function.isGraphFunction()) {
    return nullptr;
  }
  return static_cast<T*>(&function);
}

template <typename T, typename F>
T& toGraphFunctionImpl(F& function) {
  if (auto* g = tryToGraphFunctionImpl<T>(function)) {
    return *g;
  }
  TORCH_INTERNAL_ASSERT(
      false,
      "Failed to downcast a Function to a GraphFunction. "
      "This probably indicates that the JIT calling context needs a "
      "special case on tryToGraphFunction() instead.");
}

static void placeholderCreator(GraphFunction& /*unused*/) {
  throw RecursiveMethodCallError();
}

} // namespace

void GraphFunction::run(Stack& /*stack*/) {
  TORCH_CHECK(
      false,
      "TorchScript function execution is no longer supported: ",
      qualname().qualifiedName());
}

c10::intrusive_ptr<c10::ivalue::Future> GraphFunction::runAsync(
    Stack& stack,
    TaskLauncher /*taskLauncher*/) {
  run(stack);
  return {};
}

void GraphFunction::ensure_defined() {
  if (function_creator_) {
    auto creator = function_creator_;
    function_creator_ = placeholderCreator;
    creator(*this);
    function_creator_ = nullptr;
  }
}

const c10::FunctionSchema& GraphFunction::getSchema() const {
  if (schema_ == nullptr) {
    schema_ = std::make_unique<c10::FunctionSchema>(defaultSchemaFor(*this));
  }
  return *schema_;
}

GraphFunction* tryToGraphFunction(Function& function) noexcept {
  return tryToGraphFunctionImpl<GraphFunction>(function);
}

GraphFunction& toGraphFunction(Function& function) {
  return toGraphFunctionImpl<GraphFunction>(function);
}

const GraphFunction& toGraphFunction(const Function& function) {
  return toGraphFunctionImpl<const GraphFunction>(function);
}

} // namespace torch::jit

#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>

namespace torch::jit {

namespace {

static ObjectPtr create_module_object(
    c10::QualifiedName class_name,
    std::shared_ptr<CompilationUnit> cu,
    bool shouldMangle = false) {
  // If the name is unqualified, prepend a `__torch__`, similar to what Python
  // does with `__main__` for top-level code.
  if (class_name.prefix().empty()) {
    class_name = c10::QualifiedName("__torch__", class_name.name());
  }
  if (shouldMangle && cu->get_class(class_name) != nullptr) {
    class_name = cu->mangle(class_name);
  }
  auto cls = ClassType::create(std::move(class_name), cu, /*is_module=*/true);
  cu->register_type(cls);
  return c10::ivalue::Object::create(
      c10::StrongTypePtr(std::move(cu), std::move(cls)), 0);
}

} // namespace

Module::Module(c10::QualifiedName class_name)
    : Object(create_module_object(
          std::move(class_name),
          std::make_shared<CompilationUnit>())) {}

Module::Module(
    std::shared_ptr<CompilationUnit> cu,
    const c10::ClassTypePtr& type)
    : Object(c10::ivalue::Object::create(
          c10::StrongTypePtr(std::move(cu), type),
          type->numAttributes())) {}

Module::Module(
    c10::QualifiedName class_name,
    std::shared_ptr<CompilationUnit> cu,
    bool shouldMangle)
    : Object(create_module_object(
          std::move(class_name),
          std::move(cu),
          shouldMangle)) {}

Method::Method(ModulePtr owner, Function* function)
    : owner_(std::move(owner)), function_(function) {}

Module Method::owner() const {
  return Module(owner_);
}

ObjectPtr Method::raw_owner() const {
  return owner_;
}

void Method::run(Stack& stack) {
  stack.insert(stack.begin(), owner()._ivalue()); // self
  RECORD_TORCHSCRIPT_FUNCTION(name(), stack);
  function_->run(stack);
}

IValue Method::operator()(std::vector<IValue> stack, const Kwargs& kwargs)
    const {
  stack.insert(stack.begin(), owner()._ivalue()); // self
  RECORD_TORCHSCRIPT_FUNCTION(name(), stack);
  return (*function_)(std::move(stack), kwargs);
}

c10::intrusive_ptr<c10::ivalue::Future> Method::run_async(
    std::vector<IValue> stack,
    const Kwargs& kwargs,
    TaskLauncher taskLauncher) {
  stack.insert(stack.begin(), owner()._ivalue());
  RECORD_TORCHSCRIPT_FUNCTION(name(), stack);

  function_->getSchema().checkAndNormalizeInputs(stack, kwargs);
  return function_->runAsync(stack, std::move(taskLauncher));
}

void Method::setArgumentNames(
    std::vector<std::string>& argumentNamesOut) const {
  TORCH_INTERNAL_ASSERT(function_);
  auto& arguments = function_->getSchema().arguments();
  argumentNamesOut.reserve(arguments.size());
  for (auto& argument : arguments) {
    if (argument.name() == "self") {
      continue;
    }
    argumentNamesOut.push_back(argument.name());
  }
}

} // namespace torch::jit

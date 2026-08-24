#include <ATen/core/ivalue.h>
#include <ATen/core/class_type.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/serialization/import.h>

namespace torch::jit {

// Decouple how to get obj from type. TorchScript __setstate__ execution is no
// longer supported, so objects are always reconstructed from their attribute
// dict.
c10::intrusive_ptr<c10::ivalue::Object> ObjLoaderFunc(
    const at::StrongTypePtr& type,
    IValue input) {
  const auto& cls = type.type_->expectRef<at::ClassType>();
  size_t n = cls.numAttributes();
  auto dict = std::move(input).toGenericDict();
  auto obj = c10::ivalue::Object::create(type, n);
  for (const auto i : c10::irange(n)) {
    obj->setSlot(i, dict.at(cls.getAttributeName(i)));
  }
  return obj;
}

} // namespace torch::jit

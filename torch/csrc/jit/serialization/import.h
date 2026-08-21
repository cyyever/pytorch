#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/Export.h>

#include <functional>

namespace torch::jit {

using ObjLoader = std::function<
    c10::intrusive_ptr<c10::ivalue::Object>(const at::StrongTypePtr&, IValue)>;

TORCH_API c10::intrusive_ptr<c10::ivalue::Object> ObjLoaderFunc(
    const at::StrongTypePtr& type,
    IValue input);

} // namespace torch::jit

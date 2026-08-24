#include <ATen/Context.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <c10/util/irange.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace torch::jit {
namespace {

// These ops outlive TorchScript: FakeTensor, functional tensor, nested
// tensor, checkpoint, the flop counter and the meta/symbolic machinery all
// dispatch on them from Python. The rest of the upstream prim op set was
// interpreter machinery and stays deleted.
void device(Stack& stack) {
  push(stack, pop(stack).toTensor().device());
}

void dtype(Stack& stack) {
  push(stack, static_cast<int64_t>(pop(stack).toTensor().scalar_type()));
}

void layout(Stack& stack) {
  push(stack, pop(stack).toTensor().layout());
}

void size(Stack& stack) {
  pack(stack, pop(stack).toTensor().sizes().vec());
}

void sym_size(Stack& stack) {
  pack(stack, pop(stack).toTensor().sym_sizes().vec());
}

void stride(Stack& stack) {
  pack(stack, pop(stack).toTensor().strides().vec());
}

void sym_stride(Stack& stack) {
  pack(stack, pop(stack).toTensor().sym_strides().vec());
}

void is_non_overlapping_and_dense(Stack& stack) {
  auto t = pop(stack).toTensor();
  push(stack, t.unsafeGetTensorImpl()->is_non_overlapping_and_dense());
}

void is_strides_like_format(Stack& stack) {
  auto memory_format = pop(stack).toMemoryFormat();
  auto t = pop(stack).toTensor();
  push(stack, t.unsafeGetTensorImpl()->is_strides_like(memory_format));
}

// reference function THPVariable_to in python_variable_methods.cpp
at::Tensor to_dispatch(
    at::Tensor self,
    std::optional<at::Device> device,
    std::optional<at::ScalarType> scalarType,
    bool non_blocking,
    bool copy) {
  if (device && device->is_cuda()) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  }
  if (!device && !scalarType && !copy) {
    return self;
  } else if (!device) {
    return self.to(*scalarType, non_blocking, copy);
  } else if (!scalarType) {
    return self.to(*device, non_blocking, copy);
  }
  return self.to(*device, *scalarType, non_blocking, copy);
}

void grad_sum_to_size(Stack& stack) {
  auto [self, size] = pop<IValue, IValue>(stack);
  if (size.isNone()) {
    push(stack, std::move(self));
  } else {
    push(stack, at::sum_to(self.toTensor(), size.toDimVector()));
  }
}

void list_to_tensor(Stack& stack) {
  c10::List<int64_t> l = pop(stack).toIntList();
  auto t = at::empty({static_cast<int64_t>(l.size())}, at::dtype(at::kInt));
  for (const auto i : c10::irange(l.size())) {
    t[i] = l.get(i);
  }
  push(stack, std::move(t));
}

void to_prim_device(Stack& stack) {
  auto [non_blocking, copy] = pop<bool, bool>(stack);
  std::optional<at::ScalarType> scalarType = pop(stack).toOptional<at::ScalarType>();
  std::optional<c10::Device> device = pop(stack).toOptional<c10::Device>();
  at::Tensor self = pop(stack).toTensor();
  push(stack, to_dispatch(self, device, scalarType, non_blocking, copy));
}

RegisterOperators reg({
    std::optional<Operator>(Operator(
        "prim::device(Tensor a) -> Device",
        device,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "prim::dtype(Tensor a) -> int",
        dtype,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "prim::layout(Tensor a) -> Layout",
        layout,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "aten::size(Tensor self) -> int[]",
        size,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "aten::sym_size(Tensor self) -> SymInt[]",
        sym_size,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "aten::stride(Tensor self) -> int[]",
        stride,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "aten::sym_stride(Tensor self) -> SymInt[]",
        sym_stride,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "aten::is_non_overlapping_and_dense(Tensor self) -> bool",
        is_non_overlapping_and_dense,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "aten::is_strides_like_format(Tensor self, MemoryFormat memory_format) -> bool",
        is_strides_like_format,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)",
        to_prim_device,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "aten::_list_to_tensor(int[] self) -> Tensor",
        list_to_tensor,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
    std::optional<Operator>(Operator(
        "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)",
        grad_sum_to_size,
        c10::AliasAnalysisKind::FROM_SCHEMA)),
});

} // namespace
} // namespace torch::jit

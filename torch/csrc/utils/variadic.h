#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Variadic.h>
#include <torch/csrc/autograd/variable.h>

#include <utility>

namespace torch {

using at::IterArgs;

struct CountTensors : IterArgs<CountTensors> {
  size_t out = 0;
  void operator()(const at::Tensor& x) {
    out += 1;
  }
  void operator()(const std::optional<at::Tensor>& x) {
    out += x.has_value();
  }
  void operator()(at::ArrayRef<at::Tensor> xs) {
    out += xs.size();
  }
};

template <typename... Args>
size_t count_tensors(Args&&... args) {
  return CountTensors().apply(std::forward<Args>(args)...).out;
}

} // namespace torch

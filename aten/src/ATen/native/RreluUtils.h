#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/util/Exception.h>

namespace at::native {

// Shared by the CPU, CUDA and XPU rrelu_with_noise kernels, which each write
// self.numel() values through noise and would otherwise repeat these checks --
// and their messages, one of which a regression test matches on -- three times.
inline void check_rrelu_with_noise_inputs(
    const TensorBase& self,
    const TensorBase& noise,
    bool training) {
  TORCH_CHECK(
      self.sym_sizes() == noise.sym_sizes(),
      "noise tensor shape must match self tensor shape. Got self.shape = ",
      self.sym_sizes(),
      " noise.shape = ",
      noise.sym_sizes());
  // The shape check above also passes for an expanded (0-stride) noise, which
  // cannot hold the self.numel() distinct values the training kernels produce:
  // CPU writes them all into one element, and CUDA and XPU write them into the
  // temporary noise.contiguous() returns and then discard it.
  if (training) {
    TORCH_CHECK(
        noise.is_contiguous(),
        "rrelu_with_noise: noise tensor must be contiguous, got one with sizes ",
        noise.sizes(),
        " and strides ",
        noise.strides());
  }
}

} // namespace at::native

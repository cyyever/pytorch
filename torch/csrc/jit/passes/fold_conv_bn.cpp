#include <torch/csrc/jit/passes/fold_conv_bn.h>

#include <ATen/TensorOperators.h>
#include <ATen/core/DimVector.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/rsqrt.h>
#endif

namespace torch::jit {

std::tuple<at::Tensor, at::Tensor> computeUpdatedConvWeightAndBias(
    const ConvBNParameters& p) {
  at::Tensor bn_var_rsqrt = at::rsqrt(p.bn_rv + p.bn_eps);
  const int64_t ndim = p.conv_w.dim();
  at::DimVector sizes(ndim, 1);
  sizes.at(0) = -1;

  auto conv_w_dtype = p.conv_w.dtype();
  auto conv_b_dtype = p.conv_b.dtype();

  at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape(sizes);
  at::Tensor new_b = (p.conv_b - p.bn_rm) * bn_var_rsqrt * p.bn_w + p.bn_b;
  return std::make_tuple(new_w.to(conv_w_dtype), new_b.to(conv_b_dtype));
}

} // namespace torch::jit

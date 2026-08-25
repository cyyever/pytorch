// Schema definitions for the mkldnn:: operators that the XPU backend implements.
//
// The CPU file that defined them went away with USE_MKLDNN, but the
// TORCH_LIBRARY_IMPL(mkldnn, XPU) blocks in Conv.cpp and Linear.cpp still
// register against them.
//
// Only the overloads XPU actually implements are declared here.

#include <torch/library.h>

namespace at::native::xpu {

TORCH_LIBRARY(mkldnn, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_linear_pointwise(Tensor X, Tensor W, Tensor? B, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_linear_pointwise.binary(Tensor X, Tensor other, Tensor W, Tensor? B, str attr) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_convolution_pointwise(Tensor X, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_convolution_pointwise.binary(Tensor X, Tensor other, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn::_convolution_pointwise_.binary(Tensor(a!) other, Tensor X, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor(a!) Y"));
}

} // namespace at::native::xpu

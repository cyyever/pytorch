// Schema definitions for the onednn:: operators that the XPU backend implements.
//
// These used to live in aten/src/ATen/native/quantized/library.cpp, which went
// away with the quantized library. The TORCH_LIBRARY_IMPL(onednn, XPU) blocks
// in qconv.cpp and qlinear.cpp still register against them.
//
// Only the overloads XPU actually implements are declared here.

#include <torch/library.h>

namespace at::native::xpu {

TORCH_LIBRARY(onednn, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qconv_prepack(Tensor weight, Tensor w_scales, float x_scale, int x_zp, int[] stride, int[] padding, int[] dilation, int groups, int[]? x_shape=None) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qconv1d_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qconv2d_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qconv3d_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qconv_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qconv_pointwise.tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qconv2d_pointwise.binary(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor(a!) qaccum, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, float accum_scale, int accum_zero_point, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qconv2d_pointwise.binary_tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor(a!) qaccum, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, float accum_scale, int accum_zero_point, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qlinear_prepack(Tensor weight, int[]? x_shape) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qlinear_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, str post_op_name, Scalar?[] post_op_args, str post_op_algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qlinear_pointwise.tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, str post_op_name, Scalar?[] post_op_args, str post_op_algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qlinear_pointwise.binary(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? other, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, float other_scale, int other_zp, str binary_post_op, float binary_alpha, str unary_post_op, Scalar?[] unary_post_op_args, str unary_post_op_algorithm) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "onednn::qlinear_pointwise.binary_tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? other, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, float other_scale, int other_zp, str binary_post_op, float binary_alpha, str unary_post_op, Scalar?[] unary_post_op_args, str unary_post_op_algorithm) -> Tensor"));
}

} // namespace at::native::xpu

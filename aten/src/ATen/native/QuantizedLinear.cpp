#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/cpp_custom_type_hack.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation_native.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_native.h>
#include <ATen/ops/fbgemm_pack_gemm_matrix_fp16_native.h>
#endif

#include <c10/util/irange.h>

#ifdef USE_FBGEMM
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wextra-semi")
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/QuantUtils.h>
C10_DIAGNOSTIC_POP()
#endif // USE_FBGEMM

namespace caffe2 {
CAFFE_KNOWN_TYPE(c10::intrusive_ptr<LinearPackedParamsBase>)
} // namespace caffe2

#ifdef USE_FBGEMM
namespace caffe2 {
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(fbgemm::PackBMatrix<int8_t>)
CAFFE_KNOWN_TYPE(c10::intrusive_ptr<PackedLinearWeightFp16>)
} // namespace caffe2
#endif // USE_FBGEMM

namespace at::native {

#ifdef USE_FBGEMM

namespace {

float RawUint16ToFp16(unsigned short value) {
  // Convert raw 16 bits half precision floating point number
  // to single precision floating point number.
  const unsigned short sign_bits = value >> 15;
  const unsigned short exponent_bits = value >> 10 & 0x1f;
  const unsigned short significand_bits = value & 0x3ff;

  const float sign = sign_bits ? -1 : 1;
  const float significand =
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      1 + significand_bits * 0.0009765625f; // 0.0009765625f = 0x1p-10 = 2^-10
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const float exponent = exponent_bits - 0xf;

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return sign * std::ldexp(significand, exponent);
}

template <typename T>
bool CheckAndSaturate(T max_val, T* element) {
  if (*element > max_val) {
    *element = max_val;
    return true;
  }
  if (*element < -max_val) {
    *element = -max_val;
    return true;
  }
  return false;
}

// The range for using FP16 quantization of weights requires that the elements
// should be in the range of [5.96e-8, 65504]. If it is out of range, then the
// number will be saturated to max or min representable values by FP16.
void HandleWeightsSaturation(int64_t N, float* weight) {
  const float kFp16Max = RawUint16ToFp16(0x7BFF);
  bool found_out_of_range = false;
  for (const auto i : c10::irange(N)) {
    if (CheckAndSaturate<float>(kFp16Max, weight + i)) {
      found_out_of_range = true;
    }
  }
  if (found_out_of_range) {
    TORCH_WARN("FOUND weight out of range ");
  }
}

} // namespace

Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor& weight) {
  TORCH_WARN_ONCE("fbgemm_pack_gemm_matrix_fp16 is deprecated "
                  "and will be removed in a future PyTorch release.")

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

  const int64_t K = weight.size(1);
  const int64_t N = weight.size(0);
  Tensor weight_contig = weight.contiguous();
  float* weight_contig_ptr = weight_contig.mutable_data_ptr<float>();
  HandleWeightsSaturation(K * N, weight_contig_ptr);

  // TODO(mingzhe09088):
  // Consider using a functor here in PackedGemmMatrixFP16
  // Comments from (XQ): Not entirely sure this make_unique is safe. make_unique
  // is created with regular "new", and freed through TypeMetaData::deleteFn in
  // this function. This is perfectly fine if the tensors are created and freed
  // within this translation unit. It might be very problematic if that tensor
  // flows across dll boundaries.
  auto ptr = std::make_unique<fbgemm::PackedGemmMatrixFP16>(
      fbgemm::matrix_op_t::Transpose, K, N, 1, weight_contig_ptr);
  c10::intrusive_ptr<LinearPackedParamsBase> packed_weight =
      c10::make_intrusive<PackedLinearWeightFp16>(std::move(ptr), std::nullopt);
  auto unique_ptr_wrapper =
      std::make_unique<decltype(packed_weight)>(std::move(packed_weight));
  return cpp_custom_type_hack::create(
      std::move(unique_ptr_wrapper), weight.options());
}

Tensor fbgemm_linear_fp16_weight_fp32_activation(
    const Tensor& input,
    const Tensor& packed_weight,
    const std::optional<Tensor>& bias,
    at::Tensor& output) {
  TORCH_WARN_ONCE("fbgemm_linear_fp16_weight_fp32_activation is deprecated "
                  "and will be removed in a future PyTorch release.")

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

  const Tensor input_contig = input.contiguous();
  const float* input_ptr = input_contig.const_data_ptr<float>();

  // Pull out the PackedGemmMatrixFP16 instance from the owning tensor
  const fbgemm::PackedGemmMatrixFP16& packed_weight_fp16 =
      *c10::dynamic_intrusive_pointer_cast<PackedLinearWeightFp16>(
           cpp_custom_type_hack::cast<
               c10::intrusive_ptr<LinearPackedParamsBase>>(packed_weight))
           ->w;

  TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
  TORCH_CHECK(input.dim() >= 2);

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const int64_t M = size_to_dim_(input.dim() - 1, input.sizes());
  const int64_t N = packed_weight_fp16.numCols();

  std::vector<int64_t> output_size = input.sizes().vec();
  output_size.back() = N;
  // Resize output Tensor
  output.resize_(output_size);

  // Call the fp16 gemm interface
  fbgemm::cblas_gemm_compute(
      fbgemm::matrix_op_t::NoTranspose,
      M,
      input_ptr,
      packed_weight_fp16,
      0.0f,
      output.mutable_data_ptr<float>());

  // Add bias term
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
  const Tensor& bias_ = *bias_maybe_owned;
  if (bias_.defined()) {
    TORCH_CHECK(bias_.dim() == 1);
    output.add_(bias_);
  }

  return output;
}

Tensor fbgemm_linear_fp16_weight_fp32_activation(
    const Tensor& input,
    const Tensor& packed_weight,
    const std::optional<Tensor>& bias) {
      at::Tensor output = at::empty({0}, input.options().dtype(at::kFloat));
      return at::native::fbgemm_linear_fp16_weight_fp32_activation(input, packed_weight, bias, output);
  }

Tensor fbgemm_linear_fp16_weight(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias) {
  return at::native::fbgemm_linear_fp16_weight_fp32_activation(
      input, packed_weight, bias);
}

Tensor fbgemm_linear_fp16_weight(
  const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias,
  at::Tensor& output) {
  return at::native::fbgemm_linear_fp16_weight_fp32_activation(
      input, packed_weight, bias, output);
}

#else // USE_FBGEMM
Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor& weight) {
  TORCH_WARN_ONCE("fbgemm_pack_gemm_matrix_fp16 is deprecated "
                  "and will be removed in a future PyTorch release.")

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_linear_fp16_weight_fp32_activation(
    const Tensor& input,
    const Tensor& packed_weight,
    const std::optional<Tensor>& bias,
    at::Tensor& output) {
  TORCH_WARN_ONCE("fbgemm_linear_fp16_weight_fp32_activation is deprecated "
                  "and will be removed in a future PyTorch release.")

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_linear_fp16_weight_fp32_activation(
    const Tensor& input,
    const Tensor& packed_weight,
    const std::optional<Tensor>& bias) {
  TORCH_WARN_ONCE("fbgemm_linear_fp16_weight_fp32_activation is deprecated "
                  "and will be removed in a future PyTorch release.")

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_linear_fp16_weight(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias,
    at::Tensor& output) {
  TORCH_WARN_ONCE("fbgemm_linear_fp16_weight is deprecated "
                  "and will be removed in a future PyTorch release.")

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_linear_fp16_weight(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias) {
  TORCH_WARN_ONCE("fbgemm_linear_fp16_weight is deprecated "
                  "and will be removed in a future PyTorch release.")

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

#endif // USE_FBGEMM

} // namespace at::native

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/FakeQuantAffine.h>
#include <c10/util/irange.h>

#include <cmath>
#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif
#if defined(__ARM_NEON__) || defined(__aarch64__)
#include <ATen/quantized/Quantizer.h>
#include <arm_neon.h>
#endif

// NOLINTBEGIN(*-c-arrays)

namespace at::native {

namespace {

void check_tensor_memory_format(const Tensor& ref, const Tensor& other) {
  TORCH_CHECK(
      ref.is_contiguous(ref.suggest_memory_format()),
      "Quantized tensor should be contiguous");
  TORCH_CHECK(
      other.is_contiguous(ref.suggest_memory_format()),
      "Float tensor should be contiguous "
      "in same memory format as quantized tensor");
}


void _fake_quantize_tensor_helper(
  Tensor& output,
  Tensor& mask,
  const Tensor& input,
  int fake_quant_on,
  float sc,
  int64_t z_point,
  int64_t quant_min,
  int64_t quant_max) {

  float inv_scale = 1.0f / sc;

  auto iter_combined = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(output)
    .add_output(mask)
    .add_input(input)
    .build();

  if (at::isReducedFloatingType(input.scalar_type())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_type_handling", [&]() {
      iter_combined.for_each([&](char** data, const int64_t* strides, int64_t n) {
        for (const auto i : c10::irange(n)) {
          scalar_t* output_val = (scalar_t*)(data[0] + i * strides[0]);
          bool* mask_val = (bool*)(data[1] + i * strides[1]);
          scalar_t* input_val = (scalar_t*)(data[2] + i * strides[2]);

          if (fake_quant_on) {
            auto qval_f = z_point + std::nearbyint(*input_val * inv_scale);
            const auto qval = static_cast<int64_t>(std::fmin(std::fmax(qval_f, quant_min), quant_max));
            *output_val = (qval - z_point) * sc;
            *mask_val = ((quant_min <= qval_f) && (qval_f <= quant_max));
          } else {
            *output_val = *input_val;
            *mask_val = 1;
          }
        }
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_type_handling", [&] {
      iter_combined.for_each([&](char** data, const int64_t* strides, int64_t n) {
        for (const auto i : c10::irange(n)) {
          scalar_t* output_val = (scalar_t*)(data[0] + i * strides[0]);
          bool* mask_val = (bool*)(data[1] + i * strides[1]);
          scalar_t* input_val = (scalar_t*)(data[2] + i * strides[2]);

          if (fake_quant_on) {
            auto qval_f = z_point + std::nearbyint(*input_val * inv_scale);
            const auto qval = static_cast<int64_t>(std::fmin(std::fmax(qval_f, quant_min), quant_max));
            *output_val = (qval - z_point) * sc;
            *mask_val = ((quant_min <= qval_f) && (qval_f <= quant_max));
          } else {
            *output_val = *input_val;
            *mask_val = 1;
          }
        }
      });
    });
  }
}

void fake_quantize_tensor_cachemask_kernel(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max) {
  _fake_quantize_tensor_helper(output, mask, input, 1, sc, z_point, quant_min, quant_max);
}

void fake_quantize_tensor_cachemask_tensor_qparams_kernel(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    const Tensor& sc,
    const Tensor& z_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max) {
  _fake_quantize_tensor_helper(output, mask, input, fake_quant_enabled.item().toInt(), sc.item().toFloat(), z_point.item().toInt(), quant_min, quant_max);
}

void fake_quantize_learnable_tensor_grad_kernel_cpu(
    TensorIterator& iter,
    float scale,
    float inv_scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float dscale_small = quant_min - zero_point;
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float dscale_big = quant_max - zero_point;
  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    /*  When a for_each call is made on a TensorIterator with multiple inputs and outputs,
        the order they are accessed follows the order they are built within the iterator.
        For example, if an iterator is built in the following order:
        auto iter = TensorIteratorConfig().
          .add_output(firstOutput)
          .add_output(secondOutput)
          .add_input(firstInput)
          .add_input(secondInput)
          .build()
        data will contain 4 pointers to pointers to values in the following order:
        firstOutput, secondOutput, firstInput, secondInput.
        Proper pointer referencing and dereferencing, along with the usage of strides
        (to move onto different elements), can allow accessing of the input and assignment
        to the right output.
    */
    for (const auto i : c10::irange(n)) {
      float* dXOutput = (float*)(data[0] + i * strides[0]);
      float* dScaleOutput = (float*)(data[1] + i * strides[1]);
      float* dZeroPointOutput = (float*)(data[2] + i * strides[2]);
      float* XInput = (float*)(data[3] + i * strides[3]);
      float* dYInput = (float*)(data[4] + i * strides[4]);
      // Calculate gradients for X.
      int64_t xqi = std::nearbyint(zero_point + (*XInput) * inv_scale);
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      *dXOutput = (*dYInput) * (xqi >= quant_min && xqi <= quant_max);
      // Calculate gradients for scale and zero point.
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      float xfqi = static_cast<float>((std::clamp(xqi, quant_min, quant_max) - zero_point) * scale);
      // Calculate gradients according to the gradient of the clamp function.
      if (xqi < quant_min || xqi > quant_max) {
        *dZeroPointOutput = (*dYInput) * (-1) * scale * grad_factor;
        *dScaleOutput = ((xqi < quant_min) ? ((*dYInput) * dscale_small) : ((*dYInput) * dscale_big)) * grad_factor;
      } else {
        *dZeroPointOutput = 0;
        *dScaleOutput = (*dYInput) * (xfqi - (*XInput)) * inv_scale * grad_factor;
      }
    }
  });
}

template <typename SelfType>
void _fake_quant_per_channel_cachemask_cpu_helper(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    const int64_t quant_min,
    const int64_t quant_max) {

  const auto& zero_point_dtype = iter.input_dtype(2);

  if(at::isFloatingType(zero_point_dtype)){
    // When zero_point is float, quantize mirroring affine quantizer equation
    // Xq = Round(Xf * inv_scale + zero_point)
    // where zero_point is in float.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(zero_point_dtype, "fake_quantize_channel_cachemask_cpu_zero_point_handling", [&] {
      // write mask
      cpu_kernel(iter_mask, [=](SelfType self, float scale, scalar_t zero_point) -> bool {
        float inv_scale = 1.0f / scale;
        const auto qval = std::lrintf(zero_point + (self * inv_scale));
        return ((quant_min <= qval) && (qval <= quant_max));
      });

      // write fake_quant
      cpu_kernel(iter, [=](SelfType self, float scale, scalar_t zero_point) -> SelfType {
        float inv_scale = 1.0f / scale;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        return (std::fmin(
                    std::fmax(
                        std::lrintf(zero_point + self * inv_scale),
                        quant_min),
                    quant_max) -
                zero_point) *
            scale;
      });
    });

  } else {
      // write mask
      cpu_kernel(iter_mask, [=](SelfType self, float scale, int32_t zero_point) -> bool {
        float inv_scale = 1.0f / scale;
        const auto qval = static_cast<int64_t>(zero_point + std::nearbyint(self * inv_scale));
        return ((quant_min <= qval) && (qval <= quant_max));
      });

      // write fake_quant
      cpu_kernel(iter, [=](SelfType self, float scale, int32_t zero_point) -> SelfType {
        float inv_scale = 1.0f / scale;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        return (std::fmin(
                    std::fmax(
                        static_cast<int64_t>(
                            zero_point + std::nearbyint(self * inv_scale)),
                        quant_min),
                    quant_max) -
                zero_point) *
            scale;
      });
  }

}


void fake_quant_per_channel_cachemask_cpu(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    int64_t quant_min,
    int64_t quant_max) {
  // TODO(future, optional): read once, write twice.  Not done at the moment
  //   for simplicity, as we do not expect this to be a bottleneck.

  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "fake_quantize_channel_cachemask_cpu_type_handling", [&]() {
      _fake_quant_per_channel_cachemask_cpu_helper<scalar_t>(iter, iter_mask, quant_min, quant_max);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "fake_quantize_channel_cachemask_cpu_type_handling", [&] {
      _fake_quant_per_channel_cachemask_cpu_helper<scalar_t>(iter, iter_mask, quant_min, quant_max);
    });
  }
}


void fake_quantize_learnable_channel_grad_kernel_cpu(
    TensorIterator& iter,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor) {
  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    /*  To see how the input and outputs are referenced and assigned,
        please see the implementation of
        fake_quantize_learnable_tensor_grad_kernel_cpu.
    */
    for (const auto i : c10::irange(n)) {
      float* dx_output = (float*)(data[0] + i * strides[0]);
      float* dscale_output = (float*)(data[1] + i * strides[1]);
      float* dzero_point_output = (float*)(data[2] + i * strides[2]);
      float* x_input = (float*)(data[3] + i * strides[3]);
      float* dy_input = (float*)(data[4] + i * strides[4]);
      float* scale_input = (float*)(data[5] + i * strides[5]);
      float* zero_point_input = (float*)(data[6] + i * strides[6]);

      float inv_scale = 1.0f / (*scale_input);
      float dscale_small = quant_min - (*zero_point_input);
      float dscale_big = quant_max - (*zero_point_input);
      // Calculate gradients for X.
      int64_t xqi = std::nearbyint((*zero_point_input) + (*x_input) * inv_scale);
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      *dx_output = (*dy_input) * (xqi >= quant_min && xqi <= quant_max);
      // Calculate gradients for scale and zero point.
      float xfqi = ((std::clamp(xqi, quant_min, quant_max) - (*zero_point_input)) * (*scale_input));
      if (xqi < quant_min || xqi > quant_max) {
        *dzero_point_output = (*dy_input) * (-1) * (*scale_input) * grad_factor;
        *dscale_output = ((xqi < quant_min) ? ((*dy_input) * dscale_small) : ((*dy_input) * dscale_big)) * grad_factor;
      } else {
        *dzero_point_output = 0;
        *dscale_output = (*dy_input) * (xfqi - (*x_input)) * inv_scale * grad_factor;
      }
    }
  });
}

#ifdef USE_FBGEMM
void quantize_tensor_per_tensor_affine_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
        check_tensor_memory_format(rtensor, qtensor);
        const float* rd = rtensor.const_data_ptr<float>();
        auto qd = reinterpret_cast<underlying_t*>(qtensor.data_ptr<scalar_t>());
        fbgemm::TensorQuantizationParams qparams{};
        qparams.scale = scale;
        qparams.zero_point = zero_point;
        qparams.precision = CHAR_BIT * sizeof(underlying_t);
        int num_tasks = at::get_num_threads();
        at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
          for (const auto task_id : c10::irange(begin, end)) {
            fbgemm::Quantize<underlying_t>(
                rd, /*src=*/
                qd, /*dst=*/
                rtensor.numel(), /*len*/
                qparams, /*qparams=*/
                task_id, /*thread_id*/
                num_tasks /*num_threads*/);
          }
        });
      });
}

void dequantize_tensor_per_tensor_affine_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
        check_tensor_memory_format(qtensor, rtensor);
        const auto* qd =
            reinterpret_cast<const underlying_t*>(qtensor.const_data_ptr<scalar_t>());
        fbgemm::TensorQuantizationParams qparams{};
        qparams.scale = scale;
        qparams.zero_point = zero_point;
        qparams.precision = CHAR_BIT * sizeof(underlying_t);
        float* rd = rtensor.data_ptr<float>();
        int num_tasks = at::get_num_threads();
        at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
          for (const auto task_id : c10::irange(begin, end)) {
            fbgemm::Dequantize<underlying_t>(
                qd, /*src=*/
                rd, /*dst=*/
                qtensor.numel(), /*len=*/
                qparams, /*qparams=*/
                task_id, /*thread_id*/
                num_tasks /*num_threads*/);
          }
        });
      });
}
#else // USE_FBGEMM

#if defined(__ARM_NEON__) || defined(__aarch64__)

constexpr static int PARALLEL_THRESHOLD = 1 << 20;

// Generic template defaults to naive quantize implementation
template <typename T>
void quantize_tensor_arm(
    const float* __restrict__ in,
    T* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  for (const auto i : c10::irange(N)) {
    out[i] = at::native::quantize_val<T>(scale, zero_point, in[i]);
  }
}

namespace quantize_tensor_arm_intrinsics {
template <typename Tx8>
C10_ALWAYS_INLINE Tx8 vqmov(int16x8_t vraw);

template <>
C10_ALWAYS_INLINE uint8x8_t vqmov<uint8x8_t>(int16x8_t vraw) {
  return vqmovun_s16(vraw);
}

template <>
C10_ALWAYS_INLINE int8x8_t vqmov<int8x8_t>(int16x8_t vraw) {
  return vqmovn_s16(vraw);
}

template <typename T, typename Tx8>
C10_ALWAYS_INLINE void vst1(T* out, Tx8 vout);

template <>
C10_ALWAYS_INLINE void vst1<uint8_t, uint8x8_t>(uint8_t* out, uint8x8_t vout) {
  vst1_u8(out, vout);
}

template <>
C10_ALWAYS_INLINE void vst1<int8_t, int8x8_t>(int8_t* out, int8x8_t vout) {
  vst1_s8(out, vout);
}
} // namespace quantize_tensor_arm_intrinsics

// Specialized implementation from caffe2::Int8Quantize.
// There may be slight accuracy difference between this and implementation of
// quantize_val
// TODO Update quantize_tensor_arm implementation to follow quantize_val,
// i.e. f = Round(value/scale + zero_point)
// TODO Make quantize_tensor_arm work for int32 datatype too.
template <typename scalar_t, typename underlying_t, typename underlying_x8_t>
void quantize_tensor_arm_q8(
    const float* __restrict__ in,
    scalar_t* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  const float inv_scale = 1.0f / scale;
  uint32_t i = 0;
  underlying_t* out_underlying = reinterpret_cast<underlying_t*>(out);
  const float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
#if defined(__ARM_NEON__)
  // magic float and magic int to take care of rounding
  // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
  // Some detail:
  // 12582912.0f is 2**23 + 2**22. The trick is based on the fact that when you
  // add a small number to a large number, the result rounds to the precision of
  // the least significant bit of the large number. For IEEE-754
  // single-precision number mantissa has 23 bits, and adding 2**23 would cause
  // rounding to the nearest even integer. The we cast to int and subtract the
  // same number (0x4B400000 is the integer representation of 12582912.0f) to
  // get only the mantissa. This works if -2**22 < x < 2**22, but preserves the
  // sign for negative numbers.
  const int32x4_t voffset = vdupq_n_s32(zero_point - 0x4B400000);
  const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);
  for (i = 0; i + 8 <= N; i += 8) {
    const float32x4_t vin0123 = vld1q_f32(in);
    in += 4;
    const float32x4_t vin4567 = vld1q_f32(in);
    in += 4;
    const int32x4_t vraw0123 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
    const int32x4_t vraw4567 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));
    const int16x8_t vraw01234567 =
        vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
    const underlying_x8_t vout01234567 =
        quantize_tensor_arm_intrinsics::vqmov<underlying_x8_t>(vraw01234567);
    quantize_tensor_arm_intrinsics::vst1<underlying_t, underlying_x8_t>(
        out_underlying, vout01234567);
    out_underlying += 8;
  }
  for (; i < N; ++i) {
    (*out_underlying++) =
        at::native::quantize_val_arm<underlying_t>(scale, zero_point, (*in++));
  }
#else
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);
  for (i = 0; i + 8 <= N; i += 8) {
    const float32x4_t vin0123 = vld1q_f32(in);
    in += 4;
    const float32x4_t vin4567 = vld1q_f32(in);
    in += 4;
    const int32x4_t v0123_rounded = vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
    const int32x4_t v4567_rounded = vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));
    const int16x8_t v01234567_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded), vzero_point);
    const underlying_x8_t vout01234567 =
        quantize_tensor_arm_intrinsics::vqmov<underlying_x8_t>(
            v01234567_packed);
    quantize_tensor_arm_intrinsics::vst1<underlying_t, underlying_x8_t>(
        out_underlying, vout01234567);
    out_underlying += 8;
  }
  for (; i < N; ++i) {
    (*out_underlying++) =
        at::native::quantize_val_arm<underlying_t>(scale, zero_point, (*in++));
  }
#endif
}

template <>
void quantize_tensor_arm<c10::quint8>(
    const float* __restrict__ in,
    c10::quint8* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  quantize_tensor_arm_q8<c10::quint8, uint8_t, uint8x8_t>(
      in, out, N, scale, zero_point);
}

template <>
void quantize_tensor_arm<c10::qint8>(
    const float* __restrict__ in,
    c10::qint8* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  quantize_tensor_arm_q8<c10::qint8, int8_t, int8x8_t>(
      in, out, N, scale, zero_point);
}

#if defined(__aarch64__)
#define VMOVL_HIGH_U8(x) vmovl_high_u8(x)
#define VMOVL_HIGH_S8(x) vmovl_high_s8(x)
#define VMOVL_HIGH_U16(x) vmovl_high_u16(x)
#define VMOVL_HIGH_S16(x) vmovl_high_s16(x)
#else // vmovl_high intrinsic not supported
#define VMOVL_HIGH_U8(x) vmovl_u8(vget_high_u8(x))
#define VMOVL_HIGH_S8(x) vmovl_s8(vget_high_s8(x))
#define VMOVL_HIGH_U16(x) vmovl_u16(vget_high_u16(x))
#define VMOVL_HIGH_S16(x) vmovl_s16(vget_high_s16(x))
#endif

// Generic template defaults to naive dequantize implementation
template <typename T>
void dequantize_tensor_arm(
    const T* __restrict__ in,
    float* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  for (int i = 0; i < N; ++i) {
    out[i] = dequantize_val<T>(scale, zero_point, in[i]);
  }
}

template <>
void dequantize_tensor_arm<c10::qint8>(
    const c10::qint8* __restrict__ in,
    float* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  const int8_t* in_underlying = reinterpret_cast<const int8_t*>(in);

  const float32x4_t scale_fp32x4 = vdupq_n_f32(scale);
  // Zero point is restricted to be in bounds of a signed 8 bit integer
  const int8x8_t zero_point_s8x8 = vget_low_s8(vdupq_n_s8(static_cast<int8_t>(zero_point)));

  int i;
  for (i = 0; i + 16 <= N; i += 16) {
    const int8x16_t vin_s8 = vld1q_s8(in_underlying);

    // Extract upper or lower values to int16x8 and subtract zero point
    // Each input element and the zero point are restricted to be in bounds of
    // a signed 8 bit integer, so the difference will fit in a signed 16 bit
    // integer
    const int16x8_t minus_zp_low_s16 = vsubl_s8(vget_low_s8(vin_s8), zero_point_s8x8); // 0 ... 7
    const int16x8_t minus_zp_high_s16 = vsubl_s8(vget_high_s8(vin_s8), zero_point_s8x8); // 8 ... 15

    const int32x4_t minus_zp_low_low = vmovl_s16(vget_low_s16(minus_zp_low_s16)); // 0 ... 3
    const int32x4_t minus_zp_low_high = VMOVL_HIGH_S16(minus_zp_low_s16); // 4 ... 7
    const int32x4_t minus_zp_high_low = vmovl_s16(vget_low_s16(minus_zp_high_s16)); // 8 ... 11
    const int32x4_t minus_zp_high_high = VMOVL_HIGH_S16(minus_zp_high_s16); // 12 ... 15

    // Store            * scale   int32->fp32
    vst1q_f32(out,      vmulq_f32(vcvtq_f32_s32(minus_zp_low_low), scale_fp32x4));
    vst1q_f32(out + 4,  vmulq_f32(vcvtq_f32_s32(minus_zp_low_high), scale_fp32x4));
    vst1q_f32(out + 8,  vmulq_f32(vcvtq_f32_s32(minus_zp_high_low), scale_fp32x4));
    vst1q_f32(out + 12, vmulq_f32(vcvtq_f32_s32(minus_zp_high_high), scale_fp32x4));

    out += 16;
    in += 16;
    in_underlying += 16;
  }

  for (; i < N; ++i) { // use default dequantize for remaining vals
    (*out++) = dequantize_val<c10::qint8>(scale, zero_point, (*in++));
  }
}

template <>
void dequantize_tensor_arm<c10::quint8>(
    const c10::quint8* __restrict__ in,
    float* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  const uint8_t* in_underlying = reinterpret_cast<const uint8_t*>(in);

  const float32x4_t scale_fp32x4 = vdupq_n_f32(scale);
  // Zero point is restricted to be in bounds of an unsigned 8 bit integer
  const uint8x8_t zero_point_u8x8 = vget_low_u8(vdupq_n_u8(static_cast<uint8_t>(zero_point)));

  int i;
  for (i = 0; i + 16 <= N; i += 16) {
    const uint8x16_t vin_u8 = vld1q_u8(in_underlying);

    // Extract upper or lower values to uint16x8 and subtract zero point
    // Each input element and the zero point are restricted to be in bounds of
    // an unsigned 8 bit integer, so the difference will fit in a signed 16 bit
    // integer
    const int16x8_t minus_zp_low_s16 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vin_u8), zero_point_u8x8)); // 0 ... 7
    const int16x8_t minus_zp_high_s16 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vin_u8), zero_point_u8x8)); // 8 ... 15

    const int32x4_t minus_zp_low_low = vmovl_s16(vget_low_s16(minus_zp_low_s16)); // 0 ... 3
    const int32x4_t minus_zp_low_high = VMOVL_HIGH_S16(minus_zp_low_s16); // 4 ... 7
    const int32x4_t minus_zp_high_low = vmovl_s16(vget_low_s16(minus_zp_high_s16)); // 8 ... 11
    const int32x4_t minus_zp_high_high = VMOVL_HIGH_S16(minus_zp_high_s16); // 12 ... 15

    // Store            * scale   int32->fp32
    vst1q_f32(out,      vmulq_f32(vcvtq_f32_s32(minus_zp_low_low), scale_fp32x4));
    vst1q_f32(out + 4,  vmulq_f32(vcvtq_f32_s32(minus_zp_low_high), scale_fp32x4));
    vst1q_f32(out + 8,  vmulq_f32(vcvtq_f32_s32(minus_zp_high_low), scale_fp32x4));
    vst1q_f32(out + 12, vmulq_f32(vcvtq_f32_s32(minus_zp_high_high), scale_fp32x4));

    out += 16;
    in += 16;
    in_underlying += 16;
  }

  for (; i < N; ++i) { // use default dequantize for remaining vals
    (*out++) = dequantize_val<c10::quint8>(scale, zero_point, (*in++));
  }
}

#endif // defined(__ARM_NEON__) || defined(__aarch64__)

void quantize_tensor_per_tensor_affine_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  check_tensor_memory_format(rtensor, qtensor);
  const float* rdata = rtensor.const_data_ptr<float>();
  int numel = rtensor.numel();
#if defined(__ARM_NEON__) || defined(__aarch64__)
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
        scalar_t* qdata = qtensor.data_ptr<scalar_t>();
        auto quantize_range = [&](int64_t begin, int64_t end) {
          quantize_tensor_arm<scalar_t>(
            rdata + begin, qdata + begin, end - begin, scale, zero_point);
        };
        if (numel >= PARALLEL_THRESHOLD) {
          at::parallel_for(0, numel, 1, quantize_range);
        } else {
          quantize_range(0, numel);
        }
      });
#else
  // Fallback path
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
        scalar_t* qdata = qtensor.data_ptr<scalar_t>();
        for (const auto i : c10::irange(numel)) {
          qdata[i] = quantize_val<scalar_t>(scale, zero_point, rdata[i]);
        }
      });
#endif // defined(__ARM_NEON__) || defined(__aarch64__)
}

void dequantize_tensor_per_tensor_affine_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  check_tensor_memory_format(qtensor, rtensor);
  float* rdata = rtensor.data_ptr<float>();
  int numel = qtensor.numel();
#if defined(__ARM_NEON__) || defined(__aarch64__)
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
        const scalar_t* qdata = qtensor.const_data_ptr<scalar_t>();
        auto dequantize_range = [&](int64_t begin, int64_t end) {
          dequantize_tensor_arm<scalar_t>(
            qdata + begin, rdata + begin, end - begin, scale, zero_point);
        };
        if (numel >= PARALLEL_THRESHOLD) {
          at::parallel_for(0, numel, 1, dequantize_range);
        } else {
          dequantize_range(0, numel);
        }
      });
#else
  // Fallback path
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
        const scalar_t* qdata = qtensor.const_data_ptr<scalar_t>();
        for (const auto i : c10::irange(numel)) {
          rdata[i] = dequantize_val<scalar_t>(scale, zero_point, qdata[i]);
        }
      });
#endif // defined(__ARM_NEON__) || defined(__aarch64__)
}
#endif // USE_FBGEMM

// TODO: add fbgemm for per channel
// Generic template defaults to naive quantize implementation
template <typename T>
void quantize_tensor_per_channel_impl(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  // TODO: channels last kernel can be made faster.
  // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
  // For channels_last/3d however axis == 0 or 1.
  // Since current implementation on channels_last format does not
  // cover per channel quant with arbitrary axis value, it is better
  // to check and fail.
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
  int64_t channels = rtensor.size(axis);
  auto scales_data = scales.const_data_ptr<double>();
  auto zero_points_data = zero_points.const_data_ptr<int64_t>();
  const float* in = rtensor.const_data_ptr<float>();
  auto out = qtensor.data_ptr<T>();
  if (axis == 1 &&
      (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
       rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // This code handles per channel quant when axis = 1 and
    // channels_last contig.
    // If axis = 0 and channels_last contig, implementation for channels
    // first (NCHW) works.
    for (const auto b : c10::irange(batches)) {
      for (const auto e : c10::irange(elements_per_channel)) {
        for (const auto c : c10::irange(channels)) {
          auto i = b * channels * elements_per_channel + e * channels + c;
          out[i] = at::native::quantize_val<T>(
              scales_data[c], zero_points_data[c], in[i]);
        }
      }
    }
  } else {
    for (const auto b : c10::irange(batches)) {
      for (const auto c : c10::irange(channels)) {
        for (const auto e : c10::irange(elements_per_channel)) {
          auto i = b * channels * elements_per_channel +
              c * elements_per_channel + e;
          out[i] = at::native::quantize_val<T>(
              scales_data[c], zero_points_data[c], in[i]);
        }
      }
    }
  }
}

#if defined(__ARM_NEON__) || defined(__aarch64__)
// Specialized implementation from caffe2::Int8Quantize.
// There may be slight accuracy difference between this and implementation of
// quantize_val
// TODO Update quantize_tensor_per_channel_impl implementation to follow
// quantize_val, i.e. f = Round(value/scale + zero_point)
// TODO Make quantize_tensor_per_channel_impl work for other datatypes too
// (int8, int32).
template <>
void quantize_tensor_per_channel_impl<c10::quint8>(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
  int64_t channels = rtensor.size(axis);
  auto scales_data = scales.const_data_ptr<double>();
  auto zero_points_data = zero_points.const_data_ptr<int64_t>();
  const float* in = rtensor.const_data_ptr<float>();
  auto out = (uint8_t*)qtensor.data_ptr<c10::quint8>();
#if defined(__ARM_NEON__)
  // magic float and magic int to take care of rounding
  // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
  // Some detail:
  // 12582912.0f is 2**23 + 2**22. The trick is based on the fact that when you
  // add a small number to a large number, the result rounds to the precision of
  // the least significant bit of the large number. For IEEE-754
  // single-precision number mantissa has 23 bits, and adding 2**23 would cause
  // rounding to the nearest even integer. The we cast to int and subtract the
  // same number (0x4B400000 is the integer representation of 12582912.0f) to
  // get only the mantissa. This works if -2**22 < x < 2**22, but preserves the
  // sign for negative numbers.
  const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);
  // Copy reciprocal of scales (double) into float array
  // Copy zero_points with magic int (int64_t) into int32_t array
  std::vector<float> inv_scales(channels);
  std::vector<int32_t> zero_points_int32t(channels);
  for (const auto i : c10::irange(channels)) {
    inv_scales[i] = 1.0f / (float)scales_data[i];
    zero_points_int32t[i] = (int32_t)(uint32_t)zero_points_data[i] - 0x4B400000;
  }
  if (axis == 1 &&
      (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
       rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // This code handles per channel quant when axis = 1 and
    // channels_last contig.
    // If axis = 0 and channels_last contig, implementation for channels
    // first (NCHW) works.
    for ([[maybe_unused]] const auto b : c10::irange(batches)) {
      for ([[maybe_unused]] const auto e : c10::irange(elements_per_channel)) {
        uint32_t c = 0;
        while (c + 8 < channels) {
          const int32x4_t voffset0123 = vld1q_s32(&zero_points_int32t[c]);
          const float32x4_t vinv_scale0123 = vld1q_f32(&inv_scales[c]);
          c += 4;
          const int32x4_t voffset4567 = vld1q_s32(&zero_points_int32t[c]);
          const float32x4_t vinv_scale4567 = vld1q_f32(&inv_scales[c]);
          c += 4;
          const float32x4_t vin0123 = vld1q_f32(in);
          in += 4;
          const float32x4_t vin4567 = vld1q_f32(in);
          in += 4;
          const int32x4_t vraw0123 = vaddq_s32(
              voffset0123,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale0123))));
          const int32x4_t vraw4567 = vaddq_s32(
              voffset4567,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale4567))));
          const int16x8_t vraw01234567 =
              vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
          const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
          vst1_u8(out, vout01234567);
          out += 8;
        }
        for (; c < channels; ++c) {
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  } else {
    for ([[maybe_unused]] const auto b : c10::irange(batches)) {
      for (const auto c : c10::irange(channels)) {
        uint32_t e = 0;
        const int32x4_t voffset = vdupq_n_s32(zero_points_int32t[c]);
        const float32x4_t vinv_scale = vdupq_n_f32(inv_scales[c]);
        for (; e + 8 < elements_per_channel; e += 8) {
          const float32x4_t vin0123 = vld1q_f32(in);
          in += 4;
          const float32x4_t vin4567 = vld1q_f32(in);
          in += 4;
          const int32x4_t vraw0123 = vaddq_s32(
              voffset,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
          const int32x4_t vraw4567 = vaddq_s32(
              voffset,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));
          const int16x8_t vraw01234567 =
              vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
          const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
          vst1_u8(out, vout01234567);
          out += 8;
        }
        for (; e < elements_per_channel; ++e) {
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  }
#else // defined(__ARM_NEON__)
  // Copy scales (double) into float array
  // Copy zero_points (int64_t) into int16_t array
  std::vector<float> inv_scales(channels);
  std::vector<int16_t> zero_points_int16t(channels);
  for (const auto i : c10::irange(channels)) {
    inv_scales[i] = 1.0f / (float)scales_data[i];
    zero_points_int16t[i] = (int16_t)(uint16_t)zero_points_data[i];
  }
  if (axis == 1 &&
      (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
       rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // This code handles per channel quant when axis = 1 and
    // channels_last contig.
    // If axis = 0 and channels_last contig, implementation for channels
    // first (NCHW) works.
    for ([[maybe_unused]] const auto b : c10::irange(batches)) {
      for ([[maybe_unused]] const auto e : c10::irange(elements_per_channel)) {
        uint32_t c = 0;
        while (c + 8 < channels) {
          const int16x8_t vzero_point = vld1q_s16(&zero_points_int16t[c]);
          const float32x4_t vinv_scale0123 = vld1q_f32(&inv_scales[c]);
          c += 4;
          const float32x4_t vinv_scale4567 = vld1q_f32(&inv_scales[c]);
          c += 4;
          const float32x4_t vin0123 = vld1q_f32(in);
          in += 4;
          const float32x4_t vin4567 = vld1q_f32(in);
          in += 4;
          const int32x4_t v0123_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale0123));
          const int32x4_t v4567_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale4567));
          const int16x8_t v01234567_packed = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded),
              vzero_point);
          const uint8x8_t vout01234567 = vqmovun_s16(v01234567_packed);
          vst1_u8(out, vout01234567);
          out += 8;
        }
        for (; c < channels; ++c) {
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  } else {
    for ([[maybe_unused]] const auto b : c10::irange(batches)) {
      for ([[maybe_unused]] const auto c : c10::irange(channels)) {
        uint32_t e = 0;
        const int16x8_t vzero_point = vdupq_n_s16(zero_points_int16t[c]);
        const float32x4_t vinv_scale = vdupq_n_f32(inv_scales[c]);
        for (; e + 8 < elements_per_channel; e += 8) {
          const float32x4_t vin0123 = vld1q_f32(in);
          in += 4;
          const float32x4_t vin4567 = vld1q_f32(in);
          in += 4;
          const int32x4_t v0123_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
          const int32x4_t v4567_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));
          const int16x8_t v01234567_packed = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded),
              vzero_point);
          const uint8x8_t vout01234567 = vqmovun_s16(v01234567_packed);
          vst1_u8(out, vout01234567);
          out += 8;
        }
        for (; e < elements_per_channel; ++e) {
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  }
#endif // defined(__ARM_NEON__)
}
#endif // defined(__ARM_NEON__) || defined(__aarch64__)

void quantize_tensor_per_channel_affine_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  TORCH_CHECK(
      rtensor.is_contiguous() || (axis <= 1),
      "If tensor is channels_last contig then per channel quantization "
      "is supported only for axis = 0 or 1.");
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_channel_affine_cpu", [&]() {
        check_tensor_memory_format(rtensor, qtensor);
        quantize_tensor_per_channel_impl<scalar_t>(
            rtensor, qtensor, scales, zero_points, axis);
      });
}

template<typename T, typename N, typename Q>
void dequantize_per_channel_affine_kernel(
      const Tensor& qtensor,
      Tensor& rtensor,
      const Tensor& scales,
      const Tensor& zero_points,
      int64_t axis,
      int bit_width=8) {

  // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
  // For channels_last/3d however axis == 0 or 1.
  // Since current implementation on channels_last format does not
  // cover per channel quant with arbitrary axis value, it is better
  // to check and fail.
  TORCH_CHECK(rtensor.is_contiguous() || (axis <=1),
      "If tensor is channels_last contig then per channel quantization "
      "is supported only for axis = 0 or 1.");
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  int64_t elements_per_channel =
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      size_from_dim_(axis + 1, rtensor.sizes());
  int64_t channel = rtensor.size(axis);
  auto scales_data = scales.const_data_ptr<T>();
  auto zero_points_data = zero_points.const_data_ptr<N>();
  check_tensor_memory_format(qtensor, rtensor);
  const auto* qd = qtensor.const_data_ptr<Q>();
  float* rd = rtensor.data_ptr<float>();
  const auto elem_per_byte = 8 / bit_width;
  if (axis == 1 && (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
      rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    for (const auto b : c10::irange(batches)) {
      for (const auto e : c10::irange(elements_per_channel)) {
        for (const auto c : c10::irange(channel)) {
          auto i = b * channel * elements_per_channel + e * channel + c;
          // We need to convert the qint8 value to float to ensure the
          // subtraction subexpression returns a float
          auto qvalue = qd[i / elem_per_byte].val_;
          if (bit_width < 8) {
            qvalue >>= (i % elem_per_byte) * bit_width;
            qvalue &= (1 << bit_width) - 1;
          }
          rd[i] = (static_cast<float>(qvalue) - zero_points_data[c]) * scales_data[c];
        }
      }
    }
  } else {
    for (const auto b : c10::irange(batches)) {
      for (const auto c : c10::irange(channel)) {
        for (const auto e : c10::irange(elements_per_channel)) {
          auto i = b * channel * elements_per_channel +
              c * elements_per_channel + e;
          // We need to convert the qint8 value to float to ensure the
          // subtraction subexpression returns a float
          auto qvalue = qd[i / elem_per_byte].val_;
          if (bit_width < 8) {
            qvalue >>= (i % elem_per_byte) * bit_width;
            qvalue &= (1 << bit_width) - 1;
          }
          rd[i] = (static_cast<float>(qvalue) - zero_points_data[c]) * scales_data[c];
        }
      }
    }
  }
}

void dequantize_tensor_per_channel_affine_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_channel_affine_cpu", [&]() {
        dequantize_per_channel_affine_kernel<double, int64_t, scalar_t>(qtensor, rtensor, scales, zero_points, axis);
      });
}

// quantize stubs for floating point scale and zero_point.
void quantize_tensor_per_channel_float_qparams_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
  // For channels_last/3d however axis == 0 or 1.
  // Since current implementation on channels_last format does not
  // cover per channel quant with arbitrary axis value, it is better
  // to check and fail.
  TORCH_CHECK(rtensor.is_contiguous() || (axis <=1),
      "If tensor is channels_last contig then per channel quantization "
      "is supported only for axis = 0 or 1.");
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_channel_float_qparams_cpu", [&]() {
        int64_t batches = size_to_dim_(axis, rtensor.sizes());
        int64_t elements_per_channel =
            size_from_dim_(axis + 1, rtensor.sizes());
        int64_t channel = rtensor.size(axis);
        auto scales_data = scales.const_data_ptr<float>();
        auto zero_points_data = zero_points.const_data_ptr<float>();
        check_tensor_memory_format(rtensor, qtensor);
        const float* rdata = rtensor.const_data_ptr<float>();
        auto qdata = reinterpret_cast<underlying_t*>(qtensor.data_ptr<scalar_t>());
        const auto elem_per_byte = bit_width < CHAR_BIT ? CHAR_BIT / bit_width : 1;
        int qvalue = 0;
        if (axis == 1 && (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
            rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
          for (const auto b : c10::irange(batches)) {
            for (const auto e : c10::irange(elements_per_channel)) {
              for (const auto c : c10::irange(channel)) {
                auto i = b * channel * elements_per_channel + e * channel + c;
                qvalue = quantize_val_float_qparams(
                    scales_data[c], zero_points_data[c], rdata[i], quant_min, quant_max);
                if (i % elem_per_byte == 0) {
                  qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
                } else {
                  qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
                }
              }
            }
          }
        } else {
          for (const auto b : c10::irange(batches)) {
            for (const auto c : c10::irange(channel)) {
              for (const auto e : c10::irange(elements_per_channel)) {
                auto i = b * channel * elements_per_channel +
                    c * elements_per_channel + e;
                qvalue = quantize_val_float_qparams(
                    scales_data[c], zero_points_data[c], rdata[i], quant_min, quant_max);
                if (i % elem_per_byte == 0) {
                  qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
                } else {
                  qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
                }
              }
            }
          }
        }
      });
}

void dequantize_tensor_per_channel_float_qparams_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_channel_float_qparams_cpu", [&]() {
        dequantize_per_channel_affine_kernel<float, float, scalar_t>(qtensor, rtensor, scales, zero_points, axis, bit_width);
      });
}

void quantize_tensor_per_tensor_affine_sub_byte_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    float scale,
    float zero_point) {
  // TODO Use fbgemm kernel to pack values
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
    qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_sub_byte_cpu", [&]() {
      check_tensor_memory_format(rtensor, qtensor);
      const float* const rdata = rtensor.const_data_ptr<float>();
      auto qdata = reinterpret_cast<underlying_t*>(qtensor.data_ptr<scalar_t>());
      auto numel = rtensor.numel();
      const auto elem_per_byte = bit_width < CHAR_BIT ? CHAR_BIT / bit_width : 1;
      for (const auto i : c10::irange(numel)) {
        float inv_scale = scale == 0 ? 1.0f : 1.0f / scale;
        int64_t qvalue = lrintf(std::nearbyint(rdata[i] * inv_scale) + zero_point);
        qvalue = std::max(quant_min, std::min(qvalue, quant_max));

        // We pack sub_byte values and align them to a byte.
        // Eg. for 4-bits Index 0 is packed in the lower 4-bits
        // and index 1 is packed in the upper 4-bits.
        if (i % elem_per_byte == 0) {
          qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
        } else {
          qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
        }
      } // for numel
    });
}

void dequantize_tensor_per_tensor_affine_sub_byte_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    float scale,
    float zero_point) {
  // TODO Use fbgemm kernel to pack values
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
    qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_sub_byte_cpu", [&]() {
      check_tensor_memory_format(rtensor, qtensor);
      auto rdata = rtensor.data_ptr<float>();
      const underlying_t* qdata = reinterpret_cast<const underlying_t*>(qtensor.const_data_ptr<scalar_t>());
      auto numel = rtensor.numel();
      const auto elem_per_byte = bit_width < CHAR_BIT ? CHAR_BIT / bit_width : 1;

      for (const auto i : c10::irange(numel)) {
        underlying_t qvalue = qdata[i / elem_per_byte];
        qvalue >>= (i % elem_per_byte) * bit_width;
        qvalue &= (1 << bit_width) - 1;
        rdata[i] = (static_cast<float>(qvalue) - zero_point) * scale;
      }
  });
}

} // anonymous namespace

// These kernels are dispatched to AVX512
ALSO_REGISTER_AVX512_DISPATCH(dequantize_tensor_per_channel_affine_stub,
                  &dequantize_tensor_per_channel_affine_cpu)
ALSO_REGISTER_AVX512_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub,
                  &dequantize_tensor_per_channel_float_qparams_cpu)
ALSO_REGISTER_AVX512_DISPATCH(fake_quant_per_channel_cachemask_stub,
                  &fake_quant_per_channel_cachemask_cpu)

// The kernels below are dispatched to AVX2 because they don't perform as well
// with AVX512. We might revisit this decision in the near future.
REGISTER_DISPATCH(dequantize_tensor_per_tensor_affine_stub,
                  &dequantize_tensor_per_tensor_affine_cpu)
REGISTER_DISPATCH(fake_quant_grad_learnable_tensor_stub,
                  &fake_quantize_learnable_tensor_grad_kernel_cpu)
REGISTER_DISPATCH(fake_quant_tensor_cachemask_stub,
                  &fake_quantize_tensor_cachemask_kernel)
REGISTER_DISPATCH(fake_quant_tensor_cachemask_tensor_qparams_stub,
                  &fake_quantize_tensor_cachemask_tensor_qparams_kernel)
REGISTER_DISPATCH(fake_quant_grad_learnable_channel_stub,
                  &fake_quantize_learnable_channel_grad_kernel_cpu)
REGISTER_DISPATCH(
    quantize_tensor_per_tensor_affine_stub,
    &quantize_tensor_per_tensor_affine_cpu)
REGISTER_DISPATCH(
    quantize_tensor_per_channel_affine_stub,
    &quantize_tensor_per_channel_affine_cpu)
REGISTER_DISPATCH(
    quantize_tensor_per_channel_float_qparams_stub,
    &quantize_tensor_per_channel_float_qparams_cpu)
REGISTER_DISPATCH(
    quantize_tensor_per_tensor_affine_sub_byte_stub,
    &quantize_tensor_per_tensor_affine_sub_byte_cpu)
REGISTER_DISPATCH(
    dequantize_tensor_per_tensor_affine_sub_byte_stub,
    &dequantize_tensor_per_tensor_affine_sub_byte_cpu)
} // namespace at::native
// NOLINTEND(*-c-arrays)

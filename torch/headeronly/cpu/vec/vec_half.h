#pragma once

#include <torch/headeronly/cpu/vec/intrinsics.h>
#include <torch/headeronly/macros/Macros.h>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, vec)
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
    !defined(__APPLE__)
static inline uint16_t float2half_scalar(float val) {
#if defined(CPU_CAPABILITY_AVX2)
  return _cvtss_sh(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#elif defined(CPU_CAPABILITY_AVX512)
  __m512 v = _mm512_set1_ps(val);
  __m256i o =
      _mm512_cvtps_ph(v, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  return static_cast<std::uint16_t>(
      _mm_cvtsi128_si32(_mm256_castsi256_si128(o)));
#endif
}

static inline float half2float_scalar(uint16_t val) {
#if defined(CPU_CAPABILITY_AVX2)
  return _cvtsh_ss(val);
#elif defined(CPU_CAPABILITY_AVX512)
  __m256i v =
      _mm256_setr_epi16(val, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  __m512 o = _mm512_cvtph_ps(v);
  return _mm512_cvtss_f32(o);
#endif
}

#endif

} // namespace CPU_CAPABILITY
HIDDEN_NAMESPACE_END(torch, headeronly, vec)

namespace at::vec {
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
    !defined(__APPLE__)
using torch::headeronly::vec::float2half_scalar;
using torch::headeronly::vec::half2float_scalar;
#endif
} // namespace at::vec

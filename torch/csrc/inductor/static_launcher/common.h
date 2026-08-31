#pragma once

#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/python_headers.h>
#include <torch/headeronly/util/BFloat16.h>
#include <cstdint>

namespace torch::inductor::static_launcher {

// Match Triton's generated launchers: fp16 packs directly from the Python
// double, while bf16 narrows to fp32 and keeps its high 16 bits, which is what
// c10::detail::bits_from_f32 does.
inline uint16_t unpackTritonFp16(PyObject* obj) {
  uint16_t bits = 0;
  TORCH_CHECK_PYTHON(
      PyFloat_Pack2(
          THPUtils_unpackDouble(obj), reinterpret_cast<char*>(&bits), 1) >= 0);
  return bits;
}

inline uint16_t unpackTritonBf16(PyObject* obj) {
  float value = static_cast<float>(THPUtils_unpackDouble(obj));
  return c10::detail::bits_from_f32(value);
}

} // namespace torch::inductor::static_launcher

#pragma once

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <torch/csrc/Export.h>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace torch::utils {

// A stream written little-endian is read back unchanged on a little-endian
// host and byte-swapped on a big-endian one; the reverse direction is the
// same operation, so one function covers both.
template <typename T>
constexpr T swapLittleEndian(T value) {
  static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big,
      "mixed-endian platforms are not supported");
  if constexpr (std::endian::native == std::endian::little) {
    return value;
  } else {
    return std::byteswap(value);
  }
}

enum THPByteOrder { THP_LITTLE_ENDIAN = 0, THP_BIG_ENDIAN = 1 };

TORCH_API THPByteOrder THP_nativeByteOrder();

template <typename T, typename U>
TORCH_API void THP_decodeBuffer(T* dst, const uint8_t* src, U type, size_t len);

template <typename T>
TORCH_API void THP_encodeBuffer(
    uint8_t* dst,
    const T* src,
    THPByteOrder order,
    size_t len);

} // namespace torch::utils

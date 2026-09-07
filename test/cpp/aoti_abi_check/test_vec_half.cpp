#include <bit>

#include <gtest/gtest.h>
#include <torch/headeronly/cpu/vec/vec_half.h>
#include <torch/headeronly/util/Half.h>

TEST(TestVecHalf, TestConversion) {
  float f32s[100];
  for (int i = 0; i < 100; i++) {
    f32s[i] = static_cast<float>(i + 0.3);
  }
  for (int i = 0; i < 100; i++) {
    const auto half = torch::headeronly::Half(f32s[i]);
    const auto u16 = std::bit_cast<uint16_t>(half);
    const auto x =
        static_cast<float>(std::bit_cast<torch::headeronly::Half>(u16));
    EXPECT_EQ(
        u16, torch::headeronly::detail::fp16_ieee_from_fp32_value(f32s[i]))
        << "Test failed for float to uint16 " << f32s[i] << '\n';
    EXPECT_EQ(x, torch::headeronly::detail::fp16_ieee_to_fp32_value(u16))
        << "Test failed for uint16 to float " << u16 << '\n';
#if (defined(__F16C__) || defined(__AVX512F__)) && !defined(__APPLE__)
    EXPECT_EQ(torch::headeronly::vec::float2half_scalar(f32s[i]), u16);
    EXPECT_EQ(torch::headeronly::vec::half2float_scalar(u16), x);
#endif
  }
}

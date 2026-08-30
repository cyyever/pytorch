#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

using namespace at;
using namespace at::native;

TEST(CUDNNTest, CUDNNTestCUDA) {
  if (!at::cuda::is_available()) return;
  manual_seed(123);
}

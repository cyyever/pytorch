#pragma once

// Device code builds at C++20 while the host is at C++26, and this header is
// reachable from .cu, so the floor asserted here is the lower of the two.
#if __cplusplus < 202002L
#error C++20 or later compatible compiler is required to use PyTorch.
#endif

#include <torch/autograd.h>
#include <torch/cuda.h>
#include <torch/fft.h>
#include <torch/mps.h>
#include <torch/nested.h>
#include <torch/print.h>
#include <torch/special.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <torch/version.h>
#include <torch/xpu.h>

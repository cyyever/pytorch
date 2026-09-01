#pragma once

#if __cplusplus < 202002L
#error C++20 or later compatible compiler is required to use PyTorch.
#endif

#include <torch/autograd.h>
#include <torch/cuda.h>
#include <torch/enum.h>
#include <torch/fft.h>
#include <torch/mps.h>
#include <torch/nested.h>
#include <torch/print.h>
#include <torch/sparse.h>
#include <torch/special.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <torch/version.h>
#include <torch/xpu.h>

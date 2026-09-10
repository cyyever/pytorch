#pragma once

#include <torch/headeronly/util/Half.h>

// need to keep the following for BC because the APIs in here were exposed
// before migrating Half to torch/headeronly
#if (defined(__F16C__) || defined(__AVX512F__)) && !defined(__APPLE__)
#include <ATen/cpu/vec/vec_half.h>
#endif

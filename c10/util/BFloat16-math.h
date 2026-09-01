#pragma once

#include <torch/headeronly/util/BFloat16-math.h>

// Kept while the rest of the pure forwarders go: vendored submodules include
// this path and cannot be told about torch/headeronly. gloo (under
// GLOO_USE_TORCH_DTYPES), fbgemm, mslk, torch-xpu-ops and aiter all do.

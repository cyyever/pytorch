#pragma once

#include <c10/macros/Export.h>

#ifdef _MSC_VER
#define JIT_TEST_API
#else
#define JIT_TEST_API TORCH_API
#endif

namespace torch::jit {
JIT_TEST_API void runJITCPPTests();
} // namespace torch::jit

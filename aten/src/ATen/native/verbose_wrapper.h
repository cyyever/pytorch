#pragma once

#include <c10/macros/Export.h>

namespace torch::verbose {
TORCH_API int _mkl_set_verbose(int enable);
} // namespace torch::verbose

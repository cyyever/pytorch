#ifndef PYTHON_COMPAT
#define PYTHON_COMPAT

#include <torch/csrc/utils/pythoncapi_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

// PyTorch-only compat functions

#define IS_PYTHON_3_15_PLUS (PY_VERSION_HEX >= 0x030F0000)
#define IS_PYTHON_3_16_PLUS (PY_VERSION_HEX >= 0x03100000)

#ifdef __cplusplus
}
#endif
#endif // PYTHON_COMPAT

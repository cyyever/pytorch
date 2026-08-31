#ifndef PYTHON_COMPAT
#define PYTHON_COMPAT

// Included from cpython_defs.c, so this header must stay valid C:
// python_headers.h pulls in <cmath> and <complex>.
#include <Python.h>

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

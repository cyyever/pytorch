#pragma once

#include <torch/csrc/utils/python_compat.h>

// Functions that need to be copied from the CPython source
// should go in cpython_defs.c. Copying is required when, e.g.,
// we need to call internal CPython functions that are not exposed.

typedef struct _PyInterpreterFrame _PyInterpreterFrame;

PyFunctionObject* _PyFunction_CopyWithNewCode(
    PyFunctionObject* o,
    PyCodeObject* code);

#if IS_PYTHON_3_15_PLUS
#define THP_PyThreadState_BumpFramePointerSlow \
  _PyThreadState_PushFrame // renamed and publicly exported in 3.15
#else
_PyInterpreterFrame* THP_PyThreadState_BumpFramePointerSlow(
    PyThreadState* tstate,
    size_t size);
#endif

// pointers to _PyOpcode_Caches for C++
#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t* THP_PyOpcode_Caches;
extern int THP_PyOpcode_Caches_size;
void init_THPCaches();

#ifdef __cplusplus
} // extern "C"
#endif

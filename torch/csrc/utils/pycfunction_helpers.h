#pragma once

#include <c10/macros/Macros.h>
#include <torch/csrc/utils/python_compat.h>

#include <Python.h>

// Casting a keyword or fastcall function to PyCFunction is how CPython's own
// method tables are populated -- PyMethodDef stores one function type and the
// flags say which signature it really has. The cast is deliberate, so silence
// the warning here rather than at each of the hundreds of call sites.
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wcast-function-type-mismatch")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wcast-function-type-mismatch")
#endif

inline PyCFunction castPyCFunctionWithKeywords(PyCFunctionWithKeywords func) {
  return reinterpret_cast<PyCFunction>(func);
}

inline PyCFunction castPyCFunctionFast(PyCFunctionFast func) {
  return reinterpret_cast<PyCFunction>(func);
}

C10_CLANG_DIAGNOSTIC_POP()

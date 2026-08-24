#pragma once

#include <torch/csrc/utils/python_compat.h>

// Problem in CPython includes when mixing core and non-core build
// https://github.com/python/cpython/issues/105268
#undef _PyGC_FINALIZED

// see https://bugs.python.org/issue35886
#define Py_BUILD_CORE
// avoid dependency on _Py_tss_tstate
#define Py_BUILD_CORE_MODULE

#ifndef __cplusplus
// C-only headers
#include <internal/pycore_pystate.h>

#endif // __cplusplus

#include <internal/pycore_ceval.h>

#include <internal/pycore_frame.h>

#include <torch/csrc/dynamo/stackref_bridge.h>
#include <internal/pycore_code.h>
#include <internal/pycore_genobject.h>
#include <internal/pycore_interpframe.h>
#include <internal/pycore_stackref.h>

#undef Py_BUILD_CORE
#undef Py_BUILD_CORE_MODULE

#ifdef __cplusplus
extern "C" {
#endif

#define PREV_INSTR(x) (x)->instr_ptr

// 3.14 stores f_executable/f_funcobj as stackrefs rather than PyObject*.
#define F_CODE(x) \
  ((PyCodeObject*)THP_PyStackRef_AsPyObjectBorrow(&(x)->f_executable))
#define FUNC(x) \
  ((PyFunctionObject*)THP_PyStackRef_AsPyObjectBorrow(&(x)->f_funcobj))

#ifdef __cplusplus
} // extern "C"
#endif

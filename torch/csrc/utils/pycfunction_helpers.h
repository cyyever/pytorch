#pragma once

#include <c10/macros/Macros.h>
#include <torch/csrc/utils/python_compat.h>

#include <Python.h>

inline PyCFunction castPyCFunctionWithKeywords(PyCFunctionWithKeywords func) {
  return reinterpret_cast<PyCFunction>(func);
}

inline PyCFunction castPyCFunctionFast(PyCFunctionFast func) {
  return reinterpret_cast<PyCFunction>(func);
}

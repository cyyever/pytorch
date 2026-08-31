#pragma once

#include <torch/csrc/python_headers.h>

#if PY_VERSION_HEX < 0x030F00A7
extern "C" PyAPI_FUNC(void) _Py_SetImmortal(PyObject* op);
#endif

namespace torch::utils {

// Set a Python object as immortal, i.e. living for the same lifetime as the
// entire runtime.
//
// Reference counting is expensive on the free-threaded build when threads
// incref/decref objects that they don't own, there's an atomic
// read-modify-write that leads to contended cache lines.  This particularly
// hurts common shared objects like singletons.
//
// Return true if the object was successfully immortalized.
inline bool set_immortal_if_possible(PyObject* obj) {
#if PY_VERSION_HEX >= 0x030F00A7
  return PyUnstable_SetImmortal(obj) > 0;
#else
  // PyUnstable_SetImmortal arrives in 3.15. On the 3.14 floor this fork
  // targets, the two pieces it is built from are already available:
  // PyUnstable_Object_IsUniquelyReferenced is public, and _Py_SetImmortal is
  // exported, so declare it here rather than reaching into pycore_object.h.
  if (!PyUnstable_Object_IsUniquelyReferenced(obj) || PyUnicode_Check(obj)) {
    return false;
  }
  _Py_SetImmortal(obj);
  return true;
#endif
}

} // namespace torch::utils

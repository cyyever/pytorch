#include <Python.h>

#define OPENREG_EXPORT __attribute__((visibility("default")))

extern OPENREG_EXPORT PyObject* initOpenRegModule(void);

#ifdef __cplusplus
extern "C"
#endif

    OPENREG_EXPORT PyObject*
    PyInit__C(void);

PyMODINIT_FUNC PyInit__C(void) {
  return initOpenRegModule();
}

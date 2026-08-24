#include <Python.h>

extern PyObject* initModule(void);

#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);

PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}

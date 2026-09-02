#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_nested_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

$ops_headers

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::OptionalIntArrayRef;
using at::Generator;
using at::TensorList;

using namespace torch::autograd::utils;

namespace torch::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef nested_functions[] = {
  {NULL, NULL, 0, NULL},
  ${py_method_defs}
  {}
};

static PyObject* THPNestedVariableFunctionsModule = NULL;

void initNestedFunctions(PyObject* module) {
  nested_functions[0] = get_nested_functions_manual()[0];
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._nested",
     NULL,
     -1,
     nested_functions,
     nullptr, /* m_slots */
     nullptr, /* m_traverse */
     nullptr, /* m_clear */
     nullptr  /* m_free */
  };
  PyObject* nested = PyModule_Create(&def);
  THPNestedVariableFunctionsModule = nested;
  TORCH_CHECK_PYTHON(nested);
  // steals a reference to nested
  TORCH_CHECK_PYTHON(PyModule_AddObject(module, "_nested", nested) == 0);
}

// generated methods start here

${py_methods}

} // namespace torch::autograd

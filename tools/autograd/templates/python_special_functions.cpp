#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_special_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/out_types.h"
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
using at::Generator;
using at::TensorList;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

namespace torch::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef special_functions[] = {
  ${py_method_defs}
  {}
};

static PyObject* THPSpecialVariableFunctionsModule = NULL;

void initSpecialFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._special",
     NULL,
     -1,
     special_functions,
     nullptr, /* m_slots */
     nullptr, /* m_traverse */
     nullptr, /* m_clear */
     nullptr  /* m_free */
  };
  PyObject* special = PyModule_Create(&def);
  THPSpecialVariableFunctionsModule = special;
  TORCH_CHECK_PYTHON(special);
  // steals a reference to special
  TORCH_CHECK_PYTHON(PyModule_AddObject(module, "_special", special) == 0);
}

// generated methods start here

${py_methods}

} // namespace torch::autograd

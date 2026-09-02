#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_sparse_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

$ops_headers

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;
using at::TensorList;

using namespace torch::autograd::utils;

namespace torch::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef sparse_functions[] = {
  ${py_method_defs}
  {}
};

static PyObject* THPSparseVariableFunctionsModule = NULL;

void initSparseFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._sparse",
     NULL,
     -1,
     sparse_functions,
     nullptr, /* m_slots */
     nullptr, /* m_traverse */
     nullptr, /* m_clear */
     nullptr  /* m_free */
  };
  PyObject* sparse = PyModule_Create(&def);
  THPSparseVariableFunctionsModule = sparse;
  TORCH_CHECK_PYTHON(sparse);
  // steals a reference to sparse
  TORCH_CHECK_PYTHON(PyModule_AddObject(module, "_sparse", sparse) == 0);
}

// generated methods start here

${py_methods}

} // namespace torch::autograd

#pragma once
#include <torch/csrc/python_headers.h>
namespace torch::autograd {

void initSpecialFunctions(PyObject* module);

}

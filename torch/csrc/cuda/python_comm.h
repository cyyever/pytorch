#pragma once

#include <torch/csrc/python_headers.h>
namespace torch::cuda::python {

void initCommMethods(PyObject* module);

} // namespace torch::cuda::python

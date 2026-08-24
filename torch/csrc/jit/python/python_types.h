#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::jit {

void initPythonTypeBindings(PyObject* module);

} // namespace torch::jit

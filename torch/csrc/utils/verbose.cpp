#include <ATen/native/verbose_wrapper.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/verbose.h>

namespace torch {

void initVerboseBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto verbose = m.def_submodule("_verbose", "MKL verbose");
  verbose.def("mkl_set_verbose", torch::verbose::_mkl_set_verbose);
}

} // namespace torch

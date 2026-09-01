#pragma once

#include <torch/types.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_tuples.h>


namespace torch::python {
namespace detail {
inline Device py_object_to_device(py::object object) {
  PyObject* obj = object.ptr();
  if (THPDevice_Check(obj)) {
    return reinterpret_cast<THPDevice*>(obj)->device;
  }
  TORCH_CHECK_TYPE(false, "Expected device");
}

inline Dtype py_object_to_dtype(py::object object) {
  PyObject* obj = object.ptr();
  if (THPDtype_Check(obj)) {
    return reinterpret_cast<THPDtype*>(obj)->scalar_type;
  }
  TORCH_CHECK_TYPE(false, "Expected dtype");
}
} // namespace detail
} // namespace torch::python

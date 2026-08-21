#pragma once
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/api/object.h>
#include <torch/csrc/utils/pybind.h>
#include <optional>

namespace py = pybind11;

namespace torch::jit {

inline std::optional<Object> as_object(py::handle obj) {
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  auto& ScriptObject =
      storage
          .call_once_and_store_result([]() -> py::object {
            return py::module_::import("torch").attr("ScriptObject");
          })
          .get_stored();
#else
  static py::handle ScriptObject =
      py::module::import("torch").attr("ScriptObject");
#endif
  if (py::isinstance(obj, ScriptObject)) {
    return py::cast<Object>(obj);
  }
  return std::nullopt;
}

} // namespace torch::jit

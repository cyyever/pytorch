#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/core/ivalue.h>
#include <ATen/core/symbol.h>
#include <c10/util/irange.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pybind11::detail {

template <>
struct type_caster<torch::jit::IValue> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::IValue, _("IValue"));

  bool load(handle src, bool /*unused*/) {
    try {
      value = torch::jit::toTypeInferredIValue(src);
      return true;
    } catch (std::exception&) {
      return false;
    }
  }

  static handle cast(
      torch::jit::IValue src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return torch::jit::toPyObject(std::move(src)).release();
  }
};

template <>
struct type_caster<torch::jit::Symbol> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::Symbol, _("Symbol"));

  bool load(handle src, bool /*unused*/) {
    // TODO: Is there a way to py::cast that doesn't raise an exception on
    // failure?  Can we catch pybind11::cast_error here instead?
    std::string src_str;
    try {
      src_str = py::cast<std::string>(src);
    } catch (std::exception&) {
      return false;
    }
    value = torch::jit::Symbol::fromQualString(src_str);
    return true;
  }

  static handle cast(
      torch::jit::Symbol src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(std::string(src.toQualString()), return_value_policy::copy)
        .release();
  }
};

} // namespace pybind11::detail

namespace torch::jit {

static inline py::tuple tuple_tail(const py::tuple& tup) {
  py::tuple r(tup.size() - 1);
  for (const auto i : c10::irange(1, tup.size())) {
    r[i - 1] = tup[i];
  }
  return r;
}

} // namespace torch::jit

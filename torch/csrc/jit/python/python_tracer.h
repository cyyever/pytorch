#pragma once

#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>
#include <string>

namespace torch::jit {

struct Module;

namespace tracer {
void initPythonTracerBindings(PyObject* module);

SourceRange getPythonInterpreterSourceRange();

Node* preRecordPythonTrace(
    THPObjectPtr pyobj,
    const std::string& arg_types,
    at::ArrayRef<autograd::Variable> inputs,
    std::vector<THPObjectPtr> scalar_args);

} // namespace tracer
} // namespace torch::jit

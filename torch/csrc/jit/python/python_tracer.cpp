#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/utils/python_strings.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <sstream>

using namespace torch::autograd;
using namespace torch::jit;
using namespace torch::jit::tracer;

namespace torch::jit::tracer {

// Python interpreter retrieval routine adapted from
// https://stackoverflow.com/a/8706144
static std::vector<StackEntry> _pythonCallstack() {
  pybind11::gil_scoped_acquire gil;
  PyFrameObject* frame = PyEval_GetFrame();
  Py_XINCREF(frame);
  std::vector<StackEntry> entries;

  while (nullptr != frame) {
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    size_t line = PyCode_Addr2Line(code.get(), PyFrame_GetLasti(frame));
    std::string filename = THPUtils_unpackString(code->co_filename);
    std::string funcname = THPUtils_unpackString(code->co_name);
    auto source = std::make_shared<Source>(funcname, filename, line);
    entries.emplace_back(
        StackEntry{funcname, SourceRange(source, 0, funcname.size())});
    auto new_frame = PyFrame_GetBack(frame);
    Py_DECREF(frame);
    frame = new_frame;
  }
  return entries;
}

SourceRange getPythonInterpreterSourceRange() {
  auto cs = pythonCallstack();
  std::optional<std::string> source_filename;
  size_t source_line = 0;
  std::stringstream stack_trace;
  for (const auto& entry : cs) {
    auto& range = entry.range;
    if (range.source()) {
      auto& src = range.source();
      if (src && src->filename()) {
        auto line =
            src->starting_line_no() + src->lineno_for_offset(range.start());
        stack_trace << *(src->filename()) << '(' << line
                    << "): " << entry.filename << '\n';
        if (!source_filename) {
          source_filename = *(src->filename());
          source_line = line;
        }
      }
    }
  }

  auto stack_trace_text = std::move(stack_trace).str();
  auto source =
      std::make_shared<Source>(stack_trace_text, source_filename, source_line);
  return SourceRange(source, 0, stack_trace_text.size());
}

Node* preRecordPythonTrace(
    THPObjectPtr pyobj,
    const std::string& arg_types,
    at::ArrayRef<Variable> inputs,
    pyobj_list scalar_args) {
  THPObjectPtr apply(PyObject_GetAttrString(pyobj.get(), "apply"));
  TORCH_CHECK_PYTHON(apply);

  auto& graph = getTracingState()->graph;

  Node* n = graph->createPythonOp(
      std::move(apply), arg_types, std::move(scalar_args));
  recordSourceLocation(n);

  for (const Variable& input : inputs) {
    n->addInput(getValueTrace(input));
  }

  graph->insertNode(n);

  return n;
}

static void pythonRecordSourceLocation(Node* n) {
  n->setSourceRange(getPythonInterpreterSourceRange());
}

static void pythonWarn(const std::string& reason) {
  pybind11::gil_scoped_acquire gil;
  PyErr_WarnEx(PyExc_RuntimeWarning, reason.c_str(), 1);
}

void initPythonTracerBindings(PyObject* module) {
  setPythonCallstack(_pythonCallstack);
  setRecordSourceLocation(pythonRecordSourceLocation);

  auto m = py::handle(module).cast<py::module>();
  py::class_<TracingState, std::shared_ptr<TracingState>>(
      m, "TracingState", py::dynamic_attr())
      // NB: no constructor; you have to get it from C++ code
      .def(
          "__repr__",
          [](const TracingState& s) {
            std::ostringstream ss;
            ss << "<TracingState " << (const void*)&s << '>';
            return std::move(ss).str();
          })
      .def(
          "__str__",
          [](const TracingState& s) -> std::string {
            std::ostringstream ss;
            ss << *s.graph;
            return std::move(ss).str();
          })
      .def(
          "push_scope",
          [](TracingState& s, const std::string& scope_name) {
            s.graph->push_scope(scope_name);
          })
      .def("pop_scope", [](TracingState& s) { s.graph->pop_scope(); })
      .def(
          "current_scope",
          [](TracingState& s) {
            return s.graph->current_scope()->name().toUnqualString();
          })
      .def(
          "set_graph",
          [](TracingState& s, std::shared_ptr<Graph> g) {
            s.graph = std::move(g);
          })
      .def("graph", [](TracingState& s) { return s.graph; });

  m.def("_tracer_warn_use_python", []() { tracer::setWarn(pythonWarn); });
  m.def("_get_tracing_state", []() { return getTracingState(); });
  m.def("_set_tracing_state", [](std::shared_ptr<TracingState> state) {
    return setTracingState(std::move(state));
  });
  m.def("_get_value_trace", [](const Variable& var) {
    return getValueTrace(var);
  });
  m.def("_set_value_trace", [](const Variable& var, Value* value) {
    return setValueTrace(var, value);
  });
  m.def("_tracer_set_get_unique_name_fn", [](const py::function& func) {
    const auto& tracing_state = getTracingState();
    AT_ASSERT(tracing_state);
    tracing_state->lookup_var_name_fn =
        [func](const Variable& var) -> std::string {
      pybind11::gil_scoped_acquire ag;
      return py::cast<std::string>(func(var));
    };
  });
  m.def("_tracer_set_force_outplace", [](bool force_outplace) {
    const auto& tracing_state = getTracingState();
    AT_ASSERT(tracing_state);
    tracing_state->force_outplace = force_outplace;
  });
}

} // namespace torch::jit::tracer

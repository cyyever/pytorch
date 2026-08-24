#include <torch/csrc/utils/throughput_benchmark.h>

namespace torch::throughput_benchmark {

std::ostream& operator<<(
    std::ostream& os,
    const BenchmarkExecutionStats& value) {
  return os << "Average latency / iter (ms): " << value.latency_avg_ms
            << "\n Total number of iters: " << value.num_iters;
}

void ThroughputBenchmark::addInput(py::args args, py::kwargs kwargs) {
  CHECK(module_.initialized());
  module_.addInput(std::move(args), std::move(kwargs));
}

py::object ThroughputBenchmark::runOnce(
    const py::args& args,
    const py::kwargs& kwargs) {
  CHECK(module_.initialized());
  return module_.runOnce(args, kwargs);
}

ThroughputBenchmark::ThroughputBenchmark(py::object module)
    : module_(std::move(module)) {}

BenchmarkExecutionStats ThroughputBenchmark::benchmark(
    const BenchmarkConfig& config) const {
  CHECK(module_.initialized());
  // Main benchmark thread doesn't hold the GIL after scheduling worker threads
  // But for now we don't release it as we will be implicitly manipulating with
  // py::object ref. counts.
  TORCH_WARN(
      "Starting benchmark on an nn.Module. This can be slow due "
      "to Python GIL.");
  return module_.benchmark(config);
}

namespace detail {

template <>
// NOLINTNEXTLINE(*-rvalue-reference-param-not-moved)
void ModuleBenchmark::runOnce(ModuleInput&& input) const {
  CHECK(initialized_);
  pybind11::gil_scoped_acquire gil_guard;
  model_(*input.args, **input.kwargs);
}

template <>
ModuleOutput ModuleBenchmark::runOnce(
    const py::args& args,
    const py::kwargs& kwargs) const {
  CHECK(initialized_);
  pybind11::gil_scoped_acquire gil_guard;
  return model_(*args, **kwargs);
}

template <>
void ModuleBenchmark::addInput(py::args&& args, py::kwargs&& kwargs) {
  inputs_.emplace_back(std::move(args), std::move(kwargs));
}

template <>
ModuleInput cloneInput<ModuleInput>(const ModuleInput& input) {
  pybind11::gil_scoped_acquire gil_guard;
  py::args args = input.args;
  py::kwargs kwargs = input.kwargs;
  return {std::move(args), std::move(kwargs)};
}

} // namespace detail

} // namespace torch::throughput_benchmark

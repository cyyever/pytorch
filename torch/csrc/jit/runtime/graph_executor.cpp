#include <torch/csrc/jit/runtime/graph_executor.h>

#include <atomic>
#include <utility>

namespace torch::jit {

GraphExecutor::GraphExecutor(
    const std::shared_ptr<Graph>& /*graph*/,
    std::string /*function_name*/) {}

GraphExecutor::GraphExecutor(
    const std::shared_ptr<Graph>& /*graph*/,
    std::string /*function_name*/,
    ExecutorExecutionMode /*executor_mode*/) {}

bool GraphExecutor::isOptimized() const {
  return getGraphExecutorOptimize();
}

namespace {
std::atomic<bool> graph_executor_optimize{true};
}

void setGraphExecutorOptimize(bool o) {
  graph_executor_optimize.store(o);
}

bool getGraphExecutorOptimize() {
  return graph_executor_optimize.load();
}

std::atomic<bool>& getProfilingMode() {
  static std::atomic<bool> profiling_mode{false};
  return profiling_mode;
}

std::atomic<bool>& getExecutorMode() {
  static std::atomic<bool> executor_mode{false};
  return executor_mode;
}

std::atomic<size_t>& getNumProfiledRuns() {
  static std::atomic<size_t> num_profiled_runs{8};
  return num_profiled_runs;
}

size_t getBailoutDepth() {
  return getFusionStrategy().front().second;
}

bool IsNewExecutorEnabled() {
  return false;
}

namespace {
FusionStrategy fusion_strategy = {{FusionBehavior::STATIC, 2}};
}

FusionStrategy getFusionStrategy() {
  return fusion_strategy;
}

FusionStrategy setFusionStrategy(FusionStrategy& new_strategy) {
  std::swap(fusion_strategy, new_strategy);
  return new_strategy;
}

bool& getInlineEverythingMode() {
  static bool inline_everything = false;
  return inline_everything;
}

} // namespace torch::jit

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <torch/csrc/jit/ir/ir.h>

TORCH_DECLARE_bool(torch_jit_enable_new_executor);

TORCH_DECLARE_bool(torch_jit_execution_plan_reuse_code_graph);

namespace torch::jit {

// Controls the behavior of ProfilingGraphExecutor. Kept for the
// torch._C._jit_set_fusion_strategy / _jit_get_fusion_strategy bindings.
enum FusionBehavior { STATIC, DYNAMIC };

using FusionStrategy = std::vector<std::pair<FusionBehavior, size_t>>;

TORCH_API FusionStrategy getFusionStrategy();
TORCH_API FusionStrategy setFusionStrategy(FusionStrategy& fusion_strategy);

TORCH_API bool& getInlineEverythingMode();

enum ExecutorExecutionMode {
  SIMPLE,
  PROFILING,
};

struct GraphExecutorImplBase;
struct TORCH_API GraphExecutor {
  GraphExecutor() = default;
  GraphExecutor(const std::shared_ptr<Graph>& graph, std::string function_name);

  GraphExecutor(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      ExecutorExecutionMode executor_mode);

  bool isOptimized() const;

 private:
  std::shared_ptr<GraphExecutorImplBase> pImpl;
};

TORCH_API void setGraphExecutorOptimize(bool o);
TORCH_API bool getGraphExecutorOptimize();

TORCH_API std::atomic<bool>& getProfilingMode();
TORCH_API std::atomic<bool>& getExecutorMode();
TORCH_API std::atomic<size_t>& getNumProfiledRuns();
TORCH_API size_t getBailoutDepth();
TORCH_API bool IsNewExecutorEnabled();

struct TORCH_API GraphOptimizerEnabledGuard {
  GraphOptimizerEnabledGuard(bool state)
      : old_state_(getGraphExecutorOptimize()) {
    setGraphExecutorOptimize(state);
  }

  ~GraphOptimizerEnabledGuard() {
    setGraphExecutorOptimize(old_state_);
  }

  bool old_state_;
};

} // namespace torch::jit

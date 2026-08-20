import torch


def set_fuser(fuser_name, executor_name):
    if fuser_name not in ["old", "none", "default"]:
        raise AssertionError(
            f"fuser_name must be one of 'old', 'none', 'default', but got '{fuser_name}'"
        )
    if fuser_name == "old":
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
    elif fuser_name == "none":
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
    elif fuser_name == "default":
        pass

    # --executor overrides settings of --fuser
    if executor_name == "profiling":
        torch._C._jit_set_profiling_executor(True)
        torch._C._get_graph_executor_optimize(True)
    elif executor_name == "simple":
        torch._C._get_graph_executor_optimize(False)
    elif executor_name == "legacy":
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(True)
    elif executor_name == "default":
        pass

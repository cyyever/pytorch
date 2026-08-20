import torch


def set_fuser(fuser_name, executor_name):
    if fuser_name not in ["none", "default"]:
        raise AssertionError(
            f"fuser_name must be one of 'none', 'default', but got '{fuser_name}'"
        )
    if fuser_name == "none":
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
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

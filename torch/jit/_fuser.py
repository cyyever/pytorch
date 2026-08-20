# mypy: allow-untyped-defs
import contextlib
import warnings

import torch


@contextlib.contextmanager
def optimized_execution(should_optimize):
    """Context manager that controls whether the JIT's executor will run optimizations before executing a function."""
    stored_flag = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(should_optimize)
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)


last_executed_optimized_graph = torch._C._last_executed_optimized_graph


def _get_differentiable_graph_node(node, diff_node) -> None:
    if node.kind() == "prim::DifferentiableGraph":
        diff_node.append(node)
    else:
        for block in node.blocks():
            for n in block.nodes():
                _get_differentiable_graph_node(n, diff_node)


def _graph_for(self, *args, **kwargs):
    return _script_method_graph_for(self, self, *args, **kwargs)


def _script_method_graph_for(self, parent, *args, **kwargs):
    try:
        dbs = parent.get_debug_state()
        eps = list(dbs.execution_plans.values())
        if len(eps) != 1:
            raise AssertionError(f"Expected exactly 1 execution plan, got {len(eps)}")
        graph = eps[0].graph.copy()

        # graph_executor_states for differentiable node
        fw_states = eps[0].code.differentiable_op_executor_states()
        diff_nodes: list[torch._C.Node] = []
        for n in graph.nodes():
            _get_differentiable_graph_node(n, diff_nodes)

        if len(fw_states) != len(diff_nodes):
            raise AssertionError(
                f"Expected fw_states ({len(fw_states)}) and diff_nodes "
                f"({len(diff_nodes)}) to have the same length"
            )
        # swap each differentiable graph with optimized graph in their execution plan
        for n, state in zip(diff_nodes, fw_states):
            fw_execution_plans = list(state.execution_plans.values())
            # we can only update the subgraph when there's a unique execution
            # plan. Avoid assert here so we would skip the ones that can't be
            # updated while try the best effort to update other nodes.
            if len(fw_execution_plans) == 1:
                n.g_("Subgraph", fw_execution_plans[0].graph)

        return graph
    except Exception:
        # fallback approach, we just ran the graph and return the recorded optimized
        # graph
        self(*args, **kwargs)
        return last_executed_optimized_graph()


def set_fusion_strategy(strategy: list[tuple[str, int]]):
    """Set the type and number of specializations that can occur during fusion.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    Usage: provide a list of pairs (type, depth) where type is one of "STATIC" or "DYNAMIC"
    and depth is an integer.

    Behavior - static vs dynamic:
        In STATIC fusion, fused ops are compiled to have fixed input shapes. The shape is determined
        based on some initial profiling runs.
        In DYNAMIC fusion, fused ops are compiled to have variable input shapes, so that multiple
        shapes are possible.

    In both cases, we also recompile on new striding behavior, device, or dtype.

    Behavior - fallback functions & depth:
        When an input doesn't match the format required by the specialized compiled op, it will run
        a fallback function. Fallback functions are recursively compiled and specialized based
        on the observed tensor shapes. Since compilation can be slow, the "depth" parameter is provided to
        limit the number of specializations that can be compiled, before giving up on recompiling and
        falling back to a completely un-fused, un-specialized implementation.

    The list of (type, depth) pairs controls the type of specializations and the number of
    specializations. For example: [("STATIC", 2), ("DYNAMIC", 2)] indicates that the first
    two specializations will use static fusions, the following two specializations will use
    dynamic fusion, and any inputs that satisfy none of the 4 options will run an
    unfused implementation.

    NB: in the future, as more and more fusion backends are added there may be more granular
    apis for specific fusers.
    """
    warnings.warn(
        "`torch.jit.set_fusion_strategy` is deprecated. Please use `torch.compile` instead.",
        DeprecationWarning,
    )
    return torch._C._jit_set_fusion_strategy(strategy)

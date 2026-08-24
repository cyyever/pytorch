# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import copy
import operator

import torch
from torch._dynamo.utils import counters

from .. import config
from ..lowering import lowerings as L
from ..pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    stable_topological_sort,
)
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern


aten = torch.ops.aten
prims = torch.ops.prims


def _is_valid_concat_linear_int8_woq_optimization_pattern():
    def fn(match):
        if not config.cpp.enable_concat_linear:
            return False
        if not all(k in match.kwargs for k in ("x", "w1", "w2", "w3", "scales")):
            raise AssertionError("expected x, w1, w2, w3, scales in match kwargs")
        if not all(
            hasattr(match.kwargs[key], "meta")
            for key in ["x", "w1", "w2", "w3", "scales"]
        ):
            return False
        x = match.kwargs["x"].meta["val"]
        w1 = match.kwargs["w1"].meta["val"]
        w2 = match.kwargs["w2"].meta["val"]
        w3 = match.kwargs["w3"].meta["val"]
        scales = match.kwargs["scales"].meta["val"]
        if len(match.kwargs["scales"].meta["val"].size()) > 1:
            return False
        num_scales = match.kwargs["scales"].meta["val"].numel()
        w1_cols = match.kwargs["w1"].meta["val"].size()[0]
        w2_cols = match.kwargs["w2"].meta["val"].size()[0]
        w3_cols = match.kwargs["w3"].meta["val"].size()[0]
        return (
            # For now, we only support woq mm kernels
            # with x.type=bfloat16 and w.type=int8
            x.dtype == torch.bfloat16
            and w1.dtype == torch.int8
            and w2.dtype == torch.int8
            and w3.dtype == torch.int8
            and scales.dtype == torch.bfloat16
            and x.device.type in ("cpu", "cuda")
            and x.device == w1.device
            and w1.device == w2.device
            and w2.device == w3.device
            and x.device == scales.device
            and num_scales == w1_cols + w2_cols + w3_cols
        )

    return fn


def _is_valid_woq_optimization_pattern():
    def fn(match):
        if not all(k in match.kwargs for k in ("x", "weight", "scales")):
            raise AssertionError("expected x, weight, scales in match kwargs")
        if not all(
            hasattr(match.kwargs[key], "meta") for key in ["x", "weight", "scales"]
        ):
            return False
        x = match.kwargs["x"].meta["val"]
        weight = match.kwargs["weight"].meta["val"]
        scales = match.kwargs["scales"].meta["val"]
        return (
            # For now, we only support woq mm kernels
            # with x.type=bfloat16 and w.type=int8
            x.dtype == torch.bfloat16
            and weight.dtype == torch.int8
            and scales.dtype == torch.bfloat16
            and x.device.type in ("cpu", "cuda", "xpu")
            and x.device == weight.device
            and x.device == scales.device
        )

    return fn


def _register_concat_linear_int8_woq_lowering(
    pattern, computation_woq, computation_reshape
):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_concat_linear_int8_woq_optimization_pattern(),
        pass_number=4,
    )
    def woq_int8(match: Match, *args, **kwargs):
        x = kwargs["x"]
        w1 = kwargs["w1"]
        w2 = kwargs["w2"]
        w3 = kwargs["w3"]
        scales = kwargs["scales"]
        counters["inductor"]["woq_matcher_count"] += 1
        counters["inductor"]["woq_matcher_nodes"] += len(match.nodes)
        out_features = (
            w1.meta["val"].size()[0]
            + w2.meta["val"].size()[0]
            + w3.meta["val"].size()[0]
        )
        origin_x_size = tuple(x.meta["val"].size())
        x_shape = [-1, origin_x_size[-1]]
        out_shape = list(origin_x_size[:-1] + (out_features,))
        mm_node_of_x = None
        for candidate in iter(x.users.keys()):
            if (
                candidate.target is aten.mm.default
                and list(candidate._input_nodes)[1].target is aten.cat.default
            ):
                mm_node_of_x = candidate
                break
        if mm_node_of_x is None:
            raise AssertionError("unable to find mm node")
        _, cat_wgt_node = mm_node_of_x._input_nodes
        scaling_node = next(iter(mm_node_of_x.users.keys()))
        user_of_scaling_node = next(iter(scaling_node.users.keys()))
        # Some other pass is making some changes that entails
        # adding a node before it's used, but it can only be found when
        # lint is run. stable_topological_sort() is being run before lint,
        # so that error was not being discovered.
        # We call stable_topological_sort here as a workaround.
        stable_topological_sort(match.graph)
        with match.graph.inserting_before(user_of_scaling_node):
            new_cat_node = match.graph.call_function(
                aten.cat.default,
                args=([w1, w2, w3], 0),
            )
            x_reshape_node = match.graph.call_function(
                computation_reshape, args=(x, x_shape)
            )
            new_woq_node = match.graph.call_function(
                computation_woq,
                args=(x_reshape_node, new_cat_node, scales),
            )
            new_woq_node.meta = copy.copy(x.meta)
            output_reshape_node = match.graph.call_function(
                computation_reshape, args=(new_woq_node, out_shape)
            )
            scaling_node.replace_all_uses_with(output_reshape_node)
            match.graph.erase_node(scaling_node)
            match.graph.erase_node(mm_node_of_x)
            match.graph.erase_node(cat_wgt_node)
            match.graph.lint()

    return woq_int8


def _register_woq_lowering(pattern, computation_woq, computation_reshape):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_woq_optimization_pattern(),
    )
    def woq_int8(match: Match, *args, **kwargs):
        x = kwargs["x"]
        weight = kwargs["weight"]
        scales = kwargs["scales"]
        counters["inductor"]["woq_matcher_count"] += 1
        counters["inductor"]["woq_matcher_nodes"] += len(match.nodes)
        out_features = weight.get_size()[0]
        origin_x_size = x.get_size()
        x_shape = [-1, origin_x_size[-1]]
        out_shape = origin_x_size[:-1] + [
            out_features,
        ]
        func1 = L[computation_reshape](x, x_shape)
        func2 = L[computation_woq](func1, weight, scales)
        return L[computation_reshape](func2, out_shape)

    return woq_int8


def _register_woq_mm_int8_pattern1():
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to mm, with x reshape
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.mm.default,
                CallFunction(aten.reshape.default, KeywordArg("x"), Arg()),
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
            ),
            Arg(),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern2():
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to mm, w/o x reshape
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.mm.default,
                KeywordArg("x"),
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
            ),
            Arg(),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern3():
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to bmm
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.bmm.default,
            CallFunction(aten.expand.default, KeywordArg("x"), Arg()),
            CallFunction(
                aten.expand.default,
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
                Arg(),
            ),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern4():
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.mm.default,
            KeywordArg("x"),
            CallFunction(
                prims.convert_element_type.default,
                CallFunction(
                    aten.permute.default,
                    KeywordArg("weight"),
                    Arg(),
                ),
                Arg(),
            ),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_int8_woq_concat_linear_pattern():
    def _create_wgt_node(wgt_node_name: str):
        return CallFunction(
            prims.convert_element_type.default,
            CallFunction(
                aten.permute.default,
                KeywordArg(wgt_node_name),
                Arg(),
            ),
            Arg(),
        )

    cat_wgt = CallFunction(
        aten.cat.default, [_create_wgt_node(wgt) for wgt in ["w1", "w2", "w3"]], 1
    )

    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(aten.mm.default, KeywordArg("x"), cat_wgt),
        KeywordArg("scales"),
    )
    _register_concat_linear_int8_woq_lowering(
        _woq_pattern, aten._weight_int8pack_mm.default, aten.reshape
    )


def _register_woq_lowerings():
    _register_woq_mm_int8_pattern1()
    _register_woq_mm_int8_pattern2()
    _register_woq_mm_int8_pattern3()
    _register_woq_mm_int8_pattern4()


def _is_valid_concat_linear_woq_int4_fusion(computation_nodes):
    computation_op = torch.ops.aten._weight_int4pack_mm_for_cpu.default
    act = computation_nodes[0].args[0]
    wgt = computation_nodes[0].args[1]
    in_feature_size = wgt.meta.get("val").size(1)  # type: ignore[union-attr]
    group_size = computation_nodes[0].args[2]
    return len(computation_nodes) >= 2 and all(
        (
            node.target == computation_op
            and node.args[0] == act  # share same activation
            and (
                node.args[1].meta.get("val").size(1) == in_feature_size
            )  # same in feature size
            and (node.args[1] != wgt or gemm_idx == 0)
            and node.args[1].op == "get_attr"  # wgt are all constants
            and node.args[2] == group_size  # same group size
        )
        for gemm_idx, node in enumerate(computation_nodes)
    )


def concat_linear_woq_int4(gm: torch.fx.GraphModule):
    """
    Concat Linear optimization pass for WOQ int4
    This pass fuses the original pattern:
    def ...
        return (woq_int4(x, w1, group_size, scale_zp1), woq_int4(x, w2, group_size, scale_zp1) ...)
    into a single operation:
    def ...
        concat_res = woq_int4(x, concat_w, group_size, concat_scale_zp)
        return split(concat_res, split_size_list)
    """

    def concat_wgt(packed_wgts, scale_zps, group_size, act_dtype):
        # Concat the wgts and scale_zps, and repack the wgt
        unpacked_wgts = []
        for packed_wgt in packed_wgts:
            # Get the unpacked weight list
            # Same as https://github.com/pytorch/pytorch/pull/156174
            K = packed_wgt.size(1) * 2
            N = packed_wgt.size(0)
            x = torch.eye(K).to(dtype=act_dtype)
            qscales_and_zeros = (
                torch.tensor([1.0, 8.0])
                .to(dtype=act_dtype)
                .expand(K // group_size, N, 2)
                .contiguous()
            )
            unpacked_wgts.append(
                torch.ops.aten._weight_int4pack_mm_for_cpu(
                    x,
                    packed_wgt,
                    group_size,
                    qscales_and_zeros,
                )
                .t()
                .contiguous()
                .to(torch.int32)  # N, K
            )
        concat_unpacked_wgt = torch.cat(unpacked_wgts, dim=0)
        repack_w = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            concat_unpacked_wgt, 1
        )
        concat_scale_zp = torch.cat(scale_zps, dim=1).contiguous()
        return repack_w, concat_scale_zp

    graph = gm.graph
    computation_op = torch.ops.aten._weight_int4pack_mm_for_cpu.default
    for node in graph.find_nodes(op="call_function", target=computation_op):
        if (
            not node._erased
            and isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].device.type == "cpu"
        ):
            act = node.args[0]
            users = list(act.users)
            if _is_valid_concat_linear_woq_int4_fusion(users):
                with graph.inserting_before(node):
                    if not all(user.args[1].op == "get_attr" for user in users):
                        raise AssertionError(
                            "expected all users to have get_attr weight"
                        )
                    computation_node_0 = users[0]
                    packed_wgts = [getattr(gm, user.args[1].target) for user in users]
                    group_size = computation_node_0.args[2]
                    scale_zps = [getattr(gm, user.args[3].target) for user in users]
                    out_feature_size_list = [
                        packed_wgt.size(0) for packed_wgt in packed_wgts
                    ]
                    repack_w, concat_scale_zp = concat_wgt(
                        packed_wgts, scale_zps, group_size, act.meta.get("val").dtype
                    )
                    repack_w_node_name = computation_node_0.args[1].target + "_concat"
                    concat_scale_zp_node_name = (
                        computation_node_0.args[3].target + "_concat"
                    )
                    gm.register_buffer(repack_w_node_name, repack_w)
                    setattr(gm, repack_w_node_name, repack_w)
                    gm.register_buffer(concat_scale_zp_node_name, concat_scale_zp)
                    setattr(gm, concat_scale_zp_node_name, concat_scale_zp)

                    repack_w_node = graph.create_node(
                        "get_attr", repack_w_node_name, (), {}
                    )
                    with graph.inserting_after(repack_w_node):
                        concat_scale_zp_node = graph.create_node(
                            "get_attr", concat_scale_zp_node_name, (), {}
                        )

                    with graph.inserting_after(concat_scale_zp_node):
                        concat_int4_gemm_node = graph.create_node(
                            "call_function",
                            computation_op,
                            (
                                act,
                                repack_w_node,
                                group_size,
                                concat_scale_zp_node,
                            ),
                        )
                    with graph.inserting_after(concat_int4_gemm_node):
                        split_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.split_with_sizes.default,
                            (
                                concat_int4_gemm_node,
                                out_feature_size_list,
                                1,  # split dim
                            ),
                        )
                    with graph.inserting_after(split_node):
                        for gemm_idx, user in enumerate(users):
                            if user.target != computation_op:
                                raise AssertionError(
                                    f"expected target {computation_op}, got {user.target}"
                                )
                            get_item = graph.create_node(
                                "call_function",
                                operator.getitem,
                                (
                                    split_node,
                                    gemm_idx,
                                ),
                            )
                            with graph.inserting_after(get_item):
                                clone_node = graph.create_node(
                                    "call_function",
                                    torch.ops.aten.clone.default,
                                    (get_item,),
                                    {"memory_format": torch.contiguous_format},
                                )
                                user.replace_all_uses_with(clone_node)
                                graph.erase_node(user)

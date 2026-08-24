# mypy: allow-untyped-defs

import torch
import torch.utils._pytree as pytree
from torch._inductor.kernel.mm_common import mm_args

from . import config, ir
from .codegen.cpp_gemm_template import CppGemmTemplate
from .codegen.cpp_grouped_gemm_template import CppGroupedGemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .ir import TensorBox
from .lowering import (
    add,
    add_needs_realized_inputs,
    aten,
    permute,
    register_lowering,
    view,
)
from .select_algorithm import (
    autotune_select_algorithm,
    ChoiceCaller,
    ExternKernelChoice,
)
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template
from .virtualized import ops, OpsValue, V


def create_int8_compensation(
    W_tensor: torch.Tensor,
    packed_weight: ir.TensorBox,
    x_scale: ir.TensorBox,
    x_zp: ir.TensorBox,
    w_scale: ir.TensorBox,
) -> tuple[
    bool,
    ir.TensorBox,
    ir.TensorBox | None,
]:
    x_w_scale: ir.TensorBox | None = None
    use_int8_fast_compensation_path = all(
        isinstance(item, ir.TensorBox)
        and item.get_name() in V.graph.constants
        and hasattr(item.data, "data")
        and isinstance(item.data.data, ir.ConstantBuffer)
        for item in [x_scale, x_zp, w_scale]
    )
    if use_int8_fast_compensation_path:
        x_w_scale_tensor = (
            V.graph.constants[x_scale.get_name()]
            * V.graph.constants[w_scale.get_name()]
        )
        x_w_scale = V.graph.add_tensor_constant(
            x_w_scale_tensor,
            name=packed_weight.get_name() + "_x_w_compens",
        )
        weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
        x_zp_tensor = V.graph.constants[x_zp.get_name()]
        weight_compens_tensor = weight_compens_tensor * x_w_scale_tensor * x_zp_tensor
        weight_compens = V.graph.add_tensor_constant(
            weight_compens_tensor,
            name=packed_weight.get_name() + "_BMatrixCompens",
        )
    else:
        weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
        weight_compens = V.graph.add_tensor_constant(
            weight_compens_tensor,
            name=packed_weight.get_name() + "_BMatrixCompens",
        )
    return (  # type: ignore[return-type]
        use_int8_fast_compensation_path,
        weight_compens,
        x_w_scale,
    )


def codegen_int8_gemm_template_compensation(
    use_int8_fast_compensation_path: bool,
    input: OpsValue,
    _weight_compo: OpsValue,
    _x_scale: OpsValue | None,
    _x_zp: OpsValue | None,
    _w_scale: OpsValue | None,
    _x_w_scale: OpsValue | None,
) -> OpsValue:
    if use_int8_fast_compensation_path:
        temp = ops.sub(
            ops.mul(
                input,
                _x_w_scale,
            ),
            _weight_compo,
        )
    else:
        temp = ops.mul(
            ops.mul(
                input,
                _x_scale,
            ),
            _w_scale,
        )
        # NOTE: We will apply compensation even if the x_zp is 0 for int8 quantization.
        # That's because when torch.compile is invoked for dynamic quantization,
        # x might coincidentally have such values that x_zp might be zero despite
        # asymmetric quantization.
        # Besides, if x_zp is dummy for int8 x, or if x is statically quantized,
        # we'd still perform that redundant compute to avoid making the code messy
        # because we discovered that redundant computation of compensation did not
        # lead to performance degradation with the input shapes tested.
        temp = ops.sub(
            temp,
            ops.mul(
                ops.mul(
                    ops.mul(
                        _x_scale,
                        _w_scale,
                    ),
                    _x_zp,
                ),
                _weight_compo,
            ),
        )
    return temp


def grouped_gemm_lowering(
    x: TensorBox,
    w: list[TensorBox],
    b: list[TensorBox],
    attr=None,
    scalars=None,
    algorithm=None,
    layout=None,
):
    x_size = x.get_size()
    if len(x_size) > 2:
        # GEMM template needs 2D input, normalize input shape here
        x = view(x, [-1, x_size[-1]])
    num_gemm = len(w)

    if not (config.max_autotune or config.max_autotune_gemm):
        raise AssertionError(
            "grouped_gemm_lowering requires max_autotune or max_autotune_gemm"
        )
    # pyrefly: ignore [bad-assignment]
    b = [bias if bias is None else ir.ExternKernel.realize_input(bias) for bias in b]

    choices: list[ChoiceCaller] = []
    *_, layout, x, _ = mm_args(x, permute(w[0], [1, 0]), layout=layout)

    kwargs = {
        "has_bias": [bias is not None for bias in b],
        "trans_w": True,
        "epilogue_creator": None,
        "act_mapping": dict.fromkeys(range(num_gemm), x),
    }

    input_nodes = [x, *w]
    input_nodes.extend([bias for bias in b if bias is not None])

    CppGroupedGemmTemplate.add_choices(
        choices,
        layout,
        input_nodes,
        **kwargs,  # type: ignore[arg-type]
    )

    if len(choices) == 0:
        raise AssertionError("expected at least one choice for grouped_gemm")
    result, _ = autotune_select_algorithm(
        "grouped_gemm",
        choices,
        input_nodes,
        layout,
    )
    template_buf = result.data.data
    return_bufs = [
        ir.MultiOutput(layout, template_buf, [(list, gemm_idx)])
        for gemm_idx in range(num_gemm)
    ]
    # pyrefly: ignore [bad-argument-type]
    template_buf.layout = ir.MultiOutputLayout(device=input_nodes[0].get_device())
    template_buf.outputs = return_bufs
    return_tensors = [
        ir.TensorBox.create(return_bufs[gemm_idx]) for gemm_idx in range(num_gemm)
    ]
    if len(x_size) > 2:
        for gemm_idx in range(num_gemm):
            return_tensors[gemm_idx] = view(
                return_tensors[gemm_idx],  # type: ignore[arg-type]
                (*x_size[:-1], return_tensors[gemm_idx].get_size()[-1]),
            )
    return return_tensors


grouped_gemm_lowering._inductor_lowering_function = True  # type: ignore[attr-defined]
grouped_gemm_lowering._inductor_lowering_output_metadata_ignores_input_storage = True  # type: ignore[attr-defined]


def _convert_to_0d_constant(
    tensor_box: ir.TensorBox,
    dtype: torch.dtype,
    name_suffix: str = "_0d",
) -> ir.TensorBox:
    """
    Normalize a tensor with all dimensions equal to 1 to a 0D ConstantBuffer.
    """
    if isinstance(tensor_box, ir.TensorBox):
        data = tensor_box.data
        if isinstance(data, ir.StorageBox):
            data = data.data
    else:
        data = tensor_box

    if isinstance(data, ir.ConstantBuffer):
        tensor = V.graph.constants.get(tensor_box.get_name())
        if tensor is not None:
            return V.graph.add_tensor_constant(
                tensor.reshape([]), name=tensor_box.get_name() + name_suffix
            )

    from torch._inductor.lowering import get_constant_value

    const_value = get_constant_value(data)
    if const_value is not None:
        return V.graph.add_tensor_constant(
            torch.tensor(const_value.value, dtype=dtype).reshape([])
        )

    tensor_box.realize()

    from .ir import ExternKernel, GenericView

    if isinstance(tensor_box.data, GenericView):
        tensor_box = ir.TensorBox(ExternKernel.require_contiguous(tensor_box.data))

    result = view(tensor_box, [])
    return result


def register_onednn_fusion_ops():
    if torch._C._has_mkldnn:
        from . import mkldnn_ir

        aten_mkldnn_linear_unary = ExternKernelChoice(
            torch.ops.mkldnn._linear_pointwise,
            "mkldnn::_linear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.LinearUnary.create,
        )
        aten_mkldnn_linear_binary = ExternKernelChoice(
            torch.ops.mkldnn._linear_pointwise.binary,
            "mkldnn::_linear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.LinearBinary.create,
        )
        cpu_needs_realized_inputs: list[
            torch._ops.OpOverload | torch._ops.OpOverloadPacket
        ] = [
            torch.ops.mkldnn._convolution_pointwise,
            torch.ops.mkldnn._convolution_pointwise_,
            torch.ops.mkldnn._convolution_transpose_pointwise,
            torch.ops.mkldnn._linear_pointwise,
            aten.mkldnn_rnn_layer.default,
        ]

        @register_lowering(torch.ops.mkldnn._convolution_pointwise)
        def convolution_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                mkldnn_ir.ConvolutionUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise.binary)
        def convolution_binary(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            return TensorBox.create(
                mkldnn_ir.ConvolutionBinary.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise_.binary)
        def convolution_binary_inplace(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            return TensorBox.create(
                mkldnn_ir.ConvolutionBinaryInplace.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._linear_pointwise)
        def linear_unary(
            x: TensorBox,
            w: TensorBox,
            b: TensorBox,
            attr,
            scalars,
            algorithm,
            layout=None,
        ):
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            if b is not None:
                b = ir.ExternKernel.realize_input(b)  # type: ignore[assignment]
            choices: list[ChoiceCaller] = []
            if config.max_autotune or config.max_autotune_gemm:
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w = mm_args(x, transposed_w, layout=layout)
                if use_cpp_gemm_template(layout, x, transposed_w):

                    def epilogue_creator(buf):
                        return create_epilogue_with_attr(
                            buf, attr, scalars=scalars, algorithm=algorithm
                        )

                    kwargs = {
                        "has_bias": b is not None,
                        "trans_w": True,
                        "epilogue_creator": (
                            None if attr == "none" else epilogue_creator
                        ),
                    }
                    if b is not None:
                        kwargs["input_indices"] = [2, 0, 1]  # type: ignore[assignment]
                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, w] if b is None else [x, w, b],
                        **kwargs,  # type: ignore[arg-type]
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(attr=attr, scalars=scalars, algorithm=algorithm)
                if b is None:
                    kwargs["B"] = None
                choices.append(
                    aten_mkldnn_linear_unary.bind(
                        [x, w] if b is None else [x, w, b],
                        layout,
                        **kwargs,
                    )
                )
            if w.get_name() not in V.graph.constants:
                raise AssertionError("weight must be a graph constant")
            input_gen_fns = {
                1: lambda x: V.graph.constants[x.get_name()],
            }
            result, _ = autotune_select_algorithm(
                "linear_unary",
                choices,
                [x, w] if b is None else [x, w, b],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(torch.ops.mkldnn._linear_pointwise.binary)
        def linear_binary(
            x: TensorBox, y: TensorBox, w: TensorBox, b: TensorBox, attr, layout=None
        ):
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            y_size = y.get_size()
            if len(y_size) > 2:
                y = view(y, [-1, y_size[-1]])
            if b is not None:
                b = ir.ExternKernel.realize_input(b)  # type: ignore[assignment]
            choices: list[ChoiceCaller] = []
            if config.max_autotune or config.max_autotune_gemm:
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w, y = mm_args(
                    x, transposed_w, y, layout=layout
                )
                if use_cpp_gemm_template(layout, x, transposed_w):

                    def epilogue_creator(buf):
                        return create_epilogue_with_attr(buf, attr, other=y)

                    kwargs = {
                        "has_bias": b is not None,
                        "trans_w": True,
                        "epilogue_creator": epilogue_creator,
                    }

                    # pyrefly: ignore [bad-typed-dict-key, unsupported-operation]
                    kwargs["input_indices"] = [0, 2, 1] if b is None else [3, 0, 2, 1]
                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, y, w] if b is None else [x, y, w, b],
                        **kwargs,  # type: ignore[arg-type]
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(attr=attr)
                if b is None:
                    kwargs["B"] = None
                choices.append(
                    aten_mkldnn_linear_binary.bind(
                        [x, y, w] if b is None else [x, y, w, b],
                        layout,
                        **kwargs,
                    )
                )
            if w.get_name() not in V.graph.constants:
                raise AssertionError("weight must be a graph constant")
            input_gen_fns = {
                2: lambda x: V.graph.constants[x.get_name()],
            }
            result, _ = autotune_select_algorithm(
                "linear_binary",
                choices,
                [x, y, w] if b is None else [x, y, w, b],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(torch.ops.mkldnn._convolution_transpose_pointwise)
        def convolution_transpose_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                mkldnn_ir.ConvolutionTransposeUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(aten.mkldnn_rnn_layer.default)
        def mkldnn_rnn_layer(
            x: TensorBox,
            w0: TensorBox,
            w1: TensorBox,
            w2: TensorBox,
            w3: TensorBox,
            hx: TensorBox,
            cx: TensorBox,
            reverse: bool,
            batch_sizes: list[int],
            mode: int,
            hidden_size: int,
            num_layers: int,
            has_biases: bool,
            bidirectional: bool,
            batch_first: bool,
            train: bool,
        ):
            return pytree.tree_map(
                TensorBox.create,
                mkldnn_ir.MkldnnRnnLayer.create(
                    x,
                    w0,
                    w1,
                    w2,
                    w3,
                    hx,
                    cx,
                    reverse,
                    batch_sizes,
                    mode,
                    hidden_size,
                    num_layers,
                    has_biases,
                    bidirectional,
                    batch_first,
                    train,
                ),
            )

        if torch._C.has_mkl:
            aten_mkl_linear = ExternKernelChoice(
                torch.ops.mkl._mkl_linear,
                "mkl::_mkl_linear",
                has_out_variant=False,
                kernel_creator=mkldnn_ir.MKLPackedLinear.create,
            )
            cpu_needs_realized_inputs.append(torch.ops.mkl._mkl_linear)

            @register_lowering(torch.ops.mkl._mkl_linear)
            def mkl_packed_linear(
                x: TensorBox,
                packed_w: TensorBox,
                orig_w: TensorBox,
                b: TensorBox | None,
                batch_size,
                *,
                layout=None,
            ):
                choices: list[ChoiceCaller] = []
                if config.max_autotune or config.max_autotune_gemm:
                    transposed_w = permute(orig_w, [1, 0])
                    *_, layout, x, transposed_w = mm_args(
                        x, transposed_w, layout=layout
                    )
                    if use_cpp_gemm_template(layout, x, transposed_w):
                        CppGemmTemplate.add_choices(
                            choices,
                            layout,
                            [x, packed_w, orig_w],
                            trans_w=True,
                            input_indices=[0, 2],
                        )

                if len(choices) == 0 or use_aten_gemm_kernels():
                    choices.append(
                        aten_mkl_linear.bind(
                            (x, packed_w, orig_w), layout, B=None, batch_size=batch_size
                        )
                    )

                if packed_w.get_name() not in V.graph.constants:
                    raise AssertionError("packed_w must be a graph constant")
                if orig_w.get_name() not in V.graph.constants:
                    raise AssertionError("orig_w must be a graph constant")
                # packed_w is a mkldnn tensor which we can't generate directly
                # so we use the weights from the original tensor in autotune.
                input_gen_fns = {
                    1: lambda x: V.graph.constants[x.get_name()],
                    2: lambda x: V.graph.constants[x.get_name()],
                }
                result: TensorBox  # annotation on separate line since tuple unpacking doesn't support inline annotation
                result, _ = autotune_select_algorithm(
                    "packed_linear",
                    choices,
                    [x, packed_w, orig_w],
                    layout,
                    input_gen_fns=input_gen_fns,
                )
                if b is not None:
                    result = add(result, b)
                return result

        add_needs_realized_inputs(cpu_needs_realized_inputs)


register_onednn_fusion_ops()

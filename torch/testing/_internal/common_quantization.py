# mypy: ignore-errors

r"""Common utilities for tests that exercise quantized tensors without the
removed torch.ao stack."""

import functools
import unittest

import numpy as np
import torch
import torch._dynamo as torchdynamo


def skipIfNoFBGEMM(fn):
    reason = "Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs with instruction set support AVX2 or newer."
    if isinstance(fn, type):
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoONEDNN(fn):
    reason = "Quantized operations require ONEDNN."
    if isinstance(fn, type):
        if "onednn" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "onednn" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoONEDNNBF16(fn):
    reason = "Quantized operations require BF16 support."
    if isinstance(fn, type):
        if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoDynamoSupport(fn):
    reason = "dynamo doesn't support."
    if isinstance(fn, type):
        if not torchdynamo.is_dynamo_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torchdynamo.is_dynamo_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoInductorSupport(fn):
    reason = "inductor doesn't support."
    if isinstance(fn, type):
        if not torchdynamo.is_inductor_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torchdynamo.is_inductor_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


try:
    import torchvision  # noqa: F401

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skip_if_no_torchvision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def _static_quantized_linear_module(
    N,
    K,
    bias,
    example_input,
    epilogue="none",
    output_dtype=torch.float32,
):
    """
    Generate a linear module using onednn.qlinear_pointwise directly
    with static quantization parameters (no choose_qparams at runtime).
    It is used to test fusion and lowering passes in Inductor for X86 CPU.
    Input quantization limit is 0-127 to avoid overflow on old platforms.
    Params:
        N: output feature dimension
        K: input feature dimension
        bias: boolean flag to indicate whether linear module has bias
        example_input: example input tensor to get scale/zero point
        epilogue: oneDNN qlinear pointwise post op, e.g. none/relu/gelu
        output_dtype: output dtype used by onednn.qlinear_pointwise
    Return:
        An instance of the quantized linear module
    """
    if epilogue not in ("none", "relu", "gelu"):
        raise AssertionError(f"Unsupported epilogue: {epilogue}")

    class Model(torch.nn.Module):
        def __init__(
            self,
            N,
            K,
            bias,
            example_input,
            epilogue,
            output_dtype,
        ):
            super().__init__()
            self.x_scale, self.x_zp = torch.ops.quantized_decomposed.choose_qparams.tensor(
                example_input,
                quant_min=0,
                quant_max=127,
                eps=torch.finfo(torch.float32).eps,
                dtype=torch.uint8,
            )
            self.x_scale, self.x_zp = self.x_scale.detach().item(), self.x_zp.detach().item()
            self.linear = torch.nn.Linear(K, N, bias)
            self.w_scales, self.w_zps = torch.ops.quantized_decomposed.choose_qparams_per_token(
                self.linear.weight, dtype=torch.int8
            )
            self.w_scales = self.w_scales.detach().to(torch.float32).squeeze()
            self.w_zps = self.w_zps.detach().to(torch.int64).squeeze()
            self.qw = torch.ops.quantized_decomposed.quantize_per_channel.default(
                self.linear.weight,
                self.w_scales,
                self.w_zps,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )

            if bias:
                self.b = self.linear.bias
            else:
                self.b = None

            self.y_scale = 1.0
            self.y_zp = 0
            self.output_dtype = output_dtype
            self.post_op = epilogue
            self.unary_post_op_args = ()
            self.post_op_algo = "none"

        def forward(self, x):
            qw_packed = torch.ops.onednn.qlinear_prepack(self.qw, None)
            if x.is_floating_point():
                qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    x,
                    self.x_scale,
                    self.x_zp,
                    quant_min=0,
                    quant_max=255,
                    dtype=torch.uint8,
                )
            else:
                qx = x
            return torch.ops.onednn.qlinear_pointwise.default(
                qx,
                self.x_scale,
                self.x_zp,
                qw_packed,
                self.w_scales,
                self.w_zps,
                self.b,
                self.y_scale,
                self.y_zp,
                self.output_dtype,
                self.post_op,
                self.unary_post_op_args,
                self.post_op_algo,
            )

    return Model(N, K, bias, example_input, epilogue, output_dtype).eval()


def _static_quantized_linear_binary_module(
    N,
    K,
    bias,
    example_input,
    example_binary_input,
    binary_post_op="add",
    epilogue="none",
    output_dtype=torch.float32,
):
    if epilogue not in ("none", "relu"):
        raise AssertionError(f"Unsupported epilogue: {epilogue}")
    if binary_post_op not in ("add", "sum"):
        raise AssertionError(f"Unsupported binary post op: {binary_post_op}")

    class Model(torch.nn.Module):
        def __init__(
            self,
            N,
            K,
            bias,
            example_input,
            example_binary_input,
            binary_post_op,
            epilogue,
            output_dtype,
        ):
            super().__init__()
            self.x_scale, self.x_zp = torch.ops.quantized_decomposed.choose_qparams.tensor(
                example_input,
                quant_min=0,
                quant_max=127,
                eps=torch.finfo(torch.float32).eps,
                dtype=torch.uint8,
            )
            self.x_scale, self.x_zp = (
                self.x_scale.detach().item(),
                self.x_zp.detach().item(),
            )
            self.linear = torch.nn.Linear(K, N, bias)
            self.w_scales, self.w_zps = torch.ops.quantized_decomposed.choose_qparams_per_token(
                self.linear.weight, dtype=torch.int8
            )
            self.w_scales = self.w_scales.detach().to(torch.float32).squeeze()
            self.w_zps = self.w_zps.detach().to(torch.int64).squeeze()
            self.qw = torch.ops.quantized_decomposed.quantize_per_channel.default(
                self.linear.weight,
                self.w_scales,
                self.w_zps,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )

            if bias:
                self.b = self.linear.bias
            else:
                self.b = None

            if output_dtype in (torch.uint8, torch.int8):
                # Calibrate output quantization params from an eager reference of this binary op.
                # This mirrors the pre-pass graphs where uint8 outputs have non-trivial y_scale/y_zp.
                example_out = self.linear(example_input) + example_binary_input
                qmin, qmax, qdtype = (0, 127, torch.uint8)
                if output_dtype == torch.int8:
                    qmin, qmax, qdtype = (-128, 127, torch.int8)
                y_scale, y_zp = torch.ops.quantized_decomposed.choose_qparams.tensor(
                    example_out,
                    quant_min=qmin,
                    quant_max=qmax,
                    eps=torch.finfo(torch.float32).eps,
                    dtype=qdtype,
                )
                self.y_scale, self.y_zp = y_scale.detach().item(), y_zp.detach().item()
            else:
                self.y_scale, self.y_zp = 1.0, 0

            self.output_dtype = output_dtype
            self.binary_post_op = binary_post_op
            self.binary_alpha = 1.0
            self.unary_post_op = epilogue
            self.unary_post_op_args = ()
            self.unary_post_op_algo = ""

        def forward(self, x, other):
            qw_packed = torch.ops.onednn.qlinear_prepack(self.qw, None)
            other_scale, other_zp = 1.0, 0
            if x.is_floating_point():
                qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    x,
                    self.x_scale,
                    self.x_zp,
                    quant_min=0,
                    quant_max=127,
                    dtype=torch.uint8,
                )
            else:
                qx = x
            return torch.ops.onednn.qlinear_pointwise.binary(
                qx,
                self.x_scale,
                self.x_zp,
                qw_packed,
                self.w_scales,
                self.w_zps,
                other,
                self.b,
                self.y_scale,
                self.y_zp,
                self.output_dtype,
                other_scale,
                other_zp,
                self.binary_post_op,
                self.binary_alpha,
                self.unary_post_op,
                self.unary_post_op_args,
                self.unary_post_op_algo,
            )

    return Model(
        N,
        K,
        bias,
        example_input,
        example_binary_input,
        binary_post_op,
        epilogue,
        output_dtype,
    ).eval()
def _group_quantize_tensor(w, n_bit=4, q_group_size=16):
    if w.dim() != 2:
        raise AssertionError(f"expected w.dim() == 2, got {w.dim()}")
    w = w.transpose(0, 1).contiguous()
    if q_group_size <= 1:
        raise AssertionError(f"expected q_group_size > 1, got {q_group_size}")
    if w.shape[-1] % q_group_size != 0:
        raise AssertionError(
            f"expected w.shape[-1] % q_group_size == 0, got w.shape[-1]={w.shape[-1]}, q_group_size={q_group_size}"
        )

    to_quant = w.reshape(-1, q_group_size)
    if torch.isnan(to_quant).sum() != 0:
        raise AssertionError("to_quant contains NaN values")

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    if torch.isnan(scales).sum() != 0:
        raise AssertionError("scales contains NaN values")

    zeros = min_val + scales * (2 ** (n_bit - 1))
    if torch.isnan(zeros).sum() != 0:
        raise AssertionError("zeros contains NaN values")

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    if torch.isnan(out).sum() != 0:
        raise AssertionError("out contains NaN values")

    out = out.to(dtype=torch.int32).reshape(w.shape)
    if out.device != torch.device("cpu"):
        out = (out[::, ::2] << 4 | out[::, 1::2]).to(torch.uint8)

    # Scales and zeros for the same q-group should be contiguous, so we can
    # load as a 32-bit word
    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1)
    scales_and_zeros = (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

    return out, scales_and_zeros


def _group_quantize_tensor_symmetric(w, n_bit=4, groupsize=32):
    # W is of shape [K x N]
    # We transpose W as Quantization is applied on [N x K]
    w = w.transpose(0, 1).contiguous()
    if w.dim() != 2:
        raise AssertionError(f"Expected w.dim() == 2, got {w.dim()}")
    if groupsize <= 1:
        raise AssertionError(f"Expected groupsize > 1, got {groupsize}")
    if w.shape[-1] % groupsize != 0:
        raise AssertionError(f"Expected w.shape[-1] % groupsize == 0, got {w.shape[-1]} % {groupsize}")
    # Calculate scale and zeros
    to_quant = w.reshape(-1, groupsize)
    max_val = to_quant.abs().amax(dim=1, keepdim=True)
    eps = torch.finfo(max_val.dtype).eps
    max_int = 2 ** (n_bit - 1) - 1  # For 4-bit, this is 7
    scales = max_val.clamp(min=eps) / max_int
    zeros = torch.zeros_like(scales)

    # Quantize the weight
    scales = scales.to(torch.float32).reshape(w.shape[0], -1)
    zeros = zeros.to(torch.float32).reshape(w.shape[0], -1)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    max_int = 2**n_bit - 1
    w_int8 = to_quant.div(scales).add(8.5).to(torch.int8).clamp(max=max_int)
    # We pack 2 signed int4 values in unsigned uint8 container.
    # This reduces the weight size by half and improves load perf
    out_uint8 = (w_int8[::, 1::2] << 4 | w_int8[::, ::2]).to(torch.uint8)

    scales_and_zeros = scales.squeeze().contiguous()

    return out_uint8, scales_and_zeros


def _dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # source: https://github.com/meta-pytorch/gpt-fast/blob/main/quantize.py
    # default setup for affine quantization of activations
    x_dtype = x.dtype
    x = x.float()
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales.to(x_dtype), zero_points



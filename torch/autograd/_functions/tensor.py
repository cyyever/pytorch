# mypy: allow-untyped-defs
import operator
from functools import reduce

from torch.autograd.function import Function


# TODO: deprecate this
class Resize(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, tensor, sizes):
        ctx.sizes = sizes
        ctx.numel = reduce(operator.mul, sizes, 1)
        if tensor.numel() != ctx.numel:
            raise RuntimeError(
                (
                    "requested resize to {} ({} elements in total), "
                    "but the given tensor has a size of {} ({} elements). "
                    "autograd's resize can only change the shape of a given "
                    "tensor, while preserving the number of elements. "
                ).format(
                    "x".join(map(str, sizes)),
                    ctx.numel,
                    "x".join(map(str, tensor.size())),
                    tensor.numel(),
                )
            )
        ctx.input_sizes = tensor.size()
        if tensor.is_quantized:
            tensor.copy_(tensor)
            return tensor.contiguous().view(*sizes)
        if tensor.is_contiguous():
            result = tensor.new(tensor).contiguous().view(*sizes)
            return result
        else:
            return tensor.contiguous().view(*sizes)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        if grad_output.numel() != ctx.numel:
            raise AssertionError(
                f"Expected grad_output to have {ctx.numel} elements, but got {grad_output.numel()}"
            )
        return grad_output.contiguous().view(ctx.input_sizes), None

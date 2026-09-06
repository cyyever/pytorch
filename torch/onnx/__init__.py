from enum import IntEnum
from typing import Any

from torch.onnx import symbolic_helper as symbolic_helper
from torch.onnx import symbolic_opset11 as symbolic_opset11


_UNSUPPORTED_MESSAGE = (
    "ONNX support is not included in this PyTorch build. "
    "Use a PyTorch distribution built with ONNX support."
)


class TensorProtoDataType(IntEnum):
    FLOAT = 1
    INT64 = 7


def register_custom_op_symbolic(
    symbolic_name: str,
    symbolic_fn: Any,
    opset_version: int,
) -> None:
    # Downstream packages register symbolic functions at import time. Keep that
    # registration harmless because this build cannot start an ONNX export.
    return None


def unregister_custom_op_symbolic(symbolic_name: str, opset_version: int) -> None:
    return None


def is_in_onnx_export() -> bool:
    return False


def export(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(_UNSUPPORTED_MESSAGE)


def dynamo_export(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(_UNSUPPORTED_MESSAGE)


__all__ = [
    "TensorProtoDataType",
    "dynamo_export",
    "export",
    "is_in_onnx_export",
    "register_custom_op_symbolic",
    "symbolic_helper",
    "symbolic_opset11",
    "unregister_custom_op_symbolic",
]

# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch.backends import (
    __allow_nonbracketed_mutation,
    _FP32Precision,
    _get_fp32_precision_getter,
    _set_fp32_precision_setter,
    ContextProp,
    PropModule,
)


def set_flags(_deterministic=None, _allow_tf32=None, _fp32_precision="none"):
    orig_flags = (
        torch._C._get_mkldnn_deterministic(),
        torch._C._get_onednn_allow_tf32(),
        torch._C._get_fp32_precision_getter("mkldnn", "all"),
    )
    if _deterministic is not None:
        torch._C._set_mkldnn_deterministic(_deterministic)
    if _allow_tf32 is not None:
        torch._C._set_onednn_allow_tf32(_allow_tf32)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("mkldnn", "all", _fp32_precision)
    return orig_flags


@contextmanager
def flags(deterministic=False, allow_tf32=True, fp32_precision="none"):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(deterministic, allow_tf32, fp32_precision)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class MkldnnModule(PropModule):
    deterministic = ContextProp(
        torch._C._get_mkldnn_deterministic, torch._C._set_mkldnn_deterministic
    )
    allow_tf32 = ContextProp(
        torch._C._get_onednn_allow_tf32, torch._C._set_onednn_allow_tf32
    )
    matmul = _FP32Precision("mkldnn", "matmul")
    conv = _FP32Precision("mkldnn", "conv")
    rnn = _FP32Precision("mkldnn", "rnn")
    fp32_precision = ContextProp(
        _get_fp32_precision_getter("mkldnn", "all"),
        _set_fp32_precision_setter("generic", "all"),
    )


if TYPE_CHECKING:
    deterministic: ContextProp
    allow_tf32: ContextProp
    fp32_precision: str

sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)

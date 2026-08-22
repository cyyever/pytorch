# mypy: allow-untyped-defs
"""
This file includes private common utilities shared by the FSDP2 fully_shard
implementation.
"""

import dataclasses
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn


_MAX_TRAVERSE_DEPTH = 128


def _is_namedtuple(obj: Any) -> bool:
    # Mirrors torch.nn.parallel.scatter_gather._is_namedtuple
    fields = getattr(type(obj), "_fields", None)
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_asdict")
        and isinstance(fields, tuple)
        and all(isinstance(f, str) for f in fields)
    )


def collect_grad_tensors(output: Any) -> tuple[torch.Tensor, ...]:
    """
    Recursively collect tensors that require gradients from a nested structure.

    Traverses dict, list, tuple, NamedTuple, and dataclass containers.
    Sets and other iterables are *not* traversed (consistent with
    ``tree_flatten``).  Uses the same traversal order as
    :func:`replace_grad_tensors`.
    """
    tensors_list: list[torch.Tensor] = []
    _collect_grad_tensors(output, tensors_list)
    return tuple(tensors_list)


def _collect_grad_tensors(
    output: Any, out: list[torch.Tensor], _depth: int = 0
) -> None:
    """Collect grad-requiring tensors in the same order as _replace_grad_tensors."""
    if _depth >= _MAX_TRAVERSE_DEPTH:
        raise RuntimeError(
            f"collect_grad_tensors exceeded max depth ({_MAX_TRAVERSE_DEPTH}), "
            "likely due to a circular reference in the output structure"
        )
    # Branch order must mirror _replace_grad_tensors exactly.
    # Only dict, list, tuple, NamedTuple, and dataclass are traversed;
    # set and other iterables are intentionally skipped (matching tree_flatten).
    if torch.is_tensor(output) and output.requires_grad:
        out.append(output)
    elif _is_namedtuple(output):
        # NamedTuple before dataclass to match _replace_grad_tensors ordering.
        for item in output:
            _collect_grad_tensors(item, out, _depth + 1)
    elif dataclasses.is_dataclass(output) and not isinstance(output, type):
        for field in dataclasses.fields(output):
            _collect_grad_tensors(getattr(output, field.name), out, _depth + 1)
    elif isinstance(output, dict):
        for v in output.values():
            _collect_grad_tensors(v, out, _depth + 1)
    elif isinstance(output, (list, tuple)):
        for item in output:
            _collect_grad_tensors(item, out, _depth + 1)


def replace_grad_tensors(output: Any, tensor_iter: Iterator[torch.Tensor]) -> Any:
    """
    Replace grad-requiring tensors in a nested structure using replacements
    from tensor_iter.

    Tensors are consumed from tensor_iter in the same traversal order as
    :func:`collect_grad_tensors`. Traverses dict, list, tuple, NamedTuple,
    and dataclass containers; sets and other iterables are *not* traversed
    (consistent with ``tree_flatten``).

    Note: dataclass reconstruction uses ``dataclasses.replace()``, which calls
    ``__init__``. Dataclasses with custom ``__init__`` validation,
    ``__post_init__`` side effects, or non-standard dict subclass constructors
    may not be compatible. In practice, FSDP module outputs are expected to be
    shallowly nested, so recursion depth is not a concern.
    """
    result = _replace_grad_tensors(output, tensor_iter)
    sentinel = object()
    leftover = next(tensor_iter, sentinel)
    if leftover is not sentinel:
        # Count remaining without holding references to all of them
        n = 1 + sum(1 for _ in tensor_iter)
        raise RuntimeError(
            f"{n} replacement tensors were not consumed while processing "
            f"{type(output).__qualname__}"
        )
    return result


def _replace_grad_tensors(
    output: Any, tensor_iter: Iterator[torch.Tensor], _depth: int = 0
) -> Any:
    # Branch order must mirror _collect_grad_tensors exactly.
    if _depth >= _MAX_TRAVERSE_DEPTH:
        raise RuntimeError(
            f"replace_grad_tensors exceeded max depth ({_MAX_TRAVERSE_DEPTH}), "
            "likely due to a circular reference in the output structure"
        )
    if torch.is_tensor(output) and output.requires_grad:
        return next(tensor_iter)
    elif _is_namedtuple(output):
        # NamedTuple before dataclass: a NamedTuple that is also a dataclass
        # should be reconstructed via positional args, not dataclasses.replace.
        new_items = []
        any_changed = False
        for item in output:
            new_item = _replace_grad_tensors(item, tensor_iter, _depth + 1)
            new_items.append(new_item)
            if new_item is not item:
                any_changed = True
        if any_changed:
            return type(output)(*new_items)
        return output
    elif dataclasses.is_dataclass(output) and not isinstance(output, type):
        changes = {}
        for field in dataclasses.fields(output):
            old_val = getattr(output, field.name)
            new_val = _replace_grad_tensors(old_val, tensor_iter, _depth + 1)
            if new_val is not old_val:
                changes[field.name] = new_val
        if changes:
            try:
                return dataclasses.replace(output, **changes)
            except TypeError as e:
                raise TypeError(
                    f"Failed to reconstruct dataclass {type(output).__qualname__} "
                    f"via dataclasses.replace(). Dataclasses used as FSDP module "
                    f"inputs/outputs must support dataclasses.replace(): {e}"
                ) from None
        return output
    elif isinstance(output, dict):
        new_dict = {}
        any_changed = False
        for k, v in output.items():
            new_v = _replace_grad_tensors(v, tensor_iter, _depth + 1)
            new_dict[k] = new_v
            if new_v is not v:
                any_changed = True
        if any_changed:
            return new_dict if type(output) is dict else type(output)(new_dict)
        return output
    elif isinstance(output, (list, tuple)):
        new_items = []
        any_changed = False
        for item in output:
            new_item = _replace_grad_tensors(item, tensor_iter, _depth + 1)
            new_items.append(new_item)
            if new_item is not item:
                any_changed = True
        if any_changed:
            typ = type(output)
            try:
                return typ(new_items)
            except TypeError:
                # Fall back to base type for subclasses with custom __init__
                return list(new_items) if isinstance(output, list) else tuple(new_items)
        return output
    else:
        return output




def _named_parameters_with_duplicates(
    module: nn.Module, **kwargs: Any
) -> list[tuple[str, nn.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    if "remove_duplicate" in kwargs:
        raise AssertionError(
            "_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument."
        )
    kwargs["remove_duplicate"] = False
    try:
        ret = list(module.named_parameters(**kwargs))
    except AssertionError:
        kwargs.pop("remove_duplicate")
        ret = list(module.named_parameters(**kwargs))
    return ret

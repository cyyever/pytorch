# mypy: allow-untyped-defs
"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

import _thread
import collections
import enum
import io
import pickle
import threading
import weakref
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import torch
from torch._C import _Await as CAwait, Future as CFuture


_P = ParamSpec("_P")
_R = TypeVar("_R")

LockType: type = _thread.LockType

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched: "weakref.WeakKeyDictionary[Callable, dict[str, Callable]]" = (
    weakref.WeakKeyDictionary()
)  # noqa: T484


# allows BroadcastingList instance to be subscriptable
class BroadcastingListCls:
    def __getitem__(self, types):
        return


# mypy doesn't support parameters on types, so we have to explicitly type each
# list size
BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()[f"BroadcastingList{i}"] = BroadcastingList1


def boolean_dispatch(
    arg_name,
    arg_index,
    default,
    if_true,
    if_false,
    module_name,
    func_name,
):
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """

    def fn(*args, **kwargs):
        dispatch_flag = default
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    if if_true.__doc__ is None and if_false.__doc__ is not None:
        doc = if_false.__doc__
        if_true.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is not None:
        doc = if_true.__doc__
        if_false.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is None:
        # neither function has a docstring
        doc = None
    else:
        raise RuntimeError("only one function can have a docstring")
    fn.__doc__ = doc

    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name

    boolean_dispatched[fn] = {
        "if_true": if_true,
        "if_false": if_false,
        "index": arg_index,
        "default": default,
        "arg_name": arg_name,
    }
    return fn


class FunctionModifiers:
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """

    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from an exported function / forward)"
    COPY_TO_SCRIPT_WRAPPER = (
        "if this method is not scripted, copy the python method onto the scripted model"
    )
    _DROP = "_drop (function is fully ignored, declaration can be unscriptable)"


def unused(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn


            class MyModule(nn.Module):
                def __init__(self, use_memory_efficient):
                    super().__init__()
                    self.use_memory_efficient = use_memory_efficient

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb

                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10


            m = torch.jit.script(MyModule(use_memory_efficient=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    """
    if isinstance(fn, property):
        prop = fn
        setattr(  # noqa: B010
            prop.fget, "_torchscript_modifier", FunctionModifiers.UNUSED
        )

        if prop.fset:
            setattr(  # noqa: B010
                prop.fset, "_torchscript_modifier", FunctionModifiers.UNUSED
            )

        return prop

    fn._torchscript_modifier = FunctionModifiers.UNUSED  # type: ignore[attr-defined]
    return fn


# No op context manager from python side


def _drop(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    fn._torchscript_modifier = FunctionModifiers._DROP  # type: ignore[attr-defined]
    return fn


def _copy_to_script_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER  # type: ignore[attr-defined]
    return fn


def _overload(func: Callable[_P, _R]) -> Callable[_P, _R]:
    return func


def _overload_method(func: Callable[_P, _R]) -> Callable[_P, _R]:
    return func


def _create_named_tuple(
    t,
    unqual_name: str,
    field_names: list[str],
    defaults: tuple[Any, ...],
):
    TupleType = collections.namedtuple(unqual_name, field_names, defaults=defaults)  # type: ignore[call-arg, no-redef, misc]
    return TupleType(*t)


class _TensorExtractor(pickle.Pickler):
    def __init__(self, *args, tensors: list[torch.Tensor], **kwargs):
        super().__init__(*args, **kwargs)
        self.tensors = tensors

    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return ""
        # Since we just want to extract tensors, we don't mind if an object is
        # unpicklable if it doesn't contain tensors, as we can just ignore/skip
        # it. To play it safe, we only do so for common objects that we're sure
        # don't contain tensors. Feel free to add new types here. Note also that
        # even if a type isn't listed here this won't block users, since they
        # can just add a __getstate__ or __reduce__ method to their class.
        if isinstance(obj, LockType):
            return ""
        # A Future does not technically contain a value, it just offers the
        # means to access one.
        if isinstance(obj, CFuture):
            return ""
        if isinstance(obj, CAwait):
            return ""
        if isinstance(obj, torch.cuda.Event):
            return ""
        if isinstance(obj, threading.Thread):
            return ""
        return None


def _extract_tensors(obj):
    r"""
    This function is exclusively called from C++.
    See ``torch/csrc/jit/python/python_ivalue.h``.

    It extracts the tensors contained in the given object, through pickling.
    """
    tensors: list[torch.Tensor] = []
    extractor = _TensorExtractor(io.BytesIO(), protocol=-1, tensors=tensors)
    extractor.dump(obj)
    return tensors


# Typed enums (IntEnum, for example) retain base class methods in the subclass,
# so drop them explicitly to preserve the previous behavior.

_drop(enum.Enum.__new__)
_drop(enum.Enum.__format__)
_drop(enum.Enum.__repr__)
_drop(enum.Enum.__str__)

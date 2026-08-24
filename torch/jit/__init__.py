from typing import Any, Final as Final, TypeVar

from torch._jit_internal import is_scripting


__all__ = [
    "annotate",
    "export",
    "Final",
    "ignore",
    "interface",
    "is_scripting",
    "is_tracing",
    "script",
    "script_if_tracing",
    "unused",
]

_T = TypeVar("_T")


def is_tracing() -> bool:
    """TorchScript has been removed, so no trace can ever be active."""
    return False


# TorchScript has been removed, so the decorators below only ever marked code up
# for a compiler that no longer exists. They stay as identity functions because
# third-party libraries decorate eager code paths with them.
def unused(fn: _T) -> _T:
    return fn


def export(fn: _T) -> _T:
    return fn


def interface(obj: _T) -> _T:
    return obj


def _overload_method(fn: _T) -> _T:
    return fn


def script_if_tracing(fn: _T) -> _T:
    return fn


_script_if_tracing = script_if_tracing


def ignore(drop: Any = False, **kwargs: Any) -> Any:
    """Accepts both the bare @ignore and the @ignore(drop=...) spelling."""
    if callable(drop):
        return drop
    return lambda fn: fn


def script(obj: Any = None, *args: Any, **kwargs: Any) -> Any:
    if obj is None:
        return lambda fn: fn
    return obj


def annotate(the_type: Any, the_value: _T) -> _T:
    return the_value


def Attribute(value: _T, type: Any) -> _T:  # noqa: A002
    return value

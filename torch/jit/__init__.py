from typing import Any, TypeVar


__all__ = [
    "annotate",
    "export",
    "ignore",
    "script",
    "script_if_tracing",
    "unused",
]

_T = TypeVar("_T")


# TorchScript has been removed, so the decorators below only ever marked code up
# for a compiler that no longer exists. They stay as identity functions because
# third-party libraries decorate eager code paths with them.
def unused(fn: _T) -> _T:
    return fn


def export(fn: _T) -> _T:
    return fn


# torchvision decorates eager forward() overloads with this.
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

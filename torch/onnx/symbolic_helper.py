from collections.abc import Callable
from typing import Any, TypeVar


_F = TypeVar("_F", bound=Callable[..., Any])


def parse_args(*arg_descriptors: str) -> Callable[[_F], _F]:
    def decorator(function: _F) -> _F:
        return function

    return decorator

from typing import Any
from collections.abc import Callable
from typing_extensions import Protocol, runtime_checkable


class TimerClass(Protocol):
    """This is the portion of the `timeit.Timer` API used by benchmark utils."""
    def __init__(
        self,
        stmt: str,
        setup: str,
        timer: Callable[[], float],
        globals: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        ...

    def timeit(self, number: int) -> float:
        ...


@runtime_checkable
class TimeitModuleType(Protocol):
    """Modules generated from `timeit_template.cpp`."""
    def timeit(self, number: int) -> float:
        ...

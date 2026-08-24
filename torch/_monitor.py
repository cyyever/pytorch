# No-op replacement for the removed torch.monitor wait counters. Internal
# callers (dynamo, inductor, c10d) instrument wait times with these; with the
# monitor event/counter backend gone the instrumentation records nothing.


class _WaitCounterTracker:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        pass


class _WaitCounter:
    def __init__(self, name: str) -> None:
        pass

    def guard(self) -> _WaitCounterTracker:
        return _WaitCounterTracker()

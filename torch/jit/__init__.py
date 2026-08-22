from torch._jit_internal import is_scripting


__all__ = ["is_scripting", "is_tracing"]


def is_tracing() -> bool:
    """TorchScript has been removed, so no trace can ever be active."""
    return False

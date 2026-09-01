# NB: Keep this file in sync with enums in aten/src/ATen/core/Reduction.h


def get_enum(reduction: str) -> int:
    if reduction == "none":
        ret = 0
    elif reduction == "mean":
        ret = 1
    elif reduction == "sum":
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return ret

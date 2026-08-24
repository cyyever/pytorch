# mypy: allow-untyped-defs
"""
Minimal ShardedTensor-era data structures retained for DCP wire compatibility.
Not part of the public API; new code should use DTensor.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce

import torch
from torch.distributed.remote_device import _remote_device

__all__ = ["Shard", "ShardMetadata", "narrow_tensor_by_index"]


@dataclass
class ShardMetadata:
    shard_offsets: list[int]
    shard_sizes: list[int]
    placement: _remote_device | None

    def __init__(
        self,
        shard_offsets: list[int],
        shard_sizes: list[int],
        placement: str | _remote_device | None = None,
    ):
        self.shard_offsets = shard_offsets
        self.shard_sizes = shard_sizes
        if isinstance(placement, str):
            self.placement = _remote_device(placement)
        else:
            self.placement = placement
        if len(self.shard_offsets) != len(self.shard_sizes):
            raise ValueError(
                f"shard_offsets and shard_sizes should have "
                f"the same number of elements, found {len(self.shard_offsets)} "
                f"and {self.shard_sizes} respectively"
            )
        for i in range(len(self.shard_offsets)):
            if self.shard_offsets[i] < 0:
                raise ValueError("shard_offsets should be >=0")
            if self.shard_sizes[i] < 0:
                raise ValueError("shard_sizes should be >= 0")

    def __hash__(self):
        def _hash_reduce(a, b):
            return (a << 8) + hash(b)

        res = reduce(_hash_reduce, self.shard_offsets, 37)
        res = reduce(_hash_reduce, self.shard_sizes, res)
        res = _hash_reduce(res, self.placement)
        return res


@dataclass
class Shard:
    metadata: ShardMetadata
    tensor: torch.Tensor


def narrow_tensor_by_index(
    tensor: torch.Tensor,
    offsets: Sequence[int],
    sizes: Sequence[int],
) -> torch.Tensor:
    """
    Narrow the tensor according to ``offsets`` and ``sizes``.
    """
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor

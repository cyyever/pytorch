from ._fully_shard import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    FSDPModule,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
    register_fsdp_forward_method,
    share_comm_ctx,
    UnshardHandle,
)


class FullyShardedDataParallel:
    """FSDP1 is gone; only fully_shard (FSDP2) remains.

    Kept as a name so that third-party isinstance() checks for an
    FSDP1-wrapped module resolve, and correctly answer False.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "FullyShardedDataParallel (FSDP1) was removed; use "
            "torch.distributed.fsdp.fully_shard instead."
        )


__all__ = [
    "FullyShardedDataParallel",
    "CPUOffloadPolicy",
    "DataParallelMeshDims",
    "FSDPModule",
    "fully_shard",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "register_fsdp_forward_method",
    "UnshardHandle",
    "share_comm_ctx",
]

# Set namespace for exposed private names
CPUOffloadPolicy.__module__ = "torch.distributed.fsdp"
DataParallelMeshDims.__module__ = "torch.distributed.fsdp"
FSDPModule.__module__ = "torch.distributed.fsdp"
fully_shard.__module__ = "torch.distributed.fsdp"
MixedPrecisionPolicy.__module__ = "torch.distributed.fsdp"
OffloadPolicy.__module__ = "torch.distributed.fsdp"
register_fsdp_forward_method.__module__ = "torch.distributed.fsdp"
UnshardHandle.__module__ = "torch.distributed.fsdp"
share_comm_ctx.__module__ = "torch.distributed.fsdp"
FullyShardedDataParallel.__module__ = "torch.distributed.fsdp"

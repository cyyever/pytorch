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


__all__ = [
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

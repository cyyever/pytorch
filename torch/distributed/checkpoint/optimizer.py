# Copyright (c) Meta Platforms, Inc. and affiliates

from collections.abc import Sequence
from typing import cast

import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed.checkpoint._nested_dict import unflatten_state_dict
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    STATE_DICT_TYPE,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import LoadPlanner

# pyrefly: ignore [deprecated]
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.storage import StorageReader


# TODO: Update docstrings for optimizer.py
__all__ = [
    "load_sharded_optimizer_state_dict",
]


def _alloc_tensor(
    props: TensorProperties, size: Sequence[int], device_type: str = "cuda"
) -> torch.Tensor:
    if device_type == "cpu":
        device = cast(torch.device, _get_device_module(device_type).current_device())
    else:
        device = torch.device(
            device_type, _get_device_module(device_type).current_device()
        )

    return torch.empty(
        size=size,
        dtype=props.dtype,
        layout=props.layout,
        requires_grad=props.requires_grad,
        pin_memory=props.pin_memory,
        device=device,
    )


def load_sharded_optimizer_state_dict(
    model_state_dict: STATE_DICT_TYPE,
    optimizer_key: str,
    storage_reader: StorageReader,
    planner: LoadPlanner | None = None,
) -> STATE_DICT_TYPE:
    """
    Load a state_dict in conjunction with FSDP sharded optimizer state.

    Examples::

    >>> # xdoctest: +SKIP
    >>> import torch.distributed.checkpoint as dist_cp
    >>> model: torch.nn.Module
    >>> optim = torch.optim.SGD(model.parameters(), lr=0.01)
    >>> state_dict = {"app": {}, "optimizer": None}
    >>> state_dict["optimizer"] = torch.distributed.checkpoint.state_dict.get_state_dict(
    ...     model, optimizers=optim
    ... )[1]
    >>> dist_cp.save(
    ...     state_dict=state_dict,
    ...     storage_writer=dist_cp.FileSystemWriter("checkpoint"),
    ... )
    >>> optim_state = dist_cp.load_sharded_optimizer_state_dict(
    ...     model.state_dict(),
    ...     optimizer_key="optimizer",
    ...     storage_reader=dist_cp.FileSystemReader("checkpoint"),
    ... )
    """
    metadata = storage_reader.read_metadata()

    device_type = dist.distributed_c10d._get_pg_default_device().type

    # Create a state_dict for optimizer state
    state_dict: STATE_DICT_TYPE = {}
    for key, value in metadata.state_dict_metadata.items():
        key_path = metadata.planner_data[key]
        if key_path[0] != optimizer_key:
            continue

        if isinstance(value, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"
        else:
            # value: TensorStorageMetadata
            state_dict[key] = _alloc_tensor(value.properties, value.size, device_type)

    load_state_dict(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=planner,
    )

    state_dict = unflatten_state_dict(state_dict, metadata.planner_data)

    return state_dict

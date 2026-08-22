# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

"""
The following example demonstrates how to use PyTorch Distributed Checkpoint to save a FSDP2 model.

This is the current recommended way to checkpoint FSDP.
torch.save() and torch.load() is not recommended when checkpointing sharded models.
"""

import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.multiprocessing as mp
from torch.distributed.fsdp import fully_shard


CHECKPOINT_DIR = f"/scratch/{os.environ.get('LOGNAME', '')}/checkpoint"


def opt_at(opt, idx):
    return list(opt.state.values())[idx]


def init_model():
    model = torch.nn.Linear(4, 4).cuda(dist.get_rank())
    fully_shard(model)
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    model(torch.rand(4, 4)).sum().backward()
    optim.step()

    return model, optim


def print_params(stage, model_1, model_2, optim_1, optim_2):
    model_1_sd = dist_cp.state_dict.get_model_state_dict(model_1)
    model_2_sd = dist_cp.state_dict.get_model_state_dict(model_2)
    print(
        f"{stage} --- rank: {dist.get_rank()}\n"
        f"model.weight: {model_1_sd['weight']}\n"
        f"model_2.weight:{model_2_sd['weight']}\n"
        f"model.bias: {model_1_sd['bias']}\n"
        f"model_2.bias: {model_2_sd['bias']}\n"
    )

    print(
        f"{stage} --- rank: {dist.get_rank()}\n"
        f"optim exp_avg:{opt_at(optim_1, 0)['exp_avg']}\n"
        f"optim_2 exp_avg:{opt_at(optim_2, 0)['exp_avg']}\n"
        f"optim exp_avg_sq:{opt_at(optim_1, 0)['exp_avg_sq']}\n"
        f"optim_2 exp_avg_sq:{opt_at(optim_2, 0)['exp_avg_sq']}\n"
    )


def run_fsdp_checkpoint_example(rank, world_size):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Create a model
    model_1, optim_1 = init_model()

    # Save the model to CHECKPOINT_DIR
    model_sd, optim_sd = dist_cp.state_dict.get_state_dict(model_1, optim_1)
    dist_cp.save(
        state_dict={"model": model_sd, "optim": optim_sd},
        storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
    )

    # Create a second model
    model_2, optim_2 = init_model()

    # Print the model parameters for both models.
    # Before loading, the parameters should be different.
    print_params("Before loading", model_1, model_2, optim_1, optim_2)

    # Load model_2 with parameters saved in CHECKPOINT_DIR
    model_sd, optim_sd = dist_cp.state_dict.get_state_dict(model_2, optim_2)
    dist_cp.load(
        state_dict={"model": model_sd, "optim": optim_sd},
        storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
    )
    dist_cp.state_dict.set_state_dict(
        model_2, optim_2, model_state_dict=model_sd, optim_state_dict=optim_sd
    )

    # Print the model parameters for both models.
    # After loading, the parameters should be the same.
    print_params("After loading", model_1, model_2, optim_1, optim_2)

    # Shut down world pg
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(
        run_fsdp_checkpoint_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

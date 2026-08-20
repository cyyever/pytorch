# mypy: allow-untyped-defs

import sys
from functools import partial, wraps

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS


TEST_GPU_NUM = 4


class ShardedTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self):
        return TEST_GPU_NUM

    def init_pg(self, backend="nccl"):
        if backend not in ["nccl", "gloo", "mpi", "hccl", "xccl"]:
            raise RuntimeError(f"Backend {backend} not supported!")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # set device for nccl pg for collectives
        if backend == "nccl" or backend == "xccl":
            torch.accelerator.set_device_index(self.rank)

    def init_comms(self, backend="nccl"):
        self.init_pg(backend=backend)

    def destroy_comms(self):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def assert_sharded_tensor_equal(self, st1, st2):
        st1_local_shards = st1.local_shards()
        st2_local_shards = st2.local_shards()
        self.assertEqual(len(st1_local_shards), len(st2_local_shards))
        for i, st1_local_shard in enumerate(st1_local_shards):
            self.assertEqual(st1_local_shard.tensor, st2_local_shards[i].tensor)
            self.assertEqual(st1_local_shard.metadata, st2_local_shards[i].metadata)

        self.assertEqual(st1.metadata(), st2.metadata())
        self.assertEqual(st1.sharding_spec(), st2.sharding_spec())


# wrapper to initialize comms (processgroup)
def with_comms(func=None, backend="nccl"):
    if func is None:
        return partial(with_comms, backend=backend)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Skip test if backend requires accelerator but not enough devices available
        acc = torch.accelerator.current_accelerator()
        if backend in ["nccl", "xccl", "hccl"]:
            if (
                acc is None
                or backend != dist.get_default_backend_for_device(acc)
                or torch.accelerator.device_count() < self.world_size
            ):
<<<<<<< HEAD
                sys.exit(TEST_SKIPS[f"multi-device-{self.world_size}"].exit_code)
        self.init_comms(init_rpc=init_rpc, backend=backend)
=======
                sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        self.init_comms(backend=backend)
>>>>>>> 14685524f87 (Remove the distributed RPC framework)
        func(self, *args, **kwargs)
        self.destroy_comms()

    return wrapper

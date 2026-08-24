# Owner(s): ["oncall: distributed"]

import sys
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint._extension import ZStandard
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import (
    get_test_extension_registry,
    Rot13Example,
    with_temp_dir,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = torch.distributed.get_default_backend_for_device(device_type)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def assert_state_dict_equal(
    self: TestCase,
    state_dict_1: dict[str, torch.Tensor],
    state_dict_2: dict[str, torch.Tensor],
) -> bool:
    self.assertEqual(
        len(state_dict_1), len(state_dict_2), "state_dict must be the same size"
    )
    self.assertEqual(
        set(state_dict_1.keys()),
        set(state_dict_2.keys()),
        "state_dict keys do not match",
    )

    for key, value_1 in state_dict_1.items():
        value_2 = state_dict_2[key]
        if isinstance(value_1, DTensor):
            self.assertTrue(
                torch.equal(value_1.to_local(), value_2.to_local()),
                lambda msg: f"{msg}\nKey {key}'s shard does not match",
            )
        elif isinstance(value_1, torch.Tensor):
            self.assertTrue(
                torch.equal(value_1, value_2),
                lambda msg: f"{msg}\nKey {key}'s tensor does not match",
            )

    return True


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


class TestDistributedStateDictSaveLoad(TestCase):
    def test_read_write_only_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()

            fs_writer = FileSystemWriter(path=path)
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
                no_dist=True,
            )

            state_dict_to_load_to = MyTestModule().state_dict()

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding
            fs_reader = FileSystemReader(path=path)
            load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
                no_dist=True,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()

            fs_writer = FileSystemWriter(path=path, single_file_per_rank=True)
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
                no_dist=True,
            )

            state_dict_to_load_to = MyTestModule().state_dict()

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding
            fs_reader = FileSystemReader(path=path)
            load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
                no_dist=True,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)


class TestDistributedStateDictSaveLoadWithSharedTensor(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def get_file_path(self) -> str:
        paths = [tempfile.mkdtemp()] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(paths)
        return paths[0]

    @with_comms(backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    @parametrize("extensions", [None, [Rot13Example()], [ZStandard()]])
    def test_read_write_shard_tensor(self, extensions) -> None:
        path = self.get_file_path()

        model_to_save = MyTestModule().to(self.device_type)
        fully_shard(model_to_save)
        state_dict_to_save = {"model": get_model_state_dict(model_to_save)}

        # Test save
        fs_writer = FileSystemWriter(path=path, _extensions=extensions)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        dist.barrier()

        # Create a new model
        model_to_load = MyTestModule().to(self.device_type)
        fully_shard(model_to_load)
        state_dict_to_load_to = {"model": get_model_state_dict(model_to_load)}

        dist.barrier()

        with self.assertRaises(AssertionError):
            assert_state_dict_equal(
                self, state_dict_to_load_to["model"], state_dict_to_save["model"]
            )

        # Test load.
        fs_reader = FileSystemReader(
            path=path, _extension_registry=get_test_extension_registry()
        )
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)
        set_model_state_dict(model_to_load, state_dict_to_load_to["model"])

        assert_state_dict_equal(
            self, get_model_state_dict(model_to_load), state_dict_to_save["model"]
        )
        dist.barrier()

    @with_comms(backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_save_load_bytes(self) -> None:
        path = self.get_file_path()

        state_dict_to_save = {"bytes0": [1], "bytes1": "string"}

        fs_writer = FileSystemWriter(path=path)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        state_dict_to_load = {"bytes0": [2], "bytes1": "other"}

        fs_reader = FileSystemReader(path=path)
        load_state_dict(state_dict=state_dict_to_load, storage_reader=fs_reader)

        self.assertEqual([1], state_dict_to_load["bytes0"])
        self.assertEqual("string", state_dict_to_load["bytes1"])

class TestDistributedStateDictSaveLoadWithCaching(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    @with_temp_dir
    def test_read_write_shard_tensor(self) -> None:
        model_to_save = MyTestModule().to(self.device_type)
        fully_shard(model_to_save)
        state_dict_to_save = {"model": get_model_state_dict(model_to_save)}

        for _ in range(2):
            # Test save
            fs_writer = FileSystemWriter(path=self.temp_dir)
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
                planner=DefaultSavePlanner(enable_plan_caching=True),
            )

            dist.barrier()

            # Create a new model
            model_to_load = MyTestModule().to(self.device_type)
            fully_shard(model_to_load)
            state_dict_to_load_to = {"model": get_model_state_dict(model_to_load)}

            dist.barrier()

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(
                    self, state_dict_to_load_to["model"], state_dict_to_save["model"]
                )

            # Test load.
            fs_reader = FileSystemReader(
                path=self.temp_dir, _extension_registry=get_test_extension_registry()
            )
            load_state_dict(
                state_dict=state_dict_to_load_to, storage_reader=fs_reader
            )
            set_model_state_dict(model_to_load, state_dict_to_load_to["model"])

            assert_state_dict_equal(
                self, get_model_state_dict(model_to_load), state_dict_to_save["model"]
            )
            dist.barrier()


instantiate_parametrized_tests(TestDistributedStateDictSaveLoadWithSharedTensor)

if __name__ == "__main__":
    run_tests()

# Owner(s): ["oncall: distributed"]

import sys
import tempfile
from typing import Any, IO

import torch
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
    load_state_dict,
    save,
    save_state_dict,
)
from torch.distributed.checkpoint._extension import ZStandard
from torch.distributed.checkpoint.stateful import Stateful
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed.checkpoint_utils import (
    get_test_extension_registry,
    Rot13Example,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


_THREAD_COUNTS = {1, 2}


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
        if isinstance(value_1, torch.Tensor):
            self.assertTrue(
                torch.equal(value_1, value_2),
                lambda msg: f"{msg}\nKey {key}'s tensor does not match",
            )
        elif isinstance(value_1, Stateful):
            self.assertEqual(value_1, value_2)

    return True


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


class BlobState:
    def __init__(self, value: IO[bytes]) -> Any:
        self.state = {"blob": value}

    def state_dict(self) -> dict[str, Any]:
        return self.state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.state = state_dict

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BlobState) and self.state == other.state

    def __repr__(self) -> str:
        return f"BlobState({self.state['blob']})"


class TestDistributedStateDictSaveLoad(TestCase):
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_only_tensor(self, thread_count) -> None:
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()

            fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
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


class TestDistributedStateDictSaveLoadRot13(TestCase):
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_tensor_and_blob(self, thread_count) -> None:
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()
            state_dict_to_save["test_blob"] = BlobState(b"SomeBlobForTesting")

            fs_writer = FileSystemWriter(
                path=path,
                thread_count=thread_count,
                _extensions=[Rot13Example()],
            )
            save(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
            )

            state_dict_to_load_to = MyTestModule().state_dict()
            state_dict_to_load_to["test_blob"] = BlobState(b"")

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding.  Note there is no extension
            # specification here; it is determined dynamically from the metadata.
            fs_reader = FileSystemReader(
                path=path, _extension_registry=get_test_extension_registry()
            )
            load(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)


class TestDistributedStateDictSaveLoadZStandard(TestCase):
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_only_tensor(self, thread_count) -> None:
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()
            state_dict_to_save["test_blob"] = BlobState(b"SomeBlobForTesting")

            fs_writer = FileSystemWriter(
                path=path,
                thread_count=thread_count,
                _extensions=[ZStandard()],
            )
            save(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
            )

            state_dict_to_load_to = MyTestModule().state_dict()
            state_dict_to_load_to["test_blob"] = BlobState(b"")

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding
            fs_reader = FileSystemReader(path=path)
            load(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)



instantiate_parametrized_tests(TestDistributedStateDictSaveLoad)
instantiate_parametrized_tests(TestDistributedStateDictSaveLoadRot13)
instantiate_parametrized_tests(TestDistributedStateDictSaveLoadZStandard)

if __name__ == "__main__":
    run_tests()

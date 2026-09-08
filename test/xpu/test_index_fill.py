# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestXpuIndexFill(TestCase):
    @parametrize("dtype", [torch.float32, torch.int64, torch.bool])
    @parametrize("dim", [0, 1])
    @parametrize("indices", [(0, 2), (-1, -3), (0, -2)])
    def test_index_fill(self, device, dtype, dim, indices):
        source = torch.zeros((4, 6), dtype=dtype)
        index = torch.tensor(indices)
        value = True if dtype == torch.bool else 3
        expected = source.index_fill(dim, index, value)

        actual = source.to(device)
        actual.index_fill_(dim, index.to(device), value)

        self.assertEqual(actual.cpu(), expected)

    def test_index_fill_noncontiguous(self, device):
        source = torch.arange(24, dtype=torch.float32).view(4, 6).t()
        index = torch.tensor([0, 2])
        expected = source.index_fill(1, index, -1)

        actual = source.to(device)
        actual.index_fill_(1, index.to(device), -1)

        self.assertEqual(actual.cpu(), expected)


instantiate_device_type_tests(
    TestXpuIndexFill, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()

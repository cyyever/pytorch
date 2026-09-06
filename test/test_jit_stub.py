import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestJitStub(TestCase):
    def test_is_tracing_is_false(self):
        self.assertFalse(torch.jit.is_tracing())


if __name__ == "__main__":
    run_tests()

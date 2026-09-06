import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class CompatibilityTests(TestCase):
    def test_is_compiling_alias(self):
        def function():
            if torch._dynamo.is_compiling():
                return torch.ones(2, 2)
            return torch.zeros(2, 2)

        compiled = torch.compile(function, backend="eager")

        self.assertEqual(function(), torch.zeros(2, 2))
        self.assertEqual(compiled(), torch.ones(2, 2))


if __name__ == "__main__":
    run_tests()

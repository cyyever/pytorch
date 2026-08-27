# Owner(s): ["oncall: mobile"]

import io
import tempfile
import unittest

import torch
import torch.utils.show_pickle
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


# Must be importable by name for torch.save to pickle it.
class MyCoolModule(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class TestShowPickle(TestCase):
    @unittest.skipIf(IS_WINDOWS, "Can't re-open temp file on Windows")
    def test_saved_model(self):
        m = MyCoolModule(torch.tensor([2.0]))

        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(m, tmp)
            tmp.flush()
            buf = io.StringIO()
            torch.utils.show_pickle.main(
                ["", tmp.name + "@*/data.pkl"], output_stream=buf
            )
            output = buf.getvalue()
            self.assertRegex(output, "MyCoolModule")
            self.assertRegex(output, "weight")


if __name__ == "__main__":
    run_tests()

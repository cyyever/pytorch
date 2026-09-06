import torch
from torch.onnx.symbolic_helper import parse_args
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOnnxStub(TestCase):
    def test_registration_allows_downstream_imports(self):
        @parse_args("v")
        def symbolic(graph):
            return graph

        self.assertIsNone(
            torch.onnx.register_custom_op_symbolic("downstream::op", symbolic, 11)
        )
        self.assertIsNone(
            torch.onnx.unregister_custom_op_symbolic("downstream::op", 11)
        )
        self.assertFalse(torch.onnx.is_in_onnx_export())

    def test_export_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "ONNX support is not included",
        ):
            torch.onnx.export(torch.nn.Identity(), (torch.ones(1),), "model.onnx")

    def test_symbolic_execution_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "ONNX support is not included",
        ):
            torch.onnx.symbolic_opset11.unsqueeze(None, None, 0)


if __name__ == "__main__":
    run_tests()

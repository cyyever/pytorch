# Owner(s): ["module: intel"]

import math

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def offset_contiguous(shape, dtype, offset):
    base = torch.empty(math.prod(shape) + offset, device="xpu", dtype=dtype)
    return base[offset:].view(shape)


def offset_channels_last(shape, dtype, offset):
    n, c, h, w = shape
    strides = (h * w * c, 1, w * c, c)
    required = offset + sum(
        (size - 1) * stride for size, stride in zip(shape, strides)
    )
    base = torch.empty(required + 1, device="xpu", dtype=dtype)
    return base.as_strided(shape, strides, offset)


class TestXpuMemoryAlignment(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_transpose_copy_unaligned(self, device, dtype):
        shape = (16, 64, 32, 32)
        src = offset_channels_last(shape, dtype, 1)
        dst = offset_contiguous(shape, dtype, 1)
        src.copy_(torch.randn(shape, device=device, dtype=dtype))

        self.assertNotEqual(src.data_ptr() % (4 * src.element_size()), 0)
        self.assertNotEqual(dst.data_ptr() % (4 * dst.element_size()), 0)
        dst.copy_(src)

        self.assertEqual(dst, src)

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_adaptive_avg_pool2d_unaligned_channels_last(self, device, dtype):
        shape = (8, 64, 32, 32)
        input = offset_channels_last(shape, dtype, 1)
        input.copy_(torch.randn(shape, device=device, dtype=dtype))

        self.assertNotEqual(input.data_ptr() % (4 * input.element_size()), 0)
        actual = F.adaptive_avg_pool2d(input, (7, 9))
        expected = F.adaptive_avg_pool2d(input.cpu(), (7, 9))

        self.assertEqual(actual.cpu(), expected)

    def test_index_select_alignment_dispatch(self, device):
        shape = (4096, 64)
        index = torch.randint(shape[0], (1024,), device=device)

        def kernel_names(input):
            with profile(activities=[ProfilerActivity.XPU]) as prof:
                torch.index_select(input, 0, index)
                torch.xpu.synchronize()
            return {event.key for event in prof.key_averages()}

        aligned = offset_contiguous(shape, torch.float16, 0)
        misaligned = offset_contiguous(shape, torch.float16, 1)

        self.assertEqual(aligned.data_ptr() % 16, 0)
        self.assertNotEqual(misaligned.data_ptr() % 16, 0)
        self.assertTrue(
            any("VectorizedGatherKernel<16" in name for name in kernel_names(aligned))
        )
        self.assertFalse(
            any(
                "VectorizedGatherKernel<16" in name
                for name in kernel_names(misaligned)
            )
        )


instantiate_device_type_tests(
    TestXpuMemoryAlignment, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()

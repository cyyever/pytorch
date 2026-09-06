from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FullyShardedDataParallelCompat,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFullyShardedDataParallelCompat(TestCase):
    def test_compatibility_module_exports_fsdp(self):
        self.assertIs(FullyShardedDataParallelCompat, FullyShardedDataParallel)


if __name__ == "__main__":
    run_tests()

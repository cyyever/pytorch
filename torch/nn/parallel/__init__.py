from warnings import deprecated

from torch.nn.parallel.data_parallel import data_parallel, DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter


__all__ = [
    "replicate",
    "scatter",
    "parallel_apply",
    "gather",
    "data_parallel",
    "DataParallel",
    "DistributedDataParallel",
]

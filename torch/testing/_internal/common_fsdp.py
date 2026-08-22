# mypy: allow-untyped-defs
# Owner(s): ["oncall: distributed"]

import contextlib
import os
import sys
import warnings
from collections.abc import Callable
from enum import auto, Enum
from functools import wraps
from typing import Any, cast, no_type_check
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable import checkpoint
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPParamGroup,
    RegisterPostBackwardFunction,
)
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    MultiThreadedTestCase,
    run_subtests,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    get_cycles_per_ms,
    set_rng_seed,
    TEST_CUDA,
    TEST_HPU,
    TEST_WITH_ROCM,
    TEST_XPU,
)
from torch.utils._triton import has_triton


if TEST_WITH_ROCM:
    DEVICE_COUNT = min(4, max(2, torch.cuda.device_count()))
else:
    DEVICE_COUNT = 4

if TEST_CUDA:
    DEVICE_TYPE = "cuda"
    DISTRIBUTED_BACKEND = "nccl"
    DEVICE_COUNT = torch.cuda.device_count()
elif TEST_HPU:
    DEVICE_TYPE = "hpu:0"
    DISTRIBUTED_BACKEND = "hccl"
elif TEST_XPU:
    DEVICE_TYPE = "xpu"
    DISTRIBUTED_BACKEND = "xccl"
    DEVICE_COUNT = torch.xpu.device_count()
else:
    DEVICE_TYPE = "cpu"
    DISTRIBUTED_BACKEND = "nccl"
    DEVICE_COUNT = 1


def get_devtype():
    return torch.device(DEVICE_TYPE)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        device: torch.device | None = None,
        *,
        bias: bool = True,
        with_buffer: bool = False,
        dim_multiplier: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim_multiplier * dim, device=device, bias=bias)
        self.out_proj = nn.Linear(dim_multiplier * dim, dim, device=device, bias=bias)
        if with_buffer:
            self.register_buffer("buffer", torch.randn((dim,), device=device))
        else:
            self.buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = F.relu(z)
        if self.buffer is not None:
            z = z + self.buffer
        return z

    def reset_parameters(self):
        if self.buffer is not None:
            torch.nn.init.normal_(self.buffer)


class MLPStack(nn.Sequential):
    def __init__(self, mlp_dim: int, *, with_seq_parallel: bool = False):
        modules: list[nn.Module] = [
            # Use multiplier of 3 to exercise uneven case
            MLP(mlp_dim, dim_multiplier=3),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=3),
        ]
        if with_seq_parallel:
            modules.append(nn.LayerNorm(mlp_dim, bias=False))
        super().__init__(*modules)
        self.with_seq_parallel = with_seq_parallel

    def parallelize(
        self,
        tp_mesh: DeviceMesh,
        dp_mesh: DeviceMesh,
        use_activation_checkpointing: bool,
        **fsdp_kwargs,
    ) -> MLPStack:
        parallelize_plan = {
            # Pass `use_local_output=False` to keep as DTensor to preserve
            # uneven activation dims
            "0.in_proj": ColwiseParallel(use_local_output=False),
            "0.out_proj": RowwiseParallel(use_local_output=False),
            "1.in_proj": ColwiseParallel(use_local_output=False),
            "1.out_proj": RowwiseParallel(use_local_output=False),
            "2.in_proj": ColwiseParallel(use_local_output=False),
            "2.out_proj": RowwiseParallel(output_layouts=Shard(1))
            if self.with_seq_parallel
            else RowwiseParallel(),
        }
        if self.with_seq_parallel:
            parallelize_plan["3"] = SequenceParallel(sequence_dim=1)
        parallelize_module(self, device_mesh=tp_mesh, parallelize_plan=parallelize_plan)
        for module in self:
            if isinstance(module, nn.LayerNorm):
                continue
            if use_activation_checkpointing:
                checkpoint(module)
            fully_shard(module, mesh=dp_mesh, **fsdp_kwargs)
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)
        return self


class DoubleLinear(nn.Module):
    """
    This can be used for returning multiple outputs from a module
    (``use_second_linear=True``) or for having an unused module (``False``).
    """

    def __init__(self, dim: int, use_second_linear: bool = True):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.use_second_linear = use_second_linear

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.use_second_linear:
            return self.relu(self.lin1(x)), self.relu(self.lin2(x))
        return self.relu(self.lin1(x))


# NOTE: For these patch methods, if we want safety under multi-threading (e.g.
# when using multi-threaded process group), then we want:
# (1) a barrier immediately after reading the original value to ensure that all
# threads see the same original value
# (2) a barrier immediately before restoring the original value to ensure that
# all threads use the patched value inside the context
@contextlib.contextmanager
def patch_all_gather(new_all_gather_into_tensor: Callable):
    orig_all_gather = dist.all_gather_single
    dist.barrier()
    dist.all_gather_single = new_all_gather_into_tensor
    try:
        yield
    finally:
        dist.barrier()
        dist.all_gather_single = orig_all_gather


@contextlib.contextmanager
def patch_foreach_all_gather(new_foreach_all_gather: Callable):
    orig_foreach_all_gather = (
        torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_all_gather
    )
    dist.barrier()
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_all_gather = (
        new_foreach_all_gather
    )
    try:
        yield
    finally:
        dist.barrier()
        torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_all_gather = (
            orig_foreach_all_gather
        )


@contextlib.contextmanager
def patch_foreach_reduce(new_foreach_reduce: Callable):
    orig_foreach_foreach_reduce = (
        torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce
    )
    dist.barrier()
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce = (
        new_foreach_reduce
    )
    try:
        yield
    finally:
        dist.barrier()
        torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce = (
            orig_foreach_foreach_reduce
        )


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter_tensor: Callable):
    orig_reduce_scatter = dist.reduce_scatter_single
    dist.barrier()
    dist.reduce_scatter_single = new_reduce_scatter_tensor
    try:
        yield
    finally:
        dist.barrier()
        dist.reduce_scatter_single = orig_reduce_scatter


@contextlib.contextmanager
def patch_all_reduce(new_all_reduce: Callable):
    orig_all_reduce = dist.all_reduce
    dist.barrier()
    dist.all_reduce = new_all_reduce
    try:
        yield
    finally:
        dist.barrier()
        dist.all_reduce = orig_all_reduce


@no_type_check
@contextlib.contextmanager
def patch_unshard(new_unshard: Callable):
    orig_unshard = FSDPParamGroup.unshard
    dist.barrier()
    FSDPParamGroup.unshard = new_unshard
    try:
        yield
    finally:
        dist.barrier()
        FSDPParamGroup.unshard = orig_unshard


@no_type_check
@contextlib.contextmanager
def patch_reshard(new_reshard: Callable):
    orig_reshard = FSDPParamGroup.reshard
    dist.barrier()
    FSDPParamGroup.reshard = new_reshard
    try:
        yield
    finally:
        dist.barrier()
        FSDPParamGroup.reshard = orig_reshard


@no_type_check
@contextlib.contextmanager
def patch_post_backward(new_post_backward: Callable):
    orig_post_backward = FSDPParamGroup.post_backward
    dist.barrier()
    FSDPParamGroup.post_backward = new_post_backward
    try:
        yield
    finally:
        dist.barrier()
        FSDPParamGroup.post_backward = orig_post_backward


@no_type_check
@contextlib.contextmanager
def patch_register_post_backward_hook_backward(new_backward: Callable):
    orig_backward = RegisterPostBackwardFunction.backward
    dist.barrier()
    RegisterPostBackwardFunction.backward = new_backward
    try:
        yield
    finally:
        dist.barrier()
        RegisterPostBackwardFunction.backward = orig_backward


def reduce_scatter_with_assert(
    cls,
    orig_reduce_scatter: Callable,
    assert_fn: Callable,  # `assert_fn(output: Tensor)`
    *args: Any,
    **kwargs: Any,
):
    if len(args) > 0:
        output = args[0]
    elif "output" in kwargs:
        output = kwargs["output"]
    else:
        raise AssertionError(
            f"Cannot get reduce-scatter output from\nargs: {args}\nkwargs: {kwargs}"
        )
    assert_fn(output)
    return orig_reduce_scatter(*args, **kwargs)


def check_sharded_parity(
    cls,  # unit test class
    replicated_module: nn.Module,
    sharded_module: nn.Module,
    prefixes_to_ignore: tuple[str, ...] = (),
):
    for (replicated_name, replicated_param), (sharded_name, sharded_param) in zip(
        replicated_module.named_parameters(),
        sharded_module.named_parameters(),
        strict=True,
    ):
        clean_sharded_name = sharded_name
        for prefix in prefixes_to_ignore:
            clean_sharded_name = clean_sharded_name.replace(prefix, "")
        cls.assertEqual(replicated_name, clean_sharded_name)
        cls.assertIsInstance(sharded_param, DTensor)
        if not isinstance(sharded_param, DTensor):
            raise AssertionError("Expected sharded_param to be a DTensor")  # mypy
        mesh, placements = sharded_param.device_mesh, sharded_param.placements
        if tuple(placements) == (Shard(0), Shard(0)):
            raise AssertionError(
                "FSDP's (Shard(0), Shard(0)) layout differs from distribute_tensor(), "
                "so we cannot check for equality using it"
            )
        sharded_ref_param = distribute_tensor(replicated_param, mesh, placements)
        cls.assertEqual(sharded_param.to_local(), sharded_ref_param.to_local())
        if replicated_param.grad is None:
            cls.assertIsNone(sharded_param.grad)
            continue
        cls.assertIsNotNone(sharded_param.grad)
        sharded_ref_grad = distribute_tensor(replicated_param.grad, mesh, placements)
        cls.assertIsInstance(sharded_param.grad, DTensor)
        if not isinstance(sharded_param.grad, DTensor):
            raise AssertionError("Expected sharded_param.grad to be a DTensor")  # mypy
        cls.assertEqual(sharded_param.grad.to_local(), sharded_ref_grad.to_local())


class FSDPTestMultiThread(MultiThreadedTestCase):
    @property
    def world_size(self):
        return DEVICE_COUNT

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)

    def perThreadSetUp(self):
        torch._dynamo.reset()

    def perThreadTearDown(self):
        torch._dynamo.reset()


class FSDPTestMixin:
    """
    Mixin class containing shared test utilities for FSDP tests.
    Provides common helper methods for both FSDPTest and FSDPTestContinuous.
    """

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe, **kwargs):  # type: ignore[override]
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        fake_pg = kwargs.get("fake_pg", False)

        print(f"dist init r={self.rank}, world={self.world_size}")
        if DEVICE_TYPE != "cpu" and torch.accelerator.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-device-{self.world_size}"].exit_code)

        # Specify gloo backend to make 'init_process_group()' succeed,
        # Actual tests will be skipped if there are not enough GPUs.
        try:
            if fake_pg:
                store = torch.testing._internal.distributed.fake_pg.FakeStore()
                dist.init_process_group(
                    backend="fake",
                    world_size=self.world_size,
                    rank=rank,
                    store=store,
                )
            else:
                dist.init_process_group(
                    init_method=self.init_method,
                    backend=DISTRIBUTED_BACKEND,
                    world_size=int(self.world_size),
                    rank=self.rank,
                )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        device_ids = None
        device_id = self.rank % DEVICE_COUNT
        if TEST_CUDA or TEST_XPU:
            torch.accelerator.set_device_index(device_id)
        device_ids = [device_id]

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier(device_ids=device_ids)

        torch._dynamo.reset()
        set_rng_seed()
        self.run_test(test_name, pipe)
        torch._dynamo.reset()

        dist.barrier(device_ids=device_ids)

        dist.destroy_process_group()

class FSDPTest(FSDPTestMixin, MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # Set TORCH_NCCL_DESYNC_DEBUG=0 to disable the NCCL `workCleanupLoop()`,
        # which can cause unit test flakiness:
        # https://github.com/pytorch/pytorch/issues/90848
        os.environ["TORCH_NCCL_DESYNC_DEBUG"] = "0"
        self._spawn_processes()

    @property
    def world_size(self):
        return DEVICE_COUNT

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def destroy_pg_upon_exit(self) -> bool:
        # Overriding base test class: do not auto destroy PG upon exit.
        return False

    @property
    def init_method(self):
        return f"{FILE_SCHEMA}{self.file_name}"

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe, **kwargs):  # type: ignore[override]
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        fake_pg = kwargs.get("fake_pg", False)

        print(f"dist init r={self.rank}, world={self.world_size}")
        if torch.accelerator.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-device-{self.world_size}"].exit_code)

        # Specify gloo backend to make 'init_process_group()' succeed,
        # Actual tests will be skipped if there are not enough GPUs.
        try:
            if fake_pg:
                store = torch.testing._internal.distributed.fake_pg.FakeStore()
                dist.init_process_group(
                    backend="fake",
                    world_size=self.world_size,
                    rank=rank,
                    store=store,
                )
            else:
                dist.init_process_group(
                    init_method=self.init_method,
                    backend=DISTRIBUTED_BACKEND,
                    world_size=int(self.world_size),
                    rank=self.rank,
                )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        device_ids = None
        device_id = self.rank % DEVICE_COUNT
        if TEST_CUDA or TEST_XPU:
            torch.accelerator.set_device_index(device_id)
        device_ids = [device_id]

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier(device_ids=device_ids)

        torch._dynamo.reset()
        set_rng_seed()
        self.run_test(test_name, pipe)
        torch._dynamo.reset()

        dist.barrier(device_ids=device_ids)

        dist.destroy_process_group()


class FSDPTestContinuous(FSDPTestMixin, MultiProcContinuousTest):
    """
    FSDP test base class using MultiProcContinuousTest for faster test execution.
    This class reuses worker processes across tests, reducing process spawn overhead.
    Use this for tests that don't require fresh process state between tests.
    """

    world_size: int = DEVICE_COUNT

    @classmethod
    def backend_str(cls) -> str:
        return DISTRIBUTED_BACKEND

    @classmethod
    def device_type(cls) -> str:
        return DEVICE_TYPE

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # Set TORCH_NCCL_DESYNC_DEBUG=0 to disable the NCCL `workCleanupLoop()`,
        # which can cause unit test flakiness:
        # https://github.com/pytorch/pytorch/issues/90848
        os.environ["TORCH_NCCL_DESYNC_DEBUG"] = "0"

        if torch.accelerator.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-device-{world_size}"].exit_code)

        device_id = rank % DEVICE_COUNT
        if TEST_CUDA or TEST_XPU:
            torch.accelerator.set_device_index(device_id)

        super()._init_pg(rank, world_size, rdvz_file)

    def setUp(self):
        super().setUp()
        # Barrier to synchronize workers before test, similar to FSDPTest._run().
        # This ensures all workers start the test together and prevents NCCL
        # collective mismatches when the process group is reused across tests.
        if self.rank != self.MAIN_PROCESS_RANK:
            dist.barrier()
        torch._dynamo.reset()
        set_rng_seed()

    def tearDown(self):
        # Barrier to synchronize workers after test, similar to FSDPTest._run().
        if self.rank != self.MAIN_PROCESS_RANK:
            dist.barrier()
        super().tearDown()
        torch._dynamo.reset()

    @property
    def process_group(self):
        return self.__class__.pg


def compiled_fsdp_test(compile_compute_on_module: type | None = None):
    def fully_shard_with_compiled_compute(*args, **kwargs):
        torch.distributed.fsdp.fully_shard(*args, **kwargs)  # type: ignore[operator]
        if compile_compute_on_module is None or isinstance(
            args[0], compile_compute_on_module
        ):
            args[0].compile()

    class FullyShardMode(Enum):
        EAGER = auto()
        COMPILED_COMPUTE = auto()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_fully_shard: Any = torch.distributed.fsdp.fully_shard
            for mode in FullyShardMode:
                if mode != FullyShardMode.EAGER and not has_triton():
                    warnings.warn(
                        "Inductor on GPU needs Triton and recent GPU arch", stacklevel=2
                    )
                    continue
                # barrier to ensure thread reading the same value
                original_compile_threads = torch._inductor.config.compile_threads
                torch.distributed.barrier()

                if mode == FullyShardMode.EAGER:
                    fully_shard_patch = original_fully_shard
                elif mode == FullyShardMode.COMPILED_COMPUTE:
                    torch._inductor.config.compile_threads = 1
                    fully_shard_patch = fully_shard_with_compiled_compute  # type: ignore[assignment]
                else:
                    raise NotImplementedError(
                        f"Need to implement FullyShardMode={mode}"
                    )

                # fully_shard is imported as a global
                # through `from ... import fully_shard`
                func.__globals__[original_fully_shard.__name__] = fully_shard_patch
                func(*args, **kwargs)
                # other threads use patched func before this thread restores
                torch.distributed.barrier()
                func.__globals__[original_fully_shard.__name__] = original_fully_shard
                torch._inductor.config.compile_threads = original_compile_threads

        return wrapper

    return decorator

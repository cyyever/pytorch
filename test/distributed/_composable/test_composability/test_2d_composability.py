# Owner(s): ["oncall: distributed"]

import copy
import functools
import unittest

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable import replicate
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
    skip_if_rocm_arch_multiprocess,
)
from torch.testing._internal.common_fsdp import FSDPTestContinuous, MLP, MLPStack
from torch.testing._internal.common_utils import (
    IS_LINUX,
    MI200_ARCH,
    run_tests,
    TEST_WITH_ROCM,
    TEST_XPU,
    xfailIf,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorContinuousTestBase,
    ModelArgs,
    Transformer,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)
curr_backend = dist.get_default_backend_for_device(device_type)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(8, 4)
        self.net3 = nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x

    def get_input(self):
        return torch.rand(4, 5, device=device_type)


class SimpleModelUneven(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = self.net4(x)
        return x

    def get_input(self):
        return torch.rand(4, 5, device=device_type)


class TestFullyShard2DTraining(FSDPTestContinuous):
    global c10d_ops
    global funcol
    c10d_ops = torch.ops.c10d
    funcol = torch.ops.c10d_functional

    world_size = 4

    def init_global_mesh(self) -> DeviceMesh:
        # Prefer to test with >=4 GPUs, but for 2 GPUs, use 2-way TP
        dp_size = 2 if self.world_size > 2 else 1
        return init_device_mesh(
            device_type,
            (dp_size, self.world_size // dp_size),
            mesh_dim_names=("dp", "tp"),
        )

    @skip_if_rocm_arch_multiprocess(MI200_ARCH)
    @skip_if_lt_x_gpu(4)
    def test_train_parity_2d_mlp(self):
        global_mesh = self.init_global_mesh()
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 16, 17],
            },
            functools.partial(self._test_train_parity_2d_mlp, global_mesh),
        )

    def _test_train_parity_2d_mlp(
        self,
        global_mesh: DeviceMesh,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
    ):
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        dp_pg = dp_mesh.get_group()  # used for `replicate()`

        torch.manual_seed(42)
        model = MLPStack(mlp_dim)
        ref_model = copy.deepcopy(model).to(device_type)
        replicate(ref_model, device_ids=[self.rank], process_group=dp_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)
        model.parallelize(
            tp_mesh,
            dp_mesh,
            use_activation_checkpointing,
            reshard_after_forward=reshard_after_forward,
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        torch.manual_seed(42 + dp_pg.rank() + 1)
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=device_type)
            losses: list[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(4)
    @xfailIf(TEST_XPU)  # https://github.com/intel/torch-xpu-ops/issues/1881
    def test_train_parity_2d_transformer(self):
        self.run_subtests(
            {"use_shard_placement_fn": [False, True]},
            self._test_train_parity_2d_transformer,
        )

    def _test_train_parity_2d_transformer(self, use_shard_placement_fn: bool):
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=3, dropout_p=0.0)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)

        dp_size, tp_size = self.world_size // 2, 2
        global_mesh = init_device_mesh(
            device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        model = Transformer.parallelize(model, global_mesh["tp"], use_seq_parallel=True)

        def _shard_placement_fn(param: nn.Parameter) -> Shard | None:
            if isinstance(param, DTensor):
                for placement in param.placements:
                    if isinstance(placement, Shard):
                        shard_dim = param.ndim - 1 - placement.dim
                        if not (shard_dim >= 0):
                            raise AssertionError(
                                f"Expected shard_dim >= 0, got {shard_dim} for {param.shape}"
                            )
                        return Shard(shard_dim)
            return Shard(0)

        shard_placement_fn = _shard_placement_fn if use_shard_placement_fn else None
        for layer in model.layers:
            fully_shard(
                layer, mesh=global_mesh["dp"], shard_placement_fn=shard_placement_fn
            )
        fully_shard(
            model, mesh=global_mesh["dp"], shard_placement_fn=shard_placement_fn
        )
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            full_param = param.full_tensor()
            self.assertEqual(full_param, ref_param)

        torch.manual_seed(42 + global_mesh.get_local_rank("dp"))
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device=device_type)
        for _ in range(5):
            ref_loss = ref_model(inp).sum()
            loss = model(inp).sum()
            self.assertEqual(ref_loss, loss)
            ref_loss.backward()
            loss.backward()
            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(
                        param.grad,
                        group=global_mesh.get_group("dp"),
                        op=dist.ReduceOp.AVG,
                    )

            # Specially check the TP placement for `pos_embeddings.weight` and
            # its which since the grad naturally has replicate placement,
            # requiring FSDP to redistribute it to shard placement before FSDP
            # runs its reduce-scatter
            self.assertIsInstance(model.pos_embeddings.weight.placements[1], Shard)
            self.assertIsInstance(model.pos_embeddings.weight.grad.placements[1], Shard)
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                full_grad = param.grad.full_tensor()
                self.assertEqual(ref_param.grad, full_grad)

            ref_optim.step()
            optim.step()
            ref_optim.zero_grad()
            optim.zero_grad()

        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            full_param = param.full_tensor()
            self.assertEqual(full_param, ref_param)

    @skip_if_lt_x_gpu(4)
    @xfailIf(TEST_XPU)  # https://github.com/pytorch/pytorch/issues/156782
    def test_tp_with_fsdp_offloading(self):
        global_mesh = init_device_mesh(
            device_type, (1, self.world_size), mesh_dim_names=("dp", "tp")
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        torch.manual_seed(42)
        mlp_dim = 16
        model = MLPStack(mlp_dim)
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)
        # Parallelize with N-way TP and 1-way FSDP
        model.parallelize(
            tp_mesh,
            dp_mesh,
            use_activation_checkpointing=False,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy(),
        )
        for param in model.parameters():
            self.assertEqual(param.device.type, "cpu")
        num_mlps = sum(isinstance(module, MLP) for module in model.modules())
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        # NOTE: We still see the FSDP all-gather/reduce-scatter c10d ops
        # called, but they will just be no-ops without issuing any kernels.
        # We prefer to keep the no-op check at the c10d level, not in FSDP.
        inp = torch.randn((4, mlp_dim), device=device_type)  # same on all ranks
        for _ in range(10):
            ref_optim.zero_grad()
            optim.zero_grad()

            with CommDebugMode() as fwd_comm_mode:
                loss = model(inp).sum()

            fwd_comm_counts = fwd_comm_mode.get_comm_counts()
            self.assertEqual(len(fwd_comm_counts), 1)
            self.assertEqual(fwd_comm_counts[funcol.all_reduce], num_mlps)
            self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], 0)
            ref_loss = ref_model(inp).sum()
            self.assertEqual(loss, ref_loss)

            with CommDebugMode() as bwd_comm_mode:
                loss.backward()
            bwd_comm_counts = bwd_comm_mode.get_comm_counts()
            self.assertEqual(len(bwd_comm_counts), 1)
            # First MLP's input gradient does not need to be all-reduced
            self.assertEqual(bwd_comm_counts[funcol.all_reduce], num_mlps - 1)
            self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], 0)
            self.assertEqual(bwd_comm_counts[c10d_ops._reduce_scatter_base_], 0)
            ref_loss.backward()

            optim.step()
            ref_optim.step()

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM, "https://github.com/pytorch/pytorch/issues/125644"
    )
    @skip_if_lt_x_gpu(4)
    @xfailIf(TEST_XPU)  # https://github.com/intel/torch-xpu-ops/issues/1881
    @with_temp_dir
    def test_train_parity_2d_transformer_checkpoint_resume(self):
        """
        Tests train parity of a 2D transformer without checkpointing against a
        2D transformer with a checkpoint save/load.
        """
        self.run_subtests(
            {
                "use_seq_parallel": [False, True],
                # If reusing, then load into the same model/optimizer instance
                # else construct new ones (requiring eager optim state init)
                "reuse_model_optim": [False, True],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
                # TODO: need to update `parallelize` before including foreach=True for testing
                "foreach": [False],
            },
            self._test_train_parity_2d_transformer_checkpoint_resume,
        )

    def _test_train_parity_2d_transformer_checkpoint_resume(
        self,
        use_seq_parallel: bool,
        reuse_model_optim: bool,
        optimizer_class: type[torch.optim.Optimizer],
        foreach: bool,
    ):
        def train_step(
            _model: nn.Module, _optim: torch.optim.Optimizer, _inp: torch.Tensor
        ) -> torch.Tensor:
            loss = _model(_inp).sum()
            loss.backward()
            _optim.step()
            _optim.zero_grad()
            return loss

        def parallelize(_model: Transformer, mesh: DeviceMesh, use_seq_parallel: bool):
            _model = Transformer.parallelize(_model, mesh["tp"], use_seq_parallel)
            for layer in _model.layers:
                fully_shard(layer, mesh=mesh["dp"])
            fully_shard(_model, mesh=mesh["dp"])
            return _model

        global_mesh = self.init_global_mesh()
        # Baseline: run two iterations without checkpointing
        seed = 42
        torch.manual_seed(seed)
        model_args = ModelArgs(dropout_p=0.0)
        model_no_cp = parallelize(
            Transformer(model_args), global_mesh, use_seq_parallel
        )
        optim_no_cp = optimizer_class(
            model_no_cp.parameters(), lr=1e-2, foreach=foreach
        )

        torch.manual_seed(42 + global_mesh["dp"].get_local_rank() + 1)
        inp = torch.randint(0, model_args.vocab_size, (3, 16), device=device_type)
        loss_no_cp1 = train_step(model_no_cp, optim_no_cp, inp)
        loss_no_cp2 = train_step(model_no_cp, optim_no_cp, inp)

        # Test: run one iteration, save checkpoint, zero states or init new
        # model/optimizer, load checkpoint, and run another iteration
        torch.manual_seed(seed)
        model_cp = parallelize(Transformer(model_args), global_mesh, use_seq_parallel)
        optim_cp = optimizer_class(model_cp.parameters(), lr=1e-2, foreach=foreach)

        loss_cp1 = train_step(model_cp, optim_cp, inp)
        self.assertEqual(loss_no_cp1, loss_cp1)

        sharded_sd = {
            "model": get_model_state_dict(model_cp),
            # Use `get_optimizer_state_dict` to handle eager optim state init
            # when constructing a new optimizer instance
            "optim": get_optimizer_state_dict(model_cp, optim_cp),
        }
        dcp.save(
            state_dict=sharded_sd,
            storage_writer=dcp.FileSystemWriter(self.temp_dir),
        )
        if reuse_model_optim:
            with torch.no_grad():
                for param in model_cp.parameters():
                    param.zero_()
                optim_sd = optim_cp.state_dict()
                for param_states in optim_sd["state"].values():
                    for state_value in param_states.values():
                        if torch.is_tensor(state_value):
                            state_value.zero_()
        else:
            torch.manual_seed(seed + 1)  # different seed
            model_cp = parallelize(
                Transformer(model_args), global_mesh, use_seq_parallel
            )
            optim_cp = optimizer_class(model_cp.parameters(), lr=1e-2, foreach=foreach)
        self.assertNotEqual(loss_no_cp2, train_step(model_cp, optim_cp, inp))

        sharded_sd = {
            "model": get_model_state_dict(model_cp),
            "optim": get_optimizer_state_dict(model_cp, optim_cp),
        }
        dcp.load(
            state_dict=sharded_sd,
            storage_reader=dcp.FileSystemReader(self.temp_dir),
        )
        self.assertGreater(len(optim_cp.state_dict()["state"]), 0)

        loss_cp2 = train_step(model_cp, optim_cp, inp)
        self.assertEqual(loss_no_cp2, loss_cp2)


class TestFullyShard2DStateDict(DTensorContinuousTestBase):
    world_size = 4

    @property
    def backend(self):
        # need to specify gloo backend for testing cpu offload
        return f"cpu:fake,{device_type}:{curr_backend}"

    @skip_if_lt_x_gpu(4)
    def test_fully_shard_tp_2d_set_full_state_dict(self):
        dummy_model = SimpleModel().to(device_type)
        mesh_2d = init_device_mesh(
            device_type,
            (2, self.world_size // 2),
            mesh_dim_names=("dp", "tp"),
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
            "net3": ColwiseParallel(),
        }
        model = parallelize_module(dummy_model, tp_mesh, parallelize_plan)
        fully_shard(model, mesh=dp_mesh)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        model(model.get_input()).sum().backward()
        optim.step()
        # ref_msd, ref_osd are both the default sharded state dict
        ref_msd = copy.deepcopy(get_model_state_dict(model))
        ref_osd = copy.deepcopy(get_optimizer_state_dict(model, optimizers=optim))

        options = StateDictOptions(
            full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
        )
        full_msd = get_model_state_dict(model, options=options)
        full_osd = get_optimizer_state_dict(model, optimizers=optim, options=options)
        # load full_msd and full_osd into model and optim.
        # this loads the slice of full tensor into each rank's local DTensor.
        set_model_state_dict(model, full_msd, options=options)
        set_optimizer_state_dict(
            model, optimizers=optim, optim_state_dict=full_osd, options=options
        )

        # check after setting full state dict, the model and optim default sharded state dict
        # are the same as the initial default sharded state dict.
        new_msd = get_model_state_dict(model)
        new_osd = get_optimizer_state_dict(model, optimizers=optim)
        self.assertEqual(ref_msd, new_msd)
        self.assertEqual(ref_osd, new_osd)


if __name__ == "__main__":
    run_tests()

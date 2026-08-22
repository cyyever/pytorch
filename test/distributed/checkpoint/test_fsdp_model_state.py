# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class FsdpModelStateCheckpoint(DTensorTestBase):
    @property
    def backend(self):
        curr_backend = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:fake,{self.device_type}:{curr_backend}"

    def _test_fsdp_model_state(self, reshard: bool) -> None:
        CHECKPOINT_DIR = self.temp_dir

        def make_model(subgroup_sharding: bool):
            model = torch.nn.Linear(8, 8, device=self.device_type)
            if subgroup_sharding:
                mesh_2d = init_device_mesh(
                    self.device_type,
                    (2, self.world_size // 2),
                    mesh_dim_names=("dp_rep", "dp_shard"),
                )
                fully_shard(model, mesh=mesh_2d["dp_shard"])
            else:
                fully_shard(model)
            return model

        model = make_model(subgroup_sharding=reshard)
        msd = get_model_state_dict(model)
        dist_cp.save(
            state_dict={"model": msd},
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # Load the checkpoint into a model sharded differently from the saver.
        model_2 = make_model(subgroup_sharding=not reshard)
        msd_2 = get_model_state_dict(model_2)
        dist_cp.load(
            state_dict={"model": msd_2},
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )
        set_model_state_dict(model_2, msd_2)

        self.assertEqual(get_model_state_dict(model), get_model_state_dict(model_2))

    @skip_if_lt_x_gpu(2)
    @with_comms
    @with_temp_dir
    def test_fsdp_model_state_no_resharding(self):
        self._test_fsdp_model_state(reshard=False)

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    def test_fsdp_model_state_with_resharding(self):
        self._test_fsdp_model_state(reshard=True)


if __name__ == "__main__":
    run_tests()

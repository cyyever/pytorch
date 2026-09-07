import subprocess
import sys
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_AMD = REPO_ROOT / "tools/amd_build/build_amd.py"


class TestBuildAMD(TestCase):
    def test_default_output_is_staged_and_refreshed(self) -> None:
        scratch_root = REPO_ROOT / "agent_space"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            project = Path(directory)
            cuda_source = project / "torch/csrc/cuda/Example.cpp"
            unchanged_api_header = (
                project / "torch/csrc/api/include/torch/types.h"
            )
            hipified_api_header = (
                project / "torch/csrc/api/include/torch/cuda.h"
            )
            c10_source = project / "c10/cuda/CUDAFunctions.h"
            hip_cmake = project / "c10/hip/CMakeLists.txt"
            mslk_source = (
                project
                / "third_party/mslk/include/mslk/utils/tuning_cache.cuh"
            )
            ignored_source = project / "build/old/torch/csrc/cuda/Ignored.cpp"

            for path in (
                cuda_source,
                unchanged_api_header,
                hipified_api_header,
                c10_source,
                hip_cmake,
                mslk_source,
                ignored_source,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
            cuda_source.write_text("#include <cuda_runtime_api.h>\n")
            unchanged_api_header.write_text("#pragma once\n")
            hipified_api_header.write_text("#include <cuda_runtime_api.h>\n")
            c10_source.write_text("#include <cuda_runtime_api.h>\n")
            hip_cmake.write_text("# handwritten HIP build file\n")
            mslk_source.write_text("#include <cuda_runtime_api.h>\n")
            ignored_source.write_text("#include <cuda_runtime_api.h>\n")

            original_cuda = cuda_source.read_text()
            original_mslk = mslk_source.read_text()

            def run_hipify(output: Path | None = None) -> None:
                command = [
                    sys.executable,
                    str(BUILD_AMD),
                    "--project-directory",
                    str(project),
                ]
                if output is not None:
                    command.extend(["--output-directory", str(output)])
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            run_hipify()
            output = project / "build/hipify"
            self.assertTrue((output / ".pytorch-hipify.json").is_file())
            self.assertTrue((output / "torch/csrc/cuda/Example.cpp").is_file())
            self.assertFalse(
                (output / "torch/csrc/api/include/torch/types.h").exists()
            )
            self.assertTrue(
                (output / "torch/csrc/api/include/torch/cuda.h").is_file()
            )
            self.assertTrue((output / "c10/hip/HIPFunctions.h").is_file())
            self.assertEqual(
                (output / "c10/hip/CMakeLists.txt").read_text(),
                hip_cmake.read_text(),
            )
            self.assertTrue(
                (
                    output
                    / "third_party/mslk/include/mslk/utils/tuning_cache_hip.cuh"
                ).is_file()
            )
            self.assertFalse(
                (output / "build/old/torch/csrc/cuda/Ignored.cpp").exists()
            )
            self.assertEqual(cuda_source.read_text(), original_cuda)
            self.assertEqual(mslk_source.read_text(), original_mslk)
            self.assertFalse(
                (
                    project
                    / "third_party/mslk/include/mslk/utils/tuning_cache_hip.cuh"
                ).exists()
            )

            stale = output / "stale"
            stale.write_text("stale\n")
            run_hipify()
            self.assertFalse(stale.exists())

            explicit_output = project / "explicit-hipify"
            run_hipify(explicit_output)
            self.assertTrue((explicit_output / ".pytorch-hipify.json").is_file())
            self.assertTrue(
                (explicit_output / "c10/hip/HIPFunctions.h").is_file()
            )


if __name__ == "__main__":
    run_tests()

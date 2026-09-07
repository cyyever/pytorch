#!/usr/bin/env python3


import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# NOTE: `tools/amd_build/build_amd.py` could be a symlink.
# The behavior of `symlink / '..'` is different from `symlink.parent`.
# Use `pardir` three times rather than using `path.parents[2]`.
REPO_ROOT = (
    Path(__file__).absolute() / os.path.pardir / os.path.pardir / os.path.pardir
).resolve()
sys.path.append(str(REPO_ROOT / "torch" / "utils"))

from hipify import hipify_python  # type: ignore[import]


INCLUDES = [
    "caffe2/operators/*",
    "caffe2/sgd/*",
    "caffe2/image/*",
    "caffe2/transforms/*",
    "caffe2/video/*",
    "caffe2/distributed/*",
    "caffe2/queue/*",
    "caffe2/contrib/aten/*",
    "binaries/*",
    "caffe2/**/*_test*",
    "caffe2/db/*",
    "caffe2/contrib/nccl/*",
    "c10/cuda/*",
    "c10/cuda/test/CMakeLists.txt",
    "modules/*",
    "third_party/nvfuser/*",
    # PyTorch paths
    # Keep this synchronized with is_pytorch_file in hipify_python.py
    "aten/src/ATen/cuda/*",
    "aten/src/ATen/native/cuda/*",
    "aten/src/ATen/native/cudnn/*",
    "aten/src/ATen/native/quantized/cudnn/*",
    "aten/src/ATen/native/nested/cuda/*",
    "aten/src/ATen/native/sparse/cuda/*",
    "aten/src/ATen/native/quantized/cuda/*",
    "aten/src/ATen/native/transformers/cuda/attention_backward.cu",
    "aten/src/ATen/native/transformers/cuda/attention.cu",
    "aten/src/ATen/native/transformers/cuda/sdp_utils.cpp",
    "aten/src/ATen/native/transformers/cuda/sdp_utils.h",
    "aten/src/ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h",
    "aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h",
    "aten/src/ATen/native/transformers/cuda/mem_eff_attention/pytorch_utils.h",
    "aten/src/ATen/test/*",
    "torch/*",
    "tools/autograd/templates/python_variable_methods.cpp",
    "torch/csrc/stable/*",
    "test/cpp/c10d/*",
]

IGNORES = [
    "caffe2/operators/depthwise_3x3_conv_op_cudnn.cu",
    "caffe2/operators/pool_op_cudnn.cu",
    "*/hip/*",
    # These files are compatible with both cuda and hip
    "aten/src/ATen/core/*",
    # Correct path to generate HIPConfig.h:
    #   CUDAConfig.h.in -> (amd_build) HIPConfig.h.in -> (cmake) HIPConfig.h
    "aten/src/ATen/cuda/CUDAConfig.h",
    "third_party/nvfuser/csrc/codegen.cpp",
    "third_party/nvfuser/runtime/block_reduction.cu",
    "third_party/nvfuser/runtime/block_sync_atomic.cu",
    "third_party/nvfuser/runtime/block_sync_default_rocm.cu",
    "third_party/nvfuser/runtime/broadcast.cu",
    "third_party/nvfuser/runtime/grid_reduction.cu",
    "third_party/nvfuser/runtime/helpers.cu",
    # generated files we shouldn't frob
    "torch/lib/tmp_install/*",
    "torch/include/*",
]

EXTRA_FILES = [
    "torch/_inductor/codegen/cuda/device_op_overrides.py",
    "torch/_inductor/codegen/cpp_wrapper_cpu.py",
    "torch/_inductor/codegen/cpp_wrapper_gpu.py",
    "torch/_inductor/codegen/wrapper.py",
]

HIPIFY_EXTENSIONS = (".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".in", ".hpp")
HIP_SOURCE_PREFIXES = (
    "aten/src/ATen/hip",
    "aten/src/ATen/native/cudnn/hip",
    "aten/src/ATen/native/hip",
    "aten/src/ATen/native/nested/hip",
    "aten/src/ATen/native/sparse/hip",
    "aten/src/ATen/native/transformers/hip",
    "c10/hip",
)
AUTOGRAD_PREFIX = "tools/autograd"
MANIFEST = ".pytorch-hipify.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top-level script for HIPifying, filling in most common parameters"
    )
    parser.add_argument(
        "--out-of-place-only",
        action="store_true",
        help="Whether to only run hipify out-of-place on source files",
    )
    parser.add_argument(
        "--project-directory",
        type=str,
        default="",
        help="The root of the project.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default="",
        help="The directory to store the hipified source tree",
    )
    parser.add_argument(
        "--extra-include-dir",
        type=str,
        default=[],
        nargs="+",
        help="The list of extra directories in caffe2 to hipify",
    )
    return parser.parse_args()


# Check if the compiler is hip-clang.
#
# This used to be a useful function but now we can safely always assume hip-clang.
# Leaving the function here avoids bc-linter errors.
def is_hip_clang() -> bool:
    return True


def _patterns(root: Path, patterns: list[str]) -> list[str]:
    return [os.fspath(root / pattern) for pattern in patterns]


def _copy_file(project_dir: Path, staging_dir: Path, source: Path) -> None:
    relative = source.relative_to(project_dir)
    destination = staging_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _tracked_files(project_dir: Path, prefixes: tuple[str, ...]) -> list[Path]:
    try:
        git_root = subprocess.run(
            ["git", "-C", os.fspath(project_dir), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if Path(git_root).resolve() != project_dir:
            raise subprocess.CalledProcessError(1, "git rev-parse")
        result = subprocess.run(
            ["git", "-C", os.fspath(project_dir), "ls-files", "-z", "--", *prefixes],
            check=True,
            stdout=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        result = None

    if result is not None:
        return [
            project_dir / path.decode()
            for path in result.stdout.split(b"\0")
            if path
        ]

    files = []
    for prefix in prefixes:
        path = project_dir / prefix
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate for candidate in path.rglob("*") if candidate.is_file()
            )
    return files


def _stage_project(
    project_dir: Path,
    staging_dir: Path,
    include_patterns: list[str],
    ignore_patterns: list[str],
    extra_files: list[str],
) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    includes = _patterns(project_dir, include_patterns)
    ignores = _patterns(project_dir, ignore_patterns)
    staged = set()

    for filename in hipify_python.matched_files_iter(
        os.fspath(project_dir),
        includes=includes,
        ignores=ignores,
        extensions=HIPIFY_EXTENSIONS,
    ):
        source = Path(filename)
        _copy_file(project_dir, staging_dir, source)
        staged.add(source.resolve())

    needed_prefixes = (*HIP_SOURCE_PREFIXES, AUTOGRAD_PREFIX)
    for source in _tracked_files(project_dir, needed_prefixes):
        if source.is_file() and source.resolve() not in staged:
            _copy_file(project_dir, staging_dir, source)
            staged.add(source.resolve())

    for relative in extra_files:
        source = project_dir / relative
        if source.is_file() and source.resolve() not in staged:
            _copy_file(project_dir, staging_dir, source)


def _replace_output(staging_dir: Path, output_dir: Path) -> None:
    backup_dir = output_dir.with_name(f".{output_dir.name}.previous")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        output_dir.rename(backup_dir)
    try:
        staging_dir.rename(output_dir)
    except Exception:
        if backup_dir.exists():
            backup_dir.rename(output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _remove_unchanged_files(
    project_dir: Path, staging_dir: Path, relative_dir: str
) -> None:
    staged_dir = staging_dir / relative_dir
    if not staged_dir.exists():
        return
    for staged_file in staged_dir.rglob("*"):
        if not staged_file.is_file():
            continue
        source_file = project_dir / staged_file.relative_to(staging_dir)
        if source_file.is_file() and filecmp.cmp(
            staged_file, source_file, shallow=False
        ):
            staged_file.unlink()


def _run_hipify(
    project_dir: Path,
    output_dir: Path,
    include_patterns: list[str],
    ignore_patterns: list[str],
    extra_files: list[str],
    out_of_place_only: bool,
) -> None:
    hipify_python.hipify(
        project_directory=os.fspath(project_dir),
        output_directory=os.fspath(output_dir),
        includes=_patterns(output_dir, include_patterns),
        ignores=_patterns(output_dir, ignore_patterns),
        extra_files=extra_files,
        out_of_place_only=out_of_place_only,
        hip_clang_launch=is_hip_clang(),
    )


def main() -> None:
    args = parse_args()

    # NOTE: `tools/amd_build/build_amd.py` could be a symlink.
    amd_build_dir = Path(os.path.realpath(__file__)).parent
    project_dir = Path(args.project_directory or amd_build_dir.parent.parent).resolve()
    output_dir = Path(
        args.output_directory or project_dir / "build" / "hipify"
    ).resolve()

    include_patterns = list(INCLUDES)
    for new_dir in args.extra_include_dir:
        if (project_dir / new_dir).exists():
            include_patterns.append(f"{new_dir}/**/*")

    buck_build = os.environ.get("FBCODE_BUILD_TOOL", "") == "buck"
    extra_files = [
        relative for relative in EXTRA_FILES if (project_dir / relative).exists()
    ]
    mslk_relative = "third_party/mslk/include/mslk/utils/tuning_cache.cuh"
    if not buck_build and (project_dir / mslk_relative).exists():
        extra_files.append(mslk_relative)

    if output_dir == project_dir:
        _run_hipify(
            project_dir,
            output_dir,
            include_patterns,
            list(IGNORES),
            extra_files,
            args.out_of_place_only,
        )
        hipify_root = output_dir
    else:
        if output_dir in project_dir.parents:
            raise RuntimeError(
                f"Refusing to replace {output_dir}, which contains the project "
                "directory"
            )

        staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
        _stage_project(
            project_dir,
            staging_dir,
            include_patterns,
            list(IGNORES),
            extra_files,
        )
        try:
            _run_hipify(
                staging_dir,
                staging_dir,
                include_patterns,
                list(IGNORES),
                extra_files,
                args.out_of_place_only,
            )
            if not buck_build and (project_dir / mslk_relative).exists():
                mslk_move_src = (
                    staging_dir
                    / "third_party/mslk/include/mslk/utils/hip/tuning_cache.cuh"
                )
                mslk_move_dst = (
                    staging_dir
                    / "third_party/mslk/include/mslk/utils/tuning_cache_hip.cuh"
                )
                if not mslk_move_src.exists():
                    raise RuntimeError(
                        f"Source file {mslk_move_src} does not exist"
                    )
                mslk_move_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(mslk_move_src, mslk_move_dst)
                print(f"{mslk_move_dst} updated")
            _remove_unchanged_files(
                project_dir, staging_dir, "torch/csrc/api"
            )
            (staging_dir / MANIFEST).write_text(
                json.dumps(
                    {
                        "project_directory": os.fspath(project_dir),
                        "output_directory": os.fspath(output_dir),
                    },
                    indent=2,
                )
                + "\n"
            )
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            _replace_output(staging_dir, output_dir)
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise
        hipify_root = output_dir

    if (
        output_dir == project_dir
        and not buck_build
        and (project_dir / mslk_relative).exists()
    ):
        mslk_move_src = (
            hipify_root / "third_party/mslk/include/mslk/utils/hip/tuning_cache.cuh"
        )
        mslk_move_dst = (
            hipify_root / "third_party/mslk/include/mslk/utils/tuning_cache_hip.cuh"
        )
        if not mslk_move_src.exists():
            sys.exit(f"Error: Source file {mslk_move_src} does not exist")
        mslk_move_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mslk_move_src, mslk_move_dst)
        print(f"{mslk_move_dst} updated")


if __name__ == "__main__":
    main()

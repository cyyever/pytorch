#!/usr/bin/env python3
"""Build a PyTorch wheel inside a manylinux container.

Usage: build_wheel.py <output_dir>

Expects all build env vars (USE_CUDA, TORCH_CUDA_ARCH_LIST, etc.) to be set
by the caller (GitHub Actions workflow env). This script only adds the BLAS
plumbing that depends on the host architecture.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def configure_blas_env() -> None:
    """Tell CMake which BLAS to use, based on architecture and GPU type.

    On x86, MKL from /opt/intel is wired in via CMAKE_{INCLUDE,LIBRARY}_PATH.
    """
    arch = platform.machine()
    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE", "")
    print(
        f"build_wheel.py: ARCH={arch} GPU_ARCH_TYPE={gpu_arch_type or 'unset'} "
        f"DESIRED_CUDA={os.environ.get('DESIRED_CUDA', 'unset')}"
    )

    if arch == "x86_64" and Path("/opt/intel/include").is_dir():
        os.environ["CMAKE_INCLUDE_PATH"] = "/opt/intel/include"
        os.environ["CMAKE_LIBRARY_PATH"] = "/opt/intel/lib:/lib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    configure_blas_env()

    subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(args.output_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

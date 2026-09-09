#!/bin/bash

# shellcheck disable=SC2086,SC2006,SC2207,SC2076,SC2155,SC2046,SC1091,SC2143
# TODO: Re-enable shellchecks above

set -eux -o pipefail

# This script checks the following things on binaries
# 1. The gcc abi matches DESIRED_DEVTOOLSET
# 2. MacOS binaries do not link against OpenBLAS
# 3. There are no protobuf symbols of any sort anywhere (turned off, because
#    this is currently not true)
# 4. Standard Python imports work
# 5. The configured BLAS backend is available
# 6. CUDA is setup correctly and does not hang
# 7. CuDNN is available for CUDA builds
#
# This script needs the env variables DESIRED_PYTHON, DESIRED_CUDA,
# DESIRED_DEVTOOLSET and PACKAGE_TYPE. EXPECTED_BLAS defaults to OpenBLAS.
#
# This script expects PyTorch to be installed into the active Python. Or, if
# this is testing a libtorch
# Pythonless binary, then it expects to be in the root folder of the unzipped
# libtorch package.

# ensure we don't link to system libraries, linked libraries should be found from RPATH
# Save the old LD_LIBRARY_PATH to restore it later
OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH

if [[ -z ${DESIRED_PYTHON:-} ]]; then
  export DESIRED_PYTHON=${MATRIX_PYTHON_VERSION:-}
fi
if [[ -z ${DESIRED_CUDA:-} ]]; then
  export DESIRED_CUDA=${MATRIX_DESIRED_CUDA:-}
fi
if [[ -z ${DESIRED_DEVTOOLSET:-} ]]; then
  export DESIRED_DEVTOOLSET=${MATRIX_DESIRED_DEVTOOLSET:-}
fi
if [[ -z ${PACKAGE_TYPE:-} ]]; then
  export PACKAGE_TYPE=${MATRIX_PACKAGE_TYPE:-}
fi

# The install root depends on both the package type and the os
# All MacOS packages use conda, even for the wheel packages.
if [[ "$PACKAGE_TYPE" == libtorch ]]; then
  # NOTE: Only $PWD works on both CentOS and Ubuntu
  export install_root="$PWD"
else

  if [[ $DESIRED_PYTHON =~ ^cp([0-9])([0-9][0-9])(-cp[0-9]+)?t?$ ]]; then
    # Handle inputs like cp310-cp310 or cp310-cp310t
    py_dot="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
  elif [[ $DESIRED_PYTHON =~ ([0-9].[0-9]+)t ]]; then
    # For python that is maj.mint keep original version
    py_dot="$DESIRED_PYTHON"
  elif [[ $DESIRED_PYTHON =~ ([0-9].[0-9]+) ]];  then
    # Strip everything but major.minor from DESIRED_PYTHON version
    py_dot="${BASH_REMATCH[0]}"
  else
    echo "Unexpected ${DESIRED_PYTHON} format"
    exit 1
  fi
  actual_py_dot="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$actual_py_dot" != "${py_dot%t}" ]]; then
    echo "Expected Python ${py_dot}, but the active Python is ${actual_py_dot}"
    exit 1
  fi
  site_packages="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
  export install_root="${site_packages}/torch/"
fi

###############################################################################
# Check GCC ABI
###############################################################################

# NOTE: As of https://github.com/pytorch/pytorch/issues/126551 we only produce
#       wheels with cxx11-abi

echo "Checking that the gcc ABI is what we expect"
if [[ "$(uname)" != 'Darwin' ]]; then
  # We also check that there are cxx11 symbols in libtorch
  #
  echo "Checking that symbols in libtorch.so have the right gcc abi"
  python3 "$(dirname ${BASH_SOURCE[0]})/smoke_test/check_binary_symbols.py"

  echo "cxx11 symbols seem to be in order"
fi # if on Darwin

###############################################################################
# Check for no OpenBLAS
# TODO Check for no Protobuf symbols (not finished)
# Print *all* runtime dependencies
###############################################################################
# We have to loop through all shared libraries for this
if [[ "$(uname)" == 'Darwin' ]]; then
  all_dylibs=($(find "$install_root" -name '*.dylib'))
  for dylib in "${all_dylibs[@]}"; do
    echo "All dependencies of $dylib are $(otool -L $dylib) with rpath $(otool -l $dylib | grep LC_RPATH -A2)"

    # Check that OpenBlas is not linked to on Macs
    echo "Checking the OpenBLAS is not linked to"
    if [[ -n "$(otool -L $dylib | grep -i openblas)" ]]; then
      echo "ERROR: Found openblas as a dependency of $dylib"
      echo "Full dependencies is: $(otool -L $dylib)"
      exit 1
    fi

    # Check for protobuf symbols
    #proto_symbols="$(nm $dylib | grep protobuf)" || true
    #if [[ -n "$proto_symbols" ]]; then
    #  echo "ERROR: Detected protobuf symbols in $dylib"
    #  echo "Symbols are $proto_symbols"
    #  exit 1
    #fi
  done
else
  all_libs=($(find "$install_root" -name '*.so'))
  for lib in "${all_libs[@]}"; do
    echo "All dependencies of $lib are $(ldd $lib) with runpath $(objdump -p $lib | grep RUNPATH)"

    # Check for protobuf symbols
    #proto_symbols=$(nm $lib | grep protobuf) || true
    #if [[ -n "$proto_symbols" ]]; then
    #  echo "ERROR: Detected protobuf symbols in $lib"
    #  echo "Symbols are $proto_symbols"
    #  exit 1
    #fi
  done
fi

setup_link_flags () {
  REF_LIB="-Wl,-R${install_root}/lib"
  if [[ "$(uname)" == 'Darwin' ]]; then
    REF_LIB="-Wl,-rpath ${install_root}/lib"
  fi
  ADDITIONAL_LINKER_FLAGS=""
  if [[ "$(uname)" == 'Linux' ]]; then
    ADDITIONAL_LINKER_FLAGS="-Wl,--no-as-needed"
  fi
  C10_LINK_FLAGS=""
  if [ -f "${install_root}/lib/libc10.so" ] || [ -f "${install_root}/lib/libc10.dylib" ]; then
    C10_LINK_FLAGS="-lc10"
  fi
  TORCH_CPU_LINK_FLAGS=""
  if [ -f "${install_root}/lib/libtorch_cpu.so" ] || [ -f "${install_root}/lib/libtorch_cpu.dylib" ]; then
    TORCH_CPU_LINK_FLAGS="-ltorch_cpu"
  fi
  TORCH_CUDA_LINK_FLAGS=""
  if [ -f "${install_root}/lib/libtorch_cuda.so" ] || [ -f "${install_root}/lib/libtorch_cuda.dylib" ]; then
    TORCH_CUDA_LINK_FLAGS="-ltorch_cuda"
  elif [ -f "${install_root}/lib/libtorch_cuda_cpp.so" ] && [ -f "${install_root}/lib/libtorch_cuda_cpp.so" ] || \
    [ -f "${install_root}/lib/libtorch_cuda_cu.dylib" ] && [ -f "${install_root}/lib/libtorch_cuda_cu.dylib" ]; then
    TORCH_CUDA_LINK_FLAGS="-ltorch_cuda_cpp -ltorch_cuda_cu"
  fi
}

TEST_CODE_DIR="$(dirname $(realpath ${BASH_SOURCE[0]}))/test_example_code"
build_and_run_example_cpp () {
  setup_link_flags
  g++ ${TEST_CODE_DIR}/$1.cpp -I${install_root}/include -I${install_root}/include/torch/csrc/api/include -std=gnu++20 -L${install_root}/lib ${REF_LIB} ${ADDITIONAL_LINKER_FLAGS} -ltorch $TORCH_CPU_LINK_FLAGS $TORCH_CUDA_LINK_FLAGS $C10_LINK_FLAGS -o $1
  ./$1
}

###############################################################################
# Check simple Python/C++ calls
###############################################################################
if [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
  # NS: Set LD_LIBRARY_PATH for CUDA builds, but perhaps it should be removed
  if [[ "$DESIRED_CUDA" == "cu"* ]]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
  fi
  build_and_run_example_cpp simple-torch-test
else
  pushd /tmp
  python -c 'import torch'
  popd
fi

###############################################################################
# Check torch.git_version
###############################################################################
if [[ "$PACKAGE_TYPE" != 'libtorch' ]]; then
  pushd /tmp
  python -c 'import torch; assert torch.version.git_version != "Unknown"'
  python -c 'import torch; assert torch.version.git_version != None'
  popd
fi


###############################################################################
# Check BLAS
###############################################################################

if [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
  if [[ "${EXPECTED_BLAS:-open}" == "mkl" ]]; then
    echo "Checking that MKL is available"
    build_and_run_example_cpp check-torch-mkl
  else
    echo "Skipping the MKL-specific check for ${EXPECTED_BLAS:-open} libtorch"
  fi
elif [[ "$(uname -m)" != "arm64" ]]; then
  if [[ "$(uname)" != 'Darwin' || "$PACKAGE_TYPE" != *wheel ]]; then
    echo "Checking that the configured BLAS backend is available"
    pushd /tmp
    python - <<'PY'
import re
import os

import torch

config = torch.__config__.show()
match = re.search(r"\bBLAS_INFO=([^,\s]+)", config)
if match is None:
    raise RuntimeError("BLAS_INFO is missing from torch.__config__.show()")

blas = match.group(1)
expected_blas = os.environ.get("EXPECTED_BLAS", "open")
if blas != expected_blas:
    raise RuntimeError(f"Expected BLAS_INFO={expected_blas}, found {blas}")
if blas == "mkl" and not torch.backends.mkl.is_available():
    raise RuntimeError("PyTorch was configured with MKL, but MKL is unavailable")

print(f"Configured BLAS backend: {blas}")
PY
    popd
  fi
fi

###############################################################################
# Check XPU configured correctly
###############################################################################
if [[ "$DESIRED_CUDA" == 'xpu' && "$PACKAGE_TYPE" != 'libtorch' ]]; then
  echo "Checking that xpu is compiled"
  pushd /tmp
  python -c 'import torch; exit(0 if torch.xpu._is_compiled() else 1)'
  popd
fi

###############################################################################
# Check ROCm configured correctly
###############################################################################
if [[ "$DESIRED_CUDA" == *rocm* && "$PACKAGE_TYPE" != 'libtorch' ]]; then
  echo "Checking that ROCm is compiled and usable"
  pushd /tmp
  timeout 120 python - <<'PY'
import torch

if not torch.cuda._is_compiled():
    raise RuntimeError("ROCm support is not compiled")
if torch.version.hip is None:
    raise RuntimeError("HIP version is missing")
if not torch.cuda.is_available():
    raise RuntimeError("ROCm device is unavailable")

x = torch.randn(32, 32, device="cuda")
torch.mm(x, x)
torch.cuda.synchronize()
PY
  popd
fi

###############################################################################
# Check CUDA configured correctly
###############################################################################
# Test that CUDA builds are setup correctly
if [[ "$DESIRED_CUDA" != 'cpu' && "$DESIRED_CUDA" != 'xpu' && "$DESIRED_CUDA" != 'cpu-cxx11-abi' && "$DESIRED_CUDA" != *"rocm"* ]]; then
  if [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
    build_and_run_example_cpp check-torch-cuda
  else
    pushd /tmp
    echo "Checking that CUDA archs are setup correctly"
    timeout 20 python -c 'import torch; torch.randn([3,5]).cuda()'

    # These have to run after CUDA is initialized

    echo "Checking that CuDNN is available"
    python -c 'import torch; exit(0 if torch.backends.cudnn.is_available() else 1)'

    # Validates builds is free of linker regressions reported in https://github.com/pytorch/pytorch/issues/57744
    echo "Checking that exception handling works"
    python -c "import torch; from unittest import TestCase;TestCase().assertRaises(RuntimeError, lambda:torch.eye(7, 7, device='cuda:7'))"

    echo "Checking that basic RNN works"
    python ${TEST_CODE_DIR}/rnn_smoke.py

    echo "Checking that basic CNN works"
    python "${TEST_CODE_DIR}/cnn_smoke.py"

    echo "Test that linalg works"
    python -c "import torch;x=torch.rand(3,3,device='cuda');print(torch.linalg.svd(torch.mm(x.t(), x)))"

    popd
  fi # if libtorch
fi # if cuda

##########################
# Run parts of smoke tests
##########################
if [[ "$PACKAGE_TYPE" != 'libtorch' ]]; then
  pushd "$(dirname ${BASH_SOURCE[0]})/smoke_test"
  python -c "from smoke_test import test_linalg; test_linalg()"
  if [[ "$DESIRED_CUDA" == cu* || "$DESIRED_CUDA" == *rocm* ]]; then
    python -c "from smoke_test import test_linalg; test_linalg('cuda')"
  fi
  popd
fi

###############################################################################
# Restore LD_LIBRARY_PATH to its original value
###############################################################################
if [[ -n "$OLD_LD_LIBRARY_PATH" ]]; then
  export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
fi

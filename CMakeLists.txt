# PyTorch build overview
#
# This file is the entry point of the native build. It is normally driven by
# scikit-build-core through `pip install -e . -v --no-build-isolation`
# (`spin develop`) rather than by invoking cmake directly; the exception is the
# standalone libtorch build in tools/build_libtorch.py. The stages below appear
# in the order they are executed in this file.
#
# 1. Bootstrap: forward the toolchain env vars needed before project() (cross
#    compilation), call project() with the version from version.txt, then
#    cmake/EnvVarForwarding.cmake turns the remaining build env vars into CMake
#    cache variables. That module is the reference for the environment
#    variables that configure the build: look there first for an env var to
#    set, and document new ones there. The exceptions are third-party locations
#    such as MKLROOT or ROCM_PATH and a handful of niche flags, which are read
#    directly where they are used.
# 2. Pre-build (cmake/PreBuildSteps.cmake): initialize and verify the git
#    submodules, and fetch the pinned NCCL for distributed CUDA builds.
# 3. Options: declare every build knob (USE_*, BUILD_*). An option added here
#    must also be added to cmake/Summary.cmake.
# 4. Dependencies (cmake/Dependencies.cmake): find or build the third-party
#    libraries and set up the accelerator toolchains (CUDA/ROCm/XPU). The
#    global compile and link flags are resolved inline right after.
# 5. Code generation, all driven from caffe2/CMakeLists.txt: cmake/Codegen.cmake
#    runs torchgen over native_functions.yaml to emit the ATen operator headers
#    and sources, and tools/setup_helpers/generate_code.py emits the autograd
#    and Python binding sources. The .pyi stubs are generated later, in
#    torch/CMakeLists.txt.
# 6. Main build: add_subdirectory() of torch/headeronly, c10 and caffe2 (which
#    in turn adds torch/). This builds c10, torch_cpu, the per-accelerator
#    torch_cuda/torch_hip/torch_xpu, the umbrella torch library and
#    torch_python, plus the C++ tests; those subdirectories also carry the
#    install rules for the headers and libraries.
# 7. Export and extras: generate the Caffe2Config.cmake / Caffe2Targets.cmake
#    files that downstream C++ projects consume, then the optional binaries/
#    and JNI bindings, which link against the libraries built above.
# 8. Python packaging: cmake/FileMirroring.cmake copies sources into
#    torchgen/packaged and into the torch/ subpackages that need files vendored
#    out of third_party/, cmake/PackageData.cmake installs the non-Python
#    package data (stubs, headers, templates) into torch/, and
#    cmake/PostBuildSteps.cmake merges compile_commands.json and registers the
#    install-time fixups (TORCH_STABLE_ONLY header wrapping, Windows runtime
#    DLL and macOS OpenMP bundling).
# 9. Summary (cmake/Summary.cmake): print the resolved configuration.

cmake_minimum_required(VERSION 3.29 FATAL_ERROR)

cmake_policy(SET CMP0126 OLD)
# Don't remove the FindCUDA module
cmake_policy(SET CMP0146 OLD)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

# Cross-compilation is configured by CMAKE_SYSTEM_NAME / CMAKE_SYSTEM_PROCESSOR,
# which must be set before project() so CMake enters cross-compile mode and
# reports the target architecture. The general environment-variable forwarding
# in cmake/EnvVarForwarding.cmake runs after project() and skips variables that
# are already defined, so project() would otherwise set these to the host values
# and the target values passed via the environment would be lost. Forward just
# the toolchain-determining variables here, before project().
foreach(_cross_var
    CMAKE_SYSTEM_NAME
    CMAKE_SYSTEM_PROCESSOR
    CMAKE_C_COMPILER
    CMAKE_CXX_COMPILER
    CMAKE_SYSROOT
    CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{${_cross_var}} AND NOT DEFINED ${_cross_var})
    set(${_cross_var} "$ENV{${_cross_var}}")
  endif()
endforeach()

# ---[ Project and semantic versioning.
# When built via scikit-build-core, these are set automatically.
# Provide fallbacks for standalone cmake invocations.
if(NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME "torch")
endif()
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" SKBUILD_PROJECT_VERSION)
  string(STRIP "${SKBUILD_PROJECT_VERSION}" SKBUILD_PROJECT_VERSION)
  # version.txt may contain PEP 440 pre-release suffixes like "a0" that are
  # not valid CMake version components. Strip everything after the last digit
  # group so project(VERSION ...) still works.
  string(REGEX MATCH "^[0-9]+\\.[0-9]+(\\.[0-9]+)?" SKBUILD_PROJECT_VERSION "${SKBUILD_PROJECT_VERSION}")
endif()
project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX C)

# When built via scikit-build-core, Python_EXECUTABLE is already set.
# For standalone cmake invocations, find it via find_package.
find_package(Python COMPONENTS Interpreter REQUIRED)

# ---[ Where the Python packages go.
# scikit-build-core exports each wheel tree to CMake as SKBUILD_<TREE>_DIR, and
# wheel.install-dir picks which one becomes CMAKE_INSTALL_PREFIX. We set it to
# "torch" so the C++ distribution lands at torch/lib and torch/include, as the
# install rpath and torch/__init__.py both require; the prefix therefore sits
# one level below platlib, and the Python-side install rules address
# SKBUILD_PLATLIB_DIR directly instead of reaching it through the prefix.
#
# Those rules are guarded on SKBUILD, so a build scikit-build-core does not
# drive (libtorch, tools/build_libtorch.py, downstream cmake --install) installs
# the C++ distribution alone and keeps package data out of a conventional prefix
# (see gh-191102). SKBUILD is set for every scikit-build-core configure,
# editable installs included, where platlib is a per-build staging directory.
#
# The extension modules resolve the C++ tree through an $ORIGIN/lib install
# rpath and torch/__init__.py resolves bin/ and lib/ off its own __file__, so
# the prefix has to be the torch package directory. Enforce it rather than let
# a mismatch produce an install that only fails at import time.
if(SKBUILD)
  get_filename_component(_torch_pkg_dir "${SKBUILD_PLATLIB_DIR}/torch" ABSOLUTE)
  get_filename_component(_torch_prefix "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
  if(NOT _torch_pkg_dir STREQUAL _torch_prefix)
    message(FATAL_ERROR
      "CMAKE_INSTALL_PREFIX must be the torch package directory: the "
      "extension modules load the C++ libraries through an $ORIGIN/lib rpath "
      "relative to themselves.\n"
      "  SKBUILD_PLATLIB_DIR  : ${SKBUILD_PLATLIB_DIR}\n"
      "  expected prefix      : ${_torch_pkg_dir}\n"
      "  CMAKE_INSTALL_PREFIX : ${_torch_prefix}\n"
      "Keep wheel.install-dir = \"torch\" in pyproject.toml, which is what "
      "makes the two agree.")
  endif()
  unset(_torch_pkg_dir)
  unset(_torch_prefix)
endif()

# Forward environment variables to CMake variables, replicating the behavior
# previously handled by setup.py / tools/setup_helpers/cmake.py.
# When built via setup.py, these are already set as -D flags; the module
# skips variables that are already defined, so it is safe to include
# unconditionally.
include(cmake/EnvVarForwarding.cmake)

# Pre-build steps: submodule init, NCCL checkout.
# When built via setup.py, these are handled by build_deps() in setup.py;
# the module checks for existing state, so it is safe to include
# unconditionally.
include(cmake/PreBuildSteps.cmake)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX TRUE)
else()
  set(LINUX FALSE)
endif()

set(CMAKE_INSTALL_MESSAGE NEVER)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
    WARNING
      "C++ standard version definition detected in environment variable. "
      "PyTorch requires -std=c++23. Please remove -std=c++ settings in your environment."
  )
endif()
set(CMAKE_CXX_STANDARD
    23
    CACHE STRING
          "The C++ standard whose features are requested to build this target.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD
    17
    CACHE STRING
          "The C standard whose features are requested to build this target.")

# ---[ Utils
include(cmake/public/utils.cmake)

# --- [ Check that minimal gcc version is 15+
# libstdc++ 15 is the first with a complete C++23 library. Older GCCs accept
# -std=c++23 but are missing pieces of it, so they fail deep in the build
# instead of here.
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15)
  message(
    FATAL_ERROR
      "GCC-15 or newer is required to compile PyTorch, but found ${CMAKE_CXX_COMPILER_VERSION}"
  )
endif()

# --- [ Check that minimal clang version is 20+
# Matches the clang that ROCm 7.0 ships, the oldest clang this fork targets.
# AppleClang uses its own versioning scheme and is excluded.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 20)
  message(
    FATAL_ERROR
      "Clang-20 or newer is required to compile PyTorch, but found ${CMAKE_CXX_COMPILER_VERSION}"
  )
endif()

# This define is needed to preserve behavior given anticpated changes to
# cccl/thrust
# https://github.com/NVIDIA/cccl/blob/main/docs/libcudacxx/standard_api/numerics_library/complex.rst
string(APPEND CMAKE_CUDA_FLAGS
       " -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# One variable that determines whether the current cmake process is being run
# with the main Caffe2 library. This is useful for building modules - if modules
# are built with the main Caffe2 library then one does not need to do find
# caffe2 in the cmake script. One can usually guard it in some way like if(NOT
# CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO) find_package(Caffe2 REQUIRED) endif()
set(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO ON)

# Googletest's cmake files are going to set it on once they are processed. Let's
# set it at the very beginning so that the entire build is deterministic.
set(THREADS_PREFER_PTHREAD_FLAG ON)

if(NOT DEFINED BLAS_SET_BY_USER)
  if(DEFINED BLAS)
    set(BLAS_SET_BY_USER TRUE)
  else()
    message(STATUS "Not forcing any particular BLAS to be found")
    set(BLAS_SET_BY_USER FALSE)
  endif()
  set(BLAS_SET_BY_USER
      ${BLAS_SET_BY_USER}
      CACHE STRING
            "Marks whether BLAS was manually set by user or auto-detected")
endif()

# Apple specific
if(APPLE)
  # These lines are an attempt to make find_package(cuda) pick up libcuda.dylib,
  # and not cuda.framework.  It doesn't work all the time, but it seems to help
  # for some users. TODO: replace this with a more robust fix
  set(CMAKE_FIND_FRAMEWORK LAST)
  set(CMAKE_FIND_APPBUNDLE LAST)

  # Get clang version on macOS
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE clang_full_version_string)
  string(REGEX REPLACE "Apple (.*) version ([0-9]+\\.[0-9]+).*" "\\2"
                       CLANG_VERSION_STRING ${clang_full_version_string})
  message(STATUS "CLANG_VERSION_STRING:         " ${CLANG_VERSION_STRING})

  # Determine if we can link against MPSGraph
  set(MPS_FOUND OFF)
  execute_process(
    COMMAND bash -c "xcrun --sdk macosx --show-sdk-version"
    RESULT_VARIABLE _exit_code
    OUTPUT_VARIABLE _macosx_sdk_version
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_exit_code EQUAL 0)
    set(_MPS_supported_os_version OFF)
    if(_macosx_sdk_version VERSION_GREATER_EQUAL 12.3)
      set(_MPS_supported_os_version ON)
    endif()
    message(
      STATUS
        "sdk version: ${_macosx_sdk_version}, mps supported: ${_MPS_supported_os_version}"
    )
    execute_process(
      COMMAND bash -c "xcrun --sdk macosx --show-sdk-path"
      OUTPUT_VARIABLE _macosx_sdk_path
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(_SDK_SEARCH_PATH "${_macosx_sdk_path}/System/Library/Frameworks/")
    set(_FRAMEWORK_SEARCH_PATH "/System/Library/Frameworks/")

    find_library(
      _MPS_fwrk_path_
      NAMES MetalPerformanceShadersGraph MetalPerformanceShaders
      PATHS ${_FRAMEWORK_SEARCH_PATH}
      NO_DEFAULT_PATH)
    find_library(
      _MPS_sdk_path_
      NAMES MetalPerformanceShadersGraph MetalPerformanceShaders
      PATHS ${_SDK_SEARCH_PATH}
      NO_DEFAULT_PATH)

    if(_MPS_supported_os_version
       AND _MPS_fwrk_path_
       AND _MPS_sdk_path_)
      set(MPS_FOUND ON)
      message(STATUS "MPSGraph framework found")
    else()
      message(STATUS "MPSGraph framework not found")
    endif()
  else()
    message(STATUS "MPS: unable to get MacOS sdk version")
    message(STATUS "MPSGraph framework not found")
  endif()
endif()

set(CPU_AARCH64 OFF)
set(CPU_INTEL OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(AMD64|x86_64)")
  set(CPU_INTEL ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)")
  # Linux reports "aarch64", macOS "arm64", and native Windows on Arm "ARM64".
  # CMake MATCHES is case-sensitive, so accept each spelling.
  set(CPU_AARCH64 ON)
endif()

# For non-supported platforms, turn USE_DISTRIBUTED off by default. It is not
# tested and likely won't work without additional changes.
if(NOT LINUX)
  set(USE_DISTRIBUTED
      OFF
      CACHE STRING "Use distributed")
  # On macOS, if USE_DISTRIBUTED is enabled (specified by the user), then make
  # Gloo build with the libuv transport.
  if(APPLE AND USE_DISTRIBUTED)
    set(USE_LIBUV
        ON
        CACHE STRING "")
  endif()
endif()

# ---[ Options. Note to developers: if you add an option below, make sure you
# also add it to cmake/Summary.cmake so that the summary prints out the option
# values.
include(CMakeDependentOption)
option(ATEN_NO_TEST "Do not build ATen test binaries" OFF)
option(BUILD_BINARY "Build C++ binaries" OFF)
option(BUILD_PYTHON "Build Python binaries" ON)
option(BUILD_SHARED_LIBS "Build libcaffe2.so" ON)
option(BUILD_TEST "Build C++ test binaries (need gtest and gbenchmark)" ON)
option(BUILD_AOT_INDUCTOR_TEST "Build C++ test binaries for aot-inductor" OFF)
option(
  BUILD_MOBILE_BENCHMARK
  "Build C++ test binaries for mobile (ARM) targets(need gtest and gbenchmark)"
  OFF)
option(
  BUILD_MOBILE_TEST
  "Build C++ test binaries for mobile (ARM) targets(need gtest and gbenchmark)"
  OFF)
cmake_dependent_option(INSTALL_TEST "Install test binaries if BUILD_TEST is on"
                       ON "BUILD_TEST" OFF)
option(USE_RELATIVE_PATHS "Use relative paths in generated files for ccache friendliness" OFF)
option(USE_CPP_CODE_COVERAGE "Compile C/C++ with code coverage flags" OFF)
option(USE_COLORIZE_OUTPUT "Colorize output during compilation" ON)
option(USE_ASAN "Use Address+Undefined Sanitizers" OFF)
option(USE_LSAN "Use Leak Sanitizer" OFF)
option(USE_TSAN "Use Thread Sanitizer" OFF)

# Track whether USE_CUDA was explicitly set by the user (before option() is called)
# If USE_CUDA is already defined in cache, it means user explicitly set it
if(DEFINED CACHE{USE_CUDA})
  set(_USE_CUDA_EXPLICITLY_SET TRUE)
else()
  set(_USE_CUDA_EXPLICITLY_SET FALSE)
endif()

# CUDA is incompatible with TSAN
cmake_dependent_option(USE_CUDA "Use CUDA" ON "NOT USE_TSAN" OFF)

# If USE_XPU is already defined in cache, it means user explicitly set it
if(DEFINED CACHE{USE_XPU})
  set(_USE_XPU_EXPLICITLY_SET TRUE)
else()
  set(_USE_XPU_EXPLICITLY_SET FALSE)
endif()

option(USE_XPU "Use XPU" ON)
cmake_dependent_option(
  BUILD_LAZY_CUDA_LINALG "Build cuda linalg ops as separate library" ON
  "USE_CUDA AND LINUX AND BUILD_PYTHON" OFF)
cmake_dependent_option(USE_ROCM "Use ROCm" ON "LINUX" OFF)
cmake_dependent_option(USE_ROCM_CK_GEMM "Use ROCm Composable Kernel for GEMMs" ON "USE_ROCM" OFF)
cmake_dependent_option(USE_ROCM_CK_SDPA "Use ROCm Composable Kernel for SDPA" ON "USE_ROCM" OFF)
# Off by default because enabling gfx12 adds CK SDPA code generation and compile
# cost to ROCm builds that already target gfx1200/gfx1201.
cmake_dependent_option(USE_ROCM_CK_SDPA_GFX12 "Also build ROCm Composable Kernel SDPA for gfx12 (RDNA4)" OFF "USE_ROCM_CK_SDPA" OFF)
option(CAFFE2_STATIC_LINK_CUDA "Statically link CUDA libraries" OFF)
cmake_dependent_option(USE_CUDNN "Use cuDNN" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_STATIC_CUDNN "Use cuDNN static libraries" OFF
                       "USE_CUDNN" OFF)
cmake_dependent_option(USE_CUSPARSELT "Use cuSPARSELt" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_HIPSPARSELT "Use hipSPARSELt" ON "USE_ROCM" OFF)
cmake_dependent_option(USE_CUDSS "Use cuDSS" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_CUFILE "Use cuFile" ON "USE_CUDA OR USE_ROCM" OFF)
option(USE_FBGEMM "Use FBGEMM (quantized 8-bit server operators)" ON)
option(USE_KINETO "Use Kineto profiling library" ON)
option(USE_CUPTI_SO "Use CUPTI as a shared library" ON)
option(USE_NATIVE_ARCH "Use -march=native" OFF)
cmake_dependent_option(USE_MPS "Use MPS for macOS build" ON "MPS_FOUND" OFF)
option(USE_DISTRIBUTED "Use distributed" ON)
cmake_dependent_option(USE_NCCL "Use NCCL" ON
                       "USE_DISTRIBUTED;USE_CUDA OR USE_ROCM;UNIX;NOT APPLE" OFF)
cmake_dependent_option(USE_XCCL "Use XCCL" ON
                       "USE_XPU;UNIX;NOT APPLE" OFF)
cmake_dependent_option(USE_RCCL "Use RCCL" ON "USE_NCCL" OFF)
cmake_dependent_option(USE_STATIC_NCCL "Use static NCCL" OFF "USE_NCCL" OFF)
cmake_dependent_option(USE_SYSTEM_NCCL "Use system-wide NCCL" OFF "USE_NCCL"
                       OFF)
# Builds NCCL-EP if true.
cmake_dependent_option(USE_NCCL_EP "Build NCCL EP contrib library" OFF
                       "USE_NCCL;USE_CUDA" OFF)
cmake_dependent_option(USE_NVSHMEM "Use NVSHMEM" ON
                       "USE_DISTRIBUTED;USE_CUDA OR USE_ROCM;UNIX;NOT APPLE" OFF)
cmake_dependent_option(USE_NUMA "Use NUMA. Only available on Linux." ON "LINUX"
                       OFF)
cmake_dependent_option(USE_NVRTC "Use NVRTC. Only available if USE_CUDA is on."
                       ON "USE_CUDA" OFF)
option(USE_NUMPY "Use NumPy" ON)
option(USE_OPENMP "Use OpenMP for parallel code" ON)
option(USE_PRECOMPILED_HEADERS "Use pre-compiled headers to accelerate build."
       OFF)

option(USE_ROCM_KERNEL_ASSERT "Use Kernel Assert for ROCm" OFF)
option(USE_STATIC_MKL "Prefer to link with MKL statically (Unix only)" OFF)
cmake_dependent_option(
    USE_GLOO "Use Gloo. Only available if USE_DISTRIBUTED is on." ON
    "USE_DISTRIBUTED" OFF)
cmake_dependent_option(
    USE_GLOO_WITH_OPENSSL "Use Gloo with OpenSSL. Only available if USE_GLOO is on." OFF
    "USE_GLOO AND LINUX" OFF)
cmake_dependent_option(
    USE_GLOO_IBVERBS "Use Gloo with ibverbs backend. Only available if USE_GLOO is on." OFF
    "USE_GLOO AND LINUX" OFF)
cmake_dependent_option(
    USE_C10D_GLOO "USE C10D GLOO" ON "USE_DISTRIBUTED;USE_GLOO" OFF)
cmake_dependent_option(
    USE_C10D_NCCL "USE C10D NCCL" ON "USE_DISTRIBUTED;USE_NCCL" OFF)
cmake_dependent_option(
    USE_C10D_XCCL "USE C10D XCCL" ON "USE_DISTRIBUTED;USE_XCCL" OFF)
option(HAVE_SOVERSION "Whether to add SOVERSION to the shared objects" OFF)
option(BUILD_LIBTORCH_CPU_WITH_DEBUG
       "Enable RelWithDebInfo for libtorch_cpu target only" OFF)
cmake_dependent_option(
  USE_CCACHE "Attempt using [S]CCache to wrap the compilation" ON "UNIX" OFF)
option(WERROR "Build with -Werror supported by the compiler" OFF)
option(
  DEBUG_CUDA
  "When compiling DEBUG, also attempt to compile CUDA with debug flags (may cause nvcc to OOM)"
  OFF)
cmake_dependent_option(BUILD_FUNCTORCH "Build Functorch" ON "BUILD_PYTHON" OFF)
cmake_dependent_option(BUILD_BUNDLE_PTXAS "Bundle PTX into torch/bin folder"
                       OFF "USE_CUDA" OFF)
cmake_dependent_option(
  USE_CMAKE_INSTRUMENTATION
  "Use cmake-instrumentation to collect build metrics"
  OFF # disabled by default because it causes CI timeouts
  "CMAKE_VERSION VERSION_GREATER_EQUAL 4.3"
  OFF
)
if(USE_CMAKE_INSTRUMENTATION)
  cmake_instrumentation(
    API_VERSION 1
    DATA_VERSION 1.0
    HOOKS postCMakeBuild
    OPTIONS trace
    CALLBACK "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_trace_summary.py" --root "${CMAKE_CURRENT_SOURCE_DIR}" --index
  )
endif()
# prioritized text linker, ON by default for AArch64+Linux, option visible to all AArch64 and x86.
set(USE_PRIORITIZED_TEXT_DEFAULT OFF)
if(LINUX AND CPU_AARCH64)
  set(USE_PRIORITIZED_TEXT_DEFAULT ON)
endif()
cmake_dependent_option(USE_PRIORITIZED_TEXT_FOR_LD "Use prioritized text linker for ld."
  "${USE_PRIORITIZED_TEXT_DEFAULT}" "CPU_INTEL OR CPU_AARCH64" OFF)

# LLVM BOLT post-link binary optimization. Visible on AArch64 and x86.
# Defaults ON for aarch64+linux+CUDA builds that also apply the linker script
# optimization and a suitable llvm-bolt (>=21) is available. Depends on the
# linker script because the committed profiles were collected with it enabled.
function(check_llvm_bolt_version var item)
  execute_process(COMMAND ${item} --version OUTPUT_VARIABLE _cmd_out)
  string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+" _bolt_version "${_cmd_out}")
  if(${_bolt_version} VERSION_LESS 21)
    set(${var} FALSE PARENT_SCOPE)
  endif()
endfunction()
find_program(LLVM_BOLT_EXECUTABLE NAMES llvm-bolt VALIDATOR check_llvm_bolt_version)

set(USE_LLVM_BOLT_DEFAULT OFF)
if(LINUX AND USE_CUDA AND CPU_AARCH64 AND USE_PRIORITIZED_TEXT_FOR_LD AND LLVM_BOLT_EXECUTABLE)
  set(USE_LLVM_BOLT_DEFAULT ON)
endif()
cmake_dependent_option(USE_LLVM_BOLT "Optimize libraries with LLVM BOLT."
  "${USE_LLVM_BOLT_DEFAULT}" "CPU_INTEL OR CPU_AARCH64" OFF)

option(USE_MIMALLOC "Use mimalloc" OFF)
# Enable third party mimalloc library to improve memory allocation performance
# on AArch64 Linux.
option(USE_MIMALLOC_ON_MKL "Use mimalloc on MKL" OFF)
if(CPU_AARCH64 AND NOT APPLE)
  set(USE_MIMALLOC ON)
  # Disable OPT flags for mimalloc on aarch64 as it breaks ARMv8.0
  # See https://github.com/pytorch/pytorch/issues/174344
  if(CPU_AARCH64)
    set(MI_NO_OPT_ARCH ON)
  endif()

  # Not enable USE_MIMALLOC_ON_MKL due to it caused issue:
  # https://github.com/pytorch/pytorch/issues/138994
  # Will turn on when we can fix USE_STATIC_MKL lost functionality:
  # https://github.com/pytorch/pytorch/pull/138996
  # set(USE_MIMALLOC_ON_MKL ON)
endif()

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  find_program(SCCACHE_EXECUTABLE sccache)
  if(CCACHE_PROGRAM)
    foreach(LANG CXX C CUDA)
      set(CMAKE_${LANG}_COMPILER_LAUNCHER
          "${CCACHE_PROGRAM}"
          CACHE STRING "${LANG} compiler launcher")
    endforeach()
  elseif(SCCACHE_EXECUTABLE)
    foreach(LANG CXX C CUDA)
      set(CMAKE_${LANG}_COMPILER_LAUNCHER
          "${SCCACHE_EXECUTABLE}"
          CACHE STRING "${LANG} compiler launcher")
    endforeach()
  else()
    message(
      STATUS
        "Could not find neither ccache nor sccache. Consider installing ccache to speed up compilation."
    )
  endif()
endif()

if(USE_GLOO_WITH_OPENSSL)
  set(USE_TCP_OPENSSL_LOAD
      ON
      CACHE STRING "")
endif()

# Linux distributions do not want too many embedded sources, in that sense we
# need to be able to build pytorch with an (almost) empty third_party directory.
# USE_SYSTEM_LIBS is a shortcut variable to toggle all the # USE_SYSTEM_*
# variables on. Individual USE_SYSTEM_* variables can be toggled with
# USE_SYSTEM_LIBS being "OFF".
option(USE_SYSTEM_LIBS "Use all available system-provided libraries." OFF)
option(USE_SYSTEM_CPUINFO "Use system-provided cpuinfo." OFF)
option(USE_SYSTEM_SLEEF "Use system-provided sleef." OFF)
option(USE_SYSTEM_GLOO "Use system-provided gloo." OFF)
option(USE_SYSTEM_PYBIND11 "Use system-provided PyBind11." OFF)
option(USE_SYSTEM_BENCHMARK "Use system-provided google benchmark." OFF)
option(USE_SYSTEM_NVTX "Use system-provided nvtx." OFF)
if(USE_SYSTEM_LIBS)
  set(USE_SYSTEM_CPUINFO ON)
  set(USE_SYSTEM_SLEEF ON)
  set(USE_SYSTEM_GLOO ON)
  set(USE_SYSTEM_EIGEN_INSTALL ON)
  set(USE_SYSTEM_BENCHMARK ON)
  set(USE_SYSTEM_PYBIND11 ON)
  if(USE_NCCL)
    set(USE_SYSTEM_NCCL ON)
  endif()
  set(USE_SYSTEM_NVTX ON)
endif()

set(SELECTED_OP_LIST
    ""
    CACHE
      STRING
      "Path to the yaml file that contains the list of operators to include for custom build. Include all operators by default."
)
option(
  STATIC_DISPATCH_BACKEND
  "Name of the backend for which static dispatch code is generated, e.g.: CPU."
  "")
option(
  USE_LIGHTWEIGHT_DISPATCH
  "Enable codegen unboxing for ATen ops, need to work with static dispatch in order to work properly."
  OFF)
if(USE_LIGHTWEIGHT_DISPATCH AND NOT STATIC_DISPATCH_BACKEND)
  message(
    FATAL_ERROR
      "Need to enable static dispatch after enabling USE_LIGHTWEIGHT_DISPATCH.")
endif()
option(TRACING_BASED
       "Master flag to build Lite Interpreter with tracing build option" OFF)
if(LINUX)
  set(ENV_LDFLAGS "$ENV{LDFLAGS}")
  string(STRIP "${ENV_LDFLAGS}" ENV_LDFLAGS)
  # Do not append linker flags passed via env var if they already there
  if(NOT ${CMAKE_SHARED_LINKER_FLAGS} MATCHES "${ENV_LDFLAGS}")
     set(CMAKE_SHARED_LINKER_FLAGS
         "${CMAKE_SHARED_LINKER_FLAGS} ${ENV_LDFLAGS}")
  endif()
endif()


string(APPEND CMAKE_CUDA_FLAGS " -compress-mode=size")

# Let nvcc split device codegen across threads. Two keeps the product with
# MAX_JOBS within the core count on the machines this fork is built on.
string(APPEND CMAKE_CUDA_FLAGS " --split-compile=2")

# Set linker max-page-size to 64KiB on AArch64 Linux
# And stub group size to 32MiB to workaround R_AARCH64_CALL26
# relocation errors
if(LINUX AND CPU_AARCH64)
  add_link_options_if_supported("-z,max-page-size=0x10000")
  add_link_options_if_supported("--stub-group-size=0x2000000")
endif()

# INTERN_BUILD_ATEN_OPS is used to control whether to build ATen/TH operators.
set(INTERN_BUILD_ATEN_OPS ON)

if(NOT DEFINED USE_BLAS)
  set(USE_BLAS ON)
endif()

# Prioritized Text Linker Optimization
if(USE_PRIORITIZED_TEXT_FOR_LD)
  append_cxx_flag_if_supported("-ffunction-sections" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-fdata-sections" CMAKE_CXX_FLAGS)
  append_c_flag_if_supported("-ffunction-sections" CMAKE_C_FLAGS)
  append_c_flag_if_supported("-fdata-sections" CMAKE_C_FLAGS)

  set(LINKER_SCRIPT_FILE_IN "${CMAKE_SOURCE_DIR}/cmake/prioritized_text.txt")
  set(LINKER_SCRIPT_FILE_OUT "${CMAKE_SOURCE_DIR}/cmake/linker_script.ld")
  set(LINKER_SCRIPT_PATH "${CMAKE_SOURCE_DIR}/tools/setup_helpers/generate_linker_script.py")
  add_custom_command(
    OUTPUT "${LINKER_SCRIPT_FILE_OUT}"
    COMMAND ${Python_EXECUTABLE}
            "${LINKER_SCRIPT_PATH}"
            --filein "${LINKER_SCRIPT_FILE_IN}"
            --fout   "${LINKER_SCRIPT_FILE_OUT}"
    DEPENDS "${LINKER_SCRIPT_PATH}"
            "${LINKER_SCRIPT_FILE_IN}"
    VERBATIM
  )
  add_custom_target(generate_linker_script DEPENDS "${LINKER_SCRIPT_FILE_OUT}")
elseif(LINUX AND CPU_AARCH64)
  message(WARNING [[
  It is strongly recommend to enable linker script optimization for all AArch64 Linux builds.
  To do so please export USE_PRIORITIZED_TEXT_FOR_LD=1
  ]])
endif()

if(USE_LLVM_BOLT)
  # In addition to the flags below, BOLT also requires --emit-relocs for the
  # linker. Since --emit-relocs increases the binary size, we limit the use to
  # libraries that are actually optimized by BOLT instead of applying globally
  # (see torch_optimize_layout_if_enabled)
  list(APPEND _bolt_flags
    "-fno-plt" # shows a perf boost in NVIDIA internal CI
    "-fno-reorder-blocks-and-partition" # required by BOLT when using GCC>=8
  )
  foreach(_flag IN LISTS _bolt_flags)
    append_c_flag_if_supported("${_flag}" CMAKE_C_FLAGS)
    append_cxx_flag_if_supported("${_flag}" CMAKE_CXX_FLAGS)
  endforeach()

  # Extract the committed BOLT profiles for the build-time optimization
  # step (torch_optimize_layout_if_enabled in cmake/public/utils.cmake).
  set(LLVM_BOLT_PROFILES_DIR "${CMAKE_BINARY_DIR}/bolt_profiles")
  file(REMOVE_RECURSE "${LLVM_BOLT_PROFILES_DIR}")
  file(ARCHIVE_EXTRACT INPUT "${CMAKE_SOURCE_DIR}/cmake/bolt_profiles/torch.tar.zst"
       DESTINATION "${LLVM_BOLT_PROFILES_DIR}")
endif()

# ---[ Version numbers for generated libraries
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS version.txt)
file(READ version.txt TORCH_DEFAULT_VERSION)
# Strip trailing newline
string(REGEX REPLACE "\n$" "" TORCH_DEFAULT_VERSION "${TORCH_DEFAULT_VERSION}")
if("${TORCH_DEFAULT_VERSION} " STREQUAL " ")
  message(WARNING "Could not get version from base 'version.txt'")
  # If we can't get the version from the version file we should probably set it
  # to something non-sensical like 0.0.0
  set(TORCH_DEFAULT_VERSION "0.0.0")
endif()
set(TORCH_BUILD_VERSION
    "${TORCH_DEFAULT_VERSION}"
    CACHE STRING "Torch build version" FORCE)
if(DEFINED ENV{PYTORCH_BUILD_VERSION})
  set(TORCH_BUILD_VERSION
      "$ENV{PYTORCH_BUILD_VERSION}"
      CACHE STRING "Torch build version" FORCE)
endif()
caffe2_parse_version_str(TORCH ${TORCH_BUILD_VERSION})
set(TORCH_SOVERSION "${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}")

# ---[ CMake scripts + modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ---[ CMake build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()

# ---[ Build variables set within the cmake tree
include(cmake/BuildVariables.cmake)
set(CAFFE2_ALLOWLIST
    ""
    CACHE STRING "A allowlist file of files that one should build.")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage."
        FORCE)
endif()

# The below means we are cross compiling for arm64 or x86_64 on MacOSX
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin"
   AND CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64|arm64)$")
  set(CROSS_COMPILING_MACOSX TRUE)
endif()

# ---[ Misc checks to cope with various compiler modes
include(cmake/MiscCheck.cmake)

# External projects
include(ExternalProject)

# ---[ Dependencies ---[ FBGEMM doesn't work on x86 32bit and
# CMAKE_SYSTEM_PROCESSOR thinks its 64bit
if(USE_FBGEMM AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|aarch64)$")
  message(WARNING
    "x86_64 or aarch64 required for FBGEMM. "
    "Not compiling with FBGEMM. "
    "Turn this warning off by USE_FBGEMM=OFF.")
  set(USE_FBGEMM OFF)
endif()


# Note for ROCM platform: 1. USE_ROCM is always ON until
# include(cmake/Dependencies.cmake) 2. USE_CUDA will become OFF during
# re-configuration Truth Table: CUDA 1st pass: USE_CUDA=True;USE_ROCM=True,
# FLASH evaluates to ON by default CUDA 2nd pass: USE_CUDA=True;USE_ROCM=False,
# FLASH evaluates to ON by default ROCM 1st pass: USE_CUDA=True;USE_ROCM=True,
# FLASH evaluates to ON by default ROCM 2nd pass: USE_CUDA=False;USE_ROCM=True,
# FLASH evaluates to ON by default CPU 1st pass: USE_CUDA=False(Cmd
# Option);USE_ROCM=True, FLASH evaluates to OFF by default CPU 2nd pass:
# USE_CUDA=False(Cmd Option);USE_ROCM=False, FLASH evaluates to OFF by default
# Thus we cannot tell ROCM 2nd pass and CPU 1st pass
#
# The only solution is to include(cmake/Dependencies.cmake), and defer the
# aotriton build decision later.

include(cmake/Dependencies.cmake)

# MKL resolves symbols between its own sub-libraries, so the linker must keep
# every one of them in DT_NEEDED even though torch references only some. Without
# this, libmkl_intel_lp64.so fails at run time with undefined symbol mkl_blas_dsyrk.
# https://software.intel.com/en-us/articles/symbol-lookup-error-when-linking-intel-mkl-with-gcc
if(LINUX AND CAFFE2_USE_MKL)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

cmake_dependent_option(
  USE_FLASH_ATTENTION
  "Whether to build the flash_attention kernel for scaled dot product attention.\
  Will be disabled if not supported by the platform"
  ON
  "USE_CUDA OR USE_ROCM"
  OFF)

# Flash Attention needs sm80+; disable it if no resolved arch reaches that.
if(USE_FLASH_ATTENTION AND USE_CUDA AND NOT USE_ROCM)
  string(REPLACE " " ";" _flash_attn_archs "${NVCC_FLAGS_EXTRA_readable}")
  set(_flash_attn_has_sm80 OFF)
  foreach(_arch IN LISTS _flash_attn_archs)
    if(_arch MATCHES "^(sm|compute)_([0-9]+)" AND CMAKE_MATCH_2 GREATER_EQUAL 80)
      set(_flash_attn_has_sm80 ON)
    endif()
  endforeach()
  if(_flash_attn_archs AND NOT _flash_attn_has_sm80)
    message(STATUS
      "Disabling USE_FLASH_ATTENTION: it needs a CUDA architecture >= 8.0, "
      "but the resolved CUDA architectures are '${_flash_attn_archs}'")
    set(USE_FLASH_ATTENTION OFF)
  endif()
endif()


IF(USE_ROCM AND ("gfx942" IN_LIST PYTORCH_ROCM_ARCH OR "gfx950" IN_LIST PYTORCH_ROCM_ARCH))
  message(WARNING "Setting USE_MSLK for gfx942/gfx950 to ON by default, doing ROCM build")
  set(USE_MSLK_DEFAULT ON)
elseif(USE_CUDA AND "$ENV{TORCH_CUDA_ARCH_LIST}" MATCHES "10.0" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  message(STATUS "Setting USE_MSLK to ON by default , doing CUDA build for SM100a")
  set(USE_MSLK_DEFAULT ON)
else()
  set(USE_MSLK_DEFAULT OFF)
endif()

cmake_dependent_option(
  USE_MSLK
  "Whether to build MSLK quantized GEMM kernels.\
  Will be disabled if not supported by the platform"
  ${USE_MSLK_DEFAULT}
  "USE_CUDA OR USE_ROCM"
  OFF)


# CAVEAT: Again, Flash Attention2 will error while building for sm52 while Mem
# Eff Attention won't
cmake_dependent_option(
  USE_MEM_EFF_ATTENTION
  "Enable memory-efficient attention for scaled dot product attention.\
  Will be disabled if not supported by the platform" ON
  "USE_CUDA OR USE_ROCM" OFF)

#
# Cannot be put into Dependencies.cmake due circular dependency:
# USE_FLASH_ATTENTION -> USE_ROCM -> Dependencies.cmake -> aotriton.cmake
#
if(USE_ROCM)
  if(USE_FLASH_ATTENTION OR USE_MEM_EFF_ATTENTION)
    include(cmake/External/aotriton.cmake)
  endif()
endif()

if(DEBUG_CUDA)
  string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -lineinfo")
  string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -lineinfo")
endif(DEBUG_CUDA)

if(USE_FBGEMM)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_FBGEMM")
endif()

if(USE_MSLK)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_MSLK")
endif()

# Enable sleef on macOS with Apple silicon by default
if((${CMAKE_SYSTEM_NAME} STREQUAL "Darwin") AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64"))
  message(STATUS "Running on macOS with Apple silicon")
  string(APPEND CMAKE_CXX_FLAGS " -DAT_BUILD_ARM_VEC256_WITH_SLEEF")
  add_definitions(-DAT_BUILD_ARM_VEC256_WITH_SLEEF)
endif()

# Enable sleef on Arm(R) architecture by default (except Android)
if((NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  AND("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64"))
  string(APPEND CMAKE_CXX_FLAGS " -DAT_BUILD_ARM_VEC256_WITH_SLEEF")
  add_definitions(-DAT_BUILD_ARM_VEC256_WITH_SLEEF)
endif()

if(TRACING_BASED)
  string(APPEND CMAKE_CXX_FLAGS " -DTRACING_BASED")
endif()

# ---[ Allowlist file if allowlist is specified
include(cmake/Allowlist.cmake)

# ---[ Build flags Re-include to override append_cxx_flag_if_supported from
# third_party/FBGEMM
include(cmake/public/utils.cmake)
if(USE_COLORIZE_OUTPUT)
  set(CMAKE_COLOR_DIAGNOSTICS ON)
endif()
string(APPEND CMAKE_CXX_FLAGS " -O2 -fPIC")

# This prevents use of `c10::optional`, `c10::nullopt` etc within the codebase
string(APPEND CMAKE_CXX_FLAGS " -DC10_NODEPRECATED")
string(APPEND CMAKE_CUDA_FLAGS " -DC10_NODEPRECATED")
string(APPEND CMAKE_OBJCXX_FLAGS " -DC10_NODEPRECATED")

# Eigen fails to build with some versions, so convert this to a warning
# Details at http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1459
string(APPEND CMAKE_CXX_FLAGS " -Wall")
string(APPEND CMAKE_CXX_FLAGS " -Wextra")
append_cxx_flag_if_supported("-Werror=return-type" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Werror=non-virtual-dtor" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Werror=braced-scalar-init" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Werror=range-loop-construct" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Werror=bool-operation" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wnarrowing" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-missing-field-initializers"
                             CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-unknown-pragmas" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-unused-parameter" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-strict-overflow" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-strict-aliasing" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-stringop-overflow" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wvla-extension" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wsuggest-override" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wnewline-eof" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Winconsistent-missing-override"
                             CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Winconsistent-missing-destructor-override"
                             CMAKE_CXX_FLAGS)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-pass-failed")
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
  # Suppress "The ABI for passing parameters with 64-byte alignment has
  # changed in GCC 4.6"
  string(APPEND CMAKE_CXX_FLAGS " -Wno-psabi")
endif()

append_cxx_flag_if_supported("-Wno-error=old-style-cast" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wconstant-conversion" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-aligned-allocation-unavailable"
                             CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Qunused-arguments" CMAKE_CXX_FLAGS)

append_cxx_flag_if_supported("-faligned-new" CMAKE_CXX_FLAGS)

if(WERROR)
  append_cxx_flag_if_supported("-Werror" CMAKE_CXX_FLAGS)
  if(NOT COMPILER_SUPPORT_WERROR)
    set(WERROR FALSE)
  endif()
endif()
append_cxx_flag_if_supported("-Wno-maybe-uninitialized" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-fstandalone-debug" CMAKE_CXX_FLAGS_DEBUG)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(WARNING "Applying -Og optimization for aarch64 GCC debug build to workaround ICE")
  endif()
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -Og")
else()
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
endif()
# aarch64 C++ stack unwinding uses frame-pointer chain walking, so frame
# pointers must be present in all build types.  The cost is negligible on
# aarch64 (31 GPRs vs x86-64's 16, so dedicating x29 rarely spills).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  append_cxx_flag_if_supported("-fno-omit-frame-pointer" CMAKE_CXX_FLAGS)
endif()
append_cxx_flag_if_supported("-fno-math-errno" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-fno-trapping-math" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Werror=format" CMAKE_CXX_FLAGS)
if(CMAKE_COMPILER_IS_GNUCXX)
  append_cxx_flag_if_supported("-Wno-dangling-reference" CMAKE_CXX_FLAGS)
  append_cxx_flag_if_supported("-Wno-error=dangling-reference" CMAKE_CXX_FLAGS)
endif()
# needed for compat with newer versions of clang that use C++20 mangling rules
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  append_cxx_flag_if_supported("-fclang-abi-compat=17" CMAKE_CXX_FLAGS)
endif()

# Add code coverage flags to supported compilers
if(USE_CPP_CODE_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    string(APPEND CMAKE_C_FLAGS " --coverage -fprofile-abs-path")
    string(APPEND CMAKE_CXX_FLAGS " --coverage -fprofile-abs-path")
    string(APPEND CMAKE_OBJCXX_FLAGS " --coverage -fprofile-abs-path")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_C_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND CMAKE_CXX_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND CMAKE_OBJCXX_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
  else()
    message(
      FATAL_ERROR
      "Code coverage for compiler ${CMAKE_CXX_COMPILER_ID} is unsupported")
  endif()

endif()

if(APPLE)
  if(USE_MPS)
    string(APPEND CMAKE_OBJCXX_FLAGS " -DUSE_MPS -fno-objc-arc")
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_MPS")
    string(
      APPEND
      CMAKE_SHARED_LINKER_FLAGS
      " -weak_framework Foundation -weak_framework MetalPerformanceShaders -weak_framework MetalPerformanceShadersGraph -weak_framework Metal -weak_framework IOKit"
    )
    # To suppress MPSGraph availability warnings
    append_cxx_flag_if_supported("-Wno-unguarded-availability-new"
                                 CMAKE_OBJCXX_FLAGS)
  endif()
  append_cxx_flag_if_supported("-Wno-missing-braces" CMAKE_CXX_FLAGS)
endif()

if(EMSCRIPTEN)
  string(
    APPEND
    CMAKE_CXX_FLAGS
    " -Wno-implicit-function-declaration -DEMSCRIPTEN -s DISABLE_EXCEPTION_CATCHING=0"
  )
endif()

append_cxx_flag_if_supported("-Wno-stringop-overflow" CMAKE_CXX_FLAGS)

if(NOT APPLE AND UNIX)
  list(APPEND Caffe2_DEPENDENCY_LIBS dl)
endif()

# Prefix path to Caffe2 headers. If a directory containing installed Caffe2
# headers was inadvertently added to the list of include directories, prefixing
# PROJECT_SOURCE_DIR means this source tree always takes precedence.
include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# Prefix path to generated Caffe2 headers. These need to take precedence over
# their empty counterparts located in PROJECT_SOURCE_DIR.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

include_directories(BEFORE ${PROJECT_SOURCE_DIR}/aten/src/)
include_directories(BEFORE ${CMAKE_BINARY_DIR}/aten/src/)

if(USE_MIMALLOC)
  set(MI_OVERRIDE OFF)
  set(MI_BUILD_SHARED OFF)
  set(MI_BUILD_OBJECT OFF)
  set(MI_BUILD_TESTS OFF)
  add_definitions(-DUSE_MIMALLOC)
  add_subdirectory(third_party/mimalloc)
  include_directories(third_party/mimalloc/include)
endif()

if(USE_MIMALLOC AND USE_MIMALLOC_ON_MKL)
  add_definitions(-DUSE_MIMALLOC_ON_MKL)
endif()

# ---[ Main build
add_subdirectory(torch/headeronly)  # headeronly headers
add_subdirectory(c10)
add_subdirectory(caffe2)

# ---[ CMake related files Uninstall option.
if(NOT TARGET caffe2_uninstall)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake IMMEDIATE @ONLY)

  add_custom_target(
    caffe2_uninstall COMMAND ${CMAKE_COMMAND} -P
                             ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# ---[ Make configuration files for cmake to allow dependent libraries easier
# access to Caffe2.



# Note(jiayq): when building static libraries, all PRIVATE dependencies will
# also become interface libraries, and as a result if there are any dependency
# libraries that are not exported, the following install export script will
# fail. As a result, we will only provide the targets cmake files for shared lib
# installation. For more info, read:
# https://cmake.org/pipermail/cmake/2016-May/063400.html
if(BUILD_SHARED_LIBS)
  configure_file(${PROJECT_SOURCE_DIR}/cmake/Caffe2Config.cmake.in
                 ${PROJECT_BINARY_DIR}/Caffe2Config.cmake @ONLY)
  install(
    FILES ${PROJECT_BINARY_DIR}/Caffe2Config.cmake
    DESTINATION share/cmake/Caffe2
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/public/cuda.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/xpu.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/mkl.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/utils.cmake
          ${PROJECT_SOURCE_DIR}/cmake/public/LoadHIP.cmake
    DESTINATION share/cmake/Caffe2/public
    COMPONENT dev)
  install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/Modules_CUDA_fix
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindCUDAToolkit.cmake
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindCUSPARSELT.cmake
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindCUDSS.cmake
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindSYCLToolkit.cmake
    DESTINATION share/cmake/Caffe2/
    COMPONENT dev)
  if(NOT BUILD_LIBTORCHLESS)
    install(
      EXPORT Caffe2Targets
      DESTINATION share/cmake/Caffe2
      FILE Caffe2Targets.cmake
      COMPONENT dev)
  endif()
else()
  message(WARNING "Generated cmake files are only available when building "
                  "shared libs.")
endif()

# ---[ Binaries Binaries will be built after the Caffe2 main libraries and the
# modules are built. For the binaries, they will be linked to the Caffe2 main
# libraries, as well as all the modules that are built with Caffe2 (the ones
# built in the previous Modules section above).
if(BUILD_BINARY)
  add_subdirectory(binaries)
endif()

# Parse custom debug info
if(DEFINED USE_CUSTOM_DEBINFO)
  string(REPLACE ";" " " SOURCE_FILES "${USE_CUSTOM_DEBINFO}")
  message(STATUS "Source files with custom debug infos: ${SOURCE_FILES}")

  string(REGEX REPLACE " +" ";" SOURCE_FILES_LIST "${SOURCE_FILES}")

  # Set the COMPILE_FLAGS property for each source file
  foreach(SOURCE_FILE ${SOURCE_FILES_LIST})
    # We have to specify the scope here. We do this by specifying the targets we
    # care about and caffe2/ for all test targets defined there
    if(BUILD_LIBTORCHLESS)
      caffe2_update_option(USE_CUDA OFF)
      set(ALL_PT_TARGETS "torch_python;${C10_LIB};${TORCH_CPU_LIB};${TORCH_LIB}")
    else()
      # @todo test if we can remove this
      set(ALL_PT_TARGETS "torch_python;c10;torch_cpu;torch")
    endif()
    set_source_files_properties(
      ${SOURCE_FILE} DIRECTORY "caffe2/" TARGET_DIRECTORY ${ALL_PT_TARGETS}
      PROPERTIES COMPILE_FLAGS "-g")
  endforeach()

  # Link everything with debug info when any file is in debug mode
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")
endif()

# Bundle PTXAS if needed
if(BUILD_BUNDLE_PTXAS AND USE_CUDA)
  message(STATUS "Copying PTXAS into the bin folder")
  file(COPY "${CUDAToolkit_BIN_DIR}/ptxas"
       DESTINATION "${PROJECT_BINARY_DIR}")
  install(PROGRAMS "${PROJECT_BINARY_DIR}/ptxas"
          DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

if(SKBUILD)
  # File mirroring: copy source files into torchgen/packaged/, vendored
  # inductor templates, and symlink-replacement copies.  Previously handled
  # by setup.py.
  include(cmake/FileMirroring.cmake)

  # Package data: install non-Python data files (type stubs, templates,
  # headers, etc.) into the torch package.  Previously declared via
  # setup.py package_data.
  include(cmake/PackageData.cmake)
endif()

include(cmake/PostBuildSteps.cmake)

include(cmake/Summary.cmake)
caffe2_print_configuration_summary()

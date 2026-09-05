# ---[ cuda

# Poor man's include guard
if(TARGET torch::cudart)
  return()
endif()

# We don't want to statically link cudart, because we rely on it's dynamic linkage in
# python (follow along torch/cuda/__init__.py and usage of cudaGetErrorName).
# Technically, we can link cudart here statically, and link libtorch_python.so
# to a dynamic libcudart.so, but that's just wasteful.
# However, on Windows, if this one gets switched off, the error "cuda: unknown error"
# will be raised when running the following code:
# >>> import torch
# >>> torch.cuda.is_available()
# >>> torch.cuda.current_device()
# More details can be found in the following links.
# https://github.com/pytorch/pytorch/issues/20635
# https://github.com/pytorch/pytorch/issues/17108
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")

# Find CUDA. CUDA_HOME is the historical PyTorch spelling of the install
# location; CMake's own module only looks at CUDAToolkit_ROOT and CUDA_PATH.
if(NOT CUDAToolkit_ROOT AND DEFINED ENV{CUDA_HOME})
  set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}")
endif()
find_package(CUDAToolkit)
if(NOT CUDAToolkit_FOUND)
  # If user explicitly set USE_CUDA=1, error out instead of falling back
  if(_USE_CUDA_EXPLICITLY_SET AND USE_CUDA)
    message(FATAL_ERROR
      "PyTorch: CUDA was explicitly requested (USE_CUDA=1) but cannot be found. "
      "Please check your CUDA installation, ensure CUDA toolkit is installed, "
      "and that CUDA_HOME or CMAKE_CUDA_COMPILER is set correctly. "
      "If you want to build without CUDA, please set USE_CUDA=0.")
  endif()

  message(WARNING
    "PyTorch: CUDA cannot be found. Depending on whether you are building "
    "PyTorch or a PyTorch dependent library, the next warning / error will "
    "give you more info.")
  set(CAFFE2_USE_CUDA OFF)
  return()
endif()

# The Find modules for the CUDA satellite libraries and cmake/External/nccl.cmake
# take the toolkit root as a hint; FindCUDAToolkit does not define it. Likewise
# CUDA_VERSION, the major.minor spelling that ATen/BuildInfo.h and the vendored
# cutlass builds read.
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_BIN_DIR}" DIRECTORY)
set(CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
# nccl.cmake passes this to NCCL's makefile as NVCC=; an empty value there
# overrides NCCL's own default instead of falling back to it. It can already be
# set from the environment (see cmake/EnvVarForwarding.cmake).
if(NOT CUDA_NVCC_EXECUTABLE)
  set(CUDA_NVCC_EXECUTABLE "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()

# enable_language(CUDA) below fills CMAKE_CUDA_ARCHITECTURES in with the
# compiler default, so whether the user asked for one has to be answered first.
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(WARNING
          "pytorch is not compatible with `CMAKE_CUDA_ARCHITECTURES` and will ignore its value. "
          "Please configure `TORCH_CUDA_ARCH_LIST` instead.")
endif()

# Enable CUDA language support. The compiler search only looks at CUDACXX and
# PATH, so hand it the nvcc that the toolkit search above settled on.
if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()
# Pass clang as host compiler, which according to the docs
# Must be done before CUDA language is enabled, see
# https://cmake.org/cmake/help/v3.15/variable/CMAKE_CUDA_HOST_COMPILER.html
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()
enable_language(CUDA)
# Device code is C++23. nvcc rejects -std=c++26 outright, so this is as far as
# device code can follow the host.
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# nvcc's host pass, and only it, is pointed at libstdc++ 15. nvcc's EDG frontend
# has not implemented P1787R6, the C++23 relaxation that lets a ref-qualified
# and a non-ref-qualified member function share one overload set; libstdc++ 16
# relies on it for P2438's `basic_string::substr() &&`, so under nvcc a bare
# `#include <string>` fails to compile. libstdc++ 15 predates P2438.
#
# Scoped to CMAKE_CUDA_FLAGS on purpose: host C++ and every other backend keep
# the standard library they were configured with. The halves then disagree on
# libstdc++ minor version, which is the ordinary `-ccbin <older gcc>` situation
# and safe across a shared ABI -- but it does mean shared headers must not
# branch on library feature-test macros, since the two passes see different
# values for them.
set(CUDA_HOST_GCC_INSTALL_DIR "" CACHE PATH
    "GCC install directory whose libstdc++ nvcc's host pass compiles against")

if(NOT CUDA_HOST_GCC_INSTALL_DIR)
  file(GLOB _cuda_gcc15_dirs LIST_DIRECTORIES true "/usr/lib/gcc/*/15" "/usr/lib/gcc/*/15.*")
  list(SORT _cuda_gcc15_dirs COMPARE NATURAL ORDER DESCENDING)
  foreach(_dir ${_cuda_gcc15_dirs})
    if(IS_DIRECTORY "${_dir}/include/c++")
      set(CUDA_HOST_GCC_INSTALL_DIR "${_dir}" CACHE PATH "" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT CUDA_HOST_GCC_INSTALL_DIR)
  message(FATAL_ERROR
    "Device code is built as C++23, which nvcc can only do against libstdc++ 15 "
    "or older. No such toolchain was found under /usr/lib/gcc. Install GCC 15, "
    "or set CUDA_HOST_GCC_INSTALL_DIR to a directory holding an older one.")
endif()

string(APPEND CMAKE_CUDA_FLAGS
       " -Xcompiler --gcc-install-dir=${CUDA_HOST_GCC_INSTALL_DIR}")
message(STATUS "CUDA host pass uses libstdc++ from ${CUDA_HOST_GCC_INSTALL_DIR}")


if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL CUDAToolkit_VERSION)
  message(FATAL_ERROR "Found two conflicting CUDA versions:\n"
                      "V${CMAKE_CUDA_COMPILER_VERSION} from ${CMAKE_CUDA_COMPILER} and\n"
                      "V${CUDAToolkit_VERSION} in '${CUDAToolkit_INCLUDE_DIRS}'")
endif()

message(STATUS "PyTorch: CUDA detected: " ${CUDAToolkit_VERSION})
message(STATUS "PyTorch: CUDA nvcc is: " ${CUDAToolkit_NVCC_EXECUTABLE})
message(STATUS "PyTorch: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})
if(CUDAToolkit_VERSION VERSION_LESS 13.3)
  message(FATAL_ERROR "PyTorch requires CUDA 13.3 or above.")
endif()

# ---[ CUDA libraries wrapper

# find libnvrtc.so
set(CUDA_NVRTC_LIB "${CUDA_nvrtc_LIBRARY}" CACHE FILEPATH "")
if(CUDA_NVRTC_LIB AND NOT CUDA_NVRTC_SHORTHASH)
  file(SHA256 "${CUDA_NVRTC_LIB}" _cuda_nvrtc_sha256)
  string(SUBSTRING "${_cuda_nvrtc_sha256}" 0 8 CUDA_NVRTC_SHORTHASH)
  message(STATUS "${CUDA_NVRTC_LIB} shorthash is ${CUDA_NVRTC_SHORTHASH}")
endif()

# Create new style imported libraries.
# Several of these libraries have a hardcoded path if CAFFE2_STATIC_LINK_CUDA
# is set. This path is where sane CUDA installations have their static
# libraries installed. This flag should only be used for binary builds, so
# end-users should never have this flag set.

# cuda
add_library(caffe2::cuda INTERFACE IMPORTED)
set_property(
    TARGET caffe2::cuda PROPERTY INTERFACE_LINK_LIBRARIES
    CUDA::cuda_driver)

# cudart
add_library(torch::cudart INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        CUDA::cudart_static)
else()
    set_property(
        TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        CUDA::cudart)
endif()


# cublas
add_library(caffe2::cublas INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
        # NOTE: cublas is always linked dynamically
        CUDA::cublas CUDA::cublasLt)
    set_property(
        TARGET caffe2::cublas APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        CUDA::cudart_static rt)
else()
    set_property(
        TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
        CUDA::cublas CUDA::cublasLt)
endif()

# cudnn interface
# static linking is handled by USE_STATIC_CUDNN environment variable
if(CAFFE2_USE_CUDNN)
  if(USE_STATIC_CUDNN)
    set(CUDNN_STATIC ON CACHE BOOL "")
  else()
    set(CUDNN_STATIC OFF CACHE BOOL "")
  endif()

  find_package(CUDNN)

  if(NOT CUDNN_FOUND)
    message(WARNING
      "Cannot find cuDNN library. Turning the option off")
    set(CAFFE2_USE_CUDNN OFF)
  else()
    if(CUDNN_VERSION VERSION_LESS "8.1.0")
      message(FATAL_ERROR "PyTorch requires cuDNN 8.1 and above.")
    endif()
  endif()

  add_library(torch::cudnn INTERFACE IMPORTED)
  target_include_directories(torch::cudnn INTERFACE ${CUDNN_INCLUDE_PATH})
  if(CUDNN_STATIC)
    target_link_options(torch::cudnn INTERFACE
        "LINKER:--exclude-libs,libcudnn_static.a")
  else()
    target_link_libraries(torch::cudnn INTERFACE ${CUDNN_LIBRARY_PATH})
  endif()
else()
  message(STATUS "USE_CUDNN is set to 0. Compiling without cuDNN support")
endif()

if(CAFFE2_USE_CUSPARSELT)
  find_package(CUSPARSELT)

  if(NOT CUSPARSELT_FOUND)
    message(WARNING
      "Cannot find cuSPARSELt library. Turning the option off")
    set(CAFFE2_USE_CUSPARSELT OFF)
  else()
    add_library(torch::cusparselt INTERFACE IMPORTED)
    target_include_directories(torch::cusparselt INTERFACE ${CUSPARSELT_INCLUDE_PATH})
    target_link_libraries(torch::cusparselt INTERFACE ${CUSPARSELT_LIBRARY_PATH})
  endif()
else()
  message(STATUS "USE_CUSPARSELT is set to 0. Compiling without cuSPARSELt support")
endif()

if(USE_CUDSS)
  find_package(CUDSS)

  if(NOT CUDSS_FOUND)
    message(WARNING
      "Cannot find CUDSS library. Turning the option off")
    set(USE_CUDSS OFF)
  else()
    add_library(torch::cudss INTERFACE IMPORTED)
    target_include_directories(torch::cudss INTERFACE ${CUDSS_INCLUDE_PATH})
    target_link_libraries(torch::cudss INTERFACE ${CUDSS_LIBRARY_PATH})
  endif()
else()
  message(STATUS "USE_CUDSS is set to 0. Compiling without cuDSS support")
endif()

# cufile
if(CAFFE2_USE_CUFILE)
  add_library(torch::cufile INTERFACE IMPORTED)
  if(CAFFE2_STATIC_LINK_CUDA)
      set_property(
          TARGET torch::cufile PROPERTY INTERFACE_LINK_LIBRARIES
          CUDA::cuFile_static)
  else()
      set_property(
          TARGET torch::cufile PROPERTY INTERFACE_LINK_LIBRARIES
          CUDA::cuFile)
  endif()
else()
  message(STATUS "USE_CUFILE is set to 0. Compiling without cuFile support")
endif()

# nvrtc
# cuDNN frontend needs libnvrtc symbols, but linking through CUDA::nvrtc pulls
# CUDA::cuda_driver transitively. Keep a driver-free target for cuDNN users and
# reserve caffe2::nvrtc for the stub library that actually needs the driver API.
add_library(caffe2::nvrtc_runtime INTERFACE IMPORTED)
set_property(
    TARGET caffe2::nvrtc_runtime PROPERTY INTERFACE_LINK_LIBRARIES
    "${CUDA_NVRTC_LIB}")

add_library(caffe2::nvrtc INTERFACE IMPORTED)
set_property(
    TARGET caffe2::nvrtc PROPERTY INTERFACE_LINK_LIBRARIES
    CUDA::nvrtc caffe2::cuda)

# setting nvcc arch flags
# The architectures go in as explicit -gencode flags below, so CMake must not
# add any of its own.
torch_cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)
set(CMAKE_CUDA_ARCHITECTURES OFF)

list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA}")

# Debug and Release symbol support
if(CUDA_DEVICE_DEBUG)
  list(APPEND CUDA_NVCC_FLAGS "-g" "-G")  # -G enables device code debugging symbols
endif()

# needed for compat with newer versions of clang that use C++20 mangling rules
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fclang-abi-compat=17")
endif()

# Required by headers that call constexpr host functions from device code.
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# Set expt-extended-lambda to support lambda on device
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

foreach(FLAG ${CUDA_NVCC_FLAGS})
  string(FIND "${FLAG}" " " flag_space_position)
  if(NOT flag_space_position EQUAL -1)
    message(FATAL_ERROR "Found spaces in CUDA_NVCC_FLAGS entry '${FLAG}'")
  endif()
  string(APPEND CMAKE_CUDA_FLAGS " ${FLAG}")
endforeach()

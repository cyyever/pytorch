# This file used to build libtorch.so.
# Now it only builds the Torch python bindings.

if(NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  cmake_minimum_required(VERSION 3.29 FATAL_ERROR)
  project(torch CXX C)
  find_package(torch REQUIRED)
  option(USE_CUDA "Use CUDA" ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(NOT BUILD_PYTHON)
  return()
endif()

set(TORCH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_ROOT "${TORCH_SRC_DIR}/..")

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

set(LIBSHM_SUBDIR libshm)

set(LIBSHM_SRCDIR ${TORCH_SRC_DIR}/lib/${LIBSHM_SUBDIR})
add_subdirectory(${LIBSHM_SRCDIR})

# Generate files
set(TOOLS_PATH "${TORCH_ROOT}/tools")


set(TORCH_PYTHON_SRCS
    ${GENERATED_CXX_PYTHON}
    )
append_filelist("libtorch_python_core_sources" TORCH_PYTHON_SRCS)

if(BUILD_TEST)
    add_definitions(-DBUILDING_TESTS)
endif()

set(TORCH_PYTHON_INCLUDE_DIRECTORIES
    ${PYTHON_INCLUDE_DIR}

    ${TORCH_ROOT}
    ${TORCH_ROOT}/aten/src

    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/aten/src
    ${CMAKE_BINARY_DIR}/caffe2/aten/src
    ${CMAKE_BINARY_DIR}/third_party

    ${TORCH_ROOT}/third_party/gloo
    ${TORCH_ROOT}/third_party/kineto/libkineto/include
    ${TORCH_ROOT}/third_party/nlohmann/include

    ${TORCH_SRC_DIR}/csrc
    ${TORCH_SRC_DIR}/csrc/api/include
    ${TORCH_SRC_DIR}/lib
    ${TORCH_SRC_DIR}/standalone
    )

list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${LIBSHM_SRCDIR})

set(TORCH_PYTHON_LINK_LIBRARIES
    Python::Module
    pybind::pybind11
    nlohmann
    moodycamel
    shm
    fmt::fmt-header-only
    ATEN_CPU_FILES_GEN_LIB)

if(USE_ASAN AND TARGET Sanitizer::address)
  list(APPEND TORCH_PYTHON_LINK_LIBRARIES Sanitizer::address)
endif()
# Skip UBSan on ROCm: see comment in cmake/Dependencies.cmake (UBSan + ASAN
# on ROCm Clang produces misaligned ASAN global metadata that aborts at
# startup).
if(USE_ASAN AND TARGET Sanitizer::undefined AND NOT USE_ROCM)
  list(APPEND TORCH_PYTHON_LINK_LIBRARIES Sanitizer::undefined)
endif()
if(USE_LSAN AND TARGET Sanitizer::leak)
  list(APPEND TORCH_PYTHON_LINK_LIBRARIES Sanitizer::leak)
endif()
if(USE_TSAN AND TARGET Sanitizer::thread)
  list(APPEND TORCH_PYTHON_LINK_LIBRARIES Sanitizer::thread)
endif()

set(TORCH_PYTHON_COMPILE_DEFINITIONS)

set(TORCH_PYTHON_COMPILE_OPTIONS)

set(TORCH_PYTHON_LINK_FLAGS "")

if(APPLE)
    string(APPEND TORCH_PYTHON_LINK_FLAGS " -undefined dynamic_lookup")
else()
    list(APPEND TORCH_PYTHON_COMPILE_OPTIONS
      -fno-strict-aliasing
      -Wno-strict-aliasing)
endif()

if(USE_CUDA)
    include(${TORCH_ROOT}/cmake/public/cuda.cmake)
    append_filelist("libtorch_python_cuda_core_sources" TORCH_PYTHON_SRCS)

    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDA)
    if(USE_CUDNN)
        list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::cudnn)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDNN)
    endif()
    if(USE_CUSPARSELT)
        list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::cusparselt)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUSPARSELT)
    endif()
    if(USE_CUFILE)
        list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::cufile)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUFILE)
    endif()

    if(TARGET torch::nvtx3)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::nvtx3)
    else()
      if(TARGET torch::nvtoolsext)
        list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::nvtoolsext)
      endif()
    endif()
endif()

if(USE_ROCM)
    append_filelist("libtorch_python_cuda_core_sources" TORCH_PYTHON_SRCS)

    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS
      USE_ROCM
      __HIP_PLATFORM_AMD__
      )
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${ROCM_ROCTX_LIB})
    if(USE_HIPSPARSELT)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_HIPSPARSELT)
    endif()
    if(USE_CUFILE)
        list(APPEND TORCH_PYTHON_LINK_LIBRARIES hip::hipfile)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUFILE)
    endif()
endif()

if(USE_XPU)
    include(${TORCH_ROOT}/cmake/public/xpu.cmake)
    append_filelist("libtorch_python_xpu_sources" TORCH_PYTHON_SRCS)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_XPU)
endif()

if(USE_CUDNN OR USE_ROCM)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/shared/cudnn.cpp
      )
    if(USE_STATIC_CUDNN)
        set_source_files_properties(
          ${TORCH_SRC_DIR}/csrc/cuda/shared/cudnn.cpp
          PROPERTIES COMPILE_DEFINITIONS "USE_STATIC_CUDNN"
        )
    endif()
endif()

if(USE_CUSPARSELT OR USE_HIPSPARSELT)
    list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/cuda/shared/cusparselt.cpp)
endif()

if(USE_MPS)
    list(APPEND TORCH_PYTHON_SRCS ${MPS_PYTHON_SRCS})
endif()

add_custom_target(torch_python_stubs DEPENDS
    "${TORCH_SRC_DIR}/_C/__init__.pyi"
    "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi"
    "${TORCH_SRC_DIR}/nn/functional.pyi"
)

file(GLOB_RECURSE torchgen_python CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/torchgen/*.py")
file(GLOB_RECURSE autograd_python CONFIGURE_DEPENDS "${TOOLS_PATH}/autograd/*.py")
file(GLOB_RECURSE pyi_python CONFIGURE_DEPENDS "${TOOLS_PATH}/pyi/*.py")
add_custom_command(
    OUTPUT
    "${TORCH_SRC_DIR}/_C/__init__.pyi"
    "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi"
    "${TORCH_SRC_DIR}/nn/functional.pyi"
    COMMAND
    "${Python_EXECUTABLE}" -mtools.pyi.gen_pyi
      --native-functions-path "aten/src/ATen/native/native_functions.yaml"
      --tags-path "aten/src/ATen/native/tags.yaml"
    DEPENDS
      "${TORCH_SRC_DIR}/_C/__init__.pyi.in"
      "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi.in"
      "${TORCH_SRC_DIR}/nn/functional.pyi.in"
      "${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml"
      "${TORCH_ROOT}/aten/src/ATen/native/tags.yaml"
      ${pyi_python}
      ${autograd_python}
      ${torchgen_python}
    WORKING_DIRECTORY
    "${TORCH_ROOT}"
)
if(USE_DISTRIBUTED)
    append_filelist("libtorch_python_distributed_sources" TORCH_PYTHON_SRCS)
    # Disable certain warnings for GCC-9.X
    if(CMAKE_COMPILER_IS_GNUCXX)
      set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/c10d/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
    endif()
    # NCCL is a private dependency of libtorch, but libtorch_python includes
    # some private headers of libtorch, which in turn include NCCL. As a hacky
    # alternative to making NCCL a public dependency of libtorch, we make it
    # a private dependency of libtorch_python as well.
    if(USE_NCCL)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES __caffe2_nccl)
    endif()
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D)

endif()

if(USE_NCCL)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/python_nccl.cpp)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NCCL)
endif()


add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})
torch_compile_options(torch_python)  # see cmake/public/utils.cmake
torch_optimize_layout_if_enabled(
  torch_python
  "libtorch_python-${Python_SOABI}.yaml")
if(APPLE)
  target_compile_options(torch_python PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>: -fvisibility=default>)
endif()

if(CAFFE2_USE_MKL AND BUILD_LIBTORCHLESS)

  # Use the RPATH of the linked libraries
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  # we need to explicitly link caffe2::mkl in order to have the
  # correct RPATH in torch_python for the split build
  target_link_libraries(torch_python PRIVATE caffe2::mkl)
endif()

# Avoid numpy for the DEPLOY build
if(USE_NUMPY)
  target_link_libraries(torch_python PRIVATE Python::NumPy)
  target_compile_definitions(torch_python PRIVATE USE_NUMPY)
endif()


if(HAVE_SOVERSION)
  set_target_properties(torch_python PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()

# in case of the split build we need to add compile definitions
if(BUILD_LIBTORCHLESS)

  if(USE_NCCL AND USE_C10D_NCCL)
    target_compile_definitions(torch_python PRIVATE USE_C10D_NCCL)
  endif()

  if(USE_DISTRIBUTED)
    target_compile_definitions(torch_python PRIVATE USE_DISTRIBUTED)
  endif()

  if(USE_GLOO AND USE_C10D_GLOO)
    target_compile_definitions(torch_python PRIVATE USE_C10D_GLOO)
  endif()

endif()

  set(EXPERIMENTAL_SINGLE_THREAD_POOL "0" CACHE STRING
  "Experimental option to use a single thread pool for inter- and intra-op parallelism")
  if("${EXPERIMENTAL_SINGLE_THREAD_POOL}")
    target_compile_definitions(torch_python PRIVATE "-DAT_EXPERIMENTAL_SINGLE_THREAD_POOL=1")
  endif()


add_dependencies(torch_python torch_python_stubs)


if(USE_PRECOMPILED_HEADERS)
  target_precompile_headers(torch_python PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:ATen/ATen.h>")
endif()

# Required workaround for generated sources
# See https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
add_dependencies(torch_python generate-torch-sources)
set_source_files_properties(
    ${GENERATED_CXX_PYTHON}
    PROPERTIES GENERATED TRUE
    )

# Disable certain warnings for GCC-9.X
if(CMAKE_COMPILER_IS_GNUCXX)
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/Module.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/autograd/python_variable.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
endif()

# Preserve CUDA_GENCODE flags
if(USE_CUDA)
  torch_cuda_get_nvcc_gencode_flag(_ARCH_FLAGS)
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/cuda/Module.cpp PROPERTIES COMPILE_FLAGS "-DCUDA_ARCH_FLAGS=\"${_ARCH_FLAGS_readable}\"")
endif()

# Preserve HIP arch flags
if(USE_ROCM)
  string(REPLACE ";" " " PYTORCH_ROCM_ARCH_readable "${PYTORCH_ROCM_ARCH}")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/cuda/Module.cpp PROPERTIES COMPILE_FLAGS "-DCUDA_ARCH_FLAGS=\"${PYTORCH_ROCM_ARCH_readable}\"")
endif()

# Preserve XPU arch flags
if(USE_XPU)
  string(REPLACE "," " " _ARCH_FLAGS_readable "${TORCH_XPU_ARCH_LIST}")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/xpu/Module.cpp PROPERTIES COMPILE_FLAGS "-DXPU_ARCH_FLAGS=\"${_ARCH_FLAGS_readable}\"")
endif()

target_compile_definitions(torch_python PRIVATE "-DTHP_BUILD_MAIN_LIB")

target_link_libraries(torch_python PRIVATE ${TORCH_LIB} ${TORCH_PYTHON_LINK_LIBRARIES})

target_compile_definitions(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

target_compile_options(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

target_include_directories(torch_python PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})

if(${CMAKE_BUILD_TYPE} STREQUAL "RelWithAssert")
   # Workaround numerous decref-without-a-gil warnings from JIT
   # see https://github.com/pytorch/pytorch/issues/130073
   target_compile_definitions(torch_python PRIVATE "-DPYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF")
endif()

if(NOT TORCH_PYTHON_LINK_FLAGS STREQUAL "")
    set_target_properties(torch_python PROPERTIES LINK_FLAGS "${TORCH_PYTHON_LINK_FLAGS}")
endif()

install(TARGETS torch_python DESTINATION "${TORCH_INSTALL_LIB_DIR}")

# Build the torch._C Python extension module from the thin stub that
# forwards to torch_python.  This was historically built by setuptools;
# building it here lets both setuptools and scikit-build-core use the
# same code path.
if(BUILD_PYTHON AND NOT BUILD_LIBTORCH_WHL)
  Python_add_library(_C MODULE WITH_SOABI "${TORCH_SRC_DIR}/csrc/stub.c")
  target_link_libraries(_C PRIVATE torch_python)
  set_target_properties(_C PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/torch"
    # libtorch_python.so is installed to lib/ relative to _C, so use
    # $ORIGIN/lib as the install RPATH (the global default is just $ORIGIN).
    INSTALL_RPATH "${_rpath_portable_origin}/lib"
  )
  set_target_properties(_C PROPERTIES
    C_VISIBILITY_PRESET "default"
  )
  # The extension module belongs to the Python package.
  if(SKBUILD)
    install(TARGETS _C
      RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/torch"
      LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/torch"
    )
  endif()

  # ---[ NCCL EP extension (torch._nccl_ep)
  # Optional standalone extension holding the NCCL EP bindings, imported lazily
  # by torch/distributed/_token_switch.py so libtorch_cuda never references
  # ncclEp*. How it links libnccl_ep follows how torch links NCCL:
  #  - USE_SYSTEM_NCCL=OFF: __caffe2_nccl_ep is libnccl_ep.a, statically linked
  #    here -- self-contained (nccl* bind to torch_cuda's embedded NCCL; JIT
  #    headers baked from the submodule build dir; no nccl4py).
  #  - USE_SYSTEM_NCCL=ON: __caffe2_nccl_ep is libnccl_ep.so, NEEDED-linked here
  #    and resolved from the nccl4py wheel at runtime (RPATH to nccl/ep/lib; JIT
  #    headers also from nccl4py, so nothing is baked).
  if(USE_NCCL_EP)
    Python_add_library(_nccl_ep MODULE WITH_SOABI
      "${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/nccl_ep.cu"
      "${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/nccl_ep_pybind.cpp")
    target_compile_definitions(_nccl_ep PRIVATE USE_NCCL_EP USE_C10D_NCCL USE_NCCL)
    # torch_python: pybind + Tensor/ProcessGroup casters; torch_cuda:
    # NCCL communicator registry, getNcclDataType, CUDA stream, embedded NCCL;
    # __caffe2_nccl_ep: libnccl_ep (static .a or shared .so) + nccl_ep headers.
    target_link_libraries(_nccl_ep PRIVATE
      torch_python torch_cuda __caffe2_nccl_ep pybind::pybind11)
    set_target_properties(_nccl_ep PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/torch")
    if(USE_SYSTEM_NCCL)
      # Dynamic: resolve nccl4py's libnccl_ep.so (site-packages/nccl/ep/lib, one
      # dir up from torch/ via $ORIGIN/..) at import.
      set_target_properties(_nccl_ep PROPERTIES
        INSTALL_RPATH "${_rpath_portable_origin}/lib;${_rpath_portable_origin}/../nccl/ep/lib")
    else()
      # Static: only torch's own libs needed; bake the in-tree JIT header dir so
      # the extension self-configures NCCL_EP_HOME / NCCL_HOME at import.
      target_compile_definitions(_nccl_ep PRIVATE
        "NCCL_EP_JIT_HOME=\"${NCCL_EP_JIT_HOME}\"")
      set_target_properties(_nccl_ep PROPERTIES
        INSTALL_RPATH "${_rpath_portable_origin}/lib")
    endif()
    set_target_properties(_nccl_ep PROPERTIES C_VISIBILITY_PRESET "default")
    if(SKBUILD)
      install(TARGETS _nccl_ep
        RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/torch"
        LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/torch")
    endif()
  endif()
endif()

# Generate torch/version.py from the appropriate CMake cache variables.
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(TORCH_VERSION_DEBUG 1)
else()
  set(TORCH_VERSION_DEBUG 0)
endif()

set(CUDA_VERSION "")
if(CUDAToolkit_VERSION_MAJOR)
  set(CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
endif()

add_custom_target(
  gen_torch_version ALL
  "${Python_EXECUTABLE}" -m tools.generate_torch_version
    --is-debug=${TORCH_VERSION_DEBUG}
    --cuda-version=${CUDA_VERSION}
    --hip-version=${HIP_VERSION_CLEAN}
    --rocm-version=${ROCM_VERSION_DEV}
    --xpu-version=${SYCL_COMPILER_VERSION}
  BYPRODUCTS ${TORCH_SRC_DIR}/version.py
  COMMENT "Regenerating version file..."
  WORKING_DIRECTORY ${TORCH_ROOT}
)
add_dependencies(torch_python gen_torch_version)
# version.py is gitignored (generated), so scikit-build-core won't include it
# via Python package collection. Install it explicitly via cmake instead.
if(SKBUILD)
  install(FILES "${TORCH_SRC_DIR}/version.py"
          DESTINATION "${SKBUILD_PLATLIB_DIR}/torch")
endif()

# _rocm_init.py is gitignored (injected by ROCm build scripts to bootstrap
# torch/__init__.py). See https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md
# scikit-build-core won't include it via Python package collection. Install it
# explicitly via cmake instead.
if(SKBUILD AND USE_ROCM AND EXISTS "${TORCH_SRC_DIR}/_rocm_init.py")
  install(FILES "${TORCH_SRC_DIR}/_rocm_init.py"
          DESTINATION "${SKBUILD_PLATLIB_DIR}/torch")
endif()

set(TORCH_PYTHON_COMPILE_OPTIONS ${TORCH_PYTHON_COMPILE_OPTIONS} PARENT_SCOPE)
set(TORCH_PYTHON_LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS} PARENT_SCOPE)

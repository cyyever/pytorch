# Install non-Python package data into the torch package.  Populates the wheel
# under scikit-build-core; only included when SKBUILD is set.
#
# Destinations are absolute paths under SKBUILD_PLATLIB_DIR rather than
# prefix-relative ones: the install prefix is the C++ root, which is a
# conventional prefix for a standalone install and only happens to coincide
# with the torch package directory under scikit-build-core.

if(NOT DEFINED TORCH_SRC_DIR)
  set(TORCH_SRC_DIR "${PROJECT_SOURCE_DIR}/torch")
endif()

set(_torch_pkg "${SKBUILD_PLATLIB_DIR}/torch")

# --- torch package data ---

if(USE_ROCM AND PYTORCH_ROCM_USE_SDK_WHEELS)
  configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/rocm_init.py.in"
    "${PROJECT_BINARY_DIR}/_rocm_init.py"
    @ONLY
  )
  install(FILES "${PROJECT_BINARY_DIR}/_rocm_init.py"
    DESTINATION "${_torch_pkg}"
  )
endif()

# Type stubs
install(DIRECTORY "${TORCH_SRC_DIR}/"
  DESTINATION "${_torch_pkg}"
  FILES_MATCHING
  PATTERN "*.pyi"
  PATTERN "py.typed"
)

# Benchmark utilities — matches setup.py package_data patterns:
#   utils/benchmark/utils/*.cpp
install(DIRECTORY "${TORCH_SRC_DIR}/utils/benchmark/utils/"
  DESTINATION "${_torch_pkg}/utils/benchmark/utils"
  FILES_MATCHING PATTERN "*.cpp"
)

# Model dump utilities
install(FILES
  "${TORCH_SRC_DIR}/utils/model_dump/skeleton.html"
  "${TORCH_SRC_DIR}/utils/model_dump/code.js"
  DESTINATION "${_torch_pkg}/utils/model_dump"
  OPTIONAL
)
install(DIRECTORY "${TORCH_SRC_DIR}/utils/model_dump/"
  DESTINATION "${_torch_pkg}/utils/model_dump"
  FILES_MATCHING PATTERN "*.mjs"
)

# Inductor data files
install(FILES "${TORCH_SRC_DIR}/_inductor/script.ld"
  DESTINATION "${_torch_pkg}/_inductor"
  OPTIONAL
)
install(DIRECTORY "${TORCH_SRC_DIR}/_inductor/codegen/"
  DESTINATION "${_torch_pkg}/_inductor/codegen"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cpp"
)
install(DIRECTORY "${TORCH_SRC_DIR}/_inductor/kernel/flex/templates/"
  DESTINATION "${_torch_pkg}/_inductor/kernel/flex/templates"
  FILES_MATCHING PATTERN "*.jinja"
)
install(DIRECTORY "${TORCH_SRC_DIR}/_inductor/kernel/templates/"
  DESTINATION "${_torch_pkg}/_inductor/kernel/templates"
  FILES_MATCHING PATTERN "*.jinja"
)

# Export serde data
install(DIRECTORY "${TORCH_SRC_DIR}/_export/serde/"
  DESTINATION "${_torch_pkg}/_export/serde"
  FILES_MATCHING
  PATTERN "*.yaml"
  PATTERN "*.thrift"
)

# AOTI runtime header
install(FILES "${TORCH_SRC_DIR}/csrc/inductor/aoti_runtime/model.h"
  DESTINATION "${_torch_pkg}/csrc/inductor/aoti_runtime"
  OPTIONAL
)

# Generated testing Python module (gitignored so not picked up by scikit-build-core
# package scanning; install explicitly so it ends up in the wheel).
set(_torch_generated_src_dir "${TORCH_SRC_DIR}")
if(USE_ROCM)
  set(_torch_generated_src_dir "${PYTORCH_HIPIFY_TORCH_DIR}")
endif()
install(FILES "${_torch_generated_src_dir}/testing/_internal/generated/annotated_fn_args.py"
  DESTINATION "${_torch_pkg}/testing/_internal/generated"
)

if(USE_ROCM)
  set(_hipified_python_files
    "_inductor/codegen/cuda/device_op_overrides.py"
    "_inductor/codegen/cpp_wrapper_cpu.py"
    "_inductor/codegen/cpp_wrapper_gpu.py"
    "_inductor/codegen/wrapper.py")
  foreach(_relative_path IN LISTS _hipified_python_files)
    get_filename_component(_destination "${_relative_path}" DIRECTORY)
    install(FILES "${PYTORCH_HIPIFY_TORCH_DIR}/${_relative_path}"
      DESTINATION "${_torch_pkg}/${_destination}")
  endforeach()
endif()

# Dynamo data
install(FILES "${TORCH_SRC_DIR}/_dynamo/graph_break_registry.json"
  DESTINATION "${_torch_pkg}/_dynamo"
  OPTIONAL
)

unset(_torch_pkg)

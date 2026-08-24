# Post-build steps previously handled by setup.py's build_ext.run().
# These run as CMake install(SCRIPT) or install(CODE) commands.

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()
if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include)
endif()

# Normalize paths to forward slashes so they survive embedding inside
# install(CODE "...") strings.
file(TO_CMAKE_PATH "${Python_EXECUTABLE}" _python_exe)
file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" _project_src)
file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}" _cmake_bindir)

# --- Header wrapping with TORCH_STABLE_ONLY guards ---
# Wrap installed headers so they error when included with TORCH_STABLE_ONLY
# or TORCH_TARGET_VERSION defined. This is done at install time via a script.
install(CODE "
  execute_process(
    COMMAND \"${_python_exe}\"
      \"${_project_src}/tools/wrap_headers.py\"
      \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_INCLUDE_DIR}\"
    COMMAND_ERROR_IS_FATAL ANY
  )
")

# --- Compile commands merging ---
# Merge compile_commands.json from build subdirectories.
add_custom_target(merge_compile_commands ALL
  COMMAND "${_python_exe}"
    "${_project_src}/tools/merge_compile_commands.py"
    "${_cmake_bindir}" "${_project_src}"
  COMMENT "Merging compile_commands.json..."
  VERBATIM
)

# --- macOS OpenMP embedding ---
# Copy libomp.dylib / libiomp5.dylib into the wheel and fix rpaths so the
# wheel is self-contained (replicates setup.py's _embed_libomp).
# Gated on USE_OPENMP as well as OpenMP_FOUND so that a user-forced
# USE_OPENMP=OFF doesn't ship an orphan libomp.dylib in the wheel.
if(APPLE AND BUILD_PYTHON AND USE_OPENMP AND OpenMP_FOUND)
  # FindOpenMP reports one variable per entry of OpenMP_CXX_LIB_NAMES; the
  # runtime is whichever of them is an omp library (libomp or libiomp5),
  # the rest being pthread and friends.
  set(_omp_runtime "")
  foreach(_name IN LISTS OpenMP_CXX_LIB_NAMES)
    if(_name MATCHES "omp" AND EXISTS "${OpenMP_${_name}_LIBRARY}")
      set(_omp_runtime "${OpenMP_${_name}_LIBRARY}")
      break()
    endif()
  endforeach()
  if(_omp_runtime)
    install(FILES "${_omp_runtime}"
            DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    # Install omp.h so Inductor's C++ backend can find it at runtime. FindOpenMP
    # exports no include directory, so look next to the runtime we just found
    # and fall back to a search that knows where Homebrew keeps it.
    get_filename_component(_omp_lib_dir "${_omp_runtime}" DIRECTORY)
    get_filename_component(_omp_prefix "${_omp_lib_dir}" DIRECTORY)
    find_path(_omp_h_dir omp.h
      HINTS "${_omp_prefix}/include" "$ENV{OMP_PREFIX}/include"
            /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include
      NO_DEFAULT_PATH)
    if(_omp_h_dir)
      install(FILES "${_omp_h_dir}/omp.h"
              DESTINATION "${TORCH_INSTALL_INCLUDE_DIR}")
    endif()
    # Fix libtorch_cpu's load command and rpaths so the bundled libomp is
    # the only one resolved at runtime. See tools/embed_libomp_macos.py
    # for the two-case logic (homebrew abs-path vs. conda @rpath build).
    install(CODE "
      execute_process(
        COMMAND \"${_python_exe}\"
          \"${_project_src}/tools/embed_libomp_macos.py\"
          --libomp-path \"${_omp_runtime}\"
          --lib-dir \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}\"
        COMMAND_ERROR_IS_FATAL ANY
      )
    ")
  endif()
endif()

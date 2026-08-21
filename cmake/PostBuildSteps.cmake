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
  # OpenMP_libomp_LIBRARY is set by our FindOpenMP module to the full path
  # of the OpenMP shared library (e.g. /path/to/libomp.dylib).
  if(OpenMP_libomp_LIBRARY AND EXISTS "${OpenMP_libomp_LIBRARY}")
    install(FILES "${OpenMP_libomp_LIBRARY}"
            DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    # Install omp.h so Inductor's C++ backend can find it at runtime.
    # FindOpenMP doesn't export an include-dir variable; OpenMP_C_FLAGS
    # carries -I<path> on macOS, so parse it for the header location.
    if(OpenMP_C_FLAGS)
      separate_arguments(_omp_c_flags UNIX_COMMAND "${OpenMP_C_FLAGS}")
      foreach(_flag IN LISTS _omp_c_flags)
        if(_flag MATCHES "^-I(.+)$")
          set(_omp_h_dir "${CMAKE_MATCH_1}")
          if(EXISTS "${_omp_h_dir}/omp.h")
            install(FILES "${_omp_h_dir}/omp.h"
                    DESTINATION "${TORCH_INSTALL_INCLUDE_DIR}")
            break()
          endif()
        endif()
      endforeach()
    endif()
    # Fix libtorch_cpu's load command and rpaths so the bundled libomp is
    # the only one resolved at runtime. See tools/embed_libomp_macos.py
    # for the two-case logic (homebrew abs-path vs. conda @rpath build).
    install(CODE "
      execute_process(
        COMMAND \"${_python_exe}\"
          \"${_project_src}/tools/embed_libomp_macos.py\"
          --libomp-path \"${OpenMP_libomp_LIBRARY}\"
          --lib-dir \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}\"
        COMMAND_ERROR_IS_FATAL ANY
      )
    ")
  endif()
endif()

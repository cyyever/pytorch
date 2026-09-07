if(TARGET roc::rocblas AND TARGET roc::hipblaslt AND TARGET MIOpen)
  return()
elseif(TARGET roc::rocblas OR TARGET roc::hipblaslt OR TARGET MIOpen)
  message(FATAL_ERROR
    "Bundled ROCm library targets are only partially defined")
endif()

include(ExternalProject)
include(FetchContent)

set(BUNDLED_ROCM_LIBS_REVISION "271344a472037d303d5220eea2b14fd2ac01c1e5")

set(_bundled_rocm_root "${PROJECT_BINARY_DIR}/bundled_rocm")
set(_bundled_rocm_install "${_bundled_rocm_root}/install")
file(MAKE_DIRECTORY "${_bundled_rocm_install}/include" "${_bundled_rocm_install}/lib")

function(_pytorch_write_bundled_rocblas_version_script output_file)
  set(_rocsolver_library "${ROCM_PATH}/lib/librocsolver.so")
  if(NOT EXISTS "${_rocsolver_library}")
    message(FATAL_ERROR
      "Bundled rocBLAS export generation requires ${_rocsolver_library}")
  endif()

  execute_process(
    COMMAND "${CMAKE_READELF}" --dyn-syms --wide "${_rocsolver_library}"
    RESULT_VARIABLE _readelf_result
    OUTPUT_VARIABLE _rocsolver_symbols
    ERROR_VARIABLE _readelf_error)
  if(NOT _readelf_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to inspect rocSOLVER symbols: ${_readelf_error}")
  endif()

  string(REPLACE "\n" ";" _rocsolver_symbol_lines "${_rocsolver_symbols}")
  set(_required_rocblas_internal_symbols)
  foreach(_line IN LISTS _rocsolver_symbol_lines)
    if(_line MATCHES "[ \t]UND[ \t]" AND
        _line MATCHES "rocblas_internal_")
      string(REGEX REPLACE ".*[ \t]([^ \t]+)$" "\\1" _symbol "${_line}")
      list(APPEND _required_rocblas_internal_symbols "${_symbol}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _required_rocblas_internal_symbols)
  list(SORT _required_rocblas_internal_symbols)
  if(NOT _required_rocblas_internal_symbols)
    message(FATAL_ERROR
      "rocSOLVER does not expose its required rocBLAS internal symbols")
  endif()

  set(_version_script "{\n  global:\n")
  foreach(_symbol IN LISTS _required_rocblas_internal_symbols)
    string(APPEND _version_script "    ${_symbol};\n")
  endforeach()
  string(APPEND _version_script [=[
  local:
    *rocblas_internal_*;
    extern "C++" {
      "std::*";
      "__gnu_cxx::*";
      "Tensile::*";
    };
};
]=])
  file(WRITE "${output_file}" "${_version_script}")
endfunction()

set(_bundled_rocblas_version_script
  "${_bundled_rocm_root}/rocblas-version-script.map")
_pytorch_write_bundled_rocblas_version_script(
  "${_bundled_rocblas_version_script}")

set(_bundled_hipblaslt_version_script
  "${_bundled_rocm_root}/hipblaslt-version-script.map")
file(WRITE "${_bundled_hipblaslt_version_script}" [=[
{
  local:
    *hipblaslt_internal_*;
    extern "C++" {
      "std::*";
      "__gnu_cxx::*";
      "TensileLite::*";
    };
};
]=])

# The nonexistent SOURCE_SUBDIR makes FetchContent populate sources without
# adding them. ExternalProject keeps each component's cache isolated from PyTorch.
FetchContent_Declare(
  pytorch_rocm_libraries_source
  URL "https://codeload.github.com/ROCm/rocm-libraries/tar.gz/${BUNDLED_ROCM_LIBS_REVISION}"
  URL_HASH SHA256=ef9876b323ebc193451d053ca9356347f92ee0ed808ec8cbf789afc0bc8527ee
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SOURCE_SUBDIR "__pytorch_fetch_only")
FetchContent_MakeAvailable(pytorch_rocm_libraries_source)

set(_bundled_rocm_libraries_source
  "${pytorch_rocm_libraries_source_SOURCE_DIR}")
set(_bundled_hipblaslt_source
  "${_bundled_rocm_libraries_source}/projects/hipblaslt")
set(_bundled_rocblas_source
  "${_bundled_rocm_libraries_source}/projects/rocblas")
set(_bundled_miopen_source
  "${_bundled_rocm_libraries_source}/projects/miopen")

string(REPLACE ";" "|" _bundled_rocm_archs "${PYTORCH_ROCM_ARCH}")
set(_bundled_rocm_prefix_path "${_bundled_rocm_install}|${ROCM_PATH}")
set(_bundled_rocm_common_cmake_args
  "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_INSTALL_PREFIX:PATH=${_bundled_rocm_install}"
  "-DCMAKE_INSTALL_LIBDIR:STRING=lib"
  "-DCMAKE_INSTALL_RPATH:STRING=$ORIGIN"
  "-DCMAKE_PREFIX_PATH:STRING=${_bundled_rocm_prefix_path}"
  "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
  "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_HIP_COMPILER}"
  "-DCMAKE_HIP_COMPILER:FILEPATH=${CMAKE_HIP_COMPILER}"
  "-DCMAKE_HIP_ARCHITECTURES:STRING=${_bundled_rocm_archs}"
  "-DGPU_TARGETS:STRING=${_bundled_rocm_archs}"
  "-DPython_EXECUTABLE:FILEPATH=${Python_EXECUTABLE}"
  "-DPython3_EXECUTABLE:FILEPATH=${Python_EXECUTABLE}"
  "-DROCM_PATH:PATH=${ROCM_PATH}"
  "-DBUILD_SHARED_LIBS:BOOL=ON"
  "-DBUILD_TESTING:BOOL=OFF")
if(CMAKE_TOOLCHAIN_FILE)
  list(APPEND _bundled_rocm_common_cmake_args
    "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}")
endif()

set(_bundled_rocm_build_environment
  "ROCM_PATH=set:${ROCM_PATH}"
  "PATH=path_list_prepend:${ROCM_PATH}/lib/llvm/bin"
  "PATH=path_list_prepend:${ROCM_PATH}/bin")

set(_bundled_rocm_build_command "${CMAKE_COMMAND}" --build <BINARY_DIR>)
set(_bundled_rocm_install_command "${CMAKE_COMMAND}" --install <BINARY_DIR>)
if(CMAKE_CONFIGURATION_TYPES)
  list(APPEND _bundled_rocm_build_command --config "$<CONFIG>")
  list(APPEND _bundled_rocm_install_command --config "$<CONFIG>")
elseif(CMAKE_BUILD_TYPE)
  list(APPEND _bundled_rocm_build_command --config "${CMAKE_BUILD_TYPE}")
  list(APPEND _bundled_rocm_install_command --config "${CMAKE_BUILD_TYPE}")
endif()
if(DEFINED ENV{MAX_JOBS} AND NOT "$ENV{MAX_JOBS}" STREQUAL "")
  list(APPEND _bundled_rocm_build_command --parallel "$ENV{MAX_JOBS}")
endif()

ExternalProject_Add(pytorch_bundled_hipblaslt
  PREFIX "${_bundled_rocm_root}/ep/hipblaslt"
  SOURCE_DIR "${_bundled_hipblaslt_source}"
  BINARY_DIR "${_bundled_rocm_root}/build/hipblaslt"
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  LIST_SEPARATOR "|"
  CMAKE_ARGS
    ${_bundled_rocm_common_cmake_args}
    "-DROCM_LIBRARIES_ROOT:PATH=${_bundled_rocm_libraries_source}"
    "-DHIPBLASLT_ENABLE_CLIENT:BOOL=OFF"
    "-DHIPBLASLT_ENABLE_FETCH:BOOL=OFF"
    "-DHIPBLASLT_ENABLE_ROCROLLER:BOOL=OFF"
    "-DHIPBLASLT_ENABLE_THEROCK:BOOL=ON"
    "-DCMAKE_DISABLE_FIND_PACKAGE_origami:BOOL=ON"
    "-DCMAKE_CXX_FLAGS:STRING=-ffunction-sections -fdata-sections"
    "-DCMAKE_HIP_FLAGS:STRING=-ffunction-sections -fdata-sections"
    "-DCMAKE_SHARED_LINKER_FLAGS:STRING=-Wl,--gc-sections -Wl,--version-script=${_bundled_hipblaslt_version_script}"
    "-DTENSILELITE_BUILD_TESTING:BOOL=OFF"
    "-DTENSILELITE_ENABLE_CLIENT:BOOL=OFF"
  BUILD_COMMAND ${_bundled_rocm_build_command}
  BUILD_ENVIRONMENT_MODIFICATION ${_bundled_rocm_build_environment}
  INSTALL_COMMAND ${_bundled_rocm_install_command}
  BUILD_BYPRODUCTS "${_bundled_rocm_install}/lib/libhipblaslt.so"
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE
  USES_TERMINAL_INSTALL TRUE)

ExternalProject_Add(pytorch_bundled_rocblas
  PREFIX "${_bundled_rocm_root}/ep/rocblas"
  SOURCE_DIR "${_bundled_rocblas_source}"
  BINARY_DIR "${_bundled_rocm_root}/build/rocblas"
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  LIST_SEPARATOR "|"
  CMAKE_ARGS
    ${_bundled_rocm_common_cmake_args}
    "-DROCM_LIBRARIES_ROOT:PATH=${_bundled_rocm_libraries_source}"
    "-DBUILD_CLIENTS:BOOL=OFF"
    "-DBUILD_CLIENTS_TESTS:BOOL=OFF"
    "-DBUILD_CLIENTS_BENCHMARKS:BOOL=OFF"
    "-DBUILD_CLIENTS_SAMPLES:BOOL=OFF"
    "-DBUILD_DOCS:BOOL=OFF"
    "-DBUILD_WITH_HIPBLASLT:BOOL=ON"
    "-DBUILD_WITH_TENSILE:BOOL=ON"
    "-DCMAKE_CXX_FLAGS:STRING=-ffunction-sections -fdata-sections"
    "-DCMAKE_HIP_FLAGS:STRING=-ffunction-sections -fdata-sections"
    "-DCMAKE_SHARED_LINKER_FLAGS:STRING=-Wl,--gc-sections -Wl,--version-script=${_bundled_rocblas_version_script}"
    "-DHIPBLASLT_VERSION:STRING=1.4.1"
    "-DTENSILE_VERSION:STRING="
    "-DTensile_TEST_LOCAL_PATH:PATH=${_bundled_rocm_libraries_source}/shared/tensile"
    "-Dhipblaslt_path:PATH=${_bundled_rocm_install}"
  BUILD_COMMAND ${_bundled_rocm_build_command}
  BUILD_ENVIRONMENT_MODIFICATION ${_bundled_rocm_build_environment}
  INSTALL_COMMAND ${_bundled_rocm_install_command}
  BUILD_BYPRODUCTS "${_bundled_rocm_install}/lib/librocblas.so"
  DEPENDS pytorch_bundled_hipblaslt
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE
  USES_TERMINAL_INSTALL TRUE)

ExternalProject_Add(pytorch_bundled_miopen
  PREFIX "${_bundled_rocm_root}/ep/miopen"
  SOURCE_DIR "${_bundled_miopen_source}"
  BINARY_DIR "${_bundled_rocm_root}/build/miopen"
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  LIST_SEPARATOR "|"
  CMAKE_ARGS
    ${_bundled_rocm_common_cmake_args}
    "-DROCM_LIBRARIES_ROOT:PATH=${_bundled_rocm_libraries_source}"
    "-DBUILD_DEV:BOOL=OFF"
    "-DBUILD_DOCS:BOOL=OFF"
    "-DBUILD_TESTING:BOOL=OFF"
    "-DMIOPEN_BACKEND:STRING=HIP"
    "-DMIOPEN_BUILD_TESTS:BOOL=OFF"
    "-DMIOPEN_BUILD_DRIVER:BOOL=OFF"
    "-DMIOPEN_BUILD_BENCHMARKS:BOOL=OFF"
    "-DMIOPEN_BUILD_DOCS:BOOL=OFF"
    "-DMIOPEN_BUILD_PYTHON:BOOL=OFF"
    "-DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK:BOOL=OFF"
    "-DMIOPEN_ENABLE_AI_KERNEL_TUNING:BOOL=OFF"
    "-DMIOPEN_INSTALL_GPU_DATABASES:STRING=${_bundled_rocm_archs}"
    "-DMIOPEN_STANDALONE_BUILD:BOOL=OFF"
    "-DMIOPEN_USE_COMPOSABLEKERNEL:BOOL=OFF"
    "-DMIOPEN_USE_HIPCONV:BOOL=OFF"
    "-Drocblas_DIR:PATH=${_bundled_rocm_install}/lib/cmake/rocblas"
    "-Dhipblaslt_DIR:PATH=${_bundled_rocm_install}/lib/cmake/hipblaslt"
  BUILD_COMMAND ${_bundled_rocm_build_command}
  BUILD_ENVIRONMENT_MODIFICATION ${_bundled_rocm_build_environment}
  INSTALL_COMMAND ${_bundled_rocm_install_command}
  BUILD_BYPRODUCTS "${_bundled_rocm_install}/lib/libMIOpen.so"
  DEPENDS pytorch_bundled_rocblas
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE
  USES_TERMINAL_INSTALL TRUE)

set(PYTORCH_ROCM_BUNDLED_PREFIX "${_bundled_rocm_install}")

add_library(roc::rocblas SHARED IMPORTED GLOBAL)
set_target_properties(roc::rocblas PROPERTIES
  IMPORTED_LOCATION "${PYTORCH_ROCM_BUNDLED_PREFIX}/lib/librocblas.so"
  INTERFACE_INCLUDE_DIRECTORIES "${PYTORCH_ROCM_BUNDLED_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES hip::host)

add_library(roc::hipblaslt SHARED IMPORTED GLOBAL)
set_target_properties(roc::hipblaslt PROPERTIES
  IMPORTED_LOCATION "${PYTORCH_ROCM_BUNDLED_PREFIX}/lib/libhipblaslt.so"
  INTERFACE_COMPILE_FEATURES cxx_std_17
  INTERFACE_INCLUDE_DIRECTORIES "${PYTORCH_ROCM_BUNDLED_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES roc::hipblas-common)

add_library(MIOpen SHARED IMPORTED GLOBAL)
set_target_properties(MIOpen PROPERTIES
  IMPORTED_LOCATION "${PYTORCH_ROCM_BUNDLED_PREFIX}/lib/libMIOpen.so"
  INTERFACE_INCLUDE_DIRECTORIES "${PYTORCH_ROCM_BUNDLED_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES hip::host)

add_dependencies(roc::rocblas pytorch_bundled_rocblas)
add_dependencies(roc::hipblaslt pytorch_bundled_hipblaslt)
add_dependencies(MIOpen pytorch_bundled_miopen)

set(rocblas_FOUND TRUE)
set(rocblas_VERSION "5.6.0")
set(rocblas_INCLUDE_DIR "${PYTORCH_ROCM_BUNDLED_PREFIX}/include")
set(rocblas_INCLUDE_DIRS "${rocblas_INCLUDE_DIR}")
set(rocblas_LIBRARY roc::rocblas)
set(rocblas_LIBRARIES roc::rocblas)
set(ROCBLAS_FOUND TRUE)
set(ROCBLAS_VERSION "${rocblas_VERSION}")
set(ROCBLAS_INCLUDE_DIR "${rocblas_INCLUDE_DIR}")
set(ROCBLAS_INCLUDE_DIRS "${rocblas_INCLUDE_DIR}")
set(ROCBLAS_LIBRARY roc::rocblas)
set(ROCBLAS_LIBRARIES roc::rocblas)

set(hipblaslt_FOUND TRUE)
set(hipblaslt_VERSION "1.4.1")
set(hipblaslt_INCLUDE_DIR "${PYTORCH_ROCM_BUNDLED_PREFIX}/include")
set(hipblaslt_INCLUDE_DIRS "${hipblaslt_INCLUDE_DIR}")
set(hipblaslt_LIBRARY roc::hipblaslt)
set(hipblaslt_LIBRARIES roc::hipblaslt)
set(HIPBLASLT_FOUND TRUE)
set(HIPBLASLT_VERSION "${hipblaslt_VERSION}")
set(HIPBLASLT_INCLUDE_DIR "${hipblaslt_INCLUDE_DIR}")
set(HIPBLASLT_INCLUDE_DIRS "${hipblaslt_INCLUDE_DIR}")
set(HIPBLASLT_LIBRARY roc::hipblaslt)
set(HIPBLASLT_LIBRARIES roc::hipblaslt)

set(miopen_FOUND TRUE)
set(miopen_VERSION "3.6.0")
set(miopen_INCLUDE_DIR "${PYTORCH_ROCM_BUNDLED_PREFIX}/include")
set(miopen_INCLUDE_DIRS "${miopen_INCLUDE_DIR}")
set(miopen_LIBRARY MIOpen)
set(miopen_LIBRARIES MIOpen)
set(MIOpen_FOUND TRUE)
set(MIOpen_VERSION "${miopen_VERSION}")
set(MIOpen_INCLUDE_DIR "${miopen_INCLUDE_DIR}")
set(MIOpen_INCLUDE_DIRS "${miopen_INCLUDE_DIR}")
set(MIOpen_LIBRARY MIOpen)
set(MIOpen_LIBRARIES MIOpen)
set(MIOPEN_FOUND TRUE)
set(MIOPEN_VERSION "${miopen_VERSION}")
set(MIOPEN_INCLUDE_DIR "${miopen_INCLUDE_DIR}")
set(MIOPEN_INCLUDE_DIRS "${miopen_INCLUDE_DIR}")
set(MIOPEN_LIBRARY MIOpen)
set(MIOPEN_LIBRARIES MIOpen)

install(DIRECTORY "${_bundled_rocm_install}/include/"
  DESTINATION include)
install(DIRECTORY "${_bundled_rocm_install}/lib/"
  DESTINATION lib
  PATTERN "libMIOpen.so*" EXCLUDE
  PATTERN "libhipblaslt.so*" EXCLUDE
  PATTERN "librocblas.so*" EXCLUDE)

torch_install_shared_library(
  "${_bundled_rocm_install}/lib/libMIOpen.so" lib)
torch_install_shared_library(
  "${_bundled_rocm_install}/lib/libhipblaslt.so" lib)
torch_install_shared_library(
  "${_bundled_rocm_install}/lib/librocblas.so" lib)

install(DIRECTORY "${_bundled_rocm_install}/share/hipblaslt"
  DESTINATION share
  OPTIONAL)
install(DIRECTORY "${_bundled_rocm_install}/share/miopen"
  DESTINATION share
  OPTIONAL)

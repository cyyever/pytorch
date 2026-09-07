if(NOT DEFINED MIOPEN_SOURCE_DIR)
  message(FATAL_ERROR "MIOPEN_SOURCE_DIR is required")
endif()

set(_miopen_cmake "${MIOPEN_SOURCE_DIR}/src/CMakeLists.txt")
file(READ "${_miopen_cmake}" _miopen_contents)

set(_filter_marker "# PyTorch bundled target kernel filter v2")
if(_miopen_contents MATCHES "${_filter_marker}")
  return()
endif()
set(_old_filter_marker "# PyTorch bundled target kernel filter")
if(_miopen_contents MATCHES "${_old_filter_marker}")
  set(_has_old_filter TRUE)
endif()

set(_kernel_globs [=[
    file(GLOB_RECURSE COMPOSABLE_KERNEL_DYNAMIC_ASM_SOURCE "kernels/dynamic_igemm/*.s")
    file(GLOB_RECURSE COMPOSABLE_KERNEL_DYNAMIC_ASM_INCLUDE "kernels/dynamic_igemm/*.inc")
    file(GLOB_RECURSE COMPOSABLE_KERNEL_DYNAMIC_CPP_SOURCE "kernels/dynamic_igemm/*.cpp")
]=])
set(_filtered_kernel_globs [=[
    file(GLOB_RECURSE COMPOSABLE_KERNEL_DYNAMIC_ASM_SOURCE "kernels/dynamic_igemm/*.s")
    file(GLOB_RECURSE COMPOSABLE_KERNEL_DYNAMIC_ASM_INCLUDE "kernels/dynamic_igemm/*.inc")
    file(GLOB_RECURSE COMPOSABLE_KERNEL_DYNAMIC_CPP_SOURCE "kernels/dynamic_igemm/*.cpp")

    # PyTorch bundled target kernel filter v2
    if(MIOPEN_EMBED_BUILD_TARGET_KERNELS_ONLY AND GPU_TARGETS STREQUAL "gfx1201")
        foreach(_kernel_list
                COMPOSABLE_KERNEL_DYNAMIC_ASM_SOURCE
                COMPOSABLE_KERNEL_DYNAMIC_ASM_INCLUDE
                COMPOSABLE_KERNEL_DYNAMIC_CPP_SOURCE)
            list(FILTER ${_kernel_list} EXCLUDE
                 REGEX "/[^/]*_gfx(908|90a|940|950|1030)/")
        endforeach()
    endif()
]=])

if(NOT _has_old_filter)
  string(FIND "${_miopen_contents}" "${_kernel_globs}" _kernel_globs_offset)
  if(_kernel_globs_offset EQUAL -1)
    message(FATAL_ERROR
      "Could not locate the MIOpen dynamic IGEMM kernel source lists")
  endif()

  string(REPLACE "${_kernel_globs}" "${_filtered_kernel_globs}"
    _miopen_contents "${_miopen_contents}")
endif()

set(_kernel_list_end [=[
        kernels/UniversalTranspose.cpp)
]=])
set(_filtered_kernel_list_end [=[
        kernels/UniversalTranspose.cpp)

    if(MIOPEN_EMBED_BUILD_TARGET_KERNELS_ONLY AND GPU_TARGETS STREQUAL "gfx1201")
        list(FILTER MIOPEN_KERNEL_INCLUDES EXCLUDE
             REGEX "/(Conv_Winograd_v(13|14|16|21|30)|conv_3x3_wheel_alpha_)")
        list(FILTER MIOPEN_KERNEL_INCLUDES EXCLUDE
             REGEX "/winograd/Conv_Winograd_(Fury_v2_|Rage_.*gfx(94|95))")
        list(FILTER MIOPEN_KERNELS EXCLUDE
             REGEX "/(Conv_Winograd_v(13|14|16|21|30)|conv_3x3_wheel_alpha_)")
        list(FILTER MIOPEN_KERNELS EXCLUDE
             REGEX "/winograd/Conv_Winograd_Fury_v2_")
    endif()
]=])

string(FIND "${_miopen_contents}" "${_kernel_list_end}" _kernel_list_end_offset)
if(_kernel_list_end_offset EQUAL -1)
  message(FATAL_ERROR
    "Could not locate the end of the MIOpen embedded kernel lists")
endif()

string(REPLACE "${_kernel_list_end}" "${_filtered_kernel_list_end}"
  _miopen_contents "${_miopen_contents}")
if(_has_old_filter)
  string(REPLACE "${_old_filter_marker}" "${_filter_marker}"
    _miopen_contents "${_miopen_contents}")
endif()
file(WRITE "${_miopen_cmake}" "${_miopen_contents}")

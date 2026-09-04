find_package(MKL QUIET)

if(TARGET caffe2::mkl)
  return()
endif()

add_library(caffe2::mkl INTERFACE IMPORTED)
target_include_directories(caffe2::mkl INTERFACE ${MKL_INCLUDE_DIR})

# MKL resolves symbols between its own sub-libraries, so the linker must keep
# every one of them in DT_NEEDED even though torch references only some. Without
# this, libmkl_intel_lp64.so fails at run time with undefined symbol
# mkl_blas_dsyrk.
# https://software.intel.com/en-us/articles/symbol-lookup-error-when-linking-intel-mkl-with-gcc
# push-state/pop-state confines this to MKL's own libraries and restores
# whatever state the surrounding link line was in.
if(UNIX AND NOT USE_STATIC_MKL AND NOT CMAKE_CXX_LINK_LIBRARY_USING_MKL_NEEDED_SUPPORTED)
  set(CMAKE_CXX_LINK_LIBRARY_USING_MKL_NEEDED
      "LINKER:--push-state,--no-as-needed" "<LINK_ITEM>" "LINKER:--pop-state")
  set(CMAKE_CXX_LINK_LIBRARY_USING_MKL_NEEDED_SUPPORTED TRUE)
endif()
if(CMAKE_CXX_LINK_LIBRARY_USING_MKL_NEEDED_SUPPORTED)
  target_link_libraries(caffe2::mkl INTERFACE
      "$<LINK_LIBRARY:MKL_NEEDED,${MKL_LIBRARIES}>")
else()
  target_link_libraries(caffe2::mkl INTERFACE ${MKL_LIBRARIES})
endif()
foreach(MKL_LIB IN LISTS MKL_LIBRARIES)
  if(EXISTS "${MKL_LIB}")
    get_filename_component(MKL_LINK_DIR "${MKL_LIB}" DIRECTORY)
    if(IS_DIRECTORY "${MKL_LINK_DIR}")
      target_link_directories(caffe2::mkl INTERFACE "${MKL_LINK_DIR}")
    endif()
  endif()
endforeach()

# TODO: This is a hack, it will not pick up architecture dependent
# MKL libraries correctly; see https://github.com/pytorch/pytorch/issues/73008
set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_LINK_DIRECTORIES
  ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/intel64_win ${MKL_ROOT}/lib/win-x64)

if(UNIX)
  if(USE_STATIC_MKL)
    foreach(MKL_LIB_PATH IN LISTS MKL_LIBRARIES)
      if(NOT EXISTS "${MKL_LIB_PATH}")
        continue()
      endif()

      get_filename_component(MKL_LIB_NAME "${MKL_LIB_PATH}" NAME)

      # Match archive libraries starting with "libmkl_"
      if(MKL_LIB_NAME MATCHES "^libmkl_" AND MKL_LIB_NAME MATCHES ".a$")
        target_link_options(caffe2::mkl INTERFACE "LINKER:--exclude-libs,${MKL_LIB_NAME}")
      endif()
    endforeach()
  endif()
endif()

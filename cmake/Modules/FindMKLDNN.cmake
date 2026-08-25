# - Build the oneDNN GPU (SYCL) library for the XPU backend.
#
# The CPU half of this module went away with USE_MKLDNN. What remains is the
# oneDNN build that libtorch_xpu.so links against; it clones oneDNN itself and
# is built with DNNL_CPU_RUNTIME=NONE.

include(ExternalProject)

  if(USE_XPU) # Build oneDNN GPU library
    if(WIN32)
      # Windows
      set(DNNL_HOST_COMPILER "DEFAULT")
      set(DNNL_C_COMPILER "icx")
      set(SYCL_CXX_DRIVER "icx")
      set(DNNL_LIB_NAME "dnnl.lib")
    elseif(LINUX)
      # Linux
      # g++ is soft linked to /usr/bin/cxx, oneDNN would not treat it as an absolute path
      set(DNNL_HOST_COMPILER "g++")
      if("${XPU_SYCL_COMPILER}" MATCHES "icx")
        set(DNNL_C_COMPILER "icx")
        set(SYCL_CXX_DRIVER "icpx")
      elseif("${XPU_SYCL_COMPILER}" MATCHES "dpclang")
        set(DNNL_C_COMPILER "dpclang")
        set(SYCL_CXX_DRIVER "dpclang++")
      else()
        message(FATAL_ERROR "Unsupported SYCL compiler: ${XPU_SYCL_COMPILER}")
      endif()
      set(DNNL_LIB_NAME "libdnnl.a")
    else()
      MESSAGE(FATAL_ERROR "OneDNN for Intel GPU in PyTorch currently supports only Windows and Linux.
                           Detected system '${CMAKE_SYSTEM_NAME}' is not supported.")
    endif()

    set(DNNL_MAKE_COMMAND "cmake" "--build" ".")
    include(ProcessorCount)
    ProcessorCount(proc_cnt)
    if((DEFINED ENV{MAX_JOBS}) AND ("$ENV{MAX_JOBS}" LESS_EQUAL ${proc_cnt}))
      list(APPEND DNNL_MAKE_COMMAND "-j" "$ENV{MAX_JOBS}")
      if(CMAKE_GENERATOR MATCHES "Make|Ninja")
        list(APPEND DNNL_MAKE_COMMAND "--" "-l" "$ENV{MAX_JOBS}")
      endif()
    endif()
    ExternalProject_Add(xpu_mkldnn_proj
      GIT_REPOSITORY https://github.com/uxlfoundation/oneDNN
      GIT_TAG v3.12.3
      PREFIX ${XPU_MKLDNN_DIR_PREFIX}
      BUILD_IN_SOURCE 0
      CMAKE_ARGS  -DCMAKE_C_COMPILER=${DNNL_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${SYCL_CXX_DRIVER}
      -DDNNL_GPU_RUNTIME=SYCL
      -DDNNL_CPU_RUNTIME=NONE
      -DDNNL_BUILD_TESTS=OFF
      -DDNNL_BUILD_EXAMPLES=OFF
      -DONEDNN_BUILD_GRAPH=ON
      -DDNNL_LIBRARY_TYPE=STATIC
      -DDNNL_DPCPP_HOST_COMPILER=${DNNL_HOST_COMPILER} # Use global cxx compiler as host compiler
      -G ${CMAKE_GENERATOR} # Align Generator to Torch
      BUILD_COMMAND ${DNNL_MAKE_COMMAND}
      BUILD_BYPRODUCTS "xpu_mkldnn_proj-prefix/src/xpu_mkldnn_proj-build/src/${DNNL_LIB_NAME}"
      INSTALL_COMMAND ""
    )

    ExternalProject_Get_Property(xpu_mkldnn_proj SOURCE_DIR BINARY_DIR)
    set(XPU_MKLDNN_LIBRARIES ${BINARY_DIR}/src/${DNNL_LIB_NAME})
    set(XPU_MKLDNN_INCLUDE ${SOURCE_DIR}/include ${BINARY_DIR}/include)
    # This target would be further linked to libtorch_xpu.so.
    # The libtorch_xpu.so would contain Conv&GEMM operators that depend on
    # oneDNN primitive implementations inside libdnnl.a.
    add_library(xpu_mkldnn INTERFACE)
    add_dependencies(xpu_mkldnn xpu_mkldnn_proj)
    target_link_libraries(xpu_mkldnn INTERFACE ${XPU_MKLDNN_LIBRARIES})
    target_include_directories(xpu_mkldnn INTERFACE ${XPU_MKLDNN_INCLUDE})
  endif()

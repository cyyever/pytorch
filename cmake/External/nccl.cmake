if(NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  if(USE_SYSTEM_NCCL)
    # NCCL_ROOT, NCCL_LIB_DIR, NCCL_INCLUDE_DIR will be accounted in the following line.
    find_package(NCCL REQUIRED)
    if(NCCL_FOUND)
      add_library(__caffe2_nccl INTERFACE)
      target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
      target_include_directories(__caffe2_nccl SYSTEM INTERFACE ${NCCL_INCLUDE_DIRS})
    endif()
  else()
    apply_third_party_patches(
        "${PROJECT_SOURCE_DIR}/third_party/nccl_patches"
        "${PROJECT_SOURCE_DIR}/third_party/nccl"
        NCCL)

    if(DEFINED ENV{MAX_JOBS})
      set(MAX_JOBS "$ENV{MAX_JOBS}")
    else()
      include(ProcessorCount)
      ProcessorCount(NUM_HARDWARE_THREADS)
      # Assume 2 hardware threads per cpu core
      math(EXPR MAX_JOBS "${NUM_HARDWARE_THREADS} / 2")
      # ProcessorCount might return 0, set to a positive number
      if(MAX_JOBS LESS 2)
        set(MAX_JOBS 2)
      endif()
    endif()

    if((NOT DEFINED TORCH_CUDA_ARCH_LIST) AND (DEFINED ENV{TORCH_CUDA_ARCH_LIST}))
      set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
    endif()
    string(REPLACE " " ";" __NCCL_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
    set(__NCCL_CMAKE_ARCHS "")
    foreach(__arch IN LISTS __NCCL_ARCH_LIST)
      if(__arch MATCHES "^([0-9]+)\\.([0-9]+)")
        set(__cmake_arch "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        list(APPEND __NCCL_CMAKE_ARCHS "${__cmake_arch}-real")
        if(__arch MATCHES "\\+PTX$")
          list(APPEND __NCCL_CMAKE_ARCHS "${__cmake_arch}-virtual")
        endif()
      endif()
    endforeach()
    list(REMOVE_DUPLICATES __NCCL_CMAKE_ARCHS)
    if(NOT __NCCL_CMAKE_ARCHS)
      message(FATAL_ERROR
        "Bundled NCCL requires numeric entries in TORCH_CUDA_ARCH_LIST")
    endif()

    set(__NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")
    set(__NCCL_CUDA_FLAGS
      "--pre-include=${PYTORCH_CUDA_HOST_COMPILER_COMPAT}")
    if(PYTORCH_X86_ARCH_FLAG)
      string(APPEND __NCCL_CUDA_FLAGS
        " -Xcompiler=${PYTORCH_X86_ARCH_FLAG}")
    endif()
    ExternalProject_Add(nccl_external
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/nccl
      BINARY_DIR ${__NCCL_BUILD_DIR}
      CMAKE_GENERATOR "${CMAKE_GENERATOR}"
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}
        "-DCMAKE_C_FLAGS=${PYTORCH_X86_ARCH_FLAG}"
        "-DCMAKE_CXX_FLAGS=${PYTORCH_X86_ARCH_FLAG}"
        "-DCMAKE_CUDA_ARCHITECTURES=${__NCCL_CMAKE_ARCHS}"
        "-DCMAKE_CUDA_FLAGS=${__NCCL_CUDA_FLAGS}"
        -DPython3_EXECUTABLE=${Python_EXECUTABLE}
      BUILD_COMMAND
        ${CMAKE_COMMAND} --build <BINARY_DIR>
        --target nccl_static
        --parallel ${MAX_JOBS}
      BUILD_BYPRODUCTS "${__NCCL_BUILD_DIR}/lib/libnccl_static.a"
      INSTALL_COMMAND ""
      )

    set(__NCCL_LIBRARY_DEP nccl_external)
    set(NCCL_LIBRARIES ${__NCCL_BUILD_DIR}/lib/libnccl_static.a)

    set(NCCL_FOUND TRUE)
    add_library(__caffe2_nccl INTERFACE)
    # The following old-style variables are set so that other libs, such as Gloo,
    # can still use it.
    set(NCCL_INCLUDE_DIRS ${__NCCL_BUILD_DIR}/include)
    add_dependencies(__caffe2_nccl ${__NCCL_LIBRARY_DEP})
    target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
    # SYSTEM: NCCL's own headers use anonymous types in anonymous unions,
    # which -Werror=pedantic rejects. Their diagnostics are not ours to fix.
    target_include_directories(__caffe2_nccl SYSTEM INTERFACE ${NCCL_INCLUDE_DIRS})
    # nccl includes calls to shm_open/shm_close and therefore must depend on librt on Linux
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_link_libraries(__caffe2_nccl INTERFACE rt)
    endif()
  endif()
endif()

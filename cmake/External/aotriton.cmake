if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_EXTERN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)

  set(__AOTRITON_VER "0.13b")
  if(DEFINED ENV{PYTORCH_AOTRITON_COMMIT})
    set(__AOTRITON_CI_COMMIT "$ENV{PYTORCH_AOTRITON_COMMIT}")
  else()
    set(__AOTRITON_CI_COMMIT "6e00ef3e335b45dfb49065259533b59c68995bfe")
  endif()
  set(__AOTRITON_IMAGE "amd-gfx120x")
  set(__AOTRITON_IMAGE_SHA256
      "6a465dbc03148bba8a2d78c4c2a3cb83155eca00f4f7f749e676402d7660968c")
  set(__AOTRITON_BASE_URL "$ENV{PYTORCH_AOTRITON_BASE_URL}")
  if(NOT __AOTRITON_BASE_URL)
    set(__AOTRITON_BASE_URL "https://github.com/ROCm/aotriton/releases/download/")  # @lint-ignore
  endif()
  set(__AOTRITON_Z "gz")
  # Set the default __AOTRITON_LIB path
  set(__AOTRITON_LIB "lib/libaotriton_v2.so")

  function(aotriton_build_from_source noimage project)
    if(noimage)
      SET(RECURSIVE "OFF")
    else()
      SET(RECURSIVE "ON")
    endif()
    message(STATUS "PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}")
    # The HIP half is compiled by amdclang++, not the compiler the baseline was
    # validated against, so it gets its own probe.
    torch_x86_arch_flag_for("${CMAKE_HIP_COMPILER}" _aotriton_hip_arch_flag)
    set(__AOTRITON_HOST_ARCH_ARGS "")
    if(PYTORCH_X86_ARCH_FLAG)
      list(APPEND __AOTRITON_HOST_ARCH_ARGS
        "-DCMAKE_C_FLAGS=${PYTORCH_X86_ARCH_FLAG}"
        "-DCMAKE_CXX_FLAGS=${PYTORCH_X86_ARCH_FLAG}")
    endif()
    if(_aotriton_hip_arch_flag)
      list(APPEND __AOTRITON_HOST_ARCH_ARGS
        "-DCMAKE_HIP_FLAGS=-Xarch_host ${_aotriton_hip_arch_flag}")
    endif()

    ExternalProject_Add(${project}
      GIT_REPOSITORY https://github.com/ROCm/aotriton.git
      GIT_SUBMODULES_RECURSE ${RECURSIVE}
      GIT_TAG ${__AOTRITON_CI_COMMIT}
      PREFIX ${__AOTRITON_EXTERN_PREFIX}
      PATCH_COMMAND
      ${CMAKE_COMMAND}
      -DAOTRITON_SOURCE_DIR:PATH=<SOURCE_DIR>
      -P "${CMAKE_CURRENT_LIST_DIR}/aotriton_gfx1201_database_filter.cmake"
      CMAKE_CACHE_ARGS
      -DAOTRITON_TARGET_ARCH:STRING=${PYTORCH_ROCM_ARCH}
      -DCMAKE_INSTALL_PREFIX:FILEPATH=${__AOTRITON_INSTALL_DIR}
      -DCMAKE_PREFIX_PATH:PATH=${ROCM_PATH}
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      ${__AOTRITON_HOST_ARCH_ARGS}
      -DAOTRITON_GPU_BUILD_TIMEOUT=0
      -DAOTRITON_NO_PYTHON=ON
      -DAOTRITON_NOIMAGE_MODE=${noimage}
      -DHIP_PLATFORM=amd
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/${__AOTRITON_LIB}"
      USES_TERMINAL_DOWNLOAD TRUE
      USES_TERMINAL_CONFIGURE TRUE
      USES_TERMINAL_BUILD TRUE
      USES_TERMINAL_INSTALL TRUE
    )
  endfunction()

  function(aotriton_download_image project)
    string(CONCAT __AOTRITON_FILE
           "aotriton-${__AOTRITON_VER}-images-"
           "${__AOTRITON_IMAGE}.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL
           "${__AOTRITON_BASE_URL}"
           "${__AOTRITON_VER}/${__AOTRITON_FILE}")

    # Set up directories
    set(__AOTRITON_DOWNLOAD_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/aotriton_download-${__AOTRITON_IMAGE})
    set(__AOTRITON_EXTRACT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/aotriton_image-${__AOTRITON_IMAGE})
    set(__AOTRITON_INSTALL_SOURCE_DIR ${__AOTRITON_EXTRACT_DIR})

    ExternalProject_Add(${project}
      URL "${__AOTRITON_URL}"
      URL_HASH SHA256=${__AOTRITON_IMAGE_SHA256}
      DOWNLOAD_DIR ${__AOTRITON_DOWNLOAD_DIR}
      SOURCE_DIR ${__AOTRITON_EXTRACT_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${__AOTRITON_INSTALL_SOURCE_DIR}"
      "${__AOTRITON_INSTALL_DIR}"
      BUILD_BYPRODUCTS
      "${__AOTRITON_INSTALL_DIR}/lib/aotriton.images/${__AOTRITON_IMAGE}/__signature__"
    )
    message(STATUS "Download AOTriton pre-compiled GPU images from ${__AOTRITON_URL}.")
  endfunction()

  # Note it is INSTALL"ED"
  if(DEFINED ENV{AOTRITON_INSTALLED_PREFIX})
    install(DIRECTORY
            $ENV{AOTRITON_INSTALLED_PREFIX}/lib
            $ENV{AOTRITON_INSTALLED_PREFIX}/include
            DESTINATION ${__AOTRITON_INSTALL_DIR})
    set(__AOTRITON_INSTALL_DIR "$ENV{AOTRITON_INSTALLED_PREFIX}")
    message(STATUS "Using Preinstalled AOTriton at ${__AOTRITON_INSTALL_DIR}")
  elseif(DEFINED ENV{AOTRITON_INSTALL_FROM_SOURCE} OR USE_ASAN)
    aotriton_build_from_source(OFF aotriton_external)
    add_dependencies(__caffe2_aotriton aotriton_external)
    message(STATUS "Using AOTriton compiled from source directory ${__AOTRITON_EXTERN_PREFIX}")
  else()
    aotriton_build_from_source(ON aotriton_runtime)
    add_dependencies(__caffe2_aotriton aotriton_runtime)
    aotriton_download_image(aotriton_image_gfx120x)
    add_dependencies(aotriton_runtime aotriton_image_gfx120x)
  endif()
  target_link_libraries(__caffe2_aotriton INTERFACE "${__AOTRITON_INSTALL_DIR}/${__AOTRITON_LIB}")
  target_include_directories(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
  set(AOTRITON_FOUND TRUE)
  # Install libaotriton_v2.so into the cmake install tree so it ends up in
  # site-packages/torch/lib/ when building with scikit-build-core.
  # aotriton's ExternalProject puts the library directly in the source tree
  # (${PROJECT_SOURCE_DIR}/torch/lib/) without a cmake install() rule, so it
  # is absent from the installed wheel and causes link failures in downstream
  # cmake builds (e.g., custom op builds) that link against installed torch.
  torch_install_shared_library(
    "${__AOTRITON_INSTALL_DIR}/${__AOTRITON_LIB}" lib)
  # Install aotriton GPU kernel images (compressed ISA blobs) into the wheel.
  install(DIRECTORY "${__AOTRITON_INSTALL_DIR}/lib/aotriton.images"
    DESTINATION "lib"
    OPTIONAL
  )
endif() # __AOTRITON_INCLUDED

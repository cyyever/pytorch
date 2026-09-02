include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

# ---[ Check if the compiler has AVX/AVX2 support. We only check AVX2.
# CXX_AVX2_FOUND gates the vectorized CPU kernels in cmake/Codegen.cmake.
find_package(AVX) # checks AVX and AVX2

# ---[ Checks if compiler supports -fvisibility=hidden
check_cxx_compiler_flag("-fvisibility=hidden" COMPILER_SUPPORTS_HIDDEN_VISIBILITY)
check_cxx_compiler_flag("-fvisibility-inlines-hidden" COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY)
if(${COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY})
  set(CAFFE2_VISIBILITY_FLAG "-fvisibility-inlines-hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CAFFE2_VISIBILITY_FLAG}")
endif()

# ---[ Checks if linker supports -rdynamic. `-rdynamic` tells linker
# -to add all (including unused) symbols into the dynamic symbol
# -table. We need this to get symbols when generating backtrace at
# -runtime. It only does anything when linking an executable; a shared
# -library already exports whatever its visibility settings allow.
check_cxx_compiler_flag("-rdynamic" COMPILER_SUPPORTS_RDYNAMIC)
  if(${COMPILER_SUPPORTS_RDYNAMIC})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
  endif()

# ---[ Create CAFFE2_BUILD_SHARED_LIBS for macros.h.in usage.
set(CAFFE2_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

if(USE_NATIVE_ARCH)
  check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    add_definitions("-march=native")
  else()
    message(
        WARNING
        "Your compiler does not support -march=native. Turn off this warning "
        "by setting -DUSE_NATIVE_ARCH=OFF.")
  endif()
endif()

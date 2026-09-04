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

# ---[ x86 baseline. Every x86 CPU released from 2020 on satisfies x86-64-v3
# (AVX2, FMA, BMI1/2, F16C), including the AVX-512-less AMD Zen 3 and the
# consumer Intel parts that fuse AVX-512 off, so v3 is the default. Raise it to
# x86-64-v4 for a build that only ever runs on AVX-512 hardware; note that the
# ATen kernels dispatch to an AVX-512 slice at run time either way, so this only
# affects the code outside those slices. -march=native, when asked for, is more
# specific and wins.
# A host that can run AVX-512 itself gets x86-64-v4. Note this makes the default
# depend on the build machine, so a binary built on such a host will not run on
# one without AVX-512; set TORCH_X86_BASELINE explicitly for a portable build.
if(CPU_HOST_HAS_AVX512)
  set(_torch_x86_baseline_default "x86-64-v4")
else()
  set(_torch_x86_baseline_default "x86-64-v3")
endif()
set(TORCH_X86_BASELINE "${_torch_x86_baseline_default}" CACHE STRING "-march baseline for x86 builds")
if(CPU_INTEL AND NOT USE_NATIVE_ARCH)
  check_cxx_compiler_flag("-march=${TORCH_X86_BASELINE}" COMPILER_SUPPORTS_X86_BASELINE)
  if(COMPILER_SUPPORTS_X86_BASELINE)
    string(APPEND CMAKE_C_FLAGS " -march=${TORCH_X86_BASELINE}")
    string(APPEND CMAKE_CXX_FLAGS " -march=${TORCH_X86_BASELINE}")
  else()
    message(WARNING "Compiler does not support -march=${TORCH_X86_BASELINE}; building for generic x86-64.")
  endif()
endif()

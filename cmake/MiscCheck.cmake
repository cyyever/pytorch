include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

# ---[ Check whether the AVX-512 slice of the CPU kernels can be built.
# CXX_AVX512_FOUND gates it in cmake/Codegen.cmake; AVX2 needs no gate, since
# the x86 baseline below already requires it. The probe compiles intrinsics
# rather than only asking whether the flags are accepted, and it uses the flags
# Codegen actually passes -- the FindAVX module this replaces also demanded
# -mavx512bf16, which that slice does not use, so a compiler without the bf16
# extension was reported as having no AVX-512 at all.
cmake_push_check_state()
set(CMAKE_REQUIRED_FLAGS
    "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma -mf16c")
check_cxx_source_compiles("
  #include <immintrin.h>
  int main() {
    __m512i a = _mm512_set1_epi8(0);
    __mmask64 m = _mm512_cmp_epi8_mask(a, a, _MM_CMPINT_EQ);
    return static_cast<int>(m);
  }" CXX_AVX512_FOUND)
cmake_pop_check_state()

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

# ---[ x86 baseline. AMD wheel builders target Zen 4, while other x86 hosts use
# the portable x86-64-v3 baseline. Explicit TORCH_X86_BASELINE values still win.
set(_torch_x86_baseline_default "x86-64-v3")
if(CPU_INTEL AND NOT CMAKE_CROSSCOMPILING)
  cmake_host_system_information(
      RESULT _torch_processor_description QUERY PROCESSOR_DESCRIPTION)
  if(_torch_processor_description MATCHES "(^| )AMD( |$)")
    set(_torch_x86_baseline_default "znver4")
  endif()
endif()
set(TORCH_X86_BASELINE "${_torch_x86_baseline_default}" CACHE STRING
    "-march baseline for x86 builds")
if(CPU_INTEL AND NOT USE_NATIVE_ARCH)
  check_cxx_compiler_flag("-march=${TORCH_X86_BASELINE}" COMPILER_SUPPORTS_X86_BASELINE)
  if(COMPILER_SUPPORTS_X86_BASELINE)
    string(APPEND CMAKE_C_FLAGS " -march=${TORCH_X86_BASELINE}")
    string(APPEND CMAKE_CXX_FLAGS " -march=${TORCH_X86_BASELINE}")
    if(USE_ROCM)
      string(APPEND CMAKE_HIP_FLAGS
          " -Xarch_host -march=${TORCH_X86_BASELINE}")
    endif()
  else()
    message(WARNING "Compiler does not support -march=${TORCH_X86_BASELINE}; building for generic x86-64.")
  endif()
endif()
unset(_torch_processor_description)
unset(_torch_x86_baseline_default)

# The ATen CPU kernels have no separate AVX2 slice any more (see
# cmake/Codegen.cmake): the DEFAULT slice is the AVX2 tier. Check the macro the
# compiler actually defines rather than trusting the flag name, so an overridden
# TORCH_X86_BASELINE or a -march=native on a pre-AVX2 host fails here instead of
# silently building scalar kernels.
if(CPU_INTEL)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS}")
  check_cxx_source_compiles("
    #if !defined(__AVX2__) || !defined(__FMA__)
    #error no avx2
    #endif
    int main() { return 0; }" TORCH_X86_BASELINE_HAS_AVX2)
  cmake_pop_check_state()
  if(NOT TORCH_X86_BASELINE_HAS_AVX2)
    message(FATAL_ERROR
        "x86 builds require AVX2 and FMA, but the compiler does not define "
        "__AVX2__/__FMA__ with the selected flags "
        "(TORCH_X86_BASELINE=${TORCH_X86_BASELINE}, USE_NATIVE_ARCH=${USE_NATIVE_ARCH}).")
  endif()
endif()

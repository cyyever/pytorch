set(JIT_TEST_ROOT ${TORCH_ROOT}/test/cpp/jit)

# Build separate libraries the define custom classes/operators used from our Python tests.
# These are intended to be used with torch.ops.load_library() in our Python test suite.
add_library(torchbind_test SHARED
  ${JIT_TEST_ROOT}/test_custom_class_registrations.h
  ${JIT_TEST_ROOT}/test_custom_class_registrations.cpp
)
target_link_libraries(torchbind_test torch)

if(INSTALL_TEST)
  install(TARGETS torchbind_test DESTINATION lib)
endif()

# Build the cpp gtest binary containing the cpp-only tests.
set(JIT_TEST_SRCS
  ${JIT_TEST_ROOT}/source_range_test.cpp
  ${JIT_TEST_ROOT}/test_class_parser.cpp
  ${JIT_TEST_ROOT}/test_custom_class_registrations.h
  ${JIT_TEST_ROOT}/test_custom_class_registrations.cpp
  ${JIT_TEST_ROOT}/test_lexer.cpp
  ${JIT_TEST_ROOT}/test_qualified_name.cpp
  ${JIT_TEST_ROOT}/test_schema_info.cpp
)

add_executable(test_jit
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${JIT_TEST_SRCS}
)

# We also build with UBSAN flag in build_asan.h
# Skip UBSan on ROCm: combining ASAN + UBSan on ROCm Clang causes
# misaligned ASAN global metadata, triggering spurious ASAN alignment
# check aborts at startup.
if(USE_ASAN AND NOT USE_ROCM)
  target_compile_options(test_jit PRIVATE "-fsanitize=undefined")
  target_link_libraries(test_jit PRIVATE "-fsanitize=undefined")
endif()



# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_jit PRIVATE USE_GTEST)

set(JIT_TEST_DEPENDENCIES torch gtest_main gmock)

target_link_libraries(test_jit PRIVATE ${JIT_TEST_DEPENDENCIES})
target_include_directories(test_jit PRIVATE ${ATen_CPU_INCLUDE})

if(LINUX)
  #Update to target_link_options when CMake version can be upgraded
endif()

if(USE_CUDA)
  target_compile_definitions(test_jit PRIVATE USE_CUDA)
  # Suppress sign compare checks for NVFUSER JIT tests
  if(NOT MSVC)
    target_compile_options(test_jit PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>)
  endif()
elseif(USE_ROCM)
  target_link_libraries(test_jit PRIVATE
    hiprtc::hiprtc
    hip::amdhip64
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_jit PRIVATE USE_ROCM)
endif()

if(INSTALL_TEST)
  set_target_properties(test_jit PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
  install(TARGETS test_jit DESTINATION bin)
  install(FILES ${JIT_TEST_ROOT}/test_interpreter_async.pt DESTINATION bin)
  install(FILES ${JIT_TEST_ROOT}/script_module_v4.ptl DESTINATION bin)
  install(DIRECTORY ${JIT_TEST_ROOT}/upgrader_models DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_jit> DESTINATION bin OPTIONAL)
  endif()
endif()

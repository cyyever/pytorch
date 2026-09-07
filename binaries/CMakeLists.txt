set(record_function_benchmark_src "record_function_benchmark.cc")
if(USE_ROCM)
  file(RELATIVE_PATH record_function_benchmark_src
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PYTORCH_HIPIFY_DIR}/binaries/hip/record_function_benchmark.cc")
endif()
caffe2_binary_target("${record_function_benchmark_src}")
target_include_directories(record_function_benchmark PUBLIC
  ${CMAKE_BINARY_DIR}/aten/src)

caffe2_binary_target("record_function_benchmark.cc")
target_include_directories(record_function_benchmark PUBLIC
  ${CMAKE_BINARY_DIR}/aten/src)

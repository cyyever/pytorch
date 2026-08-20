caffe2_binary_target("parallel_info.cc")
target_include_directories(parallel_info PUBLIC
  ${CMAKE_BINARY_DIR}/aten/src) # provides "ATen/TypeExtendedInterface.h" to ATen.h

caffe2_binary_target("record_function_benchmark.cc")
target_include_directories(record_function_benchmark PUBLIC
  ${CMAKE_BINARY_DIR}/aten/src)


if(BUILD_TEST)
  # Core overhead benchmark
  caffe2_binary_target("core_overhead_benchmark.cc")
  target_link_libraries(core_overhead_benchmark benchmark)
endif()

caffe2_binary_target("dump_operator_names.cc")

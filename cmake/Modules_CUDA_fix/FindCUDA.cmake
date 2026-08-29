# This forwards to the vendored `./upstream/FindCUDA.cmake`, which sits ahead of
# CMake's own copy on CMAKE_MODULE_PATH.
# See ./README.md for details.

set(UPSTREAM_FIND_CUDA_DIR "${CMAKE_CURRENT_LIST_DIR}/upstream/")

include("${UPSTREAM_FIND_CUDA_DIR}/FindCUDA.cmake")

#pragma once

#include <cstddef>
#include <new>

namespace c10 {

// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr size_t gAlignment = 64;

constexpr size_t gPagesize = 4096;
// since the default thp pagesize is 2MB, enable thp only
// for buffers of size 2MB or larger to avoid memory bloating
constexpr size_t gAlloc_threshold_thp = static_cast<size_t>(2) * 1024 * 1024;

// Cache line size used to avoid false sharing between threads. libc++ still
// does not implement this, so the fallback stays.
#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif
} // namespace c10

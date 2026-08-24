#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>
#include <unistd.h>

#define F_PROT_NONE 0x0
#define F_PROT_READ 0x1
#define F_PROT_WRITE 0x2

namespace openreg {

void* mmap(size_t size) {
  void* addr = ::mmap(
      nullptr,
      size,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0);
  return (addr == MAP_FAILED) ? nullptr : addr;
}

void munmap(void* addr, size_t size) {
  ::munmap(addr, size);
}

int mprotect(void* addr, size_t size, int prot) {
  int native_prot = 0;
  if (prot == F_PROT_NONE)
    native_prot = PROT_NONE;
  else {
    if (prot & F_PROT_READ)
      native_prot |= PROT_READ;
    if (prot & F_PROT_WRITE)
      native_prot |= PROT_WRITE;
  }

  return ::mprotect(addr, size, native_prot);
}

int alloc(void** mem, size_t alignment, size_t size) {
  return posix_memalign(mem, alignment, size);
}

void free(void* mem) {
  ::free(mem);
}

long get_pagesize() {
  return sysconf(_SC_PAGESIZE);
}

} // namespace openreg

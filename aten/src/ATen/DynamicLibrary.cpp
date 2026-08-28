#include <c10/util/Exception.h>
#include <ATen/DynamicLibrary.h>

#include <dlfcn.h>
#include <libgen.h>

namespace at {

// Unix

static void* checkDL(void* x) {
  if (!x) {
    TORCH_CHECK_WITH(DynamicLibraryError, false, "Error in dlopen or dlsym: ", dlerror());
  }

  return x;
}
DynamicLibrary::DynamicLibrary(const char* name, const char* alt_name, bool leak_handle_): leak_handle(leak_handle_), handle(dlopen(name, RTLD_LOCAL | RTLD_NOW)) {
  if (!handle) {
    if (alt_name) {
      handle = dlopen(alt_name, RTLD_LOCAL | RTLD_NOW);
      if (!handle) {
        TORCH_CHECK_WITH(DynamicLibraryError, false, "Error in dlopen for library ", name, "and ", alt_name);
      }
    } else {
      TORCH_CHECK_WITH(DynamicLibraryError, false, "Error in dlopen: ", dlerror());
    }
  }
}

void* DynamicLibrary::sym(const char* name) {
  AT_ASSERT(handle);
  return checkDL(dlsym(handle, name));
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle || leak_handle) {
    return;
  }
  dlclose(handle);
}

} // namespace at

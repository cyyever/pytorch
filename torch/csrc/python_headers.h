#pragma once
// Python.h redefines _XOPEN_SOURCE and _POSIX_C_SOURCE; save and restore them
// so we do not clash with whatever the including translation unit set.
// See https://bugs.python.org/issue17120
#pragma push_macro("_XOPEN_SOURCE")
#pragma push_macro("_POSIX_C_SOURCE")
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE

#include <Python.h>
// Not pulled in by Python.h, and the frame API users below us do not include
// it themselves.
#include <frameobject.h>

#pragma pop_macro("_XOPEN_SOURCE")
#pragma pop_macro("_POSIX_C_SOURCE")

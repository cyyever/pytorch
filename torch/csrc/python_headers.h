#pragma once
// workaround for Python 2 issue: https://bugs.python.org/issue17120
// NOTE: It looks like this affects Python 3 as well.
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

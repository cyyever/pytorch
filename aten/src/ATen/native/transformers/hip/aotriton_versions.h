#pragma once

#ifdef USE_ROCM

#include <aotriton/config.h>

#define AOTRITON_VERSION_INT(x, y) (x * 100 + y)
#define AOTRITON_VERSION_CURRENT (AOTRITON_VERSION_MAJOR * 100 + AOTRITON_VERSION_MINOR)

// cmake/External/aotriton.cmake downloads 0.13b, but AOTRITON_INSTALLED_PREFIX
// and PYTORCH_AOTRITON_COMMIT let a build supply its own. Declare the floor so
// the V3 API, the compact varlen LSE layout and the hdim_qk != hdim_vo support
// can be assumed unconditionally.
#if AOTRITON_VERSION_CURRENT < AOTRITON_VERSION_INT(0, 12)
#error "PyTorch requires AOTriton 0.12 or newer."
#endif

#endif

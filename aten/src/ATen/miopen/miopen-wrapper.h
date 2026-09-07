#pragma once

// Enable MIOpen Beta APIs including miopenSetTensorDescriptorV2 which supports
// 64-bit tensor dimensions/strides for large tensors (numel > INT32_MAX).
// Reference: https://github.com/ROCm/MIOpen/pull/2838
#ifndef MIOPEN_BETA_API
#define MIOPEN_BETA_API 1
#endif

#include <miopen/miopen.h>
#include <miopen/version.h>

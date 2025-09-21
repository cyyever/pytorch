#pragma once

#include <ATen/cuda/CUDAContext.h>
#if defined(USE_ROCM)
#include <hipsparse/hipsparse-version.h>
#define HIPSPARSE_VERSION ((hipsparseVersionMajor*100000) + (hipsparseVersionMinor*100) + hipsparseVersionPatch)
#endif

// cuSparse Generic API descriptor pointers were changed to const in CUDA 12.0
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && \
    (CUSPARSE_VERSION < 12000)
#define AT_USE_CUSPARSE_NON_CONST_DESCRIPTORS() 1
#else
#define AT_USE_CUSPARSE_NON_CONST_DESCRIPTORS() 0
#endif

#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && \
    (CUSPARSE_VERSION >= 12000)
#define AT_USE_CUSPARSE_CONST_DESCRIPTORS() 1
#else
#define AT_USE_CUSPARSE_CONST_DESCRIPTORS() 0
#endif

#if defined(USE_ROCM)
// hipSparse const API added in v2.4.0
#if HIPSPARSE_VERSION >= 200400
#define AT_USE_HIPSPARSE_CONST_DESCRIPTORS() 1
#define AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_GENERIC_API() 1
#else
#define AT_USE_HIPSPARSE_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS() 1
#define AT_USE_HIPSPARSE_GENERIC_API() 1
#endif
#else // USE_ROCM
#define AT_USE_HIPSPARSE_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_GENERIC_API() 0
#endif // USE_ROCM

// BSR triangular solve functions were added in hipSPARSE 1.11.2 (ROCm 4.5.0)
#if defined(CUDART_VERSION) || defined(USE_ROCM)
#define AT_USE_HIPSPARSE_TRIANGULAR_SOLVE() 1
#else
#define AT_USE_HIPSPARSE_TRIANGULAR_SOLVE() 0
#endif

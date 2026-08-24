#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPUCC
#define FUNCAPI __host__ __device__
#define INLINE __forceinline__
#else
#define FUNCAPI
#define INLINE inline
#endif

#define RESTRICT __restrict__

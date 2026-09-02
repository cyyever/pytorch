#pragma once
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#elif defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

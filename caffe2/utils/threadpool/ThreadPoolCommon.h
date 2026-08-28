#if !defined(CAFFE2_UTILS_THREADPOOL_COMMON_H_)
#define CAFFE2_UTILS_THREADPOOL_COMMON_H_

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// caffe2 depends upon NNPACK, which depends upon this threadpool, so
// unfortunately we can't reference core/common.h here

// Define enabled when building for iOS devices
#if (defined(__APPLE__) &&                                            \
       (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define C10_IOS 1
#endif

#endif

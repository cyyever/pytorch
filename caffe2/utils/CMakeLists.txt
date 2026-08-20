list(APPEND Caffe2_CPU_SRCS
  utils/string_utils.cc
  utils/threadpool/ThreadPool.cc
)

if(USE_PTHREADPOOL)
  list(APPEND Caffe2_CPU_SRCS
    utils/threadpool/pthreadpool-cpp.cc
    utils/threadpool/thread_pool_guard.cpp
  )
endif()

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)

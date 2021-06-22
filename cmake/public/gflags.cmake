# ---[ gflags

# We will try to use the config mode first, and then manual find.
find_package(gflags CONFIG QUIET)
if(NOT TARGET gflags)
  find_package(gflags MODULE QUIET)
endif()

# After above, we should have the gflags target now.
if(NOT TARGET gflags)
  message(WARNING
      "Caffe2: gflags cannot be found. Depending on whether you are building "
      "Caffe2 or a Caffe2 dependent library, the next warning / error will "
      "give you more info.")
endif()

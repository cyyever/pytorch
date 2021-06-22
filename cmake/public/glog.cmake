# ---[ glog

# We will try to use the config mode first, and then manual find.
find_package(glog CONFIG QUIET)
if(NOT TARGET glog::glog)
  find_package(glog MODULE QUIET)
endif()

# After above, we should have the glog::glog target now.
if(NOT TARGET glog::glog)
  message(WARNING
      "Caffe2: glog cannot be found. Depending on whether you are building "
      "Caffe2 or a Caffe2 dependent library, the next warning / error will "
      "give you more info.")
endif()

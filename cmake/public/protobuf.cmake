# ---[ Protobuf

# We will try to use the config mode first, and then manual find.
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
  find_package(Protobuf MODULE QUIET)
endif()

# After above, we should have the protobuf related target now.
if((NOT TARGET protobuf::libprotobuf) AND (NOT TARGET protobuf::libprotobuf-lite))
  message(WARNING
      "Protobuf cannot be found. Depending on whether you are building Caffe2 "
      "or a Caffe2 dependent library, the next warning / error will give you "
      "more info.")
endif()

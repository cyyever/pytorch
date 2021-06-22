# Finds Google Protocol Buffers library and compilers and extends
# the standard cmake script with version and python generation support

# Main entry for protobuf. If we are building on Android, iOS or we have hard
# coded BUILD_CUSTOM_PROTOBUF, we will hard code the use of custom protobuf
# in the submodule.

find_package(protobuf)
if((NOT TARGET protobuf::libprotobuf) AND (NOT TARGET protobuf::libprotobuf-lite))
  message(FATAL_ERROR
      "Protobuf cannot be found. Caffe2 will have to build with libprotobuf. "
      "Please set the proper paths so that I can find protobuf correctly.")
endif()

get_target_property(__tmp protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Caffe2 protobuf include directory: " ${__tmp})
include_directories(BEFORE SYSTEM ${__tmp})

# If Protobuf_VERSION is known (true in most cases, false if we are building
# local protobuf), then we will add a protobuf version check in
# Caffe2Config.cmake.in.
if(DEFINED ${Protobuf_VERSION})
  set(CAFFE2_KNOWN_PROTOBUF_VERSION TRUE)
else()
  set(CAFFE2_KNOWN_PROTOBUF_VERSION FALSE)
  set(Protobuf_VERSION "Protobuf_VERSION_NOTFOUND")
endif()


# Figure out which protoc to use.
# If CAFFE2_CUSTOM_PROTOC_EXECUTABLE is set, we assume the user knows
# what they're doing and we blindly use the specified protoc. This
# is typically the case when cross-compiling where protoc must be
# compiled for the host architecture and libprotobuf must be
# compiled for the target architecture.
# If CAFFE2_CUSTOM_PROTOC_EXECUTABLE is NOT set, we use the protoc
# target that is built as part of including the protobuf project.
if(EXISTS "${CAFFE2_CUSTOM_PROTOC_EXECUTABLE}")
  set(CAFFE2_PROTOC_EXECUTABLE ${CAFFE2_CUSTOM_PROTOC_EXECUTABLE})
else()
  set(CAFFE2_PROTOC_EXECUTABLE protobuf::protoc)
endif()

################################################################################################
# Modification of standard 'protobuf_generate_cpp()' with output dir parameter and python support
# Usage:
#   caffe2_protobuf_generate_cpp_py(<srcs_var> <hdrs_var> <python_var> <proto_files>)
function(caffe2_protobuf_generate_cpp_py srcs_var hdrs_var python_var)
  if(NOT ARGN)
    message(SEND_ERROR "Error: caffe_protobuf_generate_cpp_py() called without any proto files")
    return()
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  set(${python_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h")
    list(APPEND ${python_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}_pb2.py")

    # Add TORCH_API prefix to protobuf classes and methods in all cases
    set(DLLEXPORT_STR "dllexport_decl=TORCH_API:")

    # Note: the following depends on PROTOBUF_PROTOC_EXECUTABLE. This
    # is done to make sure protoc is built before attempting to
    # generate sources if we're using protoc from the third_party
    # directory and are building it as part of the Caffe2 build. If
    # points to an existing path, it is a no-op.

    if(${CAFFE2_LINK_LOCAL_PROTOBUF})
      # We need to rewrite the pb.h files to route GetEmptyStringAlreadyInited
      # through our wrapper in proto_utils so the memory location test
      # is correct.
      add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h"
               "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}_pb2.py"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
        COMMAND ${CAFFE2_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --cpp_out=${DLLEXPORT_STR}${PROJECT_BINARY_DIR} ${abs_fil}
        COMMAND ${CAFFE2_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --python_out "${PROJECT_BINARY_DIR}" ${abs_fil}

        # If we remove all reference to these pb.h files from external
        # libraries and binaries this rewrite can be removed.
        COMMAND ${CMAKE_COMMAND} -DFILENAME=${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h -DNAMESPACES=caffe\;caffe2\;onnx\;torch -DLOCAL_PROTOBUF=${CAFFE2_LINK_LOCAL_PROTOBUF} -P ${PROJECT_SOURCE_DIR}/cmake/ProtoBufPatch.cmake

        DEPENDS ${CAFFE2_PROTOC_EXECUTABLE} ${abs_fil}
        COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
    else()
      add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h"
               "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}_pb2.py"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
        COMMAND ${CAFFE2_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --cpp_out=${DLLEXPORT_STR}${PROJECT_BINARY_DIR} ${abs_fil}
        COMMAND ${CAFFE2_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --python_out "${PROJECT_BINARY_DIR}" ${abs_fil}
        COMMAND ${CMAKE_COMMAND} -DFILENAME=${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h -DNAMESPACES=caffe\;caffe2\;onnx\;torch -DLOCAL_PROTOBUF=${CAFFE2_LINK_LOCAL_PROTOBUF} -P ${PROJECT_SOURCE_DIR}/cmake/ProtoBufPatch.cmake
        DEPENDS ${CAFFE2_PROTOC_EXECUTABLE} ${abs_fil}
        COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
    endif()
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()

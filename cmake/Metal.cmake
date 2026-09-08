if(NOT APPLE)
    return()
endif()

set(METAL_CFLAGS -Wall -Wextra -fno-fast-math)
if(WERROR)
    string(APPEND METAL_CFLAGS -Werror)
endif()

# Headers transitively included by .metal sources. Any change to these must
# retrigger the metal -> air step, since xcrun metal does not emit depfiles
# we can hand to ninja.
file(GLOB METAL_HEADER_DEPS CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/c10/metal/*.h"
     "${CMAKE_SOURCE_DIR}/aten/src/ATen/native/mps/kernels/*.h")

function(metal_to_air SRC TARGET FLAGS)
    add_custom_command(COMMAND xcrun metal -c ${SRC} -I ${CMAKE_SOURCE_DIR} -I ${CMAKE_SOURCE_DIR}/aten/src -o ${TARGET} ${FLAGS} ${METAL_CFLAGS}
                       DEPENDS ${SRC} ${METAL_HEADER_DEPS}
                       OUTPUT ${TARGET}
                       COMMENT "Compiling ${SRC} to ${TARGET}"
                       VERBATIM)
endfunction()

function(air_to_metallib TARGET OBJECTS)
    set(_OBJECTS ${OBJECTS} ${ARGN})
    add_custom_command(COMMAND xcrun metallib -o ${TARGET} ${_OBJECTS}
                       DEPENDS ${_OBJECTS}
                       OUTPUT ${TARGET}
                       COMMENT "Linking ${TARGET}"
                       VERBATIM)
endfunction()

function(metal_to_metallib_h SHADER)
    cmake_path(ABSOLUTE_PATH SHADER OUTPUT_VARIABLE SHADER_ABSOLUTE)

    cmake_path(GET SHADER STEM SHADER_STEM)
    cmake_path(APPEND ${CMAKE_CURRENT_BINARY_DIR} native mps ${SHADER_STEM} OUTPUT_VARIABLE SHADER_HDR)
    cmake_path(APPEND_STRING SHADER_HDR "_metallib.h")

    add_custom_command(COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/write_metallib_headers.py ${SHADER_ABSOLUTE} ${SHADER_HDR}
                       DEPENDS ${SHADER_ABSOLUTE} ${CMAKE_SOURCE_DIR}/scripts/write_metallib_headers.py ${METAL_HEADER_DEPS}
                       OUTPUT ${SHADER_HDR}
                       COMMENT "Generating metallib wrapper header for ${SHADER}"
                       VERBATIM)

    return(PROPAGATE SHADER_HDR)
endfunction()

set(BFLOAT_METAL_CODE "
  kernel void inc(device bfloat* ptr,
                   uint idx [[thread_position_in_grid]]) {
    ptr[idx] += 1;
  }
")
# The sentinel names the probe, not just the fact that one ran: a tree
# configured when the shaders were built for Metal 3 cached a CAN_COMPILE_METAL
# that answered a different question, and must ask again.
if(NOT CAN_COMPILE_METAL_40_FOUND)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/bfloat_inc.metal" "${BFLOAT_METAL_CODE}")
    # Same -std and deployment target the shaders are built with, so a toolchain
    # whose SDK predates 27.0 fails here rather than on every shader.
    execute_process(COMMAND xcrun metal -std=metal4.0 -mmacos-version-min=27.0 bfloat_inc.metal
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                    OUTPUT_VARIABLE XCRUN_OUTPUT
                    ERROR_VARIABLE XCRUN_OUTPUT
                    RESULT_VARIABLE XCRUN_RC)
    if(${XCRUN_RC} EQUAL 0)
        message(STATUS "Machine can compile Metal 4.0 shaders")
        set(CAN_COMPILE_METAL YES CACHE BOOL "Host can compile metal shaders" FORCE)
    else()
        message(WARNING "Machine can not compile Metal 4.0 shaders, fails with ${XCRUN_OUTPUT}")
        set(CAN_COMPILE_METAL NO CACHE BOOL "Host can compile metal shaders" FORCE)
    endif()
    set(CAN_COMPILE_METAL_40_FOUND YES CACHE INTERNAL "Run check for shader compiler")
endif()

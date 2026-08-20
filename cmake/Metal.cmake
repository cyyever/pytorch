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
                       DEPENDS ${SHADER_ABSOLUTE} ${CMAKE_SOURCE_DIR}/scripts/write_metallib_headers.py
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
set(LAMBDA_METAL_CODE "
  kernel void test(device float* ptr,
                   uint idx [[thread_position_in_grid]]) {
    auto fn = [](float x) { return x + 1.0; };
    ptr[idx] = fn(ptr[idx]);
  }
")
if(NOT CAN_COMPILE_METAL_FOUND)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/bfloat_inc.metal" "${BFLOAT_METAL_CODE}")
    execute_process(COMMAND xcrun metal -std=metal3.1 bfloat_inc.metal
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                    OUTPUT_VARIABLE XCRUN_OUTPUT
                    ERROR_VARIABLE XCRUN_OUTPUT
                    RESULT_VARIABLE XCRUN_RC)
    if(${XCRUN_RC} EQUAL 0)
        message(STATUS "Machine can compile metal shaders")
        set(CAN_COMPILE_METAL YES CACHE BOOL "Host can compile metal shaders")
    else()
        message(WARNING "Machine can not compile metal shaders, fails with ${XCRUN_OUTPUT}")
        set(CAN_COMPILE_METAL NO CACHE BOOL "Host can compile metal shaders")
    endif()
    if(CAN_COMPILE_METAL)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/lambda_test.metal" "${LAMBDA_METAL_CODE}")
        execute_process(COMMAND xcrun metal -std=metal4.0 -c lambda_test.metal -o /dev/null
                        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                        OUTPUT_VARIABLE XCRUN_OUTPUT
                        ERROR_VARIABLE XCRUN_OUTPUT
                        RESULT_VARIABLE XCRUN_RC)
        if(${XCRUN_RC} EQUAL 0)
            message(STATUS "Metal toolchain supports Metal 4.0")
            set(CAN_COMPILE_METAL_40 YES CACHE BOOL "Host can compile Metal 4.0 shaders" FORCE)
        else()
            message(STATUS "Metal toolchain does not support Metal 4.0")
            set(CAN_COMPILE_METAL_40 NO CACHE BOOL "Host can compile Metal 4.0 shaders" FORCE)
        endif()
    else()
        set(CAN_COMPILE_METAL_40 NO CACHE BOOL "Host can compile Metal 4.0 shaders" FORCE)
    endif()
    set(CAN_COMPILE_METAL_FOUND YES CACHE INTERNAL "Run check for shader compiler")
endif()

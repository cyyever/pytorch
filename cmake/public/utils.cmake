################################################################################################
# Exclude and prepend functionalities
function(exclude OUTPUT INPUT)
set(EXCLUDES ${ARGN})
foreach(EXCLUDE ${EXCLUDES})
        list(REMOVE_ITEM INPUT "${EXCLUDE}")
endforeach()
set(${OUTPUT} ${INPUT} PARENT_SCOPE)
endfunction(exclude)

function(prepend OUTPUT PREPEND)
set(OUT "")
foreach(ITEM ${ARGN})
        list(APPEND OUT "${PREPEND}${ITEM}")
endforeach()
set(${OUTPUT} ${OUT} PARENT_SCOPE)
endfunction(prepend)

################################################################################################
# Parses a version string that might have values beyond major, minor, and patch
# and set version variables for the library.
# Usage:
#   caffe2_parse_version_str(<library_name> <version_string>)
function(caffe2_parse_version_str LIBNAME VERSIONSTR)
  string(REGEX REPLACE "^([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MAJOR "${VERSIONSTR}")
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MINOR  "${VERSIONSTR}")
  string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_PATCH "${VERSIONSTR}")
  set(${LIBNAME}_VERSION_MAJOR ${${LIBNAME}_VERSION_MAJOR} ${ARGN} PARENT_SCOPE)
  set(${LIBNAME}_VERSION_MINOR ${${LIBNAME}_VERSION_MINOR} ${ARGN} PARENT_SCOPE)
  set(${LIBNAME}_VERSION_PATCH ${${LIBNAME}_VERSION_PATCH} ${ARGN} PARENT_SCOPE)
  set(${LIBNAME}_VERSION "${${LIBNAME}_VERSION_MAJOR}.${${LIBNAME}_VERSION_MINOR}.${${LIBNAME}_VERSION_PATCH}" PARENT_SCOPE)
endfunction()

###
# Removes common indentation from a block of text to produce code suitable for
# setting to `python -c`, or using with pycmd. This allows multiline code to be
# nested nicely in the surrounding code structure.
#
# This function respsects Python_EXECUTABLE if it defined, otherwise it uses
# `python` and hopes for the best. An error will be thrown if it is not found.
#
# Args:
#     outvar : variable that will hold the stdout of the python command
#     text   : text to remove indentation from
#
function(dedent outvar text)
  # Use Python_EXECUTABLE if it is defined, otherwise default to python
  if("${Python_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python3")
  else()
    set(_python_exe "${Python_EXECUTABLE}")
  endif()
  set(_fixup_cmd "import sys; from textwrap import dedent; print(dedent(sys.stdin.read()))")
  file(WRITE "${CMAKE_BINARY_DIR}/indented.txt" "${text}")
  execute_process(
    COMMAND "${_python_exe}" -c "${_fixup_cmd}"
    INPUT_FILE "${CMAKE_BINARY_DIR}/indented.txt"
    RESULT_VARIABLE _dedent_exitcode
    OUTPUT_VARIABLE _dedent_text)
  if(NOT _dedent_exitcode EQUAL 0)
    message(ERROR " Failed to remove indentation from: \n\"\"\"\n${text}\n\"\"\"
    Python dedent failed with error code: ${_dedent_exitcode}")
    message(FATAL_ERROR " Python dedent failed with error code: ${_dedent_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_dedent_text}" _dedent_text)
  set(${outvar} "${_dedent_text}" PARENT_SCOPE)
endfunction()


function(pycmd_no_exit outvar exitcode cmd)
  # Use Python_EXECUTABLE if it is defined, otherwise default to python
  if("${Python_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python")
  else()
    set(_python_exe "${Python_EXECUTABLE}")
  endif()
  # run the actual command
  execute_process(
    COMMAND "${_python_exe}" -c "${cmd}"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _output)
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_output}" _output)
  set(${outvar} "${_output}" PARENT_SCOPE)
  set(${exitcode} "${_exitcode}" PARENT_SCOPE)
endfunction()


###
# Helper function to run `python -c "<cmd>"` and capture the results of stdout
#
# Runs a python command and populates an outvar with the result of stdout.
# Common indentation in the text of `cmd` is removed before the command is
# executed, so the caller does not need to worry about indentation issues.
#
# This function respsects Python_EXECUTABLE if it defined, otherwise it uses
# `python` and hopes for the best. An error will be thrown if it is not found.
#
# Args:
#     outvar : variable that will hold the stdout of the python command
#     cmd    : text representing a (possibly multiline) block of python code
#
function(pycmd outvar cmd)
  dedent(_dedent_cmd "${cmd}")
  pycmd_no_exit(_output _exitcode "${_dedent_cmd}")

  if(NOT _exitcode EQUAL 0)
    message(ERROR " Failed when running python code: \"\"\"\n${_dedent_cmd}\n\"\"\"")
    message(FATAL_ERROR " Python command failed with error code: ${_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_output}" _output)
  set(${outvar} "${_output}" PARENT_SCOPE)
endfunction()


##############################################################################
# Macro to update cached options.
macro(caffe2_update_option variable value)
  if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
    get_property(__help_string CACHE ${variable} PROPERTY HELPSTRING)
    set(${variable} ${value} CACHE BOOL ${__help_string} FORCE)
  else()
    set(${variable} ${value})
  endif()
endmacro()


##############################################################################
# Add an interface library definition that is dependent on the source.
#
# It's probably easiest to explain why this macro exists, by describing
# what things would look like if we didn't have this macro.
#
# Let's suppose we want to statically link against torch.  We've defined
# a library in cmake called torch, and we might think that we just
# target_link_libraries(my-app PUBLIC torch).  This will result in a
# linker argument 'libtorch.a' getting passed to the linker.
#
# Unfortunately, this link command is wrong!  We have static
# initializers in libtorch.a that would get improperly pruned by
# the default link settings.  What we actually need is for you
# to do -Wl,--whole-archive,libtorch.a -Wl,--no-whole-archive to ensure
# that we keep all symbols, even if they are (seemingly) not used.
#
# What caffe2_interface_library does is create an interface library
# that indirectly depends on the real library, but sets up the link
# arguments so that you get all of the extra link settings you need.
# The result is not a "real" library, and so we have to manually
# copy over necessary properties from the original target.
#
# (The discussion above is about static libraries, but a similar
# situation occurs for dynamic libraries: if no symbols are used from
# a dynamic library, it will be pruned unless you are --no-as-needed)
macro(caffe2_interface_library SRC DST)
  add_library(${DST} INTERFACE)
  add_dependencies(${DST} ${SRC})
  # Depending on the nature of the source library as well as the compiler,
  # determine the needed compilation flags.
  get_target_property(__src_target_type ${SRC} TYPE)
  # Depending on the type of the source library, we will set up the
  # link command for the specific SRC library.
  if(${__src_target_type} STREQUAL "STATIC_LIBRARY")
    # In the case of static library, we will need to add whole-static flags.
    target_link_libraries(${DST} INTERFACE $<LINK_LIBRARY:WHOLE_ARCHIVE,${SRC}>)
  elseif(${__src_target_type} STREQUAL "SHARED_LIBRARY")
    target_link_libraries(${DST} INTERFACE ${SRC})
  else()
    message(FATAL_ERROR
        "You made a CMake build file error: target " ${SRC}
        " must be of type either STATIC_LIBRARY or SHARED_LIBRARY. However, "
        "I got " ${__src_target_type} ".")
  endif()
  # For all other interface properties, manually inherit from the source target.
  set_target_properties(${DST} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_DEFINITIONS>
    INTERFACE_COMPILE_OPTIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_OPTIONS>
    INTERFACE_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_INCLUDE_DIRECTORIES>
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>)
endmacro()


##############################################################################
# Creating a Caffe2 binary target with sources specified with relative path.
# Usage:
#   caffe2_binary_target(target_name_or_src <src1> [<src2>] [<src3>] ...)
# If only target_name_or_src is specified, this target is build with one single
# source file and the target name is autogen from the filename. Otherwise, the
# target name is given by the first argument and the rest are the source files
# to build the target.
function(caffe2_binary_target target_name_or_src)
  # https://cmake.org/cmake/help/latest/command/function.html
  # Checking that ARGC is greater than # is the only way to ensure
  # that ARGV# was passed to the function as an extra argument.
  if(ARGC GREATER 1)
    set(__target ${target_name_or_src})
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${ARGN}")
  else()
    get_filename_component(__target ${target_name_or_src} NAME_WE)
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${target_name_or_src}")
  endif()
  add_executable(${__target} ${__srcs})
  target_link_libraries(${__target} torch_library)
  # If we have Caffe2_MODULES defined, we will also link with the modules.
  if(DEFINED Caffe2_MODULES)
    target_link_libraries(${__target} ${Caffe2_MODULES})
  endif()
  install(TARGETS ${__target} DESTINATION bin)
endfunction()

function(caffe2_hip_binary_target target_name_or_src)
  if(ARGC GREATER 1)
    set(__target ${target_name_or_src})
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${ARGN}")
  else()
    get_filename_component(__target ${target_name_or_src} NAME_WE)
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${target_name_or_src}")
  endif()

  caffe2_binary_target(${target_name_or_src})

  target_compile_options(${__target} PRIVATE ${HIP_CXX_FLAGS})
  target_include_directories(${__target} PRIVATE ${Caffe2_HIP_INCLUDE})
endfunction()


##############################################################################
# Multiplex between adding libraries for CUDA versus HIP (AMD Software Stack).
# Usage:
#   torch_cuda_based_add_library(cuda_target)
#
macro(torch_cuda_based_add_library cuda_target)
  if(USE_ROCM)
    add_library(${cuda_target} ${ARGN})
  elseif(USE_CUDA)
    add_library(${cuda_target} ${ARGN})
  else()
  endif()
endmacro()

##############################################################################
# Get the HIP arch flags specified by PYTORCH_ROCM_ARCH.
# Usage:
#   torch_hip_get_arch_list(variable_to_store_flags)
#
macro(torch_hip_get_arch_list store_var)
  if(DEFINED ENV{PYTORCH_ROCM_ARCH})
    set(_TMP $ENV{PYTORCH_ROCM_ARCH})
  else()
    # Use arch of installed GPUs as default
    execute_process(COMMAND "rocm_agent_enumerator" COMMAND bash "-c" "grep -v gfx000 | sort -u | xargs | tr -d '\n'"
                    RESULT_VARIABLE ROCM_AGENT_ENUMERATOR_RESULT
                    OUTPUT_VARIABLE ROCM_ARCH_INSTALLED)
    if(NOT ROCM_AGENT_ENUMERATOR_RESULT EQUAL 0)
      message(FATAL_ERROR " Could not detect ROCm arch for GPUs on machine. Result: '${ROCM_AGENT_ENUMERATOR_RESULT}'")
    endif()
    set(_TMP ${ROCM_ARCH_INSTALLED})
  endif()
  string(REPLACE " " ";" ${store_var} "${_TMP}")
endmacro()

##############################################################################
# Get the XPU arch flags specified by TORCH_XPU_ARCH_LIST.
# Usage:
#   torch_xpu_get_arch_list(variable_to_store_flags)
#
macro(torch_xpu_get_arch_list store_var)
  if(DEFINED ENV{TORCH_XPU_ARCH_LIST})
    set(${store_var} $ENV{TORCH_XPU_ARCH_LIST})
  endif()
endmacro()

##############################################################################
# GPU architectures this build knows about. sm_89 (Ada) is the floor: the
# lists are also what clamps autodetection, so keep the numeric entries sorted
# by release, not by value (Rubin's 10.7 is below Blackwell's 12.0).
# Sets, in the caller's scope: _cuda_known_archs, _cuda_common_archs,
# _cuda_min_arch and _cuda_limit_arch.
macro(torch_cuda_architecture_lists)
  set(_cuda_known_archs "Ada" "Hopper" "Blackwell")
  set(_cuda_common_archs "8.9" "9.0" "9.0a" "10.0" "10.0a" "11.0a" "12.0" "12.0a")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.4)
    list(APPEND _cuda_known_archs "Rubin")
    list(APPEND _cuda_common_archs "10.7" "10.7a")
  endif()

  # Oldest and newest arch this toolkit can compile for, used to clamp
  # autodetection.
  set(_cuda_plain_archs ${_cuda_common_archs})
  list(FILTER _cuda_plain_archs INCLUDE REGEX "^[0-9]+\\.[0-9]+$")
  list(SORT _cuda_plain_archs COMPARE NATURAL)
  list(GET _cuda_plain_archs 0 _cuda_min_arch)
  list(GET _cuda_plain_archs -1 _cuda_limit_arch)
endmacro()

##############################################################################
# Detect the compute capabilities of the GPUs installed on this machine.
# Usage:
#   torch_cuda_detect_installed_gpus(variable_to_store_archs)
function(torch_cuda_detect_installed_gpus out_variable)
  torch_cuda_architecture_lists()

  if(NOT CUDA_GPU_DETECT_OUTPUT)
    set(file "${PROJECT_BINARY_DIR}/detect_cuda_compute_capabilities.cu")
    file(WRITE ${file} ""
      "#include <cuda_runtime.h>\n"
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    # The probe only calls the runtime API and has no device code, so it needs
    # no architecture; saying so explicitly keeps it from inheriting whatever
    # CMAKE_CUDA_ARCHITECTURES the enclosing project happens to be in.
    try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file}
            CMAKE_FLAGS "-DCMAKE_CUDA_ARCHITECTURES=OFF"
            RUN_OUTPUT_VARIABLE compute_capabilities)

    # Filter unrelated content out of the output.
    string(REGEX MATCHALL "[0-9]+\\.[0-9]+" compute_capabilities "${compute_capabilities}")

    if(run_result EQUAL 0)
      set(CUDA_GPU_DETECT_OUTPUT ${compute_capabilities}
        CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
    endif()
  endif()

  if(NOT CUDA_GPU_DETECT_OUTPUT)
    message(STATUS "Automatic GPU detection failed. Building for common architectures.")
    set(${out_variable} ${_cuda_common_archs} PARENT_SCOPE)
    return()
  endif()

  set(detected "")
  separate_arguments(CUDA_GPU_DETECT_OUTPUT)
  foreach(item IN ITEMS ${CUDA_GPU_DETECT_OUTPUT})
    if(item VERSION_GREATER _cuda_limit_arch)
      # Too new for SASS; fall back to the newest known arch's PTX for JIT.
      list(APPEND detected "${_cuda_limit_arch}+PTX")
    elseif(item VERSION_LESS _cuda_min_arch)
      # Below the supported floor. Build for the floor instead of failing, so
      # configuration still succeeds; the detected GPU cannot run the result.
      message(STATUS "Detected GPU architecture ${item} is below the minimum supported "
                     "architecture ${_cuda_min_arch}; building for "
                     "${_cuda_min_arch} instead.")
      list(APPEND detected "${_cuda_min_arch}")
    else()
      list(APPEND detected "${item}")
    endif()
  endforeach()

  set(${out_variable} ${detected} PARENT_SCOPE)
endfunction()

##############################################################################
# Translate a list of CUDA architectures into nvcc -gencode flags.
#   arch_list : Auto | Common | All | LIST(ARCH_AND_PTX ...)
#     - "Auto" builds for the GPUs installed on this machine
#     - "Common" and "All" cover the common and the entire known subsets
#   ARCH_AND_PTX : NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
#     NAME: Ada Hopper Blackwell Rubin
# Usage:
#   torch_cuda_select_nvcc_arch_flags(variable_to_store_flags [arch_list])
# Additionally sets ${variable_to_store_flags}_readable to the numeric list.
function(torch_cuda_select_nvcc_arch_flags out_variable)
  torch_cuda_architecture_lists()

  set(arch_list "${ARGN}")
  if("X${arch_list}" STREQUAL "X")
    set(arch_list "Auto")
  endif()

  if("${arch_list}" STREQUAL "All")
    set(arch_list ${_cuda_known_archs})
  elseif("${arch_list}" STREQUAL "Common")
    set(arch_list ${_cuda_common_archs})
  elseif("${arch_list}" STREQUAL "Auto")
    torch_cuda_detect_installed_gpus(arch_list)
    message(STATUS "Autodetected CUDA architecture(s): ${arch_list}")
  endif()

  set(cuda_arch_bin)
  set(cuda_arch_ptx)

  string(REGEX REPLACE "[ \t]+" ";" arch_list "${arch_list}")
  list(REMOVE_DUPLICATES arch_list)
  foreach(arch_name ${arch_list})
    set(arch_bin)
    set(arch_ptx)
    set(add_ptx FALSE)
    if(arch_name MATCHES "(.*)\\+PTX$")
      set(add_ptx TRUE)
      set(arch_name ${CMAKE_MATCH_1})
    endif()
    if(arch_name MATCHES "^([0-9]+\\.[0-9][af]?(\\([0-9]+\\.[0-9]\\))?)$")
      set(arch_bin ${CMAKE_MATCH_1})
      set(arch_ptx ${arch_bin})
    elseif(arch_name STREQUAL "Ada")
      set(arch_bin 8.9)
      set(arch_ptx 8.9)
    elseif(arch_name STREQUAL "Hopper")
      set(arch_bin 9.0)
      set(arch_ptx 9.0)
    elseif(arch_name STREQUAL "Blackwell+Tegra")
      set(arch_bin 10.1)
    elseif(arch_name STREQUAL "Blackwell")
      set(arch_bin 10.0 12.0)
      set(arch_ptx 10.0 12.0)
    elseif(arch_name STREQUAL "Rubin")
      set(arch_bin 10.7)
      set(arch_ptx 10.7)
    else()
      message(FATAL_ERROR "Unknown CUDA architecture name in TORCH_CUDA_ARCH_LIST: ${arch_name}")
    endif()
    list(APPEND cuda_arch_bin ${arch_bin})
    if(add_ptx)
      if(NOT arch_ptx)
        set(arch_ptx ${arch_bin})
      endif()
      list(APPEND cuda_arch_ptx ${arch_ptx})
    endif()
  endforeach()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" cuda_arch_ptx "${cuda_arch_ptx}")
  string(REGEX MATCHALL "[0-9()]+[af]?" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+[af]?"   cuda_arch_ptx "${cuda_arch_ptx}")

  if(cuda_arch_bin)
    list(REMOVE_DUPLICATES cuda_arch_bin)
  endif()
  if(cuda_arch_ptx)
    list(REMOVE_DUPLICATES cuda_arch_ptx)
  endif()

  set(nvcc_flags "")
  set(nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(arch ${cuda_arch_bin})
    if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified ARCH for the concrete CODE
      list(APPEND nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND nvcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify ARCH for the concrete CODE, we assume ARCH=CODE
      list(APPEND nvcc_flags -gencode arch=compute_${arch},code=sm_${arch})
      list(APPEND nvcc_archs_readable sm_${arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(arch ${cuda_arch_ptx})
    list(APPEND nvcc_flags -gencode arch=compute_${arch},code=compute_${arch})
    list(APPEND nvcc_archs_readable compute_${arch})
  endforeach()

  string(REPLACE ";" " " nvcc_archs_readable "${nvcc_archs_readable}")
  set(${out_variable}          ${nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${nvcc_archs_readable} PARENT_SCOPE)
endfunction()

##############################################################################
# Get the NVCC arch flags specified by TORCH_CUDA_ARCH_LIST.
# Usage:
#   torch_cuda_get_nvcc_gencode_flag(variable_to_store_flags)
#
macro(torch_cuda_get_nvcc_gencode_flag store_var)
  # setting nvcc arch flags
  # We need to support the explicitly and conveniently defined TORCH_CUDA_ARCH_LIST
  if((NOT DEFINED TORCH_CUDA_ARCH_LIST) AND (DEFINED ENV{TORCH_CUDA_ARCH_LIST}))
    set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
  endif()

  # sm_89 (Ada) is the oldest architecture this build supports, so anything
  # older cannot be built even if it is asked for.
  foreach(_torch_arch ${TORCH_CUDA_ARCH_LIST})
    if(_torch_arch MATCHES "^([0-9]+\\.[0-9]+)")
      if(CMAKE_MATCH_1 VERSION_LESS 8.9)
        message(FATAL_ERROR
            "PyTorch needs compute capability 8.9 or above, but TORCH_CUDA_ARCH_LIST "
            "contains ${_torch_arch}.")
      endif()
    endif()
  endforeach()

  torch_cuda_select_nvcc_arch_flags(${store_var} ${TORCH_CUDA_ARCH_LIST})
endmacro()


##############################################################################
# Add standard compile options.
# Usage:
#   torch_compile_options(lib_name)
function(torch_compile_options libname)
  set_property(TARGET ${libname} PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})

  # until they can be unified, keep these lists synced with setup.py
  set(private_compile_options
    -Wall
    -Wextra
    -Wdeprecated
    -Wunused
    -Wno-array-bounds
    -Wno-unknown-pragmas
    -Wno-strict-overflow
    -Wno-strict-aliasing
    )
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND private_compile_options -Wredundant-move)
    list(APPEND private_compile_options -Wno-interference-size)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(NOT USE_CUDA)
      # NS: One can not compile CUDA code with extra-semi flag as nvcc generates code like
      # namespace MemoryOps_cu_d8602b38_109889 __attribute__((visibility("hidden")))  { };
      list(APPEND private_compile_options -Wextra-semi)
    endif()
    list(APPEND private_compile_options -Wmove)
  endif()

  if(WERROR)
    list(APPEND private_compile_options
      -Werror
      -Werror=ignored-attributes
      -Werror=inconsistent-missing-override
      -Werror=inconsistent-missing-destructor-override
      -Werror=pedantic
      -Werror=unused
      -Wno-error=unused-parameter
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      list(APPEND private_compile_options -Werror=unused-but-set-variable -Werror=cpp)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      list(APPEND private_compile_options -Werror=macro-redefined -Werror=deprecated-copy-with-dtor)
    endif()
  endif()

  # After the WERROR block on purpose: -Werror=pedantic re-enables whatever was
  # switched off ahead of it. C10_ANONYMOUS_VARIABLE needs __COUNTER__, which
  # only C2y standardizes, so pedantic rejects every use of it.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND private_compile_options -Wno-c2y-extensions)
  endif()


  target_compile_options(${libname} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${private_compile_options}>)
  if(USE_CUDA)
    foreach(option IN LISTS private_compile_options)
      if(CMAKE_CUDA_HOST_COMPILER_ID STREQUAL "GNU")
        if("${option}" STREQUAL "-Wextra-semi")
          continue()
        endif()
      endif()
      # nvcc hands the host compiler its own preprocessed output, whose `#line`
      # directives that compiler reports as a GNU extension, so -Werror=pedantic
      # fails on every CUDA translation unit regardless of what the source says.
      if("${option}" STREQUAL "-Werror=pedantic")
        continue()
      endif()
      # -Xcompiler=<opt>, not -Xcompiler <opt>: target_compile_options
      # de-duplicates the repeated -Xcompiler token, so the space form leaves
      # one -Xcompiler followed by the whole option list and nvcc hands only
      # the first entry to the host compiler.
      target_compile_options(${libname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${option}>)
    endforeach()
  endif()

  if(NOT USE_ASAN AND NOT USE_UBSAN)
    # Enable hidden visibility by default to make it easier to debug issues with
    # TORCH_API annotations. Hidden visibility with selective default visibility
    # behaves close enough to Windows' dllimport/dllexport.
    #
    # Unfortunately, hidden visibility messes up some ubsan warnings because
    # templated classes crossing library boundary get duplicated (but identical)
    # definitions. It's easier to just disable it.
    #
    # Device translation units need this spelled out per language or they keep
    # default visibility and export their internals. nvcc has to route it to
    # its host pass; hipcc is clang and takes it directly.
    target_compile_options(${libname} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>: -fvisibility=hidden>
        $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fvisibility=hidden>
        $<$<COMPILE_LANGUAGE:HIP>: -fvisibility=hidden>)
  endif()

endfunction()

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(CheckLinkerFlag)

##############################################################################
# Check if given flag is supported and append it to provided outputvar
# Also define HAS_UPPER_CASE_FLAG_NAME variable
# Usage:
#   append_cxx_flag_if_supported("-Werror" CMAKE_CXX_FLAGS)
function(append_cxx_flag_if_supported flag outputvar)
    string(TOUPPER "HAS${flag}" _FLAG_NAME)
    string(REGEX REPLACE "[=-]" "_" _FLAG_NAME "${_FLAG_NAME}")
    # GCC silents unknown -Wno-XXX flags, so we detect the corresponding -WXXX.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      string(REGEX REPLACE "Wno-" "W" new_flag "${flag}")
    else()
      set(new_flag ${flag})
    endif()
    check_cxx_compiler_flag("${new_flag}" ${_FLAG_NAME})
    if(${_FLAG_NAME})
        string(APPEND ${outputvar} " ${flag}")
        set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
    endif()
endfunction()

function(append_c_flag_if_supported flag outputvar)
    string(TOUPPER "HAS${flag}" _FLAG_NAME)
    string(REGEX REPLACE "[=-]" "_" _FLAG_NAME "${_FLAG_NAME}")

    # GCC silences unknown -Wno-XXX flags, so test the corresponding -WXXX.
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        string(REGEX REPLACE "^Wno-" "W" new_flag "${flag}")
    else()
        set(new_flag "${flag}")
    endif()

    check_c_compiler_flag("${new_flag}" ${_FLAG_NAME})
    if(${_FLAG_NAME})
        string(APPEND ${outputvar} " ${flag}")
        set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
    endif()
endfunction()

function(target_compile_options_if_supported target flag)
  set(_compile_options "")
  append_cxx_flag_if_supported("${flag}" _compile_options)
  if(NOT "${_compile_options}" STREQUAL "")
    target_compile_options(${target} PRIVATE ${flag})
  endif()
endfunction()

function(target_link_options_if_supported tgt flag)
  string(TOUPPER "HAS_LINKER${flag}" _FLAG_NAME)
  string(REGEX REPLACE "[=,-]" "_" _FLAG_NAME "${_FLAG_NAME}")
  check_linker_flag(C "LINKER:${flag}" ${_FLAG_NAME})
  if(${_FLAG_NAME})
    target_link_options("${tgt}" PRIVATE "LINKER:${flag}")
  else()
    message(WARNING "Attempted to use unsupported link option : ${flag}.")
  endif()
endfunction()

##############################################################################
# Apply binary layout optimization to ${tgt}. This includes using an
# optimized symbol order (USE_PRIORITIZED_TEXT_FOR_LD) and post-link
# optimization using LLVM BOLT (USE_LLVM_BOLT).
#
# When USE_LLVM_BOLT is enabled, original libraries are moved to the
# prebolt/ subdirectory and bolted libraries are written in their place.
# Pass the target followed by profile names in priority order:
# torch_optimize_layout_if_enabled(<target> [<profile>...])
# Falls back to lib<target>.yaml if specified profiles don't exist.
function(torch_optimize_layout_if_enabled tgt)
  if(USE_PRIORITIZED_TEXT_FOR_LD)
    if(CMAKE_LINKER_TYPE STREQUAL "LLD")
      target_link_options("${tgt}" PRIVATE "LINKER:--no-warn-symbol-ordering")
      target_link_options("${tgt}" PRIVATE "LINKER:--symbol-ordering-file=${LINKER_SCRIPT_FILE_IN}")
    else()
      add_dependencies("${tgt}" generate_linker_script)
      target_link_options("${tgt}" PRIVATE "LINKER:-T${LINKER_SCRIPT_FILE_OUT}")
    endif()
  endif()

  if(USE_LLVM_BOLT)
    # BOLT needs --emit-relocs. This flag increases the binary size so we
    # scope it to bolt optimized targets rather than applying globally.
    target_link_options_if_supported(${tgt} "--emit-relocs")
    find_file(
      _bolt_profile
      NAMES ${ARGN} "lib${tgt}.yaml"
      PATHS "${LLVM_BOLT_PROFILES_DIR}"
      NO_DEFAULT_PATH
      NO_CMAKE_FIND_ROOT_PATH
      NO_CACHE
      REQUIRED
    )
    message(STATUS "Using BOLT profile for ${tgt}: ${_bolt_profile}")
    set_property(TARGET ${tgt} APPEND PROPERTY LINK_DEPENDS "${_bolt_profile}")
    set(_logfile "${CMAKE_BINARY_DIR}/logs/llvm-bolt-lib${tgt}.txt")
    set(_prebolt "$<TARGET_FILE_DIR:${tgt}>/prebolt/$<TARGET_FILE_NAME:${tgt}>")
    add_custom_command(
      TARGET ${tgt} POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E make_directory "$<PATH:GET_PARENT_PATH,${_logfile}>"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "$<PATH:GET_PARENT_PATH,${_prebolt}>"
      COMMAND "${CMAKE_COMMAND}" -E rename "$<TARGET_FILE:${tgt}>" "${_prebolt}"
      COMMAND "${LLVM_BOLT_EXECUTABLE}" "${_prebolt}"
              -o "$<TARGET_FILE:${tgt}>"
              "-data=${_bolt_profile}" "-log-file=${_logfile}"
              -lite -infer-stale-profile
              -reorder-blocks=ext-tsp -reorder-functions=cdsort
              -split-functions -split-all-cold -split-eh -dyno-stats
              --update-debug-sections
      COMMENT "Optimizing $<TARGET_FILE_NAME:${tgt}> with LLVM BOLT (original kept in prebolt/)"
      VERBATIM
    )
  endif()
endfunction()

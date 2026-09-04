# libstdc++ implements the C++17 parallel algorithms with the vendored LLVM
# PSTL, whose default backend is oneTBB, so <execution> compiles but does not
# link until TBB is present. oneTBB ships its own config package, so there is
# nothing to find by hand.
#
# x86 only: this is where the parallel algorithms are wanted, and it keeps the
# dependency off the macOS/arm64 build, where oneTBB is community-supported.

if(TARGET TBB::tbb)
  return()
endif()

# A config package found is not a library found. Arch's CUDA package symlinks
# ${CUDA_HOME}/lib/cmake to /usr/lib/cmake, so the system TBBConfig.cmake is
# also reachable under the CUDA prefix -- where, being relocatable, it derives
# its import prefix from its apparent path and points at a libtbb that is not
# there. That resolves at link time, not here, so validate and keep looking.
foreach(_attempt RANGE 3)
  find_package(TBB CONFIG QUIET)
  if(NOT TARGET TBB::tbb)
    break()
  endif()

  set(_tbb_lib "")
  get_target_property(_tbb_lib TBB::tbb IMPORTED_LOCATION)
  if(NOT _tbb_lib)
    get_target_property(_tbb_cfgs TBB::tbb IMPORTED_CONFIGURATIONS)
    foreach(_cfg IN LISTS _tbb_cfgs)
      get_target_property(_tbb_lib TBB::tbb IMPORTED_LOCATION_${_cfg})
      if(_tbb_lib)
        break()
      endif()
    endforeach()
  endif()

  if(_tbb_lib AND EXISTS "${_tbb_lib}")
    message(STATUS "Found oneTBB ${TBB_VERSION}: ${_tbb_lib}")
    return()
  endif()

  message(STATUS "Ignoring oneTBB at ${TBB_DIR}: ${_tbb_lib} does not exist")
  get_filename_component(_tbb_bad_prefix "${TBB_DIR}/../../.." ABSOLUTE)
  list(APPEND CMAKE_IGNORE_PREFIX_PATH "${_tbb_bad_prefix}")
  unset(TBB_DIR CACHE)
endforeach()

set(_aotriton_database_cmake
    "${AOTRITON_SOURCE_DIR}/v3src/CMakeLists.txt")
file(READ "${_aotriton_database_cmake}" _aotriton_database_contents)

set(_aotriton_unfiltered
    "file(GLOB_RECURSE DB_TARS CONFIGURE_DEPENDS \"\${DBXZ_BASE}/*.sqlite3.tar.xz\")")
set(_aotriton_filtered_without_cleanup
    "file(GLOB DB_TARS CONFIGURE_DEPENDS \"\${DBXZ_BASE}/*.sqlite3.tar.xz\")\nfile(GLOB_RECURSE ARCH_DB_TARS CONFIGURE_DEPENDS \"\${DBXZ_BASE}/amd/gfx1201/*.sqlite3.tar.xz\")\nlist(APPEND DB_TARS \${ARCH_DB_TARS})")
set(_aotriton_filtered
    "file(REMOVE_RECURSE \"\${AOTRITON_V2_BUILD_DIR}/database\")\nfile(GLOB DB_TARS CONFIGURE_DEPENDS \"\${DBXZ_BASE}/*.sqlite3.tar.xz\")\nfile(GLOB_RECURSE ARCH_DB_TARS CONFIGURE_DEPENDS \"\${DBXZ_BASE}/amd/gfx1201/*.sqlite3.tar.xz\")\nlist(APPEND DB_TARS \${ARCH_DB_TARS})")

if(_aotriton_database_contents MATCHES
   "REMOVE_RECURSE.*AOTRITON_V2_BUILD_DIR}/database")
  return()
endif()

string(FIND "${_aotriton_database_contents}" "${_aotriton_unfiltered}"
       _aotriton_unfiltered_offset)
if(NOT _aotriton_unfiltered_offset EQUAL -1)
  set(_aotriton_search "${_aotriton_unfiltered}")
else()
  string(FIND "${_aotriton_database_contents}"
         "${_aotriton_filtered_without_cleanup}"
         _aotriton_filtered_offset)
  if(_aotriton_filtered_offset EQUAL -1)
    message(FATAL_ERROR
      "AOTriton database extraction logic changed; update the gfx1201 filter")
  endif()
  set(_aotriton_search "${_aotriton_filtered_without_cleanup}")
endif()

string(REPLACE "${_aotriton_search}" "${_aotriton_filtered}"
       _aotriton_database_contents "${_aotriton_database_contents}")
file(WRITE "${_aotriton_database_cmake}" "${_aotriton_database_contents}")

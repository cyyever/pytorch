# Apply the patches this fork carries for a vendored dependency.
#
# CMake's own mechanism for this is FetchContent's PATCH_COMMAND, which runs
# once and remembers it with a stamp file. That fits content CMake fetched
# itself; it does not fit a git submodule, whose tree the user can reset at any
# time -- the stamp would survive and the patches would silently stop being
# applied. So this checks the tree instead of trusting a marker: a patch either
# applies, or has to prove it is already applied by reversing cleanly.
#
#   apply_third_party_patches(<patch_dir> <source_dir> <label>)
#
# Patches are applied in filename order, which is what their numeric prefixes
# are for.
function(apply_third_party_patches patch_dir source_dir label)
  file(GLOB _patches "${patch_dir}/*.patch")
  list(SORT _patches)
  foreach(_patch IN LISTS _patches)
    execute_process(
      COMMAND git apply --check --unidiff-zero ${_patch}
      WORKING_DIRECTORY ${source_dir}
      RESULT_VARIABLE _applies
      ERROR_QUIET)
    if(_applies EQUAL 0)
      execute_process(
        COMMAND git apply --unidiff-zero ${_patch}
        WORKING_DIRECTORY ${source_dir}
        COMMAND_ERROR_IS_FATAL ANY)
    else()
      execute_process(
        COMMAND git apply --reverse --check --unidiff-zero ${_patch}
        WORKING_DIRECTORY ${source_dir}
        RESULT_VARIABLE _already_applied
        ERROR_QUIET)
      if(NOT _already_applied EQUAL 0)
        message(FATAL_ERROR
            "${_patch} neither applies nor is already applied to ${label}")
      endif()
    endif()
  endforeach()
endfunction()

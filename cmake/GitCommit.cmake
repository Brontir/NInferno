# Writes the commit the tree is built at into OUTPUT, a C++ source defining
# ninferno::runtime::build_git_commit(), for the build section of the decision fingerprint
# (src/runtime/engine/decision_fingerprint.h). Run at every build by the ninferno_git_commit target
# (src/runtime/CMakeLists.txt), so a commit checked out since the last configure is seen; OUTPUT is
# rewritten only when its text changes, so an unchanged commit rebuilds nothing.
#
# Inputs: SOURCE_DIR (the project's source tree), OUTPUT, GIT_EXECUTABLE (empty: git absent).
# The commit is the full `git rev-parse HEAD` hash, with "-dirty" when tracked files differ from it
# (the same name on every clone, shallow or not, with or without tags), or "unknown" when git is absent, fails, or this
# source tree is not itself the top of a git checkout (a tree unpacked inside another repository
# does not take that one's commit).
cmake_minimum_required(VERSION 3.28)

set(commit "unknown")
if(GIT_EXECUTABLE AND EXISTS "${GIT_EXECUTABLE}")
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --show-toplevel
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE git_top OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE git_result ERROR_QUIET)
  if(git_result EQUAL 0 AND NOT git_top STREQUAL "")
    file(REAL_PATH "${git_top}" git_top)
    file(REAL_PATH "${SOURCE_DIR}" source_top)
    if(git_top STREQUAL source_top)
      execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE head OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE git_result ERROR_QUIET)
      if(git_result EQUAL 0 AND head MATCHES "^[0-9a-f]+$")
        set(commit "${head}")
        execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
          WORKING_DIRECTORY "${SOURCE_DIR}"
          OUTPUT_VARIABLE changed RESULT_VARIABLE dirty_result ERROR_QUIET)
        if(NOT dirty_result EQUAL 0 OR NOT changed STREQUAL "")
          string(APPEND commit "-dirty")
        endif()
      else()
        message(STATUS "GitCommit: git failed in ${SOURCE_DIR}; build commit is \"unknown\"")
      endif()
    endif()
  endif()
endif()

set(text "// Generated at build time by cmake/GitCommit.cmake. Do not edit.
namespace ninferno::runtime {
const char* build_git_commit() noexcept { return \"${commit}\"; }
} // namespace ninferno::runtime
")
set(old "")
if(EXISTS "${OUTPUT}")
  file(READ "${OUTPUT}" old)
endif()
if(NOT old STREQUAL text)
  file(WRITE "${OUTPUT}" "${text}")
endif()

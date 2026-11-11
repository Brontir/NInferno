# Test declarations are included from tests/CMakeLists.txt so executable paths and
# CTest working directories stay under build/tests.
function(ninferno_test_includes target)
  ninferno_internal_includes(${target})
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

# A member archive a test names (ninferno_core, ninferno_ops, ...: NINFERNO_SHARED_MEMBERS) is linked
# through its redirect ninferno::<m> (cmake/NInfernoTargets.cmake): the archive in a static test,
# the shared test library in a shared one. Writes the mapped list to `out`.
function(ninferno_test_libraries out)
  set(libs "")
  foreach(lib IN LISTS ARGN)
    if(lib MATCHES "^ninferno_(.+)$" AND CMAKE_MATCH_1 IN_LIST NINFERNO_SHARED_MEMBERS)
      set(lib "ninferno::${CMAKE_MATCH_1}")
    endif()
    list(APPEND libs "${lib}")
  endforeach()
  set(${out} ${libs} PARENT_SCOPE)
endfunction()

# STATIC_LINK: the test keeps the product's static link in the shared test tree
# (NINFERNO_SHARED_OP_TESTS); give the reason where it is declared. Every other test with device
# code takes the shared library (ninferno_decide_link_modes).
function(ninferno_add_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "NEEDS_SOURCE_DIR;NO_DEFAULT_TEST;STATIC_LINK" ""
    "SOURCES;LIBRARIES")
  add_executable(${name} ${arg_SOURCES})
  ninferno_test_includes(${name})
  ninferno_test_libraries(libs ${arg_LIBRARIES})
  target_link_libraries(${name} PRIVATE ${libs})
  if(arg_STATIC_LINK)
    set_target_properties(${name} PROPERTIES NINFERNO_STATIC_LINK ON)
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFERNO_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFERNO_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  # NO_DEFAULT_TEST: the caller registers the executable's runs itself (e.g. one per case group).
  if(NOT arg_NO_DEFAULT_TEST)
    add_test(NAME ${name} COMMAND ${name})
  endif()
endfunction()

# Apply these to the translation unit containing the oracle, including shared
# test support libraries. Executable options do not propagate into those libraries.
function(ninferno_op_oracle_options target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ninferno_add_op_test name)
  ninferno_add_test(${name} ${ARGN})
  if(NOT "NO_DEFAULT_TEST" IN_LIST ARGN)
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  ninferno_op_oracle_options(${name})
endfunction()

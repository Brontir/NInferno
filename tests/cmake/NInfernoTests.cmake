# Test declarations are included from tests/CMakeLists.txt so executable paths and
# CTest working directories stay under build/tests.
function(ninferno_test_includes target)
  ninferno_internal_includes(${target})
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

# The shared test tree's library (NINFERNO_SHARED_OP_TESTS: ninferno_ops_shared) is defined once,
# in cmake/SharedTestLibrary.cmake; tests reach it through the link redirects of
# cmake/NInfernoTargets.cmake.

# Every test carries labels (tools/audit/ctest_labels.sh fails on a test without):
#   host    runs and passes with no GPU (CUDA_VISIBLE_DEVICES= ctest -L host)
#   gpu     needs a CUDA device
#   real    needs a model artifact or fixtures it does not carry, and skips (exit 77) without
#   golden  compares against a committed golden (decision fingerprint, selector, choice and plan goldens, including the MoE plan)
#   slow    over 60 s on one development GPU board
# A test combines them: a gpu test is also slow, a real one is also gpu.
set(NINFERNO_TEST_LABEL_NAMES host gpu real golden slow)

# ninferno_label_tests(<test>... LABELS <label>...): sets the labels of tests registered with add_test.
function(ninferno_label_tests)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "" "LABELS")
  if(NOT arg_UNPARSED_ARGUMENTS OR NOT arg_LABELS)
    message(FATAL_ERROR "ninferno_label_tests(<test>... LABELS <label>...): tests and labels are both needed")
  endif()
  foreach(label IN LISTS arg_LABELS)
    if(NOT label IN_LIST NINFERNO_TEST_LABEL_NAMES)
      message(FATAL_ERROR "unknown test label '${label}' on ${arg_UNPARSED_ARGUMENTS}; known: ${NINFERNO_TEST_LABEL_NAMES}")
    endif()
  endforeach()
  set_tests_properties(${arg_UNPARSED_ARGUMENTS} PROPERTIES LABELS "${arg_LABELS}")
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

# ninferno_add_test(<name> LABELS <label>... SOURCES ... LIBRARIES ... [NEEDS_SOURCE_DIR] [NO_DEFAULT_TEST]
#                   [STATIC_LINK])
# LABELS is required unless NO_DEFAULT_TEST: the caller then labels the runs it registers.
# STATIC_LINK: the test keeps the product's static link in the shared test tree
# (NINFERNO_SHARED_OP_TESTS); give the reason where it is declared. Every other test with device
# code takes the shared library (ninferno_decide_link_modes).
function(ninferno_add_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "NEEDS_SOURCE_DIR;NO_DEFAULT_TEST;STATIC_LINK" ""
    "SOURCES;LIBRARIES;LABELS")
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
  # NO_DEFAULT_TEST: the caller registers the executable's runs itself (e.g. one per case group)
  # and labels them with ninferno_label_tests; LABELS here would label nothing.
  if(arg_NO_DEFAULT_TEST AND arg_LABELS)
    message(FATAL_ERROR "ninferno_add_test(${name}): LABELS with NO_DEFAULT_TEST labels nothing; "
                        "label the registered runs with ninferno_label_tests")
  endif()
  if(NOT arg_NO_DEFAULT_TEST)
    if(NOT arg_LABELS)
      message(FATAL_ERROR "ninferno_add_test(${name}): LABELS host|gpu|real|golden|slow is required")
    endif()
    add_test(NAME ${name} COMMAND ${name})
    ninferno_label_tests(${name} LABELS ${arg_LABELS})
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

# An op test runs kernels against an oracle: gpu unless LABELS says otherwise (a NO_DEFAULT_TEST
# caller labels its runs itself).
function(ninferno_add_op_test name)
  if("LABELS" IN_LIST ARGN OR "NO_DEFAULT_TEST" IN_LIST ARGN)
    ninferno_add_test(${name} ${ARGN})
  else()
    ninferno_add_test(${name} LABELS gpu ${ARGN})
  endif()
  if(NOT "NO_DEFAULT_TEST" IN_LIST ARGN)
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  ninferno_op_oracle_options(${name})
endfunction()

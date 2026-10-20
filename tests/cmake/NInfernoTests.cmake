# Test declarations are included from tests/CMakeLists.txt so executable paths and
# CTest working directories stay under build/tests.
function(ninferno_test_includes target)
  ninferno_internal_includes(${target})
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

# NINFERNO_SHARED_OP_TESTS: ninferno_ops, ninferno_core (and the non-RDC NVFP4 archive) whole in one
# shared library. The archive already carries its device link (CUDA_RESOLVE_DEVICE_SYMBOLS), so
# the library links no device code of its own.
if(NINFERNO_SHARED_OP_TESTS)
  add_library(ninferno_ops_shared SHARED "${CMAKE_CURRENT_LIST_DIR}/ops_shared_anchor.cpp")
  set(ninferno_whole $<TARGET_FILE:ninferno_ops> $<TARGET_FILE:ninferno_core>)
  if(TARGET ninferno_nvfp4_non_rdc)
    list(APPEND ninferno_whole $<TARGET_FILE:ninferno_nvfp4_non_rdc>)
  endif()
  target_link_options(ninferno_ops_shared PRIVATE
    "LINKER:--whole-archive" ${ninferno_whole} "LINKER:--no-whole-archive")
  target_link_libraries(ninferno_ops_shared PRIVATE ninferno_ops
    PUBLIC ninferno::cudart CUDA::cuda_driver Threads::Threads)
  set_target_properties(ninferno_ops_shared PROPERTIES
    LINKER_LANGUAGE CXX
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
endif()

# Links ninferno_ops into a test target: the shared library under NINFERNO_SHARED_OP_TESTS (ninferno_ops
# then only for its usage requirements), the archive otherwise.
function(ninferno_link_ops target scope)
  if(NINFERNO_SHARED_OP_TESTS)
    target_link_libraries(${target} ${scope} ninferno_ops_shared "$<COMPILE_ONLY:ninferno_ops>")
  else()
    target_link_libraries(${target} ${scope} ninferno_ops)
  endif()
endfunction()

function(ninferno_add_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "NEEDS_SOURCE_DIR;NO_DEFAULT_TEST" "" "SOURCES;LIBRARIES")
  add_executable(${name} ${arg_SOURCES})
  ninferno_test_includes(${name})
  if(arg_LIBRARIES STREQUAL "ninferno_ops")
    ninferno_link_ops(${name} PRIVATE)
  else()
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
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

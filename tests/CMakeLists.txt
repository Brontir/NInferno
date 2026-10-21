find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NInfernoTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/dense/tests.cmake)
include(models/gemma4/tests.cmake)
include(models/qwen3_5/tests.cmake)
include(models/speculation/tests.cmake)
include(ops/tests.cmake)
# The converter's and the oracle's tests run the ninferno-convert executable.
if(NINFERNO_BUILD_APPS)
  include(convert_native/tests.cmake)
  include(oracle/tests.cmake)
  include(cmake/AppRefusalTests.cmake)
endif()

# Speculation support (registry, corpus, fingerprints, profiles, watchdog, manifests): no GPU.
ninferno_add_test(ninferno_spec_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec.cpp"
  LIBRARIES ninferno_spec)

# tools/spec/registry_add.py: the "*" fallback and the load table warnings, no GPU.
add_test(NAME ninferno_registry_add_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_registry_add.py)

# Repository hygiene, no GPU: only scripts (a #! first line) may be committed executable.
add_test(NAME ninferno_file_modes_test
  COMMAND bash "${PROJECT_SOURCE_DIR}/tools/check_file_modes.sh")
set_tests_properties(ninferno_file_modes_test PROPERTIES SKIP_RETURN_CODE 77)

# Determinism, no GPU: decision code reads no clock, event timer or NVML (the monitor is
# write-only; docs/maintainer/deterministic-decisions.md, principle 3).
add_test(NAME ninferno_decision_inputs_test
  COMMAND bash "${PROJECT_SOURCE_DIR}/tools/audit/decision_inputs.sh")

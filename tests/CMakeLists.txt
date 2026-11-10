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
  # ninferno-ui's update of a moved hub model against a stand-in hub and ninferno-serve: one
  # restart when idle, one revert when the new conversion does not start. No GPU.
  add_test(NAME ninferno_ui_moved_update_test
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/ui_moved_update_test.py
            $<TARGET_FILE:ninferno-ui> ${CMAKE_CURRENT_LIST_DIR}/convert_native)
  set_tests_properties(ninferno_ui_moved_update_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ninferno-connect (#339): a stand-in router dialed over loopback (Upgrade, then HTTP/2 with the
# roles reversed; plain and TLS) and a stand-in engine; registration, concurrent streams, a reset
# that closes the engine's connection (also before any byte of the answer), ping timeout, GOAWAY
# with a replacement while the old session drains, a refused key, the router's certificate checked.
# No GPU.
if(TARGET ninferno_connect)
  ninferno_add_test(ninferno_connect_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_connect.cpp"
    LIBRARIES ninferno_connect PkgConfig::NGHTTP2 OpenSSL::SSL OpenSSL::Crypto)
endif()

# Speculation support (registry, corpus, fingerprints, profiles, watchdog, manifests): no GPU.
ninferno_add_test(ninferno_spec_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec.cpp"
  LIBRARIES ninferno_spec)

# Speculation per measured pair (spec/pairs.h) on the lab's spec-tune measurements: no GPU.
ninferno_add_test(ninferno_spec_pairs_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_pairs.cpp"
  LIBRARIES ninferno_spec)

# A speculation round's cost from facts (spec/round_cost.h) on the registry's pairs: no GPU.
ninferno_add_test(ninferno_spec_round_cost_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_round_cost.cpp"
  LIBRARIES ninferno_spec)

# The rows a restricted draft head keeps (spec/draft_head.h) on the Qwen ranking: no GPU.
ninferno_add_test(ninferno_spec_draft_head_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_draft_head.cpp"
  LIBRARIES ninferno_spec)

# tools/spec/registry_add.py: the "*" fallback and the load table warnings, no GPU.
add_test(NAME ninferno_registry_add_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_registry_add.py)

# tools/kit/validate.py: the user measurement kit's rejection rules and verdicts, no GPU.
add_test(NAME ninferno_kit_validate_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_kit_validate.py)

# tools/bench/selection_regret.py: the offline selection regret's statuses, holdouts and exit codes
# (#301), no GPU.
add_test(NAME ninferno_selection_regret_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_selection_regret.py)
# tools/bench/regret_collect.py: route sweeps and the production selection into its input, no GPU.
add_test(NAME ninferno_regret_collect_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_regret_collect.py)

# tools/bench/gap_map.py: measured times against the counted DRAM and tensor laws (#619), on
# synthetic inputs, no GPU.
add_test(NAME ninferno_gap_map_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_gap_map.py)
set_tests_properties(ninferno_gap_map_test PROPERTIES LABELS host)

# Repository hygiene, no GPU: only scripts (a #! first line) may be committed executable.
add_test(NAME ninferno_file_modes_test
  COMMAND bash "${PROJECT_SOURCE_DIR}/tools/check_file_modes.sh")
set_tests_properties(ninferno_file_modes_test PROPERTIES SKIP_RETURN_CODE 77)

# Determinism, no GPU: decision code reads no clock, event timer or NVML (the monitor is
# write-only; docs/maintainer/deterministic-decisions.md, principle 3). A POSIX shell check: on
# native Windows `bash` may be WSL's, which cannot take the Windows source path, so it is
# registered only on UNIX hosts (the Linux builds and CI run it).
if(UNIX)
  add_test(NAME ninferno_decision_inputs_test
    COMMAND bash "${PROJECT_SOURCE_DIR}/tools/audit/decision_inputs.sh")
endif()

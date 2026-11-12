find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NInfernoTests.cmake)
# The decision core's host-only tests: the same registrations a decisions-only tree builds
# (tests/decisions/CMakeLists.txt).
include(cmake/DecisionTests.cmake)
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
  ninferno_label_tests(ninferno_ui_moved_update_test LABELS host)
  set_tests_properties(ninferno_ui_moved_update_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ninferno-connect (#339): a stand-in router dialed over loopback (Upgrade, then HTTP/2 with the
# roles reversed; plain and TLS) and a stand-in engine; registration, concurrent streams, a reset
# that closes the engine's connection (also before any byte of the answer), ping timeout, GOAWAY
# with a replacement while the old session drains, a refused key, the router's certificate checked.
# No GPU.
if(TARGET ninferno_connect)
  ninferno_add_test(ninferno_connect_test LABELS host
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_connect.cpp"
    LIBRARIES ninferno_connect PkgConfig::NGHTTP2 OpenSSL::SSL OpenSSL::Crypto)
endif()

# Speculation support (registry, corpus, fingerprints, profiles, watchdog, manifests): no GPU.
ninferno_add_test(ninferno_spec_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec.cpp"
  LIBRARIES ninferno_spec)

# Speculation per measured pair (spec/pairs.h) on the lab's spec-tune measurements: no GPU.
ninferno_add_test(ninferno_spec_pairs_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_pairs.cpp"
  LIBRARIES ninferno_spec)

# A speculation round's cost from facts (spec/round_cost.h) on the registry's pairs: no GPU.
ninferno_add_test(ninferno_spec_round_cost_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_round_cost.cpp"
  LIBRARIES ninferno_spec)
set_tests_properties(ninferno_spec_round_cost_test PROPERTIES LABELS host)

# The rows a restricted draft head keeps (spec/draft_head.h) on the Qwen ranking: no GPU.
ninferno_add_test(ninferno_spec_draft_head_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_draft_head.cpp"
  LIBRARIES ninferno_spec)

# DSpine (spec/dspine.h): the published config's refusals and tensor contract, the arithmetic's
# invariants. No GPU.
ninferno_add_test(ninferno_spec_dspine_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_dspine.cpp"
  LIBRARIES ninferno_spec)

# tools/spec/registry_add.py: the "*" fallback and the load table warnings, no GPU.
add_test(NAME ninferno_registry_add_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_registry_add.py)
ninferno_label_tests(ninferno_registry_add_test LABELS host)

# tools/dev/qc.sh: the dyndep early-stop pattern is a real regex.
add_test(NAME ninferno_qc_sh_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_qc_sh.py)
ninferno_label_tests(ninferno_qc_sh_test LABELS host)

# tools/kit/validate.py: the user measurement kit's rejection rules and verdicts, no GPU.
add_test(NAME ninferno_kit_validate_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_kit_validate.py)
ninferno_label_tests(ninferno_kit_validate_test LABELS host)

# The validation protocol of user measurements (docs/maintainer/validation-protocol.md): the size
# cap, JSON only, known schema versions and keys, the build held to the release manifest, the
# switch-factor and KLD runs, and every file under docs/performance/user-runs/, no GPU.
add_test(NAME ninferno_kit_protocol_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_kit_protocol.py)
ninferno_label_tests(ninferno_kit_protocol_test LABELS host)

# tools/audit/fusion_sites.json against the FUSION-SITE markers in the code: a fusion site missing from
# the registry, or an entry whose site is gone, fails (#617 step A), no GPU.
add_test(NAME ninferno_fusion_sites_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_fusion_sites.py)
set_tests_properties(ninferno_fusion_sites_test PROPERTIES LABELS host)

# The fusion sites' generated manifest table is current, every registry site has a row, and
# tools/bench/site_regret.py validates the regret document against the registry (#617 step H), no GPU.
add_test(NAME ninferno_fusion_site_manifest_py_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_fusion_site_manifest.py)
set_tests_properties(ninferno_fusion_site_manifest_py_test PROPERTIES LABELS host)

# The facts-vs-driver comparison (tools/audit/facts_vs_driver.py) on canned driver output (#612
# step 2), no GPU.
add_test(NAME ninferno_facts_vs_driver_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_facts_vs_driver.py)
set_tests_properties(ninferno_facts_vs_driver_test PROPERTIES LABELS host)

# tools/bench/selection_regret.py: the offline selection regret's statuses, holdouts and exit codes
# (#301), no GPU.
add_test(NAME ninferno_selection_regret_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_selection_regret.py)
ninferno_label_tests(ninferno_selection_regret_test LABELS host)
# tools/bench/regret_collect.py: route sweeps and the production selection into its input, no GPU.
add_test(NAME ninferno_regret_collect_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_regret_collect.py)
ninferno_label_tests(ninferno_regret_collect_test LABELS host)
# tools/bench/policy_moves_predict.py (#614 F1): the A5 ratio of the route policy's moved cells.
add_test(NAME ninferno_policy_moves_predict_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_policy_moves_predict.py)
ninferno_label_tests(ninferno_policy_moves_predict_test LABELS host)

# tools/bench/mmq_width_grid.py (#524): the measurement against a fake server, the grid and A/B readers' refusals, no GPU.
add_test(NAME ninferno_mmq_width_grid_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_mmq_width_grid.py)
ninferno_label_tests(ninferno_mmq_width_grid_test LABELS host)
# tools/kernel_facts/sass_counts.py and the loop columns of kernel_facts.py (#613): the K loop
# found from a small SASS text's control flow, its counts, the snapshot round trip and the check's
# verdicts. No GPU, no CUDA toolkit.
add_test(NAME ninferno_kernel_facts_sass_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_kernel_facts_sass.py)
ninferno_label_tests(ninferno_kernel_facts_sass_test LABELS host)

# src/core/data/isa_facts.json and tools/bench/isa_probe/isa_probe.py (#613 PR-2): the ISA facts
# table's schema (every row Measured, every fact present and positive, card and tool named) and the
# probe output's round trip into a row. No GPU, no CUDA toolkit.
add_test(NAME ninferno_isa_facts_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_isa_facts.py)
set_tests_properties(ninferno_isa_facts_test PROPERTIES LABELS host)

# tools/bench/warp_inflight/warp_inflight.py (Refs #690, #735): the per-SM DRAM bytes-in-flight
# probe's parser and its b_w / P derivation; the probe source and the parser agree on variants,
# grid and columns, and odd, incomplete or busy runs are refused. No GPU, no CUDA toolkit.
add_test(NAME ninferno_warp_inflight_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_warp_inflight.py)
set_tests_properties(ninferno_warp_inflight_test PROPERTIES LABELS host)

# tools/bench/mmq_launch_law.py (#785): the launch law of ggml's matrix kernel from the
# microbench's JSON, a generation's row fitted on two cards with each predicted from the other, and
# a card checked against a core/generation.h row, on synthetic cards. No GPU, no CUDA toolkit.
add_test(NAME ninferno_mmq_launch_law_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_mmq_launch_law.py)
set_tests_properties(ninferno_mmq_launch_law_test PROPERTIES LABELS host)

# tools/bench/gap_map.py: measured times against the counted DRAM and tensor laws (#619), on
# synthetic inputs, no GPU.
add_test(NAME ninferno_gap_map_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_gap_map.py)
set_tests_properties(ninferno_gap_map_test PROPERTIES LABELS host)

# tools/bench/regret_bands.py: regret weighted by T band, per board, format and route family
# (#613), no GPU.
add_test(NAME ninferno_regret_bands_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_regret_bands.py)
set_tests_properties(ninferno_regret_bands_test PROPERTIES LABELS host)

# tools/bench/sweep_manifest.py: the route sweep archive's manifest (#613 PR-0), no GPU.
add_test(NAME ninferno_sweep_manifest_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_sweep_manifest.py)
set_tests_properties(ninferno_sweep_manifest_test PROPERTIES LABELS host)

# tools/audit/diff_lint.py: broken string literals, duplicated added blocks, removed definitions
# still called and removed includes still needed, on planted diffs (#611), no GPU.
add_test(NAME ninferno_diff_lint_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_diff_lint.py)
set_tests_properties(ninferno_diff_lint_test PROPERTIES LABELS host)

# tools/audit/affected_arches.py: a diff to the arches and stands to test, synthetic diffs, no GPU
# (#611).
add_test(NAME ninferno_affected_arches_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_affected_arches.py)
set_tests_properties(ninferno_affected_arches_test PROPERTIES LABELS host)

# routes/*.json: evidence_us / tuned_on provenance (#634), no GPU.
add_test(NAME ninferno_route_table_evidence_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_route_table_evidence.py)
ninferno_label_tests(ninferno_route_table_evidence_test LABELS host)

# tools/bench/mmq_transfer_audit.py: same-generation MMQ transfer by workload regime (#373), no GPU.
add_test(NAME ninferno_mmq_transfer_audit_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_mmq_transfer_audit.py)
ninferno_label_tests(ninferno_mmq_transfer_audit_test LABELS host)

# src/spec/round_cost.h kSmallTEta against tools/fits/round_cost/fit_eta.py's fit of the shipped
# registry, no GPU.
add_test(NAME ninferno_round_cost_eta_fit_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_round_cost_eta_fit.py)
set_tests_properties(ninferno_round_cost_eta_fit_test PROPERTIES LABELS host)

# tools/audit/board_literal_lint.py: no hand-written `ninferno::Board` literal or bare five-number
# board tuple outside the registry (roadmap #612 step 9), and the tree is clean. What it does not
# see is listed in the script's docstring. No GPU.
add_test(NAME ninferno_board_literal_lint_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_board_literal_lint.py)
set_tests_properties(ninferno_board_literal_lint_test PROPERTIES LABELS host)

# tools/audit/golden_diff.py: the compact summary of a selector golden change (#620), no GPU.
add_test(NAME ninferno_golden_diff_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_golden_diff.py)
set_tests_properties(ninferno_golden_diff_test PROPERTIES LABELS host)

# tools/audit/golden_gate.py: the CI gate on decision-path diffs without a golden change (#620), no GPU.
add_test(NAME ninferno_golden_gate_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_golden_gate.py)
set_tests_properties(ninferno_golden_gate_test PROPERTIES LABELS host)

# Repository hygiene, no GPU: only scripts (a #! first line) may be committed executable.
add_test(NAME ninferno_file_modes_test
  COMMAND bash "${PROJECT_SOURCE_DIR}/tools/check_file_modes.sh")
ninferno_label_tests(ninferno_file_modes_test LABELS host)
set_tests_properties(ninferno_file_modes_test PROPERTIES SKIP_RETURN_CODE 77)

# Determinism, no GPU: decision code reads no clock, event timer or NVML (the monitor is
# write-only; docs/maintainer/deterministic-decisions.md, principle 3). A POSIX shell check: on
# native Windows `bash` may be WSL's, which cannot take the Windows source path, so it is
# registered only on UNIX hosts (the Linux builds run it; no CI workflow does).
if(UNIX)
  add_test(NAME ninferno_decision_inputs_test
    COMMAND bash "${PROJECT_SOURCE_DIR}/tools/audit/decision_inputs.sh")
  ninferno_label_tests(ninferno_decision_inputs_test LABELS host)
endif()

# Every test carries a label (host, gpu, real, golden, slow). Reads the build's own test list; needs
# python3, ctest and a POSIX shell, so it is registered on UNIX hosts only. It runs with a Linux
# build's ctest; no CI workflow configures a build, so CI does not run it. The build type goes
# along (a multi-config generator lists no test without ctest -C <config>).
if(UNIX)
  add_test(NAME ninferno_ctest_labels_test
    COMMAND bash "${PROJECT_SOURCE_DIR}/tools/audit/ctest_labels.sh" "${PROJECT_BINARY_DIR}" "$<CONFIG>")
  set_tests_properties(ninferno_ctest_labels_test PROPERTIES
    ENVIRONMENT "CTEST=${CMAKE_CTEST_COMMAND};PYTHON=${Python3_EXECUTABLE}")
  ninferno_label_tests(ninferno_ctest_labels_test LABELS host)
endif()

# tools/bench/route_table_prune.py: the exit tooling for measured route overrides (#614), on
# synthetic regret reports, and the tables' evidence_us check (gaps listed in
# route_evidence_gaps.json, which may only shrink), no GPU.
add_test(NAME ninferno_route_table_prune_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/test_route_table_prune.py)
ninferno_label_tests(ninferno_route_table_prune_test LABELS host)

# The shared test tree's link modes on the built binaries (NINFERNO_SHARED_OP_TESTS, a Linux-only
# option): every test, app and bench as the configure decided, no second copy of a member archive
# or of the CUDA runtime beside the shared library. No GPU.
if(NINFERNO_SHARED_OP_TESTS)
  add_test(NAME ninferno_link_mode_test
    COMMAND bash "${PROJECT_SOURCE_DIR}/tools/audit/link_mode.sh" "${CMAKE_BINARY_DIR}")
  set_tests_properties(ninferno_link_mode_test PROPERTIES LABELS host)
endif()

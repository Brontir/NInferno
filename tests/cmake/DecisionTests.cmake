# The decision core's host-only tests (#620 step 13): the selector, the price and the choice built
# with no CUDA runtime in their link (each a host-only closure, ninferno_assert_host_only checks it
# at configure), run with no GPU. Included by every test tree (tests/CMakeLists.txt) and by a
# decisions-only tree (NINFERNO_DECISIONS_ONLY, tests/decisions/CMakeLists.txt), where they are the
# whole suite: `ctest -L host` there runs exactly these. The decisions workflow
# (.github/workflows/decisions.yml) builds them in that tree, one per golden build set.

# The offline GPU catalogue audit: PCI subsystem identities survive parsing, non-GPU PCI
# functions are excluded, and marketing series cannot silently choose an architecture.
add_test(NAME ninferno_gpu_catalog_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/../test_gpu_catalog.py)
set_tests_properties(ninferno_gpu_catalog_test PROPERTIES LABELS host)

# The selector golden's host twin: the same dump code as ninferno_selector_golden_test
# (tests/ops/linear/tests.cmake), built and linked with no CUDA runtime (ninferno_ops_host): the
# committed goldens must read the same from a host-only selector (#620). The stubs are the device
# seams (current_device_class, the device ordinal, the dense tuner, which never times anything on a
# host with no device, and the description of an executed kernel, which a host with no device has
# none of). The Volta CUTLASS route's sizes are host code (generic_volta_cutlass_sizes.cpp, in
# ninferno_ops_host), so an sm_70 tree built with CUTLASS answers here as its device build does:
# every tree registers the twin (#620 step 13).
ninferno_add_test(ninferno_selector_golden_host_test LABELS host golden slow
  SOURCES "${PROJECT_SOURCE_DIR}/tests/ops/linear/test_selector_golden.cpp"
  LIBRARIES ninferno_ops_host)
target_sources(ninferno_selector_golden_host_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/core/device_class_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/core/per_device_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/ops/generic/generic_nodevice.cpp)
target_compile_definitions(ninferno_selector_golden_host_test PRIVATE
  NINFERNO_HOST_ONLY
  NINFERNO_SELECTOR_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/ops/linear/selector_golden")
cmake_language(DEFER CALL ninferno_assert_host_only ninferno_selector_golden_host_test)
# It dumps every board twice over (once a process per board, then all in one): minutes of CPU.
set_tests_properties(ninferno_selector_golden_host_test PROPERTIES TIMEOUT 1800)

# The route policy (#614 F1, --route-policy): on the A5 regret sweep's boards and shapes, legacy is
# unchanged, a borrowed or transferred route-table answer yields to the ranker under evidence, and
# an answer of any other evidence is never demoted. `--write FILE` writes the moved decisions
# (tools/bench/data/route_regret/policy_evidence_moves.json; each board from its own build set,
# the others kept). Host only, with the golden twin's link.
ninferno_add_test(ninferno_route_policy_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${PROJECT_SOURCE_DIR}/tests/ops/linear/test_route_policy.cpp"
  LIBRARIES ninferno_ops_host)
target_sources(ninferno_route_policy_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/core/device_class_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/core/per_device_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/ops/generic/generic_nodevice.cpp)
target_link_libraries(ninferno_route_policy_test PRIVATE ninferno::json)
target_compile_definitions(ninferno_route_policy_test PRIVATE NINFERNO_HOST_ONLY)
cmake_language(DEFER CALL ninferno_assert_host_only ninferno_route_policy_test)

# The host part of the generic selector (ops/generic/generic_routes.cpp, kernel_facts.cpp and the
# rest of ninferno_ops_host, with the device seams of generic_nodevice.cpp): route names, the
# register-MMA variant rules, the kernel-facts snapshots; no CUDA runtime, no GPU (#620 step 11).
# Linked through ninferno_decisions (src/CMakeLists.txt), which brings those stubs and
# NINFERNO_HOST_ONLY.
ninferno_add_test(ninferno_generic_routes_host_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_generic_routes_host.cpp"
  LIBRARIES ninferno_decisions)
cmake_language(DEFER CALL ninferno_assert_host_only ninferno_generic_routes_host_test)

# The configuration choice: the speed model reproduces the stand measurements its constants were
# fitted on (tests/fixtures/choice/speed_measured.json), and the rule picks, keeps the pins and
# explains. CPU.
ninferno_add_test(ninferno_choice_test LABELS host golden slow STATIC_LINK
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_choice.cpp"
  LIBRARIES ninferno_product_choice ninferno_core_host ninferno::json)
target_include_directories(ninferno_choice_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
# Host-only: no CUDA runtime in its link closure (#620 step 11), so the static archive of
# ninferno_core_host in the shared test tree too (STATIC_LINK); the device of a build without one
# is the stub that throws (src/core/device_class_nodevice.cpp).
target_sources(ninferno_choice_test PRIVATE ${PROJECT_SOURCE_DIR}/src/core/device_class_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/core/per_device_nodevice.cpp)
cmake_language(DEFER CALL ninferno_assert_host_only ninferno_choice_test)
target_compile_definitions(ninferno_choice_test PRIVATE
  NINFERNO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# The MMQ prefill gate's decision atlas over the architecture table (#373, core/gate_atlas.h);
# `--json` prints it. No GPU.
ninferno_add_test(ninferno_gate_atlas_test LABELS host STATIC_LINK
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gate_atlas.cpp"
  LIBRARIES ninferno_core_host ninferno::json)
# Host-only: no CUDA runtime in its link closure (#620 step 11), so the static archive of
# ninferno_core_host in the shared test tree too (STATIC_LINK).
target_sources(ninferno_gate_atlas_test PRIVATE ${PROJECT_SOURCE_DIR}/src/core/device_class_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/core/per_device_nodevice.cpp)
cmake_language(DEFER CALL ninferno_assert_host_only ninferno_gate_atlas_test)

# The route cost model's constants, each tagged fact / counted / fitted
# (ops/generic/route_cost_constants.h); `--list` prints them. The Counted ones are held to the
# kernel families' records and the architecture table (core/arch_registry.h: the generated rows
# of ninferno_core_host's include directory). No GPU.
ninferno_add_test(ninferno_route_cost_constants_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_route_cost_constants.cpp")
target_include_directories(ninferno_route_cost_constants_test PRIVATE
  $<TARGET_PROPERTY:ninferno_core_host,INTERFACE_INCLUDE_DIRECTORIES>)
set_tests_properties(ninferno_route_cost_constants_test PROPERTIES LABELS host)

# The holdout records of the fitted constants it covers (core/fitted_holdouts.h, the one mechanism;
# the test's header says which): the tagged constants from the compiled tables, the listed files'
# definitions, and the untagged constants a sweep found; an Unset record or a wrong total fails.
# No GPU.
ninferno_add_test(ninferno_fitted_holdouts_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_fitted_holdouts.cpp")

# The generic path's literal rules (ops/generic/route_rules.h: boards, cards, tolerance), held to
# the sources they describe; `--list` prints them. No GPU.
ninferno_add_test(ninferno_route_rules_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_route_rules.cpp")

# The measured per-format rules (ops/generic/format_rules.h, #615 F6): each rule's rows name two
# cards or are tagged single_card / carried, the thresholds equal the constants and their
# route_rules.h descriptors; `--list` prints the table. No GPU.
ninferno_add_test(ninferno_format_rules_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_format_rules.cpp")

# The register-MMA kernel's static shared memory as constexpr geometry (ops/generic/rmma_geometry.h,
# static_asserted by the kernel, with the decoder rows pinned to their decoders) equals every row of
# every kernel-facts snapshot (#612 step 5); a plain host test: header only, no GPU, no library.
ninferno_add_test(ninferno_rmma_geometry_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${PROJECT_SOURCE_DIR}/tests/ops/linear/test_rmma_geometry.cpp")
# The full build's architectures (the NVFP4 kernels, so the cluster hand-over): the one list of
# CMakeLists.txt, comma-joined, not parsed from its text.
target_compile_definitions(ninferno_rmma_geometry_test PRIVATE
  NINFERNO_FULL_CUDA_ARCHS="$<JOIN:${NINFERNO_FULL_CUDA_ARCHS},$<COMMA>>")

# Which board a CUDA ordinal names (core/device_identity.h): the default GPU is the same under
# every enumeration order of the same boards. No GPU.
ninferno_add_test(ninferno_device_identity_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device_identity.cpp")

# The format facts table (core/format_facts.h) against the switches and spelling tables it
# replaced, for every QType value. No GPU.
ninferno_add_test(ninferno_format_facts_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_format_facts.cpp")

# The format predicates' arch requirements (core/arch_requires.h, ops/generic/format_requires.h;
# #615 F5): each resolves from the board registry row of every board of gpu_facts.json and answers
# as the code it replaced, under every build; the selector reads them under each board as the
# substituted device. Host only, with the golden twin's link. No GPU.
ninferno_add_test(ninferno_arch_requires_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arch_requires.cpp"
  LIBRARIES ninferno_ops_host)
target_sources(ninferno_arch_requires_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/core/device_class_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/core/per_device_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/ops/generic/generic_nodevice.cpp)
target_compile_definitions(ninferno_arch_requires_test PRIVATE NINFERNO_HOST_ONLY)
cmake_language(DEFER CALL ninferno_assert_host_only ninferno_arch_requires_test)

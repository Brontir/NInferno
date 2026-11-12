add_library(ninferno_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
ninferno_test_includes(ninferno_linear_test_support)
ninferno_op_oracle_options(ninferno_linear_test_support)
target_link_libraries(ninferno_linear_test_support PUBLIC ninferno::ops)

ninferno_add_op_test(ninferno_linear_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninferno_linear_test_support)

ninferno_add_op_test(ninferno_linear_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninferno_linear_test_support)

ninferno_add_op_test(ninferno_linear_q6_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES ninferno_linear_test_support)

ninferno_add_op_test(ninferno_linear_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninferno_linear_test_support)

# NVFP4 kernels ship only in the 120a build.
if(NINFERNO_WITH_NVFP4)
  ninferno_add_op_test(ninferno_linear_nvfp4_a16_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
    LIBRARIES ninferno_linear_test_support)

  ninferno_add_op_test(ninferno_linear_nvfp4_a4_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
    LIBRARIES ninferno_linear_test_support)
endif()

ninferno_add_op_test(ninferno_linear_fp8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
  LIBRARIES ninferno_linear_test_support)

ninferno_add_op_test(ninferno_linear_fp8_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
  LIBRARIES ninferno_linear_test_support)

ninferno_add_op_test(ninferno_linear_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninferno_ops)

# Gemma 4 26B-A4B / 31B projection shapes, one executable per weight format (NVFP4 in the 120a
# build only).
set(NINFERNO_GEMMA4_LINEAR_FORMATS Q8 Q6)
if(NINFERNO_WITH_NVFP4)
  list(PREPEND NINFERNO_GEMMA4_LINEAR_FORMATS NVFP4)
endif()
foreach(format IN LISTS NINFERNO_GEMMA4_LINEAR_FORMATS)
  string(TOLOWER "${format}" format_name)
  ninferno_add_op_test(ninferno_linear_gemma4_${format_name}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_gemma4_shapes.cpp"
    LIBRARIES ninferno_linear_test_support)
  target_compile_definitions(ninferno_linear_gemma4_${format_name}_test PRIVATE
    NINFERNO_GEMMA4_FORMAT_${format})
endforeach()

# A Gemma 4 projection's column does not depend on the call's token count.
ninferno_add_op_test(ninferno_linear_gemma4_token_invariance_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_gemma4_token_invariance.cpp"
  LIBRARIES ninferno_linear_test_support)

# A Qwen 3.x prefill op's column does not depend on the call's token count or start.
ninferno_add_op_test(ninferno_qwen_prefill_token_invariance_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qwen_prefill_token_invariance.cpp"
  LIBRARIES ninferno_linear_test_support)

# Route tables of the generic projection path: lookup, prefill route, refusals.
ninferno_add_op_test(ninferno_route_table_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_table.cpp"
  LIBRARIES ninferno_linear_test_support)
# Every format and activation path over the scale domain (tensor scale 3e-4..2.5, block
# scales across 2^16 with E4M3 and FP16 subnormals, N and K off the 512 grid, M = 1, 3, 127, 129).
ninferno_add_op_test(ninferno_linear_scale_domain_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_scale_domain.cpp"
  LIBRARIES ninferno_linear_test_support)

add_test(NAME ninferno_route_table_malformed_default_test
  COMMAND ninferno_route_table_test --malformed-default)
ninferno_label_tests(ninferno_route_table_malformed_default_test LABELS gpu)
set_tests_properties(ninferno_route_table_malformed_default_test PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME ninferno_route_table_similar_default_test
  COMMAND ninferno_route_table_test --similar-default)
ninferno_label_tests(ninferno_route_table_similar_default_test LABELS gpu)
set_tests_properties(ninferno_route_table_similar_default_test PROPERTIES SKIP_RETURN_CODE 77)
# Every shipped table's gpu.pci_id is the board of its class, read on that board (substituted, no
# GPU): the table is its own, at distance 0 but for a recorded conflict (#612 step 9).
add_test(NAME ninferno_route_table_shipped_boards_test
  COMMAND ninferno_route_table_test --shipped-boards)
set_tests_properties(ninferno_route_table_shipped_boards_test PROPERTIES LABELS host)

# The generic projection path: random and awkward shapes in every format, every op and route,
# against an FP64 reference GEMM, plus the token invariance of the prefill-admissible routes.
ninferno_add_op_test(ninferno_linear_generic_test LABELS gpu slow
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_generic_linear.cpp"
  LIBRARIES ninferno_linear_test_support)

# A generic launch's overlap with its neighbours (#309): join, release and prefetch with the
# reason each is refused, as run_route applies them (host only).
ninferno_add_op_test(ninferno_launch_overlap_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_launch_overlap.cpp"
  LIBRARIES ninferno_linear_test_support)

# The register MMA's pre-summed activation group plane (#619): which calls take it, its workspace
# and demotion (host), the pre-pass bit for bit against the CPU and the instance the price
# describes (GPU; skipped without one).
ninferno_add_op_test(ninferno_group_sums_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_group_sums.cpp"
  LIBRARIES ninferno_linear_test_support)
set_tests_properties(ninferno_group_sums_test PROPERTIES SKIP_RETURN_CODE 77)

# Scratch read lifetimes of generic launches (#309): which launch reads or writes which workspace
# region until when, and that no launch starting under its predecessor takes a region it still
# uses (host only).
ninferno_add_op_test(ninferno_scratch_lifetime_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_scratch_lifetime.cpp"
  LIBRARIES ninferno_linear_test_support)

# A route's tile and logical traffic (#298): the schedules' declared geometry (host only).
ninferno_add_op_test(ninferno_route_traffic_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_traffic.cpp"
  LIBRARIES ninferno_linear_test_support)

# A slice of a fused weight and a row view are priced, described and routed as a standalone weight
# of their own shape on every golden board, the R8 verify gate included (#640, #704, host only).
ninferno_add_op_test(ninferno_slice_price_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_slice_price.cpp"
  LIBRARIES ninferno_linear_test_support)

# A route's description (#298): the fields agree with the functions the launch and selector ask.
ninferno_add_op_test(ninferno_route_description_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_description.cpp"
  LIBRARIES ninferno_linear_test_support)

set_tests_properties(ninferno_route_description_test PROPERTIES SKIP_RETURN_CODE 77)

# The same on a substituted GV100: a tiled call of a CUTLASS-only format without room for the
# decoded copy runs scalar and the description says so (#653).
ninferno_add_op_test(ninferno_route_description_volta_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_description_volta.cpp"
  LIBRARIES ninferno_linear_test_support)

set_tests_properties(ninferno_route_description_volta_test PROPERTIES SKIP_RETURN_CODE 77)

# Route selection on the GPU classes this machine lacks (substituted DeviceClass, one process
# per board): every choice admitted, no feature the board lacks, its own route table.
ninferno_add_op_test(ninferno_route_classes_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_classes.cpp"
  LIBRARIES ninferno_linear_test_support)

# The generic path's coverage lint (#619 K2): on the golden boards of this build's architecture,
# every served format, activation mode and width band has a family on the device's own
# instructions, not only the scalar kernel or an emulated MMA; the known holes are listed in the
# test with their reasons. The served modes are convert's kept_policy / head_policy under each
# --activations mode (ninferno_convert). It reads no GPU's timing, but needs one visible: ggml's
# matrix kernel reads the device's properties when it describes a route, as in the selector golden
# test (without a device: cudaErrorNoDevice; host again once #763 is fixed). Skips (77) on an
# architecture the release does not ship.
ninferno_add_op_test(ninferno_coverage_report_test LABELS gpu
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_coverage_report.cpp"
  LIBRARIES ninferno_linear_test_support ninferno_convert)
target_compile_definitions(ninferno_coverage_report_test PRIVATE
  NINFERNO_RELEASE_CUDA_ARCHS="$<JOIN:${NINFERNO_CUDA_ARCHS},$<COMMA>>")

# The format traits table against the predicates it replaced, for every QType.
ninferno_add_op_test(ninferno_format_traits_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_format_traits.cpp"
  LIBRARIES ninferno_linear_test_support)

# The codec traits rows (ops/generic/codec_traits.h, #615 F3; generic_codec_traits.cu proves them
# against the codec structs) against the literals r8_fits, w4a8_fits and mmq_plane_bytes carried
# before, for every QType; no GPU needed.
ninferno_add_op_test(ninferno_codec_traits_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_codec_traits.cpp"
  LIBRARIES ninferno_linear_test_support)
set_tests_properties(ninferno_codec_traits_test PROPERTIES LABELS host)

# The kernel roofline (#613 PR-4): price_routes prices equal the golden table of the formula
# it replaced (fixed outputs from before the move), and
# the selector's priced routes are the roofline's terms (host only).
ninferno_add_op_test(ninferno_kernel_roofline_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernel_roofline.cpp"
  LIBRARIES ninferno_linear_test_support)

# The generator of that golden table (tests/ops/linear/gen_kernel_roofline_golden.cpp): built so that
# it keeps compiling, not run by ctest (it is run once, against the code before the move).
ninferno_add_test(ninferno_gen_kernel_roofline_golden
  NO_DEFAULT_TEST
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/gen_kernel_roofline_golden.cpp"
  LIBRARIES ninferno_linear_test_support)

# ops::linear's candidates per call (#306): the owned Q8 kernel's launch and the generic route,
# one selected as dispatch_linear runs.
# The activation each generic route prepares and which preparations serve one another (#308).
ninferno_add_op_test(ninferno_prepared_rules_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepared_rules.cpp"
  LIBRARIES ninferno_linear_test_support)

ninferno_add_op_test(ninferno_linear_candidates_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_candidates.cpp"
  LIBRARIES ninferno_linear_test_support)

# The generic path's choices on thirteen substituted boards against selector_golden/<board>.txt:
# a selector change shows line by line; a refactor must leave every file as it was.
ninferno_add_op_test(ninferno_selector_golden_test LABELS gpu golden slow
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_selector_golden.cpp"
  LIBRARIES ninferno_linear_test_support)
target_compile_definitions(ninferno_selector_golden_test PRIVATE
  NINFERNO_SELECTOR_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/selector_golden")
# It dumps every board twice over (once a process per board, then all in one): minutes of CPU.
set_tests_properties(ninferno_selector_golden_test PROPERTIES TIMEOUT 1800)

# The selector's register-MMA kernel description (rmma_resources: facts, no CUDA) equals the launch's
# own (rmma_resources_by_launch, the real launch run in describe mode), per format, schedule,
# epilogue and width on every board with facts (#620). No GPU.
ninferno_add_op_test(ninferno_rmma_describe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmma_describe.cpp"
  LIBRARIES ninferno_linear_test_support)

# The selector on the real device with no substituted board: select_route and w4a8_ws_extra return,
# and the R8 decode table of a board that has one is found (#620). Skips (77) without a GPU.
ninferno_add_op_test(ninferno_selector_real_device_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_selector_real_device.cpp"
  LIBRARIES ninferno_linear_test_support)

# The register-MMA kernel's static shared memory as constexpr geometry (ops/generic/rmma_geometry.h,
# static_asserted by the kernel, with the decoder rows pinned to their decoders) equals every row of
# every kernel-facts snapshot (#612 step 5); a plain host test: header only, no GPU, no library.
ninferno_add_test(ninferno_rmma_geometry_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmma_geometry.cpp")
# The full build's architectures (the NVFP4 kernels, so the cluster hand-over): the one list of
# CMakeLists.txt, comma-joined, not parsed from its text.
target_compile_definitions(ninferno_rmma_geometry_test PRIVATE
  NINFERNO_FULL_CUDA_ARCHS="$<JOIN:${NINFERNO_FULL_CUDA_ARCHS},$<COMMA>>")

# The kernel-facts snapshots the register-MMA price reads (ops/generic/kernel_facts.h, #612).
# Three checks, from the data inwards:
# - ninferno_kernel_facts_test: this build's register-MMA kernels, read off the linked binary with
#   cuobjdump (tools/kernel_facts/kernel_facts.py), equal the committed snapshot of its
#   architecture; no GPU. Skips (77) where this nvcc is not the snapshot's (ptxas then allocates
#   other registers by construction; the price reads the snapshot either way). A mismatch names
#   the kernels and the command that regenerates the file. Where the snapshot carries loop counts
#   and the nvcc is its own, the check disassembles every register-MMA kernel (about 2400), which
#   takes minutes (560 s in one sm_120a build); a snapshot without loop counts (schema 1: sm_70
#   alone, waived by name in ninferno_kernel_facts_lookup_test, which fails on any other) is
#   compared in seconds, and another nvcc skips. The test checks this build's architecture alone,
#   so a snapshot no stand builds (sm_80, sm_90, sm_100) is checked only by the release's
#   package.sh (#758). The label is slow in every build, so `ctest -L host -LE slow` leaves it out
#   even where it would be quick: a label that depended on the snapshot would make the fast set
#   differ between builds.
# - ninferno_kernel_facts_lookup_test: every snapshot parses and carries the loop counts (schema
#   4; sm_70 waived until #762), the release engine of each lab capability is found, and every facts-priced golden board's
#   price reads facts; no GPU.
# - ninferno_kernel_facts_live_test: on a GPU that runs this build's SASS, the running kernels'
#   attributes and the driver's occupancy equal the snapshot and the pure occupancy.
get_filename_component(ninferno_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
if(NINFERNO_SHARED_OP_TESTS)
  set(ninferno_kernel_facts_binary $<TARGET_FILE:ninferno_ops_shared>)
else()
  set(ninferno_kernel_facts_binary $<TARGET_FILE:ninferno_selector_golden_test>)
endif()
set(ninferno_kernel_facts_snapshot
  "${PROJECT_SOURCE_DIR}/src/ops/generic/data/kernel_facts/sm_${NINFERNO_BUILD_ARCH_LABEL}.txt")
string(TOUPPER "${CMAKE_BUILD_TYPE}" ninferno_build_type_upper)
if(NINFERNO_WITH_NVFP4)
  set(ninferno_kernel_facts_rdc OFF)
else()
  set(ninferno_kernel_facts_rdc ON)
endif()
string(STRIP
  "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${ninferno_build_type_upper}} rdc=${ninferno_kernel_facts_rdc} lineinfo=${NINFERNO_CUDA_LINEINFO} ptx=${NINFERNO_CUDA_EMBED_PTX}"
  ninferno_kernel_facts_flags)
add_test(NAME ninferno_kernel_facts_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/kernel_facts/kernel_facts.py check
    --binary ${ninferno_kernel_facts_binary} --expected ${ninferno_kernel_facts_snapshot}
    --nvcc ${CMAKE_CUDA_COMPILER_VERSION} --cuda-bin ${ninferno_cuda_bin})
ninferno_label_tests(ninferno_kernel_facts_test LABELS host slow)
set_tests_properties(ninferno_kernel_facts_test PROPERTIES SKIP_RETURN_CODE 77)
# `ninja ninferno_kernel_facts_update`: the snapshot of this build's architecture rewritten from
# its binary into the source tree (then regenerate the selector goldens that move with it).
add_custom_target(ninferno_kernel_facts_update
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/kernel_facts/kernel_facts.py extract
    --binary ${ninferno_kernel_facts_binary} --arch ${NINFERNO_BUILD_ARCH_LABEL}
    --nvcc ${CMAKE_CUDA_COMPILER_VERSION} --cuda-bin ${ninferno_cuda_bin}
    --flags "${ninferno_kernel_facts_flags}" --source-root ${PROJECT_SOURCE_DIR}
    -o ${ninferno_kernel_facts_snapshot}
  DEPENDS ninferno_selector_golden_test
  VERBATIM)

ninferno_add_op_test(ninferno_kernel_facts_lookup_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernel_facts.cpp"
  LIBRARIES ninferno_linear_test_support)

ninferno_add_op_test(ninferno_kernel_facts_live_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernel_facts_live.cpp"
  LIBRARIES ninferno_linear_test_support)
target_compile_definitions(ninferno_kernel_facts_live_test PRIVATE
  NINFERNO_NVCC_VERSION="${CMAKE_CUDA_COMPILER_VERSION}")

# The selector's named authorities (#614): the lists are well formed, on every facts board each
# answer's evidence is consistent with its kind, and known cases name the authority that answers.
ninferno_add_op_test(ninferno_route_authorities_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_authorities.cpp"
  LIBRARIES ninferno_linear_test_support)

# The route catalogue (#614): for every (family, schedule) the summation order, token invariance,
# activation class and workspace need agree, profile_admits says what they say, and every
# routes/*.json passes the filter. No GPU.
ninferno_add_op_test(ninferno_route_catalogue_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_catalogue.cpp"
  LIBRARIES ninferno_linear_test_support ninferno::json)

# The generic path's dense tuner: tuned at the first eager call, within the activation class, R8
# decode bytes kept, the cache file read back, nothing timed inside a graph capture.
ninferno_add_op_test(ninferno_dense_tune_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dense_tune.cpp"
  LIBRARIES ninferno_linear_test_support)

# The SwiGLU / GeGLU of a gate and an up weight of one or two formats (segmented_swiglu).
ninferno_add_op_test(ninferno_segmented_swiglu_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_segmented_swiglu.cpp"
  LIBRARIES ninferno_linear_test_support)

# The bounded twin of ggml's 4-column vector kernel: bytes equal the plain instance's, and the launch's
# choice (twin only when it saves a round and does not spill) read from its description.
ninferno_add_op_test(ninferno_ggml_vec_twin_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ggml_vec_twin.cpp"
  LIBRARIES ninferno_linear_test_support ninferno_ggml_mmq ninferno_core)

# Side-by-side q/k/v and gate projections (segmented, joined, spread) equal the separate ones.
ninferno_add_op_test(ninferno_side_by_side_projection_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_side_by_side_projection.cpp"
  LIBRARIES ninferno_linear_test_support)

# The generic path under a route pin (--prefill-route, --decode-route) on substituted boards: the
# pinned family takes every call it can compute, every other call keeps its automatic route, the
# registered shapes' hand-off gates follow, and clearing the pins restores every route.
ninferno_add_op_test(ninferno_route_pin_selector_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_pin_selector.cpp"
  LIBRARIES ninferno_linear_test_support)

# The same dump code as ninferno_selector_golden_test, built and linked with no CUDA runtime
# (ninferno_ops_host): the committed goldens must read the same from a host-only selector (#620).
# The stubs are the device seams (current_device_class, the device ordinal, the dense tuner, which
# never times anything on a host with no device, and the description of an executed kernel, which a
# host with no device has none of). The Volta CUTLASS route's sizes are host code
# (generic_volta_cutlass_sizes.cpp, in ninferno_ops_host), so an sm_70 tree built with CUTLASS
# answers here as its device build does: every tree registers the twin (#620 step 13).
ninferno_add_test(ninferno_selector_golden_host_test LABELS host golden slow
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_selector_golden.cpp"
  LIBRARIES ninferno_ops_host)
target_sources(ninferno_selector_golden_host_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/core/device_class_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/core/per_device_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/ops/generic/generic_nodevice.cpp)
target_compile_definitions(ninferno_selector_golden_host_test PRIVATE
  NINFERNO_HOST_ONLY
  NINFERNO_SELECTOR_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/selector_golden")
cmake_language(DEFER CALL ninferno_assert_host_only ninferno_selector_golden_host_test)
# It dumps every board twice over (once a process per board, then all in one): minutes of CPU.
set_tests_properties(ninferno_selector_golden_host_test PROPERTIES TIMEOUT 1800)

# The route policy (#614 F1, --route-policy): on the A5 regret sweep's boards and shapes, legacy is
# unchanged, a borrowed or transferred route-table answer yields to the ranker under evidence, and
# an answer of any other evidence is never demoted. `--write FILE` writes the moved decisions
# (tools/bench/data/route_regret/policy_evidence_moves.json). Host only, as the golden twin.
ninferno_add_test(ninferno_route_policy_test LABELS host NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_policy.cpp"
  LIBRARIES ninferno_ops_host)
target_sources(ninferno_route_policy_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/core/device_class_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/core/per_device_nodevice.cpp
  ${PROJECT_SOURCE_DIR}/src/ops/generic/generic_nodevice.cpp)
target_compile_definitions(ninferno_route_policy_test PRIVATE NINFERNO_HOST_ONLY)
cmake_language(DEFER CALL ninferno_assert_host_only ninferno_route_policy_test)

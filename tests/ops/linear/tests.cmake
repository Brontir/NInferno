add_library(ninferno_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
ninferno_test_includes(ninferno_linear_test_support)
ninferno_op_oracle_options(ninferno_linear_test_support)
ninferno_link_ops(ninferno_linear_test_support PUBLIC)

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
set_tests_properties(ninferno_route_table_malformed_default_test PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME ninferno_route_table_similar_default_test
  COMMAND ninferno_route_table_test --similar-default)
set_tests_properties(ninferno_route_table_similar_default_test PROPERTIES SKIP_RETURN_CODE 77)

# The generic projection path: random and awkward shapes in every format, every op and route,
# against an FP64 reference GEMM, plus the token invariance of the prefill-admissible routes.
ninferno_add_op_test(ninferno_linear_generic_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_generic_linear.cpp"
  LIBRARIES ninferno_linear_test_support)

# A generic launch's overlap with its neighbours (#309): join, release and prefetch with the
# reason each is refused, as run_route applies them (host only).
ninferno_add_op_test(ninferno_launch_overlap_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_launch_overlap.cpp"
  LIBRARIES ninferno_linear_test_support)

# Scratch read lifetimes of generic launches (#309): which launch reads or writes which workspace
# region until when, and that no launch starting under its predecessor takes a region it still
# uses (host only).
ninferno_add_op_test(ninferno_scratch_lifetime_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_scratch_lifetime.cpp"
  LIBRARIES ninferno_linear_test_support)

# A route's tile and logical traffic (#298): the schedules' declared geometry (host only).
ninferno_add_op_test(ninferno_route_traffic_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_traffic.cpp"
  LIBRARIES ninferno_linear_test_support)

# Route selection on the GPU classes this machine lacks (substituted DeviceClass, one process
# per board): every choice admitted, no feature the board lacks, its own route table.
ninferno_add_op_test(ninferno_route_classes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_classes.cpp"
  LIBRARIES ninferno_linear_test_support)

# The format traits table against the predicates it replaced, for every QType.
ninferno_add_op_test(ninferno_format_traits_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_format_traits.cpp"
  LIBRARIES ninferno_linear_test_support)

# ops::linear's candidates per call (#306): the owned Q8 kernel's launch and the generic route,
# one selected as dispatch_linear runs.
# The activation each generic route prepares and which preparations serve one another (#308).
ninferno_add_op_test(ninferno_prepared_rules_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepared_rules.cpp"
  LIBRARIES ninferno_linear_test_support)

ninferno_add_op_test(ninferno_linear_candidates_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_candidates.cpp"
  LIBRARIES ninferno_linear_test_support)

# The generic path's choices on thirteen substituted boards against selector_golden/<board>.txt:
# a selector change shows line by line; a refactor must leave every file as it was.
ninferno_add_op_test(ninferno_selector_golden_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_selector_golden.cpp"
  LIBRARIES ninferno_linear_test_support)
target_compile_definitions(ninferno_selector_golden_test PRIVATE
  NINFERNO_SELECTOR_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/selector_golden")

# The register-MMA kernel's static shared memory as constexpr geometry (ops/generic/rmma_geometry.h,
# static_asserted by the kernel, with the decoder rows pinned to their decoders) equals every row of
# every kernel-facts snapshot (#612 step 5); a plain host test: header only, no GPU, no library.
ninferno_add_test(ninferno_rmma_geometry_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmma_geometry.cpp")
set_tests_properties(ninferno_rmma_geometry_test PROPERTIES LABELS host)

# The kernel-facts snapshots the register-MMA price reads (ops/generic/kernel_facts.h, #612).
# Three checks, from the data inwards:
# - ninferno_kernel_facts_test: this build's register-MMA kernels, read off the linked binary with
#   cuobjdump (tools/kernel_facts/kernel_facts.py), equal the committed snapshot of its
#   architecture; no GPU. Skips (77) where this nvcc is not the snapshot's (ptxas then allocates
#   other registers by construction; the price reads the snapshot either way). A mismatch names
#   the kernels and the command that regenerates the file.
# - ninferno_kernel_facts_lookup_test: every snapshot parses, the release engine of each lab
#   capability is found, and every facts-priced golden board's price reads facts; no GPU.
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

ninferno_add_op_test(ninferno_kernel_facts_lookup_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernel_facts.cpp"
  LIBRARIES ninferno_linear_test_support)

ninferno_add_op_test(ninferno_kernel_facts_live_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernel_facts_live.cpp"
  LIBRARIES ninferno_linear_test_support)
target_compile_definitions(ninferno_kernel_facts_live_test PRIVATE
  NINFERNO_NVCC_VERSION="${CMAKE_CUDA_COMPILER_VERSION}")

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

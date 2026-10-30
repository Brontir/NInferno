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

# Route selection on the GPU classes this machine lacks (substituted DeviceClass, one process
# per board): every choice admitted, no feature the board lacks, its own route table.
ninferno_add_op_test(ninferno_route_classes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_classes.cpp"
  LIBRARIES ninferno_linear_test_support)

# The generic path's choices on thirteen substituted boards against selector_golden/<board>.txt:
# a selector change shows line by line; a refactor must leave every file as it was.
ninferno_add_op_test(ninferno_selector_golden_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_selector_golden.cpp"
  LIBRARIES ninferno_linear_test_support)
target_compile_definitions(ninferno_selector_golden_test PRIVATE
  NINFERNO_SELECTOR_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/selector_golden")

# The generic path's dense tuner: tuned at the first eager call, within the activation class, R8
# decode bytes kept, the cache file read back, nothing timed inside a graph capture.
ninferno_add_op_test(ninferno_dense_tune_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dense_tune.cpp"
  LIBRARIES ninferno_linear_test_support)

# The SwiGLU / GeGLU of a gate and an up weight of one or two formats (segmented_swiglu).
ninferno_add_op_test(ninferno_segmented_swiglu_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_segmented_swiglu.cpp"
  LIBRARIES ninferno_linear_test_support)

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

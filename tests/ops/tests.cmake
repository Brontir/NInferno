set(ninferno_op_tests
  add_bias
  gelu
  logit_softcap
  dense_fused
  gemma4_fused
  gemma4_vision
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  rmsnorm_pack_tail
  gated_rmsnorm
  l2norm
  gated_delta_net
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  finite_check
  gdn_gating
  gdn_gating_proj
  rope
  gemma4_attention
  gemma4_kv_bindings
  vision_pos_embed
  sampling
  scalar
  prepare_ragged_prefix
  scatter
  scatter_bf16_batch
  target_logprobs
  token_distribution
  position)
foreach(op IN LISTS ninferno_op_tests)
  ninferno_add_op_test(ninferno_${op}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.cpp"
    LIBRARIES ninferno_ops)
endforeach()

# Four runs of one executable (test_moe.cpp --part): the random formats in three shares and the
# shipped layers, plans and refusals took 125 s in one process on the RTX 5090; the runs are
# independent, so ctest -j spreads them. ninferno_moe_test keeps its name for the layers.
ninferno_add_op_test(ninferno_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_moe.cpp"
  LIBRARIES ninferno_ops
  NO_DEFAULT_TEST)
add_test(NAME ninferno_moe_test COMMAND ninferno_moe_test --part layers)
set(ninferno_moe_runs ninferno_moe_test)
foreach(share 0 1 2)
  add_test(NAME ninferno_moe_random${share}_test COMMAND ninferno_moe_test --part random${share})
  list(APPEND ninferno_moe_runs ninferno_moe_random${share}_test)
endforeach()
# MXFP4 banks on the block-scaled MMA with MXFP8 activations (sm_120a default; skips elsewhere).
add_test(NAME ninferno_moe_mx8_test COMMAND ninferno_moe_test --part mx8)
# The int8 W4A8 of MXFP4 banks, which sm_120a builds otherwise leave for the block-scaled MMA.
add_test(NAME ninferno_moe_int8_test COMMAND ninferno_moe_test --part int8)
list(APPEND ninferno_moe_runs ninferno_moe_mx8_test ninferno_moe_int8_test)
set_tests_properties(${ninferno_moe_runs} PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_op_test(ninferno_linear_topk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_topk.cu"
  LIBRARIES ninferno_ops)

# gemma4_attention / gemma4_kv_append over FP8 planes with row scales: the codec against the BF16
# rows and the attention against the FP64 formula over the represented values.
ninferno_add_op_test(ninferno_gemma4_attention_fp8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_attention_fp8.cu"
  LIBRARIES ninferno_ops)

# ggml's 32-value blocks and i-quants on the device (decode_element and the register-MMA
# decoders) bit for bit against the converter's ggml dequantization.
ninferno_add_op_test(ninferno_ggml_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ggml_decode.cu"
  LIBRARIES ninferno_ops ninferno_convert)

# E2M1 pair decoding (hardware conversion and the bit form) for all 256 bytes.
ninferno_add_op_test(ninferno_e2m1_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_e2m1_decode.cu"
  LIBRARIES ninferno_ops)

# E4M3 pair widening (the bit form below sm_89, the hardware's from it) for every finite pair.
ninferno_add_op_test(ninferno_fp8_e4m3_widen_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_e4m3.cu"
  LIBRARIES ninferno_ops)

# The register-fragment MMAs and ldmatrix forms (ops/common/mma.cuh) against a host product, bit for
# bit: sm_80 instructions, Turing's composed k8 shapes or the sm_70 FMA / dp4a forms.
ninferno_add_op_test(ninferno_mma_fragments_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mma_fragments.cu"
  LIBRARIES ninferno_ops)

# The SM-count launch rules (ops/common/sm_wave_rules.h): the RTX 5090's decisions unchanged at
# 170 SMs, the other boards' answers printed and pinned. No GPU.
ninferno_add_test(ninferno_sm_wave_rules_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sm_wave_rules.cpp"
  LIBRARIES ninferno_core)

# Per-GPU-class route_as overrides of the shape-owned Linear tables: loader and lookup.
ninferno_add_op_test(ninferno_shape_routes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_shape_routes.cpp"
  LIBRARIES ninferno_ops)

# Fused NVFP4 plans (attention/GDN input, GDN conv, LinearAdd, SwiGLU): A16 at every width under
# the A16Only policy (ActivationMode::Exact), and the route probe's forced routes. A build without
# the NVFP4 family (sm_89, sm_86) has none of these plans.
if(NINFERNO_WITH_NVFP4)
  ninferno_add_op_test(ninferno_nvfp4_fused_activation_routes_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_fused_activation_routes.cpp"
    LIBRARIES ninferno_ops)
endif()

ninferno_add_op_test(ninferno_candidate_selector_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_candidate_selector.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_softmax_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/causal_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/plain_and_packed.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/context.cpp"
  LIBRARIES ninferno_ops
  NO_DEFAULT_TEST)

# One run per causal KV dtype: four geometries of CPU oracles take over 20 minutes in one
# process on the RTX 5090 host, past the timeout, and the runs are independent.
add_test(NAME ninferno_softmax_attention_test
  COMMAND ninferno_softmax_attention_test --other-only)
set(ninferno_softmax_attention_runs ninferno_softmax_attention_test)
foreach(kv bf16 int8 fp8 nvfp4 k8v4)
  add_test(NAME ninferno_softmax_attention_${kv}_test
    COMMAND ninferno_softmax_attention_test --kv-dtype ${kv})
  list(APPEND ninferno_softmax_attention_runs ninferno_softmax_attention_${kv}_test)
endforeach()
add_test(NAME ninferno_softmax_attention_unwritten_test
  COMMAND ninferno_softmax_attention_test --unwritten-only)
list(APPEND ninferno_softmax_attention_runs ninferno_softmax_attention_unwritten_test)
set_tests_properties(${ninferno_softmax_attention_runs} PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_op_test(ninferno_sliding_window_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sliding_window_attention.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_kv_cache_append_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_append.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_rmsnorm_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm_rope.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_context_kv_materialize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_kv_materialize.cpp"
  LIBRARIES ninferno_ops)

add_test(NAME ninferno_kv_cache_append_nvfp4_test
  COMMAND ninferno_kv_cache_append_test --nvfp4-only)

add_test(NAME ninferno_kv_cache_append_k8v4_test
  COMMAND ninferno_kv_cache_append_test --k8v4-only)

set_tests_properties(
  ninferno_kv_cache_append_nvfp4_test
  ninferno_kv_cache_append_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_op_test(ninferno_prepare_masked_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare_masked_block.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_gemma4_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_moe.cpp"
  LIBRARIES ninferno_ops)

# ggml_frag_v1 routed-expert banks in the grouped MoE kernel: byte-identical to the published
# banks at T = 1..64, under both route profiles and on every compiled schedule pair.
ninferno_add_op_test(ninferno_moe_frag_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_moe_frag.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_sparse_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe.cpp"
  LIBRARIES ninferno_ops)

# SASS audit of programmatic dependent launch: no global load before griddepcontrol.wait
# in the ops library except the (kernel, parameter) pairs tools/pdl_prewait_loads.txt reviews.
# No GPU. The sabotage fixture carries a producer read hoisted above the wait that the audit must
# flag, and a correct twin it must pass.
get_filename_component(NINFERNO_CUDA_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
add_test(NAME ninferno_pdl_sass_audit_test
  COMMAND ${Python3_EXECUTABLE} -B "${PROJECT_SOURCE_DIR}/tools/pdl_sass_audit.py"
    --build "${PROJECT_BINARY_DIR}" --source "${PROJECT_SOURCE_DIR}"
    --allow "${PROJECT_SOURCE_DIR}/tools/pdl_prewait_loads.txt"
    --cuobjdump "${NINFERNO_CUDA_BIN}/cuobjdump")
set_tests_properties(ninferno_pdl_sass_audit_test PROPERTIES SKIP_RETURN_CODE 77)
add_library(ninferno_pdl_audit_sabotage OBJECT "${CMAKE_CURRENT_LIST_DIR}/pdl_audit_sabotage.cu")
ninferno_test_includes(ninferno_pdl_audit_sabotage)
add_test(NAME ninferno_pdl_sass_audit_sabotage_test
  COMMAND ${Python3_EXECUTABLE} -B "${PROJECT_SOURCE_DIR}/tools/pdl_sass_audit.py"
    --objects $<TARGET_OBJECTS:ninferno_pdl_audit_sabotage>
    --allow "${CMAKE_CURRENT_LIST_DIR}/pdl_audit_sabotage_allow.txt"
    --expect "pdl_audit_hoisted_consumer:0" --cuobjdump "${NINFERNO_CUDA_BIN}/cuobjdump")
set_tests_properties(ninferno_pdl_sass_audit_sabotage_test PROPERTIES SKIP_RETURN_CODE 77)

# The Qwen3.5 and Gemma 4 warp top-8 selections against FP64: negative logits, exact ties at
# rank 8, equal logits, tiny weights, renormalisation.
ninferno_add_op_test(ninferno_moe_routing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_moe_routing.cu"
  LIBRARIES ninferno_ops)

# The Mamba2 selective state scan against FP64: NemotronH's mixer and small geometries, slot pools,
# single-token calls against multi-token calls.
ninferno_add_op_test(ninferno_mamba2_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mamba2.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_mtp_pack_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_pack.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_mtp_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_round.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_attn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_gdn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_dynamic_grouped_conv_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dynamic_grouped_conv_prepare.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_linear_dynamic_grouped_conv_add_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_dynamic_grouped_conv_add.cpp"
  LIBRARIES ninferno_ops)

# DFlash2 outside the fused Q8/BF16 profile (the dflash GGUF's Q4_0, a K-quant or NVFP4 head):
# linear_topk over Linear's logits and the decomposed dynamic grouped convolutions.
ninferno_add_op_test(ninferno_dflash2_generic_routes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash2_generic_routes.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_gdn_input_proj_conv_snapshot_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_snapshot.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_gdn_input_proj_conv_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_record.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_gated_delta_net_replay_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta_net_replay_record.cpp"
  LIBRARIES ninferno_ops)

ninferno_add_op_test(ninferno_gdn_replay_fold_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_replay_fold.cpp"
  LIBRARIES ninferno_ops)

include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/tests.cmake")

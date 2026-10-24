ninferno_add_op_bench(ninferno_silu_mul_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/silu_mul_bench.cu")
ninferno_add_op_bench(ninferno_residual_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/residual_add_bench.cu")
ninferno_add_op_bench(ninferno_sigmoid_mul_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/sigmoid_mul_bench.cu")
ninferno_add_op_bench(ninferno_rmsnorm_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_bench.cu")
ninferno_add_op_bench(ninferno_rmsnorm_pack_tail_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_pack_tail_bench.cu")
ninferno_add_op_bench(ninferno_l2norm_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/l2norm_bench.cu")
ninferno_add_op_bench(ninferno_gdn_gating_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/gdn_gating_bench.cu")
ninferno_add_op_bench(ninferno_gdn_gating_proj_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj_bench.cu")
ninferno_add_op_bench(ninferno_dynamic_grouped_conv_prepare_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv_prepare_bench.cu")
ninferno_add_op_bench(ninferno_linear_dynamic_grouped_conv_add_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_dynamic_grouped_conv_add_bench.cu")
ninferno_add_op_bench(ninferno_gated_delta_net_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net_bench.cu")
ninferno_add_op_bench(ninferno_gdn_replay_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/gdn_replay_bench.cu")
ninferno_add_op_bench(ninferno_rope_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/rope_bench.cu")
ninferno_add_op_bench(ninferno_rmsnorm_rope_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope_bench.cu")
ninferno_add_op_bench(ninferno_add_bias_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/add_bias_bench.cu")
ninferno_add_op_bench(ninferno_gelu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/gelu_bench.cu")
ninferno_add_op_bench(ninferno_layer_norm_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/layer_norm_bench.cu")
ninferno_add_op_bench(ninferno_vision_pos_embed_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/vision_pos_embed_bench.cu")
ninferno_add_op_bench(ninferno_scatter_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/scatter_bench.cu")
ninferno_add_op_bench(ninferno_scatter_bf16_batch_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/scatter_bf16_batch_bench.cu")
ninferno_add_op_bench(ninferno_packed_softmax_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/packed_softmax_attention_bench.cu")
ninferno_add_op_bench(ninferno_prepare_ragged_prefix_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/prepare_ragged_prefix_bench.cu")
ninferno_add_op_bench(ninferno_embedding_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/embedding_bench.cu")
ninferno_add_op_bench(ninferno_position_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/position_bench.cu")
ninferno_add_op_bench(ninferno_argmax_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/argmax_bench.cu")
ninferno_add_op_bench(ninferno_causal_conv1d_silu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/causal_conv1d_silu_bench.cu")
ninferno_add_op_bench(ninferno_linear_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_bench.cu")
ninferno_add_op_bench(ninferno_linear_route_sweep_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_route_sweep_bench.cu")
# Offline autotuner of the generic projection path: writes a per-GPU-class route table.
ninferno_add_op_bench(ninferno_route_tune SOURCES "${CMAKE_CURRENT_LIST_DIR}/route_tune.cpp")
# ggml's matrix kernel at prefill widths: stream-k against one CTA per tile (the prefill profile's).
ninferno_add_op_bench(ninferno_ggml_mmq_prefill_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_prefill_bench.cpp")
# Every R8 decode schedule on given problems, next to the dispatcher's pick.
ninferno_add_op_bench(ninferno_r8_schedule_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/r8_schedule_bench.cpp")
ninferno_add_op_bench(ninferno_linear_topk_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_topk_bench.cu")
ninferno_add_op_bench(ninferno_candidate_selector_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/candidate_selector_bench.cu")
ninferno_add_op_bench(ninferno_gdn_input_proj_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj_bench.cu")
ninferno_add_op_bench(ninferno_gdn_input_proj_conv_snapshot_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj_conv_snapshot_bench.cu")
ninferno_add_op_bench(ninferno_attn_input_proj_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/attn_input_proj_bench.cu")
ninferno_add_op_bench(ninferno_q8_linear_swiglu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/q8_linear_swiglu_bench.cu")
ninferno_add_op_bench(ninferno_nvfp4_linear_swiglu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/nvfp4_linear_swiglu_bench.cu")
ninferno_add_op_bench(ninferno_nvfp4_a16_mma_sweep_bench SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a16_mma_sweep_bench.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a16_mma_sweep_n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a16_mma_sweep_n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a16_mma_sweep_n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a16_mma_sweep_n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a16_mma_sweep_n5120_k17408.cu")
ninferno_add_op_bench(ninferno_fp8_linear_swiglu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/fp8_linear_swiglu_bench.cu")
ninferno_add_op_bench(ninferno_q4_linear_swiglu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/q4_linear_swiglu_bench.cu")
ninferno_add_op_bench(ninferno_q8_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/q8_linear_add_bench.cu")
ninferno_add_op_bench(ninferno_q5_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/q5_linear_add_bench.cu")
ninferno_add_op_bench(ninferno_bf16_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/bf16_linear_add_bench.cu")
ninferno_add_op_bench(ninferno_nvfp4_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/nvfp4_linear_add_bench.cu")
ninferno_add_op_bench(ninferno_fp8_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/fp8_linear_add_bench.cu")
ninferno_add_op_bench(ninferno_causal_softmax_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/causal_softmax_attention_bench.cu")
ninferno_add_op_bench(ninferno_context_softmax_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context_softmax_attention_bench.cu")
ninferno_add_op_bench(ninferno_sliding_window_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/sliding_window_attention_bench.cu")
ninferno_add_op_bench(ninferno_gemma4_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/gemma4_attention_bench.cu")
ninferno_add_op_bench(ninferno_kv_cache_append_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/kv_cache_append_bench.cu")
ninferno_add_op_bench(ninferno_prepare_masked_block_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/prepare_masked_block_bench.cu")
ninferno_add_op_bench(ninferno_sampling_select_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/sampling_select_bench.cu")
ninferno_add_op_bench(ninferno_gemma4_moe_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/gemma4_moe_bench.cu")
ninferno_add_op_bench(ninferno_sparse_moe_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/sparse_moe_bench.cu")
# The generic MoE Op: any geometry and expert format, public route or a schedule sweep.
ninferno_add_op_bench(ninferno_moe_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/moe_bench.cu")
ninferno_add_op_bench(ninferno_linear_pair_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_pair_bench.cu")
ninferno_add_op_bench(ninferno_context_kv_materialize_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize_bench.cu")
ninferno_add_op_bench(ninferno_mtp_pack_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/mtp_pack_bench.cu")
ninferno_add_op_bench(ninferno_proposal_remap_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/proposal_remap_bench.cu")

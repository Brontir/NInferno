ninferno_add_test(ninferno_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninferno_model_loading)

ninferno_add_test(ninferno_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninferno_model_loading)

set_tests_properties(
  ninferno_qwen3_5_loading_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninferno_engine ninferno_core ninferno::json)

# Output presentation is invariant to how generated tokens are cut into rounds.
ninferno_add_test(ninferno_qwen3_5_output_slicing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_output_slicing.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninferno_engine ninferno_core ninferno::json)

# Parity with transformers on a checkpoint's own tokenizer files (hf_tokenizer_parity.py writes
# them and the expected ids; NINFERNO_TOKENIZER_PARITY_DIR names them). Skipped without them.
ninferno_add_test(ninferno_qwen3_5_tokenizer_parity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tokenizer_parity.cpp"
  LIBRARIES ninferno_engine ninferno_core ninferno::json)

set_tests_properties(
  ninferno_qwen3_5_tokenizer_parity_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Prompts rendered through the serve request path and the Qwen frontend equal
# transformers' apply_chat_template on a checkpoint's own files (hf_chat_template_parity.py
# writes them and the reference; NINFERNO_CHAT_TEMPLATE_PARITY_DIR names them). Skipped without them.
ninferno_add_test(ninferno_qwen3_5_chat_template_parity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_chat_template_parity.cpp"
  LIBRARIES ninferno_serve ninferno_engine ninferno_core ninferno::json)

set_tests_properties(
  ninferno_qwen3_5_chat_template_parity_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninferno_engine ninferno_core)

# Decode round pacing (host only): steps and waits on quiet, lagging and fixed hosts.
ninferno_add_test(ninferno_qwen3_5_round_pacing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_round_pacing.cpp")

# Ordinary decode run-ahead through the Engine: eight steps per round emit exactly the tokens of
# one step per round (stops mid-round, stop strings across rounds, limits, two rows, sampling, a
# follow-up past the sliding window, cancellation). NINFERNO_TEST_RUN_AHEAD_ARTIFACTS names the
# artifacts; skipped without it. GPU + artifacts.
ninferno_add_test(ninferno_qwen3_5_ordinary_run_ahead_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ordinary_run_ahead_real.cpp"
  LIBRARIES ninferno_engine)

set_tests_properties(
  ninferno_qwen3_5_ordinary_run_ahead_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

# A speculative engine that blocks on every round's end emits exactly the tokens and rounds of one
# that spins (stops, limits, two rows, sampling, prefix reuse, cancellation) under
# --activations exact. NINFERNO_TEST_SPEC_WAIT_ARTIFACT names the artifact (DFlash K=3 unless
# NINFERNO_TEST_SPEC_WAIT_BACKEND / _K / _DRAFTER say otherwise); skipped without it. GPU + artifact.
ninferno_add_test(ninferno_qwen3_5_speculative_wait_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_wait_real.cpp"
  LIBRARIES ninferno_engine)

set_tests_properties(
  ninferno_qwen3_5_speculative_wait_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninferno_engine ninferno_core)

set_tests_properties(
  ninferno_qwen3_5_state_image_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninferno_engine ninferno_core)

ninferno_add_test(ninferno_qwen3_5_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_store.cpp"
  LIBRARIES ninferno_engine ninferno_core)

set_tests_properties(
  ninferno_qwen3_5_context_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninferno_engine ninferno_artifact)

set_tests_properties(
  ninferno_qwen3_5_prefix_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninferno_engine)

set_tests_properties(
  ninferno_qwen3_5_score_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_vision_workspace_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninferno_model_runtime ninferno_engine)

set_tests_properties(
  ninferno_qwen3_5_vision_workspace_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninferno_engine ninferno_artifact)

set_tests_properties(
  ninferno_qwen3_5_dflash2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_dflash_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES ninferno_model_runtime ninferno_model_loading ninferno_core)

set_tests_properties(
  ninferno_qwen3_5_dflash_prefill_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninferno_engine ninferno_artifact)

set_tests_properties(
  ninferno_qwen3_5_moe_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninferno_engine ninferno_artifact)

set_tests_properties(
  ninferno_qwen3_5_dflash_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_qwen3_5_speculative_invariants_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_speculative_invariants_real.cpp"
  LIBRARIES ninferno_engine ninferno_artifact)

set_tests_properties(
  ninferno_qwen3_5_speculative_invariants_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../../test_tool_call_parser.cpp"
  LIBRARIES ninferno_engine ninferno_structured ninferno::json)

ninferno_add_test(ninferno_qwen3_5_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visual_scatter.cpp"
  LIBRARIES ninferno_engine ninferno_core)

set_tests_properties(
  ninferno_qwen3_5_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Measurement tool for prefill chunk invariance (not registered as a test).
add_executable(ninferno_qwen3_5_chunk_probe "${CMAKE_CURRENT_LIST_DIR}/chunk_probe.cpp")
target_link_libraries(ninferno_qwen3_5_chunk_probe PRIVATE ninferno_engine)

# Measurement tool for batched speculative verify A/B (not registered as a test).
add_executable(ninferno_qwen3_5_spec_probe "${CMAKE_CURRENT_LIST_DIR}/spec_probe.cpp")
target_link_libraries(ninferno_qwen3_5_spec_probe PRIVATE ninferno_engine)

# Measurement tool for prefix-reuse identity (not registered as a test).
add_executable(ninferno_qwen3_5_prefix_probe "${CMAKE_CURRENT_LIST_DIR}/prefix_probe.cpp")
target_link_libraries(ninferno_qwen3_5_prefix_probe PRIVATE ninferno_engine ninferno::json)

# Determinism contracts (solo, sampling parameters, batch composition, A->B->A with the prefix
# cache, process restarts) on NINFERNO_TEST_ARTIFACT; skipped without it.
ninferno_add_test(ninferno_qwen3_5_determinism_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_determinism_real.cpp"
  LIBRARIES ninferno_engine ninferno::json ninferno::cudart)

set_tests_properties(
  ninferno_qwen3_5_determinism_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_dense_session_real_test LABELS real gpu
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_session_real.cpp"
  LIBRARIES ninferno_dense)
set_tests_properties(ninferno_dense_session_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The plan without the weights for the dense decoder, moe_decoder and NemotronH (#615 W3): --plan's
# dry run over a lab model's artifact directory (Qwen3-8B, Llama-3.1-8B, gpt-oss-20b, Nemotron 3
# Nano) sizes the start planner's capacity over the described model, to the byte, on substituted
# RTX 5090, 4070 Ti and 4060 Ti boards; the declines keep the formula and say why. No GPU, no
# files (NINFERNO_PLAN_WORKSPACE_ARTIFACTS adds real artifact files).
ninferno_add_test(ninferno_decoder_plan_workspace_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_plan_workspace.cpp"
  LIBRARIES ninferno_product_plan ninferno_engine ninferno_model_runtime ninferno_core
    ninferno::json)
target_include_directories(ninferno_decoder_plan_workspace_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src)

# The same against a real start (#615 W3): one dense_decoder, one moe_decoder and one nemotron_h
# artifact loaded as ninferno-serve loads them; the start's planned arena equals --plan's workspace
# to the byte, fast and exact. NINFERNO_TEST_PLAN_WORKSPACE_ARTIFACTS names them, else the converted
# artifacts in the Hugging Face cache (models--*/ninferno/<commit>/[options-<key>/] and any
# .ninferno-cache/ under snapshots/), each family's smallest that fits and runs on this build and
# card. None skips; a refused file of this container warns beside a tested artifact and fails where
# none was tested. `--scan-only` scans and picks without a device. GPU + artifacts.
ninferno_add_test(ninferno_decoder_plan_workspace_real_test LABELS real gpu
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_plan_workspace_real.cpp"
  LIBRARIES ninferno_product_plan ninferno_engine ninferno_model_runtime ninferno_convert
    ninferno_core ninferno::json)
target_include_directories(ninferno_decoder_plan_workspace_real_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
set_tests_properties(ninferno_decoder_plan_workspace_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Prefix reuse through the Engine on a dense artifact whose prompt ends on an execution boundary
# (SmolLM3): a zero-suffix restore of a shared prefix, cold == warm. GPU + artifact.
ninferno_add_test(ninferno_dense_reuse_real_test LABELS real gpu
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reuse_real.cpp"
  LIBRARIES ninferno_engine)
set_tests_properties(ninferno_dense_reuse_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The frontend's end-of-generation ids (generation_config.json joined with the tokenizer's eos). CPU.
ninferno_add_test(ninferno_dense_frontend_stops_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend_stops.cpp"
  LIBRARIES ninferno_dense)

# LongRoPE (Phi-4-mini, Phi-3.5-mini) through the Engine against ninferno-dense's Session: the
# session's context fixes the rotation mode on both paths; a prompt past long_after in a long
# context and one in a short context give the same greedy tokens. GPU + artifact.
ninferno_add_test(ninferno_dense_longrope_real_test LABELS real gpu
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_longrope_real.cpp"
  LIBRARIES ninferno_engine ninferno_dense)
set_tests_properties(ninferno_dense_longrope_real_test PROPERTIES SKIP_RETURN_CODE 77)

# --activations exact through the Engine (ninferno-serve) against the CLI Session's exact profile on a
# real dense artifact: the same greedy tokens, byte for byte. GPU + artifact.
ninferno_add_test(ninferno_dense_engine_exact_real_test LABELS real gpu
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_exact_real.cpp"
  LIBRARIES ninferno_engine ninferno_dense)
set_tests_properties(ninferno_dense_engine_exact_real_test PROPERTIES SKIP_RETURN_CODE 77)

# DFlash speculation through the Engine (ninferno-serve) on a real dense target and its published
# drafter: under --activations exact the speculative greedy tokens equal plain decode's at every
# depth, with two requests in one launch and after a chunk wider than the drafter's window.
# GPU + artifacts.
ninferno_add_test(ninferno_dense_dflash_real_test LABELS real gpu
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_real.cpp"
  LIBRARIES ninferno_engine)
set_tests_properties(ninferno_dense_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)

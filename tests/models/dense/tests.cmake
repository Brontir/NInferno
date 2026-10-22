ninferno_add_test(ninferno_dense_session_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_session_real.cpp"
  LIBRARIES ninferno_dense)
set_tests_properties(ninferno_dense_session_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Prefix reuse through the Engine on a dense artifact whose prompt ends on an execution boundary
# (SmolLM3): a zero-suffix restore of a shared prefix, cold == warm. GPU + artifact.
ninferno_add_test(ninferno_dense_reuse_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reuse_real.cpp"
  LIBRARIES ninferno_engine)
set_tests_properties(ninferno_dense_reuse_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The frontend's end-of-generation ids (generation_config.json joined with the tokenizer's eos). CPU.
ninferno_add_test(ninferno_dense_frontend_stops_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend_stops.cpp"
  LIBRARIES ninferno_dense)

# LongRoPE (Phi-4-mini, Phi-3.5-mini) through the Engine against ninferno-dense's Session: the
# session's context fixes the rotation mode on both paths; a prompt past long_after in a long
# context and one in a short context give the same greedy tokens. GPU + artifact.
ninferno_add_test(ninferno_dense_longrope_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_longrope_real.cpp"
  LIBRARIES ninferno_engine ninferno_dense)
set_tests_properties(ninferno_dense_longrope_real_test PROPERTIES SKIP_RETURN_CODE 77)

# --activations exact through the Engine (ninferno-serve) against the CLI Session's exact profile on a
# real dense artifact: the same greedy tokens, byte for byte. GPU + artifact.
ninferno_add_test(ninferno_dense_engine_exact_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_exact_real.cpp"
  LIBRARIES ninferno_engine ninferno_dense)
set_tests_properties(ninferno_dense_engine_exact_real_test PROPERTIES SKIP_RETURN_CODE 77)

# DFlash speculation through the Engine (ninferno-serve) on a real dense target and its published
# drafter: under --activations exact the speculative greedy tokens equal plain decode's at every
# depth, with two requests in one launch and after a chunk wider than the drafter's window.
# GPU + artifacts.
ninferno_add_test(ninferno_dense_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_real.cpp"
  LIBRARIES ninferno_engine)
set_tests_properties(ninferno_dense_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)

# --activations auto|16|8|4 through the Engine on real artifacts of any family (dense, Gemma 4,
# Nemotron-H, Qwen): every mode starts and answers, or 8/4 is refused by the startup resolution
# only. NINFERNO_TEST_PRECISION_ARTIFACTS lists them (';'). GPU + artifacts.
ninferno_add_test(ninferno_engine_precision_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_engine_precision_real.cpp"
  LIBRARIES ninferno_engine)
set_tests_properties(ninferno_engine_precision_real_test PROPERTIES SKIP_RETURN_CODE 77)

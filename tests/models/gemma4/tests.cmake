ninferno_add_test(ninferno_gemma4_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninferno_gemma4)

ninferno_add_test(ninferno_gemma4_speculation_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculation_real.cpp"
  LIBRARIES ninferno_gemma4)

ninferno_add_test(ninferno_gemma4_prompt_lookup_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prompt_lookup_real.cpp"
  LIBRARIES ninferno_gemma4 ninferno_spec)

ninferno_add_test(ninferno_gemma4_window_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_window_real.cpp"
  LIBRARIES ninferno_gemma4)
set_tests_properties(ninferno_gemma4_window_real_test PROPERTIES SKIP_RETURN_CODE 77)

if(TARGET ninferno-gemma4)
  add_test(NAME ninferno_gemma4_cli_generate_context_real_test
    COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/cli_generate_context_real.sh $<TARGET_FILE:ninferno-gemma4>)
  set_tests_properties(ninferno_gemma4_cli_generate_context_real_test PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME ninferno_gemma4_cli_options_test
    COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/cli_options.sh $<TARGET_FILE:ninferno-gemma4>)
  # Gemma 4 E2B/E4B (per-layer embeddings, shared KV) against a reference on the same weights.
  foreach(case e2b_bf16 e2b_qat_q4_0 e4b_bf16)
    add_test(NAME ninferno_gemma4_${case}_score_real_test
      COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/cli_score_ref_real.sh $<TARGET_FILE:ninferno-gemma4> ${case})
    set_tests_properties(ninferno_gemma4_${case}_score_real_test PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()

set_tests_properties(
  ninferno_gemma4_loading_real_test
  ninferno_gemma4_speculation_real_test
  ninferno_gemma4_prompt_lookup_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

# --activations exact through the Engine (ninferno-serve) against the CLI Session's exact profile on a
# real Gemma 4 artifact: the same greedy tokens, byte for byte. GPU + artifact.
ninferno_add_test(ninferno_gemma4_engine_exact_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_exact_real.cpp"
  LIBRARIES ninferno_engine ninferno_gemma4)
set_tests_properties(ninferno_gemma4_engine_exact_real_test PROPERTIES SKIP_RETURN_CODE 77)

# A separate-drafter family whose automatic prefill chunk the fit narrows starts and speculates:
# Gemma 4 with its MTP assistant, the free memory it sees made to fit only a narrower chunk
# (cudaMemGetInfo wrapped). GPU + artifacts. --wrap is an ELF linker option (GNU ld, gold, lld,
# mold); CMake names the linker from 3.29, before that a non-Apple Unix stands for it. Elsewhere
# the test is not built.
if(DEFINED CMAKE_CXX_COMPILER_LINKER_ID)
  set(ninferno_linker_wraps FALSE)
  if(CMAKE_CXX_COMPILER_LINKER_ID MATCHES "^(GNU|GNUgold|LLD|MOLD)$")
    set(ninferno_linker_wraps TRUE)
  endif()
elseif(UNIX AND NOT APPLE)
  set(ninferno_linker_wraps TRUE)
else()
  set(ninferno_linker_wraps FALSE)
endif()
if(ninferno_linker_wraps)
  ninferno_add_test(ninferno_gemma4_narrowed_chunk_real_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_narrowed_chunk_real.cpp"
    LIBRARIES ninferno_engine ninferno::cudart)
  target_link_options(ninferno_gemma4_narrowed_chunk_real_test PRIVATE
    "LINKER:--wrap=cudaMemGetInfo")
  set_tests_properties(ninferno_gemma4_narrowed_chunk_real_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

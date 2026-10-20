# H1 diff oracle (tests/oracle/hf_oracle.py): synthetic mini Qwen3.5 and Gemma 4 checkpoints through
# ninferno-convert and the engine, against Hugging Face transformers in FP32. The runner is the
# engine side: teacher-forced logits of an artifact over given token sequences. Skipped without
# torch/transformers in NINFERNO_CONVERT_PYTHON or without a CUDA device.
add_executable(ninferno_hf_oracle_runner "${CMAKE_CURRENT_LIST_DIR}/oracle_runner.cpp")
ninferno_test_includes(ninferno_hf_oracle_runner)
target_link_libraries(ninferno_hf_oracle_runner PRIVATE ninferno_engine ninferno_gemma4)

add_test(NAME ninferno_hf_oracle_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/hf_oracle.py"
    $<TARGET_FILE:ninferno-convert> $<TARGET_FILE:ninferno_hf_oracle_runner> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_hf_oracle_test PROPERTIES SKIP_RETURN_CODE 77)

# Greedy ids of raw token input through the public Engine (quick family cross-checks).
add_executable(ninferno_greedy_ids "${CMAKE_CURRENT_LIST_DIR}/greedy_ids.cpp")
ninferno_test_includes(ninferno_greedy_ids)
target_link_libraries(ninferno_greedy_ids PRIVATE ninferno_engine)

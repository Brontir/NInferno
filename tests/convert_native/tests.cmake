ninferno_add_test(ninferno_convert_numerics_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_numerics.cpp"
  LIBRARIES ninferno_convert)

ninferno_add_test(ninferno_convert_ggml_quants_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ggml_quants.cpp"
  LIBRARIES ninferno_convert)
target_compile_options(ninferno_convert_ggml_quants_test PRIVATE -ffp-contract=off)

# ggml_frag_v1: the maps, row round trips, the auto rule and --relayout (in place, resumed,
# --out). With GGUF paths as arguments it also round-trips real routed-expert rows.
ninferno_add_test(ninferno_convert_ggml_frag_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ggml_frag.cpp"
  LIBRARIES ninferno_convert)

# --relayout takes no conversion flag (it only reads and rewrites an artifact); the other argument
# refusals are in tests/cmake/AppRefusalTests.cmake.
add_test(NAME ninferno_convert_relayout_flags_test
  COMMAND $<TARGET_FILE:ninferno-convert> --relayout auto --target-artifact missing.ninferno
    --recipe dense_q8)
set_tests_properties(ninferno_convert_relayout_flags_test PROPERTIES
  PASS_REGULAR_EXPRESSION "--relayout takes only --target-artifact, --out, --max-file-bytes and --quiet, not --recipe")

ninferno_add_test(ninferno_convert_gguf_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf.cpp"
  LIBRARIES ninferno_convert)

# The parity test runs the Python converter, which needs torch; point this at an interpreter
# that has it (e.g. -DNINFERNO_CONVERT_PYTHON=<venv>/bin/python). Without torch it is skipped.
set(NINFERNO_CONVERT_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH
  "Python interpreter with torch for the native/Python converter parity test")
add_test(NAME ninferno_convert_python_parity_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_gemma4.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_python_parity_test PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME ninferno_convert_qwen_parity_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_qwen3_5.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_qwen_parity_test PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_convert_checkpoint_index_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_checkpoint_index.cpp"
  LIBRARIES ninferno_convert)

ninferno_add_test(ninferno_convert_consume_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_consume.cpp"
  LIBRARIES ninferno_convert)

# Gemma 4 model properties (attention scale, GELU-tanh, plain-w RMSNorm, k_eq_v, KV sharing,
# the BF16 embedding normalizer, final softcap), each against transformers and the converter.
add_test(NAME ninferno_convert_gemma4_properties_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/gemma4_properties.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_gemma4_properties_test PROPERTIES SKIP_RETURN_CODE 77)

# NVFP4 scales through the converter: fused q/k/v and gate/up with differing divisors, stacked
# expert planes, per-expert activation scales and dead experts (0, 1e-23, NaN, inf scales).
add_test(NAME ninferno_convert_scales_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_scales.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_scales_test PROPERTIES SKIP_RETURN_CODE 77)

# Converted NVFP4 artifacts (fused and stacked planes) uploaded into a 0xFF-poisoned arena
# must come back byte for byte, with the poison intact in the gaps (tests/artifact).
add_test(NAME ninferno_artifact_poisoned_upload_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${PROJECT_SOURCE_DIR}/tests/artifact/poisoned_upload.py"
    $<TARGET_FILE:ninferno-convert> $<TARGET_FILE:ninferno_artifact_materialization_test>
    "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_artifact_poisoned_upload_test PROPERTIES SKIP_RETURN_CODE 77)

# Quantized or not is decided by the tensors: recipe choice without --recipe against checkpoints
# whose quantization_config disagrees with their headers, and per-site formats in a mixed one.
add_test(NAME ninferno_convert_detection_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_detection.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_detection_test PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME ninferno_convert_formats_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_formats.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_formats_test PROPERTIES SKIP_RETURN_CODE 77)

# A tied output head is one copy of the embedding table (dense and Qwen3.5 adapters); a scaled
# table (Granite's embedding_multiplier) keeps a separate head.
add_test(NAME ninferno_convert_tied_head_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_tied.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_tied_head_test PROPERTIES SKIP_RETURN_CODE 77)

# Published NVFP4 without calibrated activations (W4A16, ModelOpt MIXED MLPs) through dense_published:
# fused planes convert A16Only with the same weights as the calibrated (AllowA4) twin.
add_test(NAME ninferno_convert_w4a16_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_w4a16.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_w4a16_test PROPERTIES SKIP_RETURN_CODE 77)

# The GGUF reader and ggml decoders against llama.cpp's gguf-py (its writer, reader and numpy
# dequantizers), bit for bit. Skipped when the interpreter lacks the gguf package.
add_executable(ninferno_gguf_dump "${CMAKE_CURRENT_LIST_DIR}/gguf_dump.cpp")
ninferno_test_includes(ninferno_gguf_dump)
target_link_libraries(ninferno_gguf_dump PRIVATE ninferno_convert)
add_test(NAME ninferno_convert_gguf_pyref_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/gguf_pyref.py"
    $<TARGET_FILE:ninferno_gguf_dump>)
set_tests_properties(ninferno_convert_gguf_pyref_test PROPERTIES SKIP_RETURN_CODE 77)

# GGUF checkpoints (written with gguf-py as llama.cpp's converter writes them) against their
# Hugging Face twins: the native converter's artifacts must match byte for byte.
add_test(NAME ninferno_convert_gguf_synthetic_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_gguf.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_gguf_synthetic_test PROPERTIES SKIP_RETURN_CODE 77)

# llama.cpp's dflash GGUF with DFlash2's convolutions and selector (a Qwen3.5 target's dflash2
# component) against its z-lab twin: BF16 byte for byte, Q4_0 kept as published.
add_test(NAME ninferno_convert_gguf_dflash2_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_dflash2_gguf.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_gguf_dflash2_test PROPERTIES SKIP_RETURN_CODE 77)

# ninferno-convert --fetch against a local stand-in hub: ranges, resume, stalls, digests, auth; and
# a GGUF file's first bytes (fetch_head_probe: fetch_gguf_head).
add_executable(fetch_head_probe "${CMAKE_CURRENT_LIST_DIR}/fetch_head_probe.cpp")
ninferno_test_includes(fetch_head_probe)
target_link_libraries(fetch_head_probe PRIVATE ninferno_convert)
add_test(NAME ninferno_convert_fetch_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/fetch_test.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}" $<TARGET_FILE:fetch_head_probe>)

# ninferno-convert attaching the drafters a model declares or the registry names (stand-in hub,
# synthetic Gemma-shaped checkpoints; skipped without torch).
add_test(NAME ninferno_convert_drafters_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/drafters_test.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_drafters_test PROPERTIES SKIP_RETURN_CODE 77)

# The drafter plan without a hub: GGUF drafters without num_target_layers, lineage by geometry,
# the dense engine's drafters for validated models only.
add_executable(ninferno_convert_drafter_plan_test "${CMAKE_CURRENT_LIST_DIR}/test_drafter_plan.cpp")
ninferno_test_includes(ninferno_convert_drafter_plan_test)
target_link_libraries(ninferno_convert_drafter_plan_test PRIVATE ninferno_convert)
add_test(NAME ninferno_convert_drafter_plan_test COMMAND ninferno_convert_drafter_plan_test)

# The dense decoder adapter: rotary tables bit-exact with transformers, resolved conventions.
add_executable(ninferno_convert_dense_config_test "${CMAKE_CURRENT_LIST_DIR}/test_dense_config.cpp")
ninferno_test_includes(ninferno_convert_dense_config_test)
target_link_libraries(ninferno_convert_dense_config_test PRIVATE ninferno_convert)
add_test(NAME ninferno_convert_dense_config_test
  COMMAND ninferno_convert_dense_config_test "${CMAKE_CURRENT_LIST_DIR}/fixtures/dense_rope.json")

# Gemma 4 as published (gemma4_published): NVFP4 projections and expert banks imported, the table,
# head, router and every other tensor BF16; no q8/q6 object.
add_test(NAME ninferno_convert_gemma4_published_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_gemma4_published.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_gemma4_published_test PROPERTIES SKIP_RETURN_CODE 77)

# The NemotronH MTP drafter sidecar (a nemotron_h_moe GGUF that carries only the folded NextN
# block): config, bindings and formats of the drafter artifact, Q4_0 kept word for word.
add_test(NAME ninferno_convert_nemotron_mtp_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_nemotron_mtp.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_nemotron_mtp_test PROPERTIES SKIP_RETURN_CODE 77)

# A NemotronH trunk GGUF that carries its MTP block after the trunk (unsloth's block_count 53):
# the converted target has the trunk's layers only, as the same trunk without the block.
add_test(NAME ninferno_convert_nemotron_trunk_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_nemotron_trunk.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_nemotron_trunk_test PROPERTIES SKIP_RETURN_CODE 77)

# A published NVFP4 output head (ModelOpt's lm_head of nvidia/Muse-Glimmer-30B-NVFP4) whose rows
# end inside a 128-row scale tile: kept word for word by the dense recipes under every --vocab.
add_test(NAME ninferno_convert_nvfp4_head_test
  COMMAND ${NINFERNO_CONVERT_PYTHON} -B "${CMAKE_CURRENT_LIST_DIR}/synthetic_nvfp4_head.py"
    $<TARGET_FILE:ninferno-convert> "${PROJECT_SOURCE_DIR}")
set_tests_properties(ninferno_convert_nvfp4_head_test PROPERTIES SKIP_RETURN_CODE 77)

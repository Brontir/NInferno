# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the public product headers stand alone.
add_executable(ninferno_public_api_test "${CMAKE_CURRENT_LIST_DIR}/../test_public_api.cpp")
target_include_directories(ninferno_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninferno_public_api_test COMMAND ninferno_public_api_test)

ninferno_add_test(ninferno_device_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device.cpp"
  LIBRARIES ninferno_core)

set(sync_modes spin blocking yield auto)
set(sync_flags 1 4 2 0)
foreach(mode flags IN ZIP_LISTS sync_modes sync_flags)
  add_test(NAME ninferno_device_sync_${mode}_test COMMAND ninferno_device_test ${mode} ${flags})
  set_tests_properties(ninferno_device_sync_${mode}_test PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
add_test(NAME ninferno_device_sync_invalid_test COMMAND ninferno_device_test --invalid-sync invalid)
add_test(NAME ninferno_device_sync_empty_test COMMAND ninferno_device_test --invalid-sync "")

# A build with a library that has no image for the device is refused at startup. The
# stand-in library is one kernel compiled for sm_90 SASS only.
add_library(ninferno_device_image_probe_sm90 OBJECT "${CMAKE_CURRENT_LIST_DIR}/../device_image_sm90.cu")
set_target_properties(ninferno_device_image_probe_sm90 PROPERTIES CUDA_ARCHITECTURES 90-real)
ninferno_add_test(ninferno_device_image_negative_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device_image_negative.cpp"
          $<TARGET_OBJECTS:ninferno_device_image_probe_sm90>
  LIBRARIES ninferno_core)
set_tests_properties(ninferno_device_image_negative_test PROPERTIES SKIP_RETURN_CODE 77)

# The PCI-ID reference table of board facts and machine_facts() (core/machine_facts.h). No GPU.
ninferno_add_test(ninferno_machine_facts_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_machine_facts.cpp"
  LIBRARIES ninferno_core)

# The formats ggml's vector kernel takes by rule at T = 2..4 (ops/generic/vec_verify_rule.h). No GPU.
ninferno_add_test(ninferno_vec_verify_rule_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_vec_verify_rule.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_decode_graph_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_graph.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_tensor_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tensor.cpp"
  LIBRARIES ninferno_core)

# The operating system calls (core/sys.h) and the 128-bit helpers (core/checked.h): the same checks
# on Linux and Windows. Temporary files; no GPU.
ninferno_add_test(ninferno_sys_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_sys.cpp"
  LIBRARIES ninferno_sys)

# Pinned Host memory: /proc/meminfo, the headroom and the refusal with numbers. No GPU.
ninferno_add_test(ninferno_host_memory_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_memory.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_arena_test        SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_device_buffer_fill_order_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device_buffer_fill_order.cpp"
  LIBRARIES ninferno_core ninferno::cudart)
set_tests_properties(ninferno_device_buffer_fill_order_test PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_layout_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_layout.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_materialization_budget_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_materialization_budget.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_host_cpu_quota_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_cpu_quota.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_cache.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_state_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_state_store.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_gdn_replay_records_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gdn_replay_records.cpp"
  LIBRARIES ninferno_core)

set_tests_properties(
  ninferno_device_test
  ninferno_decode_graph_test
  ninferno_arena_test
  ninferno_kv_cache_test
  ninferno_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_host_timing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_timing.cpp"
  LIBRARIES ninferno_core)

ninferno_add_test(ninferno_jinja_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_jinja.cpp"
  LIBRARIES ninferno_jinja ninferno::json)

add_test(NAME ninferno_chat_templates_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_chat_templates.py
          $<TARGET_FILE:ninferno_jinja_test>)
set_tests_properties(ninferno_chat_templates_test PROPERTIES SKIP_RETURN_CODE 77)

# Synchronous copies and fills on the legacy default stream are reviewed, file by file.
add_test(NAME ninferno_legacy_stream_audit_test
  COMMAND bash ${PROJECT_SOURCE_DIR}/tools/audit/legacy_stream_calls.sh ${PROJECT_SOURCE_DIR})

# The engine against Python Jinja2 over an expression corpus and the maintained templates.
add_test(NAME ninferno_jinja_differential_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/jinja_differential.py
          $<TARGET_FILE:ninferno_jinja_test>
          ${PROJECT_SOURCE_DIR}/tools/chat_templates/qwen3_6.jinja
          ${PROJECT_SOURCE_DIR}/tools/chat_templates/qwen3_8.jinja)
set_tests_properties(ninferno_jinja_differential_test PROPERTIES SKIP_RETURN_CODE 77)

# The generic tokenizer.json tokenizer: its regex subset and refusals here; the byte-for-byte
# comparison with the Hugging Face tokenizers library over real tokenizers (fixtures written by
# tools/text/tokenizer_fixtures.py into NINFERNO_TEST_TOKENIZER_FIXTURES) in the real variant.
ninferno_add_test(ninferno_tokenizer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_tokenizer.cpp"
  LIBRARIES ninferno_text ninferno::json)
target_include_directories(ninferno_tokenizer_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
add_test(NAME ninferno_tokenizer_real_test
  COMMAND sh -c "test -n \"$NINFERNO_TEST_TOKENIZER_FIXTURES\" || exit 77; exec \"$0\" \"$NINFERNO_TEST_TOKENIZER_FIXTURES\"/*/"
          $<TARGET_FILE:ninferno_tokenizer_test>)
set_tests_properties(ninferno_tokenizer_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The shared chat layer: conventions from template markers, every tool-call syntax (parser,
# streaming decoder, structural tag) and the generic tokenizer codec.
ninferno_add_test(ninferno_chat_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../chat/test_chat.cpp"
  LIBRARIES ninferno_chat ninferno_text ninferno_structured ninferno::json)

# YaRN rotary frequencies (core/yarn.h, the converter and --rope-yarn-factor), no GPU.
ninferno_add_test(ninferno_yarn_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_yarn.cpp"
  LIBRARIES ninferno_core)

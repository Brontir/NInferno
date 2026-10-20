ninferno_add_test(ninferno_admission_policy_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_admission_policy.cpp"
  LIBRARIES ninferno_runtime_support)

ninferno_add_test(ninferno_structured_output_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_structured_output.cpp"
  LIBRARIES ninferno_structured)

ninferno_add_test(ninferno_context_cost_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost.cpp"
  LIBRARIES ninferno_runtime_support ninferno::json)

ninferno_add_test(ninferno_resource_manager_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_resource_manager.cpp"
  LIBRARIES ninferno_runtime_support)

ninferno_add_test(ninferno_kv_capacity_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_capacity.cpp"
  LIBRARIES ninferno_runtime_support)

ninferno_add_test(ninferno_context_cache_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_context_cache_defaults.cpp"
  LIBRARIES ninferno_engine ninferno_core ninferno::json)

ninferno_add_test(ninferno_sampling_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_sampling_defaults.cpp"
  LIBRARIES ninferno_engine ninferno_core)

# What a load refuses before the device: Qwen-only options on other families, unknown tuner
# variable values. CPU.
ninferno_add_test(ninferno_load_refusals_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_load_refusals.cpp"
  LIBRARIES ninferno_engine ninferno_core ninferno::json)

# The kernel warm-up's prefill width set: the exact band, the route table's breakpoints, and every
# kernel class of the exported plan choosers at or below chunk. CPU.
ninferno_add_test(ninferno_warmup_widths_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_warmup_widths.cpp"
  LIBRARIES ninferno_engine ninferno_core ninferno::json)

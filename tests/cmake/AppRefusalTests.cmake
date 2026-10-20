# The executables' argument refusals: each case runs a binary with arguments it must refuse while
# parsing, before it opens a model or the device, and passes when the output names the refusal.
# No GPU, no model.

# ninferno_add_refusal_test(<name> <target> "<expected regex>" <argument>...)
function(ninferno_add_refusal_test name target expected)
  add_test(NAME ${name} COMMAND $<TARGET_FILE:${target}> ${ARGN})
  set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${expected}")
endfunction()

ninferno_add_refusal_test(ninferno_convert_switch_value_test ninferno-convert
  "--quiet takes no value" --model x --out y --quiet=0)
ninferno_add_refusal_test(ninferno_convert_drafters_twice_test ninferno-convert
  "give one of --with-drafters and --no-drafters" --with-drafters --no-drafters)
ninferno_add_refusal_test(ninferno_convert_hub_test ninferno-convert
  "--hub must be huggingface \\(hf\\) or modelscope \\(ms\\)" --fetch org/m --hub foo)
ninferno_add_refusal_test(ninferno_convert_drafter_quant_test ninferno-convert
  "--drafter-quant must be q8, q4 or nvfp4" --drafter-quant q5)
ninferno_add_refusal_test(ninferno_convert_relayout_drafters_test ninferno-convert
  "give one of --plan-drafters, --attach-drafters and --relayout"
  --plan-drafters --relayout auto --target-artifact x)
ninferno_add_refusal_test(ninferno_convert_threads_test ninferno-convert
  "--threads: invalid integer value" --threads x)
ninferno_add_refusal_test(ninferno_convert_model_without_out_test ninferno-convert
  "error: --model needs --out \\(only --fetch" --model x)
ninferno_add_refusal_test(ninferno_convert_rows_per_chunk_test ninferno-convert
  "--rows-per-chunk must be positive" --model x --out y --rows-per-chunk 0)

ninferno_add_refusal_test(ninferno_spec_tune_unknown_flag_test ninferno
  "spec-tune: unknown flag --budget-secs" spec-tune missing.ninferno --budget-secs 5)
# --gpu and --engine are spec-reselect's; spec-tune reads its GPU from the device.
ninferno_add_refusal_test(ninferno_spec_tune_reselect_flag_test ninferno
  "spec-tune: unknown flag --engine" spec-tune missing.ninferno --engine qwen3_5)
ninferno_add_refusal_test(ninferno_spec_tune_device_test ninferno
  "invalid device: abc" spec-tune missing.ninferno --device abc)

ninferno_add_refusal_test(ninferno_ui_port_text_test ninferno-ui "invalid port: abc" --port abc)
ninferno_add_refusal_test(ninferno_ui_port_range_test ninferno-ui
  "--port must be in \\[1,65535\\]" --port 70000)
ninferno_add_refusal_test(ninferno_ui_catalog_url_test ninferno-ui
  "--catalog-url needs a value" --catalog-url)

ninferno_add_refusal_test(ninferno_perplexity_device_test ninferno-perplexity
  "invalid device: 2147483648" missing.ninferno --device 2147483648)
ninferno_add_refusal_test(ninferno_perplexity_context_test ninferno-perplexity
  "invalid context: 12a" missing.ninferno --context 12a)

# The programs a user runs answer --version with their version (the installer and the release
# scripts call serve's). The output alone decides: serve exits 1 without a driver.
foreach(program ninferno ninferno-serve ninferno-ui ninferno-convert ninferno-perplexity)
  string(REPLACE "-" "_" test_name "${program}_version_test")
  add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${program}> --version)
  set_tests_properties(${test_name} PROPERTIES PASS_REGULAR_EXPRESSION "^NInferno ")
endforeach()

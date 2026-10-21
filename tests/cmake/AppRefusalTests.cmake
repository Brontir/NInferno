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

ninferno_add_refusal_test(ninferno_perplexity_decode_steps_test ninferno-perplexity
  "--decode-steps does not apply: ninferno-perplexity scores" missing.ninferno --text t
  --decode-steps 4)
ninferno_add_refusal_test(ninferno_perplexity_decode_wait_test ninferno-perplexity
  "--decode-wait does not apply" missing.ninferno --text t --decode-wait block)
ninferno_add_refusal_test(ninferno_gemma4_moe_tune_test ninferno-gemma4
  "--moe-tune does not apply: ninferno-gemma4" missing.ninferno score --moe-tune on)
ninferno_add_refusal_test(ninferno_dense_moe_tune_test ninferno-dense
  "--moe-tune does not apply: ninferno-dense" missing.ninferno score --moe-tune on)
ninferno_add_refusal_test(ninferno_dense_unknown_option_test ninferno-dense
  "unknown option --frobnicate" missing.ninferno score --frobnicate 1)
ninferno_add_refusal_test(ninferno_dense_cuda_sync_test ninferno-dense
  "--cuda-sync must be spin, blocking, yield or auto" missing.ninferno score --cuda-sync busy)
ninferno_add_refusal_test(ninferno_dense_dense_tune_test ninferno-dense
  "--dense-tune must be off, on or retune" missing.ninferno score --dense-tune yes)

# A removed NINFERNO_* decision variable still set is named with its replacement, once, after the
# arguments parse (the run then fails on the missing model).
ninferno_add_refusal_test(ninferno_removed_environment_warning_test ninferno-dense
  "warning: NINFERNO_DECODE_STEPS is set but no longer read and changes nothing; use --decode-steps instead"
  missing.ninferno score)
set_tests_properties(ninferno_removed_environment_warning_test PROPERTIES
  ENVIRONMENT "NINFERNO_DECODE_STEPS=4")

# The programs a user runs answer --version with their version (the installer and the release
# scripts call serve's). The output alone decides: serve exits 1 without a driver.
foreach(program ninferno ninferno-serve ninferno-ui ninferno-convert ninferno-perplexity)
  string(REPLACE "-" "_" test_name "${program}_version_test")
  add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${program}> --version)
  set_tests_properties(${test_name} PROPERTIES PASS_REGULAR_EXPRESSION "^NInferno ")
endforeach()

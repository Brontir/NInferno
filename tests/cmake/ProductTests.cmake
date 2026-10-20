ninferno_add_test(ninferno_media_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_decode.cpp"
  LIBRARIES ninferno_media_decode)

ninferno_add_test(ninferno_pdf_text_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_pdf_text.cpp"
  LIBRARIES ninferno_media_pdf)

ninferno_add_test(ninferno_media_acquire_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_acquire.cpp"
  LIBRARIES ninferno_media_acquire ninferno_serve)

ninferno_add_test(ninferno_prompt_input_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prompt_input.cpp"
  LIBRARIES ninferno_product_prompt_input)

# The drafter an explicit --spec finds in <artifact>.speculation.json (model_drafter), and where
# load bounds move when their own graphs do not fit (snap_load_to_captured): no GPU.
ninferno_add_test(ninferno_product_speculation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_product_speculation.cpp"
  LIBRARIES ninferno_product_speculation)

ninferno_add_test(ninferno_pretty_logging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_pretty_logging.cpp"
  LIBRARIES ninferno_product_logging)

ninferno_add_test(ninferno_perplexity_evaluation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_perplexity_evaluation.cpp"
          ${PROJECT_SOURCE_DIR}/apps/perplexity/evaluation.cpp
          ${PROJECT_SOURCE_DIR}/apps/perplexity/kld.cpp
  LIBRARIES ninferno_core)

target_include_directories(ninferno_perplexity_evaluation_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)

ninferno_add_test(ninferno_cli_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cli_options.cpp" ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
  LIBRARIES ninferno_runtime_support ninferno_product_logging)

target_include_directories(ninferno_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)

ninferno_add_test(ninferno_openai_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_schema.cpp"
  LIBRARIES ninferno_serve)

ninferno_add_test(ninferno_openai_responses_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses.cpp"
  LIBRARIES ninferno_serve)

ninferno_add_test(ninferno_openai_responses_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses_store.cpp"
  LIBRARIES ninferno_serve)

ninferno_add_test(ninferno_anthropic_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_anthropic_schema.cpp"
  LIBRARIES ninferno_serve)

ninferno_add_test(ninferno_serve_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_options.cpp"
  LIBRARIES ninferno_serve)

ninferno_add_test(ninferno_request_log_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_request_log.cpp"
  LIBRARIES ninferno_serve)

ninferno_add_test(ninferno_http_error_handler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_error_handler.cpp"
  LIBRARIES ninferno_serve)

ninferno_add_test(ninferno_http_routes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_routes.cpp"
  LIBRARIES ninferno_serve)

ninferno_add_test(ninferno_http_transport_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_transport.cpp"
  LIBRARIES ninferno_serve)

# Chat-template parity with transformers for every template NInferno serves (the current lineup and
# every convention of src/chat/data/conventions.json): the prompt ids ninferno-serve prepares equal
# apply_chat_template(..., tokenize=True). CPU; tokenizer files only, fetched by
# tests/chat/hf_chat_template_parity.py (NINFERNO_CHAT_TEMPLATE_PARITY_ROOT names them); skipped without.
ninferno_add_test(ninferno_chat_template_parity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../chat/test_chat_template_parity.cpp"
  LIBRARIES ninferno_serve ninferno_engine ninferno_core ninferno::json)
set_tests_properties(ninferno_chat_template_parity_test PROPERTIES SKIP_RETURN_CODE 77)

# ninferno-ui: access checks, launch commands, catalog estimates, process pool, routes. No GPU,
# no network (offline listings, NVML replaced).
ninferno_add_test(ninferno_ui_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_ui.cpp"
  LIBRARIES ninferno_ui ninferno_convert)
target_include_directories(ninferno_ui_test PRIVATE ${PROJECT_SOURCE_DIR}/src)

# The release check: version order, manifests, the daily cache and its switches, against a
# manifest served on loopback. No network.
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
ninferno_add_test(ninferno_update_check_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_update_check.cpp"
  LIBRARIES ninferno_product_update ninferno::httplib ninferno::json OpenSSL::Crypto Threads::Threads)
target_include_directories(ninferno_update_check_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(ninferno_update_check_test PRIVATE NINFERNO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# The installer and the updater act only on a release manifest the release key signed.
add_test(NAME ninferno_signed_install_test
  COMMAND bash ${PROJECT_SOURCE_DIR}/tests/release/signed_install_test.sh ${PROJECT_SOURCE_DIR})
set_tests_properties(ninferno_signed_install_test PROPERTIES SKIP_RETURN_CODE 77)

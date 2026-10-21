add_library(ninferno_serve STATIC
  http_transport.cpp
  request_validation.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  openai_common.cpp
  output_constraint_request.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  request_events.cpp
  operational_log.cpp
  plan_report.cpp
  request_log.cpp
  http_server.cpp
  openai_responses_http.cpp
  serve_options.cpp
  model_source.cpp
)
ninferno_internal_includes(ninferno_serve)
target_link_libraries(ninferno_serve
  PUBLIC ninferno_engine ninferno_product_speculation Threads::Threads ninferno::json ninferno::httplib
  PRIVATE ninferno_media_acquire ninferno_media_pdf ninferno_product_logging ninferno_convert ninferno::cudart)

# The chat conventions every family shares: turn, reasoning and tool-call markup read from a data
# table by the markers a chat template contains, and one tool-call parser per syntax.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/data/conventions.json)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/data/conventions.json NINFERNO_CHAT_CONVENTIONS_JSON)
configure_file(embedded_conventions.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/chat/embedded_conventions.h @ONLY)

add_library(ninferno_chat STATIC
  conventions.cpp
  tool_calls.cpp
)
ninferno_internal_includes(ninferno_chat)
target_include_directories(ninferno_chat PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(ninferno_chat PUBLIC ninferno::json)

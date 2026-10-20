# Constrained decoding (structured output): grammar compilation, per-request matching and the
# host side of the token-mask protocol. Model-independent; a Frontend supplies the vocabulary.
add_library(ninferno_structured STATIC
  constraint.cpp
  request_constraint.cpp
  schema_support.cpp
)
ninferno_internal_includes(ninferno_structured)
target_link_libraries(ninferno_structured PUBLIC ninferno::json PRIVATE ninferno::xgrammar)

add_library(ninferno_text STATIC
  pattern.cpp
  hf_token_codec.cpp
  tokenizer.cpp
  unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)
ninferno_internal_includes(ninferno_text)
target_include_directories(ninferno_text PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
# utf8proc is compiled in, not a DLL: without this its header declares it dllimport on Windows.
target_compile_definitions(ninferno_text PUBLIC UTF8PROC_STATIC)

# Language evaluation reuses the engine's Unicode and JSON dependencies.
target_sources(ninferno_jinja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jinja.cpp)
ninferno_internal_includes(ninferno_jinja)
target_include_directories(ninferno_jinja PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninferno_jinja PRIVATE ninferno_text ninferno::json)

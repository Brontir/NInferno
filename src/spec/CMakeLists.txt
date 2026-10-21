# Speculation discovery and tuning support shared by the converter and the engines: vocabulary
# fingerprints, the curated drafter registry, the built-in tuning corpus, profiles and the runtime
# acceptance watchdog. The registry and the corpus are JSON files compiled in.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/data/drafter_registry.json
  ${CMAKE_CURRENT_SOURCE_DIR}/data/spec_corpus.json)
ninferno_embed_text(${CMAKE_CURRENT_SOURCE_DIR}/data/drafter_registry.json NINFERNO_SPEC
  NINFERNO_DRAFTER_REGISTRY_JSON)
ninferno_embed_text(${CMAKE_CURRENT_SOURCE_DIR}/data/spec_corpus.json NINFERNO_SPEC NINFERNO_SPEC_CORPUS_JSON)
configure_file(embedded_data.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/spec/embedded_data.h @ONLY)

add_library(ninferno_spec STATIC
  config.cpp
  corpus.cpp
  fingerprint.cpp
  load_tune.cpp
  load_model.cpp
  manifest.cpp
  menu.cpp
  profile.cpp
  registry.cpp
)
ninferno_internal_includes(ninferno_spec)
target_include_directories(ninferno_spec PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(ninferno_spec PUBLIC ninferno_artifact ninferno::json)

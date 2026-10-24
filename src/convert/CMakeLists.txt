add_library(ninferno_convert STATIC
  pyjson.cpp
  numerics.cpp
  safetensors.cpp
  sources.cpp
  model.cpp
  checks.cpp
  dense.cpp
  gemma4.cpp
  qwen3_5.cpp
  planner.cpp
  expert_layout.cpp
  relayout.cpp
  writer.cpp
  convert.cpp
  cache.cpp
  consume.cpp
  formats.cpp
  ggml_quants.cpp
  gguf.cpp
  gguf_model.cpp
  gguf_resources.cpp
  fetch.cpp
  drafters.cpp
  support.cpp
  acquire.cpp
)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
ninferno_internal_includes(ninferno_convert)
target_link_libraries(ninferno_convert PUBLIC ninferno_artifact ninferno_spec ninferno_chat ninferno::json Threads::Threads
  PRIVATE PkgConfig::LIBCURL OpenSSL::Crypto)
if(NINFERNO_SOURCE_TREE_PATHS)
  target_compile_definitions(ninferno_convert PRIVATE NINFERNO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()
# The converter reproduces the Python converter's rounding bit for bit: no contraction of a
# multiply into a following add, no fast-math reassociation.
# (MSVC's default /fp:precise neither contracts nor reassociates.)
if(NOT MSVC)
  target_compile_options(ninferno_convert PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>)
endif()

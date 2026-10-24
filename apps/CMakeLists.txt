add_executable(ninferno
  cli/main.cpp
  cli/options.cpp
  cli/spec_tune.cpp)
ninferno_internal_includes(ninferno)
target_link_libraries(ninferno PRIVATE
  ninferno_engine
  ninferno_spec
  ninferno_product_logging
  ninferno_product_prompt_input
  ninferno_product_speculation
  ninferno_product_plan
  ninferno::cudart)

add_executable(ninferno-ui ui/main.cpp)
ninferno_internal_includes(ninferno-ui)
target_link_libraries(ninferno-ui PRIVATE ninferno_ui ninferno_product_update)

add_executable(ninferno-serve serve/main.cpp)
ninferno_internal_includes(ninferno-serve)
target_link_libraries(ninferno-serve PRIVATE ninferno_serve ninferno_product_logging ninferno_product_update ninferno::cudart)

add_executable(ninferno-gemma4 gemma4/main.cpp)
ninferno_internal_includes(ninferno-gemma4)
target_link_libraries(ninferno-gemma4 PRIVATE ninferno_gemma4 ninferno_convert)

add_executable(ninferno-dense dense/main.cpp)
ninferno_internal_includes(ninferno-dense)
target_link_libraries(ninferno-dense PRIVATE ninferno_dense ninferno_convert)

add_executable(ninferno-gemma4-inspect gemma4/inspect.cpp)
ninferno_internal_includes(ninferno-gemma4-inspect)
target_link_libraries(ninferno-gemma4-inspect PRIVATE ninferno_gemma4)

add_executable(ninferno-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp
  perplexity/kld.cpp)
ninferno_internal_includes(ninferno-perplexity)
target_link_libraries(ninferno-perplexity PRIVATE
  ninferno_engine ninferno_product_logging ninferno::json)

# The release's entry point: installed as bin/ninferno-serve, it starts bin/ninferno-serve-sm<tag>, the
# engine built for the GPU the command runs on (dispatch/main.cpp). No CUDA code: one for every tree.
add_executable(ninferno-dispatch dispatch/main.cpp)
ninferno_internal_includes(ninferno-dispatch)
target_link_libraries(ninferno-dispatch PRIVATE ninferno::json)

add_executable(ninferno-convert convert/main.cpp)
ninferno_internal_includes(ninferno-convert)
target_link_libraries(ninferno-convert PRIVATE ninferno_convert)

# The programs a user runs answer --version with the version a release names (CMakeLists.txt);
# the family CLIs and the inspector above are maintainer tools.
foreach(program ninferno ninferno-ui ninferno-serve ninferno-perplexity ninferno-dispatch
        ninferno-convert)
  target_compile_definitions(${program} PRIVATE NINFERNO_VERSION_STRING="${NINFERNO_VERSION_STRING}")
endforeach()


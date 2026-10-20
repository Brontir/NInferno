# Static context-cost calibrator. Transfer measurements call production Paged-KV copy primitives,
# then release their synthetic fixtures before the public Engine optionally loads one real artifact
# for text/Vision prefill measurements.
add_executable(ninferno_context_cost_bench
  "${CMAKE_CURRENT_LIST_DIR}/context_cost_bench.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/context_cost_measure.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model_context_fixture.cpp")
ninferno_internal_includes(ninferno_context_cost_bench)
target_include_directories(ninferno_context_cost_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ninferno_context_cost_bench PRIVATE
  NINFERNO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninferno_context_cost_bench PRIVATE
  ninferno_engine ninferno_core ninferno::cudart ninferno::json)

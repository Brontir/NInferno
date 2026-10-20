ninferno_add_test(ninferno_bench_support_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_ninferno_bench_support.cpp"
          ${PROJECT_SOURCE_DIR}/bench/inference/ninferno_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninferno_engine ninferno::json)

target_include_directories(ninferno_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/inference)

ninferno_add_op_test(ninferno_bench_fixtures_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_bench_fixtures.cu"
  LIBRARIES ninferno_ops)
target_include_directories(ninferno_bench_fixtures_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/ops)

add_executable(ninferno_context_cost_measure_test
  "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost_measure.cpp"
  ${PROJECT_SOURCE_DIR}/bench/context_cost/context_cost_measure.cpp)

target_include_directories(ninferno_context_cost_measure_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/context_cost)

add_test(NAME ninferno_context_cost_measure_test COMMAND ninferno_context_cost_measure_test)

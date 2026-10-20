# End-to-end product benchmark: public Engine API and native .ninferno artifacts only.
add_executable(ninferno_bench
  "${CMAKE_CURRENT_LIST_DIR}/ninferno_bench.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ninferno_bench_support.cpp")
ninferno_internal_includes(ninferno_bench)
target_include_directories(ninferno_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ninferno_bench PRIVATE NINFERNO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninferno_bench PRIVATE ninferno_engine ninferno::cudart)

# Token-level outputs (teacher-forced log-probabilities and argmax, greedy continuations) for
# comparison against an independent reference of a model being onboarded.
add_executable(ninferno_quality_probe "${CMAKE_CURRENT_LIST_DIR}/quality_probe.cpp")
ninferno_internal_includes(ninferno_quality_probe)
target_link_libraries(ninferno_quality_probe PRIVATE ninferno_engine)

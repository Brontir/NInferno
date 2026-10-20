ninferno_add_test(ninferno_artifact_reader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reader.cpp"
  LIBRARIES ninferno_artifact)

ninferno_add_test(ninferno_artifact_materialization_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp" "${CMAKE_CURRENT_LIST_DIR}/materialization_cuda_errors.cpp"
  LIBRARIES ninferno_artifact)

target_link_options(ninferno_artifact_materialization_test PRIVATE
  "LINKER:--wrap=cudaMalloc"
  "LINKER:--wrap=cudaMallocHost"
  "LINKER:--wrap=cudaFree"
  "LINKER:--wrap=cudaFreeHost"
  "LINKER:--wrap=cudaEventCreateWithFlags"
  "LINKER:--wrap=cudaEventRecord"
  "LINKER:--wrap=cudaMemcpyAsync"
  "LINKER:--wrap=cudaStreamSynchronize")

add_test(NAME ninferno_artifact_writer_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/writer_interop.py"
    $<TARGET_FILE:ninferno_artifact_materialization_test>)

set_tests_properties(
  ninferno_artifact_writer_interop_test
  PROPERTIES SKIP_RETURN_CODE 77)

set_tests_properties(
  ninferno_artifact_materialization_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninferno_add_test(ninferno_artifact_expert_offload_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_offload.cpp"
  LIBRARIES ninferno_artifact)

set_tests_properties(
  ninferno_artifact_expert_offload_test
  PROPERTIES SKIP_RETURN_CODE 77)

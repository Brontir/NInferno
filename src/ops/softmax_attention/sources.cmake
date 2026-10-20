ninferno_target_sources(ninferno_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/common/causal_column_chunks.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/tiled_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/volta_splitd.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/tiled_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gemma4/gemma4_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gemma4/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gemma4/volta_decode.cu"
)

ninferno_target_sources(ninferno_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/tiled_launch.cu"
)

# Volta prefill attention over the INT8 cache on the vendored split-D flash kernel (CuTe), on the
# sm_70 tree alone: its include directory and namespace stay private to this translation unit.
if(NINFERNO_CUTLASS_INCLUDE)
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/volta_splitd.cu"
    TARGET_DIRECTORY ninferno_ops
    PROPERTIES INCLUDE_DIRECTORIES "${NINFERNO_CUTLASS_INCLUDE};${PROJECT_SOURCE_DIR}/third_party/sm70_flash_d256"
               COMPILE_DEFINITIONS "NINFERNO_VOLTA_SPLITD;FLASH_NAMESPACE=ninferno_sm70_flash"
               COMPILE_OPTIONS "--expt-relaxed-constexpr")
endif()

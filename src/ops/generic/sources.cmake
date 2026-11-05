# Generic projection path: runtime-geometry kernels for every weight format. Its tuned routes come
# from the per-GPU-class route table (src/ops/route_table.cpp, include/ninferno/ops/route_table.h).
ninferno_target_sources(ninferno_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_ggml.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/build_manifest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_tune.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/route_query.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/route_tune_tool.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_i8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_bf16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_q4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_q5.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_q6.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_q3.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_q8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_mxfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_fp8_row_f32.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_fp8_b128.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_u4zp.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_u4zp_g32.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_u4zp_g64.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gather.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_mmq_epilogue.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_volta_cutlass.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q4k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q5k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q6k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q3k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q2k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_iq4xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_iq3xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_iq3s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q40.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q41.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q50.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q51.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_iq4nl.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_iq2xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_iq2xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_iq2s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_iq1s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_iq1m.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_tq10.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_tq20.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q10.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gemm_ggml_q20.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_w4a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_scalar.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_volta_gemv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_volta_hmma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_w4a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_w4a8_ggml32.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_w4a8_superblock.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_w4a8_codebook.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_r8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_r8_iq3.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_r8_iq2.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_r8_ggml4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_r8_kquant.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_tiled.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_staged.cu"
  "${CMAKE_CURRENT_LIST_DIR}/generic_nvfp4_tiled.cu"
  "${CMAKE_CURRENT_LIST_DIR}/segmented_projection.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/segmented_gating.cu"
)
target_link_libraries(ninferno_ops PRIVATE ninferno::json)
include("${CMAKE_CURRENT_LIST_DIR}/ggml_mmq/sources.cmake")
# The compilers the compiled-variant manifest names (build_manifest.cpp, #307), on that file alone.
string(REPLACE ";" "," ninferno_manifest_archs "${CMAKE_CUDA_ARCHITECTURES}")
set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/build_manifest.cpp"
  TARGET_DIRECTORY ninferno_ops
  PROPERTIES COMPILE_DEFINITIONS
    "NINFERNO_CUDA_COMPILER=\"${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}\";NINFERNO_CXX_COMPILER=\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\";NINFERNO_MANIFEST_CUDA_ARCHITECTURES=\"${ninferno_manifest_archs}\"")

# CUTLASS's headers for the Volta prefill GEMM, on the sm_70 tree alone (CMakeLists.txt).
if(NINFERNO_CUTLASS_INCLUDE)
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/generic_volta_cutlass.cu"
    TARGET_DIRECTORY ninferno_ops
    PROPERTIES INCLUDE_DIRECTORIES "${NINFERNO_CUTLASS_INCLUDE}"
               COMPILE_DEFINITIONS NINFERNO_VOLTA_CUTLASS
               COMPILE_OPTIONS "--expt-relaxed-constexpr")
endif()

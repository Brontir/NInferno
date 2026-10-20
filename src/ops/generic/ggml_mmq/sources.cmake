# ggml's integer tensor-core matrix kernel (third_party/ggml-quants) for the generic path's ggml
# block formats at speculative-verify widths (generic_gemm.cpp, Family::GgmlMmq). Its own archive:
# the ggml headers define CUDA_CHECK and friends, so no other NInferno unit may see them.
set(NINFERNO_GGML_QUANTS_ROOT "${PROJECT_SOURCE_DIR}/third_party/ggml-quants")
add_library(ninferno_ggml_mmq STATIC
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q5_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q6_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq4_xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq3_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq3_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_1.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q5_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q5_1.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq4_nl.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q2_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q3_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq2_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq2_xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq2_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq1_s.cu")
target_include_directories(ninferno_ggml_mmq PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}"
  "${NINFERNO_GGML_QUANTS_ROOT}/include"
  "${NINFERNO_GGML_QUANTS_ROOT}/src"
  # NInferno's own headers after ggml's (the matrix kernel's programmatic launch, core/pdl.cuh).
  "${CMAKE_CURRENT_LIST_DIR}/../../..")
# Its matrix kernels call no cuBLAS (the vendored headers leave it out under this macro).
target_compile_definitions(ninferno_ggml_mmq PRIVATE NINFERNO_GGML_NO_CUBLAS)
ninferno_cuda_non_rdc_archive(ninferno_ggml_mmq)
target_compile_options(ninferno_ggml_mmq PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>)
target_link_libraries(ninferno_ggml_mmq PRIVATE ninferno::cudart)
target_link_libraries(ninferno_ops PRIVATE ninferno_ggml_mmq)

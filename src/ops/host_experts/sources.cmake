# MoE expert offload's decode path on the CPU (src/ops/host_experts/host_moe.h): the canonical row
# dots per instruction set, the pinned worker pool, the layer computations and their CUDA glue.
set(_ninferno_host_experts_dir "${CMAKE_CURRENT_LIST_DIR}")
ninferno_target_sources(ninferno_ops PRIVATE
  "${_ninferno_host_experts_dir}/host_kernels.cpp"
  "${_ninferno_host_experts_dir}/host_kernels_avx2.cpp"
  "${_ninferno_host_experts_dir}/host_kernels_avx512.cpp"
  "${_ninferno_host_experts_dir}/host_pool.cpp"
  "${_ninferno_host_experts_dir}/host_moe.cpp"
  "${_ninferno_host_experts_dir}/host_experts.cpp"
  "${_ninferno_host_experts_dir}/handoff.cu"
)
# One arithmetic on every path: no contraction of a multiply and an add the code keeps apart.
if(MSVC)
  # /fp:precise (the default) does not contract; /arch: enables the instruction sets.
  set_source_files_properties("${_ninferno_host_experts_dir}/host_kernels_avx2.cpp"
    PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties("${_ninferno_host_experts_dir}/host_kernels_avx512.cpp"
    PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
set_source_files_properties(
  "${_ninferno_host_experts_dir}/host_kernels.cpp"
  "${_ninferno_host_experts_dir}/host_moe.cpp"
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties("${_ninferno_host_experts_dir}/host_kernels_avx2.cpp"
  PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off")
set_source_files_properties("${_ninferno_host_experts_dir}/host_kernels_avx512.cpp"
  PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx2;-mfma;-ffp-contract=off")
endif()

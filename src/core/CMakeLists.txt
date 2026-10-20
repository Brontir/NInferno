add_library(ninferno_core STATIC
  arena.cu
  device.cu
  device_class.cpp
  device_image_probe.cu
  tune_timer.cu
  dtype.cpp
  host_worker_pool.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_kv_arena.cpp
  host_memory.cpp
  expert_residency.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  ggml_frag.cpp
  decode_graph.cpp
  debug_poison.cpp
  debug_fault.cu
)
ninferno_internal_includes(ninferno_core)
ninferno_cuda_archive(ninferno_core)
target_link_libraries(ninferno_core PUBLIC ninferno::cudart CUDA::nvtx3 Threads::Threads)

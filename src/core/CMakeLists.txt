# The PCI-ID reference table of board facts (machine_facts.cpp), a JSON file compiled in.
ninferno_embed_text(${CMAKE_CURRENT_SOURCE_DIR}/data/gpu_facts.json NINFERNO_FACTS
  NINFERNO_GPU_FACTS_JSON)
configure_file(embedded_gpu_facts.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/core/embedded_gpu_facts.h @ONLY)

add_library(ninferno_core STATIC
  arena.cu
  device.cu
  device_class.cpp
  device_traits.cpp
  machine_facts.cpp
  device_image_probe.cu
  tune_timer.cu
  dtype.cpp
  host_worker_pool.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_kv_arena.cpp
  host_memory.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  ggml_frag.cpp
  decode_graph.cpp
  debug_poison.cpp
  sha256.cpp
  debug_fault.cu
)
ninferno_internal_includes(ninferno_core)
target_include_directories(ninferno_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
ninferno_cuda_archive(ninferno_core)
target_link_libraries(ninferno_core PUBLIC ninferno::cudart CUDA::nvtx3 Threads::Threads
  PRIVATE ninferno::json)

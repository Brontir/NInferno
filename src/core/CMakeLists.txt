# The PCI-ID reference table of board facts (machine_facts.cpp), a JSON file compiled in.
ninferno_embed_text(${CMAKE_CURRENT_SOURCE_DIR}/data/gpu_facts.json NINFERNO_FACTS
  NINFERNO_GPU_FACTS_JSON)
configure_file(embedded_gpu_facts.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/core/embedded_gpu_facts.h @ONLY)
# The same table as constexpr rows (core/board_registry.h), for the data that is keyed by a board
# at compile time (core/measured_boards.h). The directory is public: the header reaches every
# target that includes it.
ninferno_generate_board_registry(${CMAKE_CURRENT_SOURCE_DIR}/data/gpu_facts.json
  ${CMAKE_CURRENT_BINARY_DIR}/generated_public/core/board_registry_rows.inc)
# And the "archs" array as constexpr rows (core/arch_registry.h): the SM limits per compute
# capability that the build's limits, the MMQ tile widths and the synthetic device class read.
ninferno_generate_arch_registry(${CMAKE_CURRENT_SOURCE_DIR}/data/gpu_facts.json
  ${CMAKE_CURRENT_BINARY_DIR}/generated_public/core/arch_registry_rows.inc)

# The decision code's part of core, with no CUDA runtime: the headers of the facts it reads
# (generation.h, cost_constants.h, mmq_columns.h, board_registry.h and its generated rows) and the
# sources they need. ninferno_core links it; a target that links only this and no cudart can be
# built and run on a machine with no GPU and no CUDA runtime library (tests/cmake/ProductTests.cmake
# asserts the closure); the toolkit's headers are still read (kernel_occupancy.cpp, below).
add_library(ninferno_core_host STATIC
  dtype.cpp
  device_class_host.cpp
  machine_facts.cpp
  gate_atlas.cpp
  layout.cpp
  weight_view.cpp
  ggml_frag.cpp
  sha256.cpp
  tensor.cpp
  kernel_occupancy.cpp
)
ninferno_internal_includes(ninferno_core_host)
target_include_directories(ninferno_core_host PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_include_directories(ninferno_core_host PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generated_public)
target_link_libraries(ninferno_core_host PRIVATE ninferno::json)
# kernel_occupancy.cpp reads the toolkit's cuda_occupancy.h, a header-only calculator: the include
# directory, no library.
target_include_directories(ninferno_core_host SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# core/device_class_nodevice.cpp is current_device_class() of a build with no CUDA runtime: it
# always throws. core/per_device_nodevice.cpp is the same for current_device_ordinal() (it throws)
# and current_device_ordinal_or_none() (-1); per_device.cpp holds the real ones. A host-only test
# adds both to its own sources (an object, so no archive order matters); nothing beside
# ninferno_core links them, which defines the real ones.

add_library(ninferno_core STATIC
  arena.cu
  device.cu
  device_class.cpp
  per_device.cpp
  device_traits.cpp
  device_image_probe.cu
  tune_timer.cu
  host_worker_pool.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_kv_arena.cpp
  host_memory.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  decode_graph.cpp
  debug_poison.cpp
  debug_fault.cu
)
ninferno_internal_includes(ninferno_core)
target_include_directories(ninferno_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_include_directories(ninferno_core PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generated_public)
ninferno_cuda_archive(ninferno_core)
target_link_libraries(ninferno_core PUBLIC ninferno_core_host ninferno::cudart CUDA::nvtx3 Threads::Threads
  PRIVATE ninferno::json)

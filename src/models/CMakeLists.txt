# Loader pieces with no family knowledge, shared by every family.
add_library(ninferno_model_common STATIC common/config_json.cpp common/loading.cpp
  common/weight_use.cpp)
ninferno_internal_includes(ninferno_model_common)
target_link_libraries(ninferno_model_common PUBLIC ninferno_artifact ninferno_ops ninferno_core)

# Greedy speculative decoding over any target that implements speculation::DraftTarget: the
# rounds and their graphs, the DFlash drafter and prompt lookup, shared by Gemma 4 and the dense
# decoder.
add_library(ninferno_speculation STATIC
  speculation/decoder.cu
  speculation/dflash.cpp
  speculation/dflash_round.cu
  speculation/driver.cpp
  speculation/ngram_round.cu)
ninferno_internal_includes(ninferno_speculation)
ninferno_cuda_lineinfo(ninferno_speculation)
target_link_libraries(ninferno_speculation PUBLIC ninferno_artifact ninferno_ops ninferno_core
  ninferno_model_common)

# Immutable loading and mutable execution have separate compile owners.
add_library(ninferno_model_loading STATIC registry.cpp)
ninferno_internal_includes(ninferno_model_loading)
target_link_libraries(ninferno_model_loading PUBLIC ninferno_artifact ninferno_text ninferno_ops
  ninferno_model_common)

# The byte arithmetic of a sequence plan that needs no bound weights (the persistent layouts every
# family lays its state out with, the planner's inputs, the CUDA graph allowance): a start and the
# plan without the weights (runtime/engine/plan_dry_run.cpp) call the same functions.
add_library(ninferno_sequence_layout STATIC
  qwen3_5/state/decoder_state.cpp
  qwen3_5/state/state_image.cpp
  qwen3_5/program/round_buffers.cpp
  qwen3_5/program/planning/graph_profiles.cpp
  qwen3_5/program/planning/sequence_bytes.cpp)
ninferno_internal_includes(ninferno_sequence_layout)
target_link_libraries(ninferno_sequence_layout PUBLIC ninferno_core PRIVATE ninferno::cudart)

add_library(ninferno_model_runtime STATIC)
ninferno_internal_includes(ninferno_model_runtime)
target_link_libraries(ninferno_model_runtime
  PUBLIC ninferno_model_loading ninferno_ops ninferno_runtime_support ninferno_sequence_layout
    ninferno_chat
  PRIVATE ninferno_media_decode ninferno_jinja)

include(qwen3_5/loading_sources.cmake)
include(qwen3_5/frontend_sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)

# Dense decoders of the Llama lineage: one family whose per-model differences are config data.
add_library(ninferno_dense STATIC
  dense/config.cpp
  dense/frontend.cpp
  dense/load.cpp
  dense/session.cu
  dense/target.cu)
ninferno_internal_includes(ninferno_dense)
ninferno_cuda_lineinfo(ninferno_dense)
target_link_libraries(ninferno_dense PUBLIC ninferno_chat ninferno_artifact ninferno_ops ninferno_core
  ninferno_model_common ninferno_speculation ninferno_text ninferno_jinja ninferno::json)
target_include_directories(ninferno_dense PRIVATE ${PROJECT_SOURCE_DIR}/third_party)

# The dense decoder as a family of the one Program (serve): its execution over the Program's state
# (dense/program_family.cpp) and the DFlash hooks of its speculative rounds (dense/dflash.cpp).
target_sources(ninferno_model_runtime PRIVATE dense/program_family.cpp dense/dflash.cpp)
target_link_libraries(ninferno_model_runtime PUBLIC ninferno_dense)

# NemotronH hybrids (Mamba2 / attention / relu^2 MoE) as a family of the one Program.
add_library(ninferno_nemotron_h STATIC
  nemotron_h/config.cpp
  nemotron_h/load.cpp
  nemotron_h/load_mtp.cpp
  nemotron_h/pass.cu)
ninferno_internal_includes(ninferno_nemotron_h)
ninferno_cuda_lineinfo(ninferno_nemotron_h)
target_link_libraries(ninferno_nemotron_h PUBLIC ninferno_artifact ninferno_ops ninferno_core
  ninferno_model_common ninferno::json)
target_sources(ninferno_model_runtime PRIVATE nemotron_h/program_family.cpp)
target_link_libraries(ninferno_model_runtime PUBLIC ninferno_nemotron_h)

# Gemma 4 text: its own family, sharing Ops, the artifact reader and core types only.
add_library(ninferno_gemma4 STATIC
  gemma4/config.cpp
  gemma4/load_mtp.cpp
  gemma4/load_target.cpp
  gemma4/vision.cpp
  gemma4/mtp_round.cu
  gemma4/session.cu
  gemma4/target.cu)
ninferno_internal_includes(ninferno_gemma4)
ninferno_cuda_lineinfo(ninferno_gemma4)
target_link_libraries(ninferno_gemma4 PUBLIC ninferno_artifact ninferno_ops ninferno_core
  ninferno_model_common ninferno_speculation)

# Gemma 4 as a family of the one Program (serve): its execution over the Program's state.
target_sources(ninferno_model_runtime PRIVATE gemma4/program_family.cpp)
target_link_libraries(ninferno_model_runtime PUBLIC ninferno_gemma4)

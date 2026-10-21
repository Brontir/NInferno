add_library(ninferno_runtime_support STATIC
  contract/token_mask_staging.cpp
  engine/admission_policy.cpp
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/decision_fingerprint.cpp
  engine/kv_capacity.cpp
  engine/public_types.cpp
)
ninferno_internal_includes(ninferno_runtime_support)
target_link_libraries(ninferno_runtime_support PUBLIC ninferno_core PRIVATE ninferno::json ninferno::cudart)
# The build section of the decision fingerprint: the commit the tree was configured at
# (git describe --always --dirty; "unknown" unless this source tree is itself the top of a git
# checkout, so a tree unpacked inside another repository does not take that one's commit; a commit
# made since without reconfiguring is not seen), every CUDA architecture entry, and the build
# options, defined on this source alone so that no other object rebuilds when they change.
find_package(Git QUIET)
set(NINFERNO_GIT_COMMIT "unknown")
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE ninferno_git_top OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ninferno_git_result ERROR_QUIET)
  if(ninferno_git_result EQUAL 0)
    file(REAL_PATH "${ninferno_git_top}" ninferno_git_top)
    file(REAL_PATH "${PROJECT_SOURCE_DIR}" ninferno_source_top)
  endif()
  if(ninferno_git_result EQUAL 0 AND ninferno_git_top STREQUAL ninferno_source_top)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE ninferno_git_commit OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE ninferno_git_result ERROR_QUIET)
    if(ninferno_git_result EQUAL 0 AND ninferno_git_commit MATCHES "^[0-9A-Za-z._/-]+$")
      set(NINFERNO_GIT_COMMIT "${ninferno_git_commit}")
    endif()
  endif()
endif()
string(REPLACE ";" "," ninferno_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
set(ninferno_embed_ptx 0)
if(NINFERNO_CUDA_EMBED_PTX)
  set(ninferno_embed_ptx 1)
endif()
set_source_files_properties(engine/decision_fingerprint.cpp PROPERTIES COMPILE_DEFINITIONS
  "NINFERNO_VERSION_STRING=\"${NINFERNO_VERSION_STRING}\";NINFERNO_GIT_COMMIT=\"${NINFERNO_GIT_COMMIT}\";NINFERNO_CUDA_EMBED_PTX=${ninferno_embed_ptx};NINFERNO_CUDA_ARCHITECTURES=\"${ninferno_cuda_architectures}\"")

# Public Engine and common contracts; model execution remains in the model layer.
add_library(ninferno_engine STATIC
  engine/family_registry.cpp
  engine/dense_family.cpp
  engine/nemotron_h_family.cpp
  engine/gemma4_family.cpp
  engine/hf_text_frontend.cpp
  engine/model_instance.cpp
  engine/warmup_widths.cpp
  contract/sampling.cpp
  engine/engine.cpp
)
ninferno_internal_includes(ninferno_engine)
target_include_directories(ninferno_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninferno_engine PRIVATE
  ninferno_model_runtime
  ninferno_artifact
  ninferno_core
  ninferno_runtime_support
  ninferno_ops
  ninferno_structured
  ninferno_text
  ninferno_media_decode)
add_library(ninferno::engine ALIAS ninferno_engine)

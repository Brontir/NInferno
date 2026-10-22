add_library(ninferno_runtime_support STATIC
  contract/token_mask_staging.cpp
  engine/admission_policy.cpp
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/decision_fingerprint.cpp
  engine/kv_capacity.cpp
  engine/plan_ladder.cpp
  engine/public_types.cpp
)
ninferno_internal_includes(ninferno_runtime_support)
target_link_libraries(ninferno_runtime_support PUBLIC ninferno_core PRIVATE ninferno::json ninferno::cudart)
# The build section of the decision fingerprint. The commit is read at every build, not at
# configure (a stand checks out a commit and rebuilds without reconfiguring): the
# ninferno_git_commit target runs cmake/GitCommit.cmake, which rewrites the generated source only
# when the commit changes, so an unchanged commit rebuilds nothing. Git absent or no checkout:
# "unknown". Every CUDA architecture entry and the build options are defined on
# decision_fingerprint.cpp alone, so that no other object rebuilds when they change.
find_package(Git QUIET)
set(ninferno_git_commit_source ${CMAKE_CURRENT_BINARY_DIR}/generated/runtime/build_git_commit.cpp)
add_custom_target(ninferno_git_commit
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
    -DOUTPUT=${ninferno_git_commit_source} "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
    -P ${PROJECT_SOURCE_DIR}/cmake/GitCommit.cmake
  BYPRODUCTS ${ninferno_git_commit_source}
  COMMENT "Reading the git commit for the decision fingerprint"
  VERBATIM)
target_sources(ninferno_runtime_support PRIVATE ${ninferno_git_commit_source})
add_dependencies(ninferno_runtime_support ninferno_git_commit)
string(REPLACE ";" "," ninferno_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
set(ninferno_embed_ptx 0)
if(NINFERNO_CUDA_EMBED_PTX)
  set(ninferno_embed_ptx 1)
endif()
set_source_files_properties(engine/decision_fingerprint.cpp PROPERTIES COMPILE_DEFINITIONS
  "NINFERNO_VERSION_STRING=\"${NINFERNO_VERSION_STRING}\";NINFERNO_CUDA_EMBED_PTX=${ninferno_embed_ptx};NINFERNO_CUDA_ARCHITECTURES=\"${ninferno_cuda_architectures}\"")

# Public Engine and common contracts; model execution remains in the model layer.
add_library(ninferno_engine STATIC
  engine/family_registry.cpp
  engine/dense_family.cpp
  engine/decision_sources.cpp
  engine/nemotron_h_family.cpp
  engine/gemma4_family.cpp
  engine/hf_text_frontend.cpp
  engine/model_instance.cpp
  engine/served_bytes.cpp
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

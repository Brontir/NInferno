add_library(ninferno_runtime_support STATIC
  contract/token_mask_staging.cpp
  engine/admission_policy.cpp
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/kv_capacity.cpp
  engine/public_types.cpp
)
ninferno_internal_includes(ninferno_runtime_support)
target_link_libraries(ninferno_runtime_support PUBLIC ninferno_core PRIVATE ninferno::json ninferno::cudart)

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

ninferno_target_sources(ninferno_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gemma4_moe_simt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gemma4_moe_prefill.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/gemma4_moe.cpp"
)

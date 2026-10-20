ninferno_target_sources(ninferno_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/gated_delta_net.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/replay.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/recurrent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/prepare.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/recurrence.cu"
)

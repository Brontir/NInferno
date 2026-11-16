ninferno_add_op_test(ninferno_linear_pair_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp" "${CMAKE_CURRENT_LIST_DIR}/linear_pair_test_common.cpp"
  LIBRARIES ninferno_ops)

# linear_pair's candidates per call (#310): the fused Q8 pair or the two separate calls, one selected
# as linear_pair runs.
ninferno_add_op_test(ninferno_linear_pair_candidates_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_pair_candidates.cpp"
  LIBRARIES ninferno_ops)

# #617 D: the residency price of linear_pair's one-launch schedules against seven boards'
# measured cells (no GPU).
ninferno_add_op_test(ninferno_linear_pair_work_test LABELS host
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_pair_work.cpp"
  LIBRARIES ninferno_ops)

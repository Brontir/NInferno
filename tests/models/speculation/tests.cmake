# The speculation rounds shared by every target (models/speculation): host-side pieces.
ninferno_add_test(ninferno_speculation_ngram_lookup_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_lookup.cpp"
  LIBRARIES ninferno_speculation)

# The DFlash feature projection split by tap (split_projection_columns) for BF16, Q4_K, Q6_K and
# Q5_0: each slice's projection against the whole matrix's, and the running sum the dense Program
# keeps against the whole projection. GPU.
ninferno_add_op_test(ninferno_speculation_dflash_split_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_split.cpp"
  LIBRARIES ninferno_speculation)
